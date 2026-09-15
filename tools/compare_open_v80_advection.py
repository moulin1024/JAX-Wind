"""Compare centered and MUSCL single-V80 runs, including unaveraged faces."""
import argparse
import csv
import json
from pathlib import Path
import tomllib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(directory):
    doc = tomllib.loads((directory / "resolved_case.toml").read_text())
    rotor = doc["physics"]["wind_farm"]["layout"][0]
    with np.load(directory / "flow_frames.npz") as a:
        frames, times = a["u_hub_yx"], a["time_seconds"]
        xf, yf, zf = (a[k] for k in ("x_faces_m", "y_faces_m", "z_faces_m"))
    xc, yc, zc = ((f[:-1] + f[1:]) / 2 for f in (xf, yf, zf))
    hub = int(np.argmin(abs(zc - rotor["hub_height_m"])))
    with np.load(directory / "checkpoint.npz") as a:
        raw = a["state/velocity/x"][hub].copy()
    with (directory / "history.csv").open() as f:
        history = list(csv.DictReader(f))
    mask = xf < rotor["x_m"] - 160.
    adjacent = np.diff(raw[:, mask], axis=1)
    # Second differences emphasize alternating-grid structure, although genuine
    # curvature and the near-inlet transition also contribute to this metric.
    second = np.diff(raw[:, mask], n=2, axis=1)
    metrics = {
        "max_sampled_cfl": max(float(r["maximum_cfl"]) for r in history),
        "max_sampled_divergence_s": max(float(r["maximum_divergence_s"]) for r in history),
        "upstream_max_adjacent_face_jump_m_s": float(abs(adjacent).max()),
        "upstream_second_difference_rms_m_s": float(np.sqrt(np.mean(second ** 2))),
        "final_lookup_power_w": float(history[-1]["farm_lookup_power_w"]),
    }
    return doc, rotor, frames, times, xf, yf, xc, yc, raw, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("centered", type=Path)
    parser.add_argument("muscl", type=Path)
    args = parser.parse_args()
    old, new = read(args.centered), read(args.muscl)
    for section in ("mesh", "time", "physics"):
        assert old[0][section] == new[0][section], section
    np.testing.assert_array_equal(old[3], new[3])
    result = {"centered": old[-1], "muscl_mc": new[-1]}
    result["adjacent_jump_reduction_percent"] = 100 * (1 - new[-1]["upstream_max_adjacent_face_jump_m_s"] / old[-1]["upstream_max_adjacent_face_jump_m_s"])
    result["second_difference_rms_reduction_percent"] = 100 * (1 - new[-1]["upstream_second_difference_rms_m_s"] / old[-1]["upstream_second_difference_rms_m_s"])
    (args.muscl / "advection_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for data, label, colour in ((old, "Centered", "C0"), (new, "MUSCL-MC", "C1")):
        _, rotor, frames, times, xf, yf, xc, yc, raw, _ = data
        mask = xf < rotor["x_m"] - 160.
        row = int(np.argmin(abs(yc - rotor["y_m"])))
        axes[0, 0].plot(xf[mask], raw[row, mask], ".-", color=colour, label=label)
        upstream = frames[:, :, xc < rotor["x_m"] - 160.]
        axes[0, 1].plot(times, upstream.min(axis=(1, 2)), color=colour, label=label + " minimum")
        axes[0, 1].plot(times, upstream.max(axis=(1, 2)), "--", color=colour, label=label + " maximum")
        for ax, diameters in zip(axes[1], (3, 10)):
            column = int(np.argmin(abs(xc - rotor["x_m"] - diameters * 80.)))
            ax.plot(yc - rotor["y_m"], frames[-1, :, column], color=colour, label=label)
            ax.set(xlim=(-240, 240), xlabel="y − rotor centre [m]", ylabel="u [m/s]",
                   title=f"Final hub-height wake at x = {xc[column]:g} m (~{diameters}D downstream)")
    axes[0, 0].set(xlabel="x [m]", ylabel="Raw face u [m/s]", title="Final upstream face profile, y = 504 m")
    axes[0, 1].set(xlabel="Time [s]", ylabel="Cell-centred u [m/s]", title="Upstream extrema: x < 352 m")
    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(args.muscl / "advection_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
