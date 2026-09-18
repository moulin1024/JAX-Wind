"""Compare archived mean profiles from several controls against sim.pdf Fig. 9.

Inputs are folders produced by compare_spray_figure9.py. No simulation is run.
Paper curves are approximate raster digitizations of steady CFD speed, whereas
archive means contain axial velocity. Core and outer-region errors are separate.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("profiles", nargs="+", help="LABEL=profile_directory")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    entries = []
    for item in args.profiles:
        label, path = item.split("=", 1)
        path = Path(path)
        with np.load(path / "profiles.npz") as a:
            data = {k: a[k] for k in a.files}
        with (path / "centreline_comparison.csv").open() as f:
            rows = list(csv.DictReader(f))
        entries.append((label, path, data, rows))
    reference = entries[0][2]
    stations = reference["stations_m"]
    zp = reference["paper_z_relative_m"]
    paper = reference["paper_profiles"][0]
    records = []
    fig, axes = plt.subplots(
        1, len(stations), figsize=(18, 4.8), sharey=True, layout="constrained"
    )
    centrefig, (ca, wa) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    pc = [float(r["paper_centre_speed_m_s"]) for r in entries[0][3]]
    pw = [float(r["paper_half_excess_width_m"]) for r in entries[0][3]]
    ca.plot(stations, pc, "ko-", label="Paper Fig. 9")
    wa.plot(stations, pw, "ko-", label="Paper Fig. 9")
    for i, ax in enumerate(axes):
        ax.plot(paper[i], zp, "k-", label="Paper Fig. 9")
        ax.set(
            title=f"x = {stations[i]:.1f} m",
            xlabel="Velocity (m/s)",
            xlim=(0, 13),
            ylim=(-0.3, 0.3),
        )
        ax.grid(alpha=0.2)
    for label, path, data, rows in entries:
        np.testing.assert_allclose(data["stations_m"], stations)
        np.testing.assert_allclose(data["paper_profiles"][0], paper, equal_nan=True)
        z = data["z_relative_m"]
        mean = data["means"][0]
        for i, ax in enumerate(axes):
            ax.plot(mean[:, i], z, label=label)
            predicted = np.interp(zp, z, mean[:, i])
            error = predicted - paper[i]
            rec = {
                "run": label,
                "x_m": float(stations[i]),
                "centre_axial_m_s": float(rows[i]["our_mean_centre_axial_m_s"]),
                "paper_centre_speed_m_s": float(rows[i]["paper_centre_speed_m_s"]),
                "half_excess_width_m": float(rows[i]["our_half_excess_width_m"]),
            }
            for region, mask in [
                ("core", abs(zp) <= 0.05),
                ("outer", (abs(zp) >= 0.1) & (abs(zp) <= 0.25)),
            ]:
                rec[region + "_profile_rmse_m_s"] = float(
                    np.sqrt(np.mean(error[mask] ** 2))
                )
            records.append(rec)
        ca.plot(
            stations,
            [float(r["our_mean_centre_axial_m_s"]) for r in rows],
            "o-",
            label=label,
        )
        wa.plot(
            stations,
            [float(r["our_half_excess_width_m"]) for r in rows],
            "o-",
            label=label,
        )
    axes[0].set_ylabel("Height relative to nozzle (m)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Saved 3–4 s axial means versus steady CFD speed in sim.pdf; approximate digitization"
    )
    ca.set(xlabel="x (m)", ylabel="Centre velocity (m/s)")
    wa.set(xlabel="x (m)", ylabel="Half-peak-excess width (m)")
    ca.legend(fontsize=8)
    for ax in (ca, wa):
        ax.grid(alpha=0.2)
    centrefig.suptitle(
        "Core amplitude and width; half excess is measured above inlet 3 m/s"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "velocity_profiles.png", dpi=160)
    centrefig.savefig(args.output / "core_amplitude_width.png", dpi=160)
    with (args.output / "velocity_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "inputs": {label: str(path) for label, path, _, _ in entries},
                "records": records,
                "limitations": __doc__,
            },
            indent=2,
        )
        + "\n"
    )
    plt.close("all")


if __name__ == "__main__":
    main()
