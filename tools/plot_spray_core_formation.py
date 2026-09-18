"""Distinguish centreline growth from transverse-peak decay in saved profiles."""

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
    ap.add_argument("profiles", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    a = np.load(args.profiles / "profiles.npz")
    z = a["z_relative_m"]
    x = a["stations_m"]
    u = a["means"][0]
    mask = abs(z) < 0.08
    rows = []
    for i, s in enumerate(x):
        imax = np.argmax(u[mask, i])
        peak = float(u[mask, i][imax])
        centre = float(np.interp(0, z, u[:, i]))
        rows.append(
            {
                "x_m": float(s),
                "centre_axial_m_s": centre,
                "maximum_axial_within_8cm_m_s": peak,
                "maximum_z_m": float(z[mask][imax]),
                "peak_minus_centre_m_s": peak - centre,
            }
        )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for i, ax in enumerate(axes[:2]):
        ax.plot(
            a["paper_profiles"][0, i],
            a["paper_z_relative_m"] * 1000,
            "k-",
            label="Paper Fig. 9 speed",
        )
        ax.plot(
            u[:, i],
            z * 1000,
            "o-",
            ms=3,
            color="#c84430",
            label="Production mean axial u",
        )
        ax.axhline(0, color="grey", ls=":", lw=1)
        ax.set(
            xlabel="Velocity (m/s)",
            ylabel="Height relative to nozzle (mm)",
            title=f"x = {x[i]:.1f} m",
            ylim=(-45, 45),
            xlim=(0, 15),
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    ax = axes[2]
    ax.plot(x, [r["centre_axial_m_s"] for r in rows], "o-", label="Centreline")
    ax.plot(
        x,
        [r["maximum_axial_within_8cm_m_s"] for r in rows],
        "s-",
        label="Maximum within ±8 cm",
    )
    ax.set(
        xlabel="x (m)",
        ylabel="Mean axial velocity (m/s)",
        title="Core filling versus peak-speed decay",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.suptitle(
        "Existing uniform 256 × 128 × 128 result, 3–4 s mean; no additional simulation"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "core_formation.png", dpi=180)
    plt.close(fig)
    with (args.output / "core_formation.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "records": rows,
                "scope": "Vertical centre-plane means. Off-axis maxima are discrete cell-centre samples; centreline is interpolated. These profiles identify core filling but do not close a mean momentum budget or establish the physical cause of insufficient mixing.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
