"""Plot all temporal splits of the source-only optical calibration assessment."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.report_directory
    unconstrained = json.loads((directory / "report.json").read_text())
    constrained = json.loads((directory / "nonnegative/report.json").read_text())
    assert unconstrained["protocol_sha256"] == constrained["protocol_sha256"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, layout="constrained")
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for row, traverse in enumerate(("X", "Y")):
        key = lambda r, axis=row: r["position_m"][axis] * 1000
        raw = sorted(
            (r for r in unconstrained["records"] if r["traverse"] == traverse), key=key
        )
        bounded = sorted(
            (r for r in constrained["records"] if r["traverse"] == traverse), key=key
        )
        for blocks, color in zip(unconstrained["protocol"]["time_blocks"], colors):
            first = [next(s for s in r["splits"] if s["blocks"] == blocks) for r in raw]
            second = [
                next(s for s in r["splits"] if s["blocks"] == blocks) for r in bounded
            ]
            assert all(s["check"] is not None for s in second)
            x = [key(r) for r in raw]
            series = (
                [s["implied_zero_radius_diameter_um"] for s in first],
                [s["check"]["cdf_distance"] for s in second],
                [100 * s["check"]["support_exceedance_fraction"] for s in second],
            )
            for col, values in enumerate(series):
                axes[row, col].plot(
                    x,
                    values,
                    ".-",
                    label=f"{blocks} time blocks",
                    color=color,
                    linewidth=1,
                )
                axes[row, col].grid(alpha=0.2)
                axes[row, col].set_xlabel(f"{traverse} traverse coordinate (mm)")
        axes[row, 0].axhline(
            0.5145,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Fit range lower bound",
        )
        axes[row, 0].set_ylabel("Unconstrained implied cutoff (µm)")
        axes[row, 1].set_ylabel("Nonnegative fit: chord CDF distance")
        axes[row, 2].set_ylabel("Nonnegative fit: outside support (%)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Cutoff lies inside the measured size range")
    axes[0, 1].set_title("Temporal check: distribution mismatch")
    axes[0, 2].set_title("Temporal check: chord-limit exceedances")
    fig.suptitle(
        "Rácz 20 mm source: conditional axial-chord model assessment\nNo correction applied; downstream transport holdout unused",
        fontsize=12,
    )
    fig.savefig(directory / "source_chord_assessment.png", dpi=160)
    fig.savefig(directory / "source_chord_assessment.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
