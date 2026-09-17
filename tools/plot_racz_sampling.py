"""Plot source-only weighting scenarios and block-bootstrap range envelopes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 7), sharex=True, constrained_layout=True
    )
    for column, traverse in enumerate(("X", "Y")):
        data = sorted(
            (r for r in report["records"] if r["traverse"] == traverse),
            key=lambda r: r["position_m"][column],
        )
        x = np.array([r["position_m"][column] for r in data]) * 1e3
        for row, (observable, scale, label) in enumerate(
            (("axial_mean_m_s", 1.0, "Axial mean (m/s)"), ("D32_m", 1e6, "D32 (µm)"))
        ):
            ax = axes[row, column]
            for weighting, color, legend in (
                ("event", "#666666", "Event weighted"),
                ("transit_time", "#0072B2", "Transit-time weighted"),
            ):
                scenarios = [
                    [q["statistics"][observable] for q in r["scenarios"][weighting]]
                    for r in data
                ]
                y = np.array([q[0]["estimate"] for q in scenarios]) * scale
                low = (
                    np.array([min(s["bootstrap_low"] for s in q) for q in scenarios])
                    * scale
                )
                high = (
                    np.array([max(s["bootstrap_high"] for s in q) for q in scenarios])
                    * scale
                )
                ax.plot(x, y, "o-", color=color, markersize=3, label=legend)
                ax.fill_between(x, low, high, color=color, alpha=0.18)
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(f"{traverse} source traverse · z = 20 mm")
            else:
                ax.set_xlabel("Signed traverse coordinate (mm)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Source sampling sensitivity — neither scenario is a qualified inlet\nShading: envelope of conditional bootstrap ranges over 8–64 time blocks",
        fontsize=12,
    )
    output = args.report.with_name("source_sampling.png")
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
