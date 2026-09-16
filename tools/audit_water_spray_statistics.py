"""Report temporal block stability of completed spray sensor histories.

Block variation is a stationarity diagnostic, not a confidence interval: blocks
can remain correlated and a short run cannot establish ensemble convergence.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def audit(directory, start=2.0, block_seconds=1.0):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError(f"Incomplete run: {directory}")
    end = float(summary["time_seconds"])
    if block_seconds <= 0 or not 0 <= start < end:
        raise ValueError("Require positive blocks and a start before the run ends")
    with (directory / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([float(r["time_hours"]) * 3600 for r in rows])
    sensors = np.array(
        [[float(r[f"sensor_{i}_dbt_c"]) for i in range(9)] for r in rows]
    )
    if np.any(np.diff(times) <= 0):
        raise ValueError("Diagnostics must have strictly increasing times")
    if times[0] > start + 1e-8 or times[-1] < end - 1e-8:
        raise ValueError("Requested interval is not covered")

    def mean_interval(a, b):
        inside = (times > a + 1e-8) & (times < b - 1e-8)
        nodes = np.r_[a, times[inside], b]
        values = np.stack(
            [np.interp(nodes, times, sensors[:, i]) for i in range(9)], axis=1
        )
        return np.trapezoid(values, nodes, axis=0) / (b - a)

    blocks = []
    for i in range(int(np.floor((end - start + 1e-8) / block_seconds))):
        a, b = start + i * block_seconds, start + (i + 1) * block_seconds
        blocks.append(
            {"interval_s": [a, b], "sensor_dbt_c": mean_interval(a, b).tolist()}
        )
    if len(blocks) < 2:
        raise ValueError("Require at least two complete blocks")
    means = np.asarray([b["sensor_dbt_c"] for b in blocks])
    middle = (start + end) / 2
    split_change = mean_interval(middle, end) - mean_interval(start, middle)
    return {
        "directory": str(directory),
        "interval_s": [start, end],
        "block_seconds": block_seconds,
        "block_count": len(blocks),
        "blocks": blocks,
        "sensor_dbt_c": mean_interval(start, end).tolist(),
        "sensor_block_range_k": np.ptp(means, axis=0).tolist(),
        "sensor_last_minus_previous_block_k": (means[-1] - means[-2]).tolist(),
        "sensor_late_minus_early_half_k": split_change.tolist(),
        "maximum_absolute_split_change_k": float(np.max(np.abs(split_change))),
        "interpretation": (
            "Trapezoidal time means with linearly interpolated interval endpoints. "
            "Not the equally weighted diagnostic means used by the archived comparison. "
            "Block spread is not a confidence interval or proof of stationarity; "
            "serial correlation and independent inlet-realization uncertainty remain."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--start", type=float, default=2.0)
    parser.add_argument("--block-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [audit(d, args.start, args.block_seconds) for d in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(result["directory"], result["maximum_absolute_split_change_k"])


if __name__ == "__main__":
    main()
