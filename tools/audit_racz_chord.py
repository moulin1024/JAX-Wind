"""Assess a frozen optical chord hypothesis on source-only temporal splits."""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from racz_chord import (
    admissible,
    assess_chords,
    fit_nonnegative_radius_squared,
    fit_radius_squared,
    radius_squared,
    temporal_training_mask,
)
from racz_pda import read_station, source_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fit-mode", choices=["unconstrained", "nonnegative"], default="unconstrained"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = root / "cases/WaterSprayRacz2025/chord_protocol.json"
    cfg = json.loads(protocol.read_text())
    hashes = {
        q["local_name"]: q["sha256"]
        for q in json.loads((args.data / "provenance.json").read_text())["members"]
    }
    paths = source_files(args.data)
    if len(paths) != cfg["expected_stations"]:
        raise ValueError("unexpected source station count")
    records = []
    lower, upper = np.array(cfg["diameter_range_um"]) * 1e-6
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != hashes[path.name]:
            raise ValueError("checksum mismatch")
        station = read_station(path, source_only=True)
        a, h = station.samples, station.header
        d, u, tt = a[:, 7], a[:, 3], a[:, 2]
        subset = (d >= lower) & (d <= upper)
        angle = np.degrees(np.arctan2(np.abs(a[:, 4]), np.abs(u)))
        record = {
            "file": path.name,
            "sha256": digest,
            "traverse": h.traverse,
            "position_m": h.position_m,
            "events": len(a),
            "outside_fit_size_range": int(np.sum(~subset)),
            "minimum_angle_above_5_degrees_events": int(
                np.sum(angle > cfg["angle_diagnostic_degrees"])
            ),
            "splits": [],
        }
        for blocks in cfg["time_blocks"]:
            train_mask = temporal_training_mask(a[:, 1], blocks)
            train, check = subset & train_mask, subset & ~train_mask
            coefficients = fit_radius_squared(d[train], u[train], tt[train])
            if args.fit_mode == "nonnegative":
                coefficients = fit_nonnegative_radius_squared(
                    d[train], u[train], tt[train], lower
                )
                # The radius may tend to zero at the domain's lower endpoint,
                # but every observed event must have strictly positive support.
                valid = bool(
                    coefficients[0] > 0
                    and np.all(radius_squared(coefficients, d[subset]) > 0)
                )
            else:
                valid = admissible(coefficients, [lower, upper])
            result = {
                "blocks": blocks,
                "training_events": int(np.sum(train)),
                "check_events": int(np.sum(check)),
                "coefficients_um2": coefficients.tolist(),
                "positive_over_declared_size_range": valid,
                "radius_squared_at_range_endpoints_um2": radius_squared(
                    coefficients, [lower, upper]
                ).tolist(),
                "observed_nonpositive_radius_events": int(
                    np.sum(radius_squared(coefficients, d[subset]) <= 0)
                ),
                "implied_zero_radius_diameter_um": float(
                    np.exp(-coefficients[1] / coefficients[0])
                )
                if coefficients[0] > 0
                else None,
                "check": None,
                "size_bins": [],
            }
            if valid:
                result["check"] = assess_chords(
                    coefficients, d[check], u[check], tt[check]
                )
                for lo, hi in pairwise(np.array(cfg["diameter_bin_edges_um"]) * 1e-6):
                    selected = (
                        check & (d >= lo) & ((d < hi) if hi < upper else (d <= hi))
                    )
                    n = int(np.sum(selected))
                    entry = {
                        "diameter_low_m": lo,
                        "diameter_high_m": hi,
                        "events": n,
                        "sufficient_for_diagnostic": n >= cfg["minimum_bin_events"],
                        "check": None,
                    }
                    if n:
                        entry["check"] = assess_chords(
                            coefficients, d[selected], u[selected], tt[selected]
                        )
                    result["size_bins"].append(entry)
            record["splits"].append(result)
        records.append(record)
    summary = []
    for blocks in cfg["time_blocks"]:
        splits = [
            next(s for s in r["splits"] if s["blocks"] == blocks) for r in records
        ]
        valid = [s for s in splits if s["positive_over_declared_size_range"]]
        sufficient_bins = [
            b for s in valid for b in s["size_bins"] if b["sufficient_for_diagnostic"]
        ]
        summary.append(
            {
                "blocks": blocks,
                "admissible_station_fits": len(valid),
                "nonpositive_station_fits": len(splits) - len(valid),
                "check_events_admissible_fits": sum(
                    s["check"]["events"] for s in valid
                ),
                "support_exceedances_admissible_fits": sum(
                    s["check"]["support_exceedances"] for s in valid
                ),
                "cdf_distance_range": [
                    min(s["check"]["cdf_distance"] for s in valid),
                    max(s["check"]["cdf_distance"] for s in valid),
                ]
                if valid
                else None,
                "supported_size_bins": len(sufficient_bins),
                "size_bins_second_moment_error_above_10_percent": sum(
                    abs(b["check"]["relative_second_moment_error"]) > 0.1
                    for b in sufficient_bins
                ),
            }
        )
    report = {
        "status": cfg["status"],
        "fit_mode": args.fit_mode,
        "protocol": cfg,
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "stations": len(records),
        "events": sum(r["events"] for r in records),
        "outside_fit_size_range": sum(r["outside_fit_size_range"] for r in records),
        "minimum_angle_above_5_degrees_events": sum(
            r["minimum_angle_above_5_degrees_events"] for r in records
        ),
        "summary": summary,
        "records": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("records", "protocol")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
