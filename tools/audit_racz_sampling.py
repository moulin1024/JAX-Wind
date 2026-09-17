#!/usr/bin/env python3
"""Source-only transit-time sensitivity and temporal-block sampling audit."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from racz_pda import read_station, source_files
from racz_sampling import OBSERVABLES, block_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = root / "cases/WaterSprayRacz2025/sampling_protocol.json"
    config = json.loads(protocol.read_text())
    provenance = json.loads((args.data / "provenance.json").read_text())
    hashes = {q["local_name"]: q["sha256"] for q in provenance["members"]}
    paths = source_files(args.data)
    if len(paths) != config["expected_stations"]:
        raise ValueError("unexpected source station count")
    args.output.mkdir(parents=True, exist_ok=True)
    rows, reports = [], []
    for station_index, path in enumerate(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != hashes[path.name]:
            raise ValueError("source checksum mismatch")
        station = read_station(path, source_only=True)
        a = station.samples
        h = station.header
        record = {
            "file": path.name,
            "sha256": digest,
            "traverse": h.traverse,
            "position_m": h.position_m,
            "transverse_component": h.transverse_component,
            "events": len(a),
            "first_arrival_s": float(a[0, 1]),
            "last_arrival_s": float(a[-1, 1]),
            "scenarios": {},
        }
        for weighting in config["weighting"]:
            scenarios = []
            for blocks in config["equal_duration_blocks"]:
                result = block_summary(
                    a,
                    weighting,
                    blocks,
                    config["bootstrap_replicates"],
                    config["seed"] + station_index * 100 + blocks,
                    config["percentiles"],
                )
                scenarios.append(result)
                for name, values in result["statistics"].items():
                    rows.append(
                        {
                            "file": path.name,
                            "traverse": h.traverse,
                            "x_m": h.position_m[0],
                            "y_m": h.position_m[1],
                            "weighting": weighting,
                            "blocks": blocks,
                            "block_duration_s": result["block_duration_s"],
                            "observable": name,
                            **values,
                        }
                    )
            record["scenarios"][weighting] = scenarios
        reports.append(record)
        print(
            "source_sampling", station_index + 1, h.traverse, h.position_m, flush=True
        )
    summary = {}
    for name in OBSERVABLES:
        changes, halfwidths, scales = [], [], []
        for record in reports:
            scenarios = record["scenarios"]
            raw = scenarios["event"][0]["statistics"][name]["estimate"]
            weighted = scenarios["transit_time"][0]["statistics"][name]["estimate"]
            changes.append(weighted - raw)
            scales.append(abs(raw))
            for q in scenarios["event"]:
                s = q["statistics"][name]
                halfwidths.append(0.5 * (s["bootstrap_high"] - s["bootstrap_low"]))
        summary[name] = {
            "max_abs_weighting_change": float(np.max(np.abs(changes))),
            "max_relative_weighting_change": float(
                np.max(np.abs(changes) / np.maximum(scales, np.finfo(float).tiny))
            ),
            "max_event_bootstrap_halfwidth": float(max(halfwidths)),
        }
    result = {
        "status": "source-only sensitivity, not a qualified inlet or physical validation",
        "protocol": config,
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "stations": len(reports),
        "events": sum(q["events"] for q in reports),
        "summary": summary,
        "records": reports,
    }
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "statistics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
