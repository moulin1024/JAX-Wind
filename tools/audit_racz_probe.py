"""Source-only compatibility of recorded transits with nominal probe support."""

import argparse
import csv
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from racz_pda import read_station, source_files
from racz_probe import scenario_ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "cases/WaterSprayRacz2025/probe_protocol.json"
    cfg = json.loads(protocol_path.read_text())
    hashes = {
        q["local_name"]: q["sha256"]
        for q in json.loads((args.data / "provenance.json").read_text())["members"]
    }
    paths = source_files(args.data)
    if len(paths) != cfg["expected_stations"]:
        raise ValueError("unexpected source station count")
    args.output.mkdir(parents=True, exist_ok=True)
    records, rows = [], []
    total = {name: 0 for name in cfg["scenarios"]}
    events = 0
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != hashes[path.name]:
            raise ValueError("checksum mismatch")
        station = read_station(path, source_only=True)
        a, h = station.samples, station.header
        ratios = scenario_ratios(a, cfg)
        n = len(a)
        events += n
        record = {
            "file": path.name,
            "sha256": digest,
            "traverse": h.traverse,
            "position_m": h.position_m,
            "events": n,
            "scenarios": {},
        }
        for name, values in ratios.items():
            bad = int(np.sum(values > 1))
            total[name] += bad
            record["scenarios"][name] = {
                "incompatible": bad,
                "fraction": bad / n,
                "ratio_quantiles_50_95_99_100": np.quantile(
                    values, [0.5, 0.95, 0.99, 1.0]
                ).tolist(),
            }
        edges = np.asarray(cfg["diameter_bin_edges_um"]) * 1e-6
        for lo, hi in pairwise(edges):
            selected = (a[:, 7] >= lo) & (a[:, 7] < hi)
            count = int(np.sum(selected))
            if count:
                for name, values in ratios.items():
                    rows.append(
                        {
                            "file": path.name,
                            "traverse": h.traverse,
                            "x_m": h.position_m[0],
                            "y_m": h.position_m[1],
                            "diameter_low_m": lo,
                            "diameter_high_m": hi,
                            "scenario": name,
                            "events": count,
                            "incompatible": int(np.sum(values[selected] > 1)),
                            "median_ratio": float(np.median(values[selected])),
                            "p99_ratio": float(np.quantile(values[selected], 0.99)),
                        }
                    )
        records.append(record)
    result = {
        "status": cfg["status"],
        "protocol": cfg,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "stations": len(records),
        "events": events,
        "summary": {
            name: {
                "incompatible": n,
                "fraction": n / events,
                "stations_with_incompatible_events": sum(
                    q["scenarios"][name]["incompatible"] > 0 for q in records
                ),
            }
            for name, n in total.items()
        },
        "records": records,
    }
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "size_bins.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
