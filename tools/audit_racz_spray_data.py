#!/usr/bin/env python3
"""Audit measured source/holdout data; this is not a simulated validation pass."""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from racz_pda import read_station


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((args.data / "provenance.json").read_text())
    rows, source, holdout = [], [], []
    for member in provenance["members"]:
        if (
            not member["local_name"].endswith(".txt")
            or "README" in member["local_name"]
        ):
            continue
        path = args.data / member["local_name"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != member["sha256"]:
            raise ValueError(f"Data checksum mismatch: {path}")
        station = read_station(path)
        h, q = station.header, station.samples
        valid = (
            np.all(np.isfinite(q), axis=1)
            & (q[:, 1] >= 0)
            & (q[:, 2] > 0)
            & (q[:, 7] > 0)
        )
        measured = q[valid]
        if not len(measured):
            raise ValueError(f"No physically usable records: {path}")
        d, u, transverse = measured[:, 7], measured[:, 3], measured[:, 4]
        rows.append(
            {
                "file": path.name,
                "point": h.point,
                "role": h.role,
                "traverse": h.traverse,
                "acquisition_label": h.acquisition_label,
                "x_m": h.position_m[0],
                "y_m": h.position_m[1],
                "z_m": h.position_m[2],
                "transverse_component": h.transverse_component,
                "events": len(q),
                "invalid_events": int(np.sum(~valid)),
                "nonmonotone_arrivals": int(np.sum(np.diff(q[:, 1]) < 0)),
                "sub_wavelength_events": int(np.sum(d < 0.5145e-6)),
                "above_reported_size_range_events": int(np.sum(d > 64.1e-6)),
                "negative_axial_events": int(np.sum(u < 0)),
                "last_valid_arrival_s": float(np.max(measured[:, 1])),
                "event_axial_mean_m_s": float(np.mean(u)),
                "event_axial_rms_m_s": float(np.std(u)),
                "event_transverse_mean_m_s": float(np.mean(transverse)),
                "event_transverse_rms_m_s": float(np.std(transverse)),
                "event_D10_m": float(np.mean(d)),
                "event_D32_m": float(np.sum(d**3) / np.sum(d**2)),
            }
        )
        (source if h.role == "source" else holdout).append(
            {
                "file": path.name,
                "sha256": digest,
                "traverse": h.traverse,
                "acquisition_label": h.acquisition_label,
                "position_m": h.position_m,
                "events": len(q),
            }
        )
    layout = Counter((round(r["z_m"] * 1000), r["traverse"]) for r in rows)
    expected = {
        (20, "X"): 17,
        (20, "Y"): 17,
        (40, "X"): 13,
        (40, "Y"): 13,
        (60, "X"): 15,
        (60, "Y"): 15,
    }
    if layout != expected:
        raise ValueError(f"Unexpected station layout: {layout}")
    with (args.output / "station_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "dataset_doi": provenance["dataset_doi"],
        "condition": provenance["condition"],
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("racz_pda.py"),
                Path("cases/WaterSprayRacz2025/protocol.json"),
            )
        },
        "provenance_sha256": hashlib.sha256(
            (args.data / "provenance.json").read_bytes()
        ).hexdigest(),
        "stations": len(rows),
        "events": sum(r["events"] for r in rows),
        "invalid_events": sum(r["invalid_events"] for r in rows),
        "sub_wavelength_events": sum(r["sub_wavelength_events"] for r in rows),
        "above_reported_size_range_events": sum(
            r["above_reported_size_range_events"] for r in rows
        ),
        "source_stations": len(source),
        "holdout_stations": len(holdout),
        "source": source,
        "holdout": holdout,
        "interpretation": "Event-weighted statistics after explicit finite/positive-size/transit checks; no outlier, transit-time, probe-volume or detection-efficiency correction. Not inlet flux distributions.",
        "transport_validation_pass": None,
        "qualified_for_simulation": False,
        "open_items": [
            "carrier velocity/stress input at source",
            "size-dependent detection/probe-volume bias and mass-flux weights",
            "axisymmetry and missing third joint velocity component",
            "breakup/evaporation applicability",
            "source geometry/mass-flow discrepancies in publications",
            "uncertainty and near-zero observable criteria",
        ],
    }
    (args.output / "audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("source", "holdout")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
