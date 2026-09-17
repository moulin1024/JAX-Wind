#!/usr/bin/env python3
"""Source-only gas proxy and sensitivity; no downstream targets are evaluated."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from racz_gas_proxy import estimate_gas_proxy
from racz_pda import read_station, source_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "cases/WaterSprayRacz2025/gas_proxy_protocol.json"
    cfg = json.loads(protocol_path.read_text())
    provenance_path = args.data / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    checksums = {m["local_name"]: m["sha256"] for m in provenance["members"]}
    paths = source_files(args.data)
    if len(paths) != 34:
        raise ValueError("Expected exactly 34 source stations")
    parameters = {
        "density": cfg["density_kg_m3"],
        "viscosity": cfg["viscosity_Pa_s"],
        "length_scale": cfg["length_scale_m"],
        "minimum_diameter": cfg["minimum_diameter_m"],
    }
    records = []
    masks = {}
    traces = []
    curves = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != checksums[path.name]:
            raise ValueError(f"Checksum mismatch: {path}")
        station = read_station(path, source_only=True)
        a = station.samples
        h = station.header
        if not np.all(np.isfinite(a)) or np.any(a[:, 1] < 0):
            raise ValueError(
                "Nonfinite source records or invalid arrival times require explicit audit"
            )
        row = {
            "file": path.name,
            "sha256": digest,
            "traverse": h.traverse,
            "transverse_component": h.transverse_component,
            "x_m": h.position_m[0],
            "y_m": h.position_m[1],
            "z_m": h.position_m[2],
            "raw_events": len(a),
        }
        estimates = {}
        for limit in cfg["stokes_limits"]:
            result = estimate_gas_proxy(
                a[:, 7], a[:, 3:5], **parameters, stokes_limit=limit
            )
            estimates[limit] = result
            traces.append(
                {"file": path.name, "stokes_limit": limit, "history": result.history}
            )
            row[f"speed_proxy_stk_{limit:g}_m_s"] = result.mean_speed
            row[f"count_stk_{limit:g}"] = int(np.sum(result.selected))
            row[f"max_selected_diameter_stk_{limit:g}_m"] = float(
                np.max(a[result.selected, 7])
            )
        nominal = estimates[cfg["nominal_stokes_limit"]]
        subset = a[nominal.selected]
        masks[path.stem] = np.flatnonzero(nominal.selected)
        row.update(
            {
                "prefilter_events": int(np.sum(nominal.preprocessed)),
                "nominal_count_screen_pass": len(subset) >= cfg["minimum_count_screen"],
                "selected_axial_mean_m_s": float(np.mean(subset[:, 3])),
                "selected_transverse_mean_m_s": float(np.mean(subset[:, 4])),
                "selected_axial_std_m_s": float(np.std(subset[:, 3], ddof=1)),
                "selected_transverse_std_m_s": float(np.std(subset[:, 4], ddof=1)),
                "selected_speed_std_m_s": float(
                    np.std(np.linalg.norm(subset[:, 3:5], axis=1), ddof=1)
                ),
            }
        )
        # Cumulative speed dispersion: diagnostic requested in paper Appendix A.
        base = a[nominal.preprocessed]
        order = np.argsort(base[:, 7], kind="stable")
        base = base[order]
        speeds = np.linalg.norm(base[:, 3:5], axis=1)
        cs = np.cumsum(speeds)
        cs2 = np.cumsum(speeds**2)
        edges = np.linspace(
            base[0, 7], base[-1, 7], round(2 * len(base) ** (1 / 3)) + 1
        )[1:]
        peak_std = -1.0
        peak_d = None
        for edge in edges:
            n = int(np.searchsorted(base[:, 7], edge, side="right"))
            if n < 2:
                continue
            mean = cs[n - 1] / n
            std = np.sqrt(max(0.0, (cs2[n - 1] - n * mean**2) / (n - 1)))
            curves.append(
                {
                    "file": path.name,
                    "diameter_limit_m": float(edge),
                    "count": n,
                    "mean_speed_m_s": float(mean),
                    "std_speed_m_s": float(std),
                }
            )
            if n >= cfg["minimum_count_screen"] and std > peak_std:
                peak_std = std
                peak_d = float(edge)
        row["cumulative_std_peak_diameter_m"] = peak_d
        row["cumulative_std_peak_m_s"] = float(peak_std) if peak_d is not None else None
        # Delete whole contiguous time intervals, not independently shuffled events.
        span = float(np.max(a[:, 1]) - np.min(a[:, 1]))
        blocks = (
            np.minimum(
                cfg["time_blocks"] - 1,
                ((a[:, 1] - np.min(a[:, 1])) / span * cfg["time_blocks"]).astype(int),
            )
            if span > 0
            else np.zeros(len(a), dtype=int)
        )
        leave_out = []
        for block in range(cfg["time_blocks"]):
            reduced = a[blocks != block]
            try:
                estimate = estimate_gas_proxy(
                    reduced[:, 7],
                    reduced[:, 3:5],
                    **parameters,
                    stokes_limit=cfg["nominal_stokes_limit"],
                )
                leave_out.append(estimate.mean_speed)
            except ValueError:
                continue
        row["successful_leave_block_out_runs"] = len(leave_out)
        row["leave_block_out_speed_min_m_s"] = min(leave_out) if leave_out else None
        row["leave_block_out_speed_max_m_s"] = max(leave_out) if leave_out else None
        records.append(row)
    for name, rows in [("source_proxy.csv", records), ("cumulative_speed.csv", curves)]:
        with (args.output / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(args.output / "selected_source_indices.npz", **masks)
    files = [
        Path(__file__),
        protocol_path,
        root / "tools/racz_gas_proxy.py",
        root / "tools/racz_pda.py",
        root / "tests/test_racz_gas_proxy.py",
        root / "tools/qualify_racz_gas_source.sbatch",
    ]
    report = {
        "job_id": os.getenv("SLURM_JOB_ID"),
        "dataset_doi": provenance["dataset_doi"],
        "protocol": cfg,
        "source_stations": len(records),
        "holdout_bodies_read": 0,
        "source_records": sum(r["raw_events"] for r in records),
        "source_stations_below_nominal_count_screen": sum(
            not r["nominal_count_screen_pass"] for r in records
        ),
        "nominal_selected_count_range": [
            min(r["count_stk_0.1"] for r in records),
            max(r["count_stk_0.1"] for r in records),
        ],
        "stations": records,
        "iteration_histories": traces,
        "carrier_independently_measured": False,
        "qualified_for_transport_validation": False,
        "transport_validation_pass": None,
        "remaining_requirements": [
            "sampling/probe bias and source mass-flux weights",
            "independent carrier/stress information or bounded proxy uncertainty",
            "source asymmetry and missing third joint velocity component",
            "breakup/evaporation applicability",
            "predeclared downstream physical observables and uncertainty",
        ],
        "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("stations", "iteration_histories", "sha256", "protocol")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
