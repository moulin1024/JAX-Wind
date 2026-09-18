#!/usr/bin/env python3
"""Assess paired centre and off-centre sensor errors for completed inertial controls."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from overlay_sim_fig7 import average, enthalpy, humidity, wet_bulb_c

ROOT = Path(__file__).resolve().parents[1]


def assess(directory, start=3.0, end=4.0):
    import tomllib

    doc = tomllib.loads((directory / "resolved_case.toml").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete" or summary["time_seconds"] < end - 1e-8:
        raise ValueError("Run must be complete and span the requested averaging window")
    if doc["case"]["spray_model"] != "inertial":
        raise ValueError("Expected inertial parcel model")
    case = int(doc["physics"]["flow"]["streamwise_velocity_m_s"])
    refs = json.loads(
        (ROOT / "cases/WaterSprayMontazeri2015/sim_fig7/reference.json").read_text()
    )
    ref = refs["cases"][case - 1]
    with (directory / "history.csv").open() as f:
        rows = [{k: float(v) for k, v in r.items() if v} for r in csv.DictReader(f)]
    t = np.array([r["time_hours"] * 3600 for r in rows])
    if np.any(np.diff(t) <= 0):
        raise ValueError("History must be strictly increasing")
    td = np.array([[r[f"sensor_{i}_dbt_c"] for i in range(9)] for r in rows])
    q = np.array([[r[f"sensor_{i}_vapor_kg_kg"] for i in range(9)] for r in rows])
    tw = np.asarray(wet_bulb_c(td, q, 101325.0, 287.05 / 461.5))
    predicted = np.array([average(t, v, start, end) for v in (td, tw, enthalpy(td, q))])
    ed, ew = np.array(ref["experimental_dbt_c"]), np.array(ref["experimental_wbt_c"])
    expected = np.array([ed, ew, enthalpy(ed, humidity(ed, ew))])
    error = predicted - expected
    off = np.arange(9) != 4
    return {
        "case": case,
        "run": str(directory),
        "window_seconds": [start, end],
        "mesh": doc["mesh"]["cells"],
        "dt": doc["time"]["dt_seconds"],
        "carrier_model": doc["case"].get("carrier_turbulence_model", "les"),
        "physical_transverse_inlet": doc["case"].get("physical_transverse_inlet", False),
        "sgs": doc["case"].get("carrier_sgs_model", "amd")
        if doc["case"].get("carrier_turbulence_model", "les") == "les"
        else None,
        "momentum_scheme": doc["numerics"].get("momentum_advection_scheme", "muscl-mc"),
        "scalar_scheme": doc["numerics"].get("scalar_advection_scheme", "upwind"),
        "scalar_transport_closure": doc["case"].get("carrier_scalar_transport", "shared-diffusivity"),
        "quantities": ["DBT_C", "WBT_C", "h_kJ_kg_dry_air"],
        "predicted": predicted.tolist(),
        "experimental": expected.tolist(),
        "centre_prediction": predicted[:, 4].tolist(),
        "centre_error": error[:, 4].tolist(),
        "offcentre_rmse": np.sqrt(np.mean(error[:, off] ** 2, axis=1)).tolist(),
        "all_sensor_rmse": np.sqrt(np.mean(error**2, axis=1)).tolist(),
        "mean_humidity_g_kg": (average(t, q, start, end) * 1000).tolist(),
        "centre_half_window_shift": [
            float(
                (
                    average(t, v, (start + end) / 2, end)
                    - average(t, v, start, (start + end) / 2)
                )[4]
            )
            for v in (td, tw, enthalpy(td, q))
        ],
        "maximum_cfl": max(r["maximum_cfl"] for r in rows),
        "maximum_mass_residual_kg": max(
            abs(r["parcel_mass_balance_error_kg"]) for r in rows
        ),
        "maximum_parcel_enthalpy_residual_J": max(
            abs(r["parcel_enthalpy_balance_error_j"]) for r in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=3.0)
    parser.add_argument("--end", type=float, default=4.0)
    args = parser.parse_args()
    results = [assess(p, args.start, args.end) for p in args.runs]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "| Run | Centre DBT / WBT / h | Centre errors | Other-eight RMSE |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        fmt = lambda key, r=r: " / ".join(f"{v:.3f}" for v in r[key])
        lines.append(
            f"| {Path(r['run']).name} | {fmt('centre_prediction')} | {fmt('centre_error')} | {fmt('offcentre_rmse')} |"
        )
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
