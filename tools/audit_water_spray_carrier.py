"""Sampled carrier heat/water budgets for completed inertial spray runs.

Boundary fluxes are integrated from saved diagnostics with trapezoidal
quadrature; these residuals are not exact RK-stage conservation ledgers.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def wet_bulb_c(dry_bulb_c, vapor, pressure, epsilon):
    """Invert the benchmark's stated ventilated-psychrometer reconstruction."""
    vapor_pressure = pressure * vapor / (epsilon + vapor)

    def residual(wet):
        t = wet + 273.15
        log_p = (
            54.842763
            - 6763.22 / t
            - 4.210 * np.log(t)
            + 0.000367 * t
            + np.tanh(0.0415 * (t - 218.8))
            * (53.878 - 1331.22 / t - 9.44523 * np.log(t) + 0.014025 * t)
        )
        return (
            np.exp(log_p)
            - 0.00066 * (1 + 0.00115 * wet) * pressure * (dry_bulb_c - wet)
            - vapor_pressure
        )

    return brentq(residual, 0.0, dry_bulb_c)


def audit(directory, window=1.0):
    from jaxwind.config.moisture import load_moisture
    from jaxwind.io.checkpoint import checkpoint_metadata

    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError(f"Incomplete run: {directory}")
    end = summary["time_seconds"]
    with (directory / "history.csv").open() as stream:
        rows = [
            {k: float(v) for k, v in row.items() if v} for row in csv.DictReader(stream)
        ]
    rows = [
        r for r in rows if end - window - 1e-8 <= r["time_hours"] * 3600 <= end + 1e-8
    ]
    times = np.array([r["time_hours"] * 3600 for r in rows])
    if len(times) < 4 or not np.isclose(
        times[-1] - times[0], window, atol=1e-8, rtol=0
    ):
        raise ValueError("Requested interval not covered by diagnostics")
    header = checkpoint_metadata(directory / "checkpoint.npz")
    doc = header["resolved_case"]
    moist, _ = load_moisture(doc["physics"])
    config = moist.thermodynamics
    _, ly, lz = doc["mesh"]["lengths_m"]
    dry_inflow = (
        config.dry_air_density
        * ly
        * lz
        * doc["physics"]["flow"]["streamwise_velocity_m_s"]
    )

    def rate(key):
        return (rows[-1][key] - rows[0][key]) / window

    def mean_flux(key):
        return float(np.trapezoid([r[key] for r in rows], times) / window)

    heat_source = rate("parcel_gas_sensible_energy_loss_j")
    storage = rate("gas_sensible_anomaly_j")
    cooling = mean_flux("cooling_power_w")
    water_storage = rate("water_inventory_kg")
    water_out = mean_flux("vapor_outflow_kg_s")
    water_in = dry_inflow * rows[0]["inlet_vapor_mixing_ratio"]
    evaporation = rate("parcel_evaporated_mass_kg")
    predicted_wbt = np.array(
        [
            [
                wet_bulb_c(
                    r[f"sensor_{i}_dbt_c"],
                    r[f"sensor_{i}_vapor_kg_kg"],
                    config.pressure,
                    config.dry_air_gas_constant / config.water_vapor_gas_constant,
                )
                for i in range(9)
            ]
            for r in rows
        ]
    )
    with np.load(directory / "checkpoint.npz", allow_pickle=False) as data:
        cloud = {
            name: float(np.max(data[f"state/moisture/{name}"]))
            for name in ("cloud_liquid", "cloud_ice")
        }
    return {
        "directory": str(directory),
        "window_s": [float(times[0]), float(times[-1])],
        "samples": len(rows),
        "maximum_sample_interval_s": float(np.max(np.diff(times))),
        "gas_sensible_storage_w": storage,
        "parcel_heat_sink_w": heat_source,
        "outlet_sensible_cooling_w": cooling,
        "sensible_residual_w": storage + heat_source - cooling,
        "water_storage_kg_s": water_storage,
        "vapor_inflow_kg_s": water_in,
        "vapor_outflow_kg_s": water_out,
        "parcel_evaporation_kg_s": evaporation,
        "water_residual_kg_s": water_storage + water_out - water_in - evaporation,
        "predicted_sensor_wbt_c": predicted_wbt.mean(axis=0).tolist(),
        "final_maximum_cloud_mixing_ratio": cloud,
        "interpretation": (
            "Sampled boundary-flux quadrature, not an exact stage ledger. Sensible balance "
            "assumes no cloud phase-change heating; final cloud maxima are reported but do "
            "not prove this for the entire interval. Water balance omits any condensate "
            "outflow. WBT uses the same psychrometric relation as inlet reconstruction; "
            "simulation station remains upstream of the unmodeled drift eliminator."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [audit(d) for d in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
