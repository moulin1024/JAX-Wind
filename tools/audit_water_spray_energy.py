#!/usr/bin/env python3
"""Audit time-integrated parcel heat and outlet-liquid budgets."""

import argparse
import csv
import json
from pathlib import Path

from jaxwind.config.moisture import load_moisture
from jaxwind.io.checkpoint import checkpoint_metadata
from jaxwind.physics.moisture import WaterDropletProperties


def audit(directory, window=1.0):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Energy audit requires a completed run")
    with (directory / "history.csv").open() as stream:
        rows = [
            {k: float(v) for k, v in row.items() if v} for row in csv.DictReader(stream)
        ]
    end = summary["time_seconds"]
    samples = [r for r in rows if r["time_hours"] * 3600 >= end - window - 1e-8]
    if len(samples) < 2:
        raise ValueError("At least two samples required")
    first, last = samples[0], samples[-1]
    duration = (last["time_hours"] - first["time_hours"]) * 3600
    if abs(duration - window) > 1e-6:
        raise ValueError("Requested budget interval not covered by history")
    doc = checkpoint_metadata(directory / "checkpoint.npz")["resolved_case"]
    moist, _ = load_moisture(doc["physics"])
    c = moist.thermodynamics
    cp = WaterDropletProperties().liquid_heat_capacity

    def rate(key):
        return (last[key] - first[key]) / duration

    escaped = rate("parcel_escaped_mass_kg")
    escape_heat = rate("parcel_escaped_enthalpy_j")
    injected = rate("parcel_injected_enthalpy_j")
    gas_heat = rate("parcel_gas_sensible_energy_loss_j")
    vapor_latent = c.water_vapor_latent_heat * rate("parcel_evaporated_mass_kg")
    stored = rate("parcel_inventory_enthalpy_j")
    wall_metrics = {}
    if "parcel_first_wall_mass_kg" in last:
        impact = rate("parcel_first_wall_mass_kg")
        wall_escape = rate("parcel_escaped_wall_mass_kg")
        wall_h = rate("parcel_escaped_wall_enthalpy_j")
        wall_metrics = {
            "first_wall_impact_flow_kg_s": impact,
            "first_wall_impact_temperature_c": rate("parcel_first_wall_enthalpy_j")
            / (impact * cp)
            + c.freezing_temperature
            - 273.15,
            "escaped_wall_contact_flow_kg_s": wall_escape,
            "escaped_wall_contact_temperature_c": wall_h / (wall_escape * cp)
            + c.freezing_temperature
            - 273.15,
            "escaped_never_wall_flow_kg_s": escaped - wall_escape,
            "escaped_never_wall_temperature_c": (escape_heat - wall_h)
            / ((escaped - wall_escape) * cp)
            + c.freezing_temperature
            - 273.15,
            "after_first_wall_gas_sensible_heat_w": rate(
                "parcel_wall_gas_sensible_energy_loss_j"
            ),
            "after_first_wall_evaporation_kg_s": rate("parcel_wall_evaporated_mass_kg"),
            "maximum_wall_population_mass_residual_kg": max(
                abs(r["wall_population_mass_balance_error_kg"]) for r in rows
            ),
            "maximum_wall_population_enthalpy_residual_j": max(
                abs(r["wall_population_enthalpy_balance_error_j"]) for r in rows
            ),
        }
    return {
        **wall_metrics,
        "directory": str(directory),
        "interval_seconds": [first["time_hours"] * 3600, last["time_hours"] * 3600],
        "escaped_liquid_flow_kg_s": escaped,
        "escaped_liquid_temperature_c": escape_heat / (escaped * cp)
        + c.freezing_temperature
        - 273.15,
        "injected_liquid_enthalpy_w": injected,
        "escaped_liquid_enthalpy_w": escape_heat,
        "gas_to_droplets_sensible_heat_w": gas_heat,
        "evaporated_water_latent_enthalpy_w": vapor_latent,
        "evaporated_water_flow_kg_s": rate("parcel_evaporated_mass_kg"),
        "liquid_enthalpy_storage_rate_w": stored,
        "energy_rate_residual_w": stored
        + escape_heat
        + vapor_latent
        - injected
        - gas_heat,
        "maximum_cumulative_enthalpy_residual_j": max(
            abs(r["parcel_enthalpy_balance_error_j"]) for r in rows
        ),
        "maximum_cumulative_water_residual_kg": max(
            abs(r["parcel_mass_balance_error_kg"]) for r in rows
        ),
        "interpretation": "Event-integrated x-boundary escape temperature, before drift plates or collector. Not necessarily the experimental collected-water observable. Side-wall droplets continue moving/evaporating under the current idealized wet-wall rule.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
