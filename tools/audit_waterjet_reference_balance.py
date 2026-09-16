"""Conditional apparatus heat/water balance from the uploaded measurements.

Nine equally weighted outlet patches and uniform dry-air flux are assumptions,
not measured outlet velocities. The enthalpy convention is the solver's dilute
constant-property convention. This is a consistency diagnostic, not validation.
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from jaxwind.physics.moisture import MoistureConfig, WaterDropletProperties
from jaxwind.simulation.water_spray_benchmark import inlet_mixing_ratio


def balance(reference, *, outlet_wbt_shift=0.0, outlet_dbt_shift=0.0):
    jax.config.update("jax_enable_x64", True)
    conditions = reference["operating_conditions"]
    config = MoistureConfig(pressure=101325.0, dry_air_density=1.125)
    cp_liquid = WaterDropletProperties().liquid_heat_capacity
    inlet_dbt = conditions["inlet_dbt_c"]
    qin = inlet_mixing_ratio(
        {"dry_bulb_c": inlet_dbt, "wet_bulb_c": conditions["inlet_wbt_c"]},
        config,
    )
    dbt = np.asarray(reference["experimental_dbt_c"]) + outlet_dbt_shift
    wbt = np.asarray(reference["experimental_wbt_c"]) + outlet_wbt_shift
    qout = np.array(
        [
            inlet_mixing_ratio({"dry_bulb_c": t, "wet_bulb_c": w}, config)
            for t, w in zip(dbt, wbt)
        ]
    )
    _, ly, lz = conditions["tunnel_dimensions_m"]
    dry_flow = config.dry_air_density * ly * lz * conditions["air_speed_m_s"]
    water_in = conditions["water_flow_l_min"] / 60000 * config.water_density
    evaporation = dry_flow * (qout.mean() - qin)
    water_out = water_in - evaporation
    sensible_cooling = (
        dry_flow * config.dry_air_heat_capacity * (inlet_dbt - dbt.mean())
    )
    liquid_heat_loss = cp_liquid * (
        water_in * conditions["water_inlet_c"]
        - water_out * conditions["water_outlet_c"]
    )
    vapor_heat_gain = config.water_vapor_latent_heat * evaporation
    return {
        "outlet_dbt_shift_c": outlet_dbt_shift,
        "outlet_wbt_shift_c": outlet_wbt_shift,
        "dry_air_flow_kg_s": dry_flow,
        "inlet_vapor_kg_kg": qin,
        "sensor_vapor_kg_kg": qout.tolist(),
        "equal_patch_mean_vapor_kg_kg": float(qout.mean()),
        "implied_evaporation_kg_s": float(evaporation),
        "implied_collected_water_flow_kg_s": float(water_out),
        "air_sensible_cooling_w": float(sensible_cooling),
        "vapor_latent_gain_w": float(vapor_heat_gain),
        "liquid_enthalpy_loss_w": float(liquid_heat_loss),
        "out_minus_in_enthalpy_residual_w": float(
            vapor_heat_gain - sensible_cooling - liquid_heat_loss
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "cases/WaterSprayMontazeri2015/uploaded_reference_validation/reference.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    result = {
        "reference": str(args.reference),
        "baseline": balance(reference),
        "uniform_outlet_wbt_shift_sensitivities": [
            balance(reference, outlet_wbt_shift=s) for s in (-0.3, 0.3)
        ],
        "uniform_outlet_dbt_shift_sensitivities": [
            balance(reference, outlet_dbt_shift=s) for s in (-0.3, 0.3)
        ],
        "interpretation": (
            "Uniform dry-air flux and equal outlet patches, no wall heat leak, "
            "steady storage and no uncollected liquid escape are assumptions. "
            "Same psychrometric reconstruction and dilute enthalpy convention as the solver. "
            "The ±0.3 C shifts illustrate coherent calibration sensitivity, not a "
            "joint confidence interval or the additional unquantified mist-carryover bias. "
            "This calculation neither supplies the missing outlet velocity profile nor "
            "makes upstream simulated measurements equivalent to downstream sensors."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
