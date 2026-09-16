#!/usr/bin/env python3
"""Conservative wet-plate sensitivity diagnostic, not a validated rig correction.

Legacy geometry/correlations: Kachhwaha et al. (1998), pp453–454,
DOI 10.1016/S0017-9310(97)00133-6. Transfer to the 2008 rig is uncertain.
Gas and collected liquid are treated as co-current streams. Liquid is mixed
across the outlet and allocated in proportion to dry-air flux. Neither this
routing, complete gas mixing, nor complete wetting is established experimentally for the 2008 rig.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.config.moisture import load_moisture
from jaxwind.io.checkpoint import checkpoint_metadata
from jaxwind.physics.moisture import WaterDropletProperties, saturation_mixing_ratio


def wet_plate(
    temperature,
    vapor,
    liquid_ratio,
    liquid_temperature,
    speed,
    config,
    *,
    length=0.195,
    spacing=0.036,
    height=0.64,
    steps=1024,
    mass_transfer=True,
):
    """March per-kg-dry-air budgets; return T, qv, liquid ratio, liquid T.

    Constant carrier speed/properties and wet perimeter approximation match the
    legacy channel correlation. Explicit integration must be convergence checked.
    Enthalpy is cp_air*T + Lv*qv + ql*cp_liquid*(Tl-Tfreeze).
    """
    props = WaterDropletProperties()
    rho, cp = config.dry_air_density, config.dry_air_heat_capacity
    cpl, lv = props.liquid_heat_capacity, config.water_vapor_latent_heat
    dh = 2 * spacing * height / (spacing + height)
    re = rho * speed * dh / props.air_dynamic_viscosity
    pr = cp * props.air_dynamic_viscosity / props.air_thermal_conductivity
    sc = props.air_dynamic_viscosity / (rho * config.vapor_diffusivity)
    friction = (0.79 * jnp.log(re) - 1.64) ** -2
    nu = (
        (friction / 8)
        * (re - 1000)
        * pr
        / (1 + 12.7 * jnp.sqrt(friction / 8) * (pr ** (2 / 3) - 1))
        * (1 + (dh / length) ** (2 / 3))
    )
    h = nu * props.air_thermal_conductivity / dh
    km = 0.023 * re**0.83 * sc**0.44 * config.vapor_diffusivity / dh
    # All hydraulic perimeter is wet, as in the legacy equivalent channel.
    area_per_dry_flow = 4 / dh / (rho * speed) * length / steps
    shape = jnp.broadcast_shapes(jnp.shape(temperature), jnp.shape(speed))
    initial = tuple(
        jnp.broadcast_to(jnp.asarray(a), shape)
        for a in (temperature, vapor, liquid_ratio, liquid_temperature)
    )

    def advance(state, _):
        ta, qv, ql, tl = state
        qs = saturation_mixing_ratio(tl, config.pressure, config)
        # Linear Gilliland mass-transfer driving force in vapor mass fraction.
        # Warm evaporation only; no supersaturated condensation closure here.
        driving = jnp.maximum(qs / (1 + qs) - qv / (1 + qv), 0)
        dm = jnp.minimum(ql * 0.5, rho * km * driving * area_per_dry_flow)
        dm = dm if mass_transfer else jnp.zeros_like(dm)
        heat = h * (ta - tl) * area_per_dry_flow
        qnew = ql - dm
        liquid_h = ql * cpl * (tl - config.freezing_temperature) + heat - lv * dm
        return (
            ta - heat / cp,
            qv + dm,
            qnew,
            config.freezing_temperature + liquid_h / (qnew * cpl),
        ), None

    return jax.lax.scan(advance, initial, None, length=steps)[0]


def diagnose(directory, steps):
    header = checkpoint_metadata(directory / "checkpoint.npz")
    doc = header["resolved_case"]
    moist, _ = load_moisture(doc["physics"])
    c = moist.thermodynamics
    nx, ny, nz = doc["mesh"]["cells"]
    lx, ly, lz = doc["mesh"]["lengths_m"]
    with np.load(directory / "checkpoint.npz", allow_pickle=False) as d:
        ta = d["state/scalar"][..., -1] + moist.temperature_offset_k
        qv = d["state/moisture/vapor"][..., -1]
        u = d["state/velocity/x"][..., -1]
        weights = (
            d["state/parcels/mass"]
            * d["state/parcels/multiplicity"]
            * d["state/parcels/active"]
            * np.maximum(d["state/parcels/velocity"][0], 0)
            * (d["state/parcels/position"][0] >= lx - lx / nx)
        )
        water_flow = float(weights.sum() / (lx / nx))
        tw = float(np.sum(weights * d["state/parcels/temperature"]) / weights.sum())
    dry_flow = c.dry_air_density * u * ly * lz / (ny * nz)
    # A bulk channel correlation applies to section-mean speed, not individual
    # turbulent cells. Mix gas conservatively before this lumped diagnostic.
    ta = float(np.sum(dry_flow * ta) / np.sum(dry_flow))
    qv = float(np.sum(dry_flow * qv) / np.sum(dry_flow))
    u = float(np.mean(u))
    dry_flow = float(dry_flow.sum())
    ql = water_flow / dry_flow
    dh = 2 * 0.036 * 0.64 / (0.036 + 0.64)
    re = c.dry_air_density * u * dh / WaterDropletProperties().air_dynamic_viscosity
    if np.any(re <= 2300) or np.any(re >= 5e6):
        raise ValueError("Wet-plate Reynolds number outside correlation range")
    out = wet_plate(ta, qv, ql, tw, u, c, steps=steps)
    tout, vout, lout, wlout = map(np.asarray, out)
    cp, lv = c.dry_air_heat_capacity, c.water_vapor_latent_heat
    cpl = WaterDropletProperties().liquid_heat_capacity
    hin = cp * ta + lv * qv + ql * cpl * (tw - c.freezing_temperature)
    hout = cp * tout + lv * vout + lout * cpl * (wlout - c.freezing_temperature)
    return {
        "directory": str(directory),
        "integration_steps": steps,
        "input_gas_flux_mean_c": float(
            np.sum(dry_flow * ta) / np.sum(dry_flow) - 273.15
        ),
        "output_gas_flux_mean_c": float(
            np.sum(dry_flow * tout) / np.sum(dry_flow) - 273.15
        ),
        "input_liquid_slab_temperature_c": tw - 273.15,
        "input_liquid_slab_flow_kg_s": water_flow,
        "output_liquid_temperature_c": float(
            np.sum(dry_flow * lout * wlout) / np.sum(dry_flow * lout) - 273.15
        ),
        "extra_evaporation_kg_s": float(np.sum(dry_flow * (vout - qv))),
        "extra_gas_sensible_loss_w": float(np.sum(dry_flow * cp * (ta - tout))),
        "max_water_budget_error_kg_per_kg": float(np.max(abs(vout + lout - qv - ql))),
        "max_enthalpy_budget_error_j_per_kg": float(np.max(abs(hout - hin))),
        "reynolds_range": [float(np.min(re)), float(np.max(re))],
        "interpretation": "Instantaneous outlet sensitivity using uncertain legacy rig geometry and fully mixed gas with co-current uniformly redistributed liquid. Flux mean is not the experimental nine-sensor mean. No measured temperature used as a model input.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    results = [diagnose(d, n) for d in args.directories for n in (256, 1024, 4096)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
