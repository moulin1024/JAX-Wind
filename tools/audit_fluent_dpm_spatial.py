"""Coarse V80 integration smoke check; no developed-wake or measured-validation claim."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind import InflowPlane, StaggeredVelocity, initial_atmospheric_solution
from jaxwind.config.stages import load_workflow
from jaxwind.fluent_dpm_atmosphere import dpm_diagnostics
from jaxwind.simulation.open_atmospheric import build_open_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.steps <= 20:
        parser.error("smoke test supports 1..20 steps at its declared parcel capacity")
    jax.config.update("jax_enable_x64", True)
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault(
        "JAXWIND_V80_FAST", str(root / "cases/HornsRev1/turbines/V80/CustomRotor.fst")
    )
    workflow = load_workflow(root / "cases/FluentDPMWater/v80_coarse_smoke.toml")
    grid = workflow.case.physical.physical_grid
    shape = (grid.nz, grid.ny, grid.nx)
    warm = initial_atmospheric_solution(
        grid,
        StaggeredVelocity(
            jnp.full(shape, 8.0),
            jnp.zeros(shape),
            jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
        ),
        dtype="float64",
    )
    plane = InflowPlane(
        warm.velocity.x[..., 0],
        warm.velocity.y[..., 0],
        warm.velocity.z[..., 0],
        warm.scalar[..., 0],
    )
    report = {
        "claim": "Short integration smoke test, not developed-wake validation",
        "cell_dimensions_m": [grid.dx, grid.dy, grid.dz],
        "steps": args.steps,
        "dt_s": 0.2,
        "les_model": workflow.water_spray.dpm.les_model,
        "assumed_amd_length_m": workflow.water_spray.dpm.amd_length_scale_m,
    }
    finals = {}
    for label, mdot in [
        ("dry", 0.0),
        ("spray", workflow.water_spray.mass_flow_rate_kg_s),
    ]:
        configured = replace(
            workflow,
            water_spray=replace(workflow.water_spray, mass_flow_rate_kg_s=mdot),
        )
        state, step = build_open_components(configured, warm, plane, return_step=True)
        step = jax.jit(step)
        initial_q = state.moisture.vapor
        for _ in range(args.steps):
            state = step(state, 0.2, plane)
            if not bool(state.accepted):
                raise RuntimeError(f"{label}: rejected at step {int(state.step)}")
        info = {key: float(value) for key, value in dpm_diagnostics(state).items()}
        info["minimum_temperature_K"] = float(jnp.min(state.scalar) + 310.0)
        info["minimum_u_m_s"] = float(jnp.min(state.velocity.x))
        info["maximum_u_m_s"] = float(jnp.max(state.velocity.x))
        gained = (
            workflow.moisture.thermodynamics.dry_air_density
            * grid.dx
            * grid.dy
            * grid.dz
            * jnp.sum(state.moisture.vapor - initial_q)
        )
        info["carrier_vapor_budget_error_kg"] = float(
            gained - state.transport_water - state.dpm_ledger.evaporated_mass
        )
        assert abs(info["dpm_water_budget_error_kg"]) < 1e-10
        assert abs(info["carrier_vapor_budget_error_kg"]) < 1e-8
        assert info["minimum_u_m_s"] < 8.0  # Active rotor/ground drag.
        report[label] = info
        finals[label] = state
        print(label, json.dumps(info), flush=True)
    delta = finals["spray"].scalar - finals["dry"].scalar
    report["minimum_spray_minus_dry_temperature_K"] = float(jnp.min(delta))
    report["maximum_spray_minus_dry_temperature_K"] = float(jnp.max(delta))
    assert report["spray"]["dpm_evaporated_mass_kg"] > 0
    assert report["minimum_spray_minus_dry_temperature_K"] < 0
    assert report["dry"]["dpm_injected_mass_kg"] == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        temperature_difference_K=np.asarray(delta),
        vapor=np.asarray(finals["spray"].moisture.vapor),
        parcel_position=np.asarray(finals["spray"].parcels.position),
    )


if __name__ == "__main__":
    main()
