"""Checkpointed single V80 with prescribed uniform inflow and a hub-height spray.

A transient startup from uniform flow, not a statistically developed wake.
All simulation and field diagnostics run on the requested GPU allocation.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind import (
    InflowPlane,
    StaggeredVelocity,
    courant_number,
    initial_atmospheric_solution,
)
from jaxwind.config.stages import load_workflow
from jaxwind.fluent_dpm_atmosphere import dpm_diagnostics
from jaxwind.io.checkpoint import load_checkpoint, save_checkpoint
from jaxwind.physics.fluent_dpm import DPMWaterMaterial
from jaxwind.physics.moisture import saturation_vapor_pressure_water
from jaxwind.simulation.open_atmospheric import build_open_components


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, allow_nan=False, default=str) + "\n"
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=Path,
        default=Path("cases/FluentDPMWater/v80_uniform10_20kgs_200um.toml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--wall-seconds", type=float, default=720.0)
    args = parser.parse_args()
    if args.steps < 1 or args.wall_seconds < 60:
        parser.error("positive steps and at least 60 seconds wall allowance required")
    started = time.monotonic()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("run this simulation on gpudev")
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault(
        "JAXWIND_V80_FAST", str(root / "cases/HornsRev1/turbines/V80/CustomRotor.fst")
    )
    workflow = load_workflow(args.case)
    material = DPMWaterMaterial()
    temperature = workflow.moisture.temperature_offset_k
    pressure = workflow.moisture.thermodynamics.pressure
    rh = workflow.moisture.ambient_relative_humidity
    partial = rh * float(saturation_vapor_pressure_water(jnp.asarray(temperature)))
    density = (pressure - partial) / (material.dry_air_gas_constant * temperature)
    workflow = replace(
        workflow,
        moisture=replace(
            workflow.moisture,
            thermodynamics=replace(
                workflow.moisture.thermodynamics, dry_air_density=density
            ),
        ),
    )
    grid = workflow.case.physical.physical_grid
    dt = workflow.options.main_dt_seconds
    shape = (grid.nz, grid.ny, grid.nx)
    speed = 10.0
    warm = initial_atmospheric_solution(
        grid,
        StaggeredVelocity(
            jnp.full(shape, speed),
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
    state, step = build_open_components(workflow, warm, plane, return_step=True)
    initial_q = float(state.moisture.vapor[0, 0, 0])
    step = jax.jit(step)
    args.output.mkdir(parents=True, exist_ok=True)
    description = {
        "case": workflow.resolved(),
        "uniform_inflow_m_s": speed,
        "initialization": "uniform carrier, simultaneous turbine and spray startup",
        "target_steps": args.steps,
        "dt_s": dt,
        "duration_s": args.steps * dt,
        "dry_density_kg_m3": density,
        "ambient_vapor_ratio": initial_q,
        "source_position_m": [
            workflow.turbine.x_m + workflow.water_spray.streamwise_offset_m,
            workflow.turbine.y_m,
            workflow.turbine.hub_height_m,
        ],
        "domain_m": [grid.lx, grid.ly, grid.lz],
        "cells": [grid.nx, grid.ny, grid.nz],
        "cell_dimensions_m": [grid.dx, grid.dy, grid.dz],
    }
    # Fingerprint includes all resolved settings and source code for exact resume.
    code_paths = sorted((root / "src/jaxwind").rglob("*.py"))
    code_digest = hashlib.sha256()
    for path in code_paths:
        code_digest.update(str(path.relative_to(root)).encode())
        code_digest.update(path.read_bytes())
    description["source_digest"] = code_digest.hexdigest()
    fingerprint = hashlib.sha256(
        json.dumps(description, sort_keys=True, default=str).encode()
    ).hexdigest()
    write_json(args.output / "configuration.json", description)
    checkpoint = args.output / "checkpoint.npz"
    if checkpoint.exists():
        state, _, _ = load_checkpoint(checkpoint, state, fingerprint=fingerprint)
        print("RESUME", int(state.step), float(state.time), flush=True)
    else:
        print(
            "START",
            json.dumps(
                {
                    k: description[k]
                    for k in (
                        "domain_m",
                        "cells",
                        "cell_dimensions_m",
                        "source_position_m",
                        "dry_density_kg_m3",
                    )
                }
            ),
            flush=True,
        )
    volume = grid.dx * grid.dy * grid.dz
    hub_index = int(
        np.argmin(np.abs(np.asarray(grid.z_centers) - workflow.turbine.hub_height_m))
    )
    yy, zz = jnp.meshgrid(jnp.asarray(grid.y_centers), jnp.asarray(grid.z_centers))
    disk = (
        (yy - workflow.turbine.y_m) ** 2 + (zz - workflow.turbine.hub_height_m) ** 2
    ) <= 40.0**2

    @jax.jit
    def diagnose(state):
        data = dpm_diagnostics(state)
        temp = temperature + state.scalar
        q = state.moisture.vapor
        vapor_pressure = (
            pressure
            * q
            / (material.dry_air_gas_constant / material.vapor_gas_constant + q)
        )
        relative = vapor_pressure / saturation_vapor_pressure_water(temp)
        data.update(
            time_s=state.time,
            step=state.step,
            minimum_temperature_K=jnp.min(temp),
            maximum_temperature_K=jnp.max(temp),
            maximum_relative_humidity=jnp.max(relative),
            minimum_relative_humidity=jnp.min(relative),
            maximum_cfl=courant_number(state.velocity, grid, dt),
            minimum_u_m_s=jnp.min(state.velocity.x),
            maximum_u_m_s=jnp.max(state.velocity.x),
            carrier_vapor_budget_error_kg=density * volume * jnp.sum(q - initial_q)
            - state.transport_water
            - state.dpm_ledger.evaporated_mass,
        )
        for diameter in (1, 2, 4, 6, 8):
            x = workflow.turbine.x_m + diameter * 80.0
            index = int(np.argmin(np.abs(np.asarray(grid.x_centers) - x)))
            data[f"x{diameter}D_actual_x_m"] = grid.x_centers[index]
            data[f"x{diameter}D_rotor_area_temperature_K"] = jnp.sum(
                temp[..., index] * disk
            ) / jnp.sum(disk)
            data[f"x{diameter}D_rotor_area_u_m_s"] = jnp.sum(
                0.5
                * (state.velocity.x[..., index] + state.velocity.x[..., index + 1])
                * disk
            ) / jnp.sum(disk)
        return data

    def record(state, status):
        data = {k: float(v) for k, v in diagnose(state).items()}
        data["status"] = status
        data["invocation_wall_s"] = time.monotonic() - started
        write_json(args.output / "status.json", data)
        with (args.output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(data, allow_nan=False) + "\n")
        save_checkpoint(
            checkpoint, state, metadata={"fingerprint": fingerprint, "status": status}
        )
        print("PROGRESS", json.dumps(data, allow_nan=False), flush=True)
        if not np.isfinite(data["maximum_cfl"]) or data["maximum_cfl"] > 0.95:
            raise RuntimeError("CFL limit exceeded; reduce timestep in a separate case")
        if abs(data["dpm_water_budget_error_kg"]) > 1e-8 * max(
            1.0, data["dpm_injected_mass_kg"]
        ):
            raise RuntimeError("parcel water budget failed")
        if abs(data["carrier_vapor_budget_error_kg"]) > 1e-6 * max(
            1.0, data["dpm_evaporated_mass_kg"]
        ):
            raise RuntimeError("carrier vapor budget failed")
        return data

    while int(state.step) < args.steps:
        before = state
        state = step(state, dt, plane)
        if not bool(state.accepted):
            save_checkpoint(
                checkpoint,
                before,
                metadata={"fingerprint": fingerprint, "status": "rejected"},
            )
            write_json(
                args.output / "failure.json",
                {
                    "step": int(before.step),
                    "time_s": float(before.time),
                    "reason": "DPM transaction rejected; saved last accepted state",
                },
            )
            raise RuntimeError(f"DPM rejected step {int(before.step) + 1}")
        done = int(state.step) >= args.steps
        pause = time.monotonic() - started >= args.wall_seconds
        if int(state.step) == 1 or int(state.step) % 25 == 0 or done or pause:
            record(
                state, "complete" if done else ("checkpointed" if pause else "running")
            )
        if pause and not done:
            print("CHECKPOINTED: resubmit the same command to continue", flush=True)
            return
    # Save physical fields and parcel inventories for plotting and independent audit.
    np.savez_compressed(
        args.output / "fields.npz",
        x_m=np.asarray(grid.x_centers),
        y_m=np.asarray(grid.y_centers),
        z_m=np.asarray(grid.z_centers),
        temperature_K=np.asarray(temperature + state.scalar),
        vapor_ratio=np.asarray(state.moisture.vapor),
        u_m_s=np.asarray(
            0.5 * (state.velocity.x[..., :-1] + state.velocity.x[..., 1:])
        ),
        parcel_position_m=np.asarray(state.parcels.position),
        parcel_mass_kg=np.asarray(state.parcels.mass),
        parcel_multiplicity=np.asarray(state.parcels.multiplicity),
        parcel_temperature_K=np.asarray(state.parcels.temperature),
        hub_index=hub_index,
    )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
