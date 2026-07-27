#!/usr/bin/env python3
"""Per-process worker for strong-scaling the distributed z-sharded solver."""

from __future__ import annotations

import argparse
import json
import time

import jax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--ny", type=int, default=80)
    parser.add_argument("--nz", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--pressure-method",
        choices=("transpose", "spike"),
        default="transpose",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=[0],
    )

    import jax.numpy as jnp
    from jax.experimental import multihost_utils

    from wireles_jax import Params
    from wireles_jax.sharding import make_distributed_mesh
    from wireles_jax.timestep_sharded import (
        initial_sharded_state,
        make_sharded_operators,
        make_step_ab2_sharded,
    )

    # Grid, dimensional scales and dt follow
    # benchmark/Nieuwstadt1993/configs/lasd_scalar_80x80x96.toml.
    params = Params(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=6400.0 / 1600.0,
        ly=6400.0 / 1600.0,
        lz=2400.0 / 1600.0,
        z_i=1600.0,
        dt=0.625 / 1600.0,
        nsteps=args.steps,
        u_fric=0.0,
        zo=0.16,
        bl_height=1600.0,
        pressure_force=0.0,
        initial_condition="default",
        momentum_wall_model="abl",
        initial_velocity_noise=0.0,
        fgr=1.5,
        tfr=2.0,
        sgs_model="lasd",
        cs_count=10,
        smagorinsky_cs=0.18,
        sgs_delta_scale=1.0,
        time_scheme="ab2",
        horizontal_dealias=True,
        pressure_filter_nyquist=False,
        sharded_pressure_solver=args.pressure_method,
        top_boundary_condition="rigid_lid",
        thermo_enabled=True,
        moisture_enabled=False,
        theta0=300.0,
        g=9.81,
        theta_bc="flux",
        theta_profile="linear",
        theta_initial_gradient=0.003,
        theta_top_gradient=0.003,
        surface_theta_flux=0.06,
        scalar_sgs_model="fixed_prandtl",
        prandtl_t=0.33,
        schmidt_t=0.33,
        scalar_stability_correction=False,
        scalar_vertical_scheme="centered",
        dtype=jnp.float32,
        sgs_dtype=jnp.float32,
        use_jit=True,
    )
    mesh = make_distributed_mesh(args.num_processes)
    state = initial_sharded_state(params, mesh, seed=0)
    operators = make_sharded_operators(params, mesh)
    step = make_step_ab2_sharded(params, operators, mesh)

    compile_start = time.perf_counter()
    compiled_step = (
        jax.jit(step)
        .lower(state, operators.pressure, operators.pressure_spike)
        .compile()
    )
    compile_s = time.perf_counter() - compile_start

    for _ in range(args.warmup):
        state = compiled_step(
            state, operators.pressure, operators.pressure_spike
        )
    state = jax.block_until_ready(state)
    multihost_utils.sync_global_devices("wireles-scaling-warmup")

    start = time.perf_counter()
    for _ in range(args.steps):
        state = compiled_step(
            state, operators.pressure, operators.pressure_spike
        )
    state = jax.block_until_ready(state)
    multihost_utils.sync_global_devices("wireles-scaling-finish")
    elapsed_s = time.perf_counter() - start

    local_u = state.u.addressable_shards
    local_pressure = operators.pressure.pressure_a.addressable_shards
    if len(local_u) != 1 or len(local_pressure) != 1:
        raise RuntimeError("Scaling worker expects exactly one addressable device per process.")
    global_cells = args.nx * args.ny * args.nz
    result = {
        "process_id": jax.process_index(),
        "num_processes": jax.process_count(),
        "global_shape": [args.nx, args.ny, args.nz],
        "local_u_shape": list(local_u[0].data.shape),
        "local_pressure_shape": list(local_pressure[0].data.shape),
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "compile_s": compile_s,
        "pressure_method": args.pressure_method,
        "elapsed_s": elapsed_s,
        "seconds_per_step": elapsed_s / args.steps,
        "million_cell_steps_per_s": global_cells * args.steps / elapsed_s / 1.0e6,
        "final_step": int(state.step.addressable_data(0)),
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
