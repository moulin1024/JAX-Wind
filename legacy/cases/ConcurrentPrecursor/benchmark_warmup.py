#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "jax"))

from run_single import RUN_DEFAULTS, load_config_file, params_from_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time the production distributed warm-up step without checkpoint I/O."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--coordinator-address", default="127.0.0.1:12690")
    parser.add_argument("--num-processes", type=int)
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--local-device-id", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--timed-steps", type=int, default=400)
    parser.add_argument("--estimate-steps", type=int, default=500000)
    parser.add_argument("--dt", type=float, help="Physical time-step override in seconds.")
    parser.add_argument("--cs-count", type=int, help="LASD update interval override.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def rank_and_size(args: argparse.Namespace) -> tuple[int, int]:
    rank = args.process_id
    size = args.num_processes
    if rank is None:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    if size is None:
        size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    return rank, size


def main() -> None:
    args = parse_args()
    rank, size = rank_and_size(args)
    if args.warmup_steps < 0 or args.timed_steps <= 0 or args.estimate_steps <= 0:
        raise SystemExit("step counts must be positive (warmup may be zero)")

    settings = dict(RUN_DEFAULTS)
    settings.update(load_config_file(args.config))
    if settings["precision"] != "float32" or settings["sgs_precision"] != "float32":
        raise SystemExit("This benchmark requires solver and SGS precision=float32")

    import jax

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=size,
        process_id=rank,
        local_device_ids=[args.local_device_id],
    )
    import jax.numpy as jnp
    from jax.experimental import multihost_utils

    from wireles_jax.sharding import make_distributed_mesh
    from wireles_jax.timestep_sharded import (
        initial_sharded_state,
        make_diagnostics_sharded,
        make_project_velocity_sharded,
        make_sharded_operators,
        make_step_ab2_sharded,
    )

    configured = params_from_settings(settings, jnp)
    if args.dt is not None:
        configured = replace(configured, dt=args.dt / configured.z_i)
    if args.cs_count is not None:
        configured = replace(configured, cs_count=args.cs_count)
    params = replace(
        configured,
        nsteps=args.timed_steps,
        actuator_disk_enabled=False,
        cold_source_enabled=False,
        fringe_enabled=False,
        horizontal_homogeneous=True,
        buoyancy_reference="plane_mean",
        sharded_pressure_solver="transpose",
    )
    mesh = make_distributed_mesh(size)
    state = initial_sharded_state(params, mesh, seed=args.seed)
    ops = make_sharded_operators(params, mesh)
    project = jax.jit(
        make_project_velocity_sharded(
            params,
            ops.pressure,
            mesh,
            spike_ops=ops.pressure_spike,
        )
    )
    u, v, w, p = project(
        state.u, state.v, state.w, ops.pressure, ops.pressure_spike
    )
    state = state._replace(u=u, v=v, w=w, p=p)

    step = make_step_ab2_sharded(params, ops, mesh)
    compile_start = time.perf_counter()
    compiled_step = (
        jax.jit(step)
        .lower(state, ops.pressure, ops.pressure_spike)
        .compile()
    )
    compile_s = time.perf_counter() - compile_start
    diagnostic = jax.jit(
        make_diagnostics_sharded(params, ops.horizontal, mesh)
    )

    for _ in range(args.warmup_steps):
        state = compiled_step(state, ops.pressure, ops.pressure_spike)
    state = jax.block_until_ready(state)
    multihost_utils.sync_global_devices("warmup-benchmark-ready")
    start_diag = jax.block_until_ready(
        diagnostic(
            state.u,
            state.v,
            state.w,
            state.theta,
            state.qv,
            state.step,
        )
    )

    start = time.perf_counter()
    for _ in range(args.timed_steps):
        state = compiled_step(state, ops.pressure, ops.pressure_spike)
    state = jax.block_until_ready(state)
    multihost_utils.sync_global_devices("warmup-benchmark-finished")
    elapsed_s = time.perf_counter() - start
    final_diag = jax.block_until_ready(
        diagnostic(
            state.u,
            state.v,
            state.w,
            state.theta,
            state.qv,
            state.step,
        )
    )
    fields_finite = bool(
        jax.device_get(
            jnp.all(jnp.isfinite(state.u))
            & jnp.all(jnp.isfinite(state.v))
            & jnp.all(jnp.isfinite(state.w))
        )
    )

    seconds_per_step = elapsed_s / args.timed_steps
    estimated_seconds = seconds_per_step * args.estimate_steps
    if rank == 0:
        result = {
            "global_shape": [params.nx, params.ny, params.nz],
            "num_processes": size,
            "dtype": str(params.dtype),
            "sgs_dtype": str(params.sgs_dtype),
            "dt_seconds": params.dt_physical,
            "cs_count": params.cs_count,
            "compile_seconds": compile_s,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "elapsed_seconds": elapsed_s,
            "seconds_per_step": seconds_per_step,
            "cell_steps_per_second": (
                params.nx * params.ny * params.nz / seconds_per_step
            ),
            "start_cfl_max_direction": max(
                float(start_diag.cfl_x),
                float(start_diag.cfl_y),
                float(start_diag.cfl_z),
            ),
            "final_cfl_max_direction": max(
                float(final_diag.cfl_x),
                float(final_diag.cfl_y),
                float(final_diag.cfl_z),
            ),
            "final_cfl_components": [
                float(final_diag.cfl_x),
                float(final_diag.cfl_y),
                float(final_diag.cfl_z),
            ],
            "final_lasd_cfl": params.cs_count
            * max(
                float(final_diag.cfl_x),
                float(final_diag.cfl_y),
                float(final_diag.cfl_z),
            ),
            "final_div_max": float(final_diag.div_max),
            "final_ustar": float(final_diag.ustar),
            "fields_finite": fields_finite,
            "estimate_steps": args.estimate_steps,
            "estimated_seconds": estimated_seconds,
            "estimated_hours": estimated_seconds / 3600.0,
        }
        print(json.dumps(result, indent=2), flush=True)
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
