#!/usr/bin/env python3
"""Benchmark transpose and compact-SPIKE distributed pressure solves."""

from __future__ import annotations

import argparse
import json
import time

import jax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--method", choices=("transpose", "spike"), required=True)
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--ny", type=int, default=80)
    parser.add_argument("--nz", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=[0],
    )

    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils

    from wireles_jax import Params
    from wireles_jax.pressure_sharded import (
        make_pressure_hat_solver_z_sharded,
        make_pressure_hat_solver_z_sharded_spike,
        make_sharded_pressure_operators,
        make_sharded_spike_operators,
    )
    from wireles_jax.sharding import (
        make_array_from_local_callback,
        make_distributed_mesh,
        rfft2_fortran_layout,
        z_slab_sharding,
    )

    params = Params(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=4.0,
        ly=4.0,
        lz=1.5,
        z_i=1600.0,
        dt=0.625 / 1600.0,
        momentum_wall_model="free_slip",
        dtype=jnp.float32,
    )
    mesh = make_distributed_mesh(args.num_processes)
    shape = (args.nx, args.ny, args.nz)
    sharding = z_slab_sharding(mesh)

    def rhs_callback(index: tuple[slice, ...]) -> np.ndarray:
        local_shape = tuple(part.stop - part.start for part in index)
        seed = 1201 + index[2].start
        return np.random.default_rng(seed).standard_normal(local_shape).astype(
            np.float32
        )

    rhs = make_array_from_local_callback(
        shape, sharding, rhs_callback, dtype=params.dtype
    )
    rhs_hat = rfft2_fortran_layout(rhs)
    if args.method == "transpose":
        operators = make_sharded_pressure_operators(params, mesh)
        solve = make_pressure_hat_solver_z_sharded(params, operators, mesh)
    else:
        operators = make_sharded_spike_operators(params, mesh)
        solve = make_pressure_hat_solver_z_sharded_spike(params, mesh)

    compile_start = time.perf_counter()
    compiled = jax.jit(solve).lower(rhs_hat, operators).compile()
    compile_s = time.perf_counter() - compile_start
    result = rhs_hat
    for _ in range(args.warmup):
        result = compiled(rhs_hat, operators)
    result = jax.block_until_ready(result)
    multihost_utils.sync_global_devices(f"pressure-{args.method}-warmup")
    start = time.perf_counter()
    for _ in range(args.steps):
        result = compiled(rhs_hat, operators)
    result = jax.block_until_ready(result)
    multihost_utils.sync_global_devices(f"pressure-{args.method}-finish")
    elapsed_s = time.perf_counter() - start
    print(
        json.dumps(
            {
                "method": args.method,
                "process_id": jax.process_index(),
                "num_processes": jax.process_count(),
                "global_shape": list(shape),
                "compile_s": compile_s,
                "timed_solves": args.steps,
                "elapsed_s": elapsed_s,
                "seconds_per_solve": elapsed_s / args.steps,
                "result_max_local": float(
                    jnp.max(jnp.abs(result.addressable_data(0)))
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
