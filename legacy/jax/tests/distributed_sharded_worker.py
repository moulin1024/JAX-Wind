#!/usr/bin/env python3
"""Two-process smoke worker for the truly distributed z-sharded solver."""

from __future__ import annotations

import argparse
import json

import jax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument(
        "--pressure-method",
        choices=("transpose", "spike"),
        default="transpose",
    )
    args = parser.parse_args()

    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=2,
        process_id=args.process_id,
        local_device_ids=[0],
    )

    import jax.numpy as jnp

    from wireles_jax import Params
    from wireles_jax.sharding import make_distributed_mesh
    from wireles_jax.timestep_sharded import (
        initial_sharded_state,
        make_diagnostics_sharded,
        make_sharded_operators,
        make_step_ab2_sharded,
    )

    params = Params(
        nx=8,
        ny=8,
        nz=8,
        lx=2.0,
        ly=2.0,
        lz=1.0,
        dt=1.0e-4,
        nsteps=1,
        momentum_wall_model="free_slip",
        initial_velocity_noise=0.01,
        thermo_enabled=True,
        theta0=300.0,
        theta_initial_gradient=0.01,
        surface_theta_flux=0.01,
        scalar_sgs_model="fixed_prandtl",
        scalar_vertical_scheme="centered",
        sgs_model="lasd",
        cs_count=1,
        sharded_pressure_solver=args.pressure_method,
        dtype=jnp.float32,
    )
    mesh = make_distributed_mesh(2)
    state = initial_sharded_state(params, mesh, seed=17)
    operators = make_sharded_operators(params, mesh)

    u_shards = state.u.addressable_shards
    pressure_shards = operators.pressure.pressure_a.addressable_shards
    assert len(u_shards) == 1
    assert len(pressure_shards) == 1
    assert u_shards[0].data.shape == (params.nx, params.ny, params.nz // 2)
    expected_pressure_shape = (
        (params.nx // 2 + 1, params.ny // 2, params.nz)
        if args.pressure_method == "transpose"
        else (1, 1, 1)
    )
    assert pressure_shards[0].data.shape == expected_pressure_shape
    assert u_shards[0].data.size < params.nx * params.ny * params.nz
    assert pressure_shards[0].data.size < (
        (params.nx // 2 + 1) * params.ny * params.nz
    )
    if args.pressure_method == "spike":
        assert operators.pressure_spike is not None
        assert operators.pressure_spike.local_a.addressable_shards[0].data.shape == (
            params.nx // 2 + 1,
            params.ny,
            params.nz // 2,
        )

    step = jax.jit(make_step_ab2_sharded(params, operators, mesh))
    state = jax.block_until_ready(
        step(state, operators.pressure, operators.pressure_spike)
    )
    diagnostics = make_diagnostics_sharded(params, operators.horizontal, mesh)
    diag = jax.block_until_ready(
        diagnostics(
            state.u,
            state.v,
            state.w,
            state.theta,
            state.qv,
            state.step,
        )
    )
    assert bool(jnp.isfinite(diag.div_max))
    assert bool(jnp.isfinite(diag.theta_v_min))
    print(
        json.dumps(
            {
                "process_id": jax.process_index(),
                "local_u_shape": u_shards[0].data.shape,
                "local_pressure_shape": pressure_shards[0].data.shape,
                "step": int(diag.step),
                "div_max": float(diag.div_max),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
