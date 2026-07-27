#!/usr/bin/env python3
"""One-process/four-device structural smoke test for the adjoint mesh."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from wireles_jax import Params
from wireles_jax.adjoint_sharded import (
    duplicate_state_for_adjoint,
    make_adjoint_chunk_step,
    make_adjoint_pipeline_batch,
    make_adjoint_pipeline_prime,
    make_empty_fringe_chunk,
    make_exchange_precursor_chunk,
)
from wireles_jax.sharding import make_adjoint_distributed_mesh, make_distributed_mesh
from wireles_jax.derivative import divergence
from wireles_jax.timestep_sharded import initial_sharded_state, make_sharded_operators


def main() -> None:
    params = Params(
        nx=8,
        ny=4,
        nz=8,
        lx=4.0,
        ly=2.0,
        lz=2.0,
        z_i=1.0,
        dt=1.0e-4,
        nsteps=1,
        momentum_wall_model="free_slip",
        initial_velocity_noise=0.001,
        thermo_enabled=True,
        theta0=300.0,
        surface_theta_flux=0.0,
        scalar_sgs_model="fixed_prandtl",
        scalar_vertical_scheme="centered",
        sgs_model="smagorinsky",
        sharded_pressure_solver="transpose",
        fringe_enabled=True,
        fringe_start_x=3.0,
        fringe_timescale=0.1,
        dtype=jnp.float32,
        sgs_dtype=jnp.float32,
    )

    warm_mesh = make_distributed_mesh(4)
    warm = initial_sharded_state(params, warm_mesh, seed=3)
    mesh = make_adjoint_distributed_mesh(4)
    legacy_state = duplicate_state_for_adjoint(warm, mesh)
    batched_state = duplicate_state_for_adjoint(warm, mesh)
    operators = make_sharded_operators(params, mesh)
    empty = make_empty_fringe_chunk(params, mesh, 1)

    legacy_prime = jax.jit(
        make_adjoint_chunk_step(
            params,
            operators,
            mesh,
            chunk_steps=1,
            advance_turbine=False,
        )
    )
    legacy_state, produced = legacy_prime(
        legacy_state, empty, operators.pressure, operators.pressure_spike
    )
    exchange = jax.jit(make_exchange_precursor_chunk(mesh))
    legacy_targets = exchange(produced)

    legacy_advance = jax.jit(
        make_adjoint_chunk_step(
            params,
            operators,
            mesh,
            chunk_steps=1,
            advance_turbine=True,
        )
    )
    for _ in range(2):
        legacy_state, produced = legacy_advance(
            legacy_state,
            legacy_targets,
            operators.pressure,
            operators.pressure_spike,
        )
        legacy_targets = exchange(produced)
    jax.block_until_ready(legacy_state)

    batched_prime = jax.jit(
        make_adjoint_pipeline_prime(
            params,
            operators,
            mesh,
            chunk_steps=1,
        )
    )
    batched_state, batched_targets = batched_prime(
        batched_state, empty, operators.pressure, operators.pressure_spike
    )
    batched_advance = jax.jit(
        make_adjoint_pipeline_batch(
            params,
            operators,
            mesh,
            chunk_steps=1,
            chunks_per_launch=2,
        )
    )
    batched_state, batched_targets = jax.block_until_ready(
        batched_advance(
            batched_state,
            batched_targets,
            operators.pressure,
            operators.pressure_spike,
        )
    )

    state_difference = max(
        float(jnp.max(jnp.abs(new - old)))
        for new, old in zip(batched_state, legacy_state, strict=True)
    )
    target_difference = float(
        jnp.max(jnp.abs(batched_targets - legacy_targets))
    )
    assert state_difference < 1.0e-6
    assert target_difference < 1.0e-6

    divergence_max = jnp.max(
        jnp.abs(
            jax.vmap(
                lambda u, v, w: divergence(
                    u, v, w, params, operators.horizontal
                )
            )(batched_state.u, batched_state.v, batched_state.w)
        )
    )

    shards = batched_state.u.addressable_shards
    assert batched_state.u.shape == (2, params.nx, params.ny, params.nz)
    assert all(s.data.shape == (1, params.nx, params.ny, params.nz // 2) for s in shards)
    assert all(s.data.size < params.nx * params.ny * params.nz for s in shards)
    assert bool(jnp.all(jnp.isfinite(batched_state.u)))
    assert float(divergence_max) < 2.0e-4
    print(
        json.dumps(
            {
                "mesh": dict(mesh.shape),
                "global_shape": batched_state.u.shape,
                "local_shapes": [s.data.shape for s in shards],
                "steps": list(map(int, batched_state.step)),
                "chunk_shape": produced.shape,
                "divergence_max": float(divergence_max),
                "state_difference": state_difference,
                "target_difference": target_difference,
            }
        )
    )


if __name__ == "__main__":
    main()
