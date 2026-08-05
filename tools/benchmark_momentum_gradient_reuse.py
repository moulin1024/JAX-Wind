#!/usr/bin/env python3
"""Compare repeated and shared velocity-gradient momentum tendency graphs."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp

from jaxwind.momentum import LASDModel, MomentumConfig, MomentumOperators
from jaxwind.momentum.operators import _cell_velocity
from jaxwind.pressure import (
    BoundaryCondition,
    FGMRESConfig,
    MACVelocity,
    MatrixFreePoissonSolver,
    PoissonBoundaryConditions,
    RectilinearGrid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def build_solver(n: int) -> MomentumOperators:
    grid = RectilinearGrid.uniform(
        n,
        n,
        n,
        lx=4000.0,
        ly=2000.0,
        lz=1500.0,
    )
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions(
            periodic,
            periodic,
            periodic,
            periodic,
            neumann,
            neumann,
        ),
        dtype=jnp.float32,
        krylov=FGMRESConfig(
            restart=10,
            max_iterations=20,
            relative_tolerance=1.0e-4,
            execution="jax",
        ),
    )
    return MomentumOperators(
        grid,
        pressure,
        MomentumConfig(
            friction_velocity=0.4,
            roughness_length=0.1,
            geostrophic_wind=(10.0, 0.0),
            coriolis_vertical=1.0e-4,
            coriolis_horizontal=1.0e-4,
            lasd=LASDModel(update_interval=1),
        ),
    )


def initial_velocity(solver: MomentumOperators) -> MACVelocity:
    nz, ny, nx = solver.grid.shape
    key_x, key_y, key_z = jax.random.split(jax.random.PRNGKey(1994), 3)
    velocity = MACVelocity(
        8.0 + 0.2 * jax.random.normal(key_x, (nz, ny, nx + 1), dtype=jnp.float32),
        0.2 * jax.random.normal(key_y, (nz, ny + 1, nx), dtype=jnp.float32),
        0.1 * jax.random.normal(key_z, (nz + 1, ny, nx), dtype=jnp.float32),
    )
    return solver.enforce_boundaries(velocity)


def elapsed_per_call(
    executable,
    velocity: MACVelocity,
    coefficient: jax.Array,
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        jax.block_until_ready(executable(velocity, coefficient))
    start = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(executable(velocity, coefficient))
    return (time.perf_counter() - start) / repeats


def main() -> None:
    args = parse_args()
    if min(args.n, args.warmup, args.repeats, args.rounds) <= 0:
        raise SystemExit("all benchmark controls must be positive")
    solver = build_solver(args.n)
    velocity = initial_velocity(solver)
    solver.reset_lasd(velocity)
    coefficient = solver.lasd_state.coefficient

    def repeated_gradient_tendency(
        stage_velocity: MACVelocity,
        lasd_coefficient: jax.Array,
    ) -> jax.Array:
        cells = _cell_velocity(stage_velocity)
        viscosity = solver.sgs_viscosity(cells, lasd_coefficient)
        tendency = (
            solver.conservative_advection(stage_velocity, cells)
            + solver.variational_sgs_tendency(cells, viscosity)
            + solver.forcing_tendency(cells)
        )
        if solver.config.mp5_dissipation_strength > 0.0:
            tendency += solver.mp5_dissipation(stage_velocity, cells)
        return tendency

    shared = (
        jax.jit(solver.cell_tendency)
        .lower(
            velocity,
            coefficient,
        )
        .compile()
    )
    repeated = (
        jax.jit(repeated_gradient_tendency)
        .lower(
            velocity,
            coefficient,
        )
        .compile()
    )
    shared_times = []
    repeated_times = []
    for round_index in range(args.rounds):
        order = (
            (shared, shared_times, repeated, repeated_times)
            if round_index % 2 == 0
            else (repeated, repeated_times, shared, shared_times)
        )
        first, first_times, second, second_times = order
        first_times.append(
            elapsed_per_call(
                first,
                velocity,
                coefficient,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
        second_times.append(
            elapsed_per_call(
                second,
                velocity,
                coefficient,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
    shared_median = statistics.median(shared_times)
    repeated_median = statistics.median(repeated_times)
    print(
        json.dumps(
            {
                "grid": [args.n, args.n, args.n],
                "shared_gradient_seconds": shared_median,
                "repeated_gradient_seconds": repeated_median,
                "speedup": repeated_median / shared_median,
                "shared_rounds": shared_times,
                "repeated_rounds": repeated_times,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
