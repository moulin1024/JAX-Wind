from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    FREE_SLIP,
    OPEN,
    Boundaries,
    FlowModel,
    InflowPlane,
    StaggeredVelocity,
    Wall,
    build_open_atmospheric_run,
    build_open_atmospheric_step,
    build_pressure_poisson,
    divergence,
    initial_atmospheric_solution,
)


def test_open_fast_rk3_enforces_inflow_projects_and_advances_exact_clock() -> None:
    grid = UniformGrid(8, 6, 4, 2.0, 1.5, 1.0)
    velocity = StaggeredVelocity(
        jnp.full((grid.nz, grid.ny, grid.nx + 1), 2.0, jnp.float64),
        jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float64),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx), jnp.float64),
    )
    plane = InflowPlane(
        jnp.full((grid.nz, grid.ny), 2.0, jnp.float64),
        jnp.zeros((grid.nz, grid.ny), jnp.float64),
        jnp.zeros((grid.nz + 1, grid.ny), jnp.float64),
        jnp.zeros((grid.nz, grid.ny), jnp.float64),
    )
    steps = 3
    inflows = InflowPlane(
        *(jnp.broadcast_to(value, (steps,) + value.shape) for value in plane)
    )
    boundaries = Boundaries(
        Wall(FREE_SLIP),
        Wall(FREE_SLIP),
        streamwise=OPEN,
    )
    step = build_open_atmospheric_step(
        grid,
        boundaries,
        build_pressure_poisson(
            grid,
            backend="gmg",
            periodic_x=False,
            dtype="float64",
        ),
        FlowModel(),
        None,
        scheme="fast-rk3",
    )
    final = build_open_atmospheric_run(step)(
        initial_atmospheric_solution(
            grid,
            velocity,
            dtype="float64",
        ),
        0.01,
        inflows,
    )

    assert int(final.step) == steps
    assert float(final.time) == 0.03
    assert float(jnp.max(jnp.abs(divergence(final.velocity, grid)))) < 1.0e-10
    assert bool(jnp.all(final.velocity.x[..., 0] == 2.0))
