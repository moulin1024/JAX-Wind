from __future__ import annotations

import jax.numpy as jnp

from jaxwind.pressure import (
    FGMRESConfig,
    MACStageProjector,
    MACVelocity,
    MatrixFreePoissonSolver,
    PoissonBoundaryConditions,
    RectilinearGrid,
    mac_divergence,
    mac_pressure_gradient,
    projected_ssprk3_step,
)


def _wall_solver(grid: RectilinearGrid) -> MatrixFreePoissonSolver:
    return MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.homogeneous_neumann(),
        dtype=jnp.float32,
        krylov=FGMRESConfig(
            restart=20,
            max_iterations=80,
            relative_tolerance=2.0e-6,
        ),
    )


def _closed_velocity(grid: RectilinearGrid, phase: float = 0.0) -> MACVelocity:
    nz, ny, nx = grid.shape
    x = jnp.sin(
        phase
        + 0.13
        * jnp.arange(nz * ny * (nx + 1), dtype=jnp.float32).reshape(
            nz, ny, nx + 1
        )
    )
    y = jnp.cos(
        phase
        + 0.11
        * jnp.arange(nz * (ny + 1) * nx, dtype=jnp.float32).reshape(
            nz, ny + 1, nx
        )
    )
    z = jnp.sin(
        phase
        + 0.07
        * jnp.arange((nz + 1) * ny * nx, dtype=jnp.float32).reshape(
            nz + 1, ny, nx
        )
    )
    return MACVelocity(
        x.at[..., 0].set(0.0).at[..., -1].set(0.0),
        y.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0),
        z.at[0, ...].set(0.0).at[-1, ...].set(0.0),
    )


def test_mac_gradient_and_divergence_form_the_poisson_operator() -> None:
    grid = RectilinearGrid.uniform(8, 6, 10)
    solver = _wall_solver(grid)
    pressure = jnp.sin(
        0.17 * jnp.arange(grid.cell_count, dtype=jnp.float32)
    ).reshape(grid.shape)

    composed = -mac_divergence(
        mac_pressure_gradient(
            pressure,
            grid,
            solver.operator.boundaries,
        ),
        grid,
    )

    assert float(jnp.max(jnp.abs(composed - solver.operator.apply(pressure)))) < (
        2.0e-4
    )


def test_mac_stage_projection_eliminates_divergence_and_preserves_walls() -> None:
    grid = RectilinearGrid.uniform(8, 6, 10)
    solver = _wall_solver(grid)
    velocity = _closed_velocity(grid)

    result = MACStageProjector(solver).project(
        velocity,
        timestep=0.01,
    )

    assert result.linear_result.converged
    assert float(solver.operator.norm(result.divergence_after)) < 3.0e-4
    assert float(jnp.max(jnp.abs(result.velocity.x[..., 0]))) == 0.0
    assert float(jnp.max(jnp.abs(result.velocity.x[..., -1]))) == 0.0
    assert float(jnp.max(jnp.abs(result.velocity.y[:, 0, :]))) == 0.0
    assert float(jnp.max(jnp.abs(result.velocity.y[:, -1, :]))) == 0.0
    assert float(jnp.max(jnp.abs(result.velocity.z[0, ...]))) == 0.0
    assert float(jnp.max(jnp.abs(result.velocity.z[-1, ...]))) == 0.0


def test_device_projection_path_avoids_diagnostic_synchronization() -> None:
    grid = RectilinearGrid.uniform(6, 4, 8)
    solver = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.homogeneous_neumann(),
        dtype=jnp.float32,
        krylov=FGMRESConfig(
            restart=10,
            max_iterations=40,
            relative_tolerance=2.0e-6,
            execution="jax",
        ),
    )
    projected = MACStageProjector(solver).project_velocity(
        _closed_velocity(grid),
        timestep=0.01,
    )
    divergence = mac_divergence(projected, grid)

    assert float(solver.operator.norm(divergence)) < 8.0e-4
    assert float(jnp.max(jnp.abs(projected.z[0, ...]))) == 0.0
    assert float(jnp.max(jnp.abs(projected.z[-1, ...]))) == 0.0


def test_ssprk3_projects_every_stage() -> None:
    grid = RectilinearGrid.uniform(6, 4, 8)
    solver = _wall_solver(grid)
    projector = MACStageProjector(solver)
    initial = _closed_velocity(grid)
    forcing = _closed_velocity(grid, phase=0.4)

    result = projected_ssprk3_step(
        initial,
        tendency=lambda _velocity, _time: forcing,
        projector=projector,
        timestep=0.005,
    )

    assert all(stage.linear_result.converged for stage in result.stages)
    assert all(
        float(solver.operator.norm(stage.divergence_after)) < 4.0e-4
        for stage in result.stages
    )
