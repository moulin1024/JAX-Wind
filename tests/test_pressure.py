from __future__ import annotations

import jax.numpy as jnp

from jaxwind.pressure import (
    GMGConfig,
    MACStageProjector,
    MACVelocity,
    MatrixFreePoissonSolver,
    PCGConfig,
    PoissonBoundaryConditions,
    RectilinearGrid,
)


def _solver(grid: RectilinearGrid) -> MatrixFreePoissonSolver:
    return MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.homogeneous_neumann(),
        dtype=jnp.float32,
        gmg=GMGConfig(coarse_smooth=30),
        krylov=PCGConfig(
            max_iterations=60,
            relative_tolerance=2.0e-6,
            execution="jax",
        ),
    )


def test_matrix_free_gmg_pcg_recovers_a_discrete_solution() -> None:
    grid = RectilinearGrid.uniform(8, 8, 8)
    solver = _solver(grid)
    exact = jnp.sin(
        0.13 * jnp.arange(grid.cell_count, dtype=jnp.float32)
    ).reshape(grid.shape)
    exact = solver.operator.project_nullspace(exact)

    result = solver.solve(solver.operator.apply(exact))
    error = solver.operator.project_nullspace(result.solution - exact)

    assert result.converged
    assert result.relative_residual < 2.0e-6
    assert float(solver.operator.norm(error) / solver.operator.norm(exact)) < 2.0e-5


def test_full_mac_projection_eliminates_divergence() -> None:
    grid = RectilinearGrid.uniform(8, 8, 8)
    solver = _solver(grid)
    nz, ny, nx = grid.shape
    velocity = MACVelocity(
        jnp.sin(jnp.arange(nz * ny * (nx + 1), dtype=jnp.float32)).reshape(
            nz, ny, nx + 1
        ),
        jnp.cos(jnp.arange(nz * (ny + 1) * nx, dtype=jnp.float32)).reshape(
            nz, ny + 1, nx
        ),
        jnp.sin(jnp.arange((nz + 1) * ny * nx, dtype=jnp.float32)).reshape(
            nz + 1, ny, nx
        ),
    )
    velocity = MACVelocity(
        velocity.x.at[..., 0].set(0.0).at[..., -1].set(0.0),
        velocity.y.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0),
        velocity.z.at[0].set(0.0).at[-1].set(0.0),
    )

    result = MACStageProjector(solver).project(velocity, timestep=0.01)

    assert result.linear_result.converged
    assert float(solver.operator.norm(result.divergence_after)) < 8.0e-4
