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
from jaxwind.pressure.matrix_free_gmg import _axis_diagonal, _solve_z_lines


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


def test_gmg_projects_only_at_v_cycle_boundary(monkeypatch) -> None:
    solver = _solver(RectilinearGrid.uniform(8, 8, 8))
    preconditioner = solver.preconditioner
    calls = [0] * len(preconditioner.operators)

    for index, operator in enumerate(preconditioner.operators):
        original = operator.project_nullspace

        def counted(field, *, _index=index, _original=original):
            calls[_index] += 1
            return _original(field)

        monkeypatch.setattr(operator, "project_nullspace", counted)

    rhs = jnp.sin(
        0.11 * jnp.arange(solver.operator.grid.cell_count, dtype=jnp.float32)
    ).reshape(solver.operator.shape)
    result = preconditioner.apply(rhs)

    assert calls == [2] + [0] * (len(calls) - 1)
    assert abs(float(solver.operator.volume_mean(result))) < 2.0e-6


def test_gmg_without_inner_projections_remains_symmetric() -> None:
    solver = _solver(RectilinearGrid.uniform(8, 8, 8))
    operator = solver.operator
    count = operator.grid.cell_count
    left = operator.project_nullspace(
        jnp.sin(0.07 * jnp.arange(count, dtype=jnp.float32)).reshape(operator.shape)
    )
    right = operator.project_nullspace(
        jnp.cos(0.13 * jnp.arange(count, dtype=jnp.float32)).reshape(operator.shape)
    )

    left_preconditioned = solver.preconditioner.apply(left)
    right_preconditioned = solver.preconditioner.apply(right)
    left_inner = operator.inner(left, right_preconditioned)
    right_inner = operator.inner(left_preconditioned, right)

    assert jnp.allclose(left_inner, right_inner, rtol=2.0e-5, atol=2.0e-5)


def test_z_semi_gmg_pcg_converges_without_inner_projections() -> None:
    grid = RectilinearGrid.uniform(8, 8, 8, lx=32.0, ly=32.0, lz=8.0)
    solver = _solver(grid)
    assert solver.preconditioner.coarsening_factors[0] == (1, 2, 2)
    exact = jnp.sin(
        0.09 * jnp.arange(grid.cell_count, dtype=jnp.float32)
    ).reshape(grid.shape)
    exact = solver.operator.project_nullspace(exact)

    result = solver.solve(solver.operator.apply(exact))
    error = solver.operator.project_nullspace(result.solution - exact)

    assert result.converged
    assert result.relative_residual < 2.0e-6
    assert float(solver.operator.norm(error) / solver.operator.norm(exact)) < 3.0e-5


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


def test_batched_z_line_solve_satisfies_every_column_system() -> None:
    grid = RectilinearGrid(
        (0.0, 1.0, 2.0, 3.0, 4.0),
        (0.0, 1.0, 2.0, 3.0, 4.0),
        (0.0, 0.1, 0.3, 0.7, 1.5, 3.0, 5.0, 8.0, 12.0),
    )
    operator = _solver(grid).operator
    rhs = jnp.sin(
        0.17 * jnp.arange(grid.cell_count, dtype=jnp.float32)
    ).reshape(grid.shape)

    solution = _solve_z_lines(operator, rhs)

    wx, wy, wz = operator._level.widths
    cx, cy, cz = operator._level.centers
    diagonal = (
        _axis_diagonal(
            wz,
            cz,
            operator.boundaries.z_lower,
            operator.boundaries.z_upper,
        )[:, None, None]
        + _axis_diagonal(
            wy,
            cy,
            operator.boundaries.y_lower,
            operator.boundaries.y_upper,
        )[None, :, None]
        + _axis_diagonal(
            wx,
            cx,
            operator.boundaries.x_lower,
            operator.boundaries.x_upper,
        )[None, None, :]
    )
    distance = cz[1:] - cz[:-1]
    lower = jnp.zeros_like(wz).at[1:].set(-1.0 / (wz[1:] * distance))
    upper = jnp.zeros_like(wz).at[:-1].set(-1.0 / (wz[:-1] * distance))
    reconstructed = diagonal * solution
    reconstructed = reconstructed.at[1:].add(
        lower[1:, None, None] * solution[:-1]
    )
    reconstructed = reconstructed.at[:-1].add(
        upper[:-1, None, None] * solution[1:]
    )

    assert jnp.allclose(reconstructed, rhs, rtol=2.0e-5, atol=2.0e-5)
