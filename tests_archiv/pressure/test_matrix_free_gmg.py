from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from jaxwind_archiv.pressure import (
    BoundaryCondition,
    FGMRESConfig,
    GMGConfig,
    MatrixFreeGMG,
    MatrixFreePoissonOperator,
    MatrixFreePoissonSolver,
    PCGConfig,
    PoissonBoundaryConditions,
    RectilinearGrid,
)


def _solver(
    grid: RectilinearGrid,
    boundaries: PoissonBoundaryConditions,
) -> MatrixFreePoissonSolver:
    return MatrixFreePoissonSolver(
        grid,
        boundaries,
        dtype=jnp.float32,
        gmg=GMGConfig(coarse_smooth=30),
        krylov=FGMRESConfig(
            restart=20,
            max_iterations=80,
            relative_tolerance=2.0e-6,
        ),
    )


def test_periodic_operator_has_constant_nullspace_and_weighted_symmetry() -> None:
    grid = RectilinearGrid(
        tuple(np.linspace(0.0, 1.0, 9) ** 1.1),
        tuple(np.linspace(0.0, 1.0, 7) ** 1.2),
        tuple(np.linspace(0.0, 1.0, 9) ** 1.4),
    )
    operator = MatrixFreePoissonOperator(
        grid,
        PoissonBoundaryConditions.periodic(),
        dtype=jnp.float32,
    )
    left = jnp.sin(jnp.arange(grid.cell_count, dtype=jnp.float32)).reshape(
        grid.shape
    )
    right = jnp.cos(
        0.37 * jnp.arange(grid.cell_count, dtype=jnp.float32)
    ).reshape(grid.shape)

    constant_residual = operator.apply(jnp.ones(grid.shape, dtype=jnp.float32))
    left_action = operator.inner(left, operator.apply(right))
    right_action = operator.inner(operator.apply(left), right)

    assert float(jnp.max(jnp.abs(constant_residual))) < 2.0e-5
    assert float(jnp.abs(left_action - right_action)) < 2.0e-4


def test_periodic_gmg_fgmres_recovers_a_discrete_manufactured_solution() -> None:
    grid = RectilinearGrid.uniform(8, 8, 8)
    solver = _solver(grid, PoissonBoundaryConditions.periodic())
    z, y, x = [
        (jnp.arange(count, dtype=jnp.float32) + 0.5) / count
        for count in grid.shape
    ]
    exact = (
        jnp.cos(2.0 * jnp.pi * x)[None, None, :]
        + 0.4 * jnp.cos(4.0 * jnp.pi * y)[None, :, None]
        + 0.2 * jnp.cos(2.0 * jnp.pi * z)[:, None, None]
    )
    rhs = solver.operator.apply(exact)

    result = solver.solve(rhs)
    error = solver.operator.project_nullspace(result.solution - exact)
    relative_error = solver.operator.norm(error) / solver.operator.norm(exact)

    assert result.converged
    assert result.relative_residual < 2.0e-6
    assert float(relative_error) < 5.0e-6
    assert solver.preconditioner.level_shapes == (
        (8, 8, 8),
        (4, 4, 4),
        (2, 2, 2),
    )


def test_device_gmres_preserves_weighted_accuracy_on_stretched_grid() -> None:
    grid = RectilinearGrid(
        tuple(np.linspace(0.0, 1.0, 9) ** 1.1),
        tuple(np.linspace(0.0, 1.0, 9) ** 1.2),
        tuple(np.linspace(0.0, 1.0, 9) ** 1.3),
    )
    solver = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.periodic(),
        dtype=jnp.float32,
        gmg=GMGConfig(coarse_smooth=30),
        krylov=FGMRESConfig(
            restart=10,
            max_iterations=40,
            relative_tolerance=2.0e-6,
            execution="jax",
        ),
    )
    exact = jnp.sin(
        0.1 * jnp.arange(grid.cell_count, dtype=jnp.float32)
    ).reshape(grid.shape)
    exact = solver.operator.project_nullspace(exact)
    result = solver.solve(solver.operator.apply(exact))
    error = solver.operator.project_nullspace(result.solution - exact)
    relative_error = solver.operator.norm(error) / solver.operator.norm(exact)

    assert result.converged
    assert result.relative_residual < 2.0e-6
    assert float(relative_error) < 2.0e-5


def test_gmg_is_a_weighted_symmetric_positive_preconditioner() -> None:
    grid = RectilinearGrid(
        tuple(np.linspace(0.0, 1.0, 9) ** 1.1),
        tuple(np.linspace(0.0, 1.0, 9) ** 1.2),
        tuple(np.linspace(0.0, 1.0, 17) ** 1.8),
    )
    operator = MatrixFreePoissonOperator(
        grid,
        PoissonBoundaryConditions.homogeneous_neumann(),
        dtype=jnp.float32,
    )
    preconditioner = MatrixFreeGMG(operator, GMGConfig(coarse_smooth=20))
    left = operator.project_nullspace(
        jnp.sin(
            0.17 * jnp.arange(grid.cell_count, dtype=jnp.float32)
        ).reshape(grid.shape)
    )
    right = operator.project_nullspace(
        jnp.cos(
            0.11 * jnp.arange(grid.cell_count, dtype=jnp.float32)
        ).reshape(grid.shape)
    )
    preconditioned_left = preconditioner.apply(left)
    preconditioned_right = preconditioner.apply(right)
    left_inner = operator.inner(left, preconditioned_right)
    right_inner = operator.inner(preconditioned_left, right)
    scale = jnp.maximum(jnp.maximum(jnp.abs(left_inner), jnp.abs(right_inner)), 1.0)

    assert float(jnp.abs(left_inner - right_inner) / scale) < 2.0e-5
    assert float(operator.inner(left, preconditioned_left)) > 0.0
    assert preconditioner.level_smoothers == (
        "z_line",
    ) * len(preconditioner.level_shapes)


def test_device_pcg_preserves_weighted_accuracy_on_stretched_grid() -> None:
    grid = RectilinearGrid(
        tuple(np.linspace(0.0, 1.0, 9) ** 1.1),
        tuple(np.linspace(0.0, 1.0, 9) ** 1.2),
        tuple(np.linspace(0.0, 1.0, 9) ** 1.3),
    )
    solver = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.periodic(),
        dtype=jnp.float32,
        gmg=GMGConfig(coarse_smooth=30),
        krylov=PCGConfig(
            max_iterations=40,
            relative_tolerance=2.0e-6,
            execution="jax",
        ),
    )
    exact = jnp.sin(
        0.1 * jnp.arange(grid.cell_count, dtype=jnp.float32)
    ).reshape(grid.shape)
    exact = solver.operator.project_nullspace(exact)
    result = solver.solve(solver.operator.apply(exact))
    error = solver.operator.project_nullspace(result.solution - exact)
    relative_error = solver.operator.norm(error) / solver.operator.norm(exact)

    assert result.converged
    assert result.relative_residual < 2.0e-6
    assert float(relative_error) < 2.0e-5


def test_python_pcg_recovers_a_discrete_manufactured_solution() -> None:
    grid = RectilinearGrid.uniform(8, 8, 8)
    solver = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.periodic(),
        dtype=jnp.float32,
        gmg=GMGConfig(coarse_smooth=30),
        krylov=PCGConfig(
            max_iterations=40,
            relative_tolerance=2.0e-6,
        ),
    )
    exact = jnp.sin(
        0.13 * jnp.arange(grid.cell_count, dtype=jnp.float32)
    ).reshape(grid.shape)
    exact = solver.operator.project_nullspace(exact)
    result = solver.solve(solver.operator.apply(exact))
    error = solver.operator.project_nullspace(result.solution - exact)

    assert result.converged
    assert result.iterations < 20
    assert result.relative_residual < 2.0e-6
    assert float(solver.operator.norm(error) / solver.operator.norm(exact)) < (
        2.0e-5
    )


def test_gmg_rejects_asymmetric_smoothing_counts() -> None:
    with pytest.raises(ValueError, match="symmetric GMG"):
        GMGConfig(pre_smooth=1, post_smooth=2)


def test_stretched_grid_honours_nonhomogeneous_dirichlet_data() -> None:
    def faces(count: int, exponent: float) -> tuple[float, ...]:
        return tuple(np.linspace(0.0, 1.0, count + 1) ** exponent)

    grid = RectilinearGrid(faces(8, 1.0), faces(6, 1.2), faces(8, 1.6))
    zero_flux = BoundaryCondition("neumann")
    boundaries = PoissonBoundaryConditions(
        BoundaryCondition("dirichlet", 0.0),
        BoundaryCondition("dirichlet", 1.0),
        zero_flux,
        zero_flux,
        zero_flux,
        zero_flux,
    )
    solver = _solver(grid, boundaries)
    x_faces = jnp.asarray(grid.x_faces, dtype=jnp.float32)
    x_centers = 0.5 * (x_faces[1:] + x_faces[:-1])
    exact = jnp.broadcast_to(x_centers[None, None, :], grid.shape)

    result = solver.solve(jnp.zeros(grid.shape, dtype=jnp.float32))
    relative_error = solver.operator.norm(result.solution - exact) / (
        solver.operator.norm(exact)
    )

    assert result.converged
    assert result.relative_residual < 2.0e-6
    assert float(relative_error) < 8.0e-6


def test_pure_neumann_rhs_is_projected_to_the_compatibility_subspace() -> None:
    grid = RectilinearGrid.uniform(8, 6, 4)
    solver = _solver(grid, PoissonBoundaryConditions.homogeneous_neumann())
    rhs = jnp.full(grid.shape, 3.25, dtype=jnp.float32)

    result = solver.solve(rhs)

    assert result.converged
    assert result.iterations == 0
    assert result.compatibility_shift == pytest.approx(3.25, abs=2.0e-6)
    assert float(jnp.max(jnp.abs(result.solution))) == 0.0


def test_outward_neumann_sign_recovers_a_linear_pressure_field() -> None:
    grid = RectilinearGrid.uniform(8, 6, 4)
    zero_flux = BoundaryCondition("neumann")
    boundaries = PoissonBoundaryConditions(
        BoundaryCondition("neumann", -1.0),
        BoundaryCondition("neumann", 1.0),
        zero_flux,
        zero_flux,
        zero_flux,
        zero_flux,
    )
    solver = _solver(grid, boundaries)
    x_faces = jnp.asarray(grid.x_faces, dtype=jnp.float32)
    x_centers = 0.5 * (x_faces[1:] + x_faces[:-1])
    exact = jnp.broadcast_to(x_centers[None, None, :], grid.shape)
    exact = solver.operator.project_nullspace(exact)

    result = solver.solve(jnp.zeros(grid.shape, dtype=jnp.float32))
    relative_error = solver.operator.norm(result.solution - exact) / (
        solver.operator.norm(exact)
    )

    assert result.converged
    assert result.compatibility_shift == pytest.approx(0.0, abs=2.0e-6)
    assert float(relative_error) < 8.0e-6


def test_periodicity_must_be_paired_on_each_axis() -> None:
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    with pytest.raises(ValueError, match="x boundaries"):
        PoissonBoundaryConditions(
            periodic,
            neumann,
            neumann,
            neumann,
            neumann,
            neumann,
        )


def test_z_line_smoothing_and_semi_coarsening_target_abl_anisotropy() -> None:
    def faces(count: int, exponent: float) -> tuple[float, ...]:
        return tuple(np.linspace(0.0, 1.0, count + 1) ** exponent)

    grid = RectilinearGrid(faces(8, 1.0), faces(8, 1.0), faces(16, 2.0))
    neumann = BoundaryCondition("neumann")
    dirichlet = BoundaryCondition("dirichlet")
    boundaries = PoissonBoundaryConditions(
        neumann,
        neumann,
        neumann,
        neumann,
        dirichlet,
        dirichlet,
    )
    operator = MatrixFreePoissonOperator(
        grid,
        boundaries,
        dtype=jnp.float32,
    )
    z_faces = jnp.asarray(grid.z_faces, dtype=jnp.float32)
    z = 0.5 * (z_faces[1:] + z_faces[:-1])
    exact = jnp.broadcast_to(
        jnp.sin(7.0 * jnp.pi * z)[:, None, None],
        grid.shape,
    )
    rhs = operator.apply(exact)
    common = dict(pre_smooth=1, post_smooth=1, coarse_smooth=20)
    baseline = MatrixFreeGMG(
        operator,
        GMGConfig(smoother="jacobi", coarsening="full", **common),
    )
    abl = MatrixFreeGMG(
        operator,
        GMGConfig(smoother="auto", coarsening="auto", **common),
    )

    baseline_error = rhs - operator.apply(baseline.apply(rhs))
    abl_error = rhs - operator.apply(abl.apply(rhs))
    baseline_ratio = operator.norm(baseline_error) / operator.norm(rhs)
    abl_ratio = operator.norm(abl_error) / operator.norm(rhs)

    assert abl.level_smoothers == ("z_line",) * len(abl.level_shapes)
    assert all(shape[0] == grid.shape[0] for shape in abl.level_shapes)
    assert float(abl_ratio) < 0.04
    assert float(abl_ratio) < 0.2 * float(baseline_ratio)
