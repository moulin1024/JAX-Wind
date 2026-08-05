from __future__ import annotations

import math

import jax.numpy as jnp

from jaxwind.momentum import LASDModel, MultilevelLASD
from jaxwind.pressure import (
    GMGConfig,
    MatrixFreePoissonSolver,
    PCGConfig,
    PoissonBoundaryConditions,
    RectilinearGrid,
)


def _closure(
    grid: RectilinearGrid,
) -> tuple[MatrixFreePoissonSolver, MultilevelLASD]:
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.homogeneous_neumann(),
        dtype=jnp.float32,
        gmg=GMGConfig(coarsening="full", min_coarse_cells=1),
        krylov=PCGConfig(max_iterations=20, relative_tolerance=1.0e-5),
    )
    return pressure, MultilevelLASD(
        multigrid=pressure.preconditioner,
        model=LASDModel(),
    )


def test_lasd_shares_pressure_grid_hierarchy() -> None:
    pressure, closure = _closure(RectilinearGrid.uniform(8, 8, 8))

    assert closure.hierarchy is pressure.preconditioner.hierarchy
    assert closure.hierarchy.level_shapes[:3] == (
        (8, 8, 8),
        (4, 4, 4),
        (2, 2, 2),
    )


def test_lasd_uses_actual_z_semi_coarsening_scale_ratio() -> None:
    grid = RectilinearGrid.uniform(8, 8, 8, lx=32.0, ly=32.0, lz=8.0)
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.homogeneous_neumann(),
        dtype=jnp.float32,
        gmg=GMGConfig(coarsening="auto", anisotropy_threshold=4.0),
        krylov=PCGConfig(max_iterations=20, relative_tolerance=1.0e-5),
    )
    closure = MultilevelLASD(
        multigrid=pressure.preconditioner,
        model=LASDModel(),
    )

    assert closure.hierarchy.coarsening_factors[:2] == ((1, 2, 2), (1, 2, 2))
    assert math.isclose(closure.test_ratio, 4.0 ** (1.0 / 3.0))


def test_conservative_restriction_preserves_constants_and_integrals() -> None:
    grid = RectilinearGrid(
        (0.0, 1.0, 3.0, 6.0, 10.0),
        (0.0, 2.0, 3.0, 7.0, 8.0),
        (0.0, 0.5, 2.0, 3.0, 6.0),
    )
    pressure, _ = _closure(grid)
    hierarchy = pressure.preconditioner.hierarchy
    constant = jnp.ones((*grid.shape, 2), dtype=jnp.float32)
    restricted = hierarchy.restrict(constant)

    fine_volume = pressure.operator.volume
    coarse_volume = pressure.preconditioner.operators[1].volume
    signal = jnp.arange(grid.cell_count, dtype=jnp.float32).reshape(grid.shape)
    coarse_signal = hierarchy.restrict(signal)

    assert jnp.allclose(restricted, 1.0)
    assert jnp.allclose(
        jnp.sum(signal * fine_volume),
        jnp.sum(coarse_signal * coarse_volume),
        rtol=2.0e-6,
    )


def test_multilevel_lasd_update_is_finite_bounded_and_coarse_state() -> None:
    grid = RectilinearGrid.uniform(8, 8, 8)
    _, closure = _closure(grid)
    z, y, x = jnp.meshgrid(
        jnp.arange(8, dtype=jnp.float32),
        jnp.arange(8, dtype=jnp.float32),
        jnp.arange(8, dtype=jnp.float32),
        indexing="ij",
    )
    velocity = jnp.stack(
        (
            jnp.sin(0.3 * x + 0.2 * y),
            jnp.cos(0.4 * y - 0.1 * z),
            0.2 * jnp.sin(0.5 * x + 0.3 * z),
        ),
        axis=-1,
    )
    gradient = jnp.stack(
        tuple(
            jnp.stack(
                (
                    jnp.gradient(velocity[..., component], axis=2),
                    jnp.gradient(velocity[..., component], axis=1),
                    jnp.gradient(velocity[..., component], axis=0),
                ),
                axis=-1,
            )
            for component in range(3)
        ),
        axis=-2,
    )
    state = closure.accumulate(closure.initialize(velocity), velocity)
    updated = closure.update(
        state,
        velocity,
        gradient,
        interval_dt=0.1,
        first_update=True,
    )

    assert updated.coefficient.shape == grid.shape
    assert all(field.shape == (4, 4, 4) for field in updated[1:])
    assert jnp.all(jnp.isfinite(updated.coefficient))
    assert jnp.all(
        updated.coefficient
        >= jnp.asarray(closure.model.minimum_coefficient, dtype=jnp.float32)
    )
    assert float(jnp.max(updated.coefficient)) <= closure.model.maximum_coefficient
    assert float(jnp.max(jnp.abs(updated.trajectory_x))) == 0.0
