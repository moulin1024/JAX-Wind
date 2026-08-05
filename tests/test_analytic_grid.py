from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.domain import AnalyticAxisMapping, RectilinearGrid, analytic_axis_faces


def test_zero_strength_is_exact_uniform_limit() -> None:
    uniform = analytic_axis_faces(8, 400.0)
    exponential = analytic_axis_faces(
        8,
        400.0,
        AnalyticAxisMapping("exponential", 0.3, 0.0),
    )

    assert exponential == uniform


def test_boundary_and_interior_focus_cluster_independently() -> None:
    center = AnalyticAxisMapping("exponential", 0.5, 2.0)
    ground = AnalyticAxisMapping("exponential", 0.0, 2.0)
    grid = RectilinearGrid.analytic(
        64,
        64,
        64,
        lx=400.0,
        ly=400.0,
        lz=400.0,
        x=center,
        y=center,
        z=ground,
    )

    assert grid.uniform_axes == (False, False, False)
    assert grid.x_faces[32] == 200.0
    assert grid.y_faces[32] == 200.0
    assert grid.x_widths[31] < grid.x_widths[0]
    assert grid.x_widths[32] < grid.x_widths[-1]
    assert grid.x_widths[0] == pytest.approx(grid.x_widths[-1])
    assert grid.z_widths[0] < grid.z_widths[-1]
    assert grid.z_faces[0] == 0.0
    assert grid.z_faces[-1] == 400.0


def test_horizontal_mean_uses_physical_cell_area() -> None:
    grid = RectilinearGrid.analytic(
        8,
        8,
        4,
        lx=4.0,
        ly=6.0,
        lz=2.0,
        x=AnalyticAxisMapping("exponential", 0.5, 1.5),
        y=AnalyticAxisMapping("exponential", 0.5, 2.0),
    )
    x = np.asarray(grid.x_centers)
    y = np.asarray(grid.y_centers)
    field_2d = x[None, :] ** 2 + 0.25 * y[:, None] ** 2
    field = jnp.broadcast_to(jnp.asarray(field_2d), grid.shape)
    area = np.asarray(grid.y_widths)[:, None] * np.asarray(grid.x_widths)[None, :]
    expected = np.sum(field_2d * area) / np.sum(area)

    from jaxwind.pressure import (
        BoundaryCondition,
        MatrixFreePoissonSolver,
        PoissonBoundaryConditions,
    )
    from jaxwind.momentum import MomentumOperators

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
    )
    momentum = MomentumOperators(grid, pressure)

    np.testing.assert_allclose(
        np.asarray(momentum.horizontal_mean(field)),
        expected,
        rtol=2.0e-6,
    )
