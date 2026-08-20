from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    StaggeredVelocity,
    assemble_pressure_matrix,
    build_pressure_poisson,
    default_tolerance,
    divergence,
    matrix_vector_product,
    pressure_gradient,
    project,
)


def dense_matrix(matrix) -> np.ndarray:
    dense = np.zeros(matrix.shape)
    for row in range(matrix.row_count):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        dense[row, matrix.indices[start:stop]] = matrix.data[start:stop]
    return dense


def random_velocity(grid: UniformGrid, seed: int) -> StaggeredVelocity:
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    cells = (grid.nz, grid.ny, grid.nx)
    z_velocity = jax.random.normal(keys[2], (grid.nz + 1, grid.ny, grid.nx))
    return StaggeredVelocity(
        jax.random.normal(keys[0], cells),
        jax.random.normal(keys[1], cells),
        z_velocity.at[0].set(0.0).at[-1].set(0.0),
    )


class PressureMatrixTest(unittest.TestCase):
    grid = UniformGrid(8, 6, 4, 2.0, 1.5, 1.0)

    def test_unpinned_matrix_is_the_divergence_of_the_gradient(self) -> None:
        """The assembled operator must be the projection's own composition."""
        matrix = assemble_pressure_matrix(self.grid, reference_cell=None)
        pressure = jax.random.normal(
            jax.random.PRNGKey(3),
            (self.grid.nz, self.grid.ny, self.grid.nx),
        )
        laplacian = divergence(pressure_gradient(pressure, self.grid), self.grid)
        applied = matrix_vector_product(matrix, pressure.reshape(-1))
        self.assertLess(
            float(jnp.max(jnp.abs(applied.reshape(laplacian.shape) + laplacian))),
            1.0e-11,
        )

    def test_matrix_is_symmetric_and_positive_definite_once_pinned(self) -> None:
        unpinned = dense_matrix(assemble_pressure_matrix(self.grid, reference_cell=None))
        pinned = dense_matrix(assemble_pressure_matrix(self.grid))
        for dense in (unpinned, pinned):
            self.assertLess(float(np.max(np.abs(dense - dense.T))), 1.0e-14)
        self.assertLess(abs(float(np.min(np.linalg.eigvalsh(unpinned)))), 1.0e-12)
        self.assertGreater(float(np.min(np.linalg.eigvalsh(pinned))), 1.0e-3)

    def test_row_sums_vanish_away_from_the_pinned_cell(self) -> None:
        """A constant pressure must produce no gradient anywhere."""
        dense = dense_matrix(assemble_pressure_matrix(self.grid, reference_cell=None))
        self.assertLess(float(np.max(np.abs(dense.sum(axis=1)))), 1.0e-12)

    def test_single_cell_directions_drop_out(self) -> None:
        """A periodic direction of one cell contributes nothing to the operator."""
        grid = UniformGrid(1, 1, 4, 1.0, 1.0, 1.0)
        dense = dense_matrix(assemble_pressure_matrix(grid, reference_cell=None))
        expected = np.zeros((4, 4))
        for row in range(4):
            for column, value in ((row - 1, -1.0), (row + 1, -1.0)):
                if 0 <= column < 4:
                    expected[row, column] = value / grid.dz**2
                    expected[row, row] += 1.0 / grid.dz**2
        self.assertLess(float(np.max(np.abs(dense - expected))), 1.0e-12)

    def test_rejects_a_reference_cell_outside_the_mesh(self) -> None:
        with self.assertRaises(ValueError):
            assemble_pressure_matrix(self.grid, reference_cell=self.grid.cell_count)


class PoissonSolveTest(unittest.TestCase):
    def test_solution_reproduces_a_manufactured_right_hand_side(self) -> None:
        grid = UniformGrid(16, 12, 8, 2.0, 1.5, 1.0)
        poisson = build_pressure_poisson(grid, backend="fft")
        x, y, z = (jnp.asarray(axis) for axis in _cell_axes(grid))
        exact = (
            jnp.cos(2.0 * jnp.pi * x[None, None, :] / grid.lx)
            * jnp.sin(2.0 * jnp.pi * y[None, :, None] / grid.ly)
            * jnp.cos(jnp.pi * z[:, None, None] / grid.lz)
        )
        right_hand_side = divergence(pressure_gradient(exact, grid), grid)
        solved = poisson.solve(right_hand_side)
        self.assertLess(
            float(jnp.max(jnp.abs(solved - (exact - jnp.mean(exact))))),
            1.0e-8,
        )

    def test_residual_of_the_solve_is_negligible(self) -> None:
        grid = UniformGrid(8, 8, 8, 1.0, 1.0, 1.0)
        poisson = build_pressure_poisson(grid, backend="fft")
        velocity = random_velocity(grid, 11)
        right_hand_side = divergence(velocity, grid)
        pressure = poisson.solve(right_hand_side)
        scale = float(jnp.linalg.norm(right_hand_side))
        residual = float(poisson.residual_norm(pressure, right_hand_side))
        self.assertLess(residual, 1.0e-9 * scale)

    def test_rejects_removed_and_unknown_backends(self) -> None:
        grid = UniformGrid(4, 4, 4, 1.0, 1.0, 1.0)
        for backend in ("cg", "multigrid"):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                build_pressure_poisson(grid, backend=backend)


class SinglePrecisionTest(unittest.TestCase):
    """Single precision must stay single precision end to end."""

    grid = UniformGrid(8, 8, 8, 1.0, 1.0, 1.0)

    def test_the_tolerance_follows_the_precision(self) -> None:
        self.assertGreater(default_tolerance("float32"), default_tolerance("float64"))

    def test_the_assembled_matrix_keeps_the_requested_dtype(self) -> None:
        matrix = assemble_pressure_matrix(self.grid, dtype="float32")
        self.assertEqual(matrix.data.dtype, np.dtype("float32"))

    def test_the_solve_neither_promotes_nor_loses_accuracy(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="gmg", dtype="float32")
        velocity = random_velocity(self.grid, 21)
        single = StaggeredVelocity(
            velocity.x.astype(jnp.float32),
            velocity.y.astype(jnp.float32),
            velocity.z.astype(jnp.float32),
        )
        right_hand_side = divergence(single, self.grid)
        pressure = poisson.solve(right_hand_side)
        self.assertEqual(pressure.dtype, jnp.dtype("float32"))
        scale = float(jnp.linalg.norm(right_hand_side))
        residual = float(poisson.residual_norm(pressure, right_hand_side))
        self.assertLess(residual, 1.0e-5 * scale)

    def test_the_projection_reaches_single_precision_round_off(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="gmg", dtype="float32")
        velocity = random_velocity(self.grid, 22)
        single = StaggeredVelocity(
            velocity.x.astype(jnp.float32),
            velocity.y.astype(jnp.float32),
            velocity.z.astype(jnp.float32),
        )
        before = float(jnp.max(jnp.abs(divergence(single, self.grid))))
        projected, _ = project(single, poisson, 0.05)
        after = float(jnp.max(jnp.abs(divergence(projected, self.grid))))
        self.assertLess(after, 1.0e-5 * before)


class ProjectionTest(unittest.TestCase):
    grid = UniformGrid(12, 10, 8, 1.5, 1.25, 1.0)

    def test_projection_removes_divergence_to_round_off(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="fft")
        velocity = random_velocity(self.grid, 5)
        before = float(jnp.max(jnp.abs(divergence(velocity, self.grid))))
        projected, _ = project(velocity, poisson, 0.05)
        after = float(jnp.max(jnp.abs(divergence(projected, self.grid))))
        self.assertGreater(before, 1.0)
        self.assertLess(after, 1.0e-9 * before)

    def test_projection_keeps_the_walls_impermeable(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="fft")
        projected, _ = project(random_velocity(self.grid, 7), poisson, 0.05)
        self.assertLess(float(jnp.max(jnp.abs(projected.z[0]))), 1.0e-14)
        self.assertLess(float(jnp.max(jnp.abs(projected.z[-1]))), 1.0e-14)

    def test_projection_leaves_a_solenoidal_field_alone(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="fft")
        solenoidal, _ = project(random_velocity(self.grid, 9), poisson, 0.05)
        again, pressure = project(solenoidal, poisson, 0.05)
        self.assertLess(
            float(jnp.max(jnp.abs(again.x - solenoidal.x))),
            1.0e-9 * float(jnp.max(jnp.abs(solenoidal.x))),
        )
        self.assertLess(float(jnp.max(jnp.abs(pressure))), 1.0e-9)


def _cell_axes(grid: UniformGrid):
    return (
        (np.arange(grid.nx) + 0.5) * grid.dx,
        (np.arange(grid.ny) + 0.5) * grid.dy,
        (np.arange(grid.nz) + 0.5) * grid.dz,
    )


if __name__ == "__main__":
    unittest.main()
