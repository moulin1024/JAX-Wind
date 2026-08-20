from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    StaggeredVelocity,
    build_pressure_poisson,
    divergence,
    pressure_gradient,
    project,
)


def random_velocity(grid: UniformGrid, seed: int) -> StaggeredVelocity:
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    cells = (grid.nz, grid.ny, grid.nx)
    z_velocity = jax.random.normal(keys[2], (grid.nz + 1, grid.ny, grid.nx))
    return StaggeredVelocity(
        jax.random.normal(keys[0], cells),
        jax.random.normal(keys[1], cells),
        z_velocity.at[0].set(0.0).at[-1].set(0.0),
    )


def _cell_axes(grid: UniformGrid):
    return (
        (np.arange(grid.nx) + 0.5) * grid.dx,
        (np.arange(grid.ny) + 0.5) * grid.dy,
        (np.arange(grid.nz) + 0.5) * grid.dz,
    )


class FftConfigurationTest(unittest.TestCase):
    def test_needs_no_external_dependency(self) -> None:
        """Unlike ``amg``, building the backend must not touch jaxamg."""
        grid = UniformGrid(4, 4, 4, 1.0, 1.0, 1.0)
        build_pressure_poisson(grid, backend="fft")

    def test_the_assembled_matrix_stays_unpinned(self) -> None:
        """The direct solve handles the null space itself, not a pinned row."""
        grid = UniformGrid(6, 6, 4, 1.0, 1.0, 1.0)
        poisson = build_pressure_poisson(grid, backend="fft")
        self.assertIsNone(poisson.matrix.reference_cell)


class FftSolveTest(unittest.TestCase):
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
            1.0e-9,
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

    def test_agrees_with_the_gmg_backend(self) -> None:
        """Both backends solve the same assembled operator."""
        grid = UniformGrid(10, 8, 6, 1.5, 1.25, 1.0)
        velocity = random_velocity(grid, 3)
        right_hand_side = divergence(velocity, grid)
        fft = build_pressure_poisson(grid, backend="fft").solve(right_hand_side)
        reference = build_pressure_poisson(grid, backend="gmg").solve(right_hand_side)
        scale = float(jnp.max(jnp.abs(reference)))
        self.assertGreater(scale, 0.0)
        self.assertLess(float(jnp.max(jnp.abs(fft - reference))), 1.0e-8 * scale)

    def test_a_non_power_of_two_mesh_is_handled(self) -> None:
        """The FFT must not silently assume a power-of-two cell count."""
        grid = UniformGrid(6, 10, 5, 1.2, 2.0, 1.0)
        velocity = random_velocity(grid, 17)
        right_hand_side = divergence(velocity, grid)
        fft = build_pressure_poisson(grid, backend="fft").solve(right_hand_side)
        reference = build_pressure_poisson(grid, backend="gmg").solve(right_hand_side)
        scale = float(jnp.max(jnp.abs(reference)))
        self.assertGreater(scale, 0.0)
        self.assertLess(float(jnp.max(jnp.abs(fft - reference))), 1.0e-8 * scale)

    def test_a_single_vertical_cell_is_handled(self) -> None:
        grid = UniformGrid(4, 4, 1, 1.0, 1.0, 1.0)
        velocity = random_velocity(grid, 19)
        right_hand_side = divergence(velocity, grid)
        poisson = build_pressure_poisson(grid, backend="fft")
        pressure = poisson.solve(right_hand_side)
        self.assertTrue(bool(jnp.all(jnp.isfinite(pressure))))
        scale = float(jnp.linalg.norm(right_hand_side))
        self.assertLess(
            float(poisson.residual_norm(pressure, right_hand_side)), 1.0e-9 * scale
        )


class FftSinglePrecisionTest(unittest.TestCase):
    grid = UniformGrid(8, 8, 8, 1.0, 1.0, 1.0)

    def test_the_solve_keeps_the_requested_dtype(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="fft", dtype="float32")
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
        self.assertLess(residual, 1.0e-4 * scale)


class FftProjectionTest(unittest.TestCase):
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

    def test_the_solve_runs_under_jit(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="fft")
        compiled = jax.jit(lambda field: project(field, poisson, 0.05)[0])
        projected = compiled(random_velocity(self.grid, 9))
        self.assertTrue(bool(jnp.all(jnp.isfinite(projected.x))))
        self.assertLess(
            float(jnp.max(jnp.abs(divergence(projected, self.grid)))),
            1.0e-9,
        )


if __name__ == "__main__":
    unittest.main()
