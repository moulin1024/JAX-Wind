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
from jaxwind.fv.poisson import (
    _build_gmg_levels,
    _coarsening_factors,
    _prolong,
    _restrict,
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


class GmgHierarchyTest(unittest.TestCase):
    def test_each_axis_coarsens_only_while_it_is_even(self) -> None:
        """A mesh like 12 = 4 * 3 should stall the x-axis, not the whole level."""
        grid = UniformGrid(16, 12, 8, 2.0, 1.5, 1.0)
        levels, factors = _build_gmg_levels(grid)
        shapes = [(level.nx, level.ny, level.nz) for level in levels]
        self.assertEqual(shapes[0], (16, 12, 8))
        self.assertEqual(shapes[-1], (1, 3, 1))
        for level, next_level, factor in zip(levels, levels[1:], factors):
            self.assertEqual(level.nx // factor[0], next_level.nx)
            self.assertEqual(level.ny // factor[1], next_level.ny)
            self.assertEqual(level.nz // factor[2], next_level.nz)

    def test_a_fully_coarsenable_mesh_bottoms_out_at_one_cell(self) -> None:
        grid = UniformGrid(8, 8, 8, 1.0, 1.0, 1.0)
        levels, _ = _build_gmg_levels(grid)
        self.assertEqual((levels[-1].nx, levels[-1].ny, levels[-1].nz), (1, 1, 1))

    def test_an_already_odd_mesh_has_a_single_level(self) -> None:
        grid = UniformGrid(3, 5, 7, 1.0, 1.0, 1.0)
        levels, factors = _build_gmg_levels(grid)
        self.assertEqual(len(levels), 1)
        self.assertEqual(factors, [])

    def test_coarsening_factors_stop_at_odd_or_unit_axes(self) -> None:
        grid = UniformGrid(1, 3, 4, 1.0, 1.0, 1.0)
        self.assertEqual(_coarsening_factors(grid), (1, 1, 2))

    def test_transfer_operators_preserve_constants(self) -> None:
        factors = (2, 2, 2)
        coarse = jnp.ones((3, 4, 5))
        fine = jnp.ones((6, 8, 10))
        self.assertTrue(bool(jnp.all(_prolong(coarse, factors) == 1.0)))
        self.assertTrue(bool(jnp.all(_restrict(fine, factors) == 1.0)))

    def test_restriction_is_the_scaled_adjoint_of_prolongation(self) -> None:
        factors = (2, 2, 2)
        coarse = jax.random.normal(jax.random.PRNGKey(31), (3, 4, 5))
        fine = jax.random.normal(jax.random.PRNGKey(32), (6, 8, 10))
        fine_inner_product = jnp.vdot(_prolong(coarse, factors), fine)
        coarse_inner_product = 8.0 * jnp.vdot(coarse, _restrict(fine, factors))
        self.assertLess(float(jnp.abs(fine_inner_product - coarse_inner_product)), 1.0e-12)


class GmgConfigurationTest(unittest.TestCase):
    def test_needs_no_external_dependency(self) -> None:
        grid = UniformGrid(4, 4, 4, 1.0, 1.0, 1.0)
        build_pressure_poisson(grid, backend="gmg")

    def test_the_assembled_matrix_stays_unpinned(self) -> None:
        grid = UniformGrid(6, 6, 4, 1.0, 1.0, 1.0)
        poisson = build_pressure_poisson(grid, backend="gmg")
        self.assertIsNone(poisson.matrix.reference_cell)


class GmgSolveTest(unittest.TestCase):
    def test_solution_reproduces_a_manufactured_right_hand_side(self) -> None:
        grid = UniformGrid(16, 12, 8, 2.0, 1.5, 1.0)
        poisson = build_pressure_poisson(grid, backend="gmg")
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
            1.0e-7,
        )

    def test_residual_of_the_solve_is_negligible(self) -> None:
        grid = UniformGrid(8, 8, 8, 1.0, 1.0, 1.0)
        poisson = build_pressure_poisson(grid, backend="gmg")
        velocity = random_velocity(grid, 11)
        right_hand_side = divergence(velocity, grid)
        pressure = poisson.solve(right_hand_side)
        scale = float(jnp.linalg.norm(right_hand_side))
        residual = float(poisson.residual_norm(pressure, right_hand_side))
        self.assertLess(residual, 1.0e-8 * scale)

    def test_agrees_with_the_fft_backend(self) -> None:
        grid = UniformGrid(10, 8, 6, 1.5, 1.25, 1.0)
        velocity = random_velocity(grid, 3)
        right_hand_side = divergence(velocity, grid)
        gmg = build_pressure_poisson(grid, backend="gmg").solve(right_hand_side)
        reference = build_pressure_poisson(grid, backend="fft").solve(right_hand_side)
        scale = float(jnp.max(jnp.abs(reference)))
        self.assertGreater(scale, 0.0)
        self.assertLess(float(jnp.max(jnp.abs(gmg - reference))), 1.0e-6 * scale)

    def test_an_odd_mesh_still_solves_via_the_coarsest_level_fallback(self) -> None:
        """No factor of two anywhere still has to fall back to an exact solve."""
        grid = UniformGrid(3, 5, 7, 1.0, 1.0, 1.0)
        velocity = random_velocity(grid, 17)
        right_hand_side = divergence(velocity, grid)
        gmg = build_pressure_poisson(grid, backend="gmg").solve(right_hand_side)
        reference = build_pressure_poisson(grid, backend="fft").solve(right_hand_side)
        scale = float(jnp.max(jnp.abs(reference)))
        self.assertGreater(scale, 0.0)
        self.assertLess(float(jnp.max(jnp.abs(gmg - reference))), 1.0e-6 * scale)


class GmgSinglePrecisionTest(unittest.TestCase):
    grid = UniformGrid(8, 8, 8, 1.0, 1.0, 1.0)

    def test_the_solve_keeps_the_requested_dtype(self) -> None:
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
        self.assertLess(residual, 1.0e-4 * scale)


class GmgProjectionTest(unittest.TestCase):
    grid = UniformGrid(12, 10, 8, 1.5, 1.25, 1.0)

    def test_projection_removes_divergence(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="gmg")
        velocity = random_velocity(self.grid, 5)
        before = float(jnp.max(jnp.abs(divergence(velocity, self.grid))))
        projected, _ = project(velocity, poisson, 0.05)
        after = float(jnp.max(jnp.abs(divergence(projected, self.grid))))
        self.assertGreater(before, 1.0)
        self.assertLess(after, 1.0e-7 * before)

    def test_projection_keeps_the_walls_impermeable(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="gmg")
        projected, _ = project(random_velocity(self.grid, 7), poisson, 0.05)
        self.assertLess(float(jnp.max(jnp.abs(projected.z[0]))), 1.0e-14)
        self.assertLess(float(jnp.max(jnp.abs(projected.z[-1]))), 1.0e-14)

    def test_the_solve_runs_under_jit(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="gmg")
        compiled = jax.jit(lambda field: project(field, poisson, 0.05)[0])
        projected = compiled(random_velocity(self.grid, 9))
        self.assertTrue(bool(jnp.all(jnp.isfinite(projected.x))))
        self.assertLess(
            float(jnp.max(jnp.abs(divergence(projected, self.grid)))),
            1.0e-7,
        )


if __name__ == "__main__":
    unittest.main()
