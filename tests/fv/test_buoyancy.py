from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    FREE_SLIP,
    Boundaries,
    FlowModel,
    LinearBoussinesqBuoyancy,
    PassiveScalar,
    Wall,
    boussinesq_tendency,
    build_atmospheric_step,
    build_pressure_poisson,
    divergence,
    initial_atmospheric_solution,
)


class BoussinesqTendencyTest(unittest.TestCase):
    grid = UniformGrid(6, 4, 5, 3.0, 2.0, 2.5)

    def test_tendency_is_hydrostatic_free_and_impermeable(self) -> None:
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        horizontal = jnp.arange(self.grid.nx, dtype=jnp.float64)[None, :]
        scalar = 300.0 + jnp.broadcast_to(horizontal, shape)
        coefficient = 9.81 / 300.0
        tendency = boussinesq_tendency(
            scalar,
            LinearBoussinesqBuoyancy(coefficient),
        )

        self.assertTrue(bool(jnp.all(tendency.x == 0.0)))
        self.assertTrue(bool(jnp.all(tendency.y == 0.0)))
        self.assertTrue(bool(jnp.all(tendency.z[0] == 0.0)))
        self.assertTrue(bool(jnp.all(tendency.z[-1] == 0.0)))
        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.mean(tendency.z[1:-1], axis=(-2, -1)),
                    0.0,
                    atol=1.0e-14,
                )
            )
        )
        expected = coefficient * (horizontal - jnp.mean(horizontal))
        self.assertTrue(bool(jnp.allclose(tendency.z[1], expected)))

    def test_coupled_step_projects_the_buoyant_acceleration(self) -> None:
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        scalar = jnp.broadcast_to(
            jnp.sin(2.0 * jnp.pi * jnp.arange(self.grid.nx) / self.grid.nx),
            shape,
        )
        boundaries = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP))
        poisson = build_pressure_poisson(self.grid, backend="fft")
        step = build_atmospheric_step(
            self.grid,
            boundaries,
            poisson,
            FlowModel(),
            PassiveScalar(),
            LinearBoussinesqBuoyancy(0.1),
        )
        initial = initial_atmospheric_solution(
            self.grid,
            scalar=scalar,
            dtype="float64",
        )
        final = step(initial, 0.1)

        self.assertGreater(float(jnp.max(jnp.abs(final.velocity.z))), 0.0)
        self.assertLess(
            float(jnp.max(jnp.abs(divergence(final.velocity, self.grid)))),
            1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
