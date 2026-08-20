from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    FREE_SLIP,
    Boundaries,
    CoriolisGeostrophic,
    FlowModel,
    PassiveScalar,
    StaggeredVelocity,
    Wall,
    build_adaptive_atmospheric_run,
    build_atmospheric_run,
    build_atmospheric_step,
    build_pressure_poisson,
    divergence,
    initial_atmospheric_solution,
)


class AtmosphericStepTest(unittest.TestCase):
    grid = UniformGrid(8, 8, 8, 2.0, 2.0, 1.0)

    def test_coupled_ab2_step_projects_and_advances_surface_scalar(self) -> None:
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        velocity = StaggeredVelocity(
            jnp.full(shape, 10.0),
            jnp.zeros(shape),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        momentum = FlowModel(
            rotation=CoriolisGeostrophic(1.0e-4, 10.0, 0.0, 1.0e-4)
        )
        scalar = PassiveScalar(lower_flux=1.0e-3)
        poisson = build_pressure_poisson(self.grid, backend="fft")
        boundaries = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP))
        step = build_atmospheric_step(
            self.grid,
            boundaries,
            poisson,
            momentum,
            scalar,
        )
        run = build_atmospheric_run(step)
        initial = initial_atmospheric_solution(
            self.grid,
            velocity,
            jnp.zeros(shape),
            dtype="float64",
        )
        dt, steps = 0.1, 3
        final = run(initial, dt, steps)
        self.assertLess(
            float(jnp.max(jnp.abs(divergence(final.velocity, self.grid)))),
            1.0e-10,
        )
        self.assertAlmostEqual(
            float(jnp.mean(final.scalar)),
            dt * steps * scalar.lower_flux / self.grid.lz,
            places=12,
        )
        self.assertEqual(int(final.step), steps)
        self.assertEqual(float(final.time), dt * steps)

    def test_fast_rk3_projects_and_advances_surface_scalar(self) -> None:
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        velocity = StaggeredVelocity(
            jnp.full(shape, 10.0),
            jnp.zeros(shape),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        scalar = PassiveScalar(lower_flux=1.0e-3)
        poisson = build_pressure_poisson(self.grid, backend="fft")
        boundaries = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP))
        step = build_atmospheric_step(
            self.grid,
            boundaries,
            poisson,
            FlowModel(),
            scalar,
            scheme="fast-rk3",
        )
        final = build_atmospheric_run(step)(
            initial_atmospheric_solution(
                self.grid,
                velocity,
                jnp.zeros(shape),
                dtype="float64",
            ),
            0.1,
            3,
        )
        self.assertLess(
            float(jnp.max(jnp.abs(divergence(final.velocity, self.grid)))),
            1.0e-10,
        )
        self.assertAlmostEqual(
            float(jnp.mean(final.scalar)),
            0.3 * scalar.lower_flux / self.grid.lz,
            places=12,
        )
        self.assertEqual(int(final.step), 3)
        self.assertAlmostEqual(float(final.time), 0.3, places=12)

    def test_full_rk3_and_adaptive_driver_hit_the_target_time(self) -> None:
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        velocity = StaggeredVelocity(
            jnp.full(shape, 10.0),
            jnp.zeros(shape),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        step = build_atmospheric_step(
            self.grid,
            Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP)),
            build_pressure_poisson(self.grid, backend="fft"),
            FlowModel(),
            PassiveScalar(),
            scheme="rk3",
        )
        run = build_adaptive_atmospheric_run(
            step,
            self.grid,
            cfl_ceiling=0.8,
            maximum_dt=0.1,
        )
        final = run(
            initial_atmospheric_solution(self.grid, velocity, dtype="float64"),
            0.03,
            8,
        )
        self.assertAlmostEqual(float(final.time), 0.03, places=12)
        self.assertEqual(int(final.step), 2)
        self.assertLess(
            float(jnp.max(jnp.abs(divergence(final.velocity, self.grid)))),
            1.0e-10,
        )


if __name__ == "__main__":
    unittest.main()
