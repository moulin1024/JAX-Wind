from __future__ import annotations

import math
import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    FREE_SLIP,
    AnisotropicMinimumDissipation,
    Boundaries,
    FlowModel,
    StaggeredVelocity,
    Wall,
    build_pressure_poisson,
    build_run,
    build_step,
    build_tendency,
    divergence,
    eddy_viscosity,
    initial_solution,
    project,
    stress_divergence,
    subfilter_tendency,
)


WALLS = Boundaries()
FREE_SLIP_WALLS = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP))
MODEL = AnisotropicMinimumDissipation()


def turbulent_velocity(grid: UniformGrid, seed: int) -> StaggeredVelocity:
    """A solenoidal, wall-bounded field with structure in all directions."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    cells = (grid.nz, grid.ny, grid.nx)
    candidate = StaggeredVelocity(
        jax.random.normal(keys[0], cells),
        jax.random.normal(keys[1], cells),
        jax.random.normal(keys[2], (grid.nz + 1, grid.ny, grid.nx))
        .at[0]
        .set(0.0)
        .at[-1]
        .set(0.0),
    )
    poisson = build_pressure_poisson(grid, backend="cg")
    projected, _ = project(candidate, poisson, 0.1)
    return projected


class EddyViscosityTest(unittest.TestCase):
    grid = UniformGrid(10, 8, 6, 1.0, 0.8, 0.6)

    def test_the_eddy_viscosity_is_never_negative(self) -> None:
        viscosity = eddy_viscosity(
            turbulent_velocity(self.grid, 1),
            self.grid,
            WALLS,
            MODEL,
        )
        self.assertGreaterEqual(float(jnp.min(viscosity)), 0.0)
        self.assertGreater(float(jnp.max(viscosity)), 0.0)

    def test_a_uniform_flow_needs_no_model(self) -> None:
        cells = (self.grid.nz, self.grid.ny, self.grid.nx)
        uniform = StaggeredVelocity(
            jnp.full(cells, 4.0),
            jnp.full(cells, -2.0),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        viscosity = eddy_viscosity(uniform, self.grid, FREE_SLIP_WALLS, MODEL)
        self.assertLess(float(jnp.max(viscosity)), 1.0e-14)

    def test_laminar_shear_produces_no_eddy_viscosity(self) -> None:
        """The property that lets AMD run without a wall damping function."""
        grid = UniformGrid(4, 4, 16, 1.0, 1.0, 1.0)
        height = (jnp.arange(grid.nz) + 0.5) * grid.dz
        profile = height * (grid.lz - height)
        cells = (grid.nz, grid.ny, grid.nx)
        shear = StaggeredVelocity(
            jnp.broadcast_to(profile[:, None, None], cells),
            jnp.zeros(cells),
            jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
        )
        viscosity = eddy_viscosity(shear, grid, WALLS, MODEL)
        self.assertLess(float(jnp.max(viscosity)), 1.0e-14)

    def test_the_model_is_indifferent_to_a_uniform_translation(self) -> None:
        velocity = turbulent_velocity(self.grid, 2)
        shifted = StaggeredVelocity(
            velocity.x + 3.0,
            velocity.y - 1.0,
            velocity.z,
        )
        first = eddy_viscosity(velocity, self.grid, FREE_SLIP_WALLS, MODEL)
        second = eddy_viscosity(shifted, self.grid, FREE_SLIP_WALLS, MODEL)
        self.assertLess(
            float(jnp.max(jnp.abs(first - second))),
            1.0e-12 * float(jnp.max(first)),
        )

    def test_the_viscosity_scales_with_the_squared_filter_width(self) -> None:
        velocity = turbulent_velocity(self.grid, 3)
        base = eddy_viscosity(velocity, self.grid, WALLS, MODEL)
        wider = eddy_viscosity(
            velocity,
            self.grid,
            WALLS,
            AnisotropicMinimumDissipation(4.0 / 3.0),
        )
        self.assertLess(
            float(jnp.max(jnp.abs(wider - 4.0 * base))),
            1.0e-12 * float(jnp.max(base)),
        )

    def test_rejects_a_non_positive_constant(self) -> None:
        with self.assertRaises(ValueError):
            AnisotropicMinimumDissipation(0.0)


class SubfilterStressTest(unittest.TestCase):
    grid = UniformGrid(10, 8, 6, 1.0, 0.8, 0.6)

    def test_the_stress_divergence_conserves_periodic_momentum(self) -> None:
        tendency, _ = subfilter_tendency(
            turbulent_velocity(self.grid, 4),
            self.grid,
            WALLS,
            MODEL,
        )
        for component in (tendency.x, tendency.y):
            self.assertLess(
                float(jnp.abs(jnp.sum(component))),
                1.0e-12 * float(jnp.sum(jnp.abs(component))),
            )

    def test_the_model_removes_energy(self) -> None:
        """A minimum-dissipation model must still dissipate."""
        velocity = turbulent_velocity(self.grid, 5)
        tendency, viscosity = subfilter_tendency(velocity, self.grid, WALLS, MODEL)
        production = (
            jnp.sum(velocity.x * tendency.x)
            + jnp.sum(velocity.y * tendency.y)
            + jnp.sum(velocity.z[1:-1] * tendency.z[1:-1])
        )
        self.assertGreater(float(jnp.max(viscosity)), 0.0)
        self.assertLess(float(production), 0.0)

    def test_a_uniform_eddy_viscosity_reduces_to_the_laplacian(self) -> None:
        """With constant nu the stress divergence is the viscous term."""
        from jaxwind.fv import diffusion

        velocity = turbulent_velocity(self.grid, 6)
        constant = jnp.full((self.grid.nz, self.grid.ny, self.grid.nx), 0.3)
        stress = stress_divergence(velocity, constant, self.grid, FREE_SLIP_WALLS)
        viscous = diffusion(velocity, self.grid, FREE_SLIP_WALLS, 0.3)
        scale = float(jnp.max(jnp.abs(viscous.x)))
        self.assertLess(float(jnp.max(jnp.abs(stress.x - viscous.x))), 1.0e-10 * scale)
        self.assertLess(float(jnp.max(jnp.abs(stress.y - viscous.y))), 1.0e-10 * scale)

    def test_the_walls_carry_no_subfilter_stress(self) -> None:
        tendency, _ = subfilter_tendency(
            turbulent_velocity(self.grid, 7),
            self.grid,
            WALLS,
            MODEL,
        )
        self.assertLess(float(jnp.max(jnp.abs(tendency.z[0]))), 1.0e-15)
        self.assertLess(float(jnp.max(jnp.abs(tendency.z[-1]))), 1.0e-15)


class LargeEddyRunTest(unittest.TestCase):
    grid = UniformGrid(8, 8, 8, 1.0, 1.0, 1.0)

    def test_the_closure_reaches_the_time_stepper(self) -> None:
        velocity = turbulent_velocity(self.grid, 8)
        direct = build_tendency(self.grid, WALLS, FlowModel(viscosity=0.01))
        large_eddy = build_tendency(
            self.grid,
            WALLS,
            FlowModel(viscosity=0.01, subfilter=MODEL),
        )
        difference = float(
            jnp.max(jnp.abs(large_eddy(velocity, 0.0).x - direct(velocity, 0.0).x))
        )
        self.assertGreater(difference, 0.0)

    def test_a_large_eddy_run_stays_solenoidal_and_finite(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="cg")
        model = FlowModel(
            viscosity=0.005,
            body_force=(1.0, 0.0, 0.0),
            subfilter=MODEL,
        )
        step = build_step(self.grid, WALLS, poisson, model, scheme="rk3")
        final = build_run(step)(initial_solution(self.grid), 0.002, 40)
        self.assertTrue(bool(jnp.all(jnp.isfinite(final.velocity.x))))
        self.assertLess(
            float(jnp.max(jnp.abs(divergence(final.velocity, self.grid)))),
            1.0e-9,
        )
        self.assertGreater(float(jnp.mean(final.velocity.x)), 0.0)

    def test_the_model_leaves_laminar_poiseuille_flow_alone(self) -> None:
        """AMD must not thicken a resolved laminar profile."""
        grid = UniformGrid(4, 4, 16, 1.0, 1.0, 1.0)
        poisson = build_pressure_poisson(grid, backend="cg")
        viscosity, forcing, end_time = 0.1, 1.0, 20.0
        model = FlowModel(
            viscosity=viscosity,
            body_force=(forcing, 0.0, 0.0),
            subfilter=MODEL,
        )
        step = build_step(grid, WALLS, poisson, model, scheme="ab2")
        steps = int(math.ceil(end_time / 0.004))
        final = build_run(step)(initial_solution(grid), end_time / steps, steps)
        height = (jnp.arange(grid.nz) + 0.5) * grid.dz
        exact = forcing / (2.0 * viscosity) * height * (grid.lz - height)
        error = float(jnp.max(jnp.abs(final.velocity.x[:, 0, 0] - exact)))
        self.assertLess(error, 1.0e-5 * float(jnp.max(exact)))


if __name__ == "__main__":
    unittest.main()
