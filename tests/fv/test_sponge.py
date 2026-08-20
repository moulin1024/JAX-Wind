from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    PLANE_MEAN,
    REST,
    FlowModel,
    StaggeredVelocity,
    build_tendency,
    monin_obukhov_boundaries,
    rayleigh_sponge_tendency,
)


def uniform_velocity(grid: UniformGrid, x: float, y: float, z: float) -> StaggeredVelocity:
    cells = (grid.nz, grid.ny, grid.nx)
    return StaggeredVelocity(
        jnp.full(cells, x),
        jnp.full(cells, y),
        jnp.full((grid.nz + 1, grid.ny, grid.nx), z),
    )


class SpongeConfigurationTest(unittest.TestCase):
    grid = UniformGrid(4, 4, 8, 1.0, 1.0, 1.0)

    def test_rejects_a_start_height_outside_the_domain(self) -> None:
        with self.assertRaises(ValueError):
            rayleigh_sponge_tendency(self.grid, start_height=1.0, timescale=1.0)
        with self.assertRaises(ValueError):
            rayleigh_sponge_tendency(self.grid, start_height=-0.1, timescale=1.0)

    def test_rejects_a_non_positive_timescale(self) -> None:
        with self.assertRaises(ValueError):
            rayleigh_sponge_tendency(self.grid, start_height=0.5, timescale=0.0)

    def test_rejects_an_unknown_target(self) -> None:
        with self.assertRaises(ValueError):
            rayleigh_sponge_tendency(
                self.grid, start_height=0.5, timescale=1.0, target="geostrophic"
            )


class SpongeTendencyTest(unittest.TestCase):
    grid = UniformGrid(4, 4, 8, 1.0, 1.0, 1.0)

    def test_leaves_the_flow_alone_below_the_start_height(self) -> None:
        """No damping strictly below the sponge -- z/H in (0, 0.5) here."""
        tendency = rayleigh_sponge_tendency(
            self.grid, start_height=0.5, timescale=1.0, target=REST
        )
        velocity = uniform_velocity(self.grid, 3.0, -2.0, 1.0)
        result = tendency(velocity, jnp.asarray(0.0))
        below = self.grid.nz // 2
        self.assertLess(float(jnp.max(jnp.abs(result.x[:below]))), 1.0e-12)
        self.assertLess(float(jnp.max(jnp.abs(result.y[:below]))), 1.0e-12)

    def test_strengthens_monotonically_toward_the_lid(self) -> None:
        tendency = rayleigh_sponge_tendency(
            self.grid, start_height=0.0, timescale=1.0, power=2.0, target=REST
        )
        velocity = uniform_velocity(self.grid, 1.0, 0.0, 0.0)
        result = tendency(velocity, jnp.asarray(0.0))
        magnitude = jnp.abs(result.x[:, 0, 0])
        self.assertTrue(bool(jnp.all(jnp.diff(magnitude) >= -1.0e-12)))

    def test_rest_target_pulls_uniform_flow_toward_zero(self) -> None:
        tendency = rayleigh_sponge_tendency(
            self.grid, start_height=0.0, timescale=2.0, power=1.0, target=REST
        )
        velocity = uniform_velocity(self.grid, 4.0, -4.0, 8.0)
        result = tendency(velocity, jnp.asarray(0.0))
        # u and v damp with a sign opposite the flow, bounded by 1 / timescale.
        self.assertTrue(-2.0 < float(result.x[-1, 0, 0]) < 0.0)
        self.assertTrue(0.0 < float(result.y[-1, 0, 0]) < 2.0)
        # w's top face sits exactly at the lid, where the ramp is exactly one.
        self.assertAlmostEqual(float(result.z[-1, 0, 0]), -8.0 / 2.0, places=10)

    def test_plane_mean_target_leaves_the_mean_tendency_at_zero(self) -> None:
        """Removing perturbations must not itself pump in (or drain) momentum."""
        tendency = rayleigh_sponge_tendency(
            self.grid, start_height=0.0, timescale=1.0, target=PLANE_MEAN
        )
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        cells = (self.grid.nz, self.grid.ny, self.grid.nx)
        velocity = StaggeredVelocity(
            jax.random.normal(keys[0], cells),
            jax.random.normal(keys[1], cells),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        result = tendency(velocity, jnp.asarray(0.0))
        self.assertLess(
            float(jnp.max(jnp.abs(jnp.mean(result.x, axis=(1, 2))))), 1.0e-12
        )
        self.assertLess(
            float(jnp.max(jnp.abs(jnp.mean(result.y, axis=(1, 2))))), 1.0e-12
        )

    def test_plane_mean_target_still_damps_a_horizontal_perturbation(self) -> None:
        tendency = rayleigh_sponge_tendency(
            self.grid, start_height=0.0, timescale=1.0, target=PLANE_MEAN
        )
        cells = (self.grid.nz, self.grid.ny, self.grid.nx)
        perturbation = jnp.zeros(cells).at[:, 0, 0].set(1.0)
        velocity = StaggeredVelocity(
            perturbation,
            jnp.zeros(cells),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        result = tendency(velocity, jnp.asarray(0.0))
        self.assertLess(float(result.x[-1, 0, 0]), 0.0)

    def test_w_is_always_relaxed_toward_zero(self) -> None:
        for target in (PLANE_MEAN, REST):
            tendency = rayleigh_sponge_tendency(
                self.grid, start_height=0.0, timescale=1.0, target=target
            )
            velocity = uniform_velocity(self.grid, 0.0, 0.0, 5.0)
            result = tendency(velocity, jnp.asarray(0.0))
            self.assertLess(float(result.z[-1, 0, 0]), 0.0)

    def test_repeated_application_decays_a_perturbation_in_the_sponge(self) -> None:
        """Euler-stepping the isolated tendency must behave like a decaying mode."""
        grid = UniformGrid(2, 2, 16, 1.0, 1.0, 1.0)
        tendency = rayleigh_sponge_tendency(
            grid, start_height=0.0, timescale=0.5, power=1.0, target=REST
        )
        velocity = uniform_velocity(grid, 1.0, 0.0, 0.0)
        dt = 1.0e-3
        for _ in range(2000):
            rate = tendency(velocity, jnp.asarray(0.0))
            velocity = StaggeredVelocity(
                velocity.x + dt * rate.x,
                velocity.y + dt * rate.y,
                velocity.z + dt * rate.z,
            )
        top_speed = float(velocity.x[-1, 0, 0])
        self.assertLess(top_speed, 0.05)
        self.assertGreater(top_speed, 0.0)


class SpongeInFlowModelTest(unittest.TestCase):
    def test_the_forcing_hook_adds_the_sponge_tendency(self) -> None:
        grid = UniformGrid(4, 4, 8, 1.0, 1.0, 1.0)
        boundaries = monin_obukhov_boundaries()
        sponge = rayleigh_sponge_tendency(
            grid, start_height=0.5, timescale=1.0, target=REST
        )
        plain = build_tendency(grid, boundaries, FlowModel())
        sponged = build_tendency(grid, boundaries, FlowModel(forcing=sponge))
        velocity = uniform_velocity(grid, 2.0, 0.0, 0.0)
        difference = sponged(velocity, jnp.asarray(0.0)).x - plain(
            velocity, jnp.asarray(0.0)
        ).x
        expected = sponge(velocity, jnp.asarray(0.0)).x
        self.assertLess(float(jnp.max(jnp.abs(difference - expected))), 1.0e-12)
        self.assertLess(float(jnp.max(jnp.abs(difference[: grid.nz // 2]))), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
