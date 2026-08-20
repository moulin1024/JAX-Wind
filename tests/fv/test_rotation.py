from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import CoriolisGeostrophic, StaggeredVelocity, coriolis_tendency


class CoriolisTendencyTest(unittest.TestCase):
    grid = UniformGrid(10, 8, 6, 2.0, 1.5, 1.0)

    def test_geostrophic_wind_is_a_fixed_point(self) -> None:
        ug, vg = 10.0, -0.5
        velocity = StaggeredVelocity(
            jnp.full((self.grid.nz, self.grid.ny, self.grid.nx), ug),
            jnp.full((self.grid.nz, self.grid.ny, self.grid.nx), vg),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        tendency = coriolis_tendency(
            velocity,
            CoriolisGeostrophic(1.0e-4, ug, vg, 1.0e-4),
        )
        self.assertEqual(float(jnp.max(jnp.abs(tendency.x))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(tendency.y))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(tendency.z))), 0.0)

    def test_rotation_is_globally_energy_skew(self) -> None:
        keys = jax.random.split(jax.random.PRNGKey(12), 3)
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        ug, vg = 8.0, -1.0
        velocity = StaggeredVelocity(
            ug + jax.random.normal(keys[0], shape),
            vg + jax.random.normal(keys[1], shape),
            jax.random.normal(keys[2], (self.grid.nz + 1, self.grid.ny, self.grid.nx))
            .at[0]
            .set(0.0)
            .at[-1]
            .set(0.0),
        )
        tendency = coriolis_tendency(
            velocity,
            CoriolisGeostrophic(1.0e-4, ug, vg, 0.7e-4),
        )
        work = jnp.sum((velocity.x - ug) * tendency.x)
        work += jnp.sum((velocity.y - vg) * tendency.y)
        work += jnp.sum(velocity.z * tendency.z)
        scale = jnp.sum(jnp.abs((velocity.x - ug) * tendency.x))
        scale += jnp.sum(jnp.abs((velocity.y - vg) * tendency.y))
        scale += jnp.sum(jnp.abs(velocity.z * tendency.z))
        self.assertLess(abs(float(work)), 1.0e-13 * float(scale))

    def test_wall_normal_tendency_respects_impermeability(self) -> None:
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        velocity = StaggeredVelocity(
            jnp.ones(shape),
            jnp.zeros(shape),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        tendency = coriolis_tendency(
            velocity,
            CoriolisGeostrophic(1.0e-4, 0.0, 0.0, 1.0e-4),
        )
        self.assertEqual(float(jnp.max(jnp.abs(tendency.z[0]))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(tendency.z[-1]))), 0.0)


if __name__ == "__main__":
    unittest.main()
