from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import PassiveScalar, StaggeredVelocity, scalar_tendency


class PassiveScalarTest(unittest.TestCase):
    grid = UniformGrid(8, 6, 5, 2.0, 1.5, 1.0)
    shape = (grid.nz, grid.ny, grid.nx)

    def test_constant_scalar_is_preserved(self) -> None:
        velocity = StaggeredVelocity(
            jnp.ones(self.shape),
            jnp.zeros(self.shape),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        tendency = scalar_tendency(
            jnp.full(self.shape, 3.0),
            velocity,
            self.grid,
            PassiveScalar(diffusivity=0.1),
        )
        self.assertEqual(float(jnp.max(jnp.abs(tendency))), 0.0)

    def test_surface_flux_changes_the_volume_mean_at_the_exact_rate(self) -> None:
        flux = 1.0e-3
        velocity = StaggeredVelocity(
            jnp.zeros(self.shape),
            jnp.zeros(self.shape),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        tendency = scalar_tendency(
            jnp.zeros(self.shape),
            velocity,
            self.grid,
            PassiveScalar(lower_flux=flux),
        )
        self.assertAlmostEqual(float(jnp.mean(tendency)), flux / self.grid.lz)

    def test_periodic_transport_conserves_the_scalar_integral(self) -> None:
        keys = jax.random.split(jax.random.PRNGKey(4), 2)
        scalar = jax.random.normal(keys[0], self.shape)
        velocity = StaggeredVelocity(
            jax.random.normal(keys[1], self.shape),
            jnp.zeros(self.shape),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        tendency = scalar_tendency(
            scalar,
            velocity,
            self.grid,
            PassiveScalar(),
        )
        self.assertLess(abs(float(jnp.sum(tendency))), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
