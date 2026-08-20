from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind._jax.surface import monin_obukhov_surface_transfer
from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    MoninObukhovSurface,
    StaggeredVelocity,
    coupled_surface_exchange,
)


class CoupledSurfaceExchangeTest(unittest.TestCase):
    grid = UniformGrid(8, 6, 5, 400.0, 300.0, 62.5)

    def test_fv_exchange_matches_the_existing_businger_dyer_kernel(self) -> None:
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        velocity = StaggeredVelocity(
            jnp.full(shape, 0.25),
            jnp.full(shape, -0.1),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        scalar = jnp.full(shape, 265.8)
        time = jnp.asarray(7.0 * 3600.0)
        model = MoninObukhovSurface(
            momentum_roughness=0.1,
            scalar_roughness=0.1,
            surface_scalar_initial=265.0,
            surface_scalar_rate=-0.25 / 3600.0,
            x_velocity_offset=8.0,
            buoyancy_coefficient=9.81 / 263.5,
        )
        actual = coupled_surface_exchange(
            velocity,
            scalar,
            time,
            self.grid,
            model,
        )
        expected = monin_obukhov_surface_transfer(
            velocity.x,
            velocity.y,
            scalar,
            time,
            self.grid.dz,
            model.momentum_roughness,
            model.scalar_roughness,
            model.surface_scalar_initial,
            model.surface_scalar_rate,
            model.x_velocity_offset,
            model.y_velocity_offset,
            model.buoyancy_coefficient,
            model.von_karman,
            model.positive_zeta_momentum_slope,
            model.positive_zeta_scalar_slope,
            model.negative_zeta_momentum_coefficient,
            model.negative_zeta_scalar_coefficient,
            model.relaxation,
            model.maximum_abs_zeta,
            bottom=0,
            iterations=model.iterations,
        )

        for actual_value, expected_value in zip(actual, expected[:7], strict=True):
            self.assertTrue(
                bool(jnp.allclose(actual_value, expected_value, rtol=1.0e-12))
            )
        self.assertLess(float(actual.scalar_flux), 0.0)
        self.assertGreater(float(actual.obukhov_length), 0.0)


if __name__ == "__main__":
    unittest.main()
