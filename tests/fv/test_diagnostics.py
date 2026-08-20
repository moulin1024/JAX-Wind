from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    CELL_CENTRE,
    FREE_SLIP,
    AnisotropicMinimumDissipation,
    Boundaries,
    MoninObukhovWall,
    PassiveScalar,
    StaggeredVelocity,
    Wall,
    atmospheric_history_diagnostics,
    atmospheric_profile_diagnostics,
)


class AtmosphericDiagnosticsTest(unittest.TestCase):
    grid = UniformGrid(6, 4, 5, 600.0, 400.0, 250.0)
    boundaries = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP))
    wall = MoninObukhovWall(0.1, sampling=CELL_CENTRE)
    subfilter = AnisotropicMinimumDissipation()

    def test_fixed_point_has_only_the_prescribed_boundary_fluxes(self) -> None:
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        velocity = StaggeredVelocity(
            jnp.full(shape, 10.0),
            jnp.zeros(shape),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        scalar_model = PassiveScalar(lower_flux=1.0e-3)
        _fields, profiles = atmospheric_profile_diagnostics(
            velocity,
            jnp.zeros(shape),
            jnp.zeros(shape),
            self.grid,
            self.boundaries,
            self.wall,
            scalar_model,
            self.subfilter,
        )

        self.assertTrue(bool(jnp.all(profiles["momentum_diffusivity"] == 0.0)))
        self.assertTrue(bool(jnp.all(profiles["scalar_diffusivity"] == 0.0)))
        self.assertTrue(
            bool(jnp.all(profiles["resolved_tke_sgs_transfer"] == 0.0))
        )
        self.assertAlmostEqual(
            float(profiles["sgs_wc"][0]), 0.5 * scalar_model.lower_flux
        )
        self.assertTrue(bool(jnp.all(profiles["sgs_wc"][1:] == 0.0)))
        self.assertLess(float(profiles["sgs_uw"][0]), 0.0)

        history = atmospheric_history_diagnostics(
            velocity,
            self.grid,
            self.wall,
            coriolis=1.0e-4,
            geostrophic_u=10.0,
            geostrophic_v=0.0,
        )
        self.assertEqual(float(history["integrated_total_tke_m3_s2"]), 0.0)
        self.assertEqual(float(history["momentum_stationarity_cu"]), 0.0)
        self.assertTrue(bool(jnp.isnan(history["momentum_stationarity_cv"])))

    def test_amd_transfer_is_dissipative_and_diffusivity_uses_prandtl(self) -> None:
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        keys = jax.random.split(jax.random.PRNGKey(14), 5)
        velocity = StaggeredVelocity(
            jax.random.normal(keys[0], shape),
            jax.random.normal(keys[1], shape),
            jnp.concatenate(
                (
                    jnp.zeros((1, self.grid.ny, self.grid.nx)),
                    jax.random.normal(
                        keys[2],
                        (self.grid.nz - 1, self.grid.ny, self.grid.nx),
                    ),
                    jnp.zeros((1, self.grid.ny, self.grid.nx)),
                )
            ),
        )
        scalar_model = PassiveScalar(turbulent_prandtl=0.7)
        _fields, profiles = atmospheric_profile_diagnostics(
            velocity,
            jax.random.normal(keys[3], shape),
            jax.random.normal(keys[4], shape),
            self.grid,
            self.boundaries,
            self.wall,
            scalar_model,
            self.subfilter,
        )

        self.assertTrue(
            bool(jnp.all(profiles["resolved_tke_sgs_transfer"] <= 0.0))
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    profiles["scalar_diffusivity"],
                    profiles["momentum_diffusivity"]
                    / scalar_model.turbulent_prandtl,
                )
            )
        )
        for values in profiles.values():
            self.assertEqual(values.shape, (self.grid.nz,))
            self.assertTrue(bool(jnp.all(jnp.isfinite(values))))


if __name__ == "__main__":
    unittest.main()
