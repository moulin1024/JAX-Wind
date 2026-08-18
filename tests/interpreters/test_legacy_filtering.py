from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    Accepted,
    AddressableField,
    Cell,
    DistributionSpec,
    EqualVerticalPartition,
    MeshAxis,
    MeshTopology,
    PotentialTemperaturePerturbation,
    Projected,
    UniformGrid,
    VerticalBoundary,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from jaxwind._jax.discretization import (  # noqa: E402
    VerticalFaceField,
    build_discretization,
)
from jaxwind.operators import VelocityVector  # noqa: E402
from jaxwind.physics import BoussinesqFields  # noqa: E402


class LegacyFilteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = UniformGrid(
            10,
            10,
            2,
            2.0 * jnp.pi,
            2.0 * jnp.pi,
            2.0,
        )
        self.decomposition = EqualVerticalPartition(
            self.grid,
            MeshTopology((MeshAxis("z", 1),)),
            DistributionSpec.vertical(),
        )
        self.algebra = build_discretization(self.decomposition)

    def velocity_with_x_mode(self, mode: int) -> VelocityVector:
        grid = self.grid
        x = 2.0 * jnp.pi * jnp.arange(grid.nx, dtype=jnp.float64) / grid.nx
        wave = jnp.cos(mode * x)[None, None, :]
        u = jnp.broadcast_to(wave, (grid.nz, grid.ny, grid.nx))[None]
        zero = jnp.zeros_like(u)
        return VelocityVector(
            AddressableField(
                XVelocity,
                Cell,
                self.decomposition.regions(Cell),
                Projected,
                u,
            ),
            AddressableField(
                YVelocity,
                Cell,
                self.decomposition.regions(Cell),
                Projected,
                zero,
            ),
            VerticalFaceField(
                AddressableField(
                    VerticalVelocity,
                    ZFace,
                    self.decomposition.regions(ZFace),
                    Projected,
                    zero,
                ),
                jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
            ),
        )

    def test_projection_does_not_apply_the_legacy_state_cutoff(self) -> None:
        unresolved = self.velocity_with_x_mode(3)
        prepared, _ = self.algebra.prepare_projection(
            unresolved,
            VerticalBoundary(0.0, 0.0),
        )
        self.assertGreater(
            float(jnp.max(jnp.abs(prepared.x_spectrum[..., 3]))),
            1.0,
        )

    def test_pre_rhs_filter_cuts_velocity_and_transported_scalar(self) -> None:
        unresolved = self.velocity_with_x_mode(3)
        scalar = AddressableField(
            PotentialTemperaturePerturbation,
            Cell,
            self.decomposition.regions(Cell),
            Accepted,
            unresolved.x.payload,
        )
        filtered = self.algebra.legacy_fortran_filter_fields(
            BoussinesqFields(unresolved, scalar)
        )
        self.assertLess(
            float(jnp.max(jnp.abs(filtered.velocity.x.payload))),
            3.0e-12,
        )
        self.assertLess(
            float(jnp.max(jnp.abs(filtered.potential_temperature.payload))),
            3.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
