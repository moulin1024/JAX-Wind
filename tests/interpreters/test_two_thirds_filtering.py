from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    AddressableField,
    Cell,
    DistributionSpec,
    EqualVerticalPartition,
    MeshAxis,
    MeshTopology,
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
from jaxwind.physics import ConservativeAdvection  # noqa: E402


class TwoThirdsFilteringTests(unittest.TestCase):
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
        self.algebra = build_discretization(
            self.decomposition,
            nonlinear_dealiasing="two_thirds",
        )

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

    def test_state_projection_strictly_keeps_modes_below_one_third(self) -> None:
        retained = self.velocity_with_x_mode(3)
        projected = self.algebra.enforce_normal_boundary(
            retained,
            VerticalBoundary(0.0, 0.0),
        )
        self.assertLess(
            float(jnp.max(jnp.abs(projected.x.payload - retained.x.payload))),
            3.0e-12,
        )

        removed = self.velocity_with_x_mode(4)
        projected = self.algebra.enforce_normal_boundary(
            removed,
            VerticalBoundary(0.0, 0.0),
        )
        self.assertLess(float(jnp.max(jnp.abs(projected.x.payload))), 3.0e-12)

    def test_quadratic_product_is_projected_back_to_retained_band(self) -> None:
        velocity = self.velocity_with_x_mode(3)
        context = self.algebra.dry_flow_context(velocity)
        tendency = self.algebra.advection_tendency(
            context,
            ConservativeAdvection(),
        )
        self.assertLess(float(jnp.max(jnp.abs(tendency.x.payload))), 3.0e-12)
        self.assertLess(float(jnp.max(jnp.abs(tendency.y.payload))), 3.0e-12)
        self.assertLess(
            float(jnp.max(jnp.abs(tendency.z.owned.payload))),
            3.0e-12,
        )

    def test_legacy_cutoff_filters_state_but_not_nonlinear_output(self) -> None:
        algebra = build_discretization(
            self.decomposition,
            nonlinear_dealiasing="legacy_two_thirds",
        )
        retained = self.velocity_with_x_mode(2)
        projected = algebra.enforce_normal_boundary(
            retained,
            VerticalBoundary(0.0, 0.0),
        )
        self.assertLess(
            float(jnp.max(jnp.abs(projected.x.payload - retained.x.payload))),
            3.0e-12,
        )

        removed = self.velocity_with_x_mode(3)
        projected = algebra.enforce_normal_boundary(
            removed,
            VerticalBoundary(0.0, 0.0),
        )
        self.assertLess(float(jnp.max(jnp.abs(projected.x.payload))), 3.0e-12)

        tendency = algebra.advection_tendency(
            algebra.dry_flow_context(retained),
            ConservativeAdvection(),
        )
        self.assertGreater(
            float(jnp.max(jnp.abs(tendency.x.payload))),
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
