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


class ThreeHalvesPaddingTests(unittest.TestCase):
    def test_high_resolved_mode_is_retained_without_quadratic_alias(self) -> None:
        grid = UniformGrid(8, 8, 2, 2.0 * jnp.pi, 2.0 * jnp.pi, 2.0)
        decomposition = EqualVerticalPartition(
            grid,
            MeshTopology((MeshAxis("z", 1),)),
            DistributionSpec.vertical(),
        )
        x = 2.0 * jnp.pi * jnp.arange(grid.nx, dtype=jnp.float64) / grid.nx
        high_mode = jnp.cos(3.0 * x)[None, None, :]
        u = jnp.broadcast_to(high_mode, (grid.nz, grid.ny, grid.nx))[None]
        v = jnp.zeros_like(u)
        w = jnp.zeros_like(u)
        velocity = VelocityVector(
            AddressableField(
                XVelocity,
                Cell,
                decomposition.regions(Cell),
                Projected,
                u,
            ),
            AddressableField(
                YVelocity,
                Cell,
                decomposition.regions(Cell),
                Projected,
                v,
            ),
            VerticalFaceField(
                AddressableField(
                    VerticalVelocity,
                    ZFace,
                    decomposition.regions(ZFace),
                    Projected,
                    w,
                ),
                jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
            ),
        )
        algebra = build_discretization(decomposition)
        accepted_bandwidth = algebra.enforce_normal_boundary(
            velocity,
            VerticalBoundary(0.0, 0.0),
        )
        self.assertLess(
            float(jnp.max(jnp.abs(accepted_bandwidth.x.payload - u))),
            2.0e-12,
        )

        context = algebra.dry_flow_context(accepted_bandwidth)
        tendency = algebra.advection_tendency(context, ConservativeAdvection())
        self.assertLess(float(jnp.max(jnp.abs(tendency.x.payload))), 2.0e-12)
        self.assertLess(float(jnp.max(jnp.abs(tendency.y.payload))), 2.0e-12)
        self.assertLess(
            float(jnp.max(jnp.abs(tendency.z.owned.payload))),
            2.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
