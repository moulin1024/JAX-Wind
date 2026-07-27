from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    Cell,
    Field,
    GlobalTestRegion,
    Projected,
    ScaleSystem,
    UniformGrid,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from jaxwind.interpreters.jax_reference import JaxReferenceProjection  # noqa: E402
from jaxwind.operators import VelocityVector  # noqa: E402
from jaxwind.physics import (  # noqa: E402
    ConservativeAdvection,
    CoriolisGeostrophic,
    FilteredNeutralLogWall,
    KinematicPressureGradient,
    NeutralLogWall,
    StaticSmagorinsky,
)


def velocity(grid, u, v, w) -> VelocityVector:
    return VelocityVector(
        Field(
            XVelocity,
            Cell,
            GlobalTestRegion(grid, Cell),
            Projected,
            u,
        ),
        Field(
            YVelocity,
            Cell,
            GlobalTestRegion(grid, Cell),
            Projected,
            v,
        ),
        Field(
            VerticalVelocity,
            ZFace,
            GlobalTestRegion(grid, ZFace),
            Projected,
            w,
        ),
    )


class DryFlowScalingNaturalityTests(unittest.TestCase):
    def test_every_dry_tendency_commutes_with_mechanical_scaling(self) -> None:
        grid = UniformGrid(4, 4, 4, 400.0, 400.0, 200.0)
        scales = ScaleSystem(100.0, 10.0)
        execution_grid = scales.to_execution_grid(grid)
        z = jnp.arange(grid.nz, dtype=jnp.float64)[:, None, None]
        zf = jnp.arange(grid.nz + 1, dtype=jnp.float64)[:, None, None]
        y = 2.0 * jnp.pi * jnp.arange(grid.ny)[None, :, None] / grid.ny
        x = 2.0 * jnp.pi * jnp.arange(grid.nx)[None, None, :] / grid.nx
        u = 7.5 + 0.2 * jnp.sin(x) + 0.03 * z + 0.0 * y
        v = -0.4 + 0.15 * jnp.cos(y) + 0.0 * x + 0.0 * z
        w = 0.1 * jnp.sin(x) * jnp.cos(y) * jnp.sin(jnp.pi * zf / grid.nz)
        reference = JaxReferenceProjection()
        physical_context = reference.dry_flow_context(velocity(grid, u, v, w))
        execution_context = reference.dry_flow_context(
            velocity(
                execution_grid,
                scales.to_execution_velocity(u),
                scales.to_execution_velocity(v),
                scales.to_execution_velocity(w),
            )
        )
        cases = (
            (
                reference.advection_tendency,
                ConservativeAdvection(),
                ConservativeAdvection(),
            ),
            (
                reference.pressure_gradient_tendency,
                KinematicPressureGradient(0.001, -0.002),
                KinematicPressureGradient(
                    scales.to_execution_acceleration(0.001),
                    scales.to_execution_acceleration(-0.002),
                ),
            ),
            (
                reference.wall_stress_tendency,
                NeutralLogWall(0.1),
                NeutralLogWall(scales.to_execution_length(0.1)),
            ),
            (
                reference.wall_stress_tendency,
                FilteredNeutralLogWall(
                    0.1,
                    filter_grid_ratio=1.0,
                    test_filter_ratio=1.0,
                ),
                FilteredNeutralLogWall(
                    scales.to_execution_length(0.1),
                    filter_grid_ratio=1.0,
                    test_filter_ratio=1.0,
                ),
            ),
            (
                reference.sgs_tendency,
                StaticSmagorinsky(0.16),
                StaticSmagorinsky(0.16),
            ),
            (
                reference.coriolis_geostrophic_tendency,
                CoriolisGeostrophic(1.0e-4, 8.0, 0.0),
                CoriolisGeostrophic(
                    scales.to_execution_inverse_time(1.0e-4),
                    scales.to_execution_velocity(8.0),
                    0.0,
                ),
            ),
        )
        for term, physical_config, execution_config in cases:
            with self.subTest(term=term.__name__):
                expected = term(physical_context, physical_config)
                actual = term(execution_context, execution_config)
                for expected_component, actual_component in zip(
                    (expected.x, expected.y, expected.z),
                    (actual.x, actual.y, actual.z),
                    strict=True,
                ):
                    recovered = scales.from_execution_acceleration(
                        actual_component.payload
                    )
                    self.assertLess(
                        float(jnp.max(jnp.abs(recovered - expected_component.payload))),
                        2.0e-14,
                    )


if __name__ == "__main__":
    unittest.main()
