"""Dry-flow laws evaluated with the independent global test oracle."""

from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    AcceptedClock,
    Cell,
    EvaluationTime,
    Field,
    GlobalTestRegion,
    Projected,
    UniformGrid,
    VerticalBoundary,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from jaxwind.integrators import AB2Config, Evaluation, cold_start, step  # noqa: E402
from tests.support.jax_oracle import (  # noqa: E402
    JaxOraclePressureSolver,
    JaxOracleProjection,
)
from jaxwind.operators import VelocityVector  # noqa: E402
from jaxwind.physics import (  # noqa: E402
    ConservativeAdvection,
    CoriolisGeostrophic,
    DryFlowModel,
    DryFlowVectorField,
    FilteredNeutralLogWall,
    KinematicPressureGradient,
    NeutralLogWall,
    NoRotation,
    StaticSmagorinsky,
)


class DryFlowReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = UniformGrid(6, 6, 4, 6.0, 6.0, 4.0)
        self.cells = GlobalTestRegion(self.grid, Cell)
        self.faces = GlobalTestRegion(self.grid, ZFace)
        self.algebra = JaxOracleProjection()

    def velocity(self, u, v, w) -> VelocityVector:
        return VelocityVector(
            Field(XVelocity, Cell, self.cells, Projected, jnp.asarray(u)),
            Field(YVelocity, Cell, self.cells, Projected, jnp.asarray(v)),
            Field(VerticalVelocity, ZFace, self.faces, Projected, jnp.asarray(w)),
        )

    def test_uniform_flow_terms_and_composition_laws(self) -> None:
        u = jnp.full(self.cells.storage_shape, 4.0)
        v = jnp.full(self.cells.storage_shape, -1.5)
        w = jnp.zeros(self.faces.storage_shape)
        velocity = self.velocity(u, v, w)
        model = DryFlowModel(
            ConservativeAdvection(),
            KinematicPressureGradient(0.002, -0.001),
            NeutralLogWall(0.01),
            StaticSmagorinsky(0.16),
        )
        vector_field = DryFlowVectorField(self.algebra, model)
        evaluation = Evaluation(
            velocity,
            EvaluationTime(0.0, 0, "dry-reference"),
            None,
        )
        contributions = vector_field.evaluate_contributions(evaluation)
        self.assertEqual(
            float(jnp.max(jnp.abs(contributions.advection.x.payload))),
            0.0,
        )
        self.assertEqual(float(jnp.max(jnp.abs(contributions.sgs.x.payload))), 0.0)
        self.assertEqual(
            float(jnp.max(jnp.abs(contributions.pressure_gradient.x.payload - 0.002))),
            0.0,
        )
        self.assertEqual(
            float(jnp.max(jnp.abs(contributions.pressure_gradient.y.payload + 0.001))),
            0.0,
        )
        wall = contributions.wall
        self.assertLess(float(jnp.sum(u * wall.x.payload + v * wall.y.payload)), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(wall.x.payload[1:]))), 0.0)

        forward = self.algebra.combine_tendencies(contributions.values())
        reverse = self.algebra.combine_tendencies(
            tuple(reversed(contributions.values()))
        )
        result = vector_field(evaluation)
        for left, right, total in zip(
            (forward.x, forward.y, forward.z),
            (reverse.x, reverse.y, reverse.z),
            (result.tendency.x, result.tendency.y, result.tendency.z),
            strict=True,
        ):
            self.assertEqual(float(jnp.max(jnp.abs(left.payload - right.payload))), 0.0)
            self.assertEqual(float(jnp.max(jnp.abs(left.payload - total.payload))), 0.0)
        self.assertEqual(result.diagnostic.shared_context_builds, 1)

    def test_conservative_advection_preserves_integrated_horizontal_momentum(
        self,
    ) -> None:
        z = jnp.arange(self.grid.nz, dtype=jnp.float64)[:, None, None]
        zf = jnp.arange(self.grid.nz + 1, dtype=jnp.float64)[:, None, None]
        y = 2.0 * jnp.pi * jnp.arange(self.grid.ny)[None, :, None] / self.grid.ny
        x = 2.0 * jnp.pi * jnp.arange(self.grid.nx)[None, None, :] / self.grid.nx
        u = 2.0 + 0.3 * jnp.sin(x) + 0.1 * jnp.cos(y) + 0.05 * z
        v = -0.4 + 0.2 * jnp.cos(x - y) - 0.03 * z
        w = 0.15 * jnp.sin(x) * jnp.cos(y) * jnp.sin(jnp.pi * zf / self.grid.nz)
        context = self.algebra.dry_flow_context(self.velocity(u, v, w))
        tendency = self.algebra.advection_tendency(context, ConservativeAdvection())
        self.assertLess(abs(float(jnp.sum(tendency.x.payload))), 2.0e-12)
        self.assertLess(abs(float(jnp.sum(tendency.y.payload))), 2.0e-12)
        boundary_faces = jnp.stack(
            (tendency.z.payload[0], tendency.z.payload[-1]),
            axis=0,
        )
        self.assertEqual(float(jnp.max(jnp.abs(boundary_faces))), 0.0)

        sgs = self.algebra.sgs_tendency(context, StaticSmagorinsky(0.16))
        sgs_work = jnp.sum(u * sgs.x.payload + v * sgs.y.payload) + jnp.sum(
            w[1:-1] * sgs.z.payload[1:-1]
        )
        self.assertLess(float(sgs_work), 0.0)

    def test_invalid_wall_resolution_is_rejected_at_interpretation(self) -> None:
        velocity = self.velocity(
            jnp.ones(self.cells.storage_shape),
            jnp.zeros(self.cells.storage_shape),
            jnp.zeros(self.faces.storage_shape),
        )
        context = self.algebra.dry_flow_context(velocity)
        with self.assertRaisesRegex(ValueError, "below the first cell centre"):
            self.algebra.wall_stress_tendency(context, NeutralLogWall(0.5))

    def test_filtered_wall_removes_unresolved_wall_modes(self) -> None:
        x = jnp.arange(self.grid.nx, dtype=jnp.float64)[None, None, :]
        u = 2.0 + 0.2 * x + jnp.zeros(self.cells.storage_shape)
        v = 0.1 + jnp.zeros(self.cells.storage_shape)
        velocity = self.velocity(u, v, jnp.zeros(self.faces.storage_shape))
        context = self.algebra.dry_flow_context(velocity)
        filtered = self.algebra.wall_stress_tendency(
            context,
            FilteredNeutralLogWall(0.01),
        )
        pointwise = self.algebra.wall_stress_tendency(
            context,
            NeutralLogWall(0.01),
        )
        self.assertLess(float(jnp.ptp(filtered.x.payload[0])), 2.0e-15)
        self.assertGreater(float(jnp.ptp(pointwise.x.payload[0])), 1.0e-3)

    def test_coriolis_fixed_point_and_relative_energy_law(self) -> None:
        z = jnp.arange(self.grid.nz, dtype=jnp.float64)[:, None, None]
        y = jnp.arange(self.grid.ny, dtype=jnp.float64)[None, :, None]
        x = jnp.arange(self.grid.nx, dtype=jnp.float64)[None, None, :]
        ug = 8.0
        vg = -1.0
        u = ug + 0.3 * jnp.sin(2.0 * jnp.pi * x / self.grid.nx) + 0.01 * z + 0.0 * y
        v = vg + 0.2 * jnp.cos(2.0 * jnp.pi * y / self.grid.ny) + 0.0 * x + 0.0 * z
        w = jnp.zeros(self.faces.storage_shape)
        context = self.algebra.dry_flow_context(self.velocity(u, v, w))
        config = CoriolisGeostrophic(-1.0e-4, ug, vg)
        tendency = self.algebra.coriolis_geostrophic_tendency(context, config)
        relative_work = (u - ug) * tendency.x.payload + (v - vg) * tendency.y.payload
        self.assertLess(float(jnp.max(jnp.abs(relative_work))), 2.0e-18)
        identity = self.algebra.coriolis_geostrophic_tendency(
            context,
            NoRotation(),
        )
        self.assertEqual(float(jnp.max(jnp.abs(identity.x.payload))), 0.0)

        fixed_context = self.algebra.dry_flow_context(
            self.velocity(
                jnp.full(self.cells.storage_shape, ug),
                jnp.full(self.cells.storage_shape, vg),
                w,
            )
        )
        fixed = self.algebra.coriolis_geostrophic_tendency(fixed_context, config)
        self.assertEqual(float(jnp.max(jnp.abs(fixed.x.payload))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(fixed.y.payload))), 0.0)

    def test_nontraditional_coriolis_is_globally_skew_on_staggered_grid(self) -> None:
        z = jnp.arange(self.grid.nz, dtype=jnp.float64)[:, None, None]
        zf = jnp.arange(self.grid.nz + 1, dtype=jnp.float64)[:, None, None]
        y = 2.0 * jnp.pi * jnp.arange(self.grid.ny)[None, :, None] / self.grid.ny
        x = 2.0 * jnp.pi * jnp.arange(self.grid.nx)[None, None, :] / self.grid.nx
        ug, vg = 8.0, -0.5
        u = ug + 0.2 * jnp.sin(x) + 0.03 * z + 0.0 * y
        v = vg + 0.1 * jnp.cos(y) + 0.0 * x + 0.0 * z
        w = 0.15 * jnp.sin(x) * jnp.cos(y) * jnp.sin(jnp.pi * zf / self.grid.nz)
        context = self.algebra.dry_flow_context(self.velocity(u, v, w))
        tendency = self.algebra.coriolis_geostrophic_tendency(
            context,
            CoriolisGeostrophic(1.0e-4, ug, vg, 1.0e-4),
        )
        relative_work = jnp.sum(
            (u - ug) * tendency.x.payload + (v - vg) * tendency.y.payload
        ) + jnp.sum(w * tendency.z.payload)
        self.assertLess(abs(float(relative_work)), 2.0e-15)
        self.assertEqual(
            float(
                jnp.max(
                    jnp.abs(
                        jnp.stack(
                            (tendency.z.payload[0], tendency.z.payload[-1]),
                            axis=0,
                        )
                    )
                )
            ),
            0.0,
        )

    def test_real_vector_field_runs_through_ab2_and_projection(self) -> None:
        velocity = self.velocity(
            jnp.full(self.cells.storage_shape, 2.0),
            jnp.zeros(self.cells.storage_shape),
            jnp.zeros(self.faces.storage_shape),
        )
        model = DryFlowModel(
            ConservativeAdvection(),
            KinematicPressureGradient(0.01),
            NeutralLogWall(0.01),
            StaticSmagorinsky(0.16),
        )
        config = AB2Config(0.02)
        state = cold_start(
            velocity,
            clock=AcceptedClock(0.0, 0),
            config=config,
        )
        result = step(
            state,
            config=config,
            environment=None,
            vector_field=DryFlowVectorField(self.algebra, model),
            normal_boundary=lambda _clock, _environment: VerticalBoundary(0.0, 0.0),
            algebra=self.algebra,
            pressure_solver=JaxOraclePressureSolver(),
        )
        self.assertEqual(result.state.clock.step, 1)
        self.assertLess(
            float(jnp.max(jnp.abs(result.diagnostic.projection.divergence.payload))),
            2.0e-12,
        )
        self.assertTrue(jnp.all(jnp.isfinite(result.state.velocity.x.payload)))


if __name__ == "__main__":
    unittest.main()
