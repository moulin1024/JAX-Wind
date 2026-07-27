from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from wireles.domain import (  # noqa: E402
    Accepted,
    AcceptedClock,
    Cell,
    Field,
    GlobalTestRegion,
    PotentialTemperaturePerturbation,
    Projected,
    UniformGrid,
    VerticalBoundary,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from wireles.integrators import (  # noqa: E402
    AB2Config,
    Evaluation,
    cold_start_boussinesq,
    step_boussinesq,
)
from wireles.effects import (  # noqa: E402
    ReferenceCheckpointLayout,
    load_boussinesq_checkpoint,
    save_boussinesq_checkpoint,
)
from wireles.interpreters.jax_reference import (  # noqa: E402
    JaxReferencePressureSolver,
    JaxReferenceProjection,
)
from wireles.operators import VelocityVector  # noqa: E402
from wireles.physics import (  # noqa: E402
    BoussinesqFields,
    BoussinesqModel,
    BoussinesqVectorField,
    ConservativeAdvection,
    ConservativeScalarAdvection,
    DryFlowModel,
    KinematicPressureGradient,
    LinearBoussinesqBuoyancy,
    NeutralLogWall,
    NoRayleighDamping,
    NoRotation,
    RayleighGeostrophicDamping,
    StaticSmagorinsky,
    StaticSmagorinskyScalarFlux,
)


class BoussinesqReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = UniformGrid(6, 6, 4, 6.0, 6.0, 4.0)
        self.cells = GlobalTestRegion(self.grid, Cell)
        self.faces = GlobalTestRegion(self.grid, ZFace)
        self.algebra = JaxReferenceProjection()

    def fields(self) -> BoussinesqFields:
        z = jnp.arange(self.grid.nz, dtype=jnp.float64)[:, None, None]
        zf = jnp.arange(self.grid.nz + 1, dtype=jnp.float64)[:, None, None]
        y = 2.0 * jnp.pi * jnp.arange(self.grid.ny)[None, :, None] / self.grid.ny
        x = 2.0 * jnp.pi * jnp.arange(self.grid.nx)[None, None, :] / self.grid.nx
        u = 2.0 + 0.2 * jnp.sin(x) + 0.0 * y + 0.03 * z
        v = -0.3 + 0.1 * jnp.cos(y) + 0.0 * x + 0.0 * z
        w = 0.08 * jnp.sin(x) * jnp.cos(y) * jnp.sin(jnp.pi * zf / self.grid.nz)
        theta = 0.4 * z + 0.05 * jnp.sin(x - y)
        velocity = VelocityVector(
            Field(XVelocity, Cell, self.cells, Projected, u),
            Field(YVelocity, Cell, self.cells, Projected, v),
            Field(VerticalVelocity, ZFace, self.faces, Projected, w),
        )
        scalar = Field(
            PotentialTemperaturePerturbation,
            Cell,
            self.cells,
            Accepted,
            theta,
        )
        return BoussinesqFields(velocity, scalar)

    def model(self) -> BoussinesqModel:
        return BoussinesqModel(
            DryFlowModel(
                ConservativeAdvection(),
                KinematicPressureGradient(0.0, 0.0),
                NeutralLogWall(0.01),
                StaticSmagorinsky(0.16),
                NoRotation(),
            ),
            ConservativeScalarAdvection(),
            StaticSmagorinskyScalarFlux(0.4),
            LinearBoussinesqBuoyancy(0.03),
        )

    def test_scalar_conservation_sgs_dissipation_and_buoyancy_location(self) -> None:
        fields = self.fields()
        vector_field = BoussinesqVectorField(self.algebra, self.model())
        evaluation = Evaluation(fields, AcceptedClock(0.0, 0), None)
        contributions = vector_field.evaluate_contributions(evaluation)
        self.assertLess(
            abs(float(jnp.sum(contributions.scalar_advection.payload))),
            2.0e-12,
        )
        scalar_work = jnp.sum(
            fields.potential_temperature.payload * contributions.scalar_sgs.payload
        )
        self.assertLess(float(scalar_work), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(contributions.buoyancy.x.payload))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(contributions.buoyancy.y.payload))), 0.0)
        self.assertEqual(
            float(jnp.max(jnp.abs(contributions.buoyancy.z.payload[0]))), 0.0
        )
        self.assertEqual(
            float(jnp.max(jnp.abs(contributions.buoyancy.z.payload[-1]))), 0.0
        )

    def test_rayleigh_layer_is_local_and_dissipates_geostrophic_relative_energy(
        self,
    ) -> None:
        fields = self.fields()
        context = self.algebra.boussinesq_context(fields)
        identity = self.algebra.rayleigh_damping_tendency(
            context,
            NoRayleighDamping(),
        )
        self.assertEqual(float(jnp.max(jnp.abs(identity.x.payload))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(identity.y.payload))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(identity.z.payload))), 0.0)

        target_u = 2.5
        target_v = -0.1
        damping = self.algebra.rayleigh_damping_tendency(
            context,
            RayleighGeostrophicDamping(2.0, 0.4, target_u, target_v),
        )
        self.assertEqual(float(jnp.max(jnp.abs(damping.x.payload[:2]))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(damping.y.payload[:2]))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(damping.z.payload[:3]))), 0.0)
        relative_work = jnp.sum(
            (fields.velocity.x.payload - target_u) * damping.x.payload
            + (fields.velocity.y.payload - target_v) * damping.y.payload
        ) + jnp.sum(fields.velocity.z.payload * damping.z.payload)
        self.assertLess(float(relative_work), 0.0)

    def test_coupled_ab2_projects_velocity_but_accepts_scalar_unchanged(self) -> None:
        fields = self.fields()
        config = AB2Config(0.01)
        state = cold_start_boussinesq(
            fields,
            clock=AcceptedClock(0.0, 0),
            config=config,
        )
        result = step_boussinesq(
            state,
            config=config,
            environment=None,
            vector_field=BoussinesqVectorField(self.algebra, self.model()),
            normal_boundary=lambda _clock, _environment: VerticalBoundary(0.0, 0.0),
            algebra=self.algebra,
            pressure_solver=JaxReferencePressureSolver(),
        )
        self.assertIs(result.state.fields.potential_temperature.phase, Accepted)
        self.assertLess(
            float(jnp.max(jnp.abs(result.diagnostic.projection.divergence.payload))),
            3.0e-12,
        )
        self.assertTrue(
            jnp.all(jnp.isfinite(result.state.fields.potential_temperature.payload))
        )

    def test_stable_background_buoyancy_is_restoring_for_a_discrete_wave(self) -> None:
        grid = self.grid
        z = (jnp.arange(grid.nz, dtype=jnp.float64) + 0.5)[:, None, None]
        zf = jnp.arange(grid.nz + 1, dtype=jnp.float64)[:, None, None]
        x = 2.0 * jnp.pi * jnp.arange(grid.nx)[None, None, :] / grid.nx
        y = jnp.zeros((1, grid.ny, 1), dtype=jnp.float64)
        vertical_factor = 2.0 * jnp.sin(jnp.pi / (2.0 * grid.nz)) / grid.dz
        horizontal_wavenumber = 2.0 * jnp.pi / grid.lx
        u = (
            vertical_factor
            / horizontal_wavenumber
            * jnp.cos(x)
            * jnp.cos(jnp.pi * z / grid.nz)
            + 0.0 * y
        )
        v = jnp.zeros_like(u)
        w = jnp.sin(x) * jnp.sin(jnp.pi * zf / grid.nz) + 0.0 * y
        theta = 0.2 * z + 0.0 * x + 0.0 * y
        fields = BoussinesqFields(
            VelocityVector(
                Field(XVelocity, Cell, self.cells, Projected, u),
                Field(YVelocity, Cell, self.cells, Projected, v),
                Field(VerticalVelocity, ZFace, self.faces, Projected, w),
            ),
            Field(
                PotentialTemperaturePerturbation,
                Cell,
                self.cells,
                Accepted,
                theta,
            ),
        )
        divergence = self.algebra.velocity_divergence(fields.velocity).payload
        self.assertLess(float(jnp.max(jnp.abs(divergence))), 2.0e-12)
        context = self.algebra.boussinesq_context(fields)
        theta_rate = self.algebra.scalar_advection_tendency(
            context,
            ConservativeScalarAdvection(),
        ).payload
        theta_rate_faces = jnp.concatenate(
            (
                theta_rate[:1],
                0.5 * (theta_rate[:-1] + theta_rate[1:]),
                theta_rate[-1:],
            ),
            axis=0,
        )
        buoyancy_rate = theta_rate_faces - jnp.mean(
            theta_rate_faces,
            axis=(-2, -1),
            keepdims=True,
        )
        restoring_work_rate = jnp.sum(w * buoyancy_rate)
        self.assertLess(float(restoring_work_rate), 0.0)

    def test_coupled_checkpoint_preserves_scalar_and_both_ab2_tendencies(self) -> None:
        config = AB2Config(0.01)
        state = cold_start_boussinesq(
            self.fields(),
            clock=AcceptedClock(0.0, 0),
            config=config,
        )
        advanced = step_boussinesq(
            state,
            config=config,
            environment=None,
            vector_field=BoussinesqVectorField(self.algebra, self.model()),
            normal_boundary=lambda _clock, _environment: VerticalBoundary(0.0, 0.0),
            algebra=self.algebra,
            pressure_solver=JaxReferencePressureSolver(),
        ).state
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_boussinesq_checkpoint(
                path,
                advanced,
                scale_fingerprint="test-scales-v1",
            )
            loaded = load_boussinesq_checkpoint(
                path,
                layout=ReferenceCheckpointLayout(self.grid, jnp.asarray),
                config=config,
                scale_fingerprint="test-scales-v1",
            )
            with self.assertRaisesRegex(ValueError, "scale fingerprint"):
                load_boussinesq_checkpoint(
                    path,
                    layout=ReferenceCheckpointLayout(self.grid, jnp.asarray),
                    config=config,
                    scale_fingerprint="different-scales",
                )
        self.assertEqual(loaded.clock, advanced.clock)
        self.assertEqual(
            float(
                jnp.max(
                    jnp.abs(
                        loaded.fields.potential_temperature.payload
                        - advanced.fields.potential_temperature.payload
                    )
                )
            ),
            0.0,
        )
        for component in ("x", "y", "z"):
            self.assertEqual(
                float(
                    jnp.max(
                        jnp.abs(
                            getattr(loaded.history.value.velocity, component).payload
                            - getattr(
                                advanced.history.value.velocity, component
                            ).payload
                        )
                    )
                ),
                0.0,
            )
        self.assertEqual(
            float(
                jnp.max(
                    jnp.abs(
                        loaded.history.value.potential_temperature.payload
                        - advanced.history.value.potential_temperature.payload
                    )
                )
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
