from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    Accepted,
    AcceptedClock,
    Cell,
    Field,
    GlobalTestRegion,
    PassiveScalarConcentration,
    Projected,
    UniformGrid,
    VerticalBoundary,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from jaxwind.effects import (  # noqa: E402
    ReferenceCheckpointLayout,
    load_boussinesq_checkpoint,
    save_boussinesq_checkpoint,
)
from jaxwind.integrators import (  # noqa: E402
    AB2Config,
    ConcurrentPrecursorState,
    cold_start_boussinesq,
    step_concurrent_boussinesq_precursor,
    step_boussinesq,
)
from jaxwind.interpreters.jax_reference import (  # noqa: E402
    JaxReferencePressureSolver,
    JaxReferenceProjection,
)
from jaxwind.operators import VelocityVector  # noqa: E402
from jaxwind.physics import (  # noqa: E402
    BoussinesqFields,
    BoussinesqModel,
    BoussinesqVectorField,
    ConservativeAdvection,
    ConservativeScalarAdvection,
    ConcurrentPrecursorEnvironment,
    ConcurrentPrecursorFringe,
    ConcurrentPrecursorLasdAcceptedStepEvent,
    DiagnosticLasdConstants,
    DryFlowModel,
    KinematicPressureGradient,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LasdAcceptedStepEvent,
    LasdClosureMemory,
    NeutralLogWall,
    NoActuatorDisk,
    NoBuoyancy,
    NoFringe,
    NoRotation,
    ScalarFluxBoundary,
    WindTunnelBoussinesqVectorField,
    WindTunnelModel,
)


class LasdReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = UniformGrid(8, 8, 6, 8.0, 8.0, 6.0)
        self.cells = GlobalTestRegion(self.grid, Cell)
        self.faces = GlobalTestRegion(self.grid, ZFace)
        self.algebra = JaxReferenceProjection()
        self.momentum_lasd = LagrangianScaleDependentDynamic(update_interval=2)
        self.scalar_lasd = LagrangianScaleDependentScalarFlux()
        self.model = BoussinesqModel(
            DryFlowModel(
                ConservativeAdvection(),
                KinematicPressureGradient(0.0, 0.0),
                NeutralLogWall(0.01),
                self.momentum_lasd,
                NoRotation(),
            ),
            ConservativeScalarAdvection(),
            self.scalar_lasd,
            NoBuoyancy(),
            scalar_boundary=ScalarFluxBoundary(1.0e-3, 0.0),
        )
        self.fingerprint = (
            self.momentum_lasd.fingerprint + "|" + self.scalar_lasd.fingerprint
        )

    def test_diagnostic_constants_have_stable_fingerprint(self) -> None:
        constants = DiagnosticLasdConstants()
        self.assertIn("ce=", constants.fingerprint)
        self.assertIn("cc=", constants.fingerprint)

    def fields(
        self,
        scalar_offset: float = 0.0,
        scalar_scale: float = 1.0,
    ) -> BoussinesqFields:
        grid = self.grid
        z = (jnp.arange(grid.nz, dtype=jnp.float64) + 0.5)[:, None, None]
        zf = jnp.arange(grid.nz + 1, dtype=jnp.float64)[:, None, None]
        y = 2.0 * jnp.pi * jnp.arange(grid.ny)[None, :, None] / grid.ny
        x = 2.0 * jnp.pi * jnp.arange(grid.nx)[None, None, :] / grid.nx
        u = 2.0 + 0.15 * jnp.sin(x) * jnp.cos(y) + 0.02 * z
        v = -0.2 + 0.11 * jnp.cos(x - y) + 0.01 * z
        w = 0.06 * jnp.sin(x) * jnp.cos(y) * jnp.sin(jnp.pi * zf / grid.nz)
        scalar = scalar_offset + scalar_scale * (0.3 * z + 0.08 * jnp.sin(x - y))
        velocity = VelocityVector(
            Field(XVelocity, Cell, self.cells, Projected, u),
            Field(YVelocity, Cell, self.cells, Projected, v),
            Field(VerticalVelocity, ZFace, self.faces, Projected, w),
        )
        concentration = Field(
            PassiveScalarConcentration,
            Cell,
            self.cells,
            Accepted,
            scalar,
        )
        return BoussinesqFields(velocity, concentration)

    def initialized_fields(
        self,
        scalar_offset: float = 0.0,
        scalar_scale: float = 1.0,
    ) -> BoussinesqFields:
        return self.algebra.initialize_lasd_closure(
            self.fields(scalar_offset, scalar_scale),
            self.model,
        )

    def test_event_accumulates_then_updates_bounded_offset_invariant_coefficients(
        self,
    ) -> None:
        fields = self.initialized_fields()
        skipped, diagnostic = self.algebra.prepare_lasd_closure(
            fields,
            self.model,
            AcceptedClock(0.0, 0),
            0.01,
        )
        self.assertFalse(diagnostic.updated)
        self.assertGreater(
            float(jnp.max(jnp.abs(skipped.closure.momentum.trajectory_x.payload))),
            0.0,
        )

        updated, diagnostic = self.algebra.prepare_lasd_closure(
            skipped,
            self.model,
            AcceptedClock(0.01, 1),
            0.01,
        )
        offset_skipped, _ = self.algebra.prepare_lasd_closure(
            self.initialized_fields(1000.0),
            self.model,
            AcceptedClock(0.0, 0),
            0.01,
        )
        offset_updated, _ = self.algebra.prepare_lasd_closure(
            offset_skipped,
            self.model,
            AcceptedClock(0.01, 1),
            0.01,
        )
        scaled_skipped, _ = self.algebra.prepare_lasd_closure(
            self.initialized_fields(scalar_scale=7.0),
            self.model,
            AcceptedClock(0.0, 0),
            0.01,
        )
        scaled_updated, _ = self.algebra.prepare_lasd_closure(
            scaled_skipped,
            self.model,
            AcceptedClock(0.01, 1),
            0.01,
        )
        self.assertTrue(diagnostic.updated)
        for memory in (updated.closure.momentum, updated.closure.scalar):
            for field in memory.fields():
                self.assertTrue(jnp.all(jnp.isfinite(field.payload)))
        self.assertGreaterEqual(
            float(jnp.min(updated.closure.momentum.coefficient.payload)),
            self.momentum_lasd.minimum_coefficient,
        )
        self.assertLessEqual(
            float(jnp.max(updated.closure.momentum.coefficient.payload)),
            self.momentum_lasd.maximum_coefficient,
        )
        self.assertLess(
            float(
                jnp.max(
                    jnp.abs(
                        updated.closure.scalar.coefficient.payload
                        - offset_updated.closure.scalar.coefficient.payload
                    )
                )
            ),
            2.0e-11,
        )
        self.assertLess(
            float(
                jnp.max(
                    jnp.abs(
                        updated.closure.scalar.coefficient.payload
                        - scaled_updated.closure.scalar.coefficient.payload
                    )
                )
            ),
            2.0e-11,
        )
        for trajectory in (
            updated.closure.momentum.trajectory_x,
            updated.closure.momentum.trajectory_y,
            updated.closure.momentum.trajectory_z,
        ):
            self.assertEqual(float(jnp.max(jnp.abs(trajectory.payload))), 0.0)

    def test_concurrent_event_relaxes_all_lasd_memory_before_update(self) -> None:
        current = self.initialized_fields()
        target, _ = self.algebra.prepare_lasd_closure(
            self.initialized_fields(),
            self.model,
            AcceptedClock(0.0, 0),
            0.1,
        )
        fringe = ConcurrentPrecursorFringe(
            4.0,
            0.4,
            rise_width=1.0,
            fall_width=1.0,
        )
        event = ConcurrentPrecursorLasdAcceptedStepEvent(
            self.algebra,
            self.model,
            0.1,
            fringe,
        )
        environment = ConcurrentPrecursorEnvironment(
            target.velocity,
            target.closure,
        )

        actual, diagnostic = event(
            current,
            AcceptedClock(0.0, 0),
            environment,
        )
        relaxed = self.algebra.relax_lasd_closure(
            current,
            target.closure,
            fringe,
            0.1,
        )
        expected, _ = self.algebra.prepare_lasd_closure(
            relaxed,
            self.model,
            AcceptedClock(0.0, 0),
            0.1,
        )

        self.assertTrue(diagnostic.closure_relaxed)
        for left, right in zip(
            actual.closure.fields(),
            expected.closure.fields(),
            strict=True,
        ):
            self.assertLess(
                float(jnp.max(jnp.abs(left.payload - right.payload))),
                2.0e-12,
            )
        x_index_on_plateau = 5
        self.assertGreater(
            float(
                relaxed.closure.momentum.trajectory_x.payload[
                    ..., x_index_on_plateau
                ].mean()
            ),
            0.0,
        )
        self.assertEqual(
            float(
                jnp.max(
                    jnp.abs(
                        relaxed.closure.momentum.trajectory_x.payload[..., :4]
                    )
                )
            ),
            0.0,
        )

    def test_concurrent_step_applies_velocity_and_lasd_fringe_together(self) -> None:
        config = AB2Config(0.01)
        clock = AcceptedClock(0.0, 0)
        current = self.initialized_fields()
        target, _ = self.algebra.prepare_lasd_closure(
            self.initialized_fields(),
            self.model,
            clock,
            config.dt,
        )
        fringe = ConcurrentPrecursorFringe(
            4.0,
            0.4,
            rise_width=1.0,
            fall_width=1.0,
        )
        base = BoussinesqVectorField(self.algebra, self.model)
        precursor_vector_field = WindTunnelBoussinesqVectorField(
            self.algebra,
            base,
            WindTunnelModel(NoActuatorDisk(), NoFringe()),
        )
        main_vector_field = WindTunnelBoussinesqVectorField(
            self.algebra,
            base,
            WindTunnelModel(NoActuatorDisk(), fringe),
        )
        precursor_event = LasdAcceptedStepEvent(
            self.algebra,
            self.model,
            config.dt,
        )
        main_event = ConcurrentPrecursorLasdAcceptedStepEvent(
            self.algebra,
            self.model,
            config.dt,
            fringe,
        )
        state = ConcurrentPrecursorState(
            cold_start_boussinesq(
                target,
                clock=clock,
                config=config,
            ),
            cold_start_boussinesq(
                current,
                clock=clock,
                config=config,
            ),
        )

        result = step_concurrent_boussinesq_precursor(
            state,
            config=config,
            precursor_vector_field=precursor_vector_field,
            main_vector_field=main_vector_field,
            normal_boundary=lambda _clock, _environment: VerticalBoundary(
                0.0,
                0.0,
            ),
            algebra=self.algebra,
            precursor_pressure_solver=JaxReferencePressureSolver(),
            main_pressure_solver=JaxReferencePressureSolver(),
            precursor_closure_event=precursor_event,
            main_closure_event=main_event,
        )
        expected, _ = main_event(
            current,
            clock,
            ConcurrentPrecursorEnvironment(
                target.velocity,
                target.closure,
            ),
        )

        self.assertTrue(
            result.diagnostic.main.vector_field.concurrent_fringe_enabled
        )
        self.assertTrue(
            result.diagnostic.main.closure_event.closure_relaxed
        )
        for actual, wanted in zip(
            result.state.main.fields.closure.fields(),
            expected.closure.fields(),
            strict=True,
        ):
            self.assertLess(
                float(jnp.max(jnp.abs(actual.payload - wanted.payload))),
                2.0e-12,
            )

    def test_scalar_wall_flux_is_globally_conservative(self) -> None:
        fields = self.initialized_fields()
        context = self.algebra.boussinesq_context(fields)
        tendency = self.algebra.scalar_sgs_tendency(
            context,
            self.momentum_lasd,
            self.scalar_lasd,
            self.model.scalar_boundary,
        )
        volume_integral = jnp.sum(tendency.payload) * (
            self.grid.dx * self.grid.dy * self.grid.dz
        )
        expected = (
            self.grid.lx
            * self.grid.ly
            * (
                self.model.scalar_boundary.lower_flux
                - self.model.scalar_boundary.upper_flux
            )
        )
        self.assertAlmostEqual(float(volume_integral), expected, places=12)

    def test_stable_stratification_correction_suppresses_scalar_diffusivity(self) -> None:
        fields = self.initialized_fields()
        context = self.algebra.boussinesq_context(fields)
        neutral = self.algebra.lasd_diagnostic_fields(
            context,
            self.momentum_lasd,
            self.scalar_lasd,
        )
        stable_scalar = LagrangianScaleDependentScalarFlux(
            stability_buoyancy_coefficient=0.05,
            stability_beta=30.0,
            stability_power=2.0,
        )
        stable_model = BoussinesqModel(
            self.model.momentum,
            self.model.scalar_advection,
            stable_scalar,
            self.model.buoyancy,
            scalar_boundary=self.model.scalar_boundary,
        )
        stable_fields = self.algebra.initialize_lasd_closure(
            self.fields(),
            stable_model,
        )
        stable = self.algebra.lasd_diagnostic_fields(
            self.algebra.boussinesq_context(stable_fields),
            self.momentum_lasd,
            stable_scalar,
        )
        self.assertTrue(jnp.all(stable.scalar_diffusivity <= neutral.scalar_diffusivity))
        self.assertLess(
            float(jnp.mean(stable.scalar_diffusivity)),
            float(jnp.mean(neutral.scalar_diffusivity)),
        )

    def test_scalar_stability_parameters_are_validated_and_fingerprinted(self) -> None:
        stable = LagrangianScaleDependentScalarFlux(
            stability_buoyancy_coefficient=0.05,
        )
        self.assertIn("stability-buoyancy=", stable.fingerprint)
        with self.assertRaisesRegex(ValueError, "stability power"):
            LagrangianScaleDependentScalarFlux(stability_power=0.0)

    def test_diagnostic_sgs_energy_uses_the_configured_log_wall_shear(self) -> None:
        fields = self.initialized_fields()
        context = self.algebra.boussinesq_context(fields)
        centered = self.algebra.lasd_diagnostic_fields(
            context,
            self.momentum_lasd,
            self.scalar_lasd,
            self.model.scalar_boundary,
        )
        wall_consistent = self.algebra.lasd_diagnostic_fields(
            context,
            self.momentum_lasd,
            self.scalar_lasd,
            self.model.scalar_boundary,
            wall=self.model.momentum.wall,
        )
        self.assertTrue(jnp.all(jnp.isfinite(wall_consistent.scalar_variance)))
        self.assertGreater(
            float(jnp.mean(wall_consistent.sgs_tke[0])),
            float(jnp.mean(centered.sgs_tke[0])),
        )
        self.assertEqual(
            float(jnp.max(jnp.abs(wall_consistent.sgs_tke[1:] - centered.sgs_tke[1:]))),
            0.0,
        )

    def test_checkpoint_restart_preserves_full_history_and_continuation(self) -> None:
        config = AB2Config(0.005)
        event = LasdAcceptedStepEvent(self.algebra, self.model, config.dt)
        vector_field = BoussinesqVectorField(self.algebra, self.model)

        def advance(state, steps):
            for _ in range(steps):
                state = step_boussinesq(
                    state,
                    config=config,
                    environment=None,
                    vector_field=vector_field,
                    normal_boundary=lambda _clock, _environment: VerticalBoundary(
                        0.0,
                        0.0,
                    ),
                    algebra=self.algebra,
                    pressure_solver=JaxReferencePressureSolver(),
                    closure_event=event,
                ).state
            return state

        initial = cold_start_boussinesq(
            self.initialized_fields(),
            clock=AcceptedClock(0.0, 0),
            config=config,
        )
        continuous = advance(initial, 4)
        interrupted = advance(initial, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lasd-checkpoint.npz"
            save_boussinesq_checkpoint(path, interrupted)
            restarted = load_boussinesq_checkpoint(
                path,
                layout=ReferenceCheckpointLayout(self.grid, jnp.asarray),
                config=config,
                closure_fingerprint=self.fingerprint,
            )
            with self.assertRaisesRegex(ValueError, "closure fingerprint"):
                load_boussinesq_checkpoint(
                    path,
                    layout=ReferenceCheckpointLayout(self.grid, jnp.asarray),
                    config=config,
                    closure_fingerprint="different-closure",
                )
        restarted = advance(restarted, 2)
        self.assertIs(
            restarted.fields.potential_temperature.quantity,
            PassiveScalarConcentration,
        )
        self.assertIsInstance(restarted.fields.closure, LasdClosureMemory)
        for actual, expected in zip(
            restarted.fields.closure.fields(),
            continuous.fields.closure.fields(),
            strict=True,
        ):
            self.assertEqual(
                float(jnp.max(jnp.abs(actual.payload - expected.payload))),
                0.0,
            )
        for actual, expected in (
            (
                restarted.fields.potential_temperature,
                continuous.fields.potential_temperature,
            ),
            (restarted.fields.velocity.x, continuous.fields.velocity.x),
            (restarted.fields.velocity.y, continuous.fields.velocity.y),
            (restarted.fields.velocity.z, continuous.fields.velocity.z),
        ):
            self.assertEqual(
                float(jnp.max(jnp.abs(actual.payload - expected.payload))),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
