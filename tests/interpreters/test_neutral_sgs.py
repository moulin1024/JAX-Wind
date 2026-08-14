from __future__ import annotations

from dataclasses import replace
import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    Accepted,
    AcceptedClock,
    AddressableField,
    Cell,
    DistributionSpec,
    EqualVerticalPartition,
    MeshAxis,
    MeshTopology,
    PassiveScalarConcentration,
    Projected,
    UniformGrid,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from jaxwind.integrators import Evaluation  # noqa: E402
from jaxwind._jax.discretization import (  # noqa: E402
    VerticalFaceField,
    build_discretization,
)
from jaxwind._jax.surface import (  # noqa: E402
    monin_obukhov_surface_transfer,
)
from jaxwind.operators import VelocityVector  # noqa: E402
from jaxwind.physics import (  # noqa: E402
    BoussinesqFields,
    BoussinesqModel,
    BoussinesqVectorField,
    ConservativeAdvection,
    ConservativeScalarAdvection,
    CoriolisGeostrophic,
    DryFlowModel,
    FilteredNeutralLogWall,
    KinematicPressureGradient,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LinearBoussinesqBuoyancy,
    MoninObukhovSurfaceTransfer,
    NoBuoyancy,
    NoRotation,
    NoSurfaceTransfer,
    RotationalAdvection,
    ScalarFluxBoundary,
    StaticSmagorinsky,
    StaticSmagorinskyScalarFlux,
)


class FusedNeutralSgsTests(unittest.TestCase):
    def test_surface_transfer_remains_one_jit_lowerable_program(self) -> None:
        self.assertTrue(callable(monin_obukhov_surface_transfer.lower))

    def setUp(self) -> None:
        grid = UniformGrid(8, 8, 4, 8.0, 8.0, 4.0)
        self.decomposition = EqualVerticalPartition(
            grid,
            MeshTopology((MeshAxis("z", 1),)),
            DistributionSpec.vertical(),
        )
        z = jnp.arange(grid.nz, dtype=jnp.float64)[:, None, None]
        zf = jnp.arange(1, grid.nz + 1, dtype=jnp.float64)[:, None, None]
        y = 2.0 * jnp.pi * jnp.arange(grid.ny)[None, :, None] / grid.ny
        x = 2.0 * jnp.pi * jnp.arange(grid.nx)[None, None, :] / grid.nx
        u = 2.0 + 0.2 * jnp.sin(x) + 0.1 * jnp.cos(y) + 0.03 * z
        v = -0.3 + 0.1 * jnp.cos(x - y) - 0.02 * z
        w = 0.08 * jnp.sin(x) * jnp.cos(y) * jnp.sin(jnp.pi * zf / grid.nz)
        shape = (1, grid.nz, grid.ny, grid.nx)
        regions = self.decomposition.regions(Cell)
        velocity = VelocityVector(
            AddressableField(XVelocity, Cell, regions, Projected, u.reshape(shape)),
            AddressableField(YVelocity, Cell, regions, Projected, v.reshape(shape)),
            VerticalFaceField(
                AddressableField(
                    VerticalVelocity,
                    ZFace,
                    self.decomposition.regions(ZFace),
                    Projected,
                    w.reshape(shape),
                ),
                jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
            ),
        )
        self.fields = BoussinesqFields(
            velocity,
            AddressableField(
                PassiveScalarConcentration,
                Cell,
                regions,
                Accepted,
                jnp.zeros(shape, dtype=jnp.float64),
            ),
        )
        theta = 0.4 * z + 0.05 * jnp.sin(x - y)
        self.active_fields = BoussinesqFields(
            velocity,
            AddressableField(
                PassiveScalarConcentration,
                Cell,
                regions,
                Accepted,
                theta.reshape(shape),
            ),
        )

    def assert_fused_matches_contributions(
        self,
        model: BoussinesqModel,
        *,
        fields: BoussinesqFields | None = None,
        frozen_zero_scalar: bool = True,
    ) -> None:
        algebra = build_discretization(
            self.decomposition,
            frozen_zero_scalar=frozen_zero_scalar,
        )
        fields = self.fields if fields is None else fields
        if isinstance(model.momentum.sgs, LagrangianScaleDependentDynamic):
            fields = algebra.initialize_lasd_closure(fields, model)
        evaluation = Evaluation(fields, AcceptedClock(0.0, 0), None)
        contributions = BoussinesqVectorField(
            algebra,
            model,
        ).evaluate_contributions(evaluation)
        expected_velocity = algebra.combine_tendencies(
            contributions.momentum_values()
        )
        expected_scalar = algebra.combine_scalar_tendencies(
            contributions.scalar_values()
        )
        self.assertIsNotNone(algebra.fused_boussinesq_tendency(fields, model))
        actual = BoussinesqVectorField(algebra, model)(evaluation).tendency
        for expected, fused_payload in (
            (expected_velocity.x.payload, actual.velocity.x.payload),
            (expected_velocity.y.payload, actual.velocity.y.payload),
            (expected_velocity.z.owned.payload, actual.velocity.z.owned.payload),
            (expected_scalar.payload, actual.potential_temperature.payload),
        ):
            self.assertLess(
                float(jnp.max(jnp.abs(expected - fused_payload))),
                3.0e-12,
            )

    def test_lasd_fused_rhs_matches_individual_contributions(self) -> None:
        self.assert_fused_matches_contributions(
            BoussinesqModel(
                DryFlowModel(
                    ConservativeAdvection(),
                    KinematicPressureGradient(0.002, -0.001),
                    FilteredNeutralLogWall(0.01),
                    LagrangianScaleDependentDynamic(update_interval=4),
                    NoRotation(),
                ),
                ConservativeScalarAdvection(),
                LagrangianScaleDependentScalarFlux(),
                NoBuoyancy(),
            )
        )

    def test_rotational_lasd_fused_rhs_matches_resolved_contributions(self) -> None:
        self.assert_fused_matches_contributions(
            BoussinesqModel(
                DryFlowModel(
                    RotationalAdvection(),
                    KinematicPressureGradient(0.002, -0.001),
                    FilteredNeutralLogWall(0.01),
                    LagrangianScaleDependentDynamic(update_interval=4),
                    NoRotation(),
                ),
                ConservativeScalarAdvection(),
                LagrangianScaleDependentScalarFlux(),
                NoBuoyancy(),
            )
        )

    def test_rotational_fused_rhs_removes_unresolved_product_alias(self) -> None:
        algebra = build_discretization(
            self.decomposition,
            frozen_zero_scalar=True,
        )
        grid = self.decomposition.grid
        x = 2.0 * jnp.pi * jnp.arange(grid.nx) / grid.nx
        shape = (1, grid.nz, grid.ny, grid.nx)
        u = jnp.broadcast_to(jnp.sin(3.0 * x), shape)
        v = jnp.broadcast_to(jnp.cos(3.0 * x), shape)
        zero_w = jnp.zeros(shape, dtype=u.dtype)
        velocity = VelocityVector(
            replace(self.fields.velocity.x, payload=u),
            replace(self.fields.velocity.y, payload=v),
            replace(
                self.fields.velocity.z,
                owned=replace(self.fields.velocity.z.owned, payload=zero_w),
            ),
        )
        fields = replace(self.fields, velocity=velocity)
        wall = FilteredNeutralLogWall(0.01)
        model = BoussinesqModel(
            DryFlowModel(
                RotationalAdvection(),
                KinematicPressureGradient(0.0),
                wall,
                StaticSmagorinsky(0.0),
                NoRotation(),
            ),
            ConservativeScalarAdvection(),
            StaticSmagorinskyScalarFlux(),
            NoBuoyancy(),
        )

        fused = algebra.fused_boussinesq_tendency(fields, model)
        unpadded = algebra.advection_tendency(
            algebra.dry_flow_context(velocity),
            RotationalAdvection(),
            wall,
        )
        fused_y = fused.velocity.y.payload[:, 1:]
        unpadded_y = unpadded.y.payload[:, 1:]
        self.assertLess(
            float(
                jnp.max(
                    jnp.abs(
                        fused_y
                        - jnp.mean(fused_y, axis=(-2, -1), keepdims=True)
                    )
                )
            ),
            3.0e-12,
        )
        self.assertGreater(
            float(
                jnp.max(
                    jnp.abs(
                        unpadded_y
                        - jnp.mean(unpadded_y, axis=(-2, -1), keepdims=True)
                    )
                )
            ),
            0.1,
        )

    def test_two_thirds_rotational_rhs_filters_unresolved_input_modes(self) -> None:
        algebra = build_discretization(
            self.decomposition,
            nonlinear_dealiasing="two_thirds",
            frozen_zero_scalar=True,
        )
        grid = self.decomposition.grid
        x = 2.0 * jnp.pi * jnp.arange(grid.nx) / grid.nx
        shape = (1, grid.nz, grid.ny, grid.nx)
        u = jnp.broadcast_to(jnp.sin(3.0 * x), shape)
        v = jnp.broadcast_to(jnp.cos(3.0 * x), shape)
        zero_w = jnp.zeros(shape, dtype=u.dtype)
        velocity = VelocityVector(
            replace(self.fields.velocity.x, payload=u),
            replace(self.fields.velocity.y, payload=v),
            replace(
                self.fields.velocity.z,
                owned=replace(self.fields.velocity.z.owned, payload=zero_w),
            ),
        )
        fields = replace(self.fields, velocity=velocity)
        model = BoussinesqModel(
            DryFlowModel(
                RotationalAdvection(),
                KinematicPressureGradient(0.0),
                FilteredNeutralLogWall(0.01),
                StaticSmagorinsky(0.0),
                NoRotation(),
            ),
            ConservativeScalarAdvection(),
            StaticSmagorinskyScalarFlux(),
            NoBuoyancy(),
        )

        fused = algebra.fused_boussinesq_tendency(fields, model)
        for payload in (
            fused.velocity.x.payload[:, 1:],
            fused.velocity.y.payload[:, 1:],
            fused.velocity.z.owned.payload[:, 1:],
        ):
            self.assertLess(float(jnp.max(jnp.abs(payload))), 3.0e-12)

    def test_lasd_fused_rhs_reuses_prepared_momentum_context(self) -> None:
        model = BoussinesqModel(
            DryFlowModel(
                ConservativeAdvection(),
                KinematicPressureGradient(0.002, -0.001),
                FilteredNeutralLogWall(0.01),
                LagrangianScaleDependentDynamic(update_interval=4),
                NoRotation(),
            ),
            ConservativeScalarAdvection(),
            LagrangianScaleDependentScalarFlux(),
            NoBuoyancy(),
        )
        algebra = build_discretization(
            self.decomposition,
            frozen_zero_scalar=True,
        )
        fields = algebra.initialize_lasd_closure(self.fields, model)
        regular = algebra.fused_boussinesq_tendency(fields, model)
        context = algebra.dry_flow_context(fields.velocity)
        reused = algebra.fused_boussinesq_tendency(
            fields,
            model,
            momentum_context=context,
        )

        for expected, actual in (
            (regular.velocity.x.payload, reused.velocity.x.payload),
            (regular.velocity.y.payload, reused.velocity.y.payload),
            (regular.velocity.z.owned.payload, reused.velocity.z.owned.payload),
            (
                regular.potential_temperature.payload,
                reused.potential_temperature.payload,
            ),
        ):
            self.assertLess(float(jnp.max(jnp.abs(expected - actual))), 3.0e-12)

    def test_momentum_only_lasd_matches_full_zero_scalar_update(self) -> None:
        momentum = LagrangianScaleDependentDynamic(update_interval=4)

        def model(*, scalar_updates: bool) -> BoussinesqModel:
            return BoussinesqModel(
                DryFlowModel(
                    ConservativeAdvection(),
                    KinematicPressureGradient(0.0),
                    FilteredNeutralLogWall(0.01),
                    momentum,
                    NoRotation(),
                ),
                ConservativeScalarAdvection(),
                LagrangianScaleDependentScalarFlux(
                    dynamic_updates_enabled=scalar_updates,
                ),
                NoBuoyancy(),
            )

        algebra = build_discretization(
            self.decomposition,
            frozen_zero_scalar=True,
        )
        full_model = model(scalar_updates=True)
        momentum_only_model = model(scalar_updates=False)
        full_fields = algebra.initialize_lasd_closure(self.fields, full_model)
        momentum_only_fields = algebra.initialize_lasd_closure(
            self.fields,
            momentum_only_model,
        )
        clock = AcceptedClock(0.015, 3)
        full, full_diagnostic = algebra.prepare_lasd_closure(
            full_fields,
            full_model,
            clock,
            0.005,
        )
        momentum_only, momentum_only_diagnostic = algebra.prepare_lasd_closure(
            momentum_only_fields,
            momentum_only_model,
            clock,
            0.005,
        )

        self.assertTrue(bool(full_diagnostic.updated))
        self.assertTrue(bool(momentum_only_diagnostic.updated))
        for full_value, momentum_only_value in zip(
            full.closure.momentum.fields(),
            momentum_only.closure.momentum.fields(),
            strict=True,
        ):
            self.assertLess(
                float(
                    jnp.max(
                        jnp.abs(full_value.payload - momentum_only_value.payload)
                    )
                ),
                3.0e-12,
            )
        for before, after in zip(
            momentum_only_fields.closure.scalar.fields(),
            momentum_only.closure.scalar.fields(),
            strict=True,
        ):
            self.assertEqual(
                float(jnp.max(jnp.abs(before.payload - after.payload))),
                0.0,
            )

    def test_static_smagorinsky_fused_rhs_matches_individual_contributions(
        self,
    ) -> None:
        self.assert_fused_matches_contributions(
            BoussinesqModel(
                DryFlowModel(
                    ConservativeAdvection(),
                    KinematicPressureGradient(0.0),
                    FilteredNeutralLogWall(0.01),
                    StaticSmagorinsky(0.16),
                    NoRotation(),
                ),
                ConservativeScalarAdvection(),
                StaticSmagorinskyScalarFlux(),
                NoBuoyancy(),
            )
        )

    def test_lasd_active_scalar_coriolis_fusion_matches_contributions(self) -> None:
        self.assert_fused_matches_contributions(
            BoussinesqModel(
                DryFlowModel(
                    ConservativeAdvection(),
                    KinematicPressureGradient(0.002, -0.001),
                    FilteredNeutralLogWall(0.01),
                    LagrangianScaleDependentDynamic(update_interval=4),
                    CoriolisGeostrophic(0.03, 1.2, -0.2, 0.01),
                ),
                ConservativeScalarAdvection(),
                LagrangianScaleDependentScalarFlux(),
                NoBuoyancy(),
                scalar_boundary=ScalarFluxBoundary(0.002, -0.001),
            ),
            fields=self.active_fields,
            frozen_zero_scalar=False,
        )

    def test_lasd_convective_buoyancy_fusion_matches_contributions(self) -> None:
        self.assert_fused_matches_contributions(
            BoussinesqModel(
                DryFlowModel(
                    ConservativeAdvection(),
                    KinematicPressureGradient(0.0),
                    FilteredNeutralLogWall(0.01),
                    LagrangianScaleDependentDynamic(update_interval=4),
                    NoRotation(),
                ),
                ConservativeScalarAdvection(),
                LagrangianScaleDependentScalarFlux(
                    stability_buoyancy_coefficient=0.025,
                ),
                LinearBoussinesqBuoyancy(0.025),
                scalar_boundary=ScalarFluxBoundary(0.002, 0.0),
            ),
            fields=self.active_fields,
            frozen_zero_scalar=False,
        )

    def test_lasd_coupled_surface_sources_match_individual_terms(self) -> None:
        self.assert_coupled_surface_sources_match_individual_terms(
            LagrangianScaleDependentDynamic(update_interval=4),
            LagrangianScaleDependentScalarFlux(),
        )

    def assert_coupled_surface_sources_match_individual_terms(
        self,
        momentum_sgs,
        scalar_sgs,
    ) -> None:
        model = BoussinesqModel(
            DryFlowModel(
                ConservativeAdvection(),
                KinematicPressureGradient(0.002, -0.001),
                FilteredNeutralLogWall(0.01),
                momentum_sgs,
                CoriolisGeostrophic(0.03, 1.2, -0.2, 0.01),
            ),
            ConservativeScalarAdvection(),
            scalar_sgs,
            LinearBoussinesqBuoyancy(0.025),
        )
        algebra = build_discretization(
            self.decomposition,
            frozen_zero_scalar=False,
        )
        fields = self.active_fields
        if isinstance(momentum_sgs, LagrangianScaleDependentDynamic):
            fields = algebra.initialize_lasd_closure(fields, model)
        evaluation = Evaluation(fields, AcceptedClock(0.0, 0), None)
        contributions = BoussinesqVectorField(
            algebra,
            model,
        ).evaluate_contributions(evaluation)
        wall_x, wall_y, scalar_source = 0.013, -0.007, -0.004
        imposed_wall = algebra._dry_tendency(
            jnp.zeros_like(fields.velocity.x.payload).at[:, 0].set(wall_x),
            jnp.zeros_like(fields.velocity.y.payload).at[:, 0].set(wall_y),
            jnp.zeros_like(fields.velocity.z.owned.payload),
        )
        expected_velocity = algebra.combine_tendencies(
            (
                contributions.advection,
                contributions.pressure_gradient,
                imposed_wall,
                contributions.momentum_sgs,
                contributions.coriolis_geostrophic,
                contributions.buoyancy,
                contributions.rayleigh_damping,
            )
        )
        context = algebra.boussinesq_context(fields)
        scalar_surface = algebra._scalar_tendency(
            context,
            jnp.zeros_like(fields.potential_temperature.payload)
            .at[:, 0]
            .set(scalar_source),
        )
        expected_scalar = algebra.combine_scalar_tendencies(
            (*contributions.scalar_values(), scalar_surface)
        )
        actual = algebra.fused_boussinesq_tendency(
            fields,
            model,
            wall_acceleration=(wall_x, wall_y),
            scalar_surface_source=scalar_source,
        )
        self.assertIsNotNone(actual)
        for expected, fused_payload in (
            (expected_velocity.x.payload, actual.velocity.x.payload),
            (expected_velocity.y.payload, actual.velocity.y.payload),
            (expected_velocity.z.owned.payload, actual.velocity.z.owned.payload),
            (expected_scalar.payload, actual.potential_temperature.payload),
        ):
            self.assertLess(
                float(jnp.max(jnp.abs(expected - fused_payload))),
                3.0e-12,
            )

    def test_monin_obukhov_transfer_drives_the_uniform_fused_rhs(self) -> None:
        model = BoussinesqModel(
            DryFlowModel(
                ConservativeAdvection(),
                KinematicPressureGradient(0.0),
                FilteredNeutralLogWall(0.01),
                LagrangianScaleDependentDynamic(update_interval=4),
                NoRotation(),
            ),
            ConservativeScalarAdvection(),
            LagrangianScaleDependentScalarFlux(),
            LinearBoussinesqBuoyancy(0.025),
            surface_transfer=MoninObukhovSurfaceTransfer(
                scalar_roughness_length=0.01,
                surface_scalar_initial=-1.0,
                surface_scalar_rate=-0.1,
            ),
        )
        algebra = build_discretization(
            self.decomposition,
            frozen_zero_scalar=False,
        )
        fields = algebra.initialize_lasd_closure(self.active_fields, model)
        clock = AcceptedClock(0.5, 2)
        transfer = algebra.surface_transfer(fields, model, clock)

        self.assertIsNotNone(transfer)
        self.assertGreater(float(transfer.stress_x), 0.0)
        self.assertLess(float(transfer.scalar_flux), 0.0)
        self.assertLess(float(transfer.wall_x_acceleration), 0.0)
        self.assertLess(float(transfer.scalar_surface_source), 0.0)
        result = BoussinesqVectorField(algebra, model)(
            Evaluation(fields, clock, None)
        )
        explicit = algebra.fused_boussinesq_tendency(
            fields,
            replace(model, surface_transfer=NoSurfaceTransfer()),
            wall_acceleration=(
                transfer.wall_x_acceleration,
                transfer.wall_y_acceleration,
            ),
            scalar_surface_source=transfer.scalar_surface_source,
            execution_time=clock.time,
        )
        for automatic, imposed in (
            (result.tendency.velocity.x.payload, explicit.velocity.x.payload),
            (result.tendency.velocity.y.payload, explicit.velocity.y.payload),
            (
                result.tendency.potential_temperature.payload,
                explicit.potential_temperature.payload,
            ),
        ):
            self.assertLess(float(jnp.max(jnp.abs(automatic - imposed))), 3.0e-12)
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(result.tendency.velocity.x.payload)))
        )
        self.assertTrue(
            bool(
                jnp.all(
                    jnp.isfinite(result.tendency.potential_temperature.payload)
                )
            )
        )

    def test_lasd_diagnoses_negative_resolved_tke_transfer(self) -> None:
        wall = FilteredNeutralLogWall(0.01)
        momentum = LagrangianScaleDependentDynamic(update_interval=4)
        model = BoussinesqModel(
            DryFlowModel(
                ConservativeAdvection(),
                KinematicPressureGradient(0.0, 0.0),
                wall,
                momentum,
                NoRotation(),
            ),
            ConservativeScalarAdvection(),
            LagrangianScaleDependentScalarFlux(),
            NoBuoyancy(),
        )
        algebra = build_discretization(self.decomposition)
        fields = algebra.initialize_lasd_closure(self.fields, model)
        context = algebra.boussinesq_context(fields).momentum
        transfer = algebra.momentum_sgs_tke_transfer(
            context,
            momentum,
            wall=wall,
        )
        self.assertEqual(transfer.shape, (1, 4, 8, 8))
        self.assertTrue(bool(jnp.all(jnp.isfinite(transfer))))
        self.assertLess(float(jnp.min(transfer)), 0.0)
        self.assertTrue(
            bool(jnp.all(jnp.mean(transfer, axis=(-2, -1)) <= 1.0e-12))
        )


if __name__ == "__main__":
    unittest.main()
