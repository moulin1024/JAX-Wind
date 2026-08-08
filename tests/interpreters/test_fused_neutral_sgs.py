from __future__ import annotations

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
    EqualZSlab,
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
from jaxwind.interpreters.jax_zslab import (  # noqa: E402
    ZFaceFieldContext,
    build_zslab_interpreter,
)
from jaxwind.operators import VelocityVector  # noqa: E402
from jaxwind.physics import (  # noqa: E402
    AnisotropicMinimumDissipation,
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
    ModulatedGradientModel,
    NoBuoyancy,
    NoRotation,
    ScalarFluxBoundary,
    StaticSmagorinskyScalarFlux,
)


class FusedNeutralSgsTests(unittest.TestCase):
    def setUp(self) -> None:
        grid = UniformGrid(8, 8, 4, 8.0, 8.0, 4.0)
        self.decomposition = EqualZSlab(
            grid,
            MeshTopology((MeshAxis("z", 1),)),
            DistributionSpec.z_slab(),
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
            ZFaceFieldContext(
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
        algebra = build_zslab_interpreter(
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

    def test_amd_fused_rhs_matches_individual_contributions(self) -> None:
        self.assert_fused_matches_contributions(
            BoussinesqModel(
                DryFlowModel(
                    ConservativeAdvection(),
                    KinematicPressureGradient(0.002, -0.001),
                    FilteredNeutralLogWall(0.01),
                    AnisotropicMinimumDissipation(),
                    NoRotation(),
                ),
                ConservativeScalarAdvection(),
                StaticSmagorinskyScalarFlux(0.4),
                NoBuoyancy(),
            )
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

    def test_amd_active_scalar_coriolis_fusion_matches_contributions(self) -> None:
        self.assert_fused_matches_contributions(
            BoussinesqModel(
                DryFlowModel(
                    ConservativeAdvection(),
                    KinematicPressureGradient(0.002, -0.001),
                    FilteredNeutralLogWall(0.01),
                    AnisotropicMinimumDissipation(),
                    CoriolisGeostrophic(0.03, 1.2, -0.2, 0.01),
                ),
                ConservativeScalarAdvection(),
                StaticSmagorinskyScalarFlux(0.4),
                NoBuoyancy(),
                scalar_boundary=ScalarFluxBoundary(0.002, -0.001),
            ),
            fields=self.active_fields,
            frozen_zero_scalar=False,
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

    def test_mgm_and_amd_diagnose_nonnegative_sgs_energy(self) -> None:
        algebra = build_zslab_interpreter(self.decomposition)
        context = algebra.boussinesq_context(self.fields).momentum
        wall = FilteredNeutralLogWall(0.01)
        models = (
            ModulatedGradientModel(gradient_norm_epsilon=1.0e-12),
            AnisotropicMinimumDissipation(),
        )
        diagnostics = (
            algebra.momentum_sgs_diagnostic_fields(
                context,
                models[0],
                wall=wall,
            ),
            algebra.momentum_sgs_diagnostic_fields(
                context,
                models[1],
                wall=wall,
            ),
        )
        for model, diagnostic in zip(models, diagnostics, strict=True):
            self.assertEqual(diagnostic.sgs_tke.shape, (1, 4, 8, 8))
            self.assertTrue(bool(jnp.all(jnp.isfinite(diagnostic.sgs_tke))))
            self.assertTrue(bool(jnp.all(diagnostic.sgs_tke >= 0.0)))
            self.assertGreater(float(jnp.max(diagnostic.sgs_tke)), 0.0)
            transfer = algebra.momentum_sgs_tke_transfer(
                context,
                model,
                wall=wall,
            )
            self.assertEqual(transfer.shape, diagnostic.sgs_tke.shape)
            self.assertTrue(bool(jnp.all(jnp.isfinite(transfer))))
            self.assertLess(float(jnp.min(transfer)), 0.0)
            self.assertTrue(
                bool(jnp.all(jnp.mean(transfer, axis=(-2, -1)) <= 1.0e-12))
            )
        self.assertGreater(
            float(jnp.max(diagnostics[1].momentum_diffusivity)),
            0.0,
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
        algebra = build_zslab_interpreter(self.decomposition)
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
