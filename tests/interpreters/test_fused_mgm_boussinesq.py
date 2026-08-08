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
    PotentialTemperaturePerturbation,
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
    BoussinesqFields,
    BoussinesqModel,
    BoussinesqVectorField,
    ConservativeAdvection,
    ConservativeScalarAdvection,
    CoriolisGeostrophic,
    DryFlowModel,
    FilteredNeutralLogWall,
    KinematicPressureGradient,
    ModulatedGradientModel,
    NoBuoyancy,
    StaticSmagorinskyScalarFlux,
)


class FusedMgmBoussinesqTests(unittest.TestCase):
    def test_fused_rhs_matches_individual_contributions(self) -> None:
        grid = UniformGrid(8, 8, 4, 8.0, 8.0, 4.0)
        decomposition = EqualZSlab(
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
        theta = 0.4 * z + 0.05 * jnp.sin(x - y)
        shape = (1, grid.nz, grid.ny, grid.nx)
        regions = decomposition.regions(Cell)
        velocity = VelocityVector(
            AddressableField(XVelocity, Cell, regions, Projected, u.reshape(shape)),
            AddressableField(YVelocity, Cell, regions, Projected, v.reshape(shape)),
            ZFaceFieldContext(
                AddressableField(
                    VerticalVelocity,
                    ZFace,
                    decomposition.regions(ZFace),
                    Projected,
                    w.reshape(shape),
                ),
                jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
            ),
        )
        fields = BoussinesqFields(
            velocity,
            AddressableField(
                PotentialTemperaturePerturbation,
                Cell,
                regions,
                Accepted,
                theta.reshape(shape),
            ),
        )
        model = BoussinesqModel(
            DryFlowModel(
                ConservativeAdvection(),
                KinematicPressureGradient(0.002, -0.001),
                FilteredNeutralLogWall(0.01),
                ModulatedGradientModel(kinematic_viscosity=1.5e-5),
                CoriolisGeostrophic(0.03, 1.2, -0.2, 0.01),
            ),
            ConservativeScalarAdvection(),
            StaticSmagorinskyScalarFlux(0.4),
            NoBuoyancy(),
        )
        algebra = build_zslab_interpreter(decomposition)
        vector_field = BoussinesqVectorField(algebra, model)
        evaluation = Evaluation(fields, AcceptedClock(0.0, 0), None)
        contributions = vector_field.evaluate_contributions(evaluation)
        expected_velocity = algebra.combine_tendencies(
            contributions.momentum_values()
        )
        expected_scalar = algebra.combine_scalar_tendencies(
            contributions.scalar_values()
        )
        self.assertIsNotNone(algebra.fused_boussinesq_tendency(fields, model))
        actual = vector_field(evaluation).tendency
        for expected, fused in (
            (expected_velocity.x.payload, actual.velocity.x.payload),
            (expected_velocity.y.payload, actual.velocity.y.payload),
            (expected_velocity.z.owned.payload, actual.velocity.z.owned.payload),
            (expected_scalar.payload, actual.potential_temperature.payload),
        ):
            self.assertLess(float(jnp.max(jnp.abs(expected - fused))), 2.0e-12)

        frozen_algebra = build_zslab_interpreter(
            decomposition,
            frozen_zero_scalar=True,
        )
        zero_fields = BoussinesqFields(
            velocity,
            AddressableField(
                PassiveScalarConcentration,
                Cell,
                regions,
                Accepted,
                jnp.zeros(shape, dtype=jnp.float64),
            ),
        )
        frozen_evaluation = Evaluation(
            zero_fields,
            AcceptedClock(0.0, 0),
            None,
        )
        frozen_vector_field = BoussinesqVectorField(frozen_algebra, model)
        expected_frozen_scalar = frozen_algebra.combine_scalar_tendencies(
            frozen_vector_field.evaluate_contributions(
                frozen_evaluation
            ).scalar_values()
        )
        actual_frozen_scalar = frozen_vector_field(
            frozen_evaluation
        ).tendency.potential_temperature
        self.assertLess(
            float(
                jnp.max(
                    jnp.abs(
                        expected_frozen_scalar.payload
                        - actual_frozen_scalar.payload
                    )
                )
            ),
            2.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
