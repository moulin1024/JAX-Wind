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
from jaxwind.interpreters.jax_zslab import (  # noqa: E402
    ZFaceFieldContext,
    build_zslab_interpreter,
)
from jaxwind.operators import VelocityVector  # noqa: E402
from jaxwind.physics import (  # noqa: E402
    BoussinesqFields,
    BoussinesqModel,
    ConservativeAdvection,
    ConservativeScalarAdvection,
    DryFlowModel,
    FilteredNeutralLogWall,
    KinematicPressureGradient,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    NoBuoyancy,
    NoRotation,
)


class NeutralSgsTests(unittest.TestCase):
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
