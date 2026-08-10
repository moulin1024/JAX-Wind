from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

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
    Field,
    GlobalTestRegion,
    MeshAxis,
    MeshTopology,
    PotentialTemperaturePerturbation,
    Projected,
    UniformGrid,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from tests.support.jax_oracle import JaxOracleProjection  # noqa: E402
from jaxwind.interpreters.jax_zslab import (  # noqa: E402
    ZFaceFieldContext,
    build_zslab_interpreter,
)
from jaxwind.operators import VelocityVector  # noqa: E402
from jaxwind.physics import (  # noqa: E402
    BoussinesqFields,
    ConservativeScalarAdvection,
    ConcurrentPrecursorFringe,
    DiagnosticLasdConstants,
    LinearBoussinesqBuoyancy,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    NeutralLogWall,
    RayleighGeostrophicDamping,
    ScalarFluxBoundary,
    StaticSmagorinsky,
    StaticSmagorinskyScalarFlux,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    args = parser.parse_args()
    dtype = getattr(jnp, args.dtype)
    grid = UniformGrid(6, 6, 8, 6.0, 6.0, 4.0)
    z = jnp.arange(grid.nz, dtype=dtype)[:, None, None]
    zf = jnp.arange(grid.nz + 1, dtype=dtype)[:, None, None]
    y = 2.0 * jnp.pi * jnp.arange(grid.ny, dtype=dtype)[None, :, None] / grid.ny
    x = 2.0 * jnp.pi * jnp.arange(grid.nx, dtype=dtype)[None, None, :] / grid.nx
    u = 2.0 + 0.2 * jnp.sin(x) + 0.03 * z + 0.0 * y
    v = -0.3 + 0.1 * jnp.cos(y) + 0.0 * x + 0.0 * z
    w = 0.08 * jnp.sin(x) * jnp.cos(y) * jnp.sin(jnp.pi * zf / grid.nz)
    theta = 0.3 * z + 0.06 * jnp.sin(x - y)

    cells = GlobalTestRegion(grid, Cell)
    faces = GlobalTestRegion(grid, ZFace)
    reference_fields = BoussinesqFields(
        VelocityVector(
            Field(XVelocity, Cell, cells, Projected, u),
            Field(YVelocity, Cell, cells, Projected, v),
            Field(VerticalVelocity, ZFace, faces, Projected, w),
        ),
        Field(PotentialTemperaturePerturbation, Cell, cells, Accepted, theta),
    )
    reference = JaxOracleProjection()
    reference_context = reference.boussinesq_context(reference_fields)

    decomposition = EqualZSlab(
        grid,
        MeshTopology((MeshAxis("z", args.devices),)),
        DistributionSpec.z_slab(),
    )
    shape = (args.devices, decomposition.cells_per_shard, grid.ny, grid.nx)
    production_fields = BoussinesqFields(
        VelocityVector(
            AddressableField(
                XVelocity,
                Cell,
                decomposition.regions(Cell),
                Projected,
                u.reshape(shape),
            ),
            AddressableField(
                YVelocity,
                Cell,
                decomposition.regions(Cell),
                Projected,
                v.reshape(shape),
            ),
            ZFaceFieldContext(
                AddressableField(
                    VerticalVelocity,
                    ZFace,
                    decomposition.regions(ZFace),
                    Projected,
                    w[1:].reshape(shape),
                ),
                w[0],
            ),
        ),
        AddressableField(
            PotentialTemperaturePerturbation,
            Cell,
            decomposition.regions(Cell),
            Accepted,
            theta.reshape(shape),
        ),
    )
    production = build_zslab_interpreter(
        decomposition,
        addressable_shards=tuple(range(args.devices)),
    )
    production_context = production.boussinesq_context(production_fields)
    scalar_advection = ConservativeScalarAdvection()
    momentum_sgs = StaticSmagorinsky(0.16)
    scalar_sgs = StaticSmagorinskyScalarFlux(0.4)
    buoyancy = LinearBoussinesqBuoyancy(0.03)
    rayleigh = RayleighGeostrophicDamping(2.0, 0.4, 2.5, -0.1)
    expected_advection = reference.scalar_advection_tendency(
        reference_context, scalar_advection
    )
    actual_advection = production.scalar_advection_tendency(
        production_context, scalar_advection
    )
    expected_sgs = reference.scalar_sgs_tendency(
        reference_context, momentum_sgs, scalar_sgs
    )
    actual_sgs = production.scalar_sgs_tendency(
        production_context, momentum_sgs, scalar_sgs
    )
    expected_buoyancy = reference.buoyancy_tendency(reference_context, buoyancy)
    actual_buoyancy = production.buoyancy_tendency(production_context, buoyancy)
    expected_rayleigh = reference.rayleigh_damping_tendency(reference_context, rayleigh)
    actual_rayleigh = production.rayleigh_damping_tendency(production_context, rayleigh)
    errors = {
        "scalar_advection": float(
            jnp.max(
                jnp.abs(
                    actual_advection.payload - expected_advection.payload.reshape(shape)
                )
            )
        ),
        "scalar_sgs": float(
            jnp.max(jnp.abs(actual_sgs.payload - expected_sgs.payload.reshape(shape)))
        ),
        "buoyancy": float(
            jnp.max(
                jnp.abs(
                    actual_buoyancy.z.owned.payload
                    - expected_buoyancy.z.payload[1:].reshape(shape)
                )
            )
        ),
        "rayleigh": float(
            max(
                jnp.max(
                    jnp.abs(
                        actual_rayleigh.x.payload
                        - expected_rayleigh.x.payload.reshape(shape)
                    )
                ),
                jnp.max(
                    jnp.abs(
                        actual_rayleigh.y.payload
                        - expected_rayleigh.y.payload.reshape(shape)
                    )
                ),
                jnp.max(
                    jnp.abs(
                        actual_rayleigh.z.owned.payload
                        - expected_rayleigh.z.payload[1:].reshape(shape)
                    )
                ),
            )
        ),
    }
    expected_total = reference.combine_scalar_tendencies(
        (expected_advection, expected_sgs)
    )
    actual_total = production.combine_scalar_tendencies((actual_advection, actual_sgs))
    errors["combined_scalar"] = float(
        jnp.max(jnp.abs(actual_total.payload - expected_total.payload.reshape(shape)))
    )
    errors["dtype_preserved"] = actual_total.payload.dtype == dtype

    momentum_lasd = LagrangianScaleDependentDynamic(update_interval=2)
    scalar_lasd = LagrangianScaleDependentScalarFlux()
    lasd_model = SimpleNamespace(
        momentum=SimpleNamespace(sgs=momentum_lasd),
        scalar_sgs=scalar_lasd,
    )
    reference_lasd = reference.initialize_lasd_closure(
        reference_fields,
        lasd_model,
    )
    production_lasd = production.initialize_lasd_closure(
        production_fields,
        lasd_model,
    )
    for step in range(2):
        reference_lasd, _ = reference.prepare_lasd_closure(
            reference_lasd,
            lasd_model,
            AcceptedClock(0.01 * step, step),
            0.01,
        )
        production_lasd, _ = production.prepare_lasd_closure(
            production_lasd,
            lasd_model,
            AcceptedClock(0.01 * step, step),
            0.01,
        )
    errors["lasd_memory"] = float(
        max(
            jnp.max(jnp.abs(actual.payload - expected.payload.reshape(shape)))
            for actual, expected in zip(
                production_lasd.closure.fields(),
                reference_lasd.closure.fields(),
                strict=True,
            )
        )
    )
    reference_target = reference.initialize_lasd_closure(
        reference_fields,
        lasd_model,
    )
    production_target = production.initialize_lasd_closure(
        production_fields,
        lasd_model,
    )
    reference_target, _ = reference.prepare_lasd_closure(
        reference_target,
        lasd_model,
        AcceptedClock(0.0, 0),
        0.01,
    )
    production_target, _ = production.prepare_lasd_closure(
        production_target,
        lasd_model,
        AcceptedClock(0.0, 0),
        0.01,
    )
    fringe = ConcurrentPrecursorFringe(
        3.0,
        0.2,
        rise_width=0.5,
        fall_width=0.5,
    )
    reference_relaxed = reference.relax_lasd_closure(
        reference_lasd,
        reference_target.closure,
        fringe,
        0.01,
    )
    production_relaxed = production.relax_lasd_closure(
        production_lasd,
        production_target.closure,
        fringe,
        0.01,
    )
    errors["lasd_fringe_memory"] = float(
        max(
            jnp.max(jnp.abs(actual.payload - expected.payload.reshape(shape)))
            for actual, expected in zip(
                production_relaxed.closure.fields(),
                reference_relaxed.closure.fields(),
                strict=True,
            )
        )
    )
    reference_lasd_context = reference.boussinesq_context(reference_lasd)
    production_lasd_context = production.boussinesq_context(production_lasd)
    expected_momentum_lasd = reference.sgs_tendency(
        reference_lasd_context.momentum,
        momentum_lasd,
    )
    actual_momentum_lasd = production.sgs_tendency(
        production_lasd_context.momentum,
        momentum_lasd,
    )
    errors["lasd_momentum_tendency"] = float(
        max(
            jnp.max(
                jnp.abs(
                    actual_momentum_lasd.x.payload
                    - expected_momentum_lasd.x.payload.reshape(shape)
                )
            ),
            jnp.max(
                jnp.abs(
                    actual_momentum_lasd.y.payload
                    - expected_momentum_lasd.y.payload.reshape(shape)
                )
            ),
            jnp.max(
                jnp.abs(
                    actual_momentum_lasd.z.owned.payload
                    - expected_momentum_lasd.z.payload[1:].reshape(shape)
                )
            ),
        )
    )
    expected_scalar_lasd = reference.scalar_sgs_tendency(
        reference_lasd_context,
        momentum_lasd,
        scalar_lasd,
    )
    actual_scalar_lasd = production.scalar_sgs_tendency(
        production_lasd_context,
        momentum_lasd,
        scalar_lasd,
    )
    errors["lasd_scalar_tendency"] = float(
        jnp.max(
            jnp.abs(
                actual_scalar_lasd.payload - expected_scalar_lasd.payload.reshape(shape)
            )
        )
    )
    expected_diagnostics = reference.lasd_diagnostic_fields(
        reference_lasd_context,
        momentum_lasd,
        scalar_lasd,
        ScalarFluxBoundary(1.0e-3, 0.0),
        constants=DiagnosticLasdConstants(horizontal_homogeneous_wall=True),
        wall=NeutralLogWall(0.01),
    )
    actual_diagnostics = production.lasd_diagnostic_fields(
        production_lasd_context,
        momentum_lasd,
        scalar_lasd,
        ScalarFluxBoundary(1.0e-3, 0.0),
        constants=DiagnosticLasdConstants(horizontal_homogeneous_wall=True),
        wall=NeutralLogWall(0.01),
    )
    diagnostic_errors = []
    for name in (
        "momentum_diffusivity",
        "scalar_diffusivity",
        "scalar_flux_x",
        "scalar_flux_y",
        "sgs_tke",
        "scalar_variance_numerator",
        "scalar_variance",
    ):
        diagnostic_errors.append(
            jnp.max(
                jnp.abs(
                    getattr(actual_diagnostics, name)
                    - getattr(expected_diagnostics, name).reshape(shape)
                )
            )
        )
    diagnostic_errors.append(
        jnp.max(
            jnp.abs(
                actual_diagnostics.scalar_flux_z
                - expected_diagnostics.scalar_flux_z[1:].reshape(shape)
            )
        )
    )
    errors["lasd_diagnostics"] = float(max(diagnostic_errors))
    print(json.dumps(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
