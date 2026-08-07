from __future__ import annotations

import argparse
import json
import math

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    AddressableField,
    Cell,
    DistributionSpec,
    EqualZSlab,
    Field,
    GlobalTestRegion,
    MeshAxis,
    MeshTopology,
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
    AnisotropicMinimumDissipation,
    ConservativeAdvection,
    CoriolisGeostrophic,
    FilteredNeutralLogWall,
    KinematicPressureGradient,
    ModulatedGradientModel,
    NeutralLogWall,
    RotationalAdvection,
    StaticSmagorinsky,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    args = parser.parse_args()
    if jax.device_count() != args.devices:
        raise RuntimeError(
            f"expected {args.devices} devices, found {jax.device_count()}"
        )
    dtype = getattr(jnp, args.dtype)
    grid = UniformGrid(6, 6, 8, 6.0, 6.0, 4.0)
    z = jnp.arange(grid.nz, dtype=dtype)[:, None, None]
    zf = jnp.arange(grid.nz + 1, dtype=dtype)[:, None, None]
    y = 2.0 * jnp.pi * jnp.arange(grid.ny, dtype=dtype)[None, :, None] / grid.ny
    x = 2.0 * jnp.pi * jnp.arange(grid.nx, dtype=dtype)[None, None, :] / grid.nx
    u = 2.0 + 0.22 * jnp.sin(x) + 0.11 * jnp.cos(y) + 0.04 * z
    v = -0.3 + 0.17 * jnp.cos(x - y) - 0.025 * z
    w = 0.12 * jnp.sin(x) * jnp.cos(y) * jnp.sin(jnp.pi * zf / grid.nz)

    cells = GlobalTestRegion(grid, Cell)
    faces = GlobalTestRegion(grid, ZFace)
    reference_velocity = VelocityVector(
        Field(XVelocity, Cell, cells, Projected, u),
        Field(YVelocity, Cell, cells, Projected, v),
        Field(VerticalVelocity, ZFace, faces, Projected, w),
    )
    reference = JaxOracleProjection()
    reference_context = reference.dry_flow_context(reference_velocity)

    decomposition = EqualZSlab(
        grid,
        MeshTopology((MeshAxis("z", args.devices),)),
        DistributionSpec.z_slab(),
    )
    shape = (args.devices, decomposition.cells_per_shard, grid.ny, grid.nx)
    production_velocity = VelocityVector(
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
    )
    production = build_zslab_interpreter(
        decomposition,
        addressable_shards=tuple(range(args.devices)),
    )
    production_context = production.dry_flow_context(production_velocity)
    uncorrected = build_zslab_interpreter(
        decomposition,
        addressable_shards=tuple(range(args.devices)),
        porte_agel_wall_correction=False,
    )
    uncorrected_context = uncorrected.dry_flow_context(production_velocity)
    term_cases = (
        (
            "advection",
            reference.advection_tendency,
            production.advection_tendency,
            ConservativeAdvection(),
        ),
        (
            "pressure_gradient",
            reference.pressure_gradient_tendency,
            production.pressure_gradient_tendency,
            KinematicPressureGradient(0.002, -0.001),
        ),
        (
            "wall",
            reference.wall_stress_tendency,
            production.wall_stress_tendency,
            NeutralLogWall(0.01),
        ),
        (
            "wall_filtered",
            reference.wall_stress_tendency,
            production.wall_stress_tendency,
            FilteredNeutralLogWall(0.01),
        ),
        (
            "sgs",
            reference.sgs_tendency,
            production.sgs_tendency,
            StaticSmagorinsky(0.16),
        ),
        (
            "mgm",
            reference.sgs_tendency,
            production.sgs_tendency,
            ModulatedGradientModel(kinematic_viscosity=1.5e-5),
        ),
        (
            "amd",
            reference.sgs_tendency,
            production.sgs_tendency,
            AnisotropicMinimumDissipation(),
        ),
        (
            "coriolis_geostrophic",
            reference.coriolis_geostrophic_tendency,
            production.coriolis_geostrophic_tendency,
            CoriolisGeostrophic(-1.0e-4, 8.0, -0.5, 1.0e-4),
        ),
    )
    errors = {}
    porte_agel_factor = 1.0 / math.log(3.0) - 1.0
    expected_dudz = uncorrected_context.arrays.dudz_upper[0, 0] + (
        porte_agel_factor * jnp.mean(uncorrected_context.arrays.dudz_upper[0, 0])
    )
    expected_dvdz = uncorrected_context.arrays.dvdz_upper[0, 0] + (
        porte_agel_factor * jnp.mean(uncorrected_context.arrays.dvdz_upper[0, 0])
    )
    errors["porte_agel_switch"] = max(
        float(
            jnp.max(jnp.abs(production_context.arrays.dudz_upper[0, 0] - expected_dudz))
        ),
        float(
            jnp.max(jnp.abs(production_context.arrays.dvdz_upper[0, 0] - expected_dvdz))
        ),
    )
    errors["porte_agel_has_effect"] = bool(
        jnp.max(
            jnp.abs(
                production_context.arrays.dudz_upper[0, 0]
                - uncorrected_context.arrays.dudz_upper[0, 0]
            )
        )
        > 0.0
    )
    demo_wall = FilteredNeutralLogWall(0.01)
    demo_mgm = ModulatedGradientModel(kinematic_viscosity=1.5e-5)
    for name, expected, actual in (
        (
            "rotational_advection",
            reference.advection_tendency(
                reference_context,
                RotationalAdvection(),
                demo_wall,
            ),
            production.advection_tendency(
                production_context,
                RotationalAdvection(),
                demo_wall,
            ),
        ),
        (
            "mgm_log_wall",
            reference.sgs_tendency(reference_context, demo_mgm, demo_wall),
            production.sgs_tendency(production_context, demo_mgm, demo_wall),
        ),
    ):
        errors[name] = max(
            float(
                jnp.max(jnp.abs(actual.x.payload - expected.x.payload.reshape(shape)))
            ),
            float(
                jnp.max(jnp.abs(actual.y.payload - expected.y.payload.reshape(shape)))
            ),
            float(
                jnp.max(
                    jnp.abs(
                        actual.z.owned.payload - expected.z.payload[1:].reshape(shape)
                    )
                )
            ),
        )
    reference_terms = []
    production_terms = []
    for name, reference_term, production_term, config in term_cases:
        expected = reference_term(reference_context, config)
        actual = production_term(production_context, config)
        reference_terms.append(expected)
        production_terms.append(actual)
        errors[name] = max(
            float(
                jnp.max(jnp.abs(actual.x.payload - expected.x.payload.reshape(shape)))
            ),
            float(
                jnp.max(jnp.abs(actual.y.payload - expected.y.payload.reshape(shape)))
            ),
            float(
                jnp.max(
                    jnp.abs(
                        actual.z.owned.payload - expected.z.payload[1:].reshape(shape)
                    )
                )
            ),
        )
    sgs_config = StaticSmagorinsky(0.16)
    expected_txz, expected_tyz = reference.sgs_vertical_flux(
        reference_context,
        sgs_config,
    )
    actual_txz, actual_tyz = production.sgs_vertical_flux(
        production_context,
        sgs_config,
    )
    errors["sgs_vertical_flux"] = max(
        float(jnp.max(jnp.abs(actual_txz - expected_txz[1:].reshape(shape)))),
        float(jnp.max(jnp.abs(actual_tyz - expected_tyz[1:].reshape(shape)))),
    )
    mgm_config = ModulatedGradientModel(kinematic_viscosity=1.5e-5)
    expected_txz, expected_tyz = reference.sgs_vertical_flux(
        reference_context,
        mgm_config,
    )
    actual_txz, actual_tyz = production.sgs_vertical_flux(
        production_context,
        mgm_config,
    )
    errors["mgm_vertical_flux"] = max(
        float(jnp.max(jnp.abs(actual_txz - expected_txz[1:].reshape(shape)))),
        float(jnp.max(jnp.abs(actual_tyz - expected_tyz[1:].reshape(shape)))),
    )
    amd_config = AnisotropicMinimumDissipation()
    expected_txz, expected_tyz = reference.sgs_vertical_flux(
        reference_context,
        amd_config,
    )
    actual_txz, actual_tyz = production.sgs_vertical_flux(
        production_context,
        amd_config,
    )
    errors["amd_vertical_flux"] = max(
        float(jnp.max(jnp.abs(actual_txz - expected_txz[1:].reshape(shape)))),
        float(jnp.max(jnp.abs(actual_tyz - expected_tyz[1:].reshape(shape)))),
    )
    expected_total = reference.combine_tendencies(tuple(reference_terms))
    actual_total = production.combine_tendencies(tuple(production_terms))
    errors["combined"] = max(
        float(
            jnp.max(
                jnp.abs(
                    actual_total.x.payload - expected_total.x.payload.reshape(shape)
                )
            )
        ),
        float(
            jnp.max(
                jnp.abs(
                    actual_total.y.payload - expected_total.y.payload.reshape(shape)
                )
            )
        ),
        float(
            jnp.max(
                jnp.abs(
                    actual_total.z.owned.payload
                    - expected_total.z.payload[1:].reshape(shape)
                )
            )
        ),
    )
    errors["dtype_preserved"] = all(
        field.dtype == dtype
        for field in (
            actual_total.x.payload,
            actual_total.y.payload,
            actual_total.z.owned.payload,
        )
    )
    print(json.dumps(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
