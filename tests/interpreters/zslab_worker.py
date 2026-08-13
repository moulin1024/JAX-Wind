from __future__ import annotations

import argparse
import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    AddressableField,
    Cell,
    DistributionSpec,
    EqualVerticalPartition,
    Evaluated,
    Field,
    GlobalTestRegion,
    MeshAxis,
    MeshTopology,
    PressureCorrection,
    Projected,
    UniformGrid,
    VerticalBoundary,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from tests.support import jax_oracle  # noqa: E402
from jaxwind._jax.discretization import (  # noqa: E402
    VerticalFaceField,
    build_discretization,
)
from jaxwind.operators import VelocityVector  # noqa: E402
from jaxwind.physics import (  # noqa: E402
    BladeElementActuatorLine,
    WindTunnelModel,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    args = parser.parse_args()
    dtype = getattr(jnp, args.dtype)
    if jax.device_count() != args.devices:
        raise RuntimeError(
            f"expected {args.devices} devices, found {jax.device_count()}"
        )

    grid = UniformGrid(4, 3, 8, 4.0, 3.0, 4.0)
    decomposition = EqualVerticalPartition(
        grid,
        MeshTopology((MeshAxis("z", args.devices),)),
        DistributionSpec.vertical(),
    )
    z = jnp.arange(grid.nz, dtype=dtype)[:, None, None]
    y = jnp.arange(grid.ny, dtype=dtype)[None, :, None]
    x = jnp.arange(grid.nx, dtype=dtype)[None, None, :]
    global_pressure = jnp.sin(0.23 * z + 0.17 * y + 0.11 * x)
    reference_pressure = Field(
        PressureCorrection,
        Cell,
        GlobalTestRegion(grid, Cell),
        Evaluated,
        global_pressure,
    )
    boundary = VerticalBoundary(dtype(0.25), dtype(-0.125))
    reference_gradient = jax_oracle.pressure_gradient_z(
        reference_pressure,
        boundary,
    )
    reference_laplacian = jax_oracle.divergence_z(reference_gradient)

    local_z = decomposition.cells_per_partition
    sharded_pressure = global_pressure.reshape(
        args.devices,
        local_z,
        grid.ny,
        grid.nx,
    )
    addressable_pressure = AddressableField(
        PressureCorrection,
        Cell,
        decomposition.regions(Cell),
        Evaluated,
        sharded_pressure,
    )
    interpreter = build_discretization(
        decomposition,
        addressable_partitions=(None if args.devices == 1 else tuple(range(args.devices))),
    )
    distributed_gradient = interpreter.pressure_gradient_z(
        addressable_pressure,
        boundary,
    )
    distributed_laplacian = interpreter.divergence_z(distributed_gradient)

    expected_gradient = reference_gradient.payload[1:].reshape(
        args.devices,
        local_z,
        grid.ny,
        grid.nx,
    )
    expected_laplacian = reference_laplacian.payload.reshape(
        args.devices,
        local_z,
        grid.ny,
        grid.nx,
    )

    z_face = jnp.arange(grid.nz + 1, dtype=dtype)[:, None, None]
    global_u = 2.0 + 0.02 * x + 0.03 * y + 0.01 * z
    global_v = -0.1 + 0.01 * x - 0.02 * y + 0.0 * z
    global_w = 0.01 * x + 0.02 * y + 0.015 * z_face
    reference_velocity = VelocityVector(
        Field(
            XVelocity,
            Cell,
            GlobalTestRegion(grid, Cell),
            Projected,
            global_u,
        ),
        Field(
            YVelocity,
            Cell,
            GlobalTestRegion(grid, Cell),
            Projected,
            global_v,
        ),
        Field(
            VerticalVelocity,
            ZFace,
            GlobalTestRegion(grid, ZFace),
            Projected,
            global_w,
        ),
    )
    distributed_velocity = VelocityVector(
        AddressableField(
            XVelocity,
            Cell,
            decomposition.regions(Cell),
            Projected,
            global_u.reshape(
                args.devices,
                local_z,
                grid.ny,
                grid.nx,
            ),
        ),
        AddressableField(
            YVelocity,
            Cell,
            decomposition.regions(Cell),
            Projected,
            global_v.reshape(
                args.devices,
                local_z,
                grid.ny,
                grid.nx,
            ),
        ),
        VerticalFaceField(
            AddressableField(
                VerticalVelocity,
                ZFace,
                decomposition.regions(ZFace),
                Projected,
                global_w[1:].reshape(
                    args.devices,
                    local_z,
                    grid.ny,
                    grid.nx,
                ),
            ),
            global_w[0],
        ),
    )
    aligned_velocity = interpreter.enforce_normal_boundary(
        distributed_velocity,
        VerticalBoundary(dtype(0.0), dtype(0.0)),
    )
    aligned_spectrum = jnp.fft.rfftn(
        distributed_velocity.x.payload,
        axes=(-2, -1),
    ).at[..., -1].set(0.0)
    expected_aligned_u = jnp.fft.irfftn(
        aligned_spectrum,
        s=(grid.ny, grid.nx),
        axes=(-2, -1),
    )
    line = BladeElementActuatorLine(
        x=2.0,
        y=1.5,
        z=2.0,
        blade_count=3,
        hub_radius=0.25,
        tip_radius=1.25,
        angular_velocity=1.5,
        smoothing_width=0.45,
        element_radii=(0.25, 0.75, 1.25),
        element_widths=(0.25, 0.5, 0.25),
        element_chords=(0.2, 0.15, 0.1),
        element_twist_degrees=(8.0, 4.0, 0.0),
        element_airfoil_ids=(0, 0, 0),
        polar_alpha_degrees=(-180.0, 0.0, 180.0),
        polar_lift_coefficients=((0.0, 0.0, 0.0),),
        polar_drag_coefficients=((0.1, 0.1, 0.1),),
        tip_loss=False,
        root_loss=False,
    )
    line_model = WindTunnelModel(actuator_line=line)
    reference_line = jax_oracle.JaxOracleProjection().wind_tunnel_tendency(
        reference_velocity,
        line_model,
        None,
    )
    distributed_line = interpreter.wind_tunnel_tendency(
        distributed_velocity,
        line_model,
        None,
    )
    line_errors = (
        jnp.max(
            jnp.abs(
                distributed_line.x.payload
                - reference_line.x.payload.reshape(
                    args.devices,
                    local_z,
                    grid.ny,
                    grid.nx,
                )
            )
        ),
        jnp.max(
            jnp.abs(
                distributed_line.y.payload
                - reference_line.y.payload.reshape(
                    args.devices,
                    local_z,
                    grid.ny,
                    grid.nx,
                )
            )
        ),
        jnp.max(
            jnp.abs(
                distributed_line.z.owned.payload
                - reference_line.z.payload[1:].reshape(
                    args.devices,
                    local_z,
                    grid.ny,
                    grid.nx,
                )
            )
        ),
    )

    packed = jnp.stack((sharded_pressure, 2.0 * sharded_pressure), axis=1)
    halo = interpreter.projection.exchange_packed(packed)
    halo.lower.block_until_ready()
    repeated_halo = interpreter.projection.exchange_packed(packed)
    expected_lower = jnp.zeros_like(halo.lower)
    expected_upper = jnp.zeros_like(halo.upper)
    expected_lower = expected_lower.at[1:].set(packed[:-1, :, -1])
    expected_upper = expected_upper.at[:-1].set(packed[1:, :, 0])

    result = {
        "gradient_error": float(
            jnp.max(jnp.abs(distributed_gradient.owned.payload - expected_gradient))
        ),
        "divergence_error": float(
            jnp.max(jnp.abs(distributed_laplacian.payload - expected_laplacian))
        ),
        "actuator_line_error": float(jnp.max(jnp.asarray(line_errors))),
        "actuator_line_component_errors": [float(value) for value in line_errors],
        "lower_halo_error": float(jnp.max(jnp.abs(halo.lower - expected_lower))),
        "upper_halo_error": float(jnp.max(jnp.abs(halo.upper - expected_upper))),
        "halo_shape_stable": (
            repeated_halo.lower.shape == halo.lower.shape
            and repeated_halo.upper.shape == halo.upper.shape
        ),
        "halo_elements_per_shard": int(halo.lower[0].size + halo.upper[0].size),
        "declared_halo_elements_per_shard": (
            interpreter.halo_context_elements_per_shard(packed.shape[1])
        ),
        "communicated_elements": [
            interpreter.halo_communicated_elements(packed.shape[1], shard)
            for shard in range(args.devices)
        ],
        "lower_flags": [
            bool(value) for value in jax.device_get(halo.lower_is_physical)
        ],
        "upper_flags": [
            bool(value) for value in jax.device_get(halo.upper_is_physical)
        ],
        "extract_identity": distributed_gradient.extract_owned()
        is distributed_gradient.owned,
        "state_bandwidth_error": float(
            jnp.max(jnp.abs(aligned_velocity.x.payload - expected_aligned_u))
        ),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
