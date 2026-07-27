from __future__ import annotations

import argparse
import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from wireles.domain import (  # noqa: E402
    AddressableField,
    Cell,
    DistributionSpec,
    EqualZSlab,
    Evaluated,
    Field,
    GlobalTestRegion,
    MeshAxis,
    MeshTopology,
    PressureCorrection,
    UniformGrid,
    VerticalBoundary,
)
from wireles.interpreters import jax_reference  # noqa: E402
from wireles.interpreters.jax_zslab import build_zslab_interpreter  # noqa: E402


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
    decomposition = EqualZSlab(
        grid,
        MeshTopology((MeshAxis("z", args.devices),)),
        DistributionSpec.z_slab(),
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
    reference_gradient = jax_reference.pressure_gradient_z(
        reference_pressure,
        boundary,
    )
    reference_laplacian = jax_reference.divergence_z(reference_gradient)

    local_z = decomposition.cells_per_shard
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
    interpreter = build_zslab_interpreter(
        decomposition,
        addressable_shards=tuple(range(args.devices)),
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

    packed = jnp.stack((sharded_pressure, 2.0 * sharded_pressure), axis=1)
    halo = interpreter.exchange_packed(packed)
    halo.lower.block_until_ready()
    repeated_halo = interpreter.exchange_packed(packed)
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
        "lower_flags": [bool(value) for value in jax.device_get(halo.lower_is_physical)],
        "upper_flags": [bool(value) for value in jax.device_get(halo.upper_is_physical)],
        "extract_identity": distributed_gradient.extract_owned()
        is distributed_gradient.owned,
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
