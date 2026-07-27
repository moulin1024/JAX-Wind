from __future__ import annotations

import argparse
import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from spectral_fd import runtime_from_initialized_jax  # noqa: E402

from wireles.domain import (  # noqa: E402
    AddressableField,
    Candidate,
    Cell,
    DistributionSpec,
    EqualZSlab,
    Field,
    GlobalTestRegion,
    MeshAxis,
    MeshTopology,
    UniformGrid,
    VerticalBoundary,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from wireles.interpreters.jax_reference import (  # noqa: E402
    JaxReferencePressureSolver,
    JaxReferenceProjection,
)
from wireles.interpreters.jax_zslab import (  # noqa: E402
    ZFaceFieldContext,
    build_zslab_interpreter,
)
from wireles.operators import VelocityVector, project  # noqa: E402
from wireles.pressure import build_spectral_fd_pressure_adapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument(
        "--method",
        choices=("transpose", "spike", "spike-adaptive"),
        default="transpose",
    )
    args = parser.parse_args()
    if jax.device_count() != args.devices:
        raise RuntimeError(f"expected {args.devices} devices, found {jax.device_count()}")
    dtype = getattr(jnp, args.dtype)
    grid = UniformGrid(4, 4, 8, 4.0, 4.0, 8.0)
    cells = GlobalTestRegion(grid, Cell)
    faces = GlobalTestRegion(grid, ZFace)
    z = jnp.arange(grid.nz, dtype=dtype)[:, None, None]
    zf = jnp.arange(grid.nz + 1, dtype=dtype)[:, None, None]
    y = jnp.arange(grid.ny, dtype=dtype)[None, :, None]
    x = jnp.arange(grid.nx, dtype=dtype)[None, None, :]
    global_velocity = VelocityVector(
        Field(
            XVelocity,
            Cell,
            cells,
            Candidate,
            jnp.sin(0.7 * x + 0.2 * y + 0.3 * z),
        ),
        Field(
            YVelocity,
            Cell,
            cells,
            Candidate,
            jnp.cos(0.4 * x + 0.6 * y - 0.2 * z),
        ),
        Field(
            VerticalVelocity,
            ZFace,
            faces,
            Candidate,
            jnp.sin(0.5 * zf + 0.3 * y + 0.2 * x),
        ),
    )
    boundary = VerticalBoundary(dtype(0.0), dtype(0.0))
    reference = project(
        global_velocity,
        dt=0.2,
        normal_boundary=boundary,
        algebra=JaxReferenceProjection(),
        pressure_solver=JaxReferencePressureSolver(),
    )

    decomposition = EqualZSlab(
        grid,
        MeshTopology((MeshAxis("z", args.devices),)),
        DistributionSpec.z_slab(),
    )
    shards = tuple(range(args.devices))
    local_z = decomposition.cells_per_shard
    interpreter = build_zslab_interpreter(
        decomposition,
        addressable_shards=shards,
    )
    velocity = VelocityVector(
        AddressableField(
            XVelocity,
            Cell,
            decomposition.regions(Cell),
            Candidate,
            global_velocity.x.payload.reshape(args.devices, local_z, 4, 4),
        ),
        AddressableField(
            YVelocity,
            Cell,
            decomposition.regions(Cell),
            Candidate,
            global_velocity.y.payload.reshape(args.devices, local_z, 4, 4),
        ),
        ZFaceFieldContext(
            AddressableField(
                VerticalVelocity,
                ZFace,
                decomposition.regions(ZFace),
                Candidate,
                global_velocity.z.payload[1:].reshape(args.devices, local_z, 4, 4),
            ),
            global_velocity.z.payload[0],
        ),
    )
    pressure_solver = build_spectral_fd_pressure_adapter(
        decomposition,
        addressable_shards=shards,
        runtime=runtime_from_initialized_jax(jax),
        dtype=args.dtype,
        method=args.method,
    )
    production = project(
        velocity,
        dt=0.2,
        normal_boundary=boundary,
        algebra=interpreter,
        pressure_solver=pressure_solver,
    )
    repeated = project(
        production.velocity,
        dt=0.2,
        normal_boundary=boundary,
        algebra=interpreter,
        pressure_solver=pressure_solver,
    )

    def error(actual, expected):
        return float(jnp.max(jnp.abs(actual - expected)))

    expected_shape = (args.devices, local_z, grid.ny, grid.nx)
    result = {
        "pressure_error": error(
            production.pressure.payload,
            reference.pressure.payload.reshape(expected_shape),
        ),
        "x_error": error(
            production.velocity.x.payload,
            reference.velocity.x.payload.reshape(expected_shape),
        ),
        "y_error": error(
            production.velocity.y.payload,
            reference.velocity.y.payload.reshape(expected_shape),
        ),
        "z_error": error(
            production.velocity.z.owned.payload,
            reference.velocity.z.payload[1:].reshape(expected_shape),
        ),
        "divergence": float(jnp.max(jnp.abs(production.divergence.payload))),
        "pressure_mean": float(jnp.abs(jnp.mean(production.pressure.payload))),
        "idempotence": max(
            error(repeated.velocity.x.payload, production.velocity.x.payload),
            error(repeated.velocity.y.payload, production.velocity.y.payload),
            error(
                repeated.velocity.z.owned.payload,
                production.velocity.z.owned.payload,
            ),
        ),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
