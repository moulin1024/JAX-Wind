from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from spectral_fd import runtime_from_initialized_jax  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    AcceptedClock,
    AddressableField,
    Cell,
    DistributionSpec,
    EqualZSlab,
    Evaluated,
    Field,
    GlobalTestRegion,
    MeshAxis,
    MeshTopology,
    Projected,
    UniformGrid,
    VerticalBoundary,
    VerticalVelocity,
    VerticalVelocityTendency,
    XVelocity,
    XVelocityTendency,
    YVelocity,
    YVelocityTendency,
    ZFace,
)
from jaxwind.effects import (  # noqa: E402
    ZSlabCheckpointLayout,
    load_ab2_checkpoint,
    save_ab2_checkpoint,
)
from jaxwind.integrators import (  # noqa: E402
    AB2Config,
    VectorFieldResult,
    cold_start,
    step,
)
from jaxwind.interpreters.jax_reference import (  # noqa: E402
    JaxReferencePressureSolver,
    JaxReferenceProjection,
)
from jaxwind.interpreters.jax_zslab import (  # noqa: E402
    ZFaceFieldContext,
    build_zslab_interpreter,
)
from jaxwind.operators import VelocityVector  # noqa: E402
from jaxwind.pressure import build_spectral_fd_pressure_adapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    args = parser.parse_args()
    if jax.device_count() != args.devices:
        raise RuntimeError(f"expected {args.devices} devices, found {jax.device_count()}")
    dtype = getattr(jnp, args.dtype)
    grid = UniformGrid(4, 4, 8, 4.0, 4.0, 8.0)
    config = AB2Config(0.05)

    def boundary_law(_clock, _environment):
        return VerticalBoundary(0.0, 0.0)

    cells = GlobalTestRegion(grid, Cell)
    faces = GlobalTestRegion(grid, ZFace)
    reference_velocity = VelocityVector(
        Field(XVelocity, Cell, cells, Projected, jnp.zeros(cells.storage_shape, dtype)),
        Field(YVelocity, Cell, cells, Projected, jnp.zeros(cells.storage_shape, dtype)),
        Field(
            VerticalVelocity,
            ZFace,
            faces,
            Projected,
            jnp.zeros(faces.storage_shape, dtype),
        ),
    )
    reference_state = cold_start(
        reference_velocity,
        clock=AcceptedClock(0.0, 0),
        config=config,
    )
    reference_z = jnp.arange(grid.nz, dtype=dtype)[:, None, None]
    reference_zf = jnp.arange(grid.nz + 1, dtype=dtype)[:, None, None]
    reference_y = jnp.arange(grid.ny, dtype=dtype)[None, :, None]
    reference_x = jnp.arange(grid.nx, dtype=dtype)[None, None, :]

    def reference_vector_field(evaluation):
        time = evaluation.time.time
        tendency = VelocityVector(
            Field(
                XVelocityTendency,
                Cell,
                cells,
                Evaluated,
                jnp.sin(
                    0.17 * reference_x
                    + 0.13 * reference_y
                    + 0.11 * reference_z
                    + time
                ),
            ),
            Field(
                YVelocityTendency,
                Cell,
                cells,
                Evaluated,
                jnp.cos(
                    0.19 * reference_x
                    - 0.07 * reference_y
                    + 0.05 * reference_z
                    - 0.5 * time
                ),
            ),
            Field(
                VerticalVelocityTendency,
                ZFace,
                faces,
                Evaluated,
                jnp.sin(
                    0.09 * reference_x
                    + 0.15 * reference_y
                    + 0.12 * reference_zf
                    + 0.25 * time
                ),
            ),
        )
        return VectorFieldResult(tendency, time)

    decomposition = EqualZSlab(
        grid,
        MeshTopology((MeshAxis("z", args.devices),)),
        DistributionSpec.z_slab(),
    )
    shards = tuple(range(args.devices))
    cell_regions = decomposition.regions(Cell)
    face_regions = decomposition.regions(ZFace)
    local_z = decomposition.cells_per_shard
    production_velocity = VelocityVector(
        AddressableField(
            XVelocity,
            Cell,
            cell_regions,
            Projected,
            jnp.zeros((args.devices, local_z, grid.ny, grid.nx), dtype),
        ),
        AddressableField(
            YVelocity,
            Cell,
            cell_regions,
            Projected,
            jnp.zeros((args.devices, local_z, grid.ny, grid.nx), dtype),
        ),
        ZFaceFieldContext(
            AddressableField(
                VerticalVelocity,
                ZFace,
                face_regions,
                Projected,
                jnp.zeros((args.devices, local_z, grid.ny, grid.nx), dtype),
            ),
            jnp.zeros((grid.ny, grid.nx), dtype),
        ),
    )
    production_state = cold_start(
        production_velocity,
        clock=AcceptedClock(0.0, 0),
        config=config,
    )
    local_cell_z = jnp.arange(grid.nz, dtype=dtype).reshape(
        args.devices,
        local_z,
        1,
        1,
    )
    local_face_z = jnp.arange(1, grid.nz + 1, dtype=dtype).reshape(
        args.devices,
        local_z,
        1,
        1,
    )
    local_y = jnp.arange(grid.ny, dtype=dtype)[None, None, :, None]
    local_x = jnp.arange(grid.nx, dtype=dtype)[None, None, None, :]

    def production_vector_field(evaluation):
        time = evaluation.time.time
        tendency = VelocityVector(
            AddressableField(
                XVelocityTendency,
                Cell,
                cell_regions,
                Evaluated,
                jnp.sin(
                    0.17 * local_x
                    + 0.13 * local_y
                    + 0.11 * local_cell_z
                    + time
                ),
            ),
            AddressableField(
                YVelocityTendency,
                Cell,
                cell_regions,
                Evaluated,
                jnp.cos(
                    0.19 * local_x
                    - 0.07 * local_y
                    + 0.05 * local_cell_z
                    - 0.5 * time
                ),
            ),
            ZFaceFieldContext(
                AddressableField(
                    VerticalVelocityTendency,
                    ZFace,
                    face_regions,
                    Evaluated,
                    jnp.sin(
                        0.09 * local_x
                        + 0.15 * local_y
                        + 0.12 * local_face_z
                        + 0.25 * time
                    ),
                ),
                jnp.sin(
                    0.09 * local_x[0, 0]
                    + 0.15 * local_y[0, 0]
                    + 0.25 * time
                ),
            ),
        )
        return VectorFieldResult(tendency, time)

    reference_algebra = JaxReferenceProjection()
    reference_solver = JaxReferencePressureSolver()
    production_algebra = build_zslab_interpreter(
        decomposition,
        addressable_shards=shards,
    )
    production_solver = build_spectral_fd_pressure_adapter(
        decomposition,
        addressable_shards=shards,
        runtime=runtime_from_initialized_jax(jax),
        dtype=args.dtype,
        method="transpose",
    )

    def advance_reference(state):
        return step(
            state,
            config=config,
            environment=None,
            vector_field=reference_vector_field,
            normal_boundary=boundary_law,
            algebra=reference_algebra,
            pressure_solver=reference_solver,
        )

    def advance_production(state):
        return step(
            state,
            config=config,
            environment=None,
            vector_field=production_vector_field,
            normal_boundary=boundary_law,
            algebra=production_algebra,
            pressure_solver=production_solver,
        )

    max_velocity_error = 0.0
    max_history_error = 0.0
    max_divergence = 0.0
    evaluation_times = []
    for _ in range(4):
        reference_result = advance_reference(reference_state)
        production_result = advance_production(production_state)
        reference_state = reference_result.state
        production_state = production_result.state
        evaluation_times.append(production_result.diagnostic.evaluation_time.time)
        shape = (args.devices, local_z, grid.ny, grid.nx)
        errors = (
            jnp.max(
                jnp.abs(
                    production_state.velocity.x.payload
                    - reference_state.velocity.x.payload.reshape(shape)
                )
            ),
            jnp.max(
                jnp.abs(
                    production_state.velocity.y.payload
                    - reference_state.velocity.y.payload.reshape(shape)
                )
            ),
            jnp.max(
                jnp.abs(
                    production_state.velocity.z.owned.payload
                    - reference_state.velocity.z.payload[1:].reshape(shape)
                )
            ),
        )
        max_velocity_error = max(max_velocity_error, *(float(value) for value in errors))
        history_error = jnp.max(
            jnp.abs(
                production_state.history.value.x.payload
                - reference_state.history.value.x.payload.reshape(shape)
            )
        )
        max_history_error = max(max_history_error, float(history_error))
        max_divergence = max(
            max_divergence,
            float(jnp.max(jnp.abs(production_result.diagnostic.projection.divergence.payload))),
        )

    checkpoint_base = cold_start(
        production_velocity,
        clock=AcceptedClock(0.0, 0),
        config=config,
    )
    continuous = checkpoint_base
    for _ in range(4):
        continuous = advance_production(continuous).state
    interrupted = checkpoint_base
    for _ in range(2):
        interrupted = advance_production(interrupted).state
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rank0000.npz"
        save_ab2_checkpoint(path, interrupted)
        restarted = load_ab2_checkpoint(
            path,
            layout=ZSlabCheckpointLayout(decomposition, shards, jnp.asarray),
            config=config,
        )
    for _ in range(2):
        restarted = advance_production(restarted).state
    restart_error = max(
        float(jnp.max(jnp.abs(restarted.velocity.x.payload - continuous.velocity.x.payload))),
        float(jnp.max(jnp.abs(restarted.velocity.y.payload - continuous.velocity.y.payload))),
        float(
            jnp.max(
                jnp.abs(
                    restarted.velocity.z.owned.payload
                    - continuous.velocity.z.owned.payload
                )
            )
        ),
        float(
            jnp.max(
                jnp.abs(
                    restarted.history.value.x.payload
                    - continuous.history.value.x.payload
                )
            )
        ),
    )
    result = {
        "clock": [production_state.clock.time, production_state.clock.step],
        "divergence": max_divergence,
        "dtype_preserved": (
            str(production_state.velocity.x.payload.dtype) == args.dtype
            and str(production_state.history.value.x.payload.dtype) == args.dtype
        ),
        "evaluation_times": evaluation_times,
        "history_error": max_history_error,
        "restart_error": restart_error,
        "velocity_error": max_velocity_error,
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
