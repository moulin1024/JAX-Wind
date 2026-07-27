from __future__ import annotations

import json
import hashlib
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .config import Params
from .sharding import make_array_from_local_callback, mesh_size, z_slab_sharding
from .timestep_sharded import ShardedFlowState


_ARRAY_FIELDS = tuple(name for name in ShardedFlowState._fields if name != "step")
_RESTART_PARAM_EXCLUSIONS = {"nsteps", "c_count", "use_jit"}


def _json_parameter(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return np.dtype(value).name
    except TypeError:
        return str(value)


def _restart_parameters(params: Params) -> dict[str, Any]:
    return {
        field.name: _json_parameter(getattr(params, field.name))
        for field in fields(params)
        if field.name not in _RESTART_PARAM_EXCLUSIONS
    }


def _restart_signature(parameters: dict[str, Any]) -> str:
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_restart_parameters(manifest: dict[str, Any], params: Params) -> None:
    saved = manifest.get("restart_parameters")
    if saved is None:
        return
    current = _restart_parameters(params)
    if saved == current:
        return
    differing = sorted(
        key for key in set(saved) | set(current) if saved.get(key) != current.get(key)
    )
    preview = ", ".join(differing[:8])
    suffix = "..." if len(differing) > 8 else ""
    raise ValueError(
        "Checkpoint numerical parameters do not match the resumed run: "
        f"{preview}{suffix}"
    )


def _one_local_shard(array: jax.Array) -> np.ndarray:
    shards = array.addressable_shards
    if len(shards) != 1:
        raise ValueError(
            "Concurrent-precursor checkpoints currently require exactly one "
            "addressable device per process."
        )
    return np.asarray(jax.device_get(shards[0].data))


def save_sharded_checkpoint(
    directory: str | Path,
    state: ShardedFlowState,
    params: Params,
    mesh: Mesh,
    *,
    rank: int | None = None,
) -> Path:
    """Write only this process's local z slab.

    One file is produced per process.  No field is gathered to a host and the
    manifest contains metadata only.
    """
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    rank = jax.process_index() if rank is None else rank
    payload = {name: _one_local_shard(getattr(state, name)) for name in _ARRAY_FIELDS}
    payload["step"] = np.asarray(jax.device_get(state.step))
    slab_path = output / f"rank_{rank:05d}.npz"
    np.savez_compressed(slab_path, **payload)

    if rank == 0:
        restart_parameters = _restart_parameters(params)
        manifest = {
            "format": "wireles-jax-zslab-v1",
            "complete_restart_state": True,
            "source_parts": mesh_size(mesh),
            "global_shape": [params.nx, params.ny, params.nz],
            "scalar_components": 2,
            "step": int(payload["step"]),
            "dtype": np.dtype(params.dtype).name,
            "sgs_dtype": np.dtype(params.sgs_dtype).name,
            "fields": list(_ARRAY_FIELDS),
            "field_dtypes": {
                name: str(payload[name].dtype) for name in _ARRAY_FIELDS
            },
            "restart_parameters": restart_parameters,
            "restart_signature_sha256": _restart_signature(restart_parameters),
            "state_semantics": {
                "time_integrator": "AB2 state includes previous RHS histories",
                "sgs": "LASD coefficients, histories, and Lagrangian accumulators included",
                "scalars": "scalar fields, RHS histories, and SGS coefficients included",
                "pressure": "projected velocity and pressure included",
            },
        }
        with (output / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    return slab_path


def read_sharded_checkpoint_manifest(directory: str | Path) -> dict[str, Any]:
    with (Path(directory) / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "wireles-jax-zslab-v1":
        raise ValueError(f"Unsupported sharded checkpoint format: {manifest.get('format')!r}")
    return manifest


def load_repartitioned_local_slabs(
    directory: str | Path,
    *,
    target_rank: int,
    target_parts: int,
) -> tuple[dict[str, np.ndarray], int, dict[str, Any]]:
    """Load the two adjacent old slabs forming one half-sized-mesh slab."""
    root = Path(directory)
    manifest = read_sharded_checkpoint_manifest(root)
    source_parts = int(manifest["source_parts"])
    if source_parts != 2 * target_parts:
        raise ValueError(
            f"Checkpoint has {source_parts} slabs; 4→2+2 repartition requires "
            f"exactly twice target_parts={target_parts}."
        )
    if not 0 <= target_rank < target_parts:
        raise ValueError(f"target_rank={target_rank} outside [0, {target_parts})")
    source_ranks = (2 * target_rank, 2 * target_rank + 1)
    archives = [np.load(root / f"rank_{rank:05d}.npz") for rank in source_ranks]
    try:
        step_values = [int(np.asarray(archive["step"])) for archive in archives]
        if step_values[0] != step_values[1]:
            raise ValueError(
                f"Adjacent checkpoint slabs have different steps: {step_values}"
            )
        fields = {
            name: np.concatenate([archive[name] for archive in archives], axis=2)
            for name in manifest["fields"]
        }
    finally:
        for archive in archives:
            archive.close()
    return fields, step_values[0], manifest


def load_repartitioned_sharded_checkpoint(
    directory: str | Path,
    params: Params,
    mesh: Mesh,
    *,
    target_rank: int,
    axis_name: str = "z",
) -> ShardedFlowState:
    """Restore a 2n-slab checkpoint onto either of two independent n meshes."""
    target_parts = mesh_size(mesh, axis_name)
    fields, step, manifest = load_repartitioned_local_slabs(
        directory, target_rank=target_rank, target_parts=target_parts
    )
    expected_shape = (params.nx, params.ny, params.nz)
    _validate_restart_parameters(manifest, params)
    if tuple(manifest["global_shape"]) != expected_shape:
        raise ValueError(
            f"Checkpoint shape {manifest['global_shape']} does not match {expected_shape}"
        )
    z_sharding = z_slab_sharding(mesh, axis_name)
    scalar_sharding = NamedSharding(mesh, P(None, None, axis_name, None))

    arrays: dict[str, jax.Array] = {}
    for name, local in fields.items():
        global_shape = expected_shape + ((2,) if name == "scalar_c" else ())
        sharding = scalar_sharding if name == "scalar_c" else z_sharding

        def callback(index, local_value=local):
            expected_local_shape = tuple(part.stop - part.start for part in index)
            if local_value.shape != expected_local_shape:
                raise ValueError(
                    f"Local checkpoint field {name} has {local_value.shape}; "
                    f"mesh callback expects {expected_local_shape}"
                )
            return local_value

        arrays[name] = make_array_from_local_callback(
            global_shape, sharding, callback, dtype=local.dtype
        )
    arrays["step"] = jax.device_put(
        jnp.asarray(step, dtype=jnp.int32), NamedSharding(mesh, P())
    )
    return ShardedFlowState(**arrays)


def load_sharded_checkpoint(
    directory: str | Path,
    params: Params,
    mesh: Mesh,
    *,
    rank: int | None = None,
    axis_name: str = "z",
) -> ShardedFlowState:
    """Restore a checkpoint onto a mesh with the same z partitioning.

    Every process reads only its own slab, so resuming a distributed run never
    materializes the global flow field on a host process.
    """
    root = Path(directory)
    manifest = read_sharded_checkpoint_manifest(root)
    _validate_restart_parameters(manifest, params)
    target_parts = mesh_size(mesh, axis_name)
    source_parts = int(manifest["source_parts"])
    if source_parts != target_parts:
        raise ValueError(
            f"Checkpoint has {source_parts} slabs but target mesh has "
            f"{target_parts}; use the repartitioned loader when changing mesh size."
        )
    expected_shape = (params.nx, params.ny, params.nz)
    if tuple(manifest["global_shape"]) != expected_shape:
        raise ValueError(
            f"Checkpoint shape {manifest['global_shape']} does not match {expected_shape}"
        )
    rank = jax.process_index() if rank is None else rank
    if not 0 <= rank < target_parts:
        raise ValueError(f"rank={rank} outside [0, {target_parts})")
    archive = np.load(root / f"rank_{rank:05d}.npz")
    try:
        step = int(np.asarray(archive["step"]))
        local_fields = {
            name: np.asarray(archive[name]) for name in manifest["fields"]
        }
    finally:
        archive.close()

    z_sharding = z_slab_sharding(mesh, axis_name)
    scalar_sharding = NamedSharding(mesh, P(None, None, axis_name, None))
    arrays: dict[str, jax.Array] = {}
    for name, local in local_fields.items():
        global_shape = expected_shape + ((2,) if name == "scalar_c" else ())
        sharding = scalar_sharding if name == "scalar_c" else z_sharding

        def callback(index, local_value=local, field_name=name):
            expected_local_shape = tuple(part.stop - part.start for part in index)
            if local_value.shape != expected_local_shape:
                raise ValueError(
                    f"Local checkpoint field {field_name} has {local_value.shape}; "
                    f"mesh callback expects {expected_local_shape}"
                )
            return local_value

        arrays[name] = make_array_from_local_callback(
            global_shape, sharding, callback, dtype=local.dtype
        )
    arrays["step"] = jax.device_put(
        jnp.asarray(step, dtype=jnp.int32), NamedSharding(mesh, P())
    )
    return ShardedFlowState(**arrays)


def pack_local_fringe_fields(state: ShardedFlowState) -> np.ndarray:
    """Pack the five carrier fields from one local slab for paired MPI transfer."""
    fields = tuple(_one_local_shard(getattr(state, name)) for name in ("u", "v", "w", "theta", "qv"))
    return np.ascontiguousarray(np.stack(fields, axis=0))


def fringe_start_index(params: Params) -> int:
    x = (np.arange(params.nx, dtype=np.float64) + 0.5) * params.dx * params.z_i
    index = int(np.searchsorted(x, params.fringe_start_x))
    if index >= params.nx:
        raise ValueError("Fringe start leaves no streamwise cells to exchange")
    return index


def local_fringe_snapshot(
    state: ShardedFlowState, params: Params
) -> tuple[jax.Array, ...]:
    """Capture device-local fringe slices without a host synchronization."""
    start = fringe_start_index(params)
    snapshots = []
    for name in ("u", "v", "w", "theta", "qv"):
        shards = getattr(state, name).addressable_shards
        if len(shards) != 1:
            raise ValueError("Concurrent pipeline requires one local device per MPI rank")
        snapshots.append(shards[0].data[start:, :, :])
    return tuple(snapshots)


def pack_local_fringe_chunk(
    snapshots: list[tuple[jax.Array, ...]],
) -> np.ndarray:
    """Stack K local device snapshots and perform one bulk device→host copy."""
    if not snapshots:
        raise ValueError("Cannot pack an empty precursor chunk")
    device_chunk = jnp.stack(
        [jnp.stack(snapshot, axis=0) for snapshot in snapshots], axis=0
    )
    return np.ascontiguousarray(jax.device_get(device_chunk))


def fringe_target_chunk_from_local_pack(
    packed: np.ndarray,
    params: Params,
    mesh: Mesh,
    *,
    axis_name: str = "z",
) -> jax.Array:
    """Move one received K-step fringe block to the distributed turbine mesh."""
    if packed.ndim != 5 or packed.shape[1] != 5:
        raise ValueError(
            "Expected packed fringe chunk (K,5,nx_fringe,ny,nz_local), "
            f"got {packed.shape}"
        )
    nx_fringe = params.nx - fringe_start_index(params)
    global_shape = (packed.shape[0], 5, nx_fringe, params.ny, params.nz)
    sharding = NamedSharding(mesh, P(None, None, None, None, axis_name))
    return make_array_from_local_callback(
        global_shape,
        sharding,
        lambda index: packed,
        dtype=packed.dtype,
    )


def fringe_target_from_local_pack(
    packed: np.ndarray,
    params: Params,
    mesh: Mesh,
    *,
    axis_name: str = "z",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if packed.shape[0] != 5:
        raise ValueError(f"Expected five packed fringe fields, got {packed.shape}")
    sharding = z_slab_sharding(mesh, axis_name)
    global_shape = (params.nx, params.ny, params.nz)
    arrays = []
    for field in packed:
        arrays.append(
            make_array_from_local_callback(
                global_shape,
                sharding,
                lambda index, value=field: value,
                dtype=field.dtype,
            )
        )
    return tuple(arrays)
