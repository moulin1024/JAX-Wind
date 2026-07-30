from __future__ import annotations

from collections.abc import Callable
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from .config import Params
from .timestep_sharded import (
    ShardedFlowState,
    ShardedOperators,
    make_step_ab2_sharded,
)


INFLOW_BATCH_FORMAT = "wireles-jax-inflow-batches-v1"
INFLOW_FIELDS = ("u", "v", "w", "theta", "qv")


def fringe_start_index(params: Params) -> int:
    x = (np.arange(params.nx, dtype=np.float64) + 0.5) * params.dx * params.z_i
    start = int(np.searchsorted(x, params.fringe_start_x))
    if start >= params.nx:
        raise ValueError("Inflow fringe contains no cell centres")
    return start


def _inflow_snapshot(state: ShardedFlowState, start: int) -> jax.Array:
    return jnp.stack(
        tuple(getattr(state, name)[start:] for name in INFLOW_FIELDS),
        axis=0,
    )


def make_precursor_inflow_batch(
    params: Params,
    ops: ShardedOperators,
    mesh,
    *,
    batch_steps: int,
    axis_name: str = "z",
) -> Callable[..., tuple[ShardedFlowState, jax.Array]]:
    """Advance a precursor and return pre-step fringe targets.

    The returned layout is ``(time, field, x_fringe, y, z)``. Sampling the
    carry before each step matches the target timing used by the concurrent
    precursor pipeline.
    """

    if batch_steps <= 0:
        raise ValueError("batch_steps must be positive")
    start = fringe_start_index(params)
    step = make_step_ab2_sharded(params, ops, mesh, axis_name)

    def advance(
        state: ShardedFlowState,
        runtime_pressure_ops,
        runtime_spike_ops,
    ) -> tuple[ShardedFlowState, jax.Array]:
        def body(carry: ShardedFlowState, _):
            snapshot = _inflow_snapshot(carry, start)
            next_state = step(
                carry,
                runtime_pressure_ops,
                runtime_spike_ops,
            )
            return next_state, snapshot

        return lax.scan(body, state, xs=None, length=batch_steps)

    return advance


def inflow_batch_filename(batch_id: int, rank: int) -> str:
    if batch_id < 0 or rank < 0:
        raise ValueError("batch_id and rank must be nonnegative")
    return f"batch_{batch_id:06d}_rank_{rank:05d}.npz"


def write_local_inflow_batch(
    directory: str | Path,
    packed: np.ndarray,
    *,
    batch_id: int,
    rank: int,
    global_start_step: int,
    z_start: int,
    z_stop: int,
    compress: bool = False,
) -> Path:
    """Write one rank-local, time-bounded inflow batch."""

    data = np.ascontiguousarray(packed)
    if data.ndim != 5 or data.shape[1] != len(INFLOW_FIELDS):
        raise ValueError(
            "Expected local inflow layout "
            f"(time,{len(INFLOW_FIELDS)},x_fringe,y,z_local), got {data.shape}"
        )
    if data.shape[0] <= 0:
        raise ValueError("An inflow batch must contain at least one timestep")
    if z_start < 0 or z_stop <= z_start or z_stop - z_start != data.shape[-1]:
        raise ValueError(
            f"Invalid z ownership [{z_start}, {z_stop}) for {data.shape[-1]} planes"
        )
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / inflow_batch_filename(batch_id, rank)
    payload = {
        "inflow": data,
        "batch_id": np.asarray(batch_id, dtype=np.int64),
        "rank": np.asarray(rank, dtype=np.int64),
        "global_start_step": np.asarray(global_start_step, dtype=np.int64),
        "step_count": np.asarray(data.shape[0], dtype=np.int64),
        "z_start": np.asarray(z_start, dtype=np.int64),
        "z_stop": np.asarray(z_stop, dtype=np.int64),
    }
    writer = np.savez_compressed if compress else np.savez
    writer(path, **payload)
    return path


def build_inflow_manifest(
    params: Params,
    *,
    source_parts: int,
    start_step: int,
    total_steps: int,
    batch_steps: int,
    compress: bool,
) -> dict[str, Any]:
    if source_parts <= 0:
        raise ValueError("source_parts must be positive")
    if total_steps <= 0 or batch_steps <= 0:
        raise ValueError("total_steps and batch_steps must be positive")
    if start_step < 0:
        raise ValueError("start_step must be nonnegative")
    if params.nz % source_parts:
        raise ValueError(
            f"nz={params.nz} must be divisible by source_parts={source_parts}"
        )
    start = fringe_start_index(params)
    nx_fringe = params.nx - start
    local_z = params.nz // source_parts
    itemsize = np.dtype(params.dtype).itemsize
    batches = []
    for batch_id, offset in enumerate(range(0, total_steps, batch_steps)):
        steps = min(batch_steps, total_steps - offset)
        batches.append(
            {
                "batch_id": batch_id,
                "step_offset": offset,
                "global_start_step": start_step + offset,
                "step_count": steps,
                "file_pattern": inflow_batch_filename(batch_id, 0).replace(
                    "rank_00000", "rank_{rank:05d}"
                ),
                "uncompressed_bytes_per_rank": (
                    steps
                    * len(INFLOW_FIELDS)
                    * nx_fringe
                    * params.ny
                    * local_z
                    * itemsize
                ),
            }
        )
    return {
        "format": INFLOW_BATCH_FORMAT,
        "fields": list(INFLOW_FIELDS),
        "layout": ["time", "field", "x_fringe", "y", "z_local"],
        "dtype": np.dtype(params.dtype).name,
        "source_parts": source_parts,
        "global_shape_per_step": [
            len(INFLOW_FIELDS),
            nx_fringe,
            params.ny,
            params.nz,
        ],
        "local_z_planes": local_z,
        "fringe_start_x_m": params.fringe_start_x,
        "fringe_start_index": start,
        "dt_seconds": params.dt_physical,
        "start_step": start_step,
        "total_steps": total_steps,
        "end_step": start_step + total_steps,
        "batch_steps": batch_steps,
        "batch_count": len(batches),
        "compressed": compress,
        "snapshot_semantics": "pre-step state, one target per solver timestep",
        "completion_semantics": (
            "manifest.json is written only after every rank-local batch and "
            "the final precursor checkpoint are complete"
        ),
        "batches": batches,
    }


def write_inflow_manifest(
    directory: str | Path,
    manifest: dict[str, Any],
) -> Path:
    validate_inflow_manifest(manifest)
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    return path


def read_inflow_manifest(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "manifest.json"
    manifest = json.loads(path.read_text())
    validate_inflow_manifest(manifest)
    return manifest


def validate_inflow_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("format") != INFLOW_BATCH_FORMAT:
        raise ValueError(f"Unsupported inflow batch format: {manifest.get('format')!r}")
    if tuple(manifest.get("fields", ())) != INFLOW_FIELDS:
        raise ValueError("Inflow field order does not match the solver contract")
    if manifest.get("layout") != [
        "time",
        "field",
        "x_fringe",
        "y",
        "z_local",
    ]:
        raise ValueError("Inflow layout does not match the solver contract")
    source_parts = int(manifest["source_parts"])
    total_steps = int(manifest["total_steps"])
    batch_steps = int(manifest["batch_steps"])
    batches = manifest.get("batches", [])
    if source_parts <= 0 or total_steps <= 0 or batch_steps <= 0:
        raise ValueError("Invalid inflow manifest dimensions")
    if len(batches) != math.ceil(total_steps / batch_steps):
        raise ValueError("Inflow manifest batch count is inconsistent")
    if int(manifest["batch_count"]) != len(batches):
        raise ValueError("Inflow manifest batch_count is inconsistent")
    shape = tuple(map(int, manifest["global_shape_per_step"]))
    if len(shape) != 4 or shape[0] != len(INFLOW_FIELDS):
        raise ValueError("Invalid global inflow shape")
    local_z = int(manifest["local_z_planes"])
    if local_z <= 0 or shape[-1] != source_parts * local_z:
        raise ValueError("Inflow z decomposition is inconsistent")
    covered = 0
    for expected_id, batch in enumerate(batches):
        if int(batch["batch_id"]) != expected_id:
            raise ValueError("Inflow batch ids must be contiguous")
        if int(batch["step_offset"]) != covered:
            raise ValueError("Inflow batch step offsets must be contiguous")
        count = int(batch["step_count"])
        if count <= 0 or count > batch_steps:
            raise ValueError("Invalid inflow batch step count")
        covered += count
    if covered != total_steps:
        raise ValueError("Inflow batches do not cover total_steps")


def load_local_inflow_batch(
    directory: str | Path,
    manifest: dict[str, Any],
    *,
    batch_id: int,
    rank: int,
) -> np.ndarray:
    """Load and validate one batch shard for a later enforced-inflow run."""

    validate_inflow_manifest(manifest)
    batches = manifest["batches"]
    source_parts = int(manifest["source_parts"])
    if not 0 <= batch_id < len(batches):
        raise ValueError(f"batch_id={batch_id} is outside the manifest")
    if not 0 <= rank < source_parts:
        raise ValueError(f"rank={rank} is outside [0, {source_parts})")
    batch = batches[batch_id]
    path = Path(directory) / inflow_batch_filename(batch_id, rank)
    with np.load(path, allow_pickle=False) as archive:
        data = np.asarray(archive["inflow"])
        metadata = {
            key: int(np.asarray(archive[key]))
            for key in (
                "batch_id",
                "rank",
                "global_start_step",
                "step_count",
                "z_start",
                "z_stop",
            )
        }
    expected_shape = (
        int(batch["step_count"]),
        *map(int, manifest["global_shape_per_step"][:-1]),
        int(manifest["local_z_planes"]),
    )
    if data.shape != expected_shape:
        raise ValueError(
            f"Batch {batch_id} rank {rank} has shape {data.shape}, "
            f"expected {expected_shape}"
        )
    expected_metadata = {
        "batch_id": batch_id,
        "rank": rank,
        "global_start_step": int(batch["global_start_step"]),
        "step_count": int(batch["step_count"]),
        "z_start": rank * int(manifest["local_z_planes"]),
        "z_stop": (rank + 1) * int(manifest["local_z_planes"]),
    }
    if metadata != expected_metadata:
        raise ValueError(
            f"Batch {batch_id} rank {rank} metadata mismatch: "
            f"{metadata} != {expected_metadata}"
        )
    if data.dtype != np.dtype(manifest["dtype"]):
        raise ValueError(f"Batch dtype {data.dtype} does not match {manifest['dtype']}")
    return data
