#!/usr/bin/env python3
"""Empirically tune the packed concurrent-fringe chunk on the target mesh."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "jax"))

from run_single import (  # noqa: E402
    RUN_DEFAULTS,
    load_config_file,
    params_from_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-time JAX device benchmark for the concurrent precursor "
            "compute/packed-ppermute balance."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", default="1,2,4,8,16,32,64")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--selection-slack", type=float, default=0.02)
    parser.add_argument("--coordinator-address", default="127.0.0.1:12671")
    parser.add_argument("--num-processes", type=int)
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--local-device-id", type=int, default=0)
    parser.add_argument(
        "--wait-for-pid",
        type=int,
        help="Wait for an existing local run to exit before initializing JAX.",
    )
    return parser.parse_args()


def _rank_and_size(args: argparse.Namespace) -> tuple[int, int]:
    rank = args.process_id
    size = args.num_processes
    if rank is None:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    if size is None:
        size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    return rank, size


def _chunk_candidates(text: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(value) for value in text.split(",")}))
    except ValueError as exc:
        raise SystemExit("--candidates must be comma-separated positive integers") from exc
    if not values or values[0] <= 0:
        raise SystemExit("--candidates must be comma-separated positive integers")
    return values


def _cost_analysis(compiled: Any) -> dict[str, float]:
    """Flatten the backend's optional per-device XLA cost estimate."""
    try:
        raw = compiled.cost_analysis()
    except (AttributeError, RuntimeError):
        return {}
    entries = raw if isinstance(raw, list) else [raw]
    result: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                result[str(key)] = result.get(str(key), 0.0) + float(value)
    return result


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    args = parse_args()
    rank, size = _rank_and_size(args)
    candidates = _chunk_candidates(args.candidates)
    if size < 4 or size % 2:
        raise SystemExit("Chunk autotuning requires an even rank count >= 4")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if not 0.0 <= args.selection_slack < 1.0:
        raise SystemExit("--selection-slack must lie in [0, 1)")
    if args.wait_for_pid is not None:
        if args.wait_for_pid <= 0:
            raise SystemExit("--wait-for-pid must be positive")
        if rank == 0:
            print(
                f"[autotune] waiting for pid={args.wait_for_pid} to exit",
                flush=True,
            )
        while True:
            try:
                os.kill(args.wait_for_pid, 0)
            except ProcessLookupError:
                break
            except PermissionError as exc:
                raise SystemExit(
                    f"Cannot inspect --wait-for-pid={args.wait_for_pid}"
                ) from exc
            time.sleep(5.0)

    settings = dict(RUN_DEFAULTS)
    settings.update(load_config_file(args.config))
    if settings["precision"] == "float64" or settings["sgs_precision"] == "float64":
        from jax import config as jax_config

        jax_config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=size,
        process_id=rank,
        local_device_ids=[args.local_device_id],
    )

    from wireles_jax.adjoint_sharded import (
        duplicate_state_for_adjoint,
        make_adjoint_chunk_step,
        make_empty_fringe_chunk,
        make_exchange_precursor_chunk,
    )
    from wireles_jax.checkpoint_sharded import load_sharded_checkpoint
    from wireles_jax.sharding import (
        make_adjoint_distributed_mesh,
        make_distributed_mesh,
    )
    from wireles_jax.timestep_sharded import make_sharded_operators

    configured = params_from_settings(settings, jnp)
    warmup_params = replace(
        configured,
        nsteps=0,
        actuator_disk_enabled=False,
        cold_source_enabled=False,
        fringe_enabled=False,
        horizontal_homogeneous=True,
        buoyancy_reference="plane_mean",
        sharded_pressure_solver="transpose",
    )
    concurrent_params = replace(
        configured,
        nsteps=0,
        horizontal_homogeneous=False,
        buoyancy_reference="ambient",
        sharded_pressure_solver="transpose",
    )

    warm_mesh = make_distributed_mesh(size)
    warm_state = load_sharded_checkpoint(
        args.checkpoint, warmup_params, warm_mesh, rank=rank
    )
    mesh = make_adjoint_distributed_mesh(size)
    base_state = duplicate_state_for_adjoint(warm_state, mesh)
    ops = make_sharded_operators(concurrent_params, mesh)
    exchange = jax.jit(make_exchange_precursor_chunk(mesh))
    rows: list[dict[str, Any]] = []

    for chunk_steps in candidates:
        empty = make_empty_fringe_chunk(concurrent_params, mesh, chunk_steps)
        prime = jax.jit(
            make_adjoint_chunk_step(
                concurrent_params,
                ops,
                mesh,
                chunk_steps=chunk_steps,
                advance_turbine=False,
            )
        )
        advance = jax.jit(
            make_adjoint_chunk_step(
                concurrent_params,
                ops,
                mesh,
                chunk_steps=chunk_steps,
                advance_turbine=True,
            )
        )

        compile_start = time.perf_counter()
        compiled_prime = prime.lower(
            base_state, empty, ops.pressure, ops.pressure_spike
        ).compile()
        primed_state, produced = compiled_prime(
            base_state, empty, ops.pressure, ops.pressure_spike
        )
        compiled_exchange = exchange.lower(produced).compile()
        targets = compiled_exchange(produced)
        jax.block_until_ready(targets)
        compiled_advance = advance.lower(
            primed_state, targets, ops.pressure, ops.pressure_spike
        ).compile()
        trial_state, trial_produced = compiled_advance(
            primed_state, targets, ops.pressure, ops.pressure_spike
        )
        jax.block_until_ready((trial_state, trial_produced))
        compile_seconds = time.perf_counter() - compile_start

        advance_seconds: list[float] = []
        exchange_seconds: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            trial_state, trial_produced = compiled_advance(
                primed_state, targets, ops.pressure, ops.pressure_spike
            )
            jax.block_until_ready((trial_state, trial_produced))
            advance_seconds.append(time.perf_counter() - start)

            start = time.perf_counter()
            received = compiled_exchange(trial_produced)
            jax.block_until_ready(received)
            exchange_seconds.append(time.perf_counter() - start)

        advance_median = _median(advance_seconds)
        exchange_median = _median(exchange_seconds)
        total_per_step = (advance_median + exchange_median) / chunk_steps
        local_payload_bytes = int(empty.addressable_shards[0].data.nbytes)
        row = {
            "chunk_steps": chunk_steps,
            "compile_seconds": compile_seconds,
            "advance_seconds_per_chunk": advance_median,
            "exchange_seconds_per_chunk": exchange_median,
            "advance_ms_per_step": 1.0e3 * advance_median / chunk_steps,
            "exchange_ms_per_step": 1.0e3 * exchange_median / chunk_steps,
            "total_ms_per_step": 1.0e3 * total_per_step,
            "pipeline_communication_fraction": exchange_median
            / (advance_median + exchange_median),
            "advance_to_pipeline_exchange_ratio": advance_median
            / max(exchange_median, 1.0e-30),
            "local_payload_bytes_per_chunk": local_payload_bytes,
            "effective_payload_gb_per_second": local_payload_bytes
            / max(exchange_median, 1.0e-30)
            / 1.0e9,
            "advance_samples_seconds": advance_seconds,
            "exchange_samples_seconds": exchange_seconds,
            "xla_advance_cost_analysis": _cost_analysis(compiled_advance),
        }
        rows.append(row)
        if rank == 0:
            print(
                f"[autotune] chunk={chunk_steps:3d}: "
                f"advance={row['advance_ms_per_step']:.3f} ms/step, "
                f"exchange={row['exchange_ms_per_step']:.3f} ms/step, "
                f"comm={100.0 * row['pipeline_communication_fraction']:.2f}%, "
                f"total={row['total_ms_per_step']:.3f} ms/step",
                flush=True,
            )

    minimum = min(row["total_ms_per_step"] for row in rows)
    near_best = [
        row
        for row in rows
        if row["total_ms_per_step"] <= minimum * (1.0 + args.selection_slack)
    ]
    recommendation = min(near_best, key=lambda row: row["chunk_steps"])
    report = {
        "method": (
            "JAX compiled execution with block_until_ready; advance and packed "
            "CollectivePermute are timed separately on the real distributed mesh"
        ),
        "backend": jax.default_backend(),
        "process_count": size,
        "mesh": {name: int(extent) for name, extent in mesh.shape.items()},
        "grid": [configured.nx, configured.ny, configured.nz],
        "fringe_x_cells": int(empty.shape[3]),
        "dtype": str(configured.dtype),
        "repeats": args.repeats,
        "selection_slack": args.selection_slack,
        "recommended_chunk_steps": recommendation["chunk_steps"],
        "raw_minimum_chunk_steps": min(
            rows, key=lambda row: row["total_ms_per_step"]
        )["chunk_steps"],
        "candidates": rows,
        "interpretation": (
            "advance includes solver-internal z-halo/FFT collectives; exchange "
            "isolates the once-per-chunk packed precursor-to-turbine ppermute"
        ),
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"[autotune] recommended chunk_steps="
            f"{report['recommended_chunk_steps']} "
            f"(raw minimum={report['raw_minimum_chunk_steps']}); "
            f"report={args.output}",
            flush=True,
        )
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
