#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


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
            "JAX-native concurrent precursor: explicit adjoint dimension on "
            "an adjoint x z device mesh."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coordinator-address", default="127.0.0.1:12670")
    parser.add_argument("--num-processes", type=int)
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--local-device-id", type=int, default=0)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=60000,
        help="Maximum warm-up steps; ustar convergence normally stops earlier.",
    )
    parser.add_argument(
        "--warmup-checkpoint",
        type=Path,
        help=(
            "Load a complete z-sharded warm-up checkpoint instead of running "
            "the precursor spin-up."
        ),
    )
    parser.add_argument(
        "--concurrent-steps",
        type=int,
        default=8,
        help="Paired concurrent steps; use 0 for a warm-up-only run.",
    )
    parser.add_argument("--chunk-steps", type=int, default=8)
    parser.add_argument(
        "--chunks-per-launch",
        type=int,
        default=64,
        help=(
            "Number of packed fringe chunks fused into one device-side scan; "
            "the final shorter batch is compiled separately."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hub-height", type=float)
    parser.add_argument("--target-hub-speed", type=float)
    parser.add_argument("--ustar-window-seconds", type=float, default=30.0)
    parser.add_argument("--ustar-tolerance", type=float, default=0.02)
    parser.add_argument("--ustar-sample-steps", type=int, default=100)
    parser.add_argument("--warmup-min-seconds", type=float, default=90.0)
    parser.add_argument(
        "--warmup-sgs-model",
        choices=("smagorinsky", "lasd"),
        help="Optional SGS override during precursor spin-up.",
    )
    parser.add_argument("--warmup-initial-velocity-noise", type=float)
    parser.add_argument("--warmup-disable-sponge", action="store_true")
    parser.add_argument(
        "--allow-unconverged-warmup",
        action="store_true",
        help="Continue at the maximum step count even if the ustar criterion fails.",
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


def _save_local_state(output_dir: Path, state, rank: int, params) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    indices: dict[str, list[list[int]]] = {}
    for name, value in zip(state._fields, state, strict=True):
        shards = value.addressable_shards
        if len(shards) != 1:
            raise RuntimeError(
                "run_native expects one addressable device per process; "
                f"field {name} has {len(shards)}"
            )
        shard = shards[0]
        arrays[name] = np.asarray(shard.data)
        indices[name] = [
            [part.start, part.stop] if isinstance(part, slice) else [part, part + 1]
            for part in shard.index
        ]
    np.savez(output_dir / f"rank_{rank:05d}.npz", **arrays)
    (output_dir / f"rank_{rank:05d}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "indices": indices,
                "global_field_shape": [2, params.nx, params.ny, params.nz],
                "layout": ["adjoint", "x", "y", "z"],
                "adjoint_roles": ["precursor", "turbine"],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    rank, size = _rank_and_size(args)
    if size < 4 or size % 2:
        raise SystemExit("JAX-native adjoint mesh requires an even rank count >= 4")
    if args.warmup_steps < 0 or args.concurrent_steps < 0:
        raise SystemExit("warmup and concurrent steps must be nonnegative")
    if args.chunk_steps <= 0 or args.concurrent_steps % args.chunk_steps:
        raise SystemExit("concurrent steps must be divisible by chunk steps")
    if args.chunks_per_launch <= 0:
        raise SystemExit("--chunks-per-launch must be positive")
    if args.target_hub_speed is not None and args.hub_height is None:
        raise SystemExit("--target-hub-speed requires --hub-height")
    if args.hub_height is not None and args.hub_height <= 0.0:
        raise SystemExit("--hub-height must be positive")
    if args.target_hub_speed is not None and args.target_hub_speed <= 0.0:
        raise SystemExit("--target-hub-speed must be positive")
    if args.ustar_window_seconds <= 0.0:
        raise SystemExit("--ustar-window-seconds must be positive")
    if args.ustar_tolerance <= 0.0:
        raise SystemExit("--ustar-tolerance must be positive")
    if args.ustar_sample_steps <= 0:
        raise SystemExit("--ustar-sample-steps must be positive")
    if args.warmup_min_seconds < 0.0:
        raise SystemExit("--warmup-min-seconds must be nonnegative")

    settings = dict(RUN_DEFAULTS)
    settings.update(load_config_file(args.config))
    if settings["precision"] == "float64" or settings["sgs_precision"] == "float64":
        from jax import config as jax_config

        jax_config.update("jax_enable_x64", True)

    import jax

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=size,
        process_id=rank,
        local_device_ids=[args.local_device_id],
    )
    import jax.numpy as jnp

    from wireles_jax.adjoint_sharded import (
        duplicate_state_for_adjoint,
        make_adjoint_pipeline_batch,
        make_adjoint_pipeline_prime,
        make_empty_fringe_chunk,
    )
    from wireles_jax.convergence import UStarSlidingWindow
    from wireles_jax.checkpoint_sharded import (
        load_sharded_checkpoint,
        save_sharded_checkpoint,
    )
    from wireles_jax.sharding import (
        make_adjoint_distributed_mesh,
        make_distributed_mesh,
    )
    from wireles_jax.timestep_sharded import (
        make_mean_u_at_height_sharded,
        make_sharded_operators,
        run_sharded,
    )

    configured = params_from_settings(settings, jnp)
    if not configured.fringe_enabled:
        raise SystemExit("Concurrent turbine domain requires [fringe] enabled=true")
    warmup_params = replace(
        configured,
        nsteps=args.warmup_steps,
        actuator_disk_enabled=False,
        cold_source_enabled=False,
        fringe_enabled=False,
        horizontal_homogeneous=True,
        buoyancy_reference="plane_mean",
        sharded_pressure_solver="transpose",
        sgs_model=(
            configured.sgs_model
            if args.warmup_sgs_model is None
            else args.warmup_sgs_model
        ),
        initial_velocity_noise=(
            configured.initial_velocity_noise
            if args.warmup_initial_velocity_noise is None
            else args.warmup_initial_velocity_noise
        ),
        sponge_enabled=(
            configured.sponge_enabled and not args.warmup_disable_sponge
        ),
    )
    window_samples = max(
        1,
        math.ceil(
            args.ustar_window_seconds
            / (warmup_params.dt_physical * args.ustar_sample_steps)
        ),
    )
    minimum_step = math.ceil(
        args.warmup_min_seconds / warmup_params.dt_physical
    )
    ustar_criterion = UStarSlidingWindow(
        target=warmup_params.pressure_ustar,
        relative_tolerance=args.ustar_tolerance,
        window_samples=window_samples,
        minimum_step=minimum_step,
    )
    report_interval_steps = max(
        args.ustar_sample_steps,
        math.ceil(
            5.0
            / (warmup_params.dt_physical * args.ustar_sample_steps)
        )
        * args.ustar_sample_steps,
    )
    warmup_wall_start = time.perf_counter()

    def warmup_converged(diag) -> bool:
        step = int(diag.step)
        if step == 0:
            return False
        instantaneous = float(diag.ustar)
        converged = ustar_criterion.update(step, instantaneous)
        if rank == 0 and (step % report_interval_steps == 0 or converged):
            elapsed = time.perf_counter() - warmup_wall_start
            simulated = step * warmup_params.dt_physical
            print(
                f"[warmup] step={step}/{args.warmup_steps}, "
                f"t={simulated:.1f}s, instantaneous ustar={instantaneous:.6f}, "
                f"window mean={ustar_criterion.mean:.6f}, "
                f"samples={ustar_criterion.sample_count}/{window_samples}, "
                f"wall={elapsed:.1f}s",
                flush=True,
            )
        return converged

    warm_mesh = make_distributed_mesh(size)
    if args.warmup_checkpoint is None:
        if rank == 0:
            print(
                f"[native] warmup: {size} z ranks, max={args.warmup_steps} steps; "
                f"target ustar={ustar_criterion.target:.6f} m/s, "
                f"window={args.ustar_window_seconds:g}s, "
                f"tolerance=+/-{100.0 * args.ustar_tolerance:g}%, "
                f"minimum={args.warmup_min_seconds:g}s",
                flush=True,
            )
        warm_state, _ = run_sharded(
            warmup_params,
            num_devices=size,
            log_every=args.ustar_sample_steps,
            seed=args.seed,
            log_callback=None,
            status_callback=(
                (lambda message: print(f"[warmup] {message}", flush=True))
                if rank == 0
                else None
            ),
            stop_callback=warmup_converged,
        )
        warmup_converged_ok = ustar_criterion.converged
        if rank == 0:
            mean_text = (
                "unavailable"
                if ustar_criterion.mean is None
                else f"{ustar_criterion.mean:.6f} m/s"
            )
            print(
                f"[warmup] ustar window mean={mean_text} at "
                f"step={ustar_criterion.last_step}; "
                f"converged={warmup_converged_ok}",
                flush=True,
            )
        warmup_checkpoint = args.output_dir / "warmup"
        save_sharded_checkpoint(
            warmup_checkpoint,
            warm_state,
            warmup_params,
            warm_mesh,
            rank=rank,
        )
        if rank == 0:
            (warmup_checkpoint / "convergence.json").write_text(
                json.dumps(
                    {
                        "target_ustar": ustar_criterion.target,
                        "relative_tolerance": ustar_criterion.relative_tolerance,
                        "window_seconds": args.ustar_window_seconds,
                        "window_samples": window_samples,
                        "minimum_seconds": args.warmup_min_seconds,
                        "step": ustar_criterion.last_step,
                        "window_mean": ustar_criterion.mean,
                        "converged": warmup_converged_ok,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"[warmup] checkpoint: {warmup_checkpoint}", flush=True)
    else:
        warm_state = load_sharded_checkpoint(
            args.warmup_checkpoint,
            warmup_params,
            warm_mesh,
            rank=rank,
        )
        convergence_path = args.warmup_checkpoint / "convergence.json"
        convergence = (
            json.loads(convergence_path.read_text())
            if convergence_path.exists()
            else {}
        )
        warmup_converged_ok = bool(convergence.get("converged", False))
        if rank == 0:
            print(
                f"[warmup] loaded checkpoint {args.warmup_checkpoint} at "
                f"step={int(jax.device_get(warm_state.step))}; "
                f"converged={warmup_converged_ok}",
                flush=True,
            )
    if not warmup_converged_ok and not args.allow_unconverged_warmup:
        jax.distributed.shutdown()
        raise SystemExit(
            "Warm-up state does not satisfy the ustar sliding-window "
            "criterion. Continue the warm-up or pass "
            "--allow-unconverged-warmup only for a smoke test."
        )
    if args.hub_height is not None:
        hub_probe = make_mean_u_at_height_sharded(
            warmup_params, warm_mesh, args.hub_height
        )
        hub_speed = float(jax.block_until_ready(hub_probe(warm_state.u)))
        if rank == 0:
            message = (
                f"[warmup] mean U(z={args.hub_height:g} m)="
                f"{hub_speed:.6f} m/s"
            )
            if args.target_hub_speed is not None:
                relative_error = (
                    hub_speed / args.target_hub_speed - 1.0
                )
                suggested_force = (
                    warmup_params.driving_pressure_force
                    * (args.target_hub_speed / hub_speed) ** 2
                )
                message += (
                    f"; target={args.target_hub_speed:.6f}, "
                    f"error={100.0 * relative_error:+.2f}%, "
                    f"next pressure_force={suggested_force:.9g} m/s^2"
                )
            print(message, flush=True)

    if args.concurrent_steps == 0:
        if rank == 0:
            print("[native] warm-up-only run complete", flush=True)
        jax.distributed.shutdown()
        return

    concurrent_params = replace(
        configured,
        nsteps=args.concurrent_steps,
        horizontal_homogeneous=False,
        buoyancy_reference="ambient",
        sharded_pressure_solver="transpose",
    )
    mesh = make_adjoint_distributed_mesh(size)
    state = duplicate_state_for_adjoint(warm_state, mesh)
    ops = make_sharded_operators(concurrent_params, mesh)
    empty = make_empty_fringe_chunk(
        concurrent_params, mesh, args.chunk_steps
    )
    prime = jax.jit(
        make_adjoint_pipeline_prime(
            concurrent_params,
            ops,
            mesh,
            chunk_steps=args.chunk_steps,
        ),
        donate_argnums=(0, 1),
    )

    if rank == 0:
        print(
            f"[native] mesh={dict(mesh.shape)}; chunk_steps={args.chunk_steps}, "
            f"chunks_per_launch={args.chunks_per_launch}; compiling prime",
            flush=True,
        )
    start = time.perf_counter()
    state, targets = prime(
        state, empty, ops.pressure, ops.pressure_spike
    )
    jax.block_until_ready(targets)
    prime_seconds = time.perf_counter() - start
    if rank == 0:
        print(
            f"[native] prime/compile complete in {prime_seconds:.2f}s",
            flush=True,
        )

    total_chunks = args.concurrent_steps // args.chunk_steps
    full_batches, tail_chunks = divmod(
        total_chunks, args.chunks_per_launch
    )

    compile_seconds: dict[int, float] = {}

    def compile_batch(batch_chunks: int):
        batched = jax.jit(
            make_adjoint_pipeline_batch(
                concurrent_params,
                ops,
                mesh,
                chunk_steps=args.chunk_steps,
                chunks_per_launch=batch_chunks,
            ),
            donate_argnums=(0, 1),
        )
        compile_start = time.perf_counter()
        executable = batched.lower(
            state,
            targets,
            ops.pressure,
            ops.pressure_spike,
        ).compile()
        compile_seconds[batch_chunks] = time.perf_counter() - compile_start
        if rank == 0:
            print(
                f"[native] compiled batch of {batch_chunks} chunk(s) in "
                f"{compile_seconds[batch_chunks]:.2f}s",
                flush=True,
            )
        return executable

    full_batch = (
        compile_batch(args.chunks_per_launch) if full_batches else None
    )
    tail_batch = compile_batch(tail_chunks) if tail_chunks else None

    start = time.perf_counter()
    completed_chunks = 0
    report_every = max(1, math.ceil(full_batches / 10))
    for batch_id in range(full_batches):
        state, targets = full_batch(
            state, targets, ops.pressure, ops.pressure_spike
        )
        completed_chunks += args.chunks_per_launch
        if rank == 0 and (
            (batch_id + 1) % report_every == 0
            or batch_id + 1 == full_batches
        ):
            print(
                f"[native] batch {batch_id + 1}/{full_batches} dispatched; "
                f"chunks={completed_chunks}/{total_chunks}",
                flush=True,
            )
    if tail_batch is not None:
        state, targets = tail_batch(
            state, targets, ops.pressure, ops.pressure_spike
        )
        completed_chunks += tail_chunks
        if rank == 0:
            print(
                f"[native] tail batch dispatched; "
                f"chunks={completed_chunks}/{total_chunks}",
                flush=True,
            )
    if completed_chunks != total_chunks:
        raise RuntimeError(
            f"Dispatched {completed_chunks} chunks, expected {total_chunks}"
        )
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - start
    _save_local_state(args.output_dir, state, rank, concurrent_params)
    if rank == 0:
        milliseconds_per_step = 1.0e3 * elapsed / args.concurrent_steps
        (args.output_dir / "pipeline_run.json").write_text(
            json.dumps(
                {
                    "concurrent_steps": args.concurrent_steps,
                    "chunk_steps": args.chunk_steps,
                    "chunks_per_launch": args.chunks_per_launch,
                    "total_chunks": total_chunks,
                    "full_batches": full_batches,
                    "tail_chunks": tail_chunks,
                    "prime_compile_seconds": prime_seconds,
                    "batch_compile_seconds": compile_seconds,
                    "execution_seconds": elapsed,
                    "milliseconds_per_paired_step": milliseconds_per_step,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"[native] completed {args.concurrent_steps} paired step(s) in "
            f"{elapsed:.3f}s ({milliseconds_per_step:.3f} ms/step); "
            f"local shards: {args.output_dir}",
            flush=True,
        )
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
