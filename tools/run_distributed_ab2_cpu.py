#!/usr/bin/env python3
"""Run accepted AB2 steps and per-rank restart on true JAX CPU processes."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from run_distributed_projection_cpu import _free_local_port, _source_environment


RESULT_PREFIX = "WIRELES_DISTRIBUTED_AB2_RESULT="


def _launch(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    coordinator = f"127.0.0.1:{_free_local_port()}"
    environment = _source_environment(root)
    processes = []
    for rank in range(args.processes):
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--processes",
                    str(args.processes),
                    "--rank",
                    str(rank),
                    "--coordinator",
                    coordinator,
                    "--dtype",
                    args.dtype,
                    "--method",
                    args.method,
                    "--steps",
                    str(args.steps),
                    "--vector-field",
                    args.vector_field,
                ],
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    deadline = time.monotonic() + args.timeout
    outputs = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(
                timeout=max(0.1, deadline - time.monotonic())
            )
            outputs.append((stdout, stderr, int(process.returncode)))
    except subprocess.TimeoutExpired:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise RuntimeError(f"distributed AB2 exceeded {args.timeout:.0f} s") from None

    failures = []
    result = None
    for rank, (stdout, stderr, returncode) in enumerate(outputs):
        if returncode:
            failures.append(
                f"rank {rank} exited {returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        for line in stdout.splitlines():
            if line.startswith(RESULT_PREFIX):
                result = json.loads(line.removeprefix(RESULT_PREFIX))
    if failures:
        raise RuntimeError("\n\n".join(failures))
    if result is None:
        raise RuntimeError("rank zero produced no distributed AB2 result")
    tolerance = 1.0e-4 if args.dtype == "float32" else 1.0e-11
    for metric in ("divergence", "restart_error", "pressure_mean"):
        if result[metric] >= tolerance:
            raise RuntimeError(
                f"AB2 {metric}={result[metric]:.6e} exceeds {tolerance:.1e}"
            )
    if not result["dtype_preserved"]:
        raise RuntimeError("AB2 did not preserve its configured dtype")
    if not result["ownership_audit_passed"]:
        raise RuntimeError("AB2 process-local ownership audit failed")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _global_max(jax, value, *, axis_name: str, world: int):
    from jax import lax

    mapped = partial(jax.pmap, axis_name=axis_name, axis_size=world)

    @mapped
    def reduce(local):
        return lax.pmax(local, axis_name)

    return reduce(value)


def _global_sum(jax, value, *, axis_name: str, world: int):
    from jax import lax

    mapped = partial(jax.pmap, axis_name=axis_name, axis_size=world)

    @mapped
    def reduce(local):
        return lax.psum(local, axis_name)

    return reduce(value)


def _worker(args: argparse.Namespace) -> int:
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=args.processes,
        process_id=args.rank,
        local_device_ids=[0],
    )
    try:
        import jax.numpy as jnp
        from spectral_fd import runtime_from_initialized_jax

        from wireles.domain import (
            AcceptedClock,
            AddressableField,
            Cell,
            DistributionSpec,
            EqualZSlab,
            Evaluated,
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
        from wireles.effects import (
            ZSlabCheckpointLayout,
            load_ab2_checkpoint,
            save_ab2_checkpoint,
        )
        from wireles.integrators import AB2Config, VectorFieldResult, cold_start, step
        from wireles.interpreters.jax_zslab import (
            ZFaceFieldContext,
            build_zslab_interpreter,
        )
        from wireles.operators import VelocityVector
        from wireles.physics import (
            ConservativeAdvection,
            DryFlowModel,
            DryFlowVectorField,
            KinematicPressureGradient,
            NeutralLogWall,
            StaticSmagorinsky,
        )
        from wireles.pressure import build_spectral_fd_pressure_adapter

        if (
            jax.process_count() != args.processes
            or jax.device_count() != args.processes
            or jax.local_device_count() != 1
        ):
            raise RuntimeError("AB2 CPU mesh must contain one local device per process")
        dtype = getattr(jnp, args.dtype)
        grid = UniformGrid(8, 8, 16, 8.0, 8.0, 16.0)
        config = AB2Config(0.02)
        decomposition = EqualZSlab(
            grid,
            MeshTopology((MeshAxis("z", args.processes),)),
            DistributionSpec.z_slab(),
        )
        shards = (args.rank,)
        cell_region = decomposition.regions(Cell)[args.rank]
        face_region = decomposition.regions(ZFace)[args.rank]
        local_z = decomposition.cells_per_shard
        shape = (1, local_z, grid.ny, grid.nx)
        zero_cells = jnp.zeros(shape, dtype)
        zero_faces = jnp.zeros(shape, dtype)
        zero_boundary = jnp.zeros((grid.ny, grid.nx), dtype)
        z = jnp.arange(
            cell_region.cell_z.start,
            cell_region.cell_z.stop,
            dtype=dtype,
        )[None, :, None, None]
        zf = jnp.arange(
            face_region.stored_z.start,
            face_region.stored_z.stop,
            dtype=dtype,
        )[None, :, None, None]
        y = jnp.arange(grid.ny, dtype=dtype)[None, None, :, None]
        x = jnp.arange(grid.nx, dtype=dtype)[None, None, None, :]
        if args.vector_field == "dry":
            initial_u = jnp.broadcast_to(
                2.0 + 0.1 * jnp.cos(2.0 * jnp.pi * y / grid.ny) + 0.01 * z,
                shape,
            )
            initial_v = jnp.broadcast_to(
                0.1 * jnp.cos(2.0 * jnp.pi * x / grid.nx),
                shape,
            )
        else:
            initial_u = zero_cells
            initial_v = zero_cells
        initial_velocity = VelocityVector(
            AddressableField(
                XVelocity,
                Cell,
                (cell_region,),
                Projected,
                initial_u,
            ),
            AddressableField(
                YVelocity,
                Cell,
                (cell_region,),
                Projected,
                initial_v,
            ),
            ZFaceFieldContext(
                AddressableField(
                    VerticalVelocity,
                    ZFace,
                    (face_region,),
                    Projected,
                    zero_faces,
                ),
                zero_boundary,
            ),
        )
        def manufactured_vector_field(evaluation):
            physical_time = evaluation.time.time
            return VectorFieldResult(
                VelocityVector(
                    AddressableField(
                        XVelocityTendency,
                        Cell,
                        (cell_region,),
                        Evaluated,
                        jnp.sin(0.17 * x + 0.13 * y + 0.11 * z + physical_time),
                    ),
                    AddressableField(
                        YVelocityTendency,
                        Cell,
                        (cell_region,),
                        Evaluated,
                        jnp.cos(
                            0.19 * x - 0.07 * y + 0.05 * z - 0.5 * physical_time
                        ),
                    ),
                    ZFaceFieldContext(
                        AddressableField(
                            VerticalVelocityTendency,
                            ZFace,
                            (face_region,),
                            Evaluated,
                            jnp.sin(
                                0.09 * x
                                + 0.15 * y
                                + 0.12 * zf
                                + 0.25 * physical_time
                            ),
                        ),
                        jnp.sin(
                            0.09 * x[0, 0]
                            + 0.15 * y[0, 0]
                            + 0.25 * physical_time
                        ),
                    ),
                ),
                evaluation.time,
            )

        def normal_boundary(_clock, _environment):
            return VerticalBoundary(0.0, 0.0)

        algebra = build_zslab_interpreter(
            decomposition,
            addressable_shards=shards,
        )
        if args.vector_field == "dry":
            vector_field = DryFlowVectorField(
                algebra,
                DryFlowModel(
                    ConservativeAdvection(),
                    KinematicPressureGradient(0.002),
                    NeutralLogWall(0.01),
                    StaticSmagorinsky(0.16),
                ),
            )
        else:
            vector_field = manufactured_vector_field
        pressure_solver = build_spectral_fd_pressure_adapter(
            decomposition,
            addressable_shards=shards,
            runtime=runtime_from_initialized_jax(jax),
            dtype=args.dtype,
            method=args.method,
        )

        def initial_state():
            return cold_start(
                initial_velocity,
                clock=AcceptedClock(0.0, 0),
                config=config,
            )

        def advance(state):
            return step(
                state,
                config=config,
                environment=None,
                vector_field=vector_field,
                normal_boundary=normal_boundary,
                algebra=algebra,
                pressure_solver=pressure_solver,
            )

        def run(state, count):
            maximum_divergence = jnp.asarray(0.0, dtype)
            last_result = None
            evaluation_times = []
            for _ in range(count):
                last_result = advance(state)
                state = last_result.state
                evaluation_times.append(last_result.diagnostic.evaluation_time.time)
                maximum_divergence = jnp.maximum(
                    maximum_divergence,
                    jnp.max(
                        jnp.abs(
                            last_result.diagnostic.projection.divergence.payload
                        )
                    ),
                )
            return state, last_result, maximum_divergence, evaluation_times

        started = time.perf_counter()
        continuous, final_result, local_divergence, evaluation_times = run(
            initial_state(),
            args.steps,
        )
        final_result.diagnostic.projection.divergence.payload.block_until_ready()
        elapsed = time.perf_counter() - started
        split = args.steps // 2
        interrupted, _, _, _ = run(initial_state(), split)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / f"rank{args.rank:04d}.npz"
            save_ab2_checkpoint(checkpoint, interrupted)
            restarted = load_ab2_checkpoint(
                checkpoint,
                layout=ZSlabCheckpointLayout(decomposition, shards, jnp.asarray),
                config=config,
            )
        restarted, _, _, restarted_times = run(restarted, args.steps - split)

        local_restart_error = jnp.maximum(
            jnp.max(jnp.abs(continuous.velocity.x.payload - restarted.velocity.x.payload)),
            jnp.maximum(
                jnp.max(
                    jnp.abs(continuous.velocity.y.payload - restarted.velocity.y.payload)
                ),
                jnp.maximum(
                    jnp.max(
                        jnp.abs(
                            continuous.velocity.z.owned.payload
                            - restarted.velocity.z.owned.payload
                        )
                    ),
                    jnp.max(
                        jnp.abs(
                            continuous.history.value.x.payload
                            - restarted.history.value.x.payload
                        )
                    ),
                ),
            ),
        )[None]
        pressure_sum = jnp.sum(final_result.diagnostic.projection.pressure.payload)[None]
        global_divergence = _global_max(
            jax,
            local_divergence[None],
            axis_name="ab2_divergence_audit",
            world=args.processes,
        )[0]
        global_restart_error = _global_max(
            jax,
            local_restart_error,
            axis_name="ab2_restart_audit",
            world=args.processes,
        )[0]
        pressure_mean = jnp.abs(
            _global_sum(
                jax,
                pressure_sum,
                axis_name="ab2_pressure_audit",
                world=args.processes,
            )[0]
            / grid.cell_count
        )
        elapsed_max = _global_max(
            jax,
            jnp.asarray([elapsed], dtype=dtype),
            axis_name="ab2_time_audit",
            world=args.processes,
        )[0]
        local_cells = local_z * grid.ny * grid.nx
        report = {
            "accepted_clock": [continuous.clock.time, continuous.clock.step],
            "backend": jax.default_backend(),
            "divergence": float(global_divergence),
            "dtype": args.dtype,
            "dtype_preserved": (
                str(continuous.velocity.x.payload.dtype) == args.dtype
                and str(continuous.history.value.x.payload.dtype) == args.dtype
            ),
            "elapsed_seconds_max": float(elapsed_max),
            "evaluation_times": evaluation_times,
            "local_cells_per_process": local_cells,
            "local_shape": [1, local_z, grid.ny, grid.nx],
            "method": args.method,
            "ownership_audit_passed": (
                local_cells * args.processes == grid.cell_count
                and (args.processes == 1 or local_cells < grid.cell_count)
            ),
            "pressure_mean": float(pressure_mean),
            "processes": args.processes,
            "restart_error": float(global_restart_error),
            "restart_evaluation_times": restarted_times,
            "steps": args.steps,
            "vector_field": args.vector_field,
        }
        if args.rank == 0:
            print(f"{RESULT_PREFIX}{json.dumps(report, sort_keys=True)}", flush=True)
        return 0
    finally:
        jax.distributed.shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--method",
        choices=("transpose", "spike", "spike-adaptive"),
        default="spike",
    )
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument(
        "--vector-field",
        choices=("manufactured", "dry"),
        default="manufactured",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--coordinator", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.processes not in (1, 2, 4):
        parser.error("the AB2 CPU gate supports 1, 2, or 4 processes")
    if args.steps < 2:
        parser.error("AB2 CPU gate requires at least two steps")
    if args.worker and not args.coordinator:
        parser.error("workers require a coordinator address")
    return args


def main() -> int:
    args = _parse_args()
    return _worker(args) if args.worker else _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
