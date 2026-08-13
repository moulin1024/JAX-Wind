#!/usr/bin/env python3
"""Launch a true multi-process CPU projection gate without MPI dependencies."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


RESULT_PREFIX = "JAXWIND_DISTRIBUTED_RESULT="


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _source_environment(root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    paths = [str(root / "src")]
    dependency = Path(
        environment.get(
            "JAXWIND_SPECTRAL_FD_SOURCE",
            root / "external" / "bw1000_benchmark",
        )
    )
    if dependency.exists():
        paths.append(str(dependency))
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_X64"] = "1"
    flags = environment.get("XLA_FLAGS", "")
    environment["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count=1 {flags}"
    ).strip()
    return environment


def _validate_result(result: dict, args: argparse.Namespace) -> None:
    tolerance = 1.0e-4 if args.dtype == "float32" else 1.0e-11
    if result["backend"] != "cpu":
        raise RuntimeError("distributed CPU gate ran on a non-CPU backend")
    if result["processes"] != args.processes:
        raise RuntimeError("reported JAX process count is incorrect")
    if result["global_devices"] != args.processes:
        raise RuntimeError("the global device mesh is not one CPU per process")
    if not result["ownership_audit_passed"]:
        raise RuntimeError("a process-local payload failed the ownership audit")
    for method, report in result["methods"].items():
        if report["velocity_dtype"] != args.dtype or report["pressure_dtype"] != args.dtype:
            raise RuntimeError(f"{method} did not preserve {args.dtype}")
        for metric in ("divergence", "idempotence", "pressure_mean"):
            if report[metric] >= tolerance:
                raise RuntimeError(
                    f"{method} {metric}={report[metric]:.6e} exceeds {tolerance:.1e}"
                )
    for method, difference in result["method_differences"].items():
        if difference >= tolerance:
            raise RuntimeError(
                f"{method} differs from the oracle by {difference:.6e}, "
                f"exceeding {tolerance:.1e}"
            )


def _launch(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    coordinator = f"127.0.0.1:{_free_local_port()}"
    environment = _source_environment(root)
    processes = []
    for rank in range(args.processes):
        command = [
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
            "--methods",
            args.methods,
        ]
        processes.append(
            subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    deadline = time.monotonic() + args.timeout
    outputs: list[tuple[str, str, int]] = []
    try:
        for process in processes:
            remaining = max(0.1, deadline - time.monotonic())
            stdout, stderr = process.communicate(timeout=remaining)
            outputs.append((stdout, stderr, int(process.returncode)))
    except subprocess.TimeoutExpired:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise RuntimeError(
            f"distributed projection exceeded the {args.timeout:.0f} s timeout"
        ) from None

    failures = []
    result = None
    for rank, (stdout, stderr, returncode) in enumerate(outputs):
        if returncode != 0:
            failures.append(
                f"rank {rank} exited {returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        for line in stdout.splitlines():
            if line.startswith(RESULT_PREFIX):
                result = json.loads(line.removeprefix(RESULT_PREFIX))
    if failures:
        raise RuntimeError("\n\n".join(failures))
    if result is None:
        raise RuntimeError("rank zero produced no distributed result")
    _validate_result(result, args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _global_max(jax, values, *, axis_name: str, world: int):
    from jax import lax

    mapped = partial(jax.pmap, axis_name=axis_name, axis_size=world)

    @mapped
    def reduce_max(value):
        return lax.pmax(value, axis_name)

    return reduce_max(values)


def _global_sum(jax, values, *, axis_name: str, world: int):
    from jax import lax

    mapped = partial(jax.pmap, axis_name=axis_name, axis_size=world)

    @mapped
    def reduce_sum(value):
        return lax.psum(value, axis_name)

    return reduce_sum(values)


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

        from jaxwind.domain import (
            AddressableField,
            Candidate,
            Cell,
            DistributionSpec,
            EqualVerticalPartition,
            MeshAxis,
            MeshTopology,
            UniformGrid,
            VerticalBoundary,
            VerticalVelocity,
            XVelocity,
            YVelocity,
            ZFace,
        )
        from jaxwind._jax.discretization import (
            VerticalFaceField,
            build_discretization,
        )
        from jaxwind.operators import VelocityVector, project
        from jaxwind.pressure import build_spectral_fd_pressure_adapter

        if jax.process_count() != args.processes:
            raise RuntimeError("JAX process count does not match the launcher")
        if jax.local_device_count() != 1:
            raise RuntimeError("the CPU gate requires exactly one local device per process")
        if jax.device_count() != args.processes:
            raise RuntimeError("the global CPU device mesh does not match process count")

        dtype = getattr(jnp, args.dtype)
        grid = UniformGrid(8, 8, 16, 8.0, 8.0, 16.0)
        decomposition = EqualVerticalPartition(
            grid,
            MeshTopology((MeshAxis("z", args.processes),)),
            DistributionSpec.vertical(),
        )
        addressable_partitions = (args.rank,)
        cell_region = decomposition.regions(Cell)[args.rank]
        face_region = decomposition.regions(ZFace)[args.rank]
        local_z = decomposition.cells_per_partition
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
        velocity = VelocityVector(
            AddressableField(
                XVelocity,
                Cell,
                (cell_region,),
                Candidate,
                jnp.sin(0.37 * x + 0.19 * y + 0.11 * z),
            ),
            AddressableField(
                YVelocity,
                Cell,
                (cell_region,),
                Candidate,
                jnp.cos(0.23 * x - 0.31 * y + 0.07 * z),
            ),
            VerticalFaceField(
                AddressableField(
                    VerticalVelocity,
                    ZFace,
                    (face_region,),
                    Candidate,
                    jnp.sin(0.17 * x + 0.13 * y + 0.29 * zf),
                ),
                jnp.zeros((grid.ny, grid.nx), dtype=dtype),
            ),
        )
        boundary = VerticalBoundary(dtype(0.0), dtype(0.0))
        interpreter = build_discretization(
            decomposition,
            addressable_partitions=addressable_partitions,
        )
        runtime = runtime_from_initialized_jax(jax)
        methods = tuple(name.strip() for name in args.methods.split(",") if name.strip())
        results = {}
        reports = {}
        for method in methods:
            pressure_solver = build_spectral_fd_pressure_adapter(
                decomposition,
                addressable_partitions=addressable_partitions,
                runtime=runtime,
                dtype=args.dtype,
                method=method,
            )
            started = time.perf_counter()
            result = project(
                velocity,
                dt=0.2,
                normal_boundary=boundary,
                algebra=interpreter,
                pressure_solver=pressure_solver,
            )
            result.divergence.payload.block_until_ready()
            first_seconds = time.perf_counter() - started
            started = time.perf_counter()
            repeated_candidate = project(
                velocity,
                dt=0.2,
                normal_boundary=boundary,
                algebra=interpreter,
                pressure_solver=pressure_solver,
            )
            repeated_candidate.divergence.payload.block_until_ready()
            steady_seconds = time.perf_counter() - started
            idempotent = project(
                result.velocity,
                dt=0.2,
                normal_boundary=boundary,
                algebra=interpreter,
                pressure_solver=pressure_solver,
            )
            idempotent.divergence.payload.block_until_ready()

            local_divergence = jnp.max(jnp.abs(result.divergence.payload))[None]
            local_idempotence = jnp.maximum(
                jnp.max(jnp.abs(result.velocity.x.payload - idempotent.velocity.x.payload)),
                jnp.maximum(
                    jnp.max(
                        jnp.abs(result.velocity.y.payload - idempotent.velocity.y.payload)
                    ),
                    jnp.max(
                        jnp.abs(
                            result.velocity.z.owned.payload
                            - idempotent.velocity.z.owned.payload
                        )
                    ),
                ),
            )[None]
            pressure_sum = jnp.sum(result.pressure.payload)[None]
            first_time = jnp.asarray([first_seconds], dtype=dtype)
            steady_time = jnp.asarray([steady_seconds], dtype=dtype)
            reports[method] = {
                "divergence": float(
                    _global_max(
                        jax,
                        local_divergence,
                        axis_name=f"audit_divergence_{method}",
                        world=args.processes,
                    )[0]
                ),
                "idempotence": float(
                    _global_max(
                        jax,
                        local_idempotence,
                        axis_name=f"audit_idempotence_{method}",
                        world=args.processes,
                    )[0]
                ),
                "pressure_mean": float(
                    jnp.abs(
                        _global_sum(
                            jax,
                            pressure_sum,
                            axis_name=f"audit_pressure_{method}",
                            world=args.processes,
                        )[0]
                        / grid.cell_count
                    )
                ),
                "first_seconds_max": float(
                    _global_max(
                        jax,
                        first_time,
                        axis_name=f"audit_first_time_{method}",
                        world=args.processes,
                    )[0]
                ),
                "steady_seconds_max": float(
                    _global_max(
                        jax,
                        steady_time,
                        axis_name=f"audit_steady_time_{method}",
                        world=args.processes,
                    )[0]
                ),
                "velocity_dtype": str(result.velocity.x.payload.dtype),
                "pressure_dtype": str(result.pressure.payload.dtype),
            }
            results[method] = result

        reference = results[methods[0]]
        method_differences = {}
        for method in methods[1:]:
            candidate = results[method]
            local_difference = jnp.maximum(
                jnp.max(jnp.abs(reference.pressure.payload - candidate.pressure.payload)),
                jnp.maximum(
                    jnp.max(
                        jnp.abs(reference.velocity.x.payload - candidate.velocity.x.payload)
                    ),
                    jnp.maximum(
                        jnp.max(
                            jnp.abs(
                                reference.velocity.y.payload - candidate.velocity.y.payload
                            )
                        ),
                        jnp.max(
                            jnp.abs(
                                reference.velocity.z.owned.payload
                                - candidate.velocity.z.owned.payload
                            )
                        ),
                    ),
                ),
            )[None]
            method_differences[method] = float(
                _global_max(
                    jax,
                    local_difference,
                    axis_name=f"audit_method_{method}",
                    world=args.processes,
                )[0]
            )

        local_cells = local_z * grid.ny * grid.nx
        ownership_audit_passed = local_cells * args.processes == grid.cell_count
        if args.processes > 1:
            ownership_audit_passed = ownership_audit_passed and (
                local_cells < grid.cell_count
            )
        report = {
            "backend": jax.default_backend(),
            "dtype": args.dtype,
            "global_devices": jax.device_count(),
            "global_shape": [grid.nz, grid.ny, grid.nx],
            "local_cells_per_process": local_cells,
            "local_devices": jax.local_device_count(),
            "local_shape": [1, local_z, grid.ny, grid.nx],
            "method_differences": method_differences,
            "methods": reports,
            "ownership_audit_passed": ownership_audit_passed,
            "processes": jax.process_count(),
            "rank_holds_entire_domain": local_cells == grid.cell_count,
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
        "--methods",
        default="transpose,spike,spike-adaptive",
        help="comma-separated pressure methods",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--coordinator", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.processes not in (1, 2, 4):
        parser.error("the CPU gate supports 1, 2, or 4 processes")
    if args.worker and not args.coordinator:
        parser.error("workers require a coordinator address")
    return args


def main() -> int:
    args = _parse_args()
    return _worker(args) if args.worker else _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
