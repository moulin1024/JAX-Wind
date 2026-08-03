#!/usr/bin/env python3
"""Stage-level multi-rank performance profile for the GABLS1 y-slab path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from mpi4py import MPI


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
for source in (ROOT, SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


from benchmark.GABLS1 import run_mpi  # noqa: E402


def _profile_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--profile-output", type=Path)
    profile, runner_argv = parser.parse_known_args(argv)
    if profile.profile_repeats <= 0:
        parser.error("profile-repeats must be positive")
    runner = run_mpi.parse_args(runner_argv)
    return profile, runner


def main(argv: list[str] | None = None) -> None:
    profile, args = _profile_args(argv)
    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    process_count = communicator.Get_size()
    if process_count not in (1, 2, 4):
        raise SystemExit("the GABLS1 y-slab profiler requires 1, 2, or 4 ranks")
    if args.ny % process_count:
        raise SystemExit("ny must divide evenly across MPI ranks")

    jax = run_mpi._initialize_jax(communicator)
    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    coupled, case, dtype = run_mpi._build_distributed(args, jax)
    if args.restart is None:
        state = run_mpi._local_initial_state(args, coupled, case, dtype, rank)
    else:
        state, _ = run_mpi._local_restart_state(
            args,
            coupled,
            dtype,
            rank,
        )

    def ready(value) -> None:
        jax.block_until_ready(value)

    def measure(function, *, repeats: int | None = None, advance=False):
        count = profile.profile_repeats if repeats is None else repeats
        value = state
        function(value)
        ready(function(value))
        communicator.Barrier()
        start = time.perf_counter()
        for _ in range(count):
            argument = value if advance else state
            value = function(argument)
            ready(value)
        communicator.Barrier()
        elapsed = time.perf_counter() - start
        maximum = communicator.allreduce(elapsed, op=MPI.MAX)
        return maximum / count, value

    timings: dict[str, float] = {}
    timings["rates_s"], _ = measure(lambda current: coupled.rates(current))
    timings["surface_fluxes_s"], _ = measure(
        lambda current: coupled.surface_layer_fluxes(current)
    )
    timings["scalar_rhs_s"], _ = measure(
        lambda current: coupled._mapped_scalar_tendency(
            current.potential_temperature,
            current.velocity,
            current.time,
        )
    )
    timings["momentum_rhs_s"], _ = measure(
        lambda current: coupled._mapped_momentum_tendency(
            current.velocity,
            current.potential_temperature,
            current.time,
        )
    )
    timings["coupled_stage_rhs_s"], _ = measure(
        lambda current: coupled._mapped_coupled_tendency(
            current.velocity,
            current.potential_temperature,
            current.time,
        )
    )

    projection_iterations: list[int] = []

    def project(current):
        result = coupled.projector.project(
            coupled.enforce_boundaries(current.velocity),
            timestep=0.5,
            initial_pressure=current.pressure,
        )
        projection_iterations.append(result.linear_result.iterations)
        return result.velocity

    timings["pressure_projection_s"], _ = measure(project)
    timings["divergence_norm_s"], _ = measure(
        lambda current: coupled.divergence_norm(current.velocity)
    )
    timings["full_step_s"], advanced = measure(
        lambda current: coupled.step(current, timestep=0.5),
        advance=True,
    )

    detailed_state = state
    for _ in range(2):
        detailed_state = coupled.step(detailed_state, timestep=0.5)
        ready(detailed_state)

    stage_timings: dict[str, list[float]] = {
        "scalar_rhs": [],
        "momentum_rhs": [],
        "coupled_rhs": [],
        "projection": [],
        "velocity_sum": [],
    }

    def timed(name, function):
        def wrapper(*call_args, **call_kwargs):
            stage_start = time.perf_counter()
            result = function(*call_args, **call_kwargs)
            ready(result)
            stage_timings[name].append(time.perf_counter() - stage_start)
            return result

        return wrapper

    original_scalar_rhs = coupled._mapped_scalar_tendency
    original_momentum_rhs = coupled._mapped_momentum_tendency
    original_coupled_rhs = coupled._mapped_coupled_tendency
    original_projection = coupled._project
    original_velocity_sum = coupled._velocity_sum
    coupled._mapped_scalar_tendency = timed("scalar_rhs", original_scalar_rhs)
    coupled._mapped_momentum_tendency = timed(
        "momentum_rhs", original_momentum_rhs
    )
    coupled._mapped_coupled_tendency = timed(
        "coupled_rhs", original_coupled_rhs
    )
    coupled._project = timed("projection", original_projection)
    coupled._velocity_sum = timed("velocity_sum", original_velocity_sum)
    communicator.Barrier()
    detailed_start = time.perf_counter()
    detailed_state = coupled.step(detailed_state, timestep=0.5)
    ready(detailed_state)
    communicator.Barrier()
    detailed_full_step = communicator.allreduce(
        time.perf_counter() - detailed_start,
        op=MPI.MAX,
    )
    coupled._mapped_scalar_tendency = original_scalar_rhs
    coupled._mapped_momentum_tendency = original_momentum_rhs
    coupled._mapped_coupled_tendency = original_coupled_rhs
    coupled._project = original_projection
    coupled._velocity_sum = original_velocity_sum

    maximum_stage_timings = {
        name: [communicator.allreduce(value, op=MPI.MAX) for value in values]
        for name, values in stage_timings.items()
    }

    payload = {
        "schema": "jaxwind.gabls1.y-slab-profile.v1",
        "processes": communicator.Get_size(),
        "shape_zyx": list(coupled.grid.shape),
        "local_shape_zyx": [coupled.nz, coupled.local_y, coupled.nx],
        "step": state.step,
        "profile_repeats": profile.profile_repeats,
        "pressure_execution": coupled.pressure_solver.krylov.execution,
        "coupling_integrator": args.coupling_integrator,
        "advection_limiter": args.advection_limiter,
        "advection_halo_width": coupled.halo_width,
        "scalar_rhs_calls_per_step": (
            3 if args.coupling_integrator == "coupled-ssprk3" else 6
        ),
        "pressure_replication_level": coupled.pressure_solver.replication_level,
        "pressure_replicated_shape_zyx": list(
            coupled.pressure_solver.replicated_shape
        ),
        "pressure_iterations_median": float(np.median(projection_iterations)),
        "advanced_step": advanced.step,
        "detailed_full_step_s": detailed_full_step,
        "detailed_stage_s": maximum_stage_timings,
        **timings,
    }
    if rank == 0:
        encoded = json.dumps(payload, indent=2, sort_keys=True)
        print(encoded, flush=True)
        if profile.profile_output is not None:
            profile.profile_output.parent.mkdir(parents=True, exist_ok=True)
            profile.profile_output.write_text(encoded + "\n")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
