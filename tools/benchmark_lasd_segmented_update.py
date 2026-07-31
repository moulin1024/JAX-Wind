#!/usr/bin/env python3
"""Benchmark monolithic and segmented accepted-step LASD updates."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp

from jaxwind.momentum import LASDModel, NeutralABLConfig, NeutralABLMomentum
from jaxwind.pressure import (
    BoundaryCondition,
    FGMRESConfig,
    MatrixFreePoissonSolver,
    PoissonBoundaryConditions,
    RectilinearGrid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def build_solver(n: int) -> NeutralABLMomentum:
    grid = RectilinearGrid.uniform(
        n,
        n,
        n,
        lx=4000.0,
        ly=2000.0,
        lz=1500.0,
    )
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions(
            periodic,
            periodic,
            periodic,
            periodic,
            neumann,
            neumann,
        ),
        dtype=jnp.float32,
        krylov=FGMRESConfig(
            restart=10,
            max_iterations=20,
            relative_tolerance=1.0e-4,
            execution="jax",
        ),
    )
    return NeutralABLMomentum(
        grid,
        pressure,
        NeutralABLConfig(
            friction_velocity=0.4,
            roughness_length=0.1,
            geostrophic_wind=(10.0, 0.0),
            coriolis_vertical=1.0e-4,
            coriolis_horizontal=1.0e-4,
            lasd=LASDModel(update_interval=1),
        ),
    )


def timed_compile(lowered) -> tuple[object, float]:
    start = time.perf_counter()
    executable = lowered.compile()
    return executable, time.perf_counter() - start


def elapsed_per_call(function, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        jax.block_until_ready(function())
    start = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(function())
    return (time.perf_counter() - start) / repeats


def main() -> None:
    args = parse_args()
    if min(args.n, args.warmup, args.repeats, args.rounds) <= 0:
        raise SystemExit("all benchmark controls must be positive")
    solver = build_solver(args.n)
    closure = solver.lasd_closure
    assert closure is not None
    velocity = solver.initial_log_profile(perturbation_amplitude=0.05)
    cells = solver.cell_centered_velocity(velocity)
    initial = closure.initialize(cells)
    interval_dt = jnp.asarray(1.0, dtype=cells.dtype)
    first_update = jnp.asarray(True)

    def legacy_update(state, values, gradient, timestep, first):
        ratio = closure.model.test_filter_ratio
        lm, mm = closure._contractions(values, gradient, ratio)
        qn, nn = closure._contractions(values, gradient, ratio**2)
        return closure.update_from_contractions(
            state,
            lm,
            mm,
            qn,
            nn,
            interval_dt=timestep,
            first_update=first,
        )

    legacy_accumulate, legacy_accumulate_compile = timed_compile(
        jax.jit(closure.accumulate).lower(initial, cells)
    )
    legacy_gradient, legacy_gradient_compile = timed_compile(
        jax.jit(solver.velocity_gradient).lower(cells)
    )
    accumulated = legacy_accumulate(initial, cells)
    gradient = legacy_gradient(cells)
    legacy_finalize, legacy_finalize_compile = timed_compile(
        jax.jit(legacy_update).lower(
            accumulated,
            cells,
            gradient,
            interval_dt,
            first_update,
        )
    )

    statistics_executable, statistics_compile = timed_compile(
        solver._compiled_lasd_statistics.lower(initial, cells)
    )
    statistics_state, stat_lm, stat_mm, stat_qn, stat_nn = (
        statistics_executable(initial, cells)
    )
    statistics_finalize, statistics_finalize_compile = timed_compile(
        solver._compiled_lasd_finalize.lower(
            statistics_state,
            stat_lm,
            stat_mm,
            stat_qn,
            stat_nn,
            interval_dt,
            first_update,
        )
    )

    def run_legacy():
        state = legacy_accumulate(initial, cells)
        local_gradient = legacy_gradient(cells)
        return legacy_finalize(
            state,
            cells,
            local_gradient,
            interval_dt,
            first_update,
        )

    def run_segmented():
        state, local_lm, local_mm, local_qn, local_nn = (
            statistics_executable(initial, cells)
        )
        return statistics_finalize(
            state,
            local_lm,
            local_mm,
            local_qn,
            local_nn,
            interval_dt,
            first_update,
        )

    legacy_times = []
    segmented_times = []
    for round_index in range(args.rounds):
        order = (
            (
                (run_legacy, legacy_times),
                (run_segmented, segmented_times),
            )
            if round_index % 2 == 0
            else (
                (run_segmented, segmented_times),
                (run_legacy, legacy_times),
            )
        )
        for function, samples in order:
            samples.append(
                elapsed_per_call(
                    function,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )
    legacy_median = statistics.median(legacy_times)
    segmented_median = statistics.median(segmented_times)
    print(
        json.dumps(
            {
                "grid": [args.n, args.n, args.n],
                "legacy_seconds": legacy_median,
                "segmented_seconds": segmented_median,
                "speedup": legacy_median / segmented_median,
                "legacy_compile_seconds": {
                    "accumulate": legacy_accumulate_compile,
                    "gradient": legacy_gradient_compile,
                    "monolithic_filter_and_finalize": legacy_finalize_compile,
                    "total": (
                        legacy_accumulate_compile
                        + legacy_gradient_compile
                        + legacy_finalize_compile
                    ),
                },
                "segmented_compile_seconds": {
                    "statistics": statistics_compile,
                    "finalize": statistics_finalize_compile,
                    "total": statistics_compile + statistics_finalize_compile,
                },
                "legacy_rounds": legacy_times,
                "segmented_rounds": segmented_times,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
