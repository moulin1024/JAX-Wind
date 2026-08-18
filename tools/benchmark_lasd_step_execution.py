#!/usr/bin/env python3
"""Compare dispatched and whole-step-JIT LASD execution on one checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PRESSURE_SOURCE = ROOT / "external" / "bw1000_benchmark"
for source in (ROOT, ROOT / "src", PRESSURE_SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "cases" / "DTU10MWPrecursor" / "config_lasd_benchmark.toml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "dtu10mw_lasd_benchmark_128x64x256"
            / "checkpoint_latest.npz"
        ),
    )
    parser.add_argument(
        "--checkpoint-update-interval",
        type=int,
        default=None,
        help=(
            "LASD update interval stored in the checkpoint fingerprint; "
            "the measured solver still uses the interval from --config"
        ),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tridiag", choices=("thomas", "pcr"), default="thomas")
    parser.add_argument("--thomas-chunk", type=int, default=1)
    parser.add_argument("--skip-compiled", action="store_true")
    parser.add_argument(
        "--cuda-profile-range",
        action="store_true",
        help="bracket only the eager timing loop with cudaProfilerStart/Stop",
    )
    parser.add_argument(
        "--reuse-rhs-momentum-context",
        choices=("true", "false"),
        default=None,
        help="override LASD-update derivative-context reuse",
    )
    parser.add_argument(
        "--lasd-filter-backend",
        choices=("jax", "cufft"),
        default=None,
        help="implementation used for the two-scale horizontal LASD filter",
    )
    return parser


def _ready(jax, value):
    return jax.block_until_ready(value)


def _measure(jax, transition, initial, steps: int) -> tuple[float, object]:
    current = initial
    started = time.perf_counter()
    for _ in range(steps):
        current = transition(current)
    _ready(jax, current)
    elapsed = time.perf_counter() - started
    return elapsed, current


def _warm(jax, transition, initial, steps: int):
    current = initial
    for _ in range(steps):
        current = transition(current)
    return _ready(jax, current)


def main() -> int:
    args = _parser().parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")

    from applications.pressure_driven_lasd.config import load_case

    case = load_case(args.config)
    import jax

    jax.config.update("jax_enable_x64", case.numerics.dtype == "float64")
    import jax.numpy as jnp

    from applications.pressure_driven_lasd.problem import (
        build_pressure_driven_problem,
    )
    from jaxwind.effects import JaxRuntime, load_boussinesq_checkpoint

    runtime = JaxRuntime.from_initialized_jax(jax)
    checkpoint_case = case
    if args.checkpoint_update_interval is not None:
        if args.checkpoint_update_interval <= 0:
            raise ValueError("checkpoint update interval must be positive")
        checkpoint_case = replace(
            case,
            sgs=replace(
                case.sgs,
                update_interval_steps=args.checkpoint_update_interval,
            ),
        )
    checkpoint_problem = build_pressure_driven_problem(
        checkpoint_case,
        runtime=runtime,
        pressure_tridiag=args.tridiag,
        pressure_thomas_chunk=args.thomas_chunk,
        lasd_filter_backend=args.lasd_filter_backend,
    )
    checkpoint_state = load_boussinesq_checkpoint(
        runtime.checkpoint_path(args.checkpoint),
        layout=checkpoint_problem.solver.checkpoint_layout(jnp.asarray),
        config=checkpoint_problem.integrator,
        scale_fingerprint=checkpoint_problem.scales.fingerprint,
        closure_fingerprint=checkpoint_problem.closure_fingerprint,
        physics_fingerprint=checkpoint_problem.physics_fingerprint,
    )
    selected_context_reuse = (
        case.sgs.reuse_rhs_momentum_context
        if args.reuse_rhs_momentum_context is None
        else args.reuse_rhs_momentum_context == "true"
    )
    selected_lasd_filter_backend = (
        case.sgs.lasd_filter_backend
        if args.lasd_filter_backend is None
        else args.lasd_filter_backend
    )
    if (
        args.checkpoint_update_interval is None
        and selected_context_reuse == case.sgs.reuse_rhs_momentum_context
    ):
        problem = checkpoint_problem
    else:
        benchmark_case = replace(
            case,
            sgs=replace(
                case.sgs,
                reuse_rhs_momentum_context=selected_context_reuse,
            ),
        )
        problem = build_pressure_driven_problem(
            benchmark_case,
            runtime=runtime,
            pressure_tridiag=args.tridiag,
            pressure_thomas_chunk=args.thomas_chunk,
            lasd_filter_backend=args.lasd_filter_backend,
        )
    reset_integrator_history = (
        args.checkpoint_update_interval is not None
    )
    benchmark_fields = checkpoint_state.fields
    if args.checkpoint_update_interval is not None:
        benchmark_fields = problem.solver.initialize_fields(benchmark_fields)
    if reset_integrator_history:
        initial = problem.solver.cold_start(
            benchmark_fields,
            clock=checkpoint_state.clock,
        )
    else:
        initial = checkpoint_state
    solver = problem.solver

    def transition(state):
        return solver.advance(
            state,
            compute_projection_residual=False,
        ).state

    eager_warm = _warm(jax, transition, initial, 8)
    cuda_profiler = None
    if args.cuda_profile_range:
        from cupy.cuda import profiler as cuda_profiler

        cuda_profiler.start()
    eager_elapsed, eager_final = _measure(
        jax,
        transition,
        eager_warm,
        args.steps,
    )
    if cuda_profiler is not None:
        cuda_profiler.stop()

    compile_elapsed = None
    compiled_elapsed = None
    compiled_rate = None
    speedup = None
    final = eager_final
    if not args.skip_compiled:
        compiled = jax.jit(transition)
        compile_started = time.perf_counter()
        compiled_warm = _warm(jax, compiled, initial, 8)
        compile_elapsed = time.perf_counter() - compile_started
        compiled_elapsed, final = _measure(
            jax,
            compiled,
            compiled_warm,
            args.steps,
        )
        compiled_rate = args.steps / compiled_elapsed
        speedup = eager_elapsed / compiled_elapsed

    velocity = final.fields.velocity
    u = velocity.x.payload
    v = velocity.y.payload
    w = velocity.z.owned.payload
    divergence = solver._algebra.velocity_divergence(velocity).payload
    advective_cfl = problem.integrator.dt * jnp.max(
        jnp.abs(u) / solver.grid.dx
        + jnp.abs(v) / solver.grid.dy
        + jnp.abs(w) / solver.grid.dz
    )
    diagnostics = _ready(
        jax,
        (
            jnp.all(jnp.isfinite(u))
            & jnp.all(jnp.isfinite(v))
            & jnp.all(jnp.isfinite(w)),
            0.5 * jnp.mean(u * u + v * v + w * w),
            jnp.max(jnp.abs(divergence)),
            advective_cfl,
        ),
    )

    print(
        json.dumps(
            {
                "steps": args.steps,
                "tridiag": args.tridiag,
                "thomas_chunk": args.thomas_chunk,
                "nonlinear_scheme": "legacy-fortran-pre-rhs-filtering",
                "reuse_rhs_momentum_context": selected_context_reuse,
                "lasd_filter_backend": selected_lasd_filter_backend,
                "update_interval_steps": case.sgs.update_interval_steps,
                "checkpoint_update_interval_steps": (
                    checkpoint_case.sgs.update_interval_steps
                ),
                "reset_integrator_history": reset_integrator_history,
                "initial_step": int(initial.clock.step),
                "final_step": int(final.clock.step),
                "warmup_steps": 8,
                "eager_elapsed_seconds": eager_elapsed,
                "eager_steps_per_second": args.steps / eager_elapsed,
                "whole_step_compile_and_first_seconds": compile_elapsed,
                "compiled_elapsed_seconds": compiled_elapsed,
                "compiled_steps_per_second": compiled_rate,
                "speedup": speedup,
                "finite_velocity": bool(diagnostics[0]),
                "mean_kinetic_energy": float(diagnostics[1]),
                "maximum_abs_divergence": float(diagnostics[2]),
                "maximum_advective_cfl": float(diagnostics[3]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
