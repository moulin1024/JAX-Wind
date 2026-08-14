#!/usr/bin/env python3
"""Synchronously time the major JAX-Wind LASD step stages on a checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT, ROOT / "src", ROOT / "external" / "bw1000_benchmark"):
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
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--tridiag", choices=("thomas", "pcr"), default="thomas")
    parser.add_argument("--thomas-chunk", type=int, default=1)
    parser.add_argument("--padding-ratio", type=float, choices=(1.0, 1.5), default=1.5)
    parser.add_argument(
        "--dealiasing",
        choices=("three_halves", "two_thirds", "legacy_two_thirds"),
        default=None,
    )
    parser.add_argument(
        "--advection",
        choices=("conservative", "rotational"),
        default=None,
    )
    parser.add_argument(
        "--lasd-filter-backend",
        choices=("jax", "cufft"),
        default="jax",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    from applications.pressure_driven_lasd.config import load_case

    case = load_case(args.config)
    import jax

    jax.config.update("jax_enable_x64", case.numerics.dtype == "float64")
    import jax.numpy as jnp

    from applications.pressure_driven_lasd.problem import (
        build_pressure_driven_problem,
    )
    from jaxwind.domain import AcceptedClock, EvaluationTime, VerticalBoundary
    from jaxwind.effects import JaxRuntime, load_boussinesq_checkpoint
    from jaxwind.integrators import Evaluation
    from jaxwind.physics import BoussinesqVectorField
    from jaxwind.physics import ConservativeAdvection, RotationalAdvection

    runtime = JaxRuntime.from_initialized_jax(jax)
    checkpoint_problem = build_pressure_driven_problem(
        case,
        runtime=runtime,
        pressure_tridiag=args.tridiag,
        pressure_thomas_chunk=args.thomas_chunk,
        nonlinear_padding_ratio=args.padding_ratio,
        lasd_filter_backend=args.lasd_filter_backend,
    )
    state = load_boussinesq_checkpoint(
        runtime.checkpoint_path(args.checkpoint),
        layout=checkpoint_problem.solver.checkpoint_layout(jnp.asarray),
        config=checkpoint_problem.integrator,
        scale_fingerprint=checkpoint_problem.scales.fingerprint,
        closure_fingerprint=checkpoint_problem.closure_fingerprint,
        physics_fingerprint=checkpoint_problem.physics_fingerprint,
    )
    selected_advection = args.advection or case.flow.advection
    selected_dealiasing = args.dealiasing or case.numerics.nonlinear_dealiasing
    if (
        selected_advection == case.flow.advection
        and selected_dealiasing == case.numerics.nonlinear_dealiasing
    ):
        problem = checkpoint_problem
    else:
        selected_case = replace(
            case,
            flow=replace(case.flow, advection=selected_advection),
        )
        problem = build_pressure_driven_problem(
            selected_case,
            runtime=runtime,
            pressure_tridiag=args.tridiag,
            pressure_thomas_chunk=args.thomas_chunk,
            nonlinear_padding_ratio=args.padding_ratio,
            nonlinear_dealiasing=selected_dealiasing,
            lasd_filter_backend=args.lasd_filter_backend,
        )
        state = problem.solver.cold_start(
            state.fields,
            clock=state.clock,
        )
    solver = problem.solver
    jax.block_until_ready(state)
    algebra = solver._algebra
    model = solver.model
    dt = solver.integrator.dt
    fields = state.fields

    results: dict[str, float] = {}

    def measure(name, operation):
        value = operation()
        jax.block_until_ready(value)
        started = time.perf_counter()
        with jax.profiler.TraceAnnotation(name):
            for _ in range(args.repeats):
                value = operation()
            jax.block_until_ready(value)
        elapsed = time.perf_counter() - started
        results[name] = 1000.0 * elapsed / args.repeats
        return value

    transport_clock = AcceptedClock(float(state.clock.time), 0)
    update_clock = AcceptedClock(float(state.clock.time), 3)
    prepared_transport, _, context_transport = measure(
        "lasd_prepare_transport_ms",
        lambda: algebra.prepare_lasd_closure_with_context(
            fields,
            model,
            transport_clock,
            dt,
        ),
    )
    prepared_update, _, context_update = measure(
        "lasd_prepare_update_ms",
        lambda: algebra.prepare_lasd_closure_with_context(
            fields,
            model,
            update_clock,
            dt,
        ),
    )

    momentum_config = model.momentum.sgs
    old_momentum = fields.closure.momentum
    trajectories = measure(
        "lasd_accumulate_only_ms",
        lambda: algebra.lasd.accumulate(
            context_update.arrays.u,
            context_update.arrays.v,
            context_update.arrays.w_at_cells,
            old_momentum.trajectory_x.payload,
            old_momentum.trajectory_y.payload,
            old_momentum.trajectory_z.payload,
            momentum_config.update_interval,
        ),
    )
    measure(
        "lasd_update_momentum_only_ms",
        lambda: algebra.lasd.update_momentum(
            context_update.arrays,
            old_momentum.lm.payload,
            old_momentum.mm.payload,
            old_momentum.qn.payload,
            old_momentum.nn.payload,
            *trajectories,
            False,
            dt * momentum_config.update_interval,
            momentum_config.filter_grid_ratio,
            momentum_config.test_filter_ratio,
            momentum_config.timescale_coefficient,
            momentum_config.initial_coefficient,
            momentum_config.minimum_coefficient,
            momentum_config.maximum_coefficient,
            momentum_config.scale_dependent,
        ),
    )
    measure(
        "standalone_conservative_advection_ms",
        lambda: algebra.advection_tendency(
            context_update,
            ConservativeAdvection(),
            model.momentum.wall,
        ),
    )
    measure(
        "standalone_rotational_advection_ms",
        lambda: algebra.advection_tendency(
            context_update,
            RotationalAdvection(),
            model.momentum.wall,
        ),
    )
    measure(
        "standalone_wall_ms",
        lambda: algebra.wall_stress_tendency(
            context_update,
            model.momentum.wall,
        ),
    )
    measure(
        "standalone_sgs_ms",
        lambda: algebra.sgs_tendency(
            replace(context_update, closure=prepared_update.closure),
            model.momentum.sgs,
            model.momentum.wall,
        ),
    )

    evaluation = Evaluation(
        prepared_update,
        EvaluationTime(float(state.clock.time), int(state.clock.step), "profile"),
        None,
    )
    vector_field = BoussinesqVectorField(algebra, model)
    evaluated = measure(
        "fused_rhs_reused_context_ms",
        lambda: vector_field.evaluate_prepared(evaluation, context_update),
    )
    def update_and_evaluate(current_fields):
        prepared_fields, _, momentum_context = (
            algebra.prepare_lasd_closure_with_context(
                current_fields,
                model,
                update_clock,
                dt,
            )
        )
        prepared_evaluation = Evaluation(
            prepared_fields,
            EvaluationTime(
                float(state.clock.time),
                int(state.clock.step),
                "profile-combined",
            ),
            None,
        )
        evaluated_fields = vector_field.evaluate_prepared(
            prepared_evaluation,
            momentum_context,
        )
        tendency = evaluated_fields.tendency
        momentum_memory = prepared_fields.closure.momentum
        return (
            tendency.velocity.x.payload,
            tendency.velocity.y.payload,
            tendency.velocity.z.owned.payload,
            tendency.potential_temperature.payload,
            *(field.payload for field in momentum_memory.fields()),
        )

    compiled_update_and_evaluate = jax.jit(update_and_evaluate)
    measure(
        "compiled_lasd_update_and_rhs_ms",
        lambda: compiled_update_and_evaluate(fields),
    )
    previous = (
        state.history.value
        if hasattr(state.history, "value")
        else evaluated.tendency
    )
    candidate_velocity = measure(
        "ab2_candidate_velocity_ms",
        lambda: algebra.ab2_candidate_velocity(
            prepared_update.velocity,
            evaluated.tendency.velocity,
            previous.velocity,
            dt=dt,
            current_weight=1.5,
            previous_weight=-0.5,
        ),
    )
    measure(
        "ab2_candidate_scalar_ms",
        lambda: algebra.ab2_candidate_scalar(
            prepared_update.potential_temperature,
            evaluated.tendency.potential_temperature,
            previous.potential_temperature,
            dt=dt,
            current_weight=1.5,
            previous_weight=-0.5,
        ),
    )
    boundary = VerticalBoundary(0.0, 0.0)
    boundary_velocity = measure(
        "projection_boundary_filter_ms",
        lambda: algebra.enforce_normal_boundary(candidate_velocity, boundary),
    )
    divergence = measure(
        "projection_divergence_ms",
        lambda: algebra.velocity_divergence(boundary_velocity),
    )
    prepared_projection, prepared_divergence = measure(
        "projection_fused_prepare_ms",
        lambda: algebra.prepare_projection(candidate_velocity, boundary),
    )
    rhs = measure(
        "projection_rhs_scale_ms",
        lambda: algebra.pressure_rhs(divergence, 1.0 / dt),
    )
    pressure = measure(
        "pressure_solve_ms",
        lambda: solver._pressure_solver.solve(rhs),
    )
    gradient = measure(
        "projection_gradient_ms",
        lambda: algebra.pressure_gradient(pressure),
    )
    measure(
        "projection_velocity_correction_ms",
        lambda: algebra.correct_velocity(boundary_velocity, gradient, dt),
    )
    measure(
        "projection_fused_finish_ms",
        lambda: algebra.finish_projection(prepared_projection, pressure, dt),
    )
    measure(
        "dry_context_only_ms",
        lambda: algebra.dry_flow_context(fields.velocity),
    )

    results["lasd_prepare_weighted_average_ms"] = (
        3.0 * results["lasd_prepare_transport_ms"]
        + results["lasd_prepare_update_ms"]
    ) / 4.0
    results["measured_stage_sum_ms"] = sum(
        value
        for name, value in results.items()
        if name
        in {
            "lasd_prepare_weighted_average_ms",
            "fused_rhs_reused_context_ms",
            "ab2_candidate_velocity_ms",
            "ab2_candidate_scalar_ms",
            "projection_boundary_filter_ms",
            "projection_divergence_ms",
            "projection_rhs_scale_ms",
            "pressure_solve_ms",
            "projection_gradient_ms",
            "projection_velocity_correction_ms",
        }
    )
    results["measured_fused_stage_sum_ms"] = (
        results["measured_stage_sum_ms"]
        - results["projection_boundary_filter_ms"]
        - results["projection_divergence_ms"]
        - results["projection_gradient_ms"]
        - results["projection_velocity_correction_ms"]
        + results["projection_fused_prepare_ms"]
        + results["projection_fused_finish_ms"]
    )
    print(
        json.dumps(
            {
                "repeats": args.repeats,
                "tridiag": args.tridiag,
                "thomas_chunk": args.thomas_chunk,
                "padding_ratio": args.padding_ratio,
                "dealiasing": selected_dealiasing,
                "advection": selected_advection,
                "lasd_filter_backend": args.lasd_filter_backend,
                **results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
