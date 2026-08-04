#!/usr/bin/env python3
"""Compare short MP5-on/off continuations from one mature GABLS1 state."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
for source in (ROOT, SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


from benchmark.GABLS1 import diagnostics, run  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--pressure-rtol", type=float, default=1.0e-5)
    parser.add_argument("--pressure-max-iterations", type=int, default=40)
    parser.add_argument("--pressure-smooth", type=int, default=1)
    parser.add_argument("--pressure-coarse-smooth", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_results" / "gabls1_mp5_short_validation.json",
    )
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        parser.error("--dt must be positive and finite")
    if not args.restart.is_file():
        parser.error(f"checkpoint does not exist: {args.restart}")
    return args


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), 1.0e-30)
    return float(np.linalg.norm(candidate - reference)) / denominator


def _load_checkpoint(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as checkpoint:
        if str(checkpoint["checkpoint_schema"]) != run.CHECKPOINT_SCHEMA:
            raise SystemExit("restart checkpoint schema is not supported")
        return {key: np.asarray(checkpoint[key]).copy() for key in checkpoint.files}


def _build_continuation(
    checkpoint: dict[str, object],
    validation: argparse.Namespace,
    mp5_strength: float,
):
    import jax.numpy as jnp
    from jaxwind.pressure import MACVelocity

    nz, ny, nx = (int(value) for value in checkpoint["shape_zyx"])
    runner_args = run.parse_args(
        [
            "--nx", str(nx),
            "--ny", str(ny),
            "--nz", str(nz),
            "--amd-coefficient", str(float(checkpoint["amd_coefficient"])),
            "--scalar-amd-coefficient",
            str(float(checkpoint["scalar_amd_coefficient"])),
            "--mp5-strength", str(mp5_strength),
            "--coupling-integrator", "coupled-ssprk3",
            "--pressure-rtol", str(validation.pressure_rtol),
            "--pressure-max-iterations",
            str(validation.pressure_max_iterations),
            "--pressure-smooth", str(validation.pressure_smooth),
            "--pressure-coarse-smooth",
            str(validation.pressure_coarse_smooth),
        ]
    )
    coupled, case, dtype = run._build_coupled(runner_args)
    state = coupled.initial_state(
        MACVelocity(
            jnp.asarray(checkpoint["velocity_x"], dtype=dtype),
            jnp.asarray(checkpoint["velocity_y"], dtype=dtype),
            jnp.asarray(checkpoint["velocity_z"], dtype=dtype),
        ),
        jnp.asarray(checkpoint["potential_temperature"], dtype=dtype),
        pressure=jnp.asarray(checkpoint["pressure"], dtype=dtype),
        time=float(checkpoint["time"]),
        step=int(checkpoint["step"]),
    )
    return coupled, case, state


def _advance(
    checkpoint: dict[str, object],
    validation: argparse.Namespace,
    mp5_strength: float,
) -> tuple[dict[str, object], dict[str, np.ndarray | float]]:
    import jax
    import jax.numpy as jnp

    coupled, case, state = _build_continuation(
        checkpoint,
        validation,
        mp5_strength,
    )
    initial_theta = jnp.mean(
        state.potential_temperature
        - coupled.config.reference_potential_temperature
    )
    integrated_heat_flux = jnp.asarray(
        0.0,
        dtype=state.potential_temperature.dtype,
    )
    initial_rates = coupled.stability_rates(state)
    compile_start = time.perf_counter()
    compiled_state = coupled.step(state, timestep=validation.dt)
    jax.block_until_ready(compiled_state.velocity.x)
    compilation = time.perf_counter() - compile_start
    jax.block_until_ready((initial_theta, initial_rates))
    start = time.perf_counter()
    for _ in range(validation.steps):
        state = coupled.step(state, timestep=validation.dt)
        integrated_heat_flux = (
            integrated_heat_flux
            + validation.dt * coupled.last_surface_heat_flux_quadrature
        )
    jax.block_until_ready(state.velocity.x)
    runtime = time.perf_counter() - start
    final_rates = coupled.stability_rates(state)
    theta_after, heat_flux_after, divergence = (
        float(value) for value in coupled.accepted_state_metrics(state)
    )
    initial_theta = float(initial_theta)
    integrated_heat_flux = float(integrated_heat_flux)
    statistics = diagnostics.snapshot_statistics(coupled, state)
    arrays = (
        state.velocity.x,
        state.velocity.y,
        state.velocity.z,
        state.potential_temperature,
        state.pressure,
    )
    finite = all(bool(jnp.all(jnp.isfinite(value))) for value in arrays)
    initial_advective = float(initial_rates[0])
    final_advective = float(final_rates[0])
    budget_residual = abs(
        theta_after
        - initial_theta
        - integrated_heat_flux / case.domain
    )
    result: dict[str, object] = {
        "mp5_strength": mp5_strength,
        "finite": finite,
        "steps": validation.steps,
        "simulated_seconds": validation.steps * validation.dt,
        "runtime_s": runtime,
        "compilation_s": compilation,
        "seconds_per_step": runtime / validation.steps,
        "initial_cfl": validation.dt * initial_advective,
        "final_cfl": validation.dt * final_advective,
        "divergence_l2": divergence,
        "surface_heat_flux": heat_flux_after,
        "theta_budget_residual": budget_residual,
        "maximum_abs_w": float(statistics["maximum_abs_w"]),
        "boundary_layer_height": float(statistics["boundary_layer_height"]),
        "friction_velocity": float(statistics["friction_velocity"]),
    }
    return result, statistics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = _load_checkpoint(args.restart)
    baseline, baseline_statistics = _advance(checkpoint, args, 1.0)
    candidate, candidate_statistics = _advance(checkpoint, args, 0.0)
    profile_keys = (
        "u_mean",
        "v_mean",
        "theta_mean",
        "tke_resolved",
        "tke_total",
        "uw_total",
        "vw_total",
        "wtheta_total",
    )
    profile_relative_l2 = {
        key: _relative_l2(
            np.asarray(baseline_statistics[key]),
            np.asarray(candidate_statistics[key]),
        )
        for key in profile_keys
    }
    mp5_off_dissipation_max = float(
        np.max(np.abs(candidate_statistics["dissipation_mp5"]))
    )
    safe_cfl = max(
        float(baseline["initial_cfl"]),
        float(baseline["final_cfl"]),
        float(candidate["initial_cfl"]),
        float(candidate["final_cfl"]),
    ) <= 0.9
    checks = {
        "both_states_finite": bool(baseline["finite"] and candidate["finite"]),
        "face_envelope_cfl_at_most_0p9": safe_cfl,
        "divergence_below_5e-4": max(
            float(baseline["divergence_l2"]),
            float(candidate["divergence_l2"]),
        ) < 5.0e-4,
        "theta_budget_below_5e-5": max(
            float(baseline["theta_budget_residual"]),
            float(candidate["theta_budget_residual"]),
        ) < 5.0e-5,
        "mp5_off_dissipation_is_zero": mp5_off_dissipation_max < 1.0e-12,
    }
    report = {
        "schema": "jaxwind.gabls1.mp5-short-validation.v1",
        "checkpoint": str(args.restart.resolve()),
        "checkpoint_time_s": float(checkpoint["time"]),
        "checkpoint_step": int(checkpoint["step"]),
        "dt": args.dt,
        "steps": args.steps,
        "baseline_mp5_1": baseline,
        "candidate_mp5_0": candidate,
        "profile_relative_l2_mp5_0_vs_1": profile_relative_l2,
        "mp5_off_dissipation_max": mp5_off_dissipation_max,
        "checks": checks,
        "short_run_checks_passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["short_run_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
