#!/usr/bin/env python3
"""Run Nieuwstadt et al. (1993) with the non-spectral MAC+AMD solver."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
for source in (ROOT, SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


from benchmark.Nieuwstadt1993 import amd_diagnostics  # noqa: E402


HERE = Path(__file__).resolve().parent
INITIAL_ZI_FRACTION = 0.844
STABLE_THETA_GRADIENT = 0.003
ROUGHNESS_LENGTH = 0.16
DOMAIN = (6400.0, 6400.0, 2400.0)
CHECKPOINT_SCHEMA = "jaxwind.nieuwstadt1993.nonspectral-amd.v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--dt-max", type=float, default=1.25)
    parser.add_argument("--end-tstar", type=float, default=11.0)
    parser.add_argument("--sample-start-tstar", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=96)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--target-cfl", type=float, default=0.8)
    parser.add_argument("--target-diffusive-cfl", type=float, default=0.5)
    parser.add_argument("--amd-coefficient", type=float, default=0.212)
    parser.add_argument("--scalar-amd-coefficient", type=float)
    parser.add_argument("--mp5-strength", type=float, default=1.0)
    parser.add_argument("--pressure-rtol", type=float, default=1.0e-5)
    parser.add_argument("--pressure-max-iterations", type=int, default=40)
    parser.add_argument("--pressure-coarse-smooth", type=int, default=20)
    parser.add_argument(
        "--sgs-time-integration",
        choices=("explicit", "imex_ark3"),
        default="explicit",
    )
    parser.add_argument(
        "--projection-method",
        choices=("full", "fpj2"),
        default="full",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--max-run-seconds", type=float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "nieuwstadt1993_nonspectral_amd_40x40x48",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.scalar_amd_coefficient is None:
        args.scalar_amd_coefficient = args.amd_coefficient
    if args.quick:
        args.nx = args.ny = args.nz = 8
        args.dt_max = 0.25
        args.end_tstar = 0.01
        args.sample_start_tstar = 0.0
        args.max_steps = 4 if args.max_steps is None else args.max_steps
        args.sample_every = 1
        args.log_every = 1
        args.checkpoint_every = 2
    if min(args.nx, args.ny, args.nz) < 4:
        parser.error("all grid dimensions must be at least four")
    positive = {
        "dt-max": args.dt_max,
        "end-tstar": args.end_tstar,
        "target-cfl": args.target_cfl,
        "target-diffusive-cfl": args.target_diffusive_cfl,
        "pressure-rtol": args.pressure_rtol,
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
        parser.error("time, CFL, and pressure controls must be positive and finite")
    if not 0.0 <= args.sample_start_tstar < args.end_tstar:
        parser.error("sample-start-tstar must lie in [0, end-tstar)")
    if min(args.sample_every, args.log_every, args.checkpoint_every) <= 0:
        parser.error("sampling and checkpoint intervals must be positive")
    if min(args.pressure_max_iterations, args.pressure_coarse_smooth) <= 0:
        parser.error("pressure iteration controls must be positive")
    for name in ("amd_coefficient", "scalar_amd_coefficient", "mp5_strength"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"{name.replace('_', '-')} must be finite and nonnegative")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if args.max_run_seconds is not None and args.max_run_seconds <= 0.0:
        parser.error("max-run-seconds must be positive")
    return args


def _initial_state(args, coupled, case, dtype):
    import jax
    import jax.numpy as jnp
    from jaxwind.pressure import MACVelocity

    nz, ny, nx = coupled.grid.shape
    shape = (nz, ny, nx)
    key = jax.random.PRNGKey(args.seed)
    random_theta = jax.random.uniform(
        key,
        shape,
        minval=-0.5,
        maxval=0.5,
        dtype=dtype,
    )
    dz = coupled.momentum.dz
    z = (jnp.arange(nz, dtype=dtype) + 0.5) * dz
    z_face = (jnp.arange(nz, dtype=dtype) + 1.0) * dz
    initial_zi = INITIAL_ZI_FRACTION * case.zi0
    cell_weight = jnp.maximum(1.0 - z / initial_zi, 0.0)
    face_weight = jnp.maximum(1.0 - z_face / initial_zi, 0.0)
    theta = jnp.where(
        (z < initial_zi)[:, None, None],
        0.1 * random_theta * cell_weight[:, None, None] * case.theta_star0,
        (z - initial_zi)[:, None, None] * STABLE_THETA_GRADIENT,
    )
    w = jnp.zeros((nz + 1, ny, nx), dtype=dtype)
    w = w.at[1:].set(
        jnp.where(
            (z_face < initial_zi)[:, None, None],
            0.1 * random_theta * face_weight[:, None, None] * case.wstar0,
            0.0,
        )
    )
    velocity = MACVelocity(
        jnp.zeros((nz, ny, nx + 1), dtype=dtype),
        jnp.zeros((nz, ny + 1, nx), dtype=dtype),
        w,
    )
    projected = coupled.momentum.projector.project_velocity_and_pressure(
        coupled.momentum.enforce_boundaries(velocity),
        timestep=1.0,
    )
    return coupled.initial_state(
        projected.velocity,
        theta,
        pressure=projected.pressure,
    )


def _restore_checkpoint(args, coupled, dtype):
    import jax.numpy as jnp
    from jaxwind.pressure import MACVelocity

    checkpoint = np.load(args.restart, allow_pickle=False)
    if str(checkpoint["checkpoint_schema"]) != CHECKPOINT_SCHEMA:
        raise SystemExit("restart checkpoint schema is not supported")
    expected_shape = np.asarray(coupled.grid.shape)
    if not np.array_equal(checkpoint["shape_zyx"], expected_shape):
        raise SystemExit("restart grid shape does not match")
    for key, expected in (
        ("amd_coefficient", args.amd_coefficient),
        ("scalar_amd_coefficient", args.scalar_amd_coefficient),
        ("mp5_strength", args.mp5_strength),
    ):
        if not np.isclose(float(checkpoint[key]), expected, rtol=0.0, atol=1.0e-12):
            raise SystemExit(f"restart {key} does not match")
    checkpoint_sgs_time_integration = (
        str(checkpoint["sgs_time_integration"])
        if "sgs_time_integration" in checkpoint.files
        else "explicit"
    )
    checkpoint_projection_method = (
        str(checkpoint["projection_method"])
        if "projection_method" in checkpoint.files
        else "full"
    )
    if checkpoint_sgs_time_integration != args.sgs_time_integration:
        raise SystemExit("restart SGS time integration does not match")
    if checkpoint_projection_method != args.projection_method:
        raise SystemExit("restart projection method does not match")
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
    if (
        args.projection_method == "fpj2"
        and "fpj2_current_pressure" in checkpoint.files
    ):
        from jaxwind.momentum import FPJ2State

        coupled.momentum.restore_fpj2(
            FPJ2State(
                jnp.asarray(checkpoint["fpj2_current_pressure"], dtype=dtype),
                jnp.asarray(checkpoint["fpj2_previous_pressure"], dtype=dtype),
                float(checkpoint["fpj2_current_timestep"]),
                float(checkpoint["fpj2_previous_timestep"]),
                int(checkpoint["fpj2_history_count"]),
            )
        )
    sample_keys = tuple(str(value) for value in checkpoint["sample_keys"])
    sample_count = int(checkpoint["sample_count"])
    samples = [
        {key: np.asarray(checkpoint[f"samples_{key}"][index]) for key in sample_keys}
        for index in range(sample_count)
    ]
    sample_times = list(np.asarray(checkpoint["sample_times"], dtype=float))
    time_keys = tuple(str(value) for value in checkpoint["time_keys"])
    time_count = int(checkpoint["time_count"])
    time_rows = [
        {key: float(checkpoint[f"time_rows_{key}"][index]) for key in time_keys}
        for index in range(time_count)
    ]
    return {
        "state": state,
        "initial_theta_mean": float(checkpoint["initial_theta_mean"]),
        "samples": samples,
        "sample_times": sample_times,
        "time_rows": time_rows,
        "timesteps": list(np.asarray(checkpoint["timesteps"], dtype=float)),
        "max_cfl": float(checkpoint["max_cfl"]),
        "max_diffusive_cfl": float(checkpoint["max_diffusive_cfl"]),
        "max_divergence": float(checkpoint["max_divergence"]),
        "max_scalar_budget_error": float(checkpoint["max_scalar_budget_error"]),
    }


def run(args: argparse.Namespace) -> dict[str, float | str]:
    from jax import config as jax_config

    if not args.single:
        jax_config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp

    from jaxwind.momentum import (
        AMDBoussinesq,
        AMDBoussinesqConfig,
        AMDModel,
        AMDPassiveScalar,
        AMDPassiveScalarModel,
        NeutralABLConfig,
        NeutralABLMomentum,
    )
    from jaxwind.pressure import (
        BoundaryCondition,
        GMGConfig,
        MatrixFreePoissonSolver,
        PCGConfig,
        PoissonBoundaryConditions,
        RectilinearGrid,
        mac_divergence,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case = amd_diagnostics.NieuwstadtCase()
    dtype = jnp.float32 if args.single else jnp.float64
    grid = RectilinearGrid.uniform(
        args.nx,
        args.ny,
        args.nz,
        lx=DOMAIN[0],
        ly=DOMAIN[1],
        lz=DOMAIN[2],
    )
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    pressure_solver = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions(
            periodic,
            periodic,
            periodic,
            periodic,
            neumann,
            neumann,
        ),
        dtype=dtype,
        gmg=GMGConfig(
            smoother="auto",
            coarsening="auto",
            coarse_smooth=args.pressure_coarse_smooth,
        ),
        krylov=PCGConfig(
            max_iterations=args.pressure_max_iterations,
            relative_tolerance=args.pressure_rtol,
            execution="jax",
        ),
    )
    momentum = NeutralABLMomentum(
        grid,
        pressure_solver,
        NeutralABLConfig(
            friction_velocity=case.wstar0,
            roughness_length=ROUGHNESS_LENGTH,
            pressure_acceleration=0.0,
            geostrophic_wind=None,
            coriolis_vertical=0.0,
            coriolis_horizontal=0.0,
            mp5_dissipation_strength=args.mp5_strength,
            amd=AMDModel(coefficient=args.amd_coefficient),
            sgs_time_integration=args.sgs_time_integration,
            projection_method=args.projection_method,
        ),
    )
    scalar = AMDPassiveScalar(
        grid,
        AMDPassiveScalarModel(
            coefficient=args.scalar_amd_coefficient,
            lower_surface_flux=case.surface_theta_flux,
            upper_surface_flux=0.0,
            mp5_dissipation_strength=args.mp5_strength,
        ),
    )
    coupled = AMDBoussinesq(
        momentum,
        scalar,
        AMDBoussinesqConfig(
            gravity=case.gravity,
            reference_potential_temperature=case.theta0,
        ),
    )

    if args.restart is None:
        state = _initial_state(args, coupled, case, dtype)
        initial_theta_mean = float(jnp.mean(state.potential_temperature))
        samples: list[dict] = []
        sample_times: list[float] = []
        time_rows: list[dict] = []
        timesteps: list[float] = []
        max_cfl = 0.0
        max_diffusive_cfl = 0.0
        max_divergence = 0.0
        max_scalar_budget_error = 0.0
    else:
        restored = _restore_checkpoint(args, coupled, dtype)
        state = restored["state"]
        initial_theta_mean = restored["initial_theta_mean"]
        samples = restored["samples"]
        sample_times = restored["sample_times"]
        time_rows = restored["time_rows"]
        timesteps = restored["timesteps"]
        max_cfl = restored["max_cfl"]
        max_diffusive_cfl = restored["max_diffusive_cfl"]
        max_divergence = restored["max_divergence"]
        max_scalar_budget_error = restored["max_scalar_budget_error"]

    diagnostic_kernel = jax.jit(coupled.diagnostic_fields)
    maximum_mode = (
        math.sqrt(2.0)
        * math.pi
        * max(args.nx, args.ny)
        * case.zi0
        / DOMAIN[0]
    )
    spectrum_edges = np.linspace(0.0, maximum_mode, args.nx // 2 + 2)

    compile_dt = min(args.dt_max, 0.25 if args.quick else args.dt_max)
    saved_fpj2 = momentum.fpj2_state
    compile_start = time.perf_counter()
    compiled_state = coupled.step(state, timestep=compile_dt)
    compiled_fields = diagnostic_kernel(state)
    jax.block_until_ready(compiled_state.velocity.x)
    jax.block_until_ready(compiled_fields.sgs_tke)
    compilation_s = time.perf_counter() - compile_start
    if saved_fpj2 is None:
        momentum.reset_fpj2()
    else:
        momentum.restore_fpj2(saved_fpj2)
    print(f"[compile] non-spectral CBL kernels ready in {compilation_s:.3f}s", flush=True)

    def collect_sample() -> dict:
        return amd_diagnostics.snapshot_statistics(
            state,
            coupled,
            diagnostic_kernel,
            case=case,
            spectrum_edges=spectrum_edges,
        )

    def append_sample(statistics: dict) -> None:
        samples.append(statistics)
        normalized_time = state.time / case.tstar0
        sample_times.append(normalized_time)
        time_rows.append(
            {
                "step": state.step,
                "time_s": state.time,
                "time_over_tstar0": normalized_time,
                "zi": statistics["zi"],
                "zi_over_zi0": statistics["zi"] / case.zi0,
                "wstar": statistics["wstar"],
                "energy_bl_over_wstar0_sq": statistics["energy_bl"]
                / case.wstar0**2,
            }
        )

    def save_checkpoint() -> None:
        sample_keys = tuple(samples[0]) if samples else ()
        time_keys = tuple(time_rows[0]) if time_rows else ()
        payload: dict[str, object] = {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "shape_zyx": np.asarray(grid.shape),
            "velocity_x": np.asarray(state.velocity.x),
            "velocity_y": np.asarray(state.velocity.y),
            "velocity_z": np.asarray(state.velocity.z),
            "potential_temperature": np.asarray(state.potential_temperature),
            "pressure": np.asarray(state.pressure),
            "time": state.time,
            "step": state.step,
            "initial_theta_mean": initial_theta_mean,
            "timesteps": np.asarray(timesteps),
            "amd_coefficient": args.amd_coefficient,
            "scalar_amd_coefficient": args.scalar_amd_coefficient,
            "mp5_strength": args.mp5_strength,
            "sgs_time_integration": args.sgs_time_integration,
            "projection_method": args.projection_method,
            "sample_keys": np.asarray(sample_keys),
            "sample_count": len(samples),
            "sample_times": np.asarray(sample_times),
            "time_keys": np.asarray(time_keys),
            "time_count": len(time_rows),
            "max_cfl": max_cfl,
            "max_diffusive_cfl": max_diffusive_cfl,
            "max_divergence": max_divergence,
            "max_scalar_budget_error": max_scalar_budget_error,
        }
        fpj2 = momentum.fpj2_state
        if fpj2 is not None:
            payload.update(
                {
                    "fpj2_current_pressure": np.asarray(fpj2.current_pressure),
                    "fpj2_previous_pressure": np.asarray(fpj2.previous_pressure),
                    "fpj2_current_timestep": fpj2.current_timestep,
                    "fpj2_previous_timestep": fpj2.previous_timestep,
                    "fpj2_history_count": fpj2.history_count,
                }
            )
        for key in sample_keys:
            payload[f"samples_{key}"] = np.stack(
                [np.asarray(sample[key]) for sample in samples]
            )
        for key in time_keys:
            payload[f"time_rows_{key}"] = np.asarray(
                [row[key] for row in time_rows]
            )
        destination = args.output_dir / "checkpoint.npz"
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, destination)

    final_time = args.end_tstar * case.tstar0
    start = time.perf_counter()
    stopped_early = False
    while state.time < final_time:
        timestep = min(
            coupled.timestep_for_cfl(
                state,
                args.target_cfl,
                args.target_diffusive_cfl,
            ),
            args.dt_max,
            final_time - state.time,
        )
        state = coupled.step(state, timestep=timestep)
        timesteps.append(timestep)
        cfl = float(timestep * momentum.cfl_rate(state.velocity))
        momentum_diffusive = float(
            timestep
            * 2.0
            * jnp.max(momentum.sgs_viscosity(momentum.cell_centered_velocity(state.velocity)))
            * (
                1.0 / momentum.dx**2
                + 1.0 / momentum.dy**2
                + 1.0 / momentum.dz**2
            )
        )
        scalar_diffusive = float(
            timestep
            * scalar.diffusive_rate(state.potential_temperature, state.velocity)
        )
        diffusive_cfl = max(momentum_diffusive, scalar_diffusive)
        divergence = float(
            pressure_solver.operator.norm(mac_divergence(state.velocity, grid))
        )
        expected_theta_mean = (
            initial_theta_mean
            + state.time * case.surface_theta_flux / DOMAIN[2]
        )
        scalar_budget_error = abs(
            float(jnp.mean(state.potential_temperature)) - expected_theta_mean
        )
        max_cfl = max(max_cfl, cfl)
        max_diffusive_cfl = max(max_diffusive_cfl, diffusive_cfl)
        max_divergence = max(max_divergence, divergence)
        max_scalar_budget_error = max(
            max_scalar_budget_error,
            scalar_budget_error,
        )
        reached_step_limit = (
            args.max_steps is not None and state.step >= args.max_steps
        )
        final = state.time >= final_time or reached_step_limit
        if state.step % args.sample_every == 0 or final:
            statistics = collect_sample()
            nonfinite = [
                name
                for name, value in statistics.items()
                if not np.all(np.isfinite(value))
            ]
            if nonfinite:
                raise FloatingPointError(
                    "non-finite diagnostic: " + ", ".join(nonfinite)
                )
            append_sample(statistics)
        if state.step % args.log_every == 0 or final:
            latest = time_rows[-1] if time_rows else None
            zi_text = "n/a" if latest is None else f"{latest['zi_over_zi0']:.4f}"
            print(
                f"step={state.step} t/t*={state.time / case.tstar0:.4f}/"
                f"{args.end_tstar:g} zi/zi0={zi_text} CFL={cfl:.4f} "
                f"CFLnu={diffusive_cfl:.4f} divL2={divergence:.3e} "
                f"theta_budget={scalar_budget_error:.3e}",
                flush=True,
            )
        if state.step % args.checkpoint_every == 0 or final:
            save_checkpoint()
        if reached_step_limit:
            stopped_early = state.time < final_time
            break
        if (
            args.max_run_seconds is not None
            and time.perf_counter() - start >= args.max_run_seconds
        ):
            save_checkpoint()
            stopped_early = True
            break

    runtime_s = time.perf_counter() - start
    if not samples:
        append_sample(collect_sample())
        save_checkpoint()
    selected = [
        sample
        for normalized_time, sample in zip(sample_times, samples, strict=True)
        if args.sample_start_tstar <= normalized_time <= args.end_tstar
    ]
    if not selected:
        selected = samples
    summary = amd_diagnostics.save_outputs(
        args.output_dir,
        args=args,
        case=case,
        coupled=coupled,
        time_rows=time_rows,
        selected=selected,
        runtime_s=runtime_s,
        max_cfl=max_cfl,
        max_diffusive_cfl=max_diffusive_cfl,
        max_divergence=max_divergence,
        max_scalar_budget_error=max_scalar_budget_error,
        spectrum_edges=spectrum_edges,
    )
    summary["compilation_s"] = compilation_s
    summary["stopped_early"] = str(stopped_early).lower()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
