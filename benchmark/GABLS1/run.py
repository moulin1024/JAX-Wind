#!/usr/bin/env python3
"""Run the GABLS1 stable-boundary-layer benchmark with MAC+AMD."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
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


from benchmark.GABLS1 import diagnostics  # noqa: E402


HERE = Path(__file__).resolve().parent
CHECKPOINT_SCHEMA = "jaxwind.gabls1.kep4-ko6-mp5.v2"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    days, remainder = divmod(total_seconds, 24 * 3600)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d{clock}" if days else clock


def _eta_log_fields(
    *,
    start_wall: float,
    start_simulation_time: float,
    simulation_time: float,
    final_simulation_time: float,
    current_wall: float | None = None,
    now: datetime | None = None,
) -> str:
    """Format elapsed time, measured throughput, and projected completion."""
    wall = time.perf_counter() if current_wall is None else current_wall
    elapsed_wall = max(0.0, wall - start_wall)
    advanced_simulation = simulation_time - start_simulation_time
    if elapsed_wall <= 0.0 or advanced_simulation <= 0.0:
        return (
            f"wall={_format_duration(elapsed_wall)} speed=calculating "
            "remain=calculating ETA=calculating"
        )
    simulation_per_wall = advanced_simulation / elapsed_wall
    remaining_simulation = max(0.0, final_simulation_time - simulation_time)
    remaining_wall = remaining_simulation / simulation_per_wall
    current_time = datetime.now().astimezone() if now is None else now
    completion = current_time + timedelta(seconds=remaining_wall)
    return (
        f"wall={_format_duration(elapsed_wall)} "
        f"speed={simulation_per_wall:.2f}x "
        f"remain={_format_duration(remaining_wall)} "
        f"ETA={completion.isoformat(timespec='seconds')}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--end-hours", type=float, default=9.0)
    parser.add_argument("--sample-start-hours", type=float, default=8.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=60.0)
    parser.add_argument("--dt-max", type=float, default=1.0)
    parser.add_argument("--target-cfl", type=float, default=0.9)
    parser.add_argument("--target-diffusive-cfl", type=float, default=0.5)
    parser.add_argument("--amd-coefficient", type=float, default=0.212)
    parser.add_argument("--scalar-amd-coefficient", type=float)
    parser.add_argument("--mp5-strength", type=float, default=1.0)
    parser.add_argument(
        "--momentum-advection",
        choices=("centered2", "kep4"),
        default="kep4",
    )
    parser.add_argument(
        "--momentum-regularization",
        choices=("none", "mp5", "ko6"),
        default="ko6",
    )
    parser.add_argument("--ko6-strength", type=float, default=1.0)
    parser.add_argument(
        "--scalar-advection",
        choices=("centered_mp5", "mp5"),
        default="mp5",
    )
    parser.add_argument(
        "--projection-method",
        choices=("full", "fpj2"),
        default="fpj2",
    )
    parser.add_argument(
        "--fpj2-timestep-ratio-limit",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--coupling-integrator",
        choices=("strang", "coupled-ssprk3"),
        default="coupled-ssprk3",
    )
    parser.add_argument("--pressure-rtol", type=float, default=1.0e-5)
    parser.add_argument("--pressure-max-iterations", type=int, default=40)
    parser.add_argument("--pressure-smooth", type=int, default=1)
    parser.add_argument("--pressure-coarse-smooth", type=int, default=20)
    parser.add_argument("--y-slab-coarse-cells-per-rank", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--metrics-every", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-run-seconds", type=float)
    parser.add_argument("--restart", type=Path)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=HERE / "reference" / "official_12p5m",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "gabls1_amd_32cubed",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.scalar_amd_coefficient is None:
        args.scalar_amd_coefficient = args.amd_coefficient
    if args.quick and args.smoke:
        parser.error("--quick and --smoke are mutually exclusive")
    if args.quick:
        args.nx = args.ny = args.nz = 8
        args.end_hours = 1.0 / 3600.0
        args.sample_start_hours = 0.0
        args.sample_interval_seconds = 0.25
        args.dt_max = 0.25
        args.max_steps = 4 if args.max_steps is None else args.max_steps
        args.checkpoint_every = 2
        args.log_every = 1
        args.metrics_every = 1
    if args.smoke:
        args.nx = args.ny = args.nz = 16
        args.end_hours = 0.02
        args.sample_start_hours = 0.0
        args.sample_interval_seconds = 10.0
        args.dt_max = min(args.dt_max, 0.5)
        args.checkpoint_every = min(args.checkpoint_every, 50)
        args.log_every = min(args.log_every, 20)
        args.metrics_every = min(args.metrics_every, 20)
    if min(args.nx, args.ny, args.nz) < 4:
        parser.error("all grid dimensions must be at least four")
    positive = {
        "end-hours": args.end_hours,
        "sample-interval-seconds": args.sample_interval_seconds,
        "dt-max": args.dt_max,
        "target-cfl": args.target_cfl,
        "target-diffusive-cfl": args.target_diffusive_cfl,
        "pressure-rtol": args.pressure_rtol,
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
        parser.error("time, CFL, and pressure controls must be positive and finite")
    if (
        not math.isfinite(args.fpj2_timestep_ratio_limit)
        or args.fpj2_timestep_ratio_limit < 1.0
    ):
        parser.error("fpj2-timestep-ratio-limit must be finite and at least one")
    if not 0.0 <= args.sample_start_hours < args.end_hours:
        parser.error("sample-start-hours must lie in [0, end-hours)")
    for name in (
        "amd_coefficient",
        "scalar_amd_coefficient",
        "mp5_strength",
        "ko6_strength",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"{name.replace('_', '-')} must be finite and nonnegative")
    if (
        min(
            args.pressure_max_iterations,
            args.pressure_coarse_smooth,
            args.y_slab_coarse_cells_per_rank,
            args.checkpoint_every,
            args.log_every,
            args.metrics_every,
        )
        <= 0
    ):
        parser.error("iteration and output intervals must be positive")
    if args.pressure_smooth < 0:
        parser.error("pressure-smooth must be nonnegative")
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
    z = (jnp.arange(nz, dtype=dtype) + 0.5) * coupled.scalar.dz
    profile = jnp.where(
        z <= case.inversion_base,
        case.theta_initial,
        case.theta_initial + case.inversion_gradient * (z - case.inversion_base),
    )
    perturbation = jax.random.uniform(
        jax.random.PRNGKey(args.seed),
        (nz, ny, nx),
        dtype=dtype,
        minval=-0.1,
        maxval=0.1,
    )
    perturbation -= jnp.mean(perturbation, axis=(1, 2), keepdims=True)
    perturbation *= (z < 50.0)[:, None, None]
    theta = profile[:, None, None] + perturbation
    velocity = MACVelocity(
        jnp.full((nz, ny, nx + 1), case.geostrophic_u, dtype=dtype),
        jnp.full((nz, ny + 1, nx), case.geostrophic_v, dtype=dtype),
        jnp.zeros((nz + 1, ny, nx), dtype=dtype),
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


def _pack_records(payload: dict[str, object], prefix: str, records: list[dict]):
    keys = tuple(records[0]) if records else ()
    payload[f"{prefix}_keys"] = np.asarray(keys)
    payload[f"{prefix}_count"] = len(records)
    for key in keys:
        payload[f"{prefix}_{key}"] = np.stack(
            [np.asarray(record[key]) for record in records]
        )


def _unpack_records(checkpoint, prefix: str) -> list[dict]:
    keys = tuple(str(value) for value in checkpoint[f"{prefix}_keys"])
    count = int(checkpoint[f"{prefix}_count"])
    return [
        {key: np.asarray(checkpoint[f"{prefix}_{key}"][index]) for key in keys}
        for index in range(count)
    ]


def _restore_checkpoint(args, coupled, dtype):
    import jax.numpy as jnp
    from jaxwind.momentum import FPJ2State
    from jaxwind.pressure import MACVelocity

    checkpoint = np.load(args.restart, allow_pickle=False)
    if str(checkpoint["checkpoint_schema"]) != CHECKPOINT_SCHEMA:
        raise SystemExit("restart checkpoint schema is not supported")
    if not np.array_equal(checkpoint["shape_zyx"], coupled.grid.shape):
        raise SystemExit("restart grid shape does not match")
    for key in (
        "amd_coefficient",
        "scalar_amd_coefficient",
        "mp5_strength",
        "ko6_strength",
    ):
        if not np.isclose(float(checkpoint[key]), getattr(args, key)):
            raise SystemExit(f"restart {key} does not match")
    for key in (
        "momentum_advection",
        "momentum_regularization",
        "scalar_advection",
    ):
        if str(checkpoint[key]) != getattr(args, key):
            raise SystemExit(f"restart {key} does not match")
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
    if args.projection_method == "fpj2" and "fpj2_current_pressure" in checkpoint:
        coupled.momentum.restore_fpj2(
            FPJ2State(
                jnp.asarray(checkpoint["fpj2_current_pressure"], dtype=dtype),
                jnp.asarray(checkpoint["fpj2_previous_pressure"], dtype=dtype),
                float(checkpoint["fpj2_current_timestep"]),
                float(checkpoint["fpj2_previous_timestep"]),
                int(checkpoint["fpj2_history_count"]),
            )
        )
    return {
        "state": state,
        "samples": _unpack_records(checkpoint, "samples"),
        "time_rows": _unpack_records(checkpoint, "time_rows"),
        "timesteps": list(np.asarray(checkpoint["timesteps"], dtype=float)),
        "max_cfl": float(checkpoint["max_cfl"]),
        "max_diffusive_cfl": float(checkpoint["max_diffusive_cfl"]),
        "max_divergence": float(checkpoint["max_divergence"]),
        "max_scalar_budget_residual": float(checkpoint["max_scalar_budget_residual"]),
    }


def _build_coupled(args: argparse.Namespace):
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
    )

    case = diagnostics.GABLS1Case()
    dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    grid = RectilinearGrid.uniform(
        args.nx,
        args.ny,
        args.nz,
        lx=case.domain,
        ly=case.domain,
        lz=case.domain,
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
            pre_smooth=args.pressure_smooth,
            post_smooth=args.pressure_smooth,
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
            friction_velocity=0.3,
            roughness_length=case.roughness_length,
            pressure_acceleration=0.0,
            geostrophic_wind=(case.geostrophic_u, case.geostrophic_v),
            coriolis_vertical=case.coriolis,
            coriolis_horizontal=0.0,
            mp5_dissipation_strength=args.mp5_strength,
            advection_scheme=args.momentum_advection,
            regularization_scheme=args.momentum_regularization,
            ko6_dissipation_strength=args.ko6_strength,
            amd=AMDModel(coefficient=args.amd_coefficient),
            sgs_time_integration="explicit",
            projection_method=args.projection_method,
            fpj2_timestep_ratio_limit=args.fpj2_timestep_ratio_limit,
        ),
    )
    scalar = AMDPassiveScalar(
        grid,
        AMDPassiveScalarModel(
            coefficient=args.scalar_amd_coefficient,
            lower_surface_flux=0.0,
            upper_surface_flux=0.0,
            mp5_dissipation_strength=args.mp5_strength,
            advection_scheme=args.scalar_advection,
        ),
    )
    coupled = AMDBoussinesq(
        momentum,
        scalar,
        AMDBoussinesqConfig(
            gravity=case.gravity,
            reference_potential_temperature=case.theta_reference,
            surface_potential_temperature=case.theta_initial,
            surface_temperature_tendency=case.surface_cooling_rate,
            thermal_roughness_length=case.roughness_length,
            coupling_integrator=args.coupling_integrator,
        ),
    )
    return coupled, case, dtype


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    from jax import config as jax_config

    if args.dtype == "float64":
        jax_config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coupled, case, dtype = _build_coupled(args)
    grid = coupled.grid
    momentum = coupled.momentum

    if args.restart is None:
        state = _initial_state(args, coupled, case, dtype)
        samples: list[dict] = []
        time_rows: list[dict] = []
        timesteps: list[float] = []
        max_cfl = 0.0
        max_diffusive_cfl = 0.0
        max_divergence = 0.0
        max_scalar_budget_residual = 0.0
    else:
        restored = _restore_checkpoint(args, coupled, dtype)
        state = restored["state"]
        samples = restored["samples"]
        time_rows = restored["time_rows"]
        timesteps = restored["timesteps"]
        max_cfl = restored["max_cfl"]
        max_diffusive_cfl = restored["max_diffusive_cfl"]
        max_divergence = restored["max_divergence"]
        max_scalar_budget_residual = restored["max_scalar_budget_residual"]

    diagnostic_kernel = jax.jit(coupled.diagnostic_fields)
    compile_start = time.perf_counter()
    saved_fpj2 = momentum.fpj2_state
    if args.projection_method == "fpj2":
        momentum.reset_fpj2()
    compiled_state = coupled.step(state, timestep=min(args.dt_max, 0.25))
    if args.projection_method == "fpj2":
        compiled_state = coupled.step(
            compiled_state,
            timestep=min(args.dt_max, 0.25),
        )
        compiled_state = coupled.step(
            compiled_state,
            timestep=min(args.dt_max, 0.25),
        )
    compiled_fields = diagnostic_kernel(state)
    compiled_rates = coupled.stability_rates(state)
    compiled_accepted_metrics = coupled.accepted_state_metrics(compiled_state)
    jax.block_until_ready(compiled_state.velocity.x)
    jax.block_until_ready(compiled_fields.surface_heat_flux)
    jax.block_until_ready(compiled_rates)
    jax.block_until_ready(compiled_accepted_metrics)
    if saved_fpj2 is None:
        momentum.reset_fpj2()
    else:
        momentum.restore_fpj2(saved_fpj2)
    compilation_s = time.perf_counter() - compile_start
    print(f"[compile] GABLS1 kernels ready in {compilation_s:.3f}s", flush=True)

    final_time = args.end_hours * 3600.0
    sample_start_time = args.sample_start_hours * 3600.0
    next_sample_time = (
        math.floor(state.time / args.sample_interval_seconds) + 1
    ) * args.sample_interval_seconds
    start_simulation_time = state.time
    start = time.perf_counter()
    stopped_early = False

    def save_checkpoint() -> None:
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
            "timesteps": np.asarray(timesteps),
            "amd_coefficient": args.amd_coefficient,
            "scalar_amd_coefficient": args.scalar_amd_coefficient,
            "mp5_strength": args.mp5_strength,
            "ko6_strength": args.ko6_strength,
            "momentum_advection": args.momentum_advection,
            "momentum_regularization": args.momentum_regularization,
            "scalar_advection": args.scalar_advection,
            "max_cfl": max_cfl,
            "max_diffusive_cfl": max_diffusive_cfl,
            "max_divergence": max_divergence,
            "max_scalar_budget_residual": max_scalar_budget_residual,
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
        _pack_records(payload, "samples", samples)
        _pack_records(payload, "time_rows", time_rows)
        destination = args.output_dir / "checkpoint.npz"
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, destination)

    def append_diagnostics() -> None:
        statistics = diagnostics.snapshot_statistics(coupled, state)
        nonfinite = [
            key
            for key, value in statistics.items()
            if not np.all(np.isfinite(value)) and key != "obukhov_length"
        ]
        if nonfinite:
            raise FloatingPointError("non-finite diagnostic: " + ", ".join(nonfinite))
        time_rows.append(
            {
                "step": float(state.step),
                "time_s": state.time,
                "time_hours": state.time / 3600.0,
                "boundary_layer_height": float(statistics["boundary_layer_height"]),
                "surface_temperature": case.surface_temperature(state.time),
                "surface_heat_flux": float(statistics["surface_heat_flux"]),
                "friction_velocity": float(statistics["friction_velocity"]),
                "obukhov_length": float(statistics["obukhov_length"]),
                "maximum_abs_w": float(statistics["maximum_abs_w"]),
                "jet_speed": float(statistics["jet_speed"]),
                "jet_height": float(statistics["jet_height"]),
            }
        )
        if state.time >= sample_start_time:
            samples.append(statistics)

    while state.time < final_time:
        advective_rate, momentum_rate, scalar_rate = (
            float(value) for value in coupled.stability_rates(state)
        )
        timestep = min(
            args.target_cfl / advective_rate,
            (
                args.target_diffusive_cfl / momentum_rate
                if momentum_rate > 0.0
                else math.inf
            ),
            (
                args.target_diffusive_cfl / scalar_rate
                if scalar_rate > 0.0
                else math.inf
            ),
            args.dt_max,
            final_time - state.time,
        )
        next_step = state.step + 1
        final_after_step = state.time + timestep >= final_time - 1.0e-12
        max_steps_after_step = (
            args.max_steps is not None and next_step >= args.max_steps
        )
        metrics_due = (
            next_step % args.metrics_every == 0
            or next_step % args.log_every == 0
            or next_step % args.checkpoint_every == 0
            or final_after_step
            or max_steps_after_step
        )
        if metrics_due:
            theta_before = float(
                jnp.mean(
                    state.potential_temperature
                    - coupled.config.reference_potential_temperature
                )
            )
            heat_flux_before = (
                float(jnp.mean(coupled.surface_layer_fluxes(state).heat_flux))
                if args.coupling_integrator == "strang"
                else math.nan
            )
        state = coupled.step(state, timestep=timestep)
        if metrics_due:
            theta_after, heat_flux_after, divergence = (
                float(value) for value in coupled.accepted_state_metrics(state)
            )
            budget_residual = abs(
                theta_after
                - theta_before
                - timestep
                * (
                    0.5 * (heat_flux_before + heat_flux_after)
                    if coupled.last_surface_heat_flux_quadrature is None
                    else float(coupled.last_surface_heat_flux_quadrature)
                )
                / case.domain
            )
        timesteps.append(timestep)
        cfl = timestep * advective_rate
        diffusive_cfl = timestep * max(momentum_rate, scalar_rate)
        max_cfl = max(max_cfl, cfl)
        max_diffusive_cfl = max(max_diffusive_cfl, diffusive_cfl)
        if metrics_due:
            max_divergence = max(max_divergence, divergence)
            max_scalar_budget_residual = max(
                max_scalar_budget_residual,
                budget_residual,
            )
        final = state.time >= final_time
        if state.time + 1.0e-9 >= next_sample_time or final:
            append_diagnostics()
            while next_sample_time <= state.time + 1.0e-9:
                next_sample_time += args.sample_interval_seconds
        if state.step % args.log_every == 0 or final:
            print(
                f"step={state.step} time={state.time / 3600.0:.4f}/"
                f"{args.end_hours:g}h CFL={cfl:.4f} CFLnu={diffusive_cfl:.4f} "
                f"divL2={divergence:.3e} "
                f"Q0={heat_flux_after:.3e} "
                f"theta_budget={budget_residual:.3e} "
                + _eta_log_fields(
                    start_wall=start,
                    start_simulation_time=start_simulation_time,
                    simulation_time=state.time,
                    final_simulation_time=final_time,
                ),
                flush=True,
            )
        if state.step % args.checkpoint_every == 0 or final:
            save_checkpoint()
        if args.max_steps is not None and state.step >= args.max_steps:
            stopped_early = state.time < final_time
            if not time_rows or time_rows[-1]["step"] != float(state.step):
                append_diagnostics()
            save_checkpoint()
            break
        if (
            args.max_run_seconds is not None
            and time.perf_counter() - start >= args.max_run_seconds
        ):
            stopped_early = True
            save_checkpoint()
            break

    if not samples:
        samples.append(diagnostics.snapshot_statistics(coupled, state))
    runtime_s = time.perf_counter() - start
    reference_dir = args.reference_dir if args.reference_dir.exists() else None
    summary = diagnostics.save_outputs(
        args.output_dir,
        samples=samples,
        time_rows=time_rows,
        reference_dir=reference_dir,
        metadata={
            "solver": "non-spectral MAC + matrix-free GMG/PCG",
            "sgs_model": "AMD",
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
            "grid_spacing_m": case.domain / args.nx,
            "end_time_hours": state.time / 3600.0,
            "runtime_s": runtime_s,
            "compilation_s": compilation_s,
            "stopped_early": str(stopped_early).lower(),
            "max_cfl": max_cfl,
            "max_diffusive_cfl": max_diffusive_cfl,
            "max_divergence": max_divergence,
            "max_scalar_budget_residual": max_scalar_budget_residual,
            "accepted_metrics_interval_steps": args.metrics_every,
            "amd_coefficient": args.amd_coefficient,
            "scalar_amd_coefficient": args.scalar_amd_coefficient,
            "mp5_dissipation_strength": args.mp5_strength,
            "ko6_dissipation_strength": args.ko6_strength,
            "momentum_advection": args.momentum_advection,
            "momentum_regularization": args.momentum_regularization,
            "scalar_advection": args.scalar_advection,
            "projection_method": args.projection_method,
            "coupling_integrator": args.coupling_integrator,
            "pressure_smooth": args.pressure_smooth,
        },
    )
    resolved = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
