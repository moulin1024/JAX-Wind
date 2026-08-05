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
CHECKPOINT_SCHEMA = "jaxwind.gabls1.nonspectral-amd.v1"


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
    parser.add_argument(
        "--wall-matching-height",
        type=float,
        help="target physical height; the nearest cell center is used",
    )
    parser.add_argument("--end-hours", type=float, default=9.0)
    parser.add_argument("--sample-start-hours", type=float, default=8.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=60.0)
    parser.add_argument("--dt-max", type=float, default=1.0)
    parser.add_argument("--target-cfl", type=float, default=0.9)
    parser.add_argument("--target-diffusive-cfl", type=float, default=0.5)
    parser.add_argument("--amd-coefficient", type=float, default=0.212)
    parser.add_argument("--scalar-amd-coefficient", type=float)
    parser.add_argument(
        "--advection-dissipation-strength",
        "--mp5-strength",
        dest="mp5_strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--coupling-integrator",
        choices=("strang", "coupled-ssprk3"),
        default="strang",
    )
    parser.add_argument(
        "--sgs-time-integration",
        choices=("explicit", "imex_ark3"),
        default="explicit",
        help="treat the frozen vertical momentum SGS operator implicitly",
    )
    parser.add_argument(
        "--rayleigh-sponge-start-height",
        type=float,
        help="activate a quadratic top sponge above this physical height (m)",
    )
    parser.add_argument(
        "--rayleigh-sponge-maximum-rate",
        type=float,
        default=0.2,
        help="relaxation rate at the domain top (1/s; inactive without start height)",
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
    if not 0.0 <= args.sample_start_hours < args.end_hours:
        parser.error("sample-start-hours must lie in [0, end-hours)")
    for name in ("amd_coefficient", "scalar_amd_coefficient", "mp5_strength"):
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
    if args.wall_matching_height is not None and (
        not math.isfinite(args.wall_matching_height) or args.wall_matching_height <= 0.0
    ):
        parser.error("wall-matching-height must be positive and finite")
    if args.rayleigh_sponge_start_height is not None and (
        not math.isfinite(args.rayleigh_sponge_start_height)
        or args.rayleigh_sponge_start_height < 0.0
        or args.rayleigh_sponge_start_height >= diagnostics.GABLS1Case().domain
    ):
        parser.error("rayleigh-sponge-start-height must lie in [0, domain top)")
    if (
        not math.isfinite(args.rayleigh_sponge_maximum_rate)
        or args.rayleigh_sponge_maximum_rate <= 0.0
    ):
        parser.error("rayleigh-sponge-maximum-rate must be positive and finite")
    if (
        args.sgs_time_integration == "imex_ark3"
        and args.coupling_integrator != "strang"
    ):
        parser.error(
            "--sgs-time-integration imex_ark3 requires --coupling-integrator strang"
        )
    return args


def _initial_state(args, coupled, case, dtype):
    import jax
    import jax.numpy as jnp
    from jaxwind.pressure import MACVelocity

    nz, ny, nx = coupled.grid.shape
    z = jnp.asarray(coupled.grid.z_centers, dtype=dtype)
    z -= coupled.grid.z_faces[0]
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
    from jaxwind.pressure import MACVelocity

    checkpoint = np.load(args.restart, allow_pickle=False)
    if str(checkpoint["checkpoint_schema"]) != CHECKPOINT_SCHEMA:
        raise SystemExit("restart checkpoint schema is not supported")
    if not np.array_equal(checkpoint["shape_zyx"], coupled.grid.shape):
        raise SystemExit("restart grid shape does not match")
    coordinate_keys = ("x_faces", "y_faces", "z_faces")
    if any(key in checkpoint for key in coordinate_keys):
        if not all(key in checkpoint for key in coordinate_keys):
            raise SystemExit("restart checkpoint has incomplete grid coordinates")
        for key in coordinate_keys:
            expected = np.asarray(getattr(coupled.grid, key))
            if not np.array_equal(np.asarray(checkpoint[key]), expected):
                raise SystemExit(f"restart {key} do not match the active mesh")
    for key in ("amd_coefficient", "scalar_amd_coefficient", "mp5_strength"):
        if not np.isclose(float(checkpoint[key]), getattr(args, key)):
            raise SystemExit(f"restart {key} does not match")
    checkpoint_limiter = (
        str(checkpoint["advection_limiter"])
        if "advection_limiter" in checkpoint
        else "mp5"
    )
    if checkpoint_limiter != "mp5":
        raise SystemExit("cannot restart a non-MP5 advection checkpoint")
    checkpoint_sgs_time_integration = (
        str(checkpoint["sgs_time_integration"])
        if "sgs_time_integration" in checkpoint
        else "explicit"
    )
    if checkpoint_sgs_time_integration != args.sgs_time_integration:
        raise SystemExit("restart SGS time integration does not match")
    checkpoint_sponge_start = (
        float(checkpoint["rayleigh_sponge_start_height_m"])
        if "rayleigh_sponge_start_height_m" in checkpoint
        else math.nan
    )
    active_sponge_start = (
        math.nan
        if args.rayleigh_sponge_start_height is None
        else args.rayleigh_sponge_start_height
    )
    if not (
        math.isnan(checkpoint_sponge_start)
        and math.isnan(active_sponge_start)
    ) and not np.isclose(checkpoint_sponge_start, active_sponge_start):
        raise SystemExit("restart Rayleigh sponge start height does not match")
    if "rayleigh_sponge_maximum_rate_s-1" in checkpoint and not np.isclose(
        float(checkpoint["rayleigh_sponge_maximum_rate_s-1"]),
        (
            args.rayleigh_sponge_maximum_rate
            if args.rayleigh_sponge_start_height is not None
            else 0.0
        ),
    ):
        raise SystemExit("restart Rayleigh sponge maximum rate does not match")
    if "wall_matching_height_m" in checkpoint and not np.isclose(
        float(checkpoint["wall_matching_height_m"]),
        coupled.momentum.wall_matching_height,
    ):
        raise SystemExit("restart wall matching height does not match")
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
    from jaxwind.domain import RectilinearGrid
    from jaxwind.pressure import (
        BoundaryCondition,
        GMGConfig,
        MatrixFreePoissonSolver,
        PCGConfig,
        PoissonBoundaryConditions,
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
            wall_matching_height=args.wall_matching_height,
            mp5_dissipation_strength=args.mp5_strength,
            amd=AMDModel(coefficient=args.amd_coefficient),
            sgs_time_integration=args.sgs_time_integration,
        ),
    )
    scalar = AMDPassiveScalar(
        grid,
        AMDPassiveScalarModel(
            coefficient=args.scalar_amd_coefficient,
            lower_surface_flux=0.0,
            upper_surface_flux=0.0,
            mp5_dissipation_strength=args.mp5_strength,
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
            rayleigh_sponge_start_height=args.rayleigh_sponge_start_height,
            rayleigh_sponge_maximum_rate=(
                args.rayleigh_sponge_maximum_rate
                if args.rayleigh_sponge_start_height is not None
                else 0.0
            ),
            rayleigh_reference_temperature_at_zero=(
                case.theta_initial
                - case.inversion_gradient * case.inversion_base
                if args.rayleigh_sponge_start_height is not None
                else None
            ),
            rayleigh_reference_temperature_gradient=case.inversion_gradient,
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
    compiled_state = coupled.step(state, timestep=min(args.dt_max, 0.25))
    compiled_fields = diagnostic_kernel(state)
    compiled_rates = coupled.stability_rates(state)
    compiled_accepted_metrics = coupled.accepted_state_metrics(compiled_state)
    jax.block_until_ready(compiled_state.velocity.x)
    jax.block_until_ready(compiled_fields.surface_heat_flux)
    jax.block_until_ready(compiled_rates)
    jax.block_until_ready(compiled_accepted_metrics)
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
            "x_faces": np.asarray(grid.x_faces),
            "y_faces": np.asarray(grid.y_faces),
            "z_faces": np.asarray(grid.z_faces),
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
            "advection_limiter": "mp5",
            "sgs_time_integration": args.sgs_time_integration,
            "rayleigh_sponge_start_height_m": (
                math.nan
                if args.rayleigh_sponge_start_height is None
                else args.rayleigh_sponge_start_height
            ),
            "rayleigh_sponge_maximum_rate_s-1": (
                args.rayleigh_sponge_maximum_rate
                if args.rayleigh_sponge_start_height is not None
                else 0.0
            ),
            "wall_matching_height_m": momentum.wall_matching_height,
            "max_cfl": max_cfl,
            "max_diffusive_cfl": max_diffusive_cfl,
            "max_divergence": max_divergence,
            "max_scalar_budget_residual": max_scalar_budget_residual,
        }
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
                coupled.scalar.volume_mean(state.potential_temperature)
                - coupled.config.reference_potential_temperature
            )
            heat_flux_before = (
                float(jnp.mean(coupled.surface_layer_fluxes(state).heat_flux))
                if args.coupling_integrator == "strang"
                else math.nan
            )
            sponge_source_before = float(
                coupled.rayleigh_scalar_volume_rate(
                    state.potential_temperature
                )
            )
        state = coupled.step(state, timestep=timestep)
        if metrics_due:
            theta_after, heat_flux_after, divergence = (
                float(value) for value in coupled.accepted_state_metrics(state)
            )
            sponge_source_after = float(
                coupled.rayleigh_scalar_volume_rate(
                    state.potential_temperature
                )
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
                - timestep * 0.5 * (sponge_source_before + sponge_source_after)
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
            "grid_spacing_m": (case.domain**3 / (args.nx * args.ny * args.nz))
            ** (1.0 / 3.0),
            "minimum_dx_m": float(np.min(grid.x_widths)),
            "maximum_dx_m": float(np.max(grid.x_widths)),
            "minimum_dy_m": float(np.min(grid.y_widths)),
            "maximum_dy_m": float(np.max(grid.y_widths)),
            "minimum_dz_m": float(np.min(grid.z_widths)),
            "maximum_dz_m": float(np.max(grid.z_widths)),
            "mesh": "uniform",
            "wall_matching_level": momentum.wall_matching_level,
            "wall_matching_height_m": momentum.wall_matching_height,
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
            "advection_dissipation_strength": args.mp5_strength,
            "advection_limiter": "mp5",
            "projection_method": "full",
            "coupling_integrator": args.coupling_integrator,
            "sgs_time_integration": args.sgs_time_integration,
            "rayleigh_sponge_start_height_m": (
                "disabled"
                if args.rayleigh_sponge_start_height is None
                else args.rayleigh_sponge_start_height
            ),
            "rayleigh_sponge_maximum_rate_s-1": (
                args.rayleigh_sponge_maximum_rate
                if args.rayleigh_sponge_start_height is not None
                else 0.0
            ),
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
