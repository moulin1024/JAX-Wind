#!/usr/bin/env python3
"""Reproduce the neutral Ekman-layer intercomparison of Andren et al. (1994)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
for source in (ROOT, SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


INITIAL_U = (
    4.44,
    5.92,
    6.91,
    7.73,
    8.43,
    9.02,
    9.52,
    9.93,
    10.25,
    10.47,
    10.62,
    10.70,
    10.71,
    10.67,
    10.59,
    10.48,
    10.36,
    10.24,
    10.13,
    10.04,
    9.99,
    9.96,
    9.95,
    9.96,
    9.98,
    9.99,
    10.00,
    9.99,
    9.99,
    9.99,
    10.00,
    10.00,
    10.00,
    10.00,
    10.00,
    10.00,
    10.00,
    10.00,
    10.00,
    10.00,
)
INITIAL_V = (
    2.18,
    2.67,
    2.83,
    2.84,
    2.75,
    2.57,
    2.34,
    2.06,
    1.75,
    1.44,
    1.12,
    0.82,
    0.55,
    0.31,
    0.12,
    -0.02,
    -0.11,
    -0.16,
    -0.17,
    -0.15,
    -0.11,
    -0.06,
    -0.02,
    0.01,
    0.02,
    0.02,
    0.02,
    0.02,
    0.02,
    0.01,
    0.01,
    0.01,
    0.01,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
)
INITIAL_TKE = (
    0.365,
    0.295,
    0.245,
    0.205,
    0.175,
    0.145,
    0.120,
    0.100,
    0.085,
    0.070,
    0.055,
    0.045,
    0.035,
    0.025,
    0.020,
    0.015,
    0.010,
    0.010,
    0.005,
    0.005,
    0.005,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
    0.000,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-ft", type=float, default=0.1)
    parser.add_argument("--sample-start-ft", type=float, default=0.05)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--target-cfl", type=float, default=0.8)
    parser.add_argument("--target-diffusive-cfl", type=float, default=0.5)
    parser.add_argument("--sgs", choices=("amd", "lasd"), default="amd")
    parser.add_argument("--amd-coefficient", type=float, default=0.212)
    parser.add_argument(
        "--passive-scalar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "advance the paper's passive scalar with prescribed surface flux "
            "(default: on for AMD, off for LASD momentum)"
        ),
    )
    parser.add_argument(
        "--scalar-amd-coefficient",
        type=float,
        help="defaults to --amd-coefficient",
    )
    parser.add_argument("--scalar-surface-flux", type=float, default=1.0e-3)
    parser.add_argument("--diagnostic-sgs-ce", type=float, default=0.93)
    parser.add_argument("--diagnostic-scalar-cc", type=float, default=2.02)
    parser.add_argument("--history-every", type=int, default=20)
    parser.add_argument("--lasd-update-interval", type=int, default=1)
    parser.add_argument("--lasd-filter-grid-ratio", type=float, default=1.0)
    parser.add_argument("--lasd-maximum-coefficient", type=float, default=0.81)
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
        "--sgs-time-integration",
        choices=("imex_ark3", "explicit"),
        default="imex_ark3",
    )
    parser.add_argument("--pressure-rtol", type=float, default=1.0e-4)
    parser.add_argument("--pressure-max-iterations", type=int, default=20)
    parser.add_argument("--pressure-restart", type=int, default=10)
    parser.add_argument(
        "--linear-solver",
        choices=("pcg", "gmres"),
        default="pcg",
    )
    parser.add_argument(
        "--projection-method",
        choices=("full", "fpj2"),
        default="fpj2",
    )
    parser.add_argument(
        "--pressure-discretization",
        choices=("centered2", "kep4"),
        default="kep4",
    )
    parser.add_argument(
        "--fpj2-timestep-ratio-limit",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--krylov-execution",
        choices=("jax", "python"),
        default="jax",
    )
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--max-run-seconds",
        type=float,
        help="pause cleanly at a checkpoint after this much stepping wall time",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/andren1994_40cubed"),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.end_ft <= 0.0:
        raise SystemExit("end-ft must be positive")
    if not 0.0 <= args.sample_start_ft < args.end_ft:
        raise SystemExit("sample-start-ft must lie in [0, end-ft)")
    if min(args.sample_every, args.log_every, args.history_every) <= 0:
        raise SystemExit("sampling intervals must be positive")
    if not math.isfinite(args.amd_coefficient) or args.amd_coefficient < 0.0:
        raise SystemExit("AMD coefficient must be finite and nonnegative")
    if not math.isfinite(args.mp5_strength) or args.mp5_strength < 0.0:
        raise SystemExit("MP5 strength must be finite and nonnegative")
    if not math.isfinite(args.ko6_strength) or args.ko6_strength < 0.0:
        raise SystemExit("KO6 strength must be finite and nonnegative")
    if args.passive_scalar is None:
        args.passive_scalar = args.sgs == "amd"
    if args.scalar_amd_coefficient is None:
        args.scalar_amd_coefficient = args.amd_coefficient
    if (
        not math.isfinite(args.scalar_amd_coefficient)
        or args.scalar_amd_coefficient < 0.0
    ):
        raise SystemExit("scalar AMD coefficient must be finite and nonnegative")
    if not math.isfinite(args.scalar_surface_flux):
        raise SystemExit("scalar surface flux must be finite")
    if min(args.diagnostic_sgs_ce, args.diagnostic_scalar_cc) <= 0.0:
        raise SystemExit("diagnostic SGS constants must be positive")
    if args.passive_scalar and args.sgs != "amd":
        raise SystemExit(
            "the non-spectral passive-scalar closure is currently paired with AMD; "
            "use --no-passive-scalar for a momentum-only LASD run"
        )
    if args.passive_scalar and args.projection_method != "fpj2":
        raise SystemExit(
            "paper pressure/scalar diagnostics require --projection-method fpj2"
        )
    if args.lasd_update_interval <= 0:
        raise SystemExit("LASD update interval must be positive")
    if args.checkpoint_every <= 0:
        raise SystemExit("checkpoint interval must be positive")
    if args.fpj2_timestep_ratio_limit < 1.0:
        raise SystemExit("FPJ-2 timestep ratio limit must be at least one")
    if args.max_run_seconds is not None and args.max_run_seconds <= 0.0:
        raise SystemExit("max-run-seconds must be positive")

    from jax import config as jax_config

    if not args.single:
        jax_config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaxwind.momentum import (
        AMDModel,
        AMDPassiveScalar,
        AMDPassiveScalarModel,
        FPJ2State,
        LASDModel,
        LASDState,
        NeutralABLConfig,
        NeutralABLMomentum,
    )
    from jaxwind.pressure import (
        BoundaryCondition,
        FGMRESConfig,
        GMGConfig,
        MACVelocity,
        MatrixFreePoissonSolver,
        PCGConfig,
        PoissonBoundaryConditions,
        RectilinearGrid,
    )
    from benchmark.Andren1994 import amd_diagnostics

    nx = ny = nz = 40
    lx, ly, height = 4000.0, 2000.0, 1500.0
    roughness = 0.1
    geostrophic = (10.0, 0.0)
    coriolis = 1.0e-4
    expected_ustar = 0.425
    dtype = jnp.float32 if args.single else jnp.float64

    grid = RectilinearGrid.uniform(
        nx,
        ny,
        nz,
        lx=lx,
        ly=ly,
        lz=height,
    )
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    krylov = (
        PCGConfig(
            max_iterations=args.pressure_max_iterations,
            relative_tolerance=args.pressure_rtol,
            execution=args.krylov_execution,
        )
        if args.linear_solver == "pcg"
        else FGMRESConfig(
            restart=args.pressure_restart,
            max_iterations=args.pressure_max_iterations,
            relative_tolerance=args.pressure_rtol,
            reorthogonalize=False,
            execution=args.krylov_execution,
        )
    )
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
        dtype=dtype,
        gmg=GMGConfig(
            smoother="auto",
            coarsening="auto",
            coarse_smooth=20,
        ),
        krylov=krylov,
        discretization=args.pressure_discretization,
    )
    solver = NeutralABLMomentum(
        grid,
        pressure,
        NeutralABLConfig(
            friction_velocity=expected_ustar,
            roughness_length=roughness,
            geostrophic_wind=geostrophic,
            coriolis_vertical=coriolis,
            coriolis_horizontal=coriolis,
            mp5_dissipation_strength=args.mp5_strength,
            advection_scheme=args.momentum_advection,
            regularization_scheme=args.momentum_regularization,
            ko6_dissipation_strength=args.ko6_strength,
            amd=AMDModel(coefficient=args.amd_coefficient),
            sgs_time_integration=args.sgs_time_integration,
            projection_method=args.projection_method,
            fpj2_timestep_ratio_limit=args.fpj2_timestep_ratio_limit,
            lasd=(
                LASDModel(
                    filter_grid_ratio=args.lasd_filter_grid_ratio,
                    update_interval=args.lasd_update_interval,
                    maximum_coefficient=args.lasd_maximum_coefficient,
                    x_boundary="periodic",
                    y_boundary="periodic",
                )
                if args.sgs == "lasd"
                else None
            ),
        ),
    )

    scalar_solver = AMDPassiveScalar(
        grid,
        AMDPassiveScalarModel(
            coefficient=args.scalar_amd_coefficient,
            lower_surface_flux=args.scalar_surface_flux,
            mp5_dissipation_strength=args.mp5_strength,
            advection_scheme=args.scalar_advection,
        ),
    )

    nominal_spectrum_level = int(
        np.argmin(
            np.abs(
                (np.arange(nz) + 0.5) * (height / nz) * coriolis / expected_ustar - 0.1
            )
        )
    )
    compiled_sample_profiles = amd_diagnostics.build_profile_kernel(
        solver,
        scalar_solver,
        diagnostic_ce=args.diagnostic_sgs_ce,
        diagnostic_cc=args.diagnostic_scalar_cc,
        spectrum_level=nominal_spectrum_level,
    )
    compiled_history = amd_diagnostics.build_history_kernel(
        solver,
        diagnostic_ce=args.diagnostic_sgs_ce,
    )
    compiled_budget = amd_diagnostics.build_budget_kernel(
        solver,
        scalar_solver,
    )

    def active_sgs_coefficient():
        lasd = solver.lasd_state
        return (
            lasd.coefficient
            if lasd is not None
            else jnp.zeros((1,), dtype=velocity.x.dtype)
        )

    def active_pressure():
        fpj2 = solver.fpj2_state
        if fpj2 is None:
            return jnp.zeros(grid.shape, dtype=dtype)
        return fpj2.current_pressure

    def active_wall_velocity():
        return solver.active_wall_velocity(velocity)

    if args.restart is None:
        velocity = solver.initial_profile(
            jnp.asarray(INITIAL_U, dtype=dtype),
            jnp.asarray(INITIAL_V, dtype=dtype),
            perturbation_tke=jnp.asarray(INITIAL_TKE, dtype=dtype),
            seed=args.seed,
        )
        solver.reset_lasd(velocity)
        scalar = jnp.zeros(grid.shape, dtype=dtype)
        samples: list[tuple[np.ndarray, ...]] = []
        sample_times: list[float] = []
        budget_samples: list[tuple[np.ndarray, ...]] = []
        budget_times: list[float] = []
        history_rows: list[dict[str, float]] = []
        timesteps: list[float] = []
        simulation_time = 0.0
        step = 0
    else:
        checkpoint = np.load(args.restart)
        if (
            "checkpoint_schema" not in checkpoint
            or str(checkpoint["checkpoint_schema"])
            != "jaxwind.andren1994.morinishi-s4-pressure-ko6-mp5.v5"
        ):
            raise SystemExit(
                "restart predates the Morinishi staggered S4 transport; "
                "start a fresh run with the current discretization"
            )
        checkpoint_sgs = (
            str(checkpoint["sgs_model"]) if "sgs_model" in checkpoint else "lasd"
        )
        if checkpoint_sgs != args.sgs:
            raise SystemExit("restart SGS model does not match this run")
        if args.sgs == "amd" and "amd_coefficient" in checkpoint:
            checkpoint_amd_coefficient = float(checkpoint["amd_coefficient"])
            if not np.isclose(
                checkpoint_amd_coefficient,
                args.amd_coefficient,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise SystemExit("restart AMD coefficient does not match this run")
        checkpoint_mp5_strength = (
            float(checkpoint["mp5_strength"]) if "mp5_strength" in checkpoint else 1.0
        )
        if not np.isclose(
            checkpoint_mp5_strength,
            args.mp5_strength,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise SystemExit("restart MP5 strength does not match this run")
        for name, expected in (("ko6_strength", args.ko6_strength),):
            if name not in checkpoint or not np.isclose(
                float(checkpoint[name]),
                expected,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise SystemExit(f"restart {name} does not match this run")
        for name, expected in (
            ("momentum_advection", args.momentum_advection),
            ("momentum_regularization", args.momentum_regularization),
            ("scalar_advection", args.scalar_advection),
            ("pressure_discretization", args.pressure_discretization),
        ):
            if name not in checkpoint or str(checkpoint[name]) != expected:
                raise SystemExit(f"restart {name} does not match this run")
        for name, expected in (
            ("scalar_amd_coefficient", args.scalar_amd_coefficient),
            ("scalar_surface_flux", args.scalar_surface_flux),
        ):
            if name in checkpoint and not np.isclose(
                float(checkpoint[name]),
                expected,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise SystemExit(f"restart {name} does not match this run")
        if "shape_zyx" in checkpoint and not np.array_equal(
            checkpoint["shape_zyx"],
            np.asarray(grid.shape),
        ):
            raise SystemExit("restart grid shape does not match this run")
        velocity = MACVelocity(
            jnp.asarray(checkpoint["velocity_x"], dtype=dtype),
            jnp.asarray(checkpoint["velocity_y"], dtype=dtype),
            jnp.asarray(checkpoint["velocity_z"], dtype=dtype),
        )
        scalar = (
            jnp.asarray(checkpoint["passive_scalar"], dtype=dtype)
            if "passive_scalar" in checkpoint
            else jnp.zeros(grid.shape, dtype=dtype)
        )
        step = int(checkpoint["step"])
        simulation_time = float(checkpoint["simulation_time"])
        if args.sgs == "lasd":
            required_lasd_fields = tuple(f"lasd_{name}" for name in LASDState._fields)
            if not all(name in checkpoint for name in required_lasd_fields):
                raise SystemExit("LASD restart is missing closure memory")
            solver.restore_lasd(
                LASDState(
                    *(
                        jnp.asarray(checkpoint[name], dtype=dtype)
                        for name in required_lasd_fields
                    )
                ),
                accepted_step=int(checkpoint["lasd_step"]),
                interval_time=float(checkpoint["lasd_interval_time"]),
            )
        if (
            args.projection_method == "fpj2"
            and "fpj2_current_pressure" in checkpoint.files
        ):
            solver.restore_fpj2(
                FPJ2State(
                    jnp.asarray(
                        checkpoint["fpj2_current_pressure"],
                        dtype=dtype,
                    ),
                    jnp.asarray(
                        checkpoint["fpj2_previous_pressure"],
                        dtype=dtype,
                    ),
                    float(checkpoint["fpj2_current_timestep"]),
                    float(checkpoint["fpj2_previous_timestep"]),
                    int(checkpoint["fpj2_history_count"]),
                )
            )
        timesteps = list(np.asarray(checkpoint["timesteps"], dtype=float))
        sample_arrays = tuple(
            np.asarray(checkpoint[f"sample_{index}"])
            for index in range(len(amd_diagnostics.PROFILE_NAMES))
        )
        samples = [
            tuple(array[index] for array in sample_arrays)
            for index in range(sample_arrays[0].shape[0])
        ]
        sample_times = list(np.asarray(checkpoint["sample_times"], dtype=float))
        budget_arrays = tuple(
            np.asarray(checkpoint[f"budget_{index}"])
            for index in range(len(amd_diagnostics.BUDGET_NAMES))
        )
        budget_samples = [
            tuple(array[index] for array in budget_arrays)
            for index in range(budget_arrays[0].shape[0])
        ]
        budget_times = list(np.asarray(checkpoint["budget_times"], dtype=float))
        history_names = (
            "time_seconds",
            "step",
            "ustar",
            "integrated_resolved_tke_m3_s2",
            "integrated_sgs_tke_m3_s2",
            "integrated_total_tke_m3_s2",
            "cu",
            "cv",
        )
        history_size = len(checkpoint["history_time_seconds"])
        history_rows = [
            {
                name: float(checkpoint[f"history_{name}"][index])
                for name in history_names
            }
            for index in range(history_size)
        ]

    saved_lasd = solver.lasd_state
    saved_lasd_progress = solver.lasd_progress
    saved_fpj2 = solver.fpj2_state
    compile_start = time.perf_counter()
    warmup_timestep = solver.timestep_for_cfl(
        velocity,
        args.target_cfl,
        args.target_diffusive_cfl,
    )
    compiled_velocity = velocity
    compiled_scalar = scalar
    warmup_steps = max(
        args.lasd_update_interval if args.sgs == "lasd" else 1,
        3 if args.projection_method == "fpj2" else 1,
    )
    for warmup_step in range(warmup_steps):
        if args.passive_scalar:
            compiled_scalar = scalar_solver.step(
                compiled_scalar,
                compiled_velocity,
                timestep=0.5 * warmup_timestep,
            )
        compiled_velocity = solver.step(
            compiled_velocity,
            timestep=warmup_timestep,
            time=warmup_step * warmup_timestep,
        )
        if args.passive_scalar:
            compiled_scalar = scalar_solver.step(
                compiled_scalar,
                compiled_velocity,
                timestep=0.5 * warmup_timestep,
            )
    jax.block_until_ready(compiled_velocity.x)
    jax.block_until_ready(compiled_scalar)
    if args.restart is None:
        solver.reset_lasd(velocity)
        solver.reset_fpj2()
    else:
        if saved_lasd is not None:
            solver.restore_lasd(
                saved_lasd,
                accepted_step=saved_lasd_progress[0],
                interval_time=saved_lasd_progress[1],
            )
        if saved_fpj2 is None:
            solver.reset_fpj2()
        else:
            solver.restore_fpj2(saved_fpj2)
    compiled_samples = compiled_sample_profiles(
        velocity,
        scalar,
        active_pressure(),
        active_sgs_coefficient(),
        active_wall_velocity(),
    )
    jax.block_until_ready(compiled_samples[0])
    compiled_history_sample = compiled_history(
        velocity,
        active_sgs_coefficient(),
        active_wall_velocity(),
    )
    jax.block_until_ready(compiled_history_sample[0])
    compiled_budget_sample = compiled_budget(
        velocity,
        scalar,
        active_pressure(),
        active_sgs_coefficient(),
        active_wall_velocity(),
    )
    jax.block_until_ready(compiled_budget_sample[0])
    solver.diagnostic(
        velocity,
        timestep=warmup_timestep,
        time=0.0,
    )
    compilation_elapsed = time.perf_counter() - compile_start
    print(f"[compile] kernels ready in {compilation_elapsed:.3f}s")

    diagnostics = []
    final_time = args.end_ft / coriolis
    sample_start = args.sample_start_ft / coriolis
    start = time.perf_counter()

    def sample_profiles() -> tuple[np.ndarray, ...]:
        return tuple(
            np.asarray(value)
            for value in compiled_sample_profiles(
                velocity,
                scalar,
                active_pressure(),
                active_sgs_coefficient(),
                active_wall_velocity(),
            )
        )

    def sample_budget() -> tuple[np.ndarray, ...]:
        return tuple(
            np.asarray(value)
            for value in compiled_budget(
                velocity,
                scalar,
                active_pressure(),
                active_sgs_coefficient(),
                active_wall_velocity(),
            )
        )

    def sample_history() -> dict[str, float]:
        values = tuple(
            float(value)
            for value in compiled_history(
                velocity,
                active_sgs_coefficient(),
                active_wall_velocity(),
            )
        )
        integrated_total = values[1] + values[2]
        return {
            "time_seconds": simulation_time,
            "step": float(step),
            "ustar": values[0],
            "integrated_resolved_tke_m3_s2": values[1],
            "integrated_sgs_tke_m3_s2": values[2],
            "integrated_total_tke_m3_s2": integrated_total,
            "cu": values[3],
            "cv": values[4],
        }

    def save_checkpoint() -> None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        lasd = solver.lasd_state
        if samples:
            stacked_samples = tuple(
                np.stack([sample[index] for sample in samples])
                for index in range(len(amd_diagnostics.PROFILE_NAMES))
            )
        else:
            stacked_samples = tuple(
                np.empty(
                    (0, *np.asarray(compiled_samples[index]).shape),
                    dtype=np.asarray(compiled_samples[index]).dtype,
                )
                for index in range(len(amd_diagnostics.PROFILE_NAMES))
            )
        if budget_samples:
            stacked_budgets = tuple(
                np.stack([sample[index] for sample in budget_samples])
                for index in range(len(amd_diagnostics.BUDGET_NAMES))
            )
        else:
            stacked_budgets = tuple(
                np.empty((0, nz), dtype=np.asarray(scalar).dtype)
                for _ in amd_diagnostics.BUDGET_NAMES
            )
        payload = {
            "checkpoint_schema": "jaxwind.andren1994.morinishi-s4-pressure-ko6-mp5.v5",
            "velocity_x": np.asarray(velocity.x),
            "velocity_y": np.asarray(velocity.y),
            "velocity_z": np.asarray(velocity.z),
            "passive_scalar": np.asarray(scalar),
            "shape_zyx": np.asarray(grid.shape),
            "step": step,
            "simulation_time": simulation_time,
            "timesteps": np.asarray(timesteps),
            "sgs_model": args.sgs,
            "amd_coefficient": args.amd_coefficient,
            "scalar_amd_coefficient": args.scalar_amd_coefficient,
            "scalar_surface_flux": args.scalar_surface_flux,
            "mp5_strength": args.mp5_strength,
            "ko6_strength": args.ko6_strength,
            "momentum_advection": args.momentum_advection,
            "momentum_regularization": args.momentum_regularization,
            "scalar_advection": args.scalar_advection,
            "pressure_discretization": args.pressure_discretization,
            "sample_times": np.asarray(sample_times),
            "budget_times": np.asarray(budget_times),
        }
        if history_rows:
            for name in history_rows[0]:
                payload[f"history_{name}"] = np.asarray(
                    [row[name] for row in history_rows]
                )
        else:
            for name in (
                "time_seconds",
                "step",
                "ustar",
                "integrated_resolved_tke_m3_s2",
                "integrated_sgs_tke_m3_s2",
                "integrated_total_tke_m3_s2",
                "cu",
                "cv",
            ):
                payload[f"history_{name}"] = np.empty((0,))
        if lasd is not None:
            lasd_step, interval_time = solver.lasd_progress
            payload.update(
                {
                    **{
                        f"lasd_{name}": np.asarray(value)
                        for name, value in zip(
                            LASDState._fields,
                            lasd,
                            strict=True,
                        )
                    },
                    "lasd_step": lasd_step,
                    "lasd_interval_time": interval_time,
                }
            )
        fpj2 = solver.fpj2_state
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
        payload.update(
            {f"sample_{index}": values for index, values in enumerate(stacked_samples)}
        )
        payload.update(
            {f"budget_{index}": values for index, values in enumerate(stacked_budgets)}
        )
        destination = args.output_dir / "checkpoint.npz"
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, destination)

    if not history_rows:
        history_rows.append(sample_history())

    while simulation_time < final_time:
        timestep = solver.timestep_for_cfl(
            velocity,
            args.target_cfl,
            args.target_diffusive_cfl,
        )
        if args.passive_scalar:
            timestep = min(
                timestep,
                scalar_solver.timestep_for_diffusive_cfl(
                    scalar,
                    velocity,
                    args.target_diffusive_cfl,
                ),
            )
        timestep = min(timestep, final_time - simulation_time)
        step_sgs_coefficient = active_sgs_coefficient()
        if args.passive_scalar:
            scalar = scalar_solver.step(
                scalar,
                velocity,
                timestep=0.5 * timestep,
            )
        velocity = solver.step(
            velocity,
            timestep=timestep,
            time=simulation_time,
        )
        if args.passive_scalar:
            scalar = scalar_solver.step(
                scalar,
                velocity,
                timestep=0.5 * timestep,
            )
        simulation_time += timestep
        timesteps.append(timestep)
        step += 1
        if simulation_time >= sample_start and (
            step % args.sample_every == 0 or simulation_time >= final_time
        ):
            samples.append(sample_profiles())
            sample_times.append(simulation_time)
            if args.projection_method == "fpj2":
                budget_samples.append(sample_budget())
                budget_times.append(simulation_time)
        if step % args.history_every == 0 or simulation_time >= final_time:
            history_rows.append(sample_history())
        if step % args.log_every == 0 or simulation_time >= final_time:
            diagnostic = solver.diagnostic(
                velocity,
                timestep=timestep,
                time=simulation_time,
            )
            step_diagnostic = solver.diagnostic(
                velocity,
                timestep=timestep,
                time=simulation_time,
                lasd_coefficient=step_sgs_coefficient,
            )
            diagnostics.append(diagnostic)
            print(
                f"step={step} ft={coriolis * simulation_time:.4f}/"
                f"{args.end_ft:g} CFL={diagnostic.maximum_cfl:.4f} "
                f"CFLnu_step={step_diagnostic.maximum_diffusive_cfl:.4f} "
                f"CFLnu_next={diagnostic.maximum_diffusive_cfl:.4f} "
                f"ustar/Ug={diagnostic.mean_wall_ustar / geostrophic[0]:.5f} "
                f"divL2={diagnostic.divergence_norm:.3e} "
                f"nu_sgs_max={diagnostic.maximum_sgs_viscosity:.3e} "
                f"Csgs_mean/max={diagnostic.mean_sgs_coefficient:.3e}/"
                f"{diagnostic.maximum_sgs_coefficient:.3e} "
                f"clipped={diagnostic.clipped_sgs_coefficient_fraction:.3f}"
            )
        if step % args.checkpoint_every == 0:
            save_checkpoint()
            if (
                args.max_run_seconds is not None
                and time.perf_counter() - start >= args.max_run_seconds
            ):
                print(
                    f"[paused] checkpointed step={step} "
                    f"ft={coriolis * simulation_time:.4f}",
                    flush=True,
                )
                return
    jax.block_until_ready(velocity.x)
    save_checkpoint()
    elapsed = time.perf_counter() - start
    if not diagnostics:
        diagnostics.append(
            solver.diagnostic(
                velocity,
                timestep=timesteps[-1],
                time=simulation_time,
            )
        )

    averaged = amd_diagnostics.average_samples(samples)
    mean = np.column_stack((averaged["u"], averaged["v"], averaged["w"]))
    variances = np.column_stack(
        (
            averaged["resolved_u_variance"],
            averaged["resolved_v_variance"],
            averaged["resolved_w_variance"],
        )
    )
    resolved_uw = averaged["resolved_uw"]
    resolved_vw = averaged["resolved_vw"]
    sgs_uw = averaged["sgs_uw"]
    sgs_vw = averaged["sgs_vw"]
    total_uw = resolved_uw + sgs_uw
    total_vw = resolved_vw + sgs_vw
    z = (np.arange(nz) + 0.5) * height / nz
    selected_history = [
        row for row in history_rows if row["time_seconds"] >= sample_start
    ] or [history_rows[-1]]
    ustar = float(np.mean([row["ustar"] for row in selected_history]))
    normalized_height = z * coriolis / ustar
    phi_m = (
        0.4
        * z
        / ustar
        * np.hypot(np.gradient(mean[:, 0], z), np.gradient(mean[:, 1], z))
    )
    phi_m[0] = 1.0
    cstar = args.scalar_surface_flux / ustar
    phi_c = -0.4 * z * np.gradient(averaged["scalar"], z) / cstar
    phi_c[0] = 1.0
    resolved_tke = averaged["resolved_tke"]
    sgs_tke = averaged["sgs_tke"]
    component_sgs_variance = (2.0 / 3.0) * sgs_tke
    scalar_sgs_variance = amd_diagnostics.diagnostic_scalar_variance(averaged)
    integrated_total_tke = float(
        np.mean(
            [
                coriolis * row["integrated_total_tke_m3_s2"] / ustar**3
                for row in selected_history
            ]
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    amd_diagnostics.write_csv(args.output_dir / "history.csv", history_rows)
    ustar2 = ustar**2
    scalar_flux_scale = ustar * cstar
    normalized_columns = {
        "z_m": z,
        "z_f_over_ustar": normalized_height,
        "u_m_s": mean[:, 0],
        "v_m_s": mean[:, 1],
        "w_m_s": mean[:, 2],
        "scalar": averaged["scalar"],
        "phi_m": phi_m,
        "phi_c": phi_c,
        "resolved_u_variance_over_ustar2": variances[:, 0] / ustar2,
        "resolved_v_variance_over_ustar2": variances[:, 1] / ustar2,
        "resolved_w_variance_over_ustar2": variances[:, 2] / ustar2,
        "sgs_component_variance_over_ustar2": component_sgs_variance / ustar2,
        "total_u_variance_over_ustar2": (variances[:, 0] + component_sgs_variance)
        / ustar2,
        "total_v_variance_over_ustar2": (variances[:, 1] + component_sgs_variance)
        / ustar2,
        "total_w_variance_over_ustar2": (variances[:, 2] + component_sgs_variance)
        / ustar2,
        "resolved_tke_over_ustar2": resolved_tke / ustar2,
        "sgs_tke_over_ustar2": sgs_tke / ustar2,
        "total_tke_over_ustar2": (resolved_tke + sgs_tke) / ustar2,
        "resolved_uw_over_ustar2": resolved_uw / ustar2,
        "resolved_vw_over_ustar2": resolved_vw / ustar2,
        "sgs_uw_over_ustar2": sgs_uw / ustar2,
        "sgs_vw_over_ustar2": sgs_vw / ustar2,
        "total_uw_over_ustar2": total_uw / ustar2,
        "total_vw_over_ustar2": total_vw / ustar2,
        "resolved_scalar_variance_over_cstar2": (
            averaged["resolved_scalar_variance"] / cstar**2
        ),
        "sgs_scalar_variance_over_cstar2": scalar_sgs_variance / cstar**2,
        "total_scalar_variance_over_cstar2": (
            averaged["resolved_scalar_variance"] + scalar_sgs_variance
        )
        / cstar**2,
        "resolved_wc_over_ustar_cstar": averaged["resolved_wc"] / scalar_flux_scale,
        "sgs_wc_over_ustar_cstar": averaged["sgs_wc"] / scalar_flux_scale,
        "total_wc_over_ustar_cstar": (averaged["resolved_wc"] + averaged["sgs_wc"])
        / scalar_flux_scale,
        "momentum_diffusivity_m2_s": averaged["momentum_diffusivity"],
        "scalar_diffusivity_m2_s": averaged["scalar_diffusivity"],
        "wp_modified_pressure_over_ustar3": (
            averaged["wp_modified_pressure"] / ustar**3
        ),
        "modified_pressure_std_over_ustar2": (
            averaged["modified_pressure_std"] / ustar2
        ),
        "resolved_tke_sgs_dissipation_over_f_ustar2": (
            averaged["resolved_tke_sgs_dissipation"] / (coriolis * ustar2)
        ),
    }
    if not args.passive_scalar:
        for name in (
            "scalar",
            "phi_c",
            "resolved_scalar_variance_over_cstar2",
            "sgs_scalar_variance_over_cstar2",
            "total_scalar_variance_over_cstar2",
            "resolved_wc_over_ustar_cstar",
            "sgs_wc_over_ustar_cstar",
            "total_wc_over_ustar_cstar",
            "scalar_diffusivity_m2_s",
        ):
            normalized_columns.pop(name)
    if args.projection_method != "fpj2":
        normalized_columns.pop("wp_modified_pressure_over_ustar3")
        normalized_columns.pop("modified_pressure_std_over_ustar2")
    normalized_matrix = np.column_stack(tuple(normalized_columns.values()))
    for filename in ("profiles.csv", "normalized_profiles.csv"):
        np.savetxt(
            args.output_dir / filename,
            normalized_matrix,
            delimiter=",",
            header=",".join(normalized_columns),
            comments="",
        )

    profile_path = args.output_dir / "andren1994_profiles.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "z_m",
                "zf_over_ustar",
                "mean_u_m_s",
                "mean_v_m_s",
                "var_u_m2_s2",
                "var_v_m2_s2",
                "var_w_m2_s2",
                "diagnostic_sgs_tke_m2_s2",
                "diagnostic_sgs_component_variance_m2_s2",
                "resolved_uw_m2_s2",
                "resolved_vw_m2_s2",
                "sgs_uw_m2_s2",
                "sgs_vw_m2_s2",
                "total_uw_m2_s2",
                "total_vw_m2_s2",
                "phi_m",
            )
        )
        writer.writerows(
            zip(
                z,
                normalized_height,
                mean[:, 0],
                mean[:, 1],
                variances[:, 0],
                variances[:, 1],
                variances[:, 2],
                sgs_tke,
                component_sgs_variance,
                resolved_uw,
                resolved_vw,
                sgs_uw,
                sgs_vw,
                total_uw,
                total_vw,
                phi_m,
            )
        )

    modes = averaged["spectrum_mode"]
    selected_modes = modes > 0.0
    wavenumber = 2.0 * np.pi * modes[selected_modes] / lx
    # ``mode * E_mode`` is the discrete counterpart of the scaled ordinate
    # printed in Andrén Fig. 15; applying the caption's factor again would
    # double-count the dimensional-to-discrete spectrum conversion.
    spectra_columns = {
        "k_ustar_over_f": wavenumber * ustar / coriolis,
        "kEu_over_ustar2": modes[selected_modes]
        * averaged["spectrum_u"][selected_modes]
        / ustar2,
        "kEv_over_ustar2": modes[selected_modes]
        * averaged["spectrum_v"][selected_modes]
        / ustar2,
        "kEw_over_ustar2": modes[selected_modes]
        * averaged["spectrum_w"][selected_modes]
        / ustar2,
        "sample_height_m": averaged["spectrum_height_m"][selected_modes],
    }
    if args.passive_scalar:
        spectra_columns["kEc_over_cstar2"] = (
            modes[selected_modes]
            * averaged["spectrum_scalar"][selected_modes]
            / cstar**2
        )
    np.savetxt(
        args.output_dir / "spectra.csv",
        np.column_stack(tuple(spectra_columns.values())),
        delimiter=",",
        header=",".join(spectra_columns),
        comments="",
    )
    if len(budget_samples) >= 2:
        uw_budget, wc_budget = amd_diagnostics.averaged_budget(
            budget_times,
            budget_samples,
            ustar=ustar,
            scalar_surface_flux=args.scalar_surface_flux,
            coriolis=coriolis,
            dz=height / nz,
        )
        amd_diagnostics.write_budget(
            args.output_dir / "fig12_budget_profiles.csv",
            uw_budget,
        )
        if args.passive_scalar:
            amd_diagnostics.write_budget(
                args.output_dir / "fig13_budget_profiles.csv",
                wc_budget,
            )

    summary = {
        "schema": "jaxwind.andren1994.morinishi-s4-pressure-amd-scalar.v4",
        "reference": "Andren et al. (1994), QJRMS 120, 1457-1484",
        "backend": jax.default_backend(),
        "dtype": str(dtype),
        "shape_zyx": grid.shape,
        "domain_m": [lx, ly, height],
        "roughness_length_m": roughness,
        "geostrophic_wind_m_s": geostrophic,
        "coriolis_vertical_s-1": coriolis,
        "coriolis_horizontal_s-1": coriolis,
        "end_ft": coriolis * simulation_time,
        "sample_start_ft": args.sample_start_ft,
        "steps": step,
        "minimum_dt_seconds": min(timesteps),
        "maximum_dt_seconds": max(timesteps),
        "elapsed_seconds": elapsed,
        "compilation_seconds": compilation_elapsed,
        "sgs_model": args.sgs,
        "amd_coefficient": (args.amd_coefficient if args.sgs == "amd" else None),
        "passive_scalar": args.passive_scalar,
        "scalar_amd_coefficient": args.scalar_amd_coefficient,
        "scalar_surface_flux": args.scalar_surface_flux,
        "diagnostic_sgs_energy": True,
        "diagnostic_sgs_scalar_variance": True,
        "sgs_energy_kind": (
            "diagnostic local equilibrium with neutral-log wall shear; not prognostic"
        ),
        "sgs_scalar_variance_kind": (
            "diagnostic local equilibrium using full SGS scalar dissipation and "
            "flux-consistent lower-wall gradient; not prognostic"
        ),
        "diagnostic_sgs_dissipation_coefficient": args.diagnostic_sgs_ce,
        "diagnostic_scalar_variance_coefficient": args.diagnostic_scalar_cc,
        "lasd": (
            {
                "update_interval": args.lasd_update_interval,
                "filter_grid_ratio": args.lasd_filter_grid_ratio,
                "sgs_delta_scale": args.lasd_filter_grid_ratio,
                "maximum_coefficient": args.lasd_maximum_coefficient,
                "filter": "three-dimensional compact top-hat convolution",
                "clipped_beta_fallback": True,
            }
            if args.sgs == "lasd"
            else None
        ),
        "mp5_dissipation_strength": args.mp5_strength,
        "ko6_dissipation_strength": args.ko6_strength,
        "momentum_advection": args.momentum_advection,
        "momentum_regularization": args.momentum_regularization,
        "scalar_advection": args.scalar_advection,
        "pressure_discretization": args.pressure_discretization,
        "sgs_time_integration": args.sgs_time_integration,
        "vertical_sgs_diffusion_is_implicit": (
            args.sgs_time_integration == "imex_ark3"
        ),
        "horizontal_sgs_diffusion_is_explicit": True,
        "target_advective_cfl": args.target_cfl,
        "target_diffusive_cfl": args.target_diffusive_cfl,
        "linear_solver": args.linear_solver,
        "pressure_relative_tolerance": args.pressure_rtol,
        "pressure_max_iterations": args.pressure_max_iterations,
        "pressure_restart": args.pressure_restart,
        "krylov_execution": args.krylov_execution,
        "projection_method": args.projection_method,
        "fpj2_timestep_ratio_limit": args.fpj2_timestep_ratio_limit,
        "elapsed_seconds_scope": (
            "fresh canonical invocation"
            if args.restart is None
            else "final invocation after restart"
        ),
        "friction_velocity_m_s": ustar,
        "friction_velocity_over_geostrophic": ustar / geostrophic[0],
        "reference_friction_velocity_ratio_range": [0.0402, 0.0448],
        "normalized_integrated_resolved_tke": float(
            np.mean(
                [
                    coriolis * row["integrated_resolved_tke_m3_s2"] / ustar**3
                    for row in selected_history
                ]
            )
        ),
        "normalized_integrated_sgs_tke": float(
            np.mean(
                [
                    coriolis * row["integrated_sgs_tke_m3_s2"] / ustar**3
                    for row in selected_history
                ]
            )
        ),
        "normalized_integrated_total_tke": integrated_total_tke,
        "reference_normalized_integrated_total_tke": 0.7,
        "final": asdict(diagnostics[-1]),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(args.output_dir / ".matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), sharey=True)
    axes[0, 0].plot(mean[:, 0] / geostrophic[0], normalized_height, label="U/Ug")
    axes[0, 0].plot(mean[:, 1] / geostrophic[0], normalized_height, label="V/Ug")
    axes[0, 0].set_xlabel("mean velocity / Ug")
    axes[0, 0].legend()
    for component, label in enumerate(("u", "v", "w")):
        axes[0, 1].plot(
            (variances[:, component] + component_sgs_variance) / ustar**2,
            normalized_height,
            label=label,
        )
    axes[0, 1].set_xlabel("total variance / u*²")
    axes[0, 1].legend()
    axes[1, 0].plot(total_uw / ustar**2, normalized_height, label="total uw")
    axes[1, 0].plot(total_vw / ustar**2, normalized_height, label="total vw")
    axes[1, 0].set_xlabel("momentum flux / u*²")
    axes[1, 0].legend()
    axes[1, 1].plot(phi_m, normalized_height)
    axes[1, 1].axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].set_xlabel("Phi_M")
    for panel in axes.flat:
        panel.set_ylabel("z f / u*")
        panel.grid(True, alpha=0.25)
        panel.set_ylim(0.0, 0.36)
    sgs_label = "AMD" if args.sgs == "amd" else "physical-space LASD"
    figure.suptitle(
        f"Andren 1994 with {sgs_label}: "
        f"ft={coriolis * simulation_time:.3f}, "
        f"u*/Ug={ustar / geostrophic[0]:.4f}"
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "andren1994_profiles.png", dpi=180)
    plt.close(figure)
    print(f"[done] elapsed={elapsed:.3f}s summary={summary_path}")


if __name__ == "__main__":
    main()
