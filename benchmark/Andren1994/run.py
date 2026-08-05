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


F_CORIOLIS = 1.0e-4

INITIAL_U = (
    4.44, 5.92, 6.91, 7.73, 8.43, 9.02, 9.52, 9.93, 10.25, 10.47,
    10.62, 10.70, 10.71, 10.67, 10.59, 10.48, 10.36, 10.24, 10.13, 10.04,
    9.99, 9.96, 9.95, 9.96, 9.98, 9.99, 10.00, 9.99, 9.99, 9.99,
    10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00,
)
INITIAL_V = (
    2.18, 2.67, 2.83, 2.84, 2.75, 2.57, 2.34, 2.06, 1.75, 1.44,
    1.12, 0.82, 0.55, 0.31, 0.12, -0.02, -0.11, -0.16, -0.17, -0.15,
    -0.11, -0.06, -0.02, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01,
    0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
)
INITIAL_TKE = (
    0.365, 0.295, 0.245, 0.205, 0.175, 0.145, 0.120, 0.100, 0.085,
    0.070, 0.055, 0.045, 0.035, 0.025, 0.020, 0.015, 0.010, 0.010,
    0.005, 0.005, 0.005, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000,
)


def _logarithmic_shear(first, second, z):
    """Differentiate two mean profiles with a logarithm-exact estimator.

    A centred difference of a ``1/z`` shear carries a large geometric bias on the
    first few levels: on a uniform mesh an exact logarithmic profile returns
    ``0.549, 1.207, 1.059, 1.029`` for ``kappa z dU/dz / u*`` instead of one, so
    the near-wall part of any similarity plot is dominated by the estimator
    rather than by the flow.  Dividing the neighbour difference by
    ``z ln(z_up / z_down)`` instead of by ``z_up - z_down`` returns ``A / z``
    exactly for ``U = A ln z`` at every level, including the one-sided ends.
    """

    import numpy as np

    z = np.asarray(z, dtype=float)
    below = np.r_[z[0], z[:-1]]
    above = np.r_[z[1:], z[-1]]
    span = z * np.log(above / below)
    shears = []
    for values in (first, second):
        values = np.asarray(values, dtype=float)
        difference = np.r_[values[1:], values[-1]] - np.r_[values[0], values[:-1]]
        shears.append(difference / span)
    return shears[0], shears[1]


def _initial_tables_on_grid(grid, height: float, roughness: float):
    """Return the published initial profiles sampled at the grid cell centres.

    The tables in this module are tabulated at the centres of forty uniform
    cells.  A stretched mesh puts its centres elsewhere, so the tables are read
    by height rather than by index; a uniform mesh recovers them exactly.

    Below the first published level the two wind components are continued
    logarithmically rather than held constant, because a mesh refined at the
    wall asks for values inside the surface layer where holding the value at
    18.75 m would start the first cells with far too much momentum.  The
    perturbation energy is held constant there, since it only sets an amplitude.
    """

    import numpy as np

    published_z = (np.arange(len(INITIAL_U)) + 0.5) * height / len(INITIAL_U)
    centers = np.asarray(grid.z_centers) - np.asarray(grid.z_faces)[0]
    surface_layer = centers < published_z[0]
    shape_ratio = np.log(np.maximum(centers, 1.001 * roughness) / roughness) / np.log(
        published_z[0] / roughness
    )
    tables = []
    for index, table in enumerate((INITIAL_U, INITIAL_V, INITIAL_TKE)):
        values = np.interp(centers, published_z, np.asarray(table, dtype=float))
        if index < 2:
            values = np.where(surface_layer, table[0] * shape_ratio, values)
        tables.append(values)
    return tuple(tables)


def _cells_from_faces(*face_profiles):
    """Average wall-normal face profiles onto the cells between them."""

    import numpy as np

    return tuple(
        0.5 * (np.asarray(profile)[:-1] + np.asarray(profile)[1:])
        for profile in face_profiles
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
    parser.add_argument("--lasd-sgs-delta-scale", type=float)
    parser.add_argument("--lasd-maximum-coefficient", type=float, default=0.81)
    parser.add_argument(
        "--advection-dissipation-strength",
        "--mp5-strength",
        dest="mp5_strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sgs-time-integration",
        choices=("imex_ark3", "explicit"),
        default="imex_ark3",
    )
    parser.add_argument("--pressure-rtol", type=float, default=1.0e-4)
    parser.add_argument("--pressure-max-iterations", type=int, default=20)
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
    if args.lasd_update_interval <= 0:
        raise SystemExit("LASD update interval must be positive")
    if args.lasd_sgs_delta_scale is not None and (
        not math.isfinite(args.lasd_sgs_delta_scale)
        or args.lasd_sgs_delta_scale <= 0.0
    ):
        raise SystemExit("LASD SGS delta scale must be positive and finite")
    if args.checkpoint_every <= 0:
        raise SystemExit("checkpoint interval must be positive")
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
        LASDModel,
        LASDState,
        NeutralABLConfig,
        NeutralABLMomentum,
    )
    from jaxwind.pressure import (
        BoundaryCondition,
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
    coriolis = F_CORIOLIS
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
    krylov = PCGConfig(
        max_iterations=args.pressure_max_iterations,
        relative_tolerance=args.pressure_rtol,
        execution="jax",
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
            amd=AMDModel(coefficient=args.amd_coefficient),
            sgs_time_integration=args.sgs_time_integration,
            lasd=(
                LASDModel(
                    update_interval=args.lasd_update_interval,
                    maximum_coefficient=args.lasd_maximum_coefficient,
                    x_boundary="periodic",
                    y_boundary="periodic",
                    sgs_delta_scale=args.lasd_sgs_delta_scale,
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
        ),
    )

    nominal_spectrum_level = int(
        np.argmin(
            np.abs(
                (np.arange(nz) + 0.5)
                * (height / nz)
                * coriolis
                / expected_ustar
                - 0.1
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
        return solver.pressure

    def active_wall_velocity():
        return solver.active_wall_velocity(velocity)

    momentum_rate_kernel = (
        solver._compiled_imex_timestep_rates
        if solver.config.sgs_time_integration == "imex_ark3"
        else solver._compiled_timestep_rates
    )

    @jax.jit
    def stability_rates(velocity, scalar, sgs_coefficient):
        """Return all three timestep rates in one device vector."""

        advective_rate, momentum_diffusive_rate = momentum_rate_kernel(
            velocity,
            sgs_coefficient,
        )
        scalar_diffusive_rate = (
            scalar_solver.diffusive_rate(scalar, velocity)
            if args.passive_scalar
            else jnp.asarray(0.0, dtype=advective_rate.dtype)
        )
        return jnp.stack(
            (
                advective_rate,
                momentum_diffusive_rate,
                scalar_diffusive_rate,
            )
        )

    if args.restart is None:
        initial_u, initial_v, initial_tke = _initial_tables_on_grid(
            grid,
            height,
            roughness,
        )
        velocity = solver.initial_profile(
            jnp.asarray(initial_u, dtype=dtype),
            jnp.asarray(initial_v, dtype=dtype),
            perturbation_tke=jnp.asarray(initial_tke, dtype=dtype),
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
        if args.passive_scalar and (
            "checkpoint_schema" not in checkpoint
            or str(checkpoint["checkpoint_schema"])
            != "jaxwind.andren1994.amd-passive-scalar.v2"
        ):
            raise SystemExit(
                "restart predates passive-scalar/complete-SGS statistics; "
                "a true paper comparison must start a fresh run"
            )
        checkpoint_sgs = (
            str(checkpoint["sgs_model"])
            if "sgs_model" in checkpoint
            else "lasd"
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
                raise SystemExit(
                    "restart AMD coefficient does not match this run"
                )
        checkpoint_mp5_strength = (
            float(checkpoint["mp5_strength"])
            if "mp5_strength" in checkpoint
            else 1.0
        )
        if not np.isclose(
            checkpoint_mp5_strength,
            args.mp5_strength,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise SystemExit("restart MP5 strength does not match this run")
        checkpoint_limiter = (
            str(checkpoint["advection_limiter"])
            if "advection_limiter" in checkpoint
            else "mp5"
        )
        if checkpoint_limiter != "mp5":
            raise SystemExit("cannot restart a non-MP5 advection checkpoint")
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
            required_lasd_fields = tuple(
                f"lasd_{name}" for name in LASDState._fields
            )
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
        if "pressure" in checkpoint.files:
            solver.restore_pressure(checkpoint["pressure"])
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
    saved_pressure = solver.pressure
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
        1,
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
    compiled_rates = stability_rates(
        compiled_velocity,
        compiled_scalar,
        active_sgs_coefficient(),
    )
    jax.block_until_ready(compiled_velocity.x)
    jax.block_until_ready(compiled_scalar)
    jax.block_until_ready(compiled_rates)
    if args.restart is None:
        solver.reset_lasd(velocity)
        solver.reset_pressure()
    else:
        if saved_lasd is not None:
            solver.restore_lasd(
                saved_lasd,
                accepted_step=saved_lasd_progress[0],
                interval_time=saved_lasd_progress[1],
            )
        solver.restore_pressure(saved_pressure)
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
            "checkpoint_schema": "jaxwind.andren1994.amd-passive-scalar.v2",
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
            "advection_limiter": "mp5",
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
        payload["pressure"] = np.asarray(solver.pressure)
        payload.update(
            {
                f"sample_{index}": values
                for index, values in enumerate(stacked_samples)
            }
        )
        payload.update(
            {
                f"budget_{index}": values
                for index, values in enumerate(stacked_budgets)
            }
        )
        destination = args.output_dir / "checkpoint.npz"
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}"
        )
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, destination)

    if not history_rows:
        history_rows.append(sample_history())

    pending_rates = stability_rates(velocity, scalar, active_sgs_coefficient())

    while simulation_time < final_time:
        # One vector transfer replaces three scalar transfers.  This is the
        # only host synchronization needed to choose the next timestep.
        advective_rate, momentum_diffusive_rate, scalar_diffusive_rate = (
            float(rate) for rate in np.asarray(pending_rates)
        )
        if advective_rate <= 0.0:
            raise ValueError("cannot choose a CFL step for zero velocity")
        timestep = args.target_cfl / advective_rate
        if momentum_diffusive_rate > 0.0:
            timestep = min(
                timestep,
                args.target_diffusive_cfl / momentum_diffusive_rate,
            )
        if args.passive_scalar and scalar_diffusive_rate > 0.0:
            timestep = min(
                timestep,
                args.target_diffusive_cfl / scalar_diffusive_rate,
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
        next_simulation_time = simulation_time + timestep
        if next_simulation_time < final_time:
            # The next loop-top read waits on this result.  Until then the
            # device computes it while the host records the accepted step.
            pending_rates = stability_rates(
                velocity,
                scalar,
                active_sgs_coefficient(),
            )
        simulation_time = next_simulation_time
        timesteps.append(timestep)
        step += 1
        if (
            simulation_time >= sample_start
            and (step % args.sample_every == 0 or simulation_time >= final_time)
        ):
            samples.append(sample_profiles())
            sample_times.append(simulation_time)
            if args.passive_scalar:
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
    mean = np.column_stack(
        (averaged["u"], averaged["v"], averaged["w"])
    )
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
    z_faces = np.asarray(solver.grid.z_faces)
    z = np.asarray(solver.grid.z_centers) - z_faces[0]
    z_face_height = averaged["face_height_m"]
    # Total flux on the faces the traction is applied to.  Its wall value is the
    # imposed surface stress by construction, which the cell-centred columns
    # cannot reproduce; see amd_diagnostics.PROFILE_NAMES.
    total_uw_face = averaged["resolved_uw_face"] + averaged["sgs_uw_face"]
    total_vw_face = averaged["resolved_vw_face"] + averaged["sgs_vw_face"]
    selected_history = [
        row for row in history_rows if row["time_seconds"] >= sample_start
    ] or [history_rows[-1]]
    ustar = float(np.mean([row["ustar"] for row in selected_history]))
    normalized_height = z * coriolis / ustar
    du_dz, dv_dz = _logarithmic_shear(mean[:, 0], mean[:, 1], z)
    phi_m = 0.4 * z * np.hypot(du_dz, dv_dz) / ustar
    # Monin-Obukhov similarity is a local-scaling statement, so the surface
    # friction velocity is only the right normalization inside the
    # constant-flux layer.  Above it the stress decays and the wind vector
    # turns, and only the locally scaled shear can approach unity.
    local_ustar = np.sqrt(
        np.maximum(np.hypot(*_cells_from_faces(total_uw_face, total_vw_face)), 0.0)
    )
    phi_m_local = np.where(
        local_ustar > 0.0,
        0.4 * z * np.hypot(du_dz, dv_dz) / np.maximum(local_ustar, 1.0e-30),
        np.nan,
    )
    cstar = args.scalar_surface_flux / ustar
    scalar_shear, _ = _logarithmic_shear(
        averaged["scalar"],
        np.zeros_like(averaged["scalar"]),
        z,
    )
    phi_c = -0.4 * z * scalar_shear / cstar
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
        "phi_m_local": phi_m_local,
        "phi_c": phi_c,
        "resolved_u_variance_over_ustar2": variances[:, 0] / ustar2,
        "resolved_v_variance_over_ustar2": variances[:, 1] / ustar2,
        "resolved_w_variance_over_ustar2": variances[:, 2] / ustar2,
        "sgs_component_variance_over_ustar2": component_sgs_variance / ustar2,
        "total_u_variance_over_ustar2": (
            variances[:, 0] + component_sgs_variance
        ) / ustar2,
        "total_v_variance_over_ustar2": (
            variances[:, 1] + component_sgs_variance
        ) / ustar2,
        "total_w_variance_over_ustar2": (
            variances[:, 2] + component_sgs_variance
        ) / ustar2,
        "resolved_tke_over_ustar2": resolved_tke / ustar2,
        "sgs_tke_over_ustar2": sgs_tke / ustar2,
        "total_tke_over_ustar2": (resolved_tke + sgs_tke) / ustar2,
        "resolved_uw_over_ustar2": resolved_uw / ustar2,
        "resolved_vw_over_ustar2": resolved_vw / ustar2,
        "sgs_uw_over_ustar2": sgs_uw / ustar2,
        "sgs_vw_over_ustar2": sgs_vw / ustar2,
        "total_uw_over_ustar2": total_uw / ustar2,
        "total_vw_over_ustar2": total_vw / ustar2,
        "face_total_uw_at_cells_over_ustar2": (
            _cells_from_faces(total_uw_face)[0] / ustar2
        ),
        "face_total_vw_at_cells_over_ustar2": (
            _cells_from_faces(total_vw_face)[0] / ustar2
        ),
        "resolved_scalar_variance_over_cstar2": (
            averaged["resolved_scalar_variance"] / cstar**2
        ),
        "sgs_scalar_variance_over_cstar2": scalar_sgs_variance / cstar**2,
        "total_scalar_variance_over_cstar2": (
            averaged["resolved_scalar_variance"] + scalar_sgs_variance
        ) / cstar**2,
        "resolved_wc_over_ustar_cstar": averaged["resolved_wc"]
        / scalar_flux_scale,
        "sgs_wc_over_ustar_cstar": averaged["sgs_wc"] / scalar_flux_scale,
        "total_wc_over_ustar_cstar": (
            averaged["resolved_wc"] + averaged["sgs_wc"]
        ) / scalar_flux_scale,
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
    if not args.passive_scalar:
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

    # Face profiles carry one more level than the cell profiles and are the only
    # place the surface stress closes exactly, so they are written separately
    # rather than interpolated into the cell table.
    face_columns = {
        "z_face_m": z_face_height,
        "z_face_f_over_ustar": z_face_height * coriolis / ustar,
        "resolved_uw_face_over_ustar2": averaged["resolved_uw_face"] / ustar2,
        "resolved_vw_face_over_ustar2": averaged["resolved_vw_face"] / ustar2,
        "sgs_uw_face_over_ustar2": averaged["sgs_uw_face"] / ustar2,
        "sgs_vw_face_over_ustar2": averaged["sgs_vw_face"] / ustar2,
        "total_uw_face_over_ustar2": total_uw_face / ustar2,
        "total_vw_face_over_ustar2": total_vw_face / ustar2,
        "total_stress_face_over_ustar2": (
            np.hypot(total_uw_face, total_vw_face) / ustar2
        ),
    }
    np.savetxt(
        args.output_dir / "face_stress_profiles.csv",
        np.column_stack(tuple(face_columns.values())),
        delimiter=",",
        header=",".join(face_columns),
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
            heights=z,
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
        "schema": "jaxwind.andren1994.amd-passive-scalar.v2",
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
        "amd_coefficient": (
            args.amd_coefficient if args.sgs == "amd" else None
        ),
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
                "filter_grid_ratio": 1.0,
                "sgs_delta_scale": (
                    1.0
                    if args.lasd_sgs_delta_scale is None
                    else args.lasd_sgs_delta_scale
                ),
                "maximum_coefficient": args.lasd_maximum_coefficient,
                "filter": "shared pressure-GMG levels with conservative restriction",
                "state_level": 1,
                "second_test_level": 2,
                "clipped_beta_fallback": True,
            }
            if args.sgs == "lasd"
            else None
        ),
        "mp5_dissipation_strength": args.mp5_strength,
        "advection_dissipation_strength": args.mp5_strength,
        "advection_limiter": "mp5",
        "sgs_time_integration": args.sgs_time_integration,
        "vertical_sgs_diffusion_is_implicit": (
            args.sgs_time_integration == "imex_ark3"
        ),
        "horizontal_sgs_diffusion_is_explicit": True,
        "target_advective_cfl": args.target_cfl,
        "target_diffusive_cfl": args.target_diffusive_cfl,
        "linear_solver": "pcg",
        "pressure_relative_tolerance": args.pressure_rtol,
        "pressure_max_iterations": args.pressure_max_iterations,
        "krylov_execution": "jax",
        "projection_method": "full",
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
    sgs_label = "AMD" if args.sgs == "amd" else "multilevel LASD"
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
