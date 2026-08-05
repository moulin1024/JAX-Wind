#!/usr/bin/env python3
"""Pressure-driven neutral log-law LES with the non-spectral MAC solver."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--lx", type=float, default=2.0 * math.pi)
    parser.add_argument("--ly", type=float, default=math.pi)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--ustar", type=float, default=0.1)
    parser.add_argument("--z0", type=float, default=1.0e-3)
    parser.add_argument(
        "--wall-matching-level",
        type=int,
        default=0,
        help="zero-based vertical cell level supplied to the wall model",
    )
    parser.add_argument(
        "--wall-filter-width",
        type=float,
        help="periodic physical top-hat width in horizontal grid cells",
    )
    parser.add_argument(
        "--wall-temporal-filter-gamma",
        type=float,
        help=(
            "enable first-order wall-input filtering with "
            "Tf=hwm/(gamma*kappa*ustar); literature baseline gamma=1"
        ),
    )
    parser.add_argument("--sgs", choices=("amd", "lasd"), default="amd")
    parser.add_argument("--amd-coefficient", type=float, default=0.212)
    parser.add_argument("--lasd-update-interval", type=int, default=2)
    parser.add_argument("--lasd-initial-coefficient", type=float, default=0.03)
    parser.add_argument("--mp5-strength", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument(
        "--target-cfl",
        type=float,
        help="adapt dt each step to this advective CFL",
    )
    parser.add_argument("--target-diffusive-cfl", type=float, default=0.5)
    parser.add_argument("--pressure-rtol", type=float, default=1.0e-6)
    parser.add_argument("--pressure-max-iterations", type=int, default=20)
    parser.add_argument("--pressure-restart", type=int, default=10)
    parser.add_argument(
        "--linear-solver",
        choices=("pcg", "gmres"),
        default="pcg",
    )
    parser.add_argument(
        "--krylov-execution",
        choices=("jax", "python"),
        default="jax",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--sample-start-step",
        type=int,
        help="first accepted step included in statistics (default: halfway)",
    )
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--flow-frame-count",
        type=int,
        default=0,
        help="number of streamwise-fluctuation flow frames to retain",
    )
    parser.add_argument(
        "--flow-frame-start-step",
        type=int,
        default=1,
        help="first step eligible for evenly spaced flow frames",
    )
    parser.add_argument(
        "--flow-frame-every",
        type=int,
        help="explicit frame stride; default distributes frames through the run",
    )
    parser.add_argument("--flow-gif-fps", type=int, default=10)
    parser.add_argument("--flow-slice-z-over-h", type=float, default=0.1)
    parser.add_argument("--perturbation", type=float, default=0.1)
    parser.add_argument("--restart", type=Path)
    parser.add_argument(
        "--reset-statistics-on-restart",
        action="store_true",
        help="retain the flow and solver state but begin a fresh averaging window",
    )
    parser.add_argument(
        "--initial-velocity",
        type=Path,
        help=(
            "load velocity from a checkpoint and reset time, statistics, and SGS memory"
        ),
    )
    parser.add_argument(
        "--max-run-seconds",
        type=float,
        help="pause at the next checkpoint after this much stepping time",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/neutral_loglaw_amd"),
    )
    parser.add_argument("--single", action="store_true")
    return parser.parse_args(argv)


def _validate(args: argparse.Namespace) -> int:
    if min(args.nx, args.ny, args.nz, args.steps) <= 0:
        raise SystemExit("grid dimensions and steps must be positive")
    if min(args.lx, args.ly, args.height, args.ustar, args.z0) <= 0.0:
        raise SystemExit("domain and wall scales must be positive")
    if args.z0 >= 0.5 * args.height / args.nz:
        raise SystemExit("z0 must lie below the first cell centre")
    if not 0 <= args.wall_matching_level < args.nz:
        raise SystemExit("wall-matching-level must lie in [0, nz)")
    if min(args.sample_every, args.log_every, args.checkpoint_every) <= 0:
        raise SystemExit("sampling and checkpoint intervals must be positive")
    if args.wall_filter_width is not None and (
        not math.isfinite(args.wall_filter_width) or args.wall_filter_width <= 0.0
    ):
        raise SystemExit("wall filter width must be positive and finite")
    if args.wall_temporal_filter_gamma is not None and (
        not math.isfinite(args.wall_temporal_filter_gamma)
        or args.wall_temporal_filter_gamma <= 0.0
    ):
        raise SystemExit("wall temporal filter gamma must be positive and finite")
    if args.flow_frame_count < 0:
        raise SystemExit("flow frame count must be nonnegative")
    if args.flow_frame_count > 0:
        if args.flow_gif_fps <= 0:
            raise SystemExit("flow GIF frame rate must be positive")
        if not 1 <= args.flow_frame_start_step <= args.steps:
            raise SystemExit("flow-frame-start-step must lie in [1, steps]")
        if args.flow_frame_every is not None and args.flow_frame_every <= 0:
            raise SystemExit("flow frame interval must be positive")
        if not 0.0 < args.flow_slice_z_over_h < 1.0:
            raise SystemExit("flow slice z/H must lie strictly between zero and one")
        available_steps = args.steps - args.flow_frame_start_step + 1
        required_steps = (
            args.flow_frame_count
            if args.flow_frame_every is None
            else 1 + (args.flow_frame_count - 1) * args.flow_frame_every
        )
        if required_steps > available_steps:
            raise SystemExit("requested flow frames do not fit in the run")
    if args.lasd_update_interval <= 0:
        raise SystemExit("LASD update interval must be positive")
    if not 0.0 <= args.lasd_initial_coefficient <= 0.81:
        raise SystemExit("LASD initial coefficient must lie in [0, 0.81]")
    if args.restart is not None and args.initial_velocity is not None:
        raise SystemExit("--restart and --initial-velocity are mutually exclusive")
    if args.reset_statistics_on_restart and args.restart is None:
        raise SystemExit("--reset-statistics-on-restart requires --restart")
    if args.target_cfl is not None and args.target_cfl <= 0.0:
        raise SystemExit("target CFL must be positive")
    if args.target_diffusive_cfl <= 0.0:
        raise SystemExit("target diffusive CFL must be positive")
    if args.pressure_rtol <= 0.0:
        raise SystemExit("pressure tolerance must be positive")
    if min(args.pressure_max_iterations, args.pressure_restart) <= 0:
        raise SystemExit("pressure iteration controls must be positive")
    if args.max_run_seconds is not None and args.max_run_seconds <= 0.0:
        raise SystemExit("max-run-seconds must be positive")
    sample_start = (
        args.steps // 2 if args.sample_start_step is None else args.sample_start_step
    )
    if not 0 <= sample_start < args.steps:
        raise SystemExit("sample-start-step must lie in [0, steps)")
    return sample_start


def main() -> None:
    args = parse_args()
    sample_start_step = _validate(args)

    from jax import config as jax_config

    if not args.single:
        jax_config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaxwind.momentum import (
        AMDModel,
        LASDModel,
        LASDState,
        MomentumConfig,
        MomentumOperators,
        WallModelState,
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

    dtype = jnp.float32 if args.single else jnp.float64
    grid = RectilinearGrid.uniform(
        args.nx,
        args.ny,
        args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.height,
    )
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    pressure_boundaries = PoissonBoundaryConditions(
        periodic,
        periodic,
        periodic,
        periodic,
        neumann,
        neumann,
    )
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
        pressure_boundaries,
        dtype=dtype,
        gmg=GMGConfig(smoother="auto", coarsening="auto", coarse_smooth=20),
        krylov=krylov,
    )
    von_karman = 0.4
    wall_matching_height = (args.wall_matching_level + 0.5) * args.height / args.nz
    wall_temporal_filter_timescale = (
        None
        if args.wall_temporal_filter_gamma is None
        else wall_matching_height
        / (args.wall_temporal_filter_gamma * von_karman * args.ustar)
    )
    config = MomentumConfig(
        friction_velocity=args.ustar,
        roughness_length=args.z0,
        von_karman=von_karman,
        wall_matching_level=args.wall_matching_level,
        wall_filter_width=args.wall_filter_width,
        wall_temporal_filter_timescale=wall_temporal_filter_timescale,
        mp5_dissipation_strength=args.mp5_strength,
        amd=AMDModel(coefficient=args.amd_coefficient),
        lasd=(
            LASDModel(
                update_interval=args.lasd_update_interval,
                initial_coefficient=args.lasd_initial_coefficient,
            )
            if args.sgs == "lasd"
            else None
        ),
    )
    solver = MomentumOperators(grid, pressure, config)

    def sample_kernel(
        sample_velocity: MACVelocity,
        sgs_coefficient,
        wall_velocity,
    ):
        cells = solver.cell_centered_velocity(sample_velocity)
        mean = jnp.mean(cells, axis=(1, 2))
        fluctuations = cells - mean[:, None, None, :]
        variances = jnp.mean(fluctuations * fluctuations, axis=(1, 2))
        resolved_minus_uw = -jnp.mean(
            solver.vertical_advective_flux(sample_velocity, cells)[..., 0],
            axis=(1, 2),
        )
        sgs_xz = jnp.mean(
            solver.vertical_sgs_stress_flux(
                cells,
                sgs_coefficient,
                wall_velocity=wall_velocity,
            )[..., 0],
            axis=(1, 2),
        )
        return (
            mean,
            variances,
            resolved_minus_uw,
            sgs_xz,
            jnp.mean(solver.wall_ustar(cells, wall_velocity=wall_velocity)),
        )

    compiled_sample = jax.jit(sample_kernel)
    flow_level = min(
        args.nz - 1,
        max(0, int(args.flow_slice_z_over_h * args.nz)),
    )

    def flow_frame_kernel(frame_velocity: MACVelocity):
        cells = solver.cell_centered_velocity(frame_velocity)
        streamwise = cells[..., 0]
        fluctuation = (
            streamwise - jnp.mean(streamwise, axis=(1, 2), keepdims=True)
        ) / args.ustar
        return (
            fluctuation[flow_level],
            fluctuation[:, args.ny // 2, :],
            fluctuation[:, :, args.nx // 2],
        )

    compiled_flow_frame = jax.jit(flow_frame_kernel)
    if args.flow_frame_count == 0:
        flow_frame_targets: tuple[int, ...] = ()
    elif args.flow_frame_every is not None:
        flow_frame_targets = tuple(
            args.flow_frame_start_step + index * args.flow_frame_every
            for index in range(args.flow_frame_count)
        )
    else:
        flow_frame_targets = tuple(
            int(value)
            for value in np.linspace(
                args.flow_frame_start_step,
                args.steps,
                args.flow_frame_count,
            ).round()
        )
        if len(set(flow_frame_targets)) != args.flow_frame_count:
            raise RuntimeError("evenly spaced flow frame steps are not unique")
    statistic_shapes = (
        (args.nz, 3),
        (args.nz, 3),
        (args.nz + 1,),
        (args.nz + 1,),
        (),
    )
    statistic_sums = [np.zeros(shape, dtype=np.float64) for shape in statistic_shapes]
    sample_count = 0
    timesteps: list[float] = []
    flow_frame_steps: list[int] = []
    flow_frame_times: list[float] = []
    flow_xy_frames: list[np.ndarray] = []
    flow_xz_frames: list[np.ndarray] = []
    flow_yz_frames: list[np.ndarray] = []
    elapsed_before_run = 0.0
    original_compilation_seconds: float | None = None

    def checkpoint_velocity(path: Path) -> MACVelocity:
        checkpoint = np.load(path)
        expected_shape = np.asarray((args.nz, args.ny, args.nx))
        if not np.array_equal(checkpoint["shape_zyx"], expected_shape):
            raise SystemExit("checkpoint grid shape does not match this run")
        return MACVelocity(
            *(
                jnp.asarray(checkpoint[f"velocity_{axis}"], dtype=dtype)
                for axis in "xyz"
            )
        )

    if args.restart is None:
        velocity = (
            solver.initial_log_profile(perturbation_amplitude=args.perturbation)
            if args.initial_velocity is None
            else checkpoint_velocity(args.initial_velocity)
        )
        simulation_time = 0.0
        step = 0
        solver.reset_lasd(velocity)
        solver.reset_wall_model(velocity)
    else:
        checkpoint = np.load(args.restart)
        expected_shape = np.asarray((args.nz, args.ny, args.nx))
        if not np.array_equal(checkpoint["shape_zyx"], expected_shape):
            raise SystemExit("restart grid shape does not match this run")
        checkpoint_sgs = (
            str(checkpoint["sgs_model"]) if "sgs_model" in checkpoint else "amd"
        )
        if checkpoint_sgs != args.sgs:
            raise SystemExit("restart SGS model does not match this run")
        checkpoint_matching_level = int(
            checkpoint["wall_matching_level"]
            if "wall_matching_level" in checkpoint
            else 0
        )
        if checkpoint_matching_level != args.wall_matching_level:
            raise SystemExit("restart wall matching level does not match")
        checkpoint_filter_width = (
            float(checkpoint["wall_filter_width"])
            if "wall_filter_width" in checkpoint
            else math.nan
        )
        requested_filter_width = (
            math.nan if args.wall_filter_width is None else args.wall_filter_width
        )
        if (
            not (
                math.isnan(checkpoint_filter_width)
                and math.isnan(requested_filter_width)
            )
            and checkpoint_filter_width != requested_filter_width
        ):
            raise SystemExit("restart wall spatial filter does not match")
        checkpoint_timescale = (
            float(checkpoint["wall_temporal_filter_timescale"])
            if "wall_temporal_filter_timescale" in checkpoint
            else math.nan
        )
        requested_timescale = (
            math.nan
            if wall_temporal_filter_timescale is None
            else wall_temporal_filter_timescale
        )
        if not (
            math.isnan(checkpoint_timescale) and math.isnan(requested_timescale)
        ) and not math.isclose(
            checkpoint_timescale,
            requested_timescale,
            rel_tol=1.0e-12,
        ):
            raise SystemExit("restart wall temporal filter does not match")
        if (
            not args.reset_statistics_on_restart
            and int(checkpoint["sample_start_step"]) != sample_start_step
        ):
            raise SystemExit("restart sample-start-step does not match this run")
        velocity = MACVelocity(
            *(
                jnp.asarray(checkpoint[f"velocity_{axis}"], dtype=dtype)
                for axis in "xyz"
            )
        )
        simulation_time = float(checkpoint["simulation_time"])
        step = int(checkpoint["step"])
        previous_summary = None
        previous_summary_path = args.output_dir / "summary.json"
        if previous_summary_path.exists():
            previous_summary = json.loads(previous_summary_path.read_text())
        elapsed_before_run = float(
            checkpoint["elapsed_seconds"]
            if "elapsed_seconds" in checkpoint
            else (
                previous_summary.get("elapsed_seconds", 0.0)
                if previous_summary is not None
                else 0.0
            )
        )
        original_compilation_seconds = (
            float(checkpoint["compilation_seconds"])
            if "compilation_seconds" in checkpoint
            else (
                float(previous_summary["compilation_seconds"])
                if previous_summary is not None
                and "compilation_seconds" in previous_summary
                else None
            )
        )
        timesteps = list(np.asarray(checkpoint["timesteps"], dtype=float))
        if not args.reset_statistics_on_restart:
            sample_count = int(checkpoint["sample_count"])
            statistic_sums = [
                np.asarray(
                    checkpoint[f"statistic_sum_{index}"],
                    dtype=np.float64,
                )
                for index in range(len(statistic_shapes))
            ]
        if args.sgs == "lasd":
            required = tuple(f"lasd_{name}" for name in LASDState._fields)
            if not all(name in checkpoint for name in required):
                raise SystemExit("LASD restart is missing closure memory")
            solver.restore_lasd(
                LASDState(
                    *(jnp.asarray(checkpoint[name], dtype=dtype) for name in required)
                ),
                accepted_step=int(checkpoint["lasd_accepted_step"]),
                interval_time=float(checkpoint["lasd_interval_time"]),
            )
        if "pressure" in checkpoint:
            solver.restore_pressure(checkpoint["pressure"])
        if wall_temporal_filter_timescale is not None:
            if "wall_filtered_velocity" not in checkpoint:
                raise SystemExit("temporal wall-model restart is missing memory")
            solver.restore_wall_model(
                WallModelState(
                    jnp.asarray(
                        checkpoint["wall_filtered_velocity"],
                        dtype=dtype,
                    )
                )
            )
    if step > args.steps:
        raise SystemExit("restart step exceeds requested steps")
    if flow_frame_targets and step >= flow_frame_targets[0]:
        raise SystemExit(
            "restart begins after requested flow frames; restart frame memory "
            "is intentionally not checkpointed"
        )

    def active_sgs_coefficient():
        lasd = solver.lasd_state
        return (
            lasd.coefficient
            if lasd is not None
            else jnp.zeros((1,), dtype=velocity.x.dtype)
        )

    def timestep_for(velocity_value: MACVelocity) -> float:
        if args.target_cfl is None:
            return args.dt
        return solver.timestep_for_cfl(
            velocity_value,
            args.target_cfl,
            args.target_diffusive_cfl,
        )

    saved_pressure = solver.pressure
    saved_lasd = solver.lasd_state
    saved_lasd_progress = solver.lasd_progress
    saved_wall_model = solver.wall_model_state
    compile_started = time.perf_counter()
    warmup_timestep = timestep_for(velocity)
    compiled_velocity = solver.step(
        velocity,
        timestep=warmup_timestep,
        time=simulation_time,
    )
    jax.block_until_ready(compiled_velocity.x)
    solver.restore_pressure(saved_pressure)
    if saved_lasd is not None:
        solver.restore_lasd(
            saved_lasd,
            accepted_step=saved_lasd_progress[0],
            interval_time=saved_lasd_progress[1],
        )
    if saved_wall_model is not None:
        solver.restore_wall_model(saved_wall_model)
    compiled_values = compiled_sample(
        velocity,
        active_sgs_coefficient(),
        solver.active_wall_velocity(velocity),
    )
    jax.block_until_ready(compiled_values[0])
    if args.flow_frame_count > 0:
        compiled_frame = compiled_flow_frame(velocity)
        jax.block_until_ready(compiled_frame[0])
    compilation_elapsed = time.perf_counter() - compile_started
    print(f"[compile] kernels ready in {compilation_elapsed:.3f}s", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint() -> None:
        payload: dict[str, object] = {
            "shape_zyx": np.asarray(grid.shape),
            "velocity_x": np.asarray(velocity.x),
            "velocity_y": np.asarray(velocity.y),
            "velocity_z": np.asarray(velocity.z),
            "step": step,
            "simulation_time": simulation_time,
            "timesteps": np.asarray(timesteps),
            "sample_start_step": sample_start_step,
            "sample_count": sample_count,
            "sgs_model": args.sgs,
            "elapsed_seconds": (elapsed_before_run + time.perf_counter() - started),
            "compilation_seconds": (
                compilation_elapsed
                if original_compilation_seconds is None
                else original_compilation_seconds
            ),
            "wall_matching_level": args.wall_matching_level,
            "wall_filter_width": (
                np.nan if args.wall_filter_width is None else args.wall_filter_width
            ),
            "wall_temporal_filter_timescale": (
                np.nan
                if wall_temporal_filter_timescale is None
                else wall_temporal_filter_timescale
            ),
        }
        payload.update(
            {
                f"statistic_sum_{index}": value
                for index, value in enumerate(statistic_sums)
            }
        )
        payload["pressure"] = np.asarray(solver.pressure)
        lasd = solver.lasd_state
        if lasd is not None:
            accepted_step, interval_time = solver.lasd_progress
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
                    "lasd_accepted_step": accepted_step,
                    "lasd_interval_time": interval_time,
                }
            )
        wall_model = solver.wall_model_state
        if wall_model is not None:
            payload["wall_filtered_velocity"] = np.asarray(wall_model.filtered_velocity)
        destination = args.output_dir / "checkpoint.npz"
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, destination)

    diagnostics = []
    started = time.perf_counter()
    while step < args.steps:
        timestep = timestep_for(velocity)
        velocity = solver.step(
            velocity,
            timestep=timestep,
            time=simulation_time,
        )
        simulation_time += timestep
        timesteps.append(timestep)
        step += 1
        if step >= sample_start_step and (
            (step - sample_start_step) % args.sample_every == 0 or step == args.steps
        ):
            values = tuple(
                np.asarray(value)
                for value in compiled_sample(
                    velocity,
                    active_sgs_coefficient(),
                    solver.active_wall_velocity(velocity),
                )
            )
            for accumulator, value in zip(statistic_sums, values, strict=True):
                accumulator += value
            sample_count += 1
        if (
            flow_frame_targets
            and len(flow_frame_steps) < args.flow_frame_count
            and step == flow_frame_targets[len(flow_frame_steps)]
        ):
            xy_frame, xz_frame, yz_frame = (
                np.asarray(value) for value in compiled_flow_frame(velocity)
            )
            flow_frame_steps.append(step)
            flow_frame_times.append(simulation_time)
            flow_xy_frames.append(xy_frame)
            flow_xz_frames.append(xz_frame)
            flow_yz_frames.append(yz_frame)
        if step % args.log_every == 0 or step == args.steps:
            diagnostic = solver.diagnostic(
                velocity,
                timestep=timestep,
                time=simulation_time,
            )
            diagnostics.append(diagnostic)
            divergence_rms = diagnostic.divergence_norm / math.sqrt(
                args.lx * args.ly * args.height
            )
            turnover_time = simulation_time * args.ustar / args.height
            sgs_status = ""
            if solver.lasd_state is not None:
                coefficient = np.asarray(solver.lasd_state.coefficient)
                sgs_status = (
                    f" Cs2={np.mean(coefficient):.4f}"
                    f"[{np.min(coefficient):.4f},{np.max(coefficient):.4f}]"
                    f" std={np.std(coefficient):.4f}"
                )
            print(
                f"step={step}/{args.steps} t*={turnover_time:.4f} "
                f"CFL={diagnostic.maximum_cfl:.4f} "
                f"CFLnu={diagnostic.maximum_diffusive_cfl:.4f} "
                f"ustar/target={diagnostic.mean_wall_ustar / args.ustar:.4f} "
                f"divRMS={divergence_rms:.3e}{sgs_status}",
                flush=True,
            )
        if step % args.checkpoint_every == 0:
            save_checkpoint()
            if (
                args.max_run_seconds is not None
                and time.perf_counter() - started >= args.max_run_seconds
            ):
                print(
                    f"[paused] checkpointed step={step}/{args.steps}",
                    flush=True,
                )
                return

    jax.block_until_ready(velocity.x)
    elapsed = elapsed_before_run + time.perf_counter() - started
    save_checkpoint()
    if len(flow_frame_steps) != args.flow_frame_count:
        raise RuntimeError(
            f"captured {len(flow_frame_steps)} of "
            f"{args.flow_frame_count} requested flow frames"
        )
    if sample_count == 0:
        raise RuntimeError("no profile statistics were sampled")
    if not diagnostics:
        diagnostics.append(
            solver.diagnostic(
                velocity,
                timestep=timesteps[-1],
                time=simulation_time,
            )
        )
    averaged = [value / sample_count for value in statistic_sums]
    mean, variances, resolved_minus_uw, sgs_xz, mean_wall_ustar = averaged
    total_stress = resolved_minus_uw + sgs_xz
    z = (np.arange(args.nz) + 0.5) * args.height / args.nz
    z_faces = np.arange(args.nz + 1) * args.height / args.nz
    target_velocity = args.ustar / config.von_karman * np.log(z / args.z0)
    target_stress = args.ustar**2 * (1.0 - z_faces / args.height)
    resolved_minus_uw_cells = 0.5 * (resolved_minus_uw[:-1] + resolved_minus_uw[1:])
    sgs_xz_cells = 0.5 * (sgs_xz[:-1] + sgs_xz[1:])
    total_stress_cells = resolved_minus_uw_cells + sgs_xz_cells
    target_stress_cells = args.ustar**2 * (1.0 - z / args.height)
    fit_mask = (z / args.height >= 0.05) & (z / args.height <= 0.3)
    design = np.column_stack((np.log(z[fit_mask]), np.ones(np.sum(fit_mask))))
    slope, intercept = np.linalg.lstsq(design, mean[fit_mask, 0], rcond=None)[0]
    fitted_ustar = config.von_karman * slope
    fitted_z0 = float(np.exp(-intercept / slope)) if slope > 0.0 else float("nan")
    fixed_z0_log = np.log(z[fit_mask] / args.z0)
    fixed_z0_slope = float(
        np.dot(fixed_z0_log, mean[fit_mask, 0]) / np.dot(fixed_z0_log, fixed_z0_log)
    )
    fixed_z0_fitted_ustar = config.von_karman * fixed_z0_slope
    loglaw_error_plus = (mean[fit_mask, 0] - target_velocity[fit_mask]) / args.ustar
    loglaw_rmse_plus = float(np.sqrt(np.mean(loglaw_error_plus**2)))
    loglaw_bias_plus = float(np.mean(loglaw_error_plus))
    stress_rmse = float(
        np.sqrt(np.mean(((total_stress - target_stress) / args.ustar**2) ** 2))
    )

    profile_path = args.output_dir / "mean_profile.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "z_over_h",
                "z_over_z0",
                "mean_u_over_ustar",
                "target_log_u_over_ustar",
                "var_u_over_ustar2",
                "var_v_over_ustar2",
                "var_w_over_ustar2",
                "resolved_minus_uw_over_ustar2",
                "sgs_xz_over_ustar2",
                "total_stress_over_ustar2",
                "target_stress_over_ustar2",
            )
        )
        writer.writerows(
            zip(
                z / args.height,
                z / args.z0,
                mean[:, 0] / args.ustar,
                target_velocity / args.ustar,
                variances[:, 0] / args.ustar**2,
                variances[:, 1] / args.ustar**2,
                variances[:, 2] / args.ustar**2,
                resolved_minus_uw_cells / args.ustar**2,
                sgs_xz_cells / args.ustar**2,
                total_stress_cells / args.ustar**2,
                target_stress_cells / args.ustar**2,
                strict=True,
            )
        )

    stress_path = args.output_dir / "stress_faces.csv"
    with stress_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "z_face_over_h",
                "resolved_minus_uw_over_ustar2",
                "sgs_xz_over_ustar2",
                "total_stress_over_ustar2",
                "target_stress_over_ustar2",
            )
        )
        writer.writerows(
            zip(
                z_faces / args.height,
                resolved_minus_uw / args.ustar**2,
                sgs_xz / args.ustar**2,
                total_stress / args.ustar**2,
                target_stress / args.ustar**2,
                strict=True,
            )
        )

    final_diagnostic = diagnostics[-1]
    final_divergence_rms = final_diagnostic.divergence_norm / math.sqrt(
        args.lx * args.ly * args.height
    )
    summary = {
        "backend": jax.default_backend(),
        "dtype": str(dtype),
        "shape_zyx": grid.shape,
        "domain_lx_ly_h": [args.lx, args.ly, args.height],
        "steps": args.steps,
        "simulation_time": simulation_time,
        "turnover_time_t_ustar_over_h": simulation_time * args.ustar / args.height,
        "sample_start_step": sample_start_step,
        "samples": sample_count,
        "minimum_dt": min(timesteps),
        "maximum_dt": max(timesteps),
        "elapsed_seconds": elapsed,
        "compilation_seconds": (
            compilation_elapsed
            if original_compilation_seconds is None
            else original_compilation_seconds
        ),
        "target_ustar": args.ustar,
        "wall_matching_level": args.wall_matching_level,
        "wall_matching_height": wall_matching_height,
        "wall_matching_height_over_z0": wall_matching_height / args.z0,
        "wall_filter_width_grid_cells": args.wall_filter_width,
        "wall_temporal_filter_gamma": args.wall_temporal_filter_gamma,
        "wall_temporal_filter_timescale": wall_temporal_filter_timescale,
        "mean_sampled_wall_ustar": float(mean_wall_ustar),
        "fitted_loglaw_ustar": float(fitted_ustar),
        "fixed_z0_fitted_loglaw_ustar": float(fixed_z0_fitted_ustar),
        "fixed_z0_fitted_loglaw_ustar_over_target": float(
            fixed_z0_fitted_ustar / args.ustar
        ),
        "target_z0": args.z0,
        "fitted_loglaw_z0": fitted_z0,
        "loglaw_fit_z_over_h": [0.05, 0.3],
        "loglaw_rmse_wall_units": loglaw_rmse_plus,
        "loglaw_bias_wall_units": loglaw_bias_plus,
        "first_cell_mean_u_over_ustar": float(mean[0, 0] / args.ustar),
        "first_cell_jump_u_over_ustar": float((mean[1, 0] - mean[0, 0]) / args.ustar),
        "total_stress_rmse_ustar2": stress_rmse,
        "sgs_model": args.sgs,
        "amd_coefficient": args.amd_coefficient if args.sgs == "amd" else None,
        "lasd": (
            {
                "update_interval": args.lasd_update_interval,
                "initial_coefficient": args.lasd_initial_coefficient,
                "final_minimum_coefficient": float(
                    np.min(np.asarray(solver.lasd_state.coefficient))
                ),
                "final_mean_coefficient": float(
                    np.mean(np.asarray(solver.lasd_state.coefficient))
                ),
                "final_maximum_coefficient": float(
                    np.max(np.asarray(solver.lasd_state.coefficient))
                ),
                "final_coefficient_std": float(
                    np.std(np.asarray(solver.lasd_state.coefficient))
                ),
            }
            if solver.lasd_state is not None
            else None
        ),
        "mp5_dissipation_strength": args.mp5_strength,
        "linear_solver": args.linear_solver,
        "pressure_relative_tolerance": args.pressure_rtol,
        "pressure_max_iterations": args.pressure_max_iterations,
        "krylov_execution": args.krylov_execution,
        "projection_method": "full",
        "flow_frames": len(flow_frame_steps),
        "flow_frame_start_step": (
            args.flow_frame_start_step if flow_frame_steps else None
        ),
        "flow_frame_every_steps": args.flow_frame_every,
        "flow_frame_steps": flow_frame_steps,
        "flow_frame_simulation_times": flow_frame_times,
        "flow_slice_z_over_h": (
            (flow_level + 0.5) / args.nz if flow_frame_steps else None
        ),
        "final_divergence_rms": final_divergence_rms,
        "final": asdict(final_diagnostic),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flow_stem = f"flow_uprime_{len(flow_frame_steps)}frames"
    flow_gif_path = args.output_dir / f"{flow_stem}.gif"
    if flow_frame_steps:
        from matplotlib.animation import FuncAnimation, PillowWriter

        xy_frames = np.stack(flow_xy_frames)
        xz_frames = np.stack(flow_xz_frames)
        yz_frames = np.stack(flow_yz_frames)
        frame_data_path = args.output_dir / f"{flow_stem}.npz"
        np.savez_compressed(
            frame_data_path,
            u_prime_over_ustar_xy=xy_frames,
            u_prime_over_ustar_xz=xz_frames,
            u_prime_over_ustar_yz=yz_frames,
            steps=np.asarray(flow_frame_steps),
            simulation_time=np.asarray(flow_frame_times),
            z_over_h=np.asarray((flow_level + 0.5) / args.nz),
        )
        color_limit = float(
            np.quantile(
                np.abs(
                    np.concatenate(
                        (xy_frames.ravel(), xz_frames.ravel(), yz_frames.ravel())
                    )
                ),
                0.995,
            )
        )
        color_limit = max(color_limit, 1.0e-6)
        flow_figure = plt.figure(figsize=(10.0, 6.0))
        flow_grid = flow_figure.add_gridspec(2, 2, width_ratios=(2.0, 1.0))
        flow_axes = (
            flow_figure.add_subplot(flow_grid[0, 0]),
            flow_figure.add_subplot(flow_grid[1, 0]),
            flow_figure.add_subplot(flow_grid[:, 1]),
        )
        flow_figure.subplots_adjust(
            left=0.07,
            right=0.89,
            bottom=0.09,
            top=0.88,
            wspace=0.24,
            hspace=0.38,
        )
        image_options = {
            "origin": "lower",
            "vmin": -color_limit,
            "vmax": color_limit,
            "cmap": "RdBu_r",
            "interpolation": "bilinear",
            "aspect": "auto",
        }
        flow_images = (
            flow_axes[0].imshow(
                xy_frames[0],
                extent=(0.0, args.lx, 0.0, args.ly),
                **image_options,
            ),
            flow_axes[1].imshow(
                xz_frames[0],
                extent=(0.0, args.lx, 0.0, args.height),
                **image_options,
            ),
            flow_axes[2].imshow(
                yz_frames[0],
                extent=(0.0, args.ly, 0.0, args.height),
                **image_options,
            ),
        )
        flow_axes[0].set(
            xlabel="x / H",
            ylabel="y / H",
            title=rf"$u'/u_*$ at $z/H={(flow_level + 0.5) / args.nz:.3f}$",
        )
        flow_axes[1].set(
            xlabel="x / H",
            ylabel="z / H",
            title=rf"$u'/u_*$ at $y/H={0.5 * args.ly:.3f}$",
        )
        flow_axes[2].set(
            xlabel="y / H",
            ylabel="z / H",
            title=rf"$u'/u_*$ at $x/H={0.5 * args.lx:.3f}$",
        )
        flow_colorbar = flow_figure.colorbar(
            flow_images[0], ax=flow_axes, shrink=0.82, pad=0.02
        )
        flow_colorbar.set_label(r"streamwise fluctuation $u'/u_*$")
        flow_title = flow_figure.suptitle("")

        def update_flow_frame(frame_index: int):
            flow_images[0].set_data(xy_frames[frame_index])
            flow_images[1].set_data(xz_frames[frame_index])
            flow_images[2].set_data(yz_frames[frame_index])
            flow_title.set_text(
                f"Filtered-wall {args.sgs.upper()}: "
                f"step {flow_frame_steps[frame_index]}, "
                f"t*={flow_frame_times[frame_index] * args.ustar / args.height:.3f}"
            )
            return (*flow_images, flow_title)

        flow_animation = FuncAnimation(
            flow_figure,
            update_flow_frame,
            frames=len(flow_frame_steps),
            interval=1000.0 / args.flow_gif_fps,
            blit=False,
        )
        flow_animation.save(
            flow_gif_path,
            writer=PillowWriter(fps=args.flow_gif_fps),
            dpi=80,
        )
        plt.close(flow_figure)
        print(
            f"[output] {len(flow_frame_steps)} flow frames: "
            f"{frame_data_path} and {flow_gif_path}",
            flush=True,
        )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axes[0].semilogy(
        mean[:, 0] / args.ustar,
        z / args.height,
        label=f"{args.sgs.upper()} LES",
    )
    axes[0].semilogy(
        target_velocity / args.ustar,
        z / args.height,
        "--",
        label="neutral log law",
    )
    axes[0].set_xlabel("U / u*")
    axes[0].set_ylabel("z / H")
    axes[0].legend()
    axes[0].set_title(
        "fixed-z0 fit u*/target="
        f"{fixed_z0_fitted_ustar / args.ustar:.3f}; "
        f"free fit={fitted_ustar / args.ustar:.3f}"
    )
    axes[1].plot(
        resolved_minus_uw / args.ustar**2,
        z_faces / args.height,
        label="resolved",
    )
    axes[1].plot(
        sgs_xz / args.ustar**2,
        z_faces / args.height,
        label="SGS",
    )
    axes[1].plot(
        total_stress / args.ustar**2,
        z_faces / args.height,
        label="total",
    )
    axes[1].plot(
        target_stress / args.ustar**2,
        z_faces / args.height,
        "--",
        label="1-z/H",
    )
    axes[1].set_xlabel("stress / u*²")
    axes[1].set_ylabel("z / H")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        f"64-compatible pressure-driven {args.sgs.upper()}, "
        f"t*={summary['turnover_time_t_ustar_over_h']:.2f}"
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "loglaw_and_stress.png", dpi=180)
    plt.close(figure)
    print(f"[done] elapsed={elapsed:.3f}s summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
