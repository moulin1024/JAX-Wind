"""Replay recorded precursor inflow through a fringe and optional turbine ADM."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

from applications.pressure_driven_lasd.config import CaseConfig
from applications.pressure_driven_lasd.evaluate import _configure_source_paths
from applications.pressure_driven_lasd.problem import build_pressure_driven_problem

from .evaluate import (
    _fringe_fingerprint,
    _load_developed_state,
    _save_state,
    _turbine_fingerprint,
)
from .visualization import (
    capture_xz_velocity,
    evenly_spaced_frame_offsets,
    save_flow_frames,
    write_flow_gif,
)


def replay_main(
    case: CaseConfig,
    *,
    restart: Path,
    recording: Path,
    output_dir: Path,
    main_steps: int,
    fringe_start_fraction: float,
    fringe_relaxation_seconds: float,
    section: str,
    read_buffer: int,
    frame_count: int,
    gif_fps: int,
    turbine: Any | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Run only the enforced main domain from an existing precursor file."""

    if isinstance(main_steps, bool) or not isinstance(main_steps, int) or main_steps <= 0:
        raise ValueError("main steps must be positive")
    frame_offsets = evenly_spaced_frame_offsets(main_steps, frame_count)
    if not 0.0 < fringe_start_fraction < 1.0:
        raise ValueError("fringe start fraction must lie strictly inside the domain")
    if (
        not math.isfinite(fringe_relaxation_seconds)
        or fringe_relaxation_seconds <= 0.0
    ):
        raise ValueError("fringe relaxation time must be finite and positive")

    _configure_source_paths()
    import jax

    jax.config.update("jax_enable_x64", case.numerics.dtype == "float64")
    import jax.numpy as jnp
    from jaxwind.effects import (
        HDF5PrecursorPlayback,
        JaxRuntime,
        PrecursorPlaybackConfig,
        run_main_with_precursor,
    )
    from jaxwind.physics import ConcurrentPrecursorFringe, WindTunnelModel

    runtime = JaxRuntime.from_initialized_jax(jax)
    for required in (runtime.checkpoint_path(restart), runtime.checkpoint_path(recording)):
        if not required.exists():
            raise FileNotFoundError(required)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if runtime.is_primary:
        (output_dir / "resolved_case.toml").write_text(case.resolved_toml())
    runtime.synchronize("jaxwind-offline-precursor-main-replay-output-ready")

    base_problem = build_pressure_driven_problem(case, runtime=runtime)
    start_x = base_problem.scales.to_execution_length(
        fringe_start_fraction * case.domain.lx_m
    )
    relaxation_time = base_problem.scales.to_execution_time(
        fringe_relaxation_seconds
    )
    fringe = ConcurrentPrecursorFringe(start_x, relaxation_time)
    actuator_disk = (
        turbine.to_actuator_disk(scales=base_problem.scales)
        if turbine is not None
        else None
    )
    wind_tunnel = (
        WindTunnelModel(fringe=fringe)
        if actuator_disk is None
        else WindTunnelModel(actuator_disk=actuator_disk, fringe=fringe)
    )
    problem = build_pressure_driven_problem(
        case,
        runtime=runtime,
        wind_tunnel_model=wind_tunnel,
    )
    main = _load_developed_state(
        problem,
        restart,
        runtime=runtime,
        jnp=jnp,
    )
    initial_step = int(main.clock.step)
    frame_schedule = set(frame_offsets)
    frames: list[Any] = []
    frame_times: list[float] = []

    def observe(frame_state, completed: int) -> None:
        if completed not in frame_schedule:
            return
        frame = capture_xz_velocity(frame_state, problem, jax=jax)
        if runtime.is_primary:
            frames.append(frame)
            frame_times.append(completed * case.time.dt_seconds)

    started = time.perf_counter()
    progress_started = time.perf_counter()

    def progress(state, completed: int, total: int) -> None:
        if not runtime.is_primary:
            return
        state.fields.velocity.x.payload.block_until_ready()
        elapsed = time.perf_counter() - progress_started
        remaining = elapsed * (total - completed) / completed
        print(
            f"main step={completed}/{total} "
            f"elapsed={elapsed / 60.0:.1f}min eta={remaining / 60.0:.1f}min",
            flush=True,
        )

    with HDF5PrecursorPlayback(
        recording,
        runtime=runtime,
        state=main,
        config=PrecursorPlaybackConfig(section=section, buffer_samples=read_buffer),
    ) as playback:
        if main_steps > playback.sample_count:
            raise ValueError("main steps exceed the precursor recording")
        initial_environment = playback.environment(main)
        target_delta = float(
            jnp.max(
                jnp.abs(
                    initial_environment.velocity.x.payload
                    - main.fields.velocity.x.payload
                )
            )
        )
        main = run_main_with_precursor(
            main,
            steps=main_steps,
            advance=problem.solver.advance,
            playback=playback,
            compute_projection_residual=False,
            observer=observe,
            progress=progress,
            compile_step=True,
            dt=problem.integrator.dt,
            observer_steps=frame_offsets,
        )
    main.fields.velocity.x.payload.block_until_ready()
    elapsed = time.perf_counter() - started

    fingerprint = _fringe_fingerprint(problem, fringe)
    fingerprint += _turbine_fingerprint(turbine)
    _save_state(
        output_dir / "main_final.npz",
        main,
        problem,
        runtime=runtime,
        physics_fingerprint=fingerprint,
    )

    frames_path = output_dir / "main_xz_frames.npz"
    gif_path = output_dir / "main_xz_velocity.gif"
    if runtime.is_primary:
        if len(frames) != frame_count:
            raise RuntimeError(f"captured {len(frames)} frames; expected {frame_count}")
        save_flow_frames(frames_path, frames, frame_times)
        write_flow_gif(
            gif_path,
            frames,
            frame_times,
            grid=problem.physical_grid,
            fringe_start_x_m=fringe_start_fraction * case.domain.lx_m,
            fps=gif_fps,
            turbine=turbine,
        )

    turbine_summary = None
    if turbine is not None:
        turbine_summary = {
            "model": "DTU-10MW simple ADM",
            "x_m": turbine.x_m,
            "y_m": turbine.y_m,
            "hub_height_m": turbine.hub_height_m,
            "rotor_diameter_m": turbine.rotor_diameter_m,
            "thrust_coefficient_prime": turbine.thrust_coefficient_prime,
            "smoothing_width_m": turbine.smoothing_width_m,
        }
    summary = {
        "schema": "jaxwind.offline-precursor-main-replay.v1",
        "case": case.name,
        "restart": str(restart),
        "recording": str(recording),
        "runtime": {
            "backend": runtime.backend,
            "process_count": runtime.process_count,
            "global_devices": runtime.global_devices,
        },
        "main": {
            "initial_step": initial_step,
            "steps": main_steps,
            "final_step": int(main.clock.step),
            "fringe_enabled": True,
            "fringe_start_fraction": fringe_start_fraction,
            "fringe_relaxation_seconds": fringe_relaxation_seconds,
            "source_section": section,
            "initial_local_target_delta": target_delta,
            "turbine": turbine_summary,
            "frame_count": frame_count,
            "frames": str(frames_path),
            "gif": str(gif_path),
            "elapsed_seconds": elapsed,
            "steps_per_second": main_steps / elapsed,
        },
    }
    if runtime.is_primary:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    return summary


__all__ = ["replay_main"]
