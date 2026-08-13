"""Pressure-driven offline precursor and main-domain fringe workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

from applications.pressure_driven_lasd.config import CaseConfig
from applications.pressure_driven_lasd.evaluate import _configure_source_paths
from applications.pressure_driven_lasd.problem import (
    PressureDrivenProblem,
    build_pressure_driven_problem,
)
from .visualization import (
    capture_xz_velocity,
    evenly_spaced_frame_offsets,
    save_flow_frames,
    write_flow_gif,
)


def _load_developed_state(problem, path, *, runtime, jnp):
    from jaxwind.effects import load_boussinesq_checkpoint

    return load_boussinesq_checkpoint(
        runtime.checkpoint_path(path),
        layout=problem.solver.checkpoint_layout(jnp.asarray),
        config=problem.integrator,
        scale_fingerprint=problem.scales.fingerprint,
        closure_fingerprint=problem.closure_fingerprint,
        physics_fingerprint=problem.physics_fingerprint,
        allow_single_process_reshard=(
            runtime.process_count == 1 and runtime.local_devices > 1
        ),
    )


def _save_state(path, state, problem, *, runtime, physics_fingerprint=None):
    from jaxwind.effects import save_boussinesq_checkpoint

    save_boussinesq_checkpoint(
        runtime.checkpoint_path(path),
        state,
        scale_fingerprint=problem.scales.fingerprint,
        physics_fingerprint=physics_fingerprint or problem.physics_fingerprint,
    )


def _fringe_fingerprint(problem: PressureDrivenProblem, fringe: Any) -> str:
    return (
        problem.physics_fingerprint
        + "|offline-fringe=v1"
        + f"|start={float(fringe.start_x).hex()}"
        + f"|relaxation={float(fringe.relaxation_time).hex()}"
    )


def evaluate(
    case: CaseConfig,
    *,
    restart: Path,
    output_dir: Path,
    precursor_steps: int,
    main_steps: int,
    fringe_start_fraction: float,
    fringe_relaxation_seconds: float,
    section: str,
    sample_buffer: int,
    read_buffer: int,
    compression: str | None,
    frame_count: int,
    gif_fps: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Generate offline planes, then replay them through the production fringe."""

    if min(precursor_steps, main_steps) <= 0:
        raise ValueError("precursor and main steps must be positive")
    if main_steps > precursor_steps:
        raise ValueError("main steps cannot exceed recorded precursor steps")
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
        PrecursorRecordingConfig,
        run_main_with_precursor,
        run_precursor,
    )
    from jaxwind.physics import ConcurrentPrecursorFringe, WindTunnelModel

    runtime = JaxRuntime.from_initialized_jax(jax)
    local_restart = runtime.checkpoint_path(restart)
    if not local_restart.exists():
        raise FileNotFoundError(local_restart)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if runtime.is_primary:
        (output_dir / "resolved_case.toml").write_text(case.resolved_toml())
    runtime.synchronize("jaxwind-offline-precursor-output-ready")

    def progress_reporter(phase: str):
        phase_started = time.perf_counter()

        def report(_state, completed: int, total: int) -> None:
            if not runtime.is_primary:
                return
            _state.fields.velocity.x.payload.block_until_ready()
            elapsed = time.perf_counter() - phase_started
            remaining = elapsed * (total - completed) / completed
            print(
                f"{phase} step={completed}/{total} "
                f"elapsed={elapsed / 60.0:.1f}min "
                f"eta={remaining / 60.0:.1f}min",
                flush=True,
            )

        return report

    started = time.perf_counter()
    precursor_problem = build_pressure_driven_problem(case, runtime=runtime)
    developed = _load_developed_state(
        precursor_problem,
        restart,
        runtime=runtime,
        jnp=jnp,
    )
    initial_step = developed.clock.step
    recording_path = output_dir / "precursor.h5"
    precursor_final = run_precursor(
        developed,
        steps=precursor_steps,
        advance=precursor_problem.solver.advance,
        path=recording_path,
        runtime=runtime,
        recording=PrecursorRecordingConfig(
            sample_every=1,
            buffer_samples=sample_buffer,
            compression=compression,
            overwrite=overwrite,
        ),
        progress=progress_reporter("precursor"),
        compile_step=True,
        dt=precursor_problem.integrator.dt,
    )
    _save_state(
        output_dir / "precursor_final.npz",
        precursor_final,
        precursor_problem,
        runtime=runtime,
    )
    precursor_elapsed = time.perf_counter() - started

    start_x = precursor_problem.scales.to_execution_length(
        fringe_start_fraction * case.domain.lx_m
    )
    relaxation_time = precursor_problem.scales.to_execution_time(
        fringe_relaxation_seconds
    )
    fringe = ConcurrentPrecursorFringe(start_x, relaxation_time)
    main_problem = build_pressure_driven_problem(
        case,
        runtime=runtime,
        wind_tunnel_model=WindTunnelModel(fringe=fringe),
    )
    main = _load_developed_state(
        main_problem,
        restart,
        runtime=runtime,
        jnp=jnp,
    )
    with HDF5PrecursorPlayback(
        recording_path,
        runtime=runtime,
        state=main,
        config=PrecursorPlaybackConfig(
            section=section,
            buffer_samples=read_buffer,
        ),
    ) as playback:
        initial_environment = playback.environment(main)
        local_target_delta = float(
            jnp.max(
                jnp.abs(
                    initial_environment.velocity.x.payload
                    - main.fields.velocity.x.payload
                )
            )
        )
        scheduled_frames = set(frame_offsets)
        frames: list[Any] = []
        frame_times: list[float] = []

        def observe(frame_state, completed: int) -> None:
            if completed not in scheduled_frames:
                return
            frame = capture_xz_velocity(frame_state, main_problem, jax=jax)
            if runtime.is_primary:
                frames.append(frame)
                frame_times.append(completed * case.time.dt_seconds)

        main = run_main_with_precursor(
            main,
            steps=main_steps,
            advance=main_problem.solver.advance,
            playback=playback,
            compute_projection_residual=False,
            observer=observe,
            progress=progress_reporter("main"),
            compile_step=True,
            dt=main_problem.integrator.dt,
            observer_steps=frame_offsets,
        )
    main.fields.velocity.x.payload.block_until_ready()
    main_elapsed = time.perf_counter() - started - precursor_elapsed
    main_fingerprint = _fringe_fingerprint(main_problem, fringe)
    _save_state(
        output_dir / "main_final.npz",
        main,
        main_problem,
        runtime=runtime,
        physics_fingerprint=main_fingerprint,
    )

    frames_path = output_dir / "main_xz_frames.npz"
    gif_path = output_dir / "main_xz_velocity.gif"
    if runtime.is_primary:
        if len(frames) != frame_count:
            raise RuntimeError(
                f"captured {len(frames)} main frames; expected {frame_count}"
            )
        save_flow_frames(frames_path, frames, frame_times)
        write_flow_gif(
            gif_path,
            frames,
            frame_times,
            grid=main_problem.physical_grid,
            fringe_start_x_m=fringe_start_fraction * case.domain.lx_m,
            fps=gif_fps,
        )

    comparison = None
    if main_steps == precursor_steps:
        comparison = float(
            jnp.max(
                jnp.abs(
                    main.fields.velocity.x.payload
                    - precursor_final.fields.velocity.x.payload
                )
            )
        )
    elapsed = time.perf_counter() - started
    summary = {
        "schema": "jaxwind.offline-precursor-main.v1",
        "case": case.name,
        "restart": str(restart),
        "recording": str(recording_path),
        "runtime": {
            "backend": runtime.backend,
            "process_count": runtime.process_count,
            "global_devices": runtime.global_devices,
        },
        "precursor": {
            "initial_step": initial_step,
            "steps": precursor_steps,
            "final_step": int(precursor_final.clock.step),
            "section_samples": precursor_steps,
            "elapsed_seconds": precursor_elapsed,
            "steps_per_second": precursor_steps / precursor_elapsed,
        },
        "main": {
            "steps": main_steps,
            "final_step": int(main.clock.step),
            "fringe_enabled": True,
            "fringe_start_fraction": fringe_start_fraction,
            "fringe_relaxation_seconds": fringe_relaxation_seconds,
            "source_section": section,
            "initial_local_target_delta": local_target_delta,
            "local_difference_from_unforced_precursor": comparison,
            "frame_count": frame_count,
            "frames": str(frames_path),
            "gif": str(gif_path),
            "elapsed_seconds": main_elapsed,
            "steps_per_second": main_steps / main_elapsed,
        },
        "elapsed_seconds": elapsed,
    }
    if runtime.is_primary:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    return summary


__all__ = ["evaluate"]
