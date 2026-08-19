"""Strict CUDA-Fortran precursor and main-domain inlet workflow."""

from __future__ import annotations

import json
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
from .legacy_inflow import (
    STRICT_LEGACY_INFLOW,
    build_accepted_state_transform,
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


def _turbine_fingerprint(turbine: Any | None) -> str:
    if turbine is None:
        return ""
    value = (
        f"|turbine={getattr(turbine, 'model_name', type(turbine).__name__)}"
        f"|x={float(turbine.x_m).hex()}"
        f"|y={float(turbine.y_m).hex()}"
        f"|hub-height={float(turbine.hub_height_m).hex()}"
        f"|diameter={float(turbine.rotor_diameter_m).hex()}"
    )
    if not hasattr(turbine, "smearing_azimuthal_elements"):
        value += f"|epsilon={float(turbine.smoothing_width_m).hex()}"
    if hasattr(turbine, "thrust_coefficient_prime"):
        value += f"|ct-prime={float(turbine.thrust_coefficient_prime).hex()}"
    if getattr(turbine, "prescribed_inflow_velocity_m_s", 0.0) > 0.0:
        value += (
            f"|prescribed-uinf={float(turbine.prescribed_inflow_velocity_m_s).hex()}"
            f"|prescribed-ct={float(turbine.prescribed_thrust_coefficient).hex()}"
        )
        value += (
            f"|force-x-offset={float(turbine.force_x_offset_m).hex()}"
            f"|force-y-offset={float(turbine.force_y_offset_m).hex()}"
        )
    if hasattr(turbine, "rotor_speed_rpm"):
        speed = turbine.rotor.rotor_speed_rpm if turbine.rotor_speed_rpm is None else turbine.rotor_speed_rpm
        pitch = turbine.rotor.pitch_degrees if turbine.pitch_degrees is None else turbine.pitch_degrees
        value += f"|rpm={float(speed).hex()}|pitch={float(pitch).hex()}"
    if hasattr(turbine, "smearing_azimuthal_elements"):
        widths = turbine.element_smoothing_widths_m
        value += (
            f"|smearing-azimuthal-elements={turbine.smearing_azimuthal_elements}"
            f"|element-epsilon-min={float(min(widths)).hex()}"
            f"|element-epsilon-max={float(max(widths)).hex()}"
        )
    if hasattr(turbine, "nacelle_length_m"):
        value += (
            f"|nacelle-length={float(turbine.nacelle_length_m).hex()}"
            f"|nacelle-diameter={float(turbine.nacelle_diameter_m).hex()}"
            f"|nacelle-cd={float(turbine.nacelle_drag_coefficient).hex()}"
            f"|tower-base-diameter={float(turbine.tower_base_diameter_m).hex()}"
            f"|tower-top-diameter={float(turbine.tower_top_diameter_m).hex()}"
            f"|tower-cd={float(turbine.tower_drag_coefficient).hex()}"
            f"|body-epsilon={float(turbine.smoothing_width_m if turbine.body_smoothing_width_m is None else turbine.body_smoothing_width_m).hex()}"
        )
    return value


def _turbine_summary(turbine: Any | None) -> dict[str, Any] | None:
    if turbine is None:
        return None
    result = {
        "model": getattr(turbine, "model_name", type(turbine).__name__),
        "x_m": turbine.x_m,
        "y_m": turbine.y_m,
        "hub_height_m": turbine.hub_height_m,
        "rotor_diameter_m": turbine.rotor_diameter_m,
    }
    if not hasattr(turbine, "smearing_azimuthal_elements"):
        result["smoothing_width_m"] = turbine.smoothing_width_m
    if hasattr(turbine, "thrust_coefficient_prime"):
        result["thrust_coefficient_prime"] = turbine.thrust_coefficient_prime
    if getattr(turbine, "prescribed_inflow_velocity_m_s", 0.0) > 0.0:
        result["prescribed_inflow_velocity_m_s"] = turbine.prescribed_inflow_velocity_m_s
        result["prescribed_thrust_coefficient"] = turbine.prescribed_thrust_coefficient
        result["force_location_m"] = [turbine.force_x_m, turbine.force_y_m]
    if hasattr(turbine, "rotor"):
        result.update(
            blade_count=turbine.rotor.blade_count,
            radial_stations=len(turbine.rotor.element_radii_m),
            rotor_speed_rpm=(turbine.rotor.rotor_speed_rpm if turbine.rotor_speed_rpm is None else turbine.rotor_speed_rpm),
            pitch_degrees=(turbine.rotor.pitch_degrees if turbine.pitch_degrees is None else turbine.pitch_degrees),
            openfast_source=str(turbine.rotor.source),
        )
    if hasattr(turbine, "smearing_azimuthal_elements"):
        widths = turbine.element_smoothing_widths_m
        result.update(
            smearing_model="legacy-admr-element-size",
            smearing_azimuthal_elements=turbine.smearing_azimuthal_elements,
            element_smoothing_width_min_m=min(widths),
            element_smoothing_width_max_m=max(widths),
        )
    if hasattr(turbine, "nacelle_length_m"):
        result.update(
            nacelle_length_m=turbine.nacelle_length_m,
            nacelle_diameter_m=turbine.nacelle_diameter_m,
            nacelle_drag_coefficient=turbine.nacelle_drag_coefficient,
            tower_base_diameter_m=turbine.tower_base_diameter_m,
            tower_top_diameter_m=turbine.tower_top_diameter_m,
            tower_drag_coefficient=turbine.tower_drag_coefficient,
            body_smoothing_width_m=(
                turbine.smoothing_width_m
                if turbine.body_smoothing_width_m is None
                else turbine.body_smoothing_width_m
            ),
        )
    return result


def evaluate(
    case: CaseConfig,
    *,
    restart: Path,
    output_dir: Path,
    precursor_steps: int,
    main_steps: int,
    sample_buffer: int,
    read_buffer: int,
    compression: str | None,
    frame_count: int,
    gif_fps: int,
    turbine: Any | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Run the strict CUDA-Fortran precursor/inlet workflow through HDF5."""

    if min(precursor_steps, main_steps) <= 0:
        raise ValueError("precursor and main steps must be positive")
    if main_steps > precursor_steps:
        raise ValueError("main steps cannot exceed recorded precursor steps")
    contract = STRICT_LEGACY_INFLOW
    if precursor_steps % contract.update_interval_steps:
        raise ValueError(
            "precursor steps must be divisible by the legacy inlet interval"
        )
    frame_offsets = evenly_spaced_frame_offsets(main_steps, frame_count)

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
    from jaxwind.physics import WindTunnelModel

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
            sample_every=contract.update_interval_steps,
            buffer_samples=sample_buffer,
            section_width=contract.width,
            inflow_start_index=contract.zero_based_start,
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

    actuator_disk = (
        turbine.to_actuator_disk(scales=precursor_problem.scales)
        if turbine is not None
        else None
    )
    turbine_body = (
        turbine.to_nacelle_tower(scales=precursor_problem.scales)
        if turbine is not None and hasattr(turbine, "to_nacelle_tower")
        else None
    )
    wind_tunnel = (
        WindTunnelModel()
        if actuator_disk is None
        else WindTunnelModel(
            actuator_disk=actuator_disk,
            **({} if turbine_body is None else {"turbine_body": turbine_body}),
        )
    )
    main_problem = build_pressure_driven_problem(
        case,
        runtime=runtime,
        wind_tunnel_model=wind_tunnel,
        pressure_acceleration_m_s2=0.0,
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
            section="inflow",
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

        legacy_transform = build_accepted_state_transform(
            contract=contract,
            jax=jax,
            jnp=jnp,
            ny=case.domain.ny,
        )

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
            accepted_state_transform=legacy_transform,
        )
    main.fields.velocity.x.payload.block_until_ready()
    main_elapsed = time.perf_counter() - started - precursor_elapsed
    main_fingerprint = (
        main_problem.physics_fingerprint
        + "|legacy-inlet-overwrite=strict"
        + f"|inflow-start-plane={contract.start_plane}"
        + f"|inflow-end-plane={contract.end_plane}"
        + f"|inflow-update-steps={contract.update_interval_steps}"
        + f"|inflow-cycle-updates={contract.cycle_interval_updates}"
        + "|main-pressure-acceleration-m-s2="
        + float(0.0).hex()
        + _turbine_fingerprint(turbine)
    )
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
            inlet_end_x_m=contract.end_plane * case.domain.dx_m,
            fps=gif_fps,
            turbine=turbine,
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
        "schema": "jaxwind.strict-fortran-precursor-main.v1",
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
            "section_samples": (
                precursor_steps // contract.update_interval_steps
            ),
            "section_width_planes": contract.width,
            "sample_interval_steps": contract.update_interval_steps,
            "elapsed_seconds": precursor_elapsed,
            "steps_per_second": precursor_steps / precursor_elapsed,
        },
        "main": {
            "steps": main_steps,
            "final_step": int(main.clock.step),
            "compatibility": "strict-cuda-fortran",
            "fringe_enabled": False,
            "inflow_enforcement": "legacy-overwrite",
            "inflow_start_plane": contract.start_plane,
            "inflow_end_plane": contract.end_plane,
            "inflow_update_steps": contract.update_interval_steps,
            "spanwise_cycle_updates": contract.cycle_interval_updates,
            "pressure_gradient_enabled": False,
            "pressure_acceleration_m_s2": 0.0,
            "source_section": "inflow",
            "turbine": _turbine_summary(turbine),
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
