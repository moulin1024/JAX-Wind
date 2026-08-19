"""Replay precursor inflow with strict CUDA-Fortran inlet semantics."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import numpy as np

from applications.pressure_driven_lasd.config import CaseConfig
from applications.pressure_driven_lasd.evaluate import _configure_source_paths
from applications.pressure_driven_lasd.problem import build_pressure_driven_problem

from .evaluate import (
    _load_developed_state,
    _save_state,
    _turbine_fingerprint,
    _turbine_summary,
)
from .legacy_inflow import (
    STRICT_LEGACY_INFLOW,
    build_accepted_state_transform,
    force_inflow_component as _legacy_force_inflow_component,
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
    legacy_inflow_directory: Path | None,
    read_buffer: int,
    frame_count: int,
    gif_fps: int,
    turbine: Any | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Run only the strict legacy-inlet main domain from recorded precursor data."""

    if isinstance(main_steps, bool) or not isinstance(main_steps, int) or main_steps <= 0:
        raise ValueError("main steps must be positive")
    contract = STRICT_LEGACY_INFLOW
    if main_steps % contract.update_interval_steps:
        raise ValueError("main steps must be divisible by the legacy inlet interval")
    frame_offsets = evenly_spaced_frame_offsets(main_steps, frame_count)

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
    from jaxwind.physics import WindTunnelModel

    runtime = JaxRuntime.from_initialized_jax(jax)
    required_paths = [runtime.checkpoint_path(restart)]
    if legacy_inflow_directory is None:
        required_paths.append(runtime.checkpoint_path(recording))
    for required in required_paths:
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
    actuator_disk = (
        turbine.to_actuator_disk(scales=base_problem.scales)
        if turbine is not None
        else None
    )
    turbine_body = (
        turbine.to_nacelle_tower(scales=base_problem.scales)
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
    problem = build_pressure_driven_problem(
        case,
        runtime=runtime,
        wind_tunnel_model=wind_tunnel,
        pressure_acceleration_m_s2=0.0,
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

    class LegacyBinaryPlayback:
        def __init__(self, directory: Path) -> None:
            self.runtime = runtime
            self.config = PrecursorPlaybackConfig(buffer_samples=read_buffer)
            self.sample_count = main_steps // contract.update_interval_steps
            self.covered_steps = main_steps
            self.progress_interval_steps = (
                read_buffer * contract.update_interval_steps
            )
            self._first_step = int(main.clock.step)
            shape = (
                contract.width,
                case.domain.ny,
                case.domain.nz,
                main_steps // contract.update_interval_steps,
            )
            self._values = tuple(
                np.memmap(
                    directory / f"p000_inflow_{name}.bin",
                    dtype=np.float32,
                    mode="r",
                    shape=shape,
                    order="F",
                )
                for name in ("u", "v", "w")
            )
            self._cached_index = -1
            self._cached_environment = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self._values = ()

        def environment(self, state, *, step=None, time=None):
            del time
            current_step = int(state.clock.step) if step is None else step
            index = (
                current_step - self._first_step
            ) // contract.update_interval_steps
            if index != self._cached_index:
                blocks = [
                    np.asarray(values[..., index]).transpose(2, 1, 0)
                    for values in self._values
                ]
                velocity = state.fields.velocity

                def target(field, block):
                    payload = jnp.zeros_like(field.payload)
                    execution = jnp.asarray(
                        problem.scales.to_execution_velocity(block),
                        dtype=payload.dtype,
                    )
                    payload = payload.at[..., : contract.width].set(
                        execution[None, ...]
                    )
                    return replace(field, payload=payload)

                target_velocity = replace(
                    velocity,
                    x=target(velocity.x, blocks[0]),
                    y=target(velocity.y, blocks[1]),
                    z=replace(
                        velocity.z,
                        owned=target(velocity.z.owned, blocks[2]),
                    ),
                )
                from jaxwind.physics import ConcurrentPrecursorEnvironment
                self._cached_environment = ConcurrentPrecursorEnvironment(target_velocity)
                self._cached_index = index
            return self._cached_environment

    def observe(frame_state, completed: int) -> None:
        if completed not in frame_schedule:
            return
        frame = capture_xz_velocity(
            frame_state,
            problem,
            jax=jax,
            y_m=(
                None
                if turbine is None
                else getattr(turbine, "force_y_m", turbine.y_m)
            ),
        )
        if runtime.is_primary:
            frames.append(frame)
            frame_times.append(completed * case.time.dt_seconds)

    started = time.perf_counter()
    progress_started = time.perf_counter()

    legacy_transform = build_accepted_state_transform(
        contract=contract,
        jax=jax,
        jnp=jnp,
        ny=case.domain.ny,
    )

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

    playback_context = (
        LegacyBinaryPlayback(legacy_inflow_directory)
        if legacy_inflow_directory is not None
        else HDF5PrecursorPlayback(
            recording,
            runtime=runtime,
            state=main,
            config=PrecursorPlaybackConfig(
                section="inflow",
                buffer_samples=read_buffer,
            ),
        )
    )
    with playback_context as playback:
        if main_steps > playback.covered_steps:
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
            accepted_state_transform=legacy_transform,
        )
    main.fields.velocity.x.payload.block_until_ready()
    elapsed = time.perf_counter() - started

    fingerprint = (
        problem.physics_fingerprint
        + "|legacy-inlet-overwrite=strict"
        + f"|inflow-start-plane={contract.start_plane}"
        + f"|inflow-end-plane={contract.end_plane}"
        + f"|inflow-update-steps={contract.update_interval_steps}"
        + f"|inflow-cycle-updates={contract.cycle_interval_updates}"
    )
    fingerprint += (
        "|main-pressure-acceleration-m-s2="
        + float(
            0.0
        ).hex()
    )
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
            inlet_end_x_m=contract.end_plane * case.domain.dx_m,
            fps=gif_fps,
            turbine=turbine,
        )

    turbine_summary = _turbine_summary(turbine)
    summary = {
        "schema": "jaxwind.strict-fortran-main-replay.v1",
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
            "compatibility": "strict-cuda-fortran",
            "fringe_enabled": False,
            "inflow_enforcement": "legacy-overwrite",
            "inflow_start_plane": contract.start_plane,
            "inflow_end_plane": contract.end_plane,
            "inflow_update_steps": contract.update_interval_steps,
            "spanwise_cycle_updates": contract.cycle_interval_updates,
            "pressure_gradient_enabled": False,
            "pressure_acceleration_m_s2": 0.0,
            "nonlinear_scheme": "legacy-fortran-pre-rhs-filtering",
            "source_section": "inflow",
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
