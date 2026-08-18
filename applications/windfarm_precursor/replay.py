"""Replay recorded precursor inflow through a fringe and optional turbine ADM."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import numpy as np

from applications.pressure_driven_lasd.config import CaseConfig
from applications.pressure_driven_lasd.evaluate import _configure_source_paths
from applications.pressure_driven_lasd.problem import build_pressure_driven_problem

from .evaluate import (
    _fringe_fingerprint,
    _load_developed_state,
    _save_state,
    _shift_fingerprint,
    _turbine_fingerprint,
    _turbine_summary,
)
from .visualization import (
    capture_xz_velocity,
    evenly_spaced_frame_offsets,
    save_flow_frames,
    write_flow_gif,
)


def _legacy_force_inflow_component(payload, target_payload, shift, blend, *, jnp):
    """Apply the literal vertical/staggered indexing of legacy ``force_inflow``."""
    source_block = jnp.roll(target_payload[..., :11], shift, axis=-2)
    source = source_block[..., 0]
    base = payload[..., 0]
    blended = payload[..., :9]
    shifted = base[:, 1:, :, None] + blend * (
        source[:, 1:, :] - base[:, 1:, :]
    )[..., None]
    blended = blended.at[:, :-1, :, :].set(shifted)
    payload = payload.at[..., :9].set(blended)
    return payload.at[..., 9:20].set(source_block)


def replay_main(
    case: CaseConfig,
    *,
    restart: Path,
    recording: Path,
    output_dir: Path,
    main_steps: int,
    fringe_start_fraction: float,
    fringe_relaxation_seconds: float,
    inflow_enforcement: str,
    legacy_inflow_update_steps: int,
    main_pressure_gradient: str,
    legacy_inflow_directory: Path | None,
    section: str,
    read_buffer: int,
    spanwise_shift_cells: int,
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
    if inflow_enforcement not in ("fringe", "legacy-overwrite"):
        raise ValueError("unsupported inflow enforcement")
    if legacy_inflow_update_steps <= 0:
        raise ValueError("legacy inflow update interval must be positive")
    if main_pressure_gradient not in ("on", "off"):
        raise ValueError("unsupported main pressure-gradient choice")

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
    turbine_body = (
        turbine.to_nacelle_tower(scales=base_problem.scales)
        if turbine is not None and hasattr(turbine, "to_nacelle_tower")
        else None
    )
    active_fringe = fringe if inflow_enforcement == "fringe" else None
    wind_tunnel = (
        WindTunnelModel(
            **({} if active_fringe is None else {"fringe": active_fringe})
        )
        if actuator_disk is None
        else WindTunnelModel(
            actuator_disk=actuator_disk,
            **({} if active_fringe is None else {"fringe": active_fringe}),
            **({} if turbine_body is None else {"turbine_body": turbine_body}),
        )
    )
    pressure_enabled = main_pressure_gradient == "on"
    problem = build_pressure_driven_problem(
        case,
        runtime=runtime,
        wind_tunnel_model=wind_tunnel,
        pressure_acceleration_m_s2=(
            None if pressure_enabled else 0.0
        ),
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
            self.sample_count = main_steps
            self._first_step = int(main.clock.step)
            shape = (11, case.domain.ny, case.domain.nz, main_steps // legacy_inflow_update_steps)
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
            index = (current_step - self._first_step) // legacy_inflow_update_steps
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
                    payload = payload.at[..., :11].set(execution[None, ...])
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

    legacy_transforms: dict[int, Any] = {}

    def legacy_transform(state, environment, completed: int):
        if (
            inflow_enforcement != "legacy-overwrite"
            or (completed - 1) % legacy_inflow_update_steps != 0
        ):
            return state
        shift = (
            (completed - 1) // (legacy_inflow_update_steps * 4)
        ) % case.domain.ny + 1
        transform = legacy_transforms.get(shift)
        if transform is None:
            blend = jnp.asarray(
                0.5
                * (
                    1.0
                    - jnp.cos(jnp.pi * jnp.arange(9, dtype=jnp.float32) / 8.0)
                )
            )

            def apply(current, target):
                velocity = current.fields.velocity
                target_velocity = target.velocity

                def component(field, target_field):
                    # Match legacy ``force_inflow`` literally.  Its blend uses
                    # ``field(..., k+1)`` and writes ``field(..., k)``; the
                    # first write targets the lower ghost and the last owned
                    # level is untouched.  Apply that one-level shift within
                    # every legacy z slab, discarding the ghost write.
                    return replace(
                        field,
                        payload=_legacy_force_inflow_component(
                            field.payload,
                            target_field.payload,
                            shift,
                            blend,
                            jnp=jnp,
                        ),
                    )

                updated_velocity = replace(
                    velocity,
                    x=component(velocity.x, target_velocity.x),
                    y=component(velocity.y, target_velocity.y),
                    z=replace(
                        velocity.z,
                        owned=component(velocity.z.owned, target_velocity.z.owned),
                    ),
                )
                return replace(
                    current,
                    fields=replace(current.fields, velocity=updated_velocity),
                )

            transform = jax.jit(apply)
            legacy_transforms[shift] = transform
        return transform(state, environment)

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
                section=section,
                buffer_samples=read_buffer,
                spanwise_shift_cells=spanwise_shift_cells,
            ),
        )
    )
    with playback_context as playback:
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
            accepted_state_transform=legacy_transform,
        )
    main.fields.velocity.x.payload.block_until_ready()
    elapsed = time.perf_counter() - started

    fingerprint = (
        _fringe_fingerprint(problem, fringe)
        if inflow_enforcement == "fringe"
        else problem.physics_fingerprint
        + f"|legacy-inlet-overwrite={legacy_inflow_update_steps}"
    )
    fingerprint += (
        "|main-pressure-acceleration-m-s2="
        + float(
            case.flow.pressure_acceleration_m_s2 if pressure_enabled else 0.0
        ).hex()
    )
    fingerprint += _turbine_fingerprint(turbine)
    fingerprint += _shift_fingerprint(spanwise_shift_cells)
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

    turbine_summary = _turbine_summary(turbine)
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
            "fringe_enabled": inflow_enforcement == "fringe",
            "inflow_enforcement": inflow_enforcement,
            "legacy_inflow_update_steps": legacy_inflow_update_steps,
            "pressure_gradient_enabled": pressure_enabled,
            "pressure_acceleration_m_s2": (
                case.flow.pressure_acceleration_m_s2 if pressure_enabled else 0.0
            ),
            "nonlinear_scheme": "legacy-fortran-pre-rhs-filtering",
            "fringe_start_fraction": fringe_start_fraction,
            "fringe_relaxation_seconds": fringe_relaxation_seconds,
            "source_section": section,
            "spanwise_shift_cells": spanwise_shift_cells,
            "spanwise_shift_m": spanwise_shift_cells * case.domain.dy_m,
            "spanwise_shift_recurrence_flowthroughs": (
                None
                if spanwise_shift_cells == 0
                else case.domain.ny
                // math.gcd(abs(spanwise_shift_cells), case.domain.ny)
            ),
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
