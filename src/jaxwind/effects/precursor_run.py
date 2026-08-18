"""Compiled execution loops for offline precursor generation and playback."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from jaxwind.integrators import PreviousTendency

from .precursor import (
    HDF5PrecursorPlayback,
    HDF5PrecursorRecorder,
    _state_payloads,
    finalize_precursor_recording,
)
from .precursor_config import PrecursorRecordingConfig
from .runtime import JaxRuntime


def run_main_with_precursor(
    state: Any,
    *,
    steps: int,
    advance: Callable[..., Any],
    playback: HDF5PrecursorPlayback,
    compute_projection_residual: bool = False,
    observer: Callable[[Any, int], None] | None = None,
    progress: Callable[[Any, int, int], None] | None = None,
    compile_step: bool = False,
    dt: float | None = None,
    observer_steps: tuple[int, ...] | None = None,
    accepted_state_transform: Callable[[Any, Any, int], Any] | None = None,
) -> Any:
    """Advance a main domain with clock-matched offline precursor inflow."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("main steps must be a positive integer")
    if not isinstance(getattr(state, "history", None), PreviousTendency):
        raise ValueError("offline fringe execution requires a warm AB2 main state")
    if compile_step and (dt is None or not math.isfinite(dt) or dt <= 0.0):
        raise ValueError("compiled main execution requires a positive finite dt")
    if observer_steps is not None:
        if tuple(sorted(set(observer_steps))) != observer_steps or any(
            value < 0 or value > steps for value in observer_steps
        ):
            raise ValueError("main observer steps must be sorted and unique")
    current = state
    host_step = int(state.clock.step)
    host_time = float(state.clock.time)
    scheduled_observers = None if observer_steps is None else set(observer_steps)
    if observer is not None and (
        scheduled_observers is None or 0 in scheduled_observers
    ):
        observer(current, 0)
    if not compile_step:
        for completed in range(1, steps + 1):
            environment = playback.environment(current)
            result = advance(
                current,
                environment=environment,
                compute_projection_residual=compute_projection_residual,
            )
            current = result.state
            if accepted_state_transform is not None:
                current = accepted_state_transform(current, environment, completed)
            if observer is not None:
                observer(current, completed)
            if progress is not None:
                progress(current, completed, steps)
        return current

    compiled_advance = playback.runtime.jax.jit(
        lambda current_state, current_environment: advance(
            current_state,
            environment=current_environment,
            compute_projection_residual=compute_projection_residual,
        ).state
    )
    for completed in range(1, steps + 1):
        environment = playback.environment(
            current,
            step=host_step,
            time=host_time,
        )
        current = compiled_advance(current, environment)
        if accepted_state_transform is not None:
            current = accepted_state_transform(current, environment, completed)
        host_step += 1
        host_time += dt
        if observer is not None and (
            scheduled_observers is None or completed in scheduled_observers
        ):
            observer(current, completed)
        if progress is not None and (
            completed == steps
            or completed % playback.config.buffer_samples == 0
        ):
            progress(current, completed, steps)
    return current


def run_precursor(
    state: Any,
    *,
    steps: int,
    advance: Callable[..., Any],
    path: str | Path,
    runtime: JaxRuntime,
    recording: PrecursorRecordingConfig = PrecursorRecordingConfig(),
    compute_projection_residual: bool = False,
    progress: Callable[[Any, int, int], None] | None = None,
    compile_step: bool = False,
    dt: float | None = None,
) -> Any:
    """Advance a warm state while recording its pre-step boundary planes."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("precursor steps must be a positive integer")
    if not isinstance(getattr(state, "history", None), PreviousTendency):
        raise ValueError(
            "offline precursor recording requires a developed warm AB2 state "
            "with previous-tendency history"
        )
    if compile_step and (dt is None or not math.isfinite(dt) or dt <= 0.0):
        raise ValueError("compiled precursor execution requires a positive finite dt")
    recorder = HDF5PrecursorRecorder(path, runtime=runtime, config=recording)
    current = state
    host_step = int(state.clock.step)
    host_time = float(state.clock.time)
    with recorder:
        if not compile_step:
            for completed in range(1, steps + 1):
                recorder.record(current)
                result = advance(
                    current,
                    compute_projection_residual=compute_projection_residual,
                )
                current = result.state
                if progress is not None:
                    progress(current, completed, steps)
        else:
            jax = runtime.jax
            compiled_blocks: dict[int, Callable[..., Any]] = {}
            recorder._initialize(current)

            def compiled_block(count: int) -> Callable[..., Any]:
                cached = compiled_blocks.get(count)
                if cached is not None:
                    return cached

                def block(current_state):
                    def body(carry, _unused):
                        velocity, scalar = _state_payloads(carry)
                        sections = recorder._extract_sections(velocity, scalar)
                        next_state = advance(
                            carry,
                            compute_projection_residual=(
                                compute_projection_residual
                            ),
                        ).state
                        return next_state, sections

                    return jax.lax.scan(
                        body,
                        current_state,
                        xs=None,
                        length=count,
                    )

                cached = jax.jit(block)
                compiled_blocks[count] = cached
                return cached

            completed = 0
            while completed < steps:
                count = min(recording.buffer_samples, steps - completed)
                block_start = current
                current, sections = compiled_block(count)(current)
                velocity, scalar = sections
                block_steps = np.arange(
                    host_step,
                    host_step + count,
                    dtype=np.int64,
                )
                block_times = np.empty(count, dtype=np.float64)
                for index in range(count):
                    block_times[index] = host_time
                    host_time += dt
                recorder.record_batch(
                    block_start,
                    velocity,
                    scalar,
                    steps=block_steps,
                    times=block_times,
                )
                host_step += count
                completed += count
                if progress is not None:
                    progress(current, completed, steps)
    finalize_precursor_recording(
        path,
        runtime=runtime,
        overwrite=recording.overwrite,
    )
    return current


__all__ = ["run_main_with_precursor", "run_precursor"]
