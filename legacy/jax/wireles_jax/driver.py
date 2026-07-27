from __future__ import annotations

import time
from collections.abc import Callable

import jax

from .config import Params
from .diagnostics import diagnostics, validate_cfl, validate_lasd_cfl
from .grid import make_operators
from .init import initial_state
from .state import Diagnostics, FlowState
from .timestep import step, step_ab2_lasd_skip, step_ab2_lasd_update


def _add_timing(diag: Diagnostics, params: Params, start_time: float) -> Diagnostics:
    diag = jax.block_until_ready(diag)
    elapsed_s = time.perf_counter() - start_time
    step_count = int(diag.step)
    if step_count > 0:
        total_s = elapsed_s * float(params.nsteps) / float(step_count)
        remaining_s = max(0.0, total_s - elapsed_s)
    else:
        total_s = 0.0
        remaining_s = 0.0
    return diag._replace(elapsed_s=elapsed_s, remaining_s=remaining_s, total_s=total_s)


def run(
    params: Params,
    seed: int = 0,
    log_every: int | None = None,
    log_callback: Callable[[Diagnostics], None] | None = None,
    log_state_callback: Callable[[FlowState, Diagnostics], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[FlowState, list[Diagnostics]]:
    ops = make_operators(params)
    state = initial_state(params, seed)
    log_every = params.c_count if log_every is None else log_every

    step_fn: Callable[[FlowState], FlowState]
    step_update_fn: Callable[[FlowState], FlowState] | None = None
    if params.use_jit and params.time_scheme == "ab2" and params.sgs_model == "lasd":
        skip_jit = jax.jit(lambda s: step_ab2_lasd_skip(s, params, ops))
        update_jit = jax.jit(lambda s: step_ab2_lasd_update(s, params, ops))
        compile_start = time.perf_counter()
        if status_callback is not None:
            status_callback(
                f"[precompile] lowering ab2 LASD skip/update step kernels "
                f"for {params.nx}x{params.ny}x{params.nz}"
            )
        skip_lowered = skip_jit.lower(state)
        update_lowered = update_jit.lower(state)
        if status_callback is not None:
            status_callback(
                f"[precompile] compiling skip/update step kernels "
                f"(lowered in {time.perf_counter() - compile_start:.1f}s)"
            )
        step_fn = skip_lowered.compile()
        step_update_fn = update_lowered.compile()
        if status_callback is not None:
            status_callback(f"[precompile] done in {time.perf_counter() - compile_start:.1f}s")
    elif params.use_jit:
        step_jit = jax.jit(lambda s: step(s, params, ops))
        compile_start = time.perf_counter()
        if status_callback is not None:
            status_callback(
                f"[precompile] lowering {params.time_scheme} step kernel "
                f"for {params.nx}x{params.ny}x{params.nz} "
                f"(projection_mode={params.projection_mode})"
            )
        lowered = step_jit.lower(state)
        if status_callback is not None:
            status_callback(
                f"[precompile] compiling step kernel "
                f"(lowered in {time.perf_counter() - compile_start:.1f}s)"
            )
        step_fn = lowered.compile()
        if status_callback is not None:
            status_callback(f"[precompile] done in {time.perf_counter() - compile_start:.1f}s")
    else:
        step_fn = lambda s: step(s, params, ops)

    first_diag = jax.block_until_ready(diagnostics(state, params, ops))._replace(
        elapsed_s=0.0,
        remaining_s=0.0,
        total_s=0.0,
    )
    validate_cfl(first_diag)
    validate_lasd_cfl(first_diag, params)
    logs: list[Diagnostics] = [first_diag]
    if log_callback is not None:
        log_callback(first_diag)
    if log_state_callback is not None:
        log_state_callback(state, first_diag)

    start_time = time.perf_counter()
    for n in range(params.nsteps):
        lasd_update = ((n + 1) % params.cs_count) == 0
        if lasd_update and (
            params.sgs_model == "lasd"
            or (params.thermo_enabled and params.scalar_sgs_model == "lasd")
        ):
            update_diag = jax.block_until_ready(diagnostics(state, params, ops))
            validate_cfl(update_diag)
            validate_lasd_cfl(update_diag, params)
        if step_update_fn is not None and lasd_update:
            state = step_update_fn(state)
        else:
            state = step_fn(state)
        if (n + 1) % log_every == 0:
            diag = diagnostics(state, params, ops)
            timed_diag = _add_timing(diag, params, start_time)
            validate_cfl(timed_diag)
            logs.append(timed_diag)
            if log_callback is not None:
                log_callback(timed_diag)
            if log_state_callback is not None:
                log_state_callback(state, timed_diag)
    return state, logs
