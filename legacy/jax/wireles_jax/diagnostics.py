from __future__ import annotations

import math
import warnings

import jax.numpy as jnp

from .config import Params
from .derivative import divergence
from .grid import upper_face_to_center
from .scalar import virtual_potential_temperature
from .state import Diagnostics, FlowState, Operators
from .wall import wall_stress


CFL_LIMIT = 0.1
LASD_CFL_LIMIT = 1.0
_WARNED_CFL_LIMITS: set[str] = set()


def _warn_limit_once(kind: str, message: str) -> None:
    """Emit one runtime warning per CFL condition without flooding long runs."""
    if kind in _WARNED_CFL_LIMITS:
        return
    _WARNED_CFL_LIMITS.add(kind)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def cfl_number(diag: Diagnostics) -> float:
    """Maximum directional advective CFL number."""
    return max(float(diag.cfl_x), float(diag.cfl_y), float(diag.cfl_z))


def validate_cfl(diag: Diagnostics, limit: float = CFL_LIMIT) -> float:
    """Report the overall advective CFL number and warn above ``limit``."""
    value = cfl_number(diag)
    if not math.isfinite(value) or value > limit:
        _warn_limit_once(
            "overall",
            f"CFL limit exceeded at step {int(diag.step)}: "
            f"max(CFL_x, CFL_y, CFL_z) = {value:.6f} > {limit:.6f}. "
            "Continuing because CFL limits are advisory; reduce dt if instability appears.",
        )
    return value


def lasd_cfl_number(diag: Diagnostics, params: Params) -> float:
    """Lagrangian backtracking displacement in grid-cell units."""
    return params.cs_count * cfl_number(diag)


def validate_lasd_cfl(diag: Diagnostics, params: Params) -> float:
    """Report LASD interpolation displacement and warn at one cell or above."""
    active = params.sgs_model == "lasd" or (
        params.thermo_enabled and params.scalar_sgs_model == "lasd"
    )
    value = lasd_cfl_number(diag, params)
    if active and (not math.isfinite(value) or value >= LASD_CFL_LIMIT):
        _warn_limit_once(
            "lasd",
            "LASD Lagrangian CFL limit exceeded at step "
            f"{int(diag.step)}: cs_count * max(CFL) = {value:.6f} >= {LASD_CFL_LIMIT:.1f}. "
            "Continuing because CFL limits are advisory; reduce dt or cs_count if instability appears.",
        )
    return value


def diagnostics(state: FlowState, params: Params, ops: Operators) -> Diagnostics:
    _, _, _, _, ustar = wall_stress(state.u, state.v, params)
    u_i = state.u
    v_i = state.v
    w_i = state.w
    w_uv = upper_face_to_center(state.w)
    ke = jnp.max(u_i * u_i + v_i * v_i + w_uv * w_uv)
    div = divergence(state.u, state.v, state.w, params, ops)
    theta_v = virtual_potential_temperature(state.theta, state.qv, params)
    qv_i = state.qv
    if params.moisture_enabled:
        qv_min = jnp.min(qv_i)
        floor_hits = jnp.sum(qv_i <= (params.qv_floor + 10.0 * jnp.finfo(params.dtype).eps))
    else:
        qv_min = jnp.asarray(0.0, dtype=params.dtype)
        floor_hits = jnp.asarray(0.0, dtype=params.dtype)
    zero_time = jnp.asarray(0.0, dtype=params.dtype)
    return Diagnostics(
        step=state.step,
        ustar=jnp.mean(ustar),
        ke_max=ke,
        div_max=jnp.max(jnp.abs(div)),
        cfl_x=jnp.max(jnp.abs(u_i)) * params.dt / params.dx,
        cfl_y=jnp.max(jnp.abs(v_i)) * params.dt / params.dy,
        cfl_z=jnp.max(jnp.abs(w_i)) * params.dt / params.dz,
        theta_v_min=jnp.min(theta_v),
        qv_min=qv_min,
        qv_floor_hits=floor_hits,
        elapsed_s=zero_time,
        remaining_s=zero_time,
        total_s=zero_time,
    )
