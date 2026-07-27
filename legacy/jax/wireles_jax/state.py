from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class FlowState(NamedTuple):
    u: jax.Array
    v: jax.Array
    w: jax.Array
    p: jax.Array
    theta: jax.Array
    qv: jax.Array
    rhs_u_prev: jax.Array
    rhs_v_prev: jax.Array
    rhs_w_prev: jax.Array
    rhs_theta_prev: jax.Array
    rhs_qv_prev: jax.Array
    lm_old: jax.Array
    mm_old: jax.Array
    qn_old: jax.Array
    nn_old: jax.Array
    cs2: jax.Array
    scalar_c: jax.Array
    scalar_lm_old: jax.Array
    scalar_mm_old: jax.Array
    scalar_qn_old: jax.Array
    scalar_nn_old: jax.Array
    u_lag: jax.Array
    v_lag: jax.Array
    w_lag: jax.Array
    step: jax.Array


class Operators(NamedTuple):
    kx: jax.Array
    ky: jax.Array
    kx_rfft: jax.Array
    ky_rfft: jax.Array
    horizontal_cutoff_rfft: jax.Array
    pressure_a: jax.Array
    pressure_b: jax.Array
    pressure_c: jax.Array
    pressure_inv_bet: jax.Array
    pressure_gam: jax.Array
    pressure_mode_keep: jax.Array


class Diagnostics(NamedTuple):
    step: jax.Array
    ustar: jax.Array
    ke_max: jax.Array
    div_max: jax.Array
    cfl_x: jax.Array
    cfl_y: jax.Array
    cfl_z: jax.Array
    theta_v_min: jax.Array | float = 0.0
    qv_min: jax.Array | float = 0.0
    qv_floor_hits: jax.Array | float = 0.0
    elapsed_s: jax.Array | float = 0.0
    remaining_s: jax.Array | float = 0.0
    total_s: jax.Array | float = 0.0


def zeros_like_velocity(u: jax.Array) -> jax.Array:
    return jnp.zeros_like(u)
