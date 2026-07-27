from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Params
from .derivative import ddx, ddy, ddz_uv, ddz_w
from .state import Operators


def div_stress(
    tx: jax.Array,
    ty: jax.Array,
    tz: jax.Array,
    params: Params,
    ops: Operators,
    stagger_w: bool,
) -> jax.Array:
    dz_term = ddz_w(tz, params) if stagger_w else ddz_uv(tz, params)
    return ddx(tx, params, ops) + ddy(ty, params, ops) + dz_term


def assemble_rhs(
    c: jax.Array,
    div_t: jax.Array,
    params: Params,
    pressure_force: bool = False,
) -> jax.Array:
    rhs = -c - div_t
    if pressure_force and params.driving_pressure_force != 0.0:
        z = (jnp.arange(params.nz, dtype=params.dtype) + 0.5) * params.dz
        mask = (z <= params.forcing_height).astype(params.dtype)
        rhs = rhs + params.driving_pressure_force * mask[None, None, :]
    return rhs


def add_coriolis_geostrophic_forcing_inner(
    rhs_u: jax.Array,
    rhs_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array]:
    f = params.coriolis_f_internal
    if f == 0.0:
        return rhs_u, rhs_v
    geostrophic_u = params.geostrophic_u
    geostrophic_v = params.geostrophic_v
    rhs_u = rhs_u + f * (v - geostrophic_v)
    rhs_v = rhs_v - f * (u - geostrophic_u)
    return rhs_u.astype(params.dtype), rhs_v.astype(params.dtype)


def add_coriolis_geostrophic_forcing(
    rhs_u: jax.Array,
    rhs_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array]:
    if params.coriolis_f_internal == 0.0:
        return rhs_u, rhs_v
    return add_coriolis_geostrophic_forcing_inner(rhs_u, rhs_v, u, v, params)
