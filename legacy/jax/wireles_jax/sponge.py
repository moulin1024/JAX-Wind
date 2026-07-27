from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Params


def _sponge_decay(z: jax.Array, params: Params, dtype: jnp.dtype) -> jax.Array:
    """Return the exact Rayleigh-relaxation factor for one complete step."""
    top = params.lz * params.z_i
    depth = max(top - params.sponge_start_height, params.dz * params.z_i)
    eta = jnp.clip((z - params.sponge_start_height) / depth, 0.0, 1.0)
    strength_dt = (params.dt_physical / params.sponge_timescale) * eta**params.sponge_power
    return jnp.exp(-strength_dt).astype(dtype)


def apply_rayleigh_sponge(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Damp upper-layer velocity using the C++ solver's exact exponential ramp.

    ``plane_mean`` removes only horizontal perturbations from ``u`` and ``v``;
    this is the appropriate target for a pressure-driven case whose mean wind
    is not prescribed by a geostrophic reference state.  As in the C++ solver,
    ``w`` is relaxed toward zero.
    """
    if not params.sponge_enabled:
        return u, v, w

    center_z = (jnp.arange(params.nz, dtype=u.dtype) + 0.5) * params.dz * params.z_i
    face_z = (jnp.arange(params.nz, dtype=w.dtype) + 1.0) * params.dz * params.z_i
    center_decay = _sponge_decay(center_z, params, u.dtype)[None, None, :]
    face_decay = _sponge_decay(face_z, params, w.dtype)[None, None, :]

    u_inner = u
    v_inner = v
    if params.sponge_target == "plane_mean":
        target_u = jnp.mean(u_inner, axis=(0, 1), keepdims=True)
        target_v = jnp.mean(v_inner, axis=(0, 1), keepdims=True)
    else:
        target_u = jnp.asarray(params.geostrophic_u, dtype=u.dtype)
        target_v = jnp.asarray(params.geostrophic_v, dtype=v.dtype)

    u_inner = target_u + (u_inner - target_u) * center_decay
    v_inner = target_v + (v_inner - target_v) * center_decay.astype(v.dtype)
    if params.sponge_target == "plane_mean":
        # Remove the tiny finite-precision drift left by subtract/multiply/add;
        # otherwise a pressure-driven mean can accumulate a sponge source over
        # many thousands of steps.
        u_inner = u_inner + target_u - jnp.mean(u_inner, axis=(0, 1), keepdims=True)
        v_inner = v_inner + target_v - jnp.mean(v_inner, axis=(0, 1), keepdims=True)
    w_inner = w * face_decay
    return u_inner, v_inner, w_inner
