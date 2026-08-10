"""Nieuwstadt et al. (1993) benchmark-specific initial fields."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from wireles_jax.config import Params
from wireles_jax.init import apply_velocity_bc, initial_state
from wireles_jax.state import FlowState


def initial_benchmark_state(
    params: Params,
    *,
    seed: int = 0,
    initial_zi_fraction: float = 0.844,
) -> FlowState:
    if not 0.0 < initial_zi_fraction < 1.0:
        raise ValueError(
            "initial_zi_fraction must lie between zero and one, "
            f"got {initial_zi_fraction:.6e}"
        )

    state = initial_state(params, seed=seed)
    key = jax.random.split(jax.random.PRNGKey(seed), 4)[3]
    z_center = (
        jnp.arange(params.nz, dtype=params.dtype) + 0.5
    ) * params.dz * params.z_i
    zi1 = jnp.asarray(initial_zi_fraction * params.z_i, dtype=params.dtype)
    gamma = jnp.asarray(params.theta_initial_gradient, dtype=params.dtype)
    wstar0 = (
        (params.g / params.theta0)
        * params.surface_theta_flux
        * params.z_i
    ) ** (1.0 / 3.0)
    theta_star0 = params.surface_theta_flux / wstar0

    random_theta = jax.random.uniform(
        key,
        (params.nx, params.ny, params.nz),
        minval=-0.5,
        maxval=0.5,
        dtype=params.dtype,
    )
    lower_weight = jnp.maximum(1.0 - z_center / zi1, 0.0)
    theta = jnp.where(
        (z_center < zi1)[None, None, :],
        params.theta0
        + 0.1
        * random_theta
        * lower_weight[None, None, :]
        * theta_star0,
        params.theta0 + (z_center - zi1)[None, None, :] * gamma,
    ).astype(params.dtype)

    z_face = (
        jnp.arange(params.nz, dtype=params.dtype) + 1.0
    ) * params.dz * params.z_i
    face_weight = jnp.maximum(1.0 - z_face / zi1, 0.0)
    w = jnp.where(
        (z_face < zi1)[None, None, :],
        0.1 * random_theta * face_weight[None, None, :] * wstar0,
        0.0,
    ).astype(params.dtype)
    u = jnp.zeros_like(state.u)
    v = jnp.zeros_like(state.v)
    u, v, w = apply_velocity_bc(u, v, w, params)
    return state._replace(u=u, v=v, w=w, theta=theta)
