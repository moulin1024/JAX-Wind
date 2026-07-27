from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Params
from .grid import center_gradient, divergence_upper_faces, gradient_to_upper_faces
from .state import Operators


def ddx(q: jax.Array, params: Params, ops: Operators) -> jax.Array:
    q_hat = jnp.fft.rfft2(q, axes=(0, 1))
    kx = ops.kx_rfft.astype(q.real.dtype)
    dq = jnp.fft.irfft2(1j * kx * q_hat, s=(params.nx, params.ny), axes=(0, 1))
    return dq.astype(q.dtype)


def ddy(q: jax.Array, params: Params, ops: Operators) -> jax.Array:
    q_hat = jnp.fft.rfft2(q, axes=(0, 1))
    ky = ops.ky_rfft.astype(q.real.dtype)
    dq = jnp.fft.irfft2(1j * ky * q_hat, s=(params.nx, params.ny), axes=(0, 1))
    return dq.astype(q.dtype)


def gradxy(q: jax.Array, params: Params, ops: Operators) -> tuple[jax.Array, jax.Array]:
    q_hat = jnp.fft.rfft2(q, axes=(0, 1))
    kx = ops.kx_rfft.astype(q.real.dtype)
    ky = ops.ky_rfft.astype(q.real.dtype)
    dqdx = jnp.fft.irfft2(1j * kx * q_hat, s=(params.nx, params.ny), axes=(0, 1))
    dqdy = jnp.fft.irfft2(1j * ky * q_hat, s=(params.nx, params.ny), axes=(0, 1))
    return (
        dqdx.astype(q.dtype),
        dqdy.astype(q.dtype),
    )


def gradxy_many(
    qs: tuple[jax.Array, ...],
    params: Params,
    ops: Operators,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    q_inner = jnp.stack(qs, axis=0)
    q_hat = jnp.fft.rfft2(q_inner, axes=(1, 2))
    kx = ops.kx_rfft.astype(q_inner.real.dtype)
    ky = ops.ky_rfft.astype(q_inner.real.dtype)
    dqdx_inner = jnp.fft.irfft2(
        1j * kx[None, :, :, :] * q_hat,
        s=(params.nx, params.ny),
        axes=(1, 2),
    )
    dqdy_inner = jnp.fft.irfft2(
        1j * ky[None, :, :, :] * q_hat,
        s=(params.nx, params.ny),
        axes=(1, 2),
    )
    dqdx = tuple(dqdx_inner[i].astype(qs[i].dtype) for i in range(len(qs)))
    dqdy = tuple(dqdy_inner[i].astype(qs[i].dtype) for i in range(len(qs)))
    return dqdx, dqdy


def horizontal_filter_many(
    qs: tuple[jax.Array, ...],
    params: Params,
    ops: Operators,
) -> tuple[jax.Array, ...]:
    """Apply the configured horizontal cutoff to complete state fields."""
    if not params.horizontal_dealias:
        return qs
    q_inner = jnp.stack(qs, axis=0)
    q_hat = jnp.fft.rfft2(q_inner, axes=(1, 2))
    q_hat = q_hat * ops.horizontal_cutoff_rfft[None, :, :, :].astype(q_hat.dtype)
    q_filtered_inner = jnp.fft.irfft2(
        q_hat,
        s=(params.nx, params.ny),
        axes=(1, 2),
    )
    return tuple(q_filtered_inner[i].astype(qs[i].dtype) for i in range(len(qs)))


def ddxy_filter_many(
    qs: tuple[jax.Array, ...],
    params: Params,
    ops: Operators,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    q_inner = jnp.stack(qs, axis=0)
    q_hat = jnp.fft.rfft2(q_inner, axes=(1, 2))
    if params.horizontal_dealias:
        q_hat = q_hat * ops.horizontal_cutoff_rfft[None, :, :, :].astype(q_hat.dtype)
    kx = ops.kx_rfft.astype(q_inner.real.dtype)
    ky = ops.ky_rfft.astype(q_inner.real.dtype)
    q_filtered_inner = jnp.fft.irfft2(q_hat, s=(params.nx, params.ny), axes=(1, 2))
    dqdx_inner = jnp.fft.irfft2(
        1j * kx[None, :, :, :] * q_hat,
        s=(params.nx, params.ny),
        axes=(1, 2),
    )
    dqdy_inner = jnp.fft.irfft2(
        1j * ky[None, :, :, :] * q_hat,
        s=(params.nx, params.ny),
        axes=(1, 2),
    )
    q_filtered = tuple(q_filtered_inner[i].astype(qs[i].dtype) for i in range(len(qs)))
    dqdx = tuple(dqdx_inner[i].astype(qs[i].dtype) for i in range(len(qs)))
    dqdy = tuple(dqdy_inner[i].astype(qs[i].dtype) for i in range(len(qs)))
    return q_filtered, dqdx, dqdy


def ddz_uv(q: jax.Array, params: Params) -> jax.Array:
    """Cell-centered vertical gradient of a cell-centered field."""
    return center_gradient(q, params.dz).astype(q.dtype)


def ddz_uv_face(
    q: jax.Array,
    params: Params,
    top_gradient: jax.Array | float = 0.0,
) -> jax.Array:
    """Vertical gradient on the owned upper face of every center cell."""
    return gradient_to_upper_faces(q, params.dz, top_gradient).astype(q.dtype)


def ddz_w(q: jax.Array, params: Params) -> jax.Array:
    return divergence_upper_faces(q, params.dz).astype(q.dtype)


def divergence(u: jax.Array, v: jax.Array, w: jax.Array, params: Params, ops: Operators) -> jax.Array:
    return ddx(u, params, ops) + ddy(v, params, ops) + ddz_w(w, params)
