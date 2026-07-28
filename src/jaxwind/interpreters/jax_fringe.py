"""Shared JAX construction of smooth concurrent-precursor fringe masks."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def cinf_step(coordinate):
    """Compact C-infinity transition from zero to one on ``0 < x < 1``."""

    dtype = coordinate.dtype
    epsilon = jnp.finfo(dtype).eps
    safe = jnp.clip(coordinate, epsilon, 1.0 - epsilon)
    interior = jax.nn.sigmoid(1.0 / (1.0 - safe) - 1.0 / safe)
    return jnp.where(
        coordinate <= 0.0,
        0.0,
        jnp.where(coordinate >= 1.0, 1.0, interior),
    )


def plateau_fringe_mask(
    x,
    *,
    start_x: float,
    end_x: float,
    rise_width: float,
    fall_width: float,
):
    """Smoothly rise, remain at one, and fall to zero at the periodic seam."""

    dtype = x.dtype
    start = jnp.asarray(start_x, dtype)
    end = jnp.asarray(end_x, dtype)
    rise = jnp.asarray(rise_width, dtype)
    fall = jnp.asarray(fall_width, dtype)
    return cinf_step((x - start) / rise) * cinf_step((end - x) / fall)
