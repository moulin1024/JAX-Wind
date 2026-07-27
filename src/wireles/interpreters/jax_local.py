"""Production local JAX kernels for the first upper-face storage mapping."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def pressure_gradient_upper_faces(
    pressure,
    *,
    dz: float,
    last_upper_gradient: Any,
):
    """Return one stored upper-face gradient per local pressure cell."""
    if pressure.ndim != 3:
        raise ValueError("local pressure must have shape (z, y, x)")
    last = jnp.broadcast_to(
        jnp.asarray(last_upper_gradient, dtype=pressure.dtype),
        pressure.shape[1:],
    )
    interior = (pressure[1:] - pressure[:-1]) / dz
    return jnp.concatenate((interior, last[None, ...]), axis=0)


def divergence_from_upper_faces(
    upper_faces,
    *,
    dz: float,
    lower_face: Any,
):
    """Return one cell divergence per stored upper face."""
    if upper_faces.ndim != 3:
        raise ValueError("local upper faces must have shape (z, y, x)")
    lower = jnp.broadcast_to(
        jnp.asarray(lower_face, dtype=upper_faces.dtype),
        upper_faces.shape[1:],
    )
    lower_faces = jnp.concatenate((lower[None, ...], upper_faces[:-1]), axis=0)
    return (upper_faces - lower_faces) / dz

