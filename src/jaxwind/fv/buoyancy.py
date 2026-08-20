"""Boussinesq coupling for the staggered finite-volume solver."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp

from .state import StaggeredVelocity


@dataclass(frozen=True, slots=True)
class LinearBoussinesqBuoyancy:
    """Vertical acceleration per transported scalar unit."""

    acceleration_per_scalar: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.acceleration_per_scalar):
            raise ValueError("the buoyancy coefficient must be finite")


def boussinesq_tendency(
    scalar: jnp.ndarray,
    model: LinearBoussinesqBuoyancy,
    *,
    x_face_count: int | None = None,
) -> StaggeredVelocity:
    """Return hydrostatic-free buoyancy on the staggered vertical faces.

    Only horizontal scalar fluctuations accelerate the resolved flow. The
    plane mean is a hydrostatic contribution that pressure balances and is
    removed explicitly, matching the distributed JAX discretisation.
    """

    upper_scalar = 0.5 * (scalar[:-1] + scalar[1:])
    fluctuation = upper_scalar - jnp.mean(
        upper_scalar,
        axis=(-2, -1),
        keepdims=True,
    )
    wall = jnp.zeros_like(scalar[:1])
    vertical = jnp.concatenate(
        (
            wall,
            model.acceleration_per_scalar * fluctuation,
            wall,
        ),
        axis=0,
    )
    x_count = scalar.shape[-1] if x_face_count is None else x_face_count
    x_shape = scalar.shape[:-1] + (x_count,)
    return StaggeredVelocity(
        jnp.zeros(x_shape, scalar.dtype),
        jnp.zeros_like(scalar),
        vertical,
    )


__all__ = ["LinearBoussinesqBuoyancy", "boussinesq_tendency"]
