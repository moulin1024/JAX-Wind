"""Coriolis and geostrophic forcing on the staggered FV mesh."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp

from .operators import cell_velocity
from .state import StaggeredVelocity


@dataclass(frozen=True, slots=True)
class CoriolisGeostrophic:
    """Constant traditional and nontraditional Coriolis components."""

    coriolis_parameter: float
    geostrophic_x_velocity: float
    geostrophic_y_velocity: float = 0.0
    horizontal_coriolis_parameter: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.coriolis_parameter,
            self.geostrophic_x_velocity,
            self.geostrophic_y_velocity,
            self.horizontal_coriolis_parameter,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Coriolis--geostrophic parameters must be finite")


def _to_x_faces(field: jnp.ndarray, *, open_x: bool = False) -> jnp.ndarray:
    if open_x:
        interior = 0.5 * (field[..., :-1] + field[..., 1:])
        return jnp.concatenate((field[..., :1], interior, field[..., -1:]), axis=2)
    return 0.5 * (field + jnp.roll(field, 1, axis=2))


def _to_y_faces(field: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (field + jnp.roll(field, 1, axis=1))


def _to_z_faces(field: jnp.ndarray, template: jnp.ndarray) -> jnp.ndarray:
    wall = jnp.zeros_like(template[:1])
    interior = 0.5 * (field[:-1] + field[1:])
    return jnp.concatenate((wall, interior, wall), axis=0)


def coriolis_tendency(
    velocity: StaggeredVelocity,
    model: CoriolisGeostrophic,
) -> StaggeredVelocity:
    """Return an energy-skew Coriolis tendency at each velocity face."""

    u_cell, v_cell, w_cell = cell_velocity(velocity)
    relative_u = u_cell - model.geostrophic_x_velocity
    relative_v = v_cell - model.geostrophic_y_velocity
    vertical = jnp.asarray(model.coriolis_parameter, velocity.x.dtype)
    horizontal = jnp.asarray(
        model.horizontal_coriolis_parameter,
        velocity.x.dtype,
    )
    open_x = velocity.x.shape[-1] == relative_v.shape[-1] + 1
    return StaggeredVelocity(
        vertical * _to_x_faces(relative_v, open_x=open_x)
        - horizontal * _to_x_faces(w_cell, open_x=open_x),
        -vertical * _to_y_faces(relative_u),
        horizontal * _to_z_faces(relative_u, velocity.z),
    )


__all__ = ["CoriolisGeostrophic", "coriolis_tendency"]
