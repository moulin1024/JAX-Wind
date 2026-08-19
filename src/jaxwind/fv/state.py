"""Staggered (MAC) velocity storage and Cartesian wall boundary conditions.

The layout is z-first to match the rest of the code base:

* ``u`` lives on x-faces, shape ``(nz, ny, nx)``; index ``i`` is the face at
  ``x = i * dx``, between cells ``i - 1`` and ``i`` (periodic in x).
* ``v`` lives on y-faces, shape ``(nz, ny, nx)``; index ``j`` is the face at
  ``y = j * dy`` (periodic in y).
* ``w`` lives on z-faces, shape ``(nz + 1, ny, nx)``; levels ``0`` and ``nz``
  are the physical walls and are always zero.
* pressure lives at cell centres, shape ``(nz, ny, nx)``.

This is the arrangement for which the discrete divergence and the discrete
gradient are exact adjoints, so ``D G`` is the compact seven-point Laplacian
and the projection removes divergence to round-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from jaxwind.domain.grid import UniformGrid


NO_SLIP = "no-slip"
FREE_SLIP = "free-slip"


class StaggeredVelocity(NamedTuple):
    """Face-normal velocity components on the MAC arrangement."""

    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray


@dataclass(frozen=True, slots=True)
class Wall:
    """One z-wall: impermeable, with a tangential velocity condition."""

    kind: str = NO_SLIP
    x_velocity: float = 0.0
    y_velocity: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in (NO_SLIP, FREE_SLIP):
            raise ValueError(f"unsupported wall kind: {self.kind!r}")
        if self.kind == FREE_SLIP and (self.x_velocity or self.y_velocity):
            raise ValueError("a free-slip wall cannot impose a tangential velocity")


@dataclass(frozen=True, slots=True)
class Boundaries:
    """Periodic in x and y, impermeable walls in z."""

    lower: Wall = Wall()
    upper: Wall = Wall()


def cell_shape(grid: UniformGrid) -> tuple[int, int, int]:
    return (grid.nz, grid.ny, grid.nx)


def z_face_shape(grid: UniformGrid) -> tuple[int, int, int]:
    return (grid.nz + 1, grid.ny, grid.nx)


def zeros(grid: UniformGrid, dtype: str = "float64") -> StaggeredVelocity:
    """Return a velocity field at rest."""
    resolved = jnp.zeros((), dtype=jnp.dtype(dtype)).dtype
    return StaggeredVelocity(
        jnp.zeros(cell_shape(grid), resolved),
        jnp.zeros(cell_shape(grid), resolved),
        jnp.zeros(z_face_shape(grid), resolved),
    )


def validate(velocity: StaggeredVelocity, grid: UniformGrid) -> None:
    """Raise when a velocity does not match the staggered layout."""
    if velocity.x.shape != cell_shape(grid):
        raise ValueError(f"u must have shape {cell_shape(grid)}")
    if velocity.y.shape != cell_shape(grid):
        raise ValueError(f"v must have shape {cell_shape(grid)}")
    if velocity.z.shape != z_face_shape(grid):
        raise ValueError(f"w must have shape {z_face_shape(grid)}")


def enforce_impermeability(velocity: StaggeredVelocity) -> StaggeredVelocity:
    """Zero the normal velocity on both walls."""
    z_velocity = velocity.z.at[0].set(0.0).at[-1].set(0.0)
    return StaggeredVelocity(velocity.x, velocity.y, z_velocity)


def face_coordinates(grid: UniformGrid) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the 1-D x-face, y-face and z-face coordinates."""
    return (
        jnp.arange(grid.nx) * grid.dx,
        jnp.arange(grid.ny) * grid.dy,
        jnp.arange(grid.nz + 1) * grid.dz,
    )


def cell_coordinates(grid: UniformGrid) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the 1-D cell-centre coordinates."""
    return (
        (jnp.arange(grid.nx) + 0.5) * grid.dx,
        (jnp.arange(grid.ny) + 0.5) * grid.dy,
        (jnp.arange(grid.nz) + 0.5) * grid.dz,
    )


__all__ = [
    "FREE_SLIP",
    "NO_SLIP",
    "Boundaries",
    "StaggeredVelocity",
    "Wall",
    "cell_coordinates",
    "cell_shape",
    "enforce_impermeability",
    "face_coordinates",
    "validate",
    "z_face_shape",
    "zeros",
]
