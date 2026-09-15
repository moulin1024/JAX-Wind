"""Staggered (MAC) velocity storage and Cartesian wall boundary conditions.

The layout is z-first to match the rest of the code base:

* ``u`` lives on x-faces, shape ``(nz, ny, nx)``; index ``i`` is the face at
  ``x = i * dx``, between cells ``i - 1`` and ``i`` (periodic in x).
* ``v`` lives on y-faces. It has shape ``(nz, ny, nx)`` for periodic y and
  ``(nz, ny + 1, nx)`` when physical side walls are represented.
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

from jaxwind.domain.grid import Grid


NO_SLIP = "no-slip"
FREE_SLIP = "free-slip"
PERIODIC = "periodic"
OPEN = "open"


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
    """Streamwise topology plus impermeable walls in z.

    The spanwise direction can be periodic, free-slip, or OPEN. OPEN gives the streamwise
    velocity a distinct face at each end of the domain, so its final dimension
    is nx + 1 instead of the periodic nx.
    """

    lower: Wall = Wall()
    upper: Wall = Wall()
    streamwise: str = PERIODIC
    spanwise: str = PERIODIC

    def __post_init__(self) -> None:
        if self.streamwise not in (PERIODIC, OPEN):
            raise ValueError(f"unsupported streamwise boundary: {self.streamwise!r}")
        if self.spanwise not in (PERIODIC, FREE_SLIP, OPEN):
            raise ValueError(f"unsupported spanwise boundary: {self.spanwise!r}")


def cell_shape(grid: Grid) -> tuple[int, int, int]:
    return (grid.nz, grid.ny, grid.nx)


def z_face_shape(grid: Grid) -> tuple[int, int, int]:
    return (grid.nz + 1, grid.ny, grid.nx)


def x_face_shape(
    grid: Grid,
    boundaries: Boundaries = Boundaries(),
) -> tuple[int, int, int]:
    count = grid.nx if boundaries.streamwise == PERIODIC else grid.nx + 1
    return (grid.nz, grid.ny, count)


def y_face_shape(
    grid: Grid,
    boundaries: Boundaries = Boundaries(),
) -> tuple[int, int, int]:
    count = grid.ny if boundaries.spanwise == PERIODIC else grid.ny + 1
    return (grid.nz, count, grid.nx)


def zeros(
    grid: Grid,
    dtype: str = "float64",
    boundaries: Boundaries = Boundaries(),
) -> StaggeredVelocity:
    """Return a velocity field at rest."""
    resolved = jnp.zeros((), dtype=jnp.dtype(dtype)).dtype
    return StaggeredVelocity(
        jnp.zeros(x_face_shape(grid, boundaries), resolved),
        jnp.zeros(y_face_shape(grid, boundaries), resolved),
        jnp.zeros(z_face_shape(grid), resolved),
    )


def validate(
    velocity: StaggeredVelocity,
    grid: Grid,
    boundaries: Boundaries = Boundaries(),
) -> None:
    """Raise when a velocity does not match the staggered layout."""
    expected_x = x_face_shape(grid, boundaries)
    if velocity.x.shape != expected_x:
        raise ValueError(f"u must have shape {expected_x}")
    expected_y = y_face_shape(grid, boundaries)
    if velocity.y.shape != expected_y:
        raise ValueError(f"v must have shape {expected_y}")
    if velocity.z.shape != z_face_shape(grid):
        raise ValueError(f"w must have shape {z_face_shape(grid)}")


def streamwise_is_periodic(
    velocity: StaggeredVelocity,
    grid: Grid,
) -> bool:
    """Infer the static streamwise topology from the x-face count."""
    if velocity.x.shape[-1] == grid.nx:
        return True
    if velocity.x.shape[-1] == grid.nx + 1:
        return False
    raise ValueError("u must carry nx periodic faces or nx + 1 open faces")


def spanwise_is_periodic(
    velocity: StaggeredVelocity,
    grid: Grid,
) -> bool:
    """Infer the static spanwise topology from the y-face count."""
    if velocity.y.shape[1] == grid.ny:
        return True
    if velocity.y.shape[1] == grid.ny + 1:
        return False
    raise ValueError("v must carry ny periodic faces or ny + 1 wall faces")


def enforce_impermeability(velocity: StaggeredVelocity, *, open_y=False) -> StaggeredVelocity:
    """Zero normal velocity on z walls and, unless open_y, y side walls."""
    z_velocity = velocity.z.at[0].set(0.0).at[-1].set(0.0)
    y_velocity = velocity.y
    if not open_y and velocity.y.shape[1] != velocity.x.shape[1]:
        y_velocity = y_velocity.at[:, 0].set(0.0).at[:, -1].set(0.0)
    return StaggeredVelocity(velocity.x, y_velocity, z_velocity)


def face_coordinates(grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the 1-D x-face, y-face and z-face coordinates."""
    return (
        jnp.asarray(grid.x_faces[:-1]),
        jnp.asarray(grid.y_faces[:-1]),
        jnp.asarray(grid.z_faces),
    )


def cell_coordinates(grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the 1-D cell-centre coordinates."""
    return (
        jnp.asarray(grid.x_centers),
        jnp.asarray(grid.y_centers),
        jnp.asarray(grid.z_centers),
    )


__all__ = [
    "FREE_SLIP",
    "NO_SLIP",
    "OPEN",
    "PERIODIC",
    "Boundaries",
    "StaggeredVelocity",
    "Wall",
    "cell_coordinates",
    "cell_shape",
    "enforce_impermeability",
    "face_coordinates",
    "spanwise_is_periodic",
    "streamwise_is_periodic",
    "validate",
    "x_face_shape",
    "y_face_shape",
    "z_face_shape",
    "zeros",
]
