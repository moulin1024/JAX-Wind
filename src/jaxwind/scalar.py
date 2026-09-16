"""Conservative passive-scalar transport on rectilinear FV meshes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp

from jaxwind.domain.grid import Grid

from .metrics import center_distances, shaped_widths
from jaxwind.numerics.discretization import _cells_to_faces
from .state import StaggeredVelocity, spanwise_is_periodic, streamwise_is_periodic


@dataclass(frozen=True, slots=True)
class PassiveScalar:
    """Molecular/SGS diffusivity and prescribed vertical boundary fluxes."""

    diffusivity: float = 0.0
    turbulent_prandtl: float = 1.0
    lower_flux: float = 0.0
    upper_flux: float = 0.0
    advection_scheme: str = "central"

    def __post_init__(self) -> None:
        values = (self.diffusivity, self.lower_flux, self.upper_flux)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("passive-scalar parameters must be finite")
        if self.diffusivity < 0.0:
            raise ValueError("scalar diffusivity must be nonnegative")
        if not math.isfinite(self.turbulent_prandtl) or self.turbulent_prandtl <= 0.0:
            raise ValueError("the turbulent Prandtl number must be positive")
        if self.advection_scheme not in ("central", "upwind"):
            raise ValueError("scalar advection must be central or upwind")


def scalar_tendency(
    scalar: jnp.ndarray,
    velocity: StaggeredVelocity,
    grid: Grid,
    model: PassiveScalar,
    *,
    eddy_viscosity: jnp.ndarray | float = 0.0,
    lower_flux: jnp.ndarray | float | None = None,
    upper_flux: jnp.ndarray | float | None = None,
) -> jnp.ndarray:
    """Return ``-div(u c - kappa grad(c))`` with prescribed wall fluxes."""
    diffusivity = jnp.asarray(model.diffusivity, scalar.dtype) + (
        jnp.asarray(eddy_viscosity, scalar.dtype) / model.turbulent_prandtl
    )
    diffusivity = jnp.broadcast_to(diffusivity, scalar.shape)
    periodic_x = streamwise_is_periodic(velocity, grid)
    periodic_y = spanwise_is_periodic(velocity, grid)

    if model.advection_scheme == "central":
        x_scalar = _cells_to_faces(
            scalar, grid, 2, periodic=periodic_x, boundary="copy"
        )
        y_scalar = _cells_to_faces(
            scalar, grid, 1, periodic=periodic_y, boundary="copy"
        )
    else:
        if periodic_x:
            x_scalar = jnp.where(
                velocity.x >= 0.0,
                jnp.roll(scalar, 1, axis=2),
                scalar,
            )
        else:
            left = jnp.concatenate((scalar[..., :1], scalar), axis=2)
            right = jnp.concatenate((scalar, scalar[..., -1:]), axis=2)
            x_scalar = jnp.where(velocity.x >= 0.0, left, right)
        if periodic_y:
            y_scalar = jnp.where(
                velocity.y >= 0.0,
                jnp.roll(scalar, 1, axis=1),
                scalar,
            )
        else:
            left = jnp.concatenate((scalar[:, :1], scalar), axis=1)
            right = jnp.concatenate((scalar, scalar[:, -1:]), axis=1)
            y_scalar = jnp.where(velocity.y >= 0.0, left, right)
    x_diffusivity = _cells_to_faces(
        diffusivity, grid, 2, periodic=periodic_x, boundary="copy"
    )
    if periodic_x:
        x_distance = center_distances(
            grid, 2, periodic=True, dtype=scalar.dtype
        )[None, None, :]
        x_gradient = (scalar - jnp.roll(scalar, 1, axis=2)) / x_distance
    else:
        x_distance = center_distances(
            grid, 2, periodic=False, dtype=scalar.dtype
        )
        zero = jnp.zeros_like(scalar[..., :1])
        x_gradient = jnp.concatenate(
            (
                zero,
                (scalar[..., 1:] - scalar[..., :-1])
                / x_distance[None, None, 1:-1],
                zero,
            ),
            axis=2,
        )

    y_diffusivity = _cells_to_faces(
        diffusivity, grid, 1, periodic=periodic_y, boundary="copy"
    )
    if periodic_y:
        y_distance = center_distances(
            grid, 1, periodic=True, dtype=scalar.dtype
        )[None, :, None]
        y_gradient = (scalar - jnp.roll(scalar, 1, axis=1)) / y_distance
    else:
        y_distance = center_distances(
            grid, 1, periodic=False, dtype=scalar.dtype
        )
        side = jnp.zeros_like(scalar[:, :1])
        y_gradient = jnp.concatenate(
            (
                side,
                (scalar[:, 1:] - scalar[:, :-1])
                / y_distance[None, 1:-1, None],
                side,
            ),
            axis=1,
        )

    x_flux = velocity.x * x_scalar - x_diffusivity * x_gradient
    y_flux = velocity.y * y_scalar - y_diffusivity * y_gradient

    z_distance = center_distances(grid, 0, periodic=False, dtype=scalar.dtype)
    interior_diffusivity = _cells_to_faces(
        diffusivity, grid, 0, periodic=False, boundary="copy"
    )[1:-1]
    interior_scalar = (
        _cells_to_faces(
            scalar, grid, 0, periodic=False, boundary="copy"
        )[1:-1]
        if model.advection_scheme == "central"
        else jnp.where(
            velocity.z[1:-1] >= 0.0,
            scalar[:-1],
            scalar[1:],
        )
    )
    interior_flux = velocity.z[1:-1] * interior_scalar - interior_diffusivity * (
        scalar[1:] - scalar[:-1]
    ) / z_distance[1:-1, None, None]
    lower_value = model.lower_flux if lower_flux is None else lower_flux
    upper_value = model.upper_flux if upper_flux is None else upper_flux
    lower = jnp.full_like(scalar[:1], lower_value)
    upper = jnp.full_like(scalar[:1], upper_value)
    z_flux = jnp.concatenate((lower, interior_flux, upper), axis=0)

    x_difference = (
        jnp.roll(x_flux, -1, axis=2) - x_flux
        if periodic_x
        else x_flux[..., 1:] - x_flux[..., :-1]
    )
    y_difference = (
        jnp.roll(y_flux, -1, axis=1) - y_flux
        if periodic_y
        else y_flux[:, 1:] - y_flux[:, :-1]
    )
    return -(
        x_difference / shaped_widths(grid, 2, scalar.dtype)
        + y_difference / shaped_widths(grid, 1, scalar.dtype)
        + (z_flux[1:] - z_flux[:-1])
        / shaped_widths(grid, 0, scalar.dtype)
    )


def open_scalar_tendency(
    scalar, velocity, grid, model, ambient, *, eddy_viscosity=0.0,
    lower_flux=None, upper_flux=None,
):
    """Conservative open-x transport with reservoir inflow and interior outflow.

    Physical end cells evolve normally, including local sources. Advective
    boundary flux uses ambient on incoming flow at either end; diffusive flux
    is zero at the open faces. This is a flux boundary, not a cell-centre
    Dirichlet condition. Lateral/vertical treatment follows scalar_tendency.
    """
    if streamwise_is_periodic(velocity, grid):
        raise ValueError("open scalar flux requires nonperiodic streamwise faces")
    tendency = scalar_tendency(
        scalar, velocity, grid, model, eddy_viscosity=eddy_viscosity,
        lower_flux=lower_flux, upper_flux=upper_flux,
    )
    widths = jnp.asarray(grid.x_widths, scalar.dtype)
    incoming_left = jnp.maximum(velocity.x[..., 0], 0) * (ambient - scalar[..., 0])
    incoming_right = jnp.minimum(velocity.x[..., -1], 0) * (ambient - scalar[..., -1])
    return tendency.at[..., 0].add(incoming_left / widths[0]).at[..., -1].add(
        -incoming_right / widths[-1]
    )


__all__ = ["PassiveScalar", "scalar_tendency", "open_scalar_tendency"]
