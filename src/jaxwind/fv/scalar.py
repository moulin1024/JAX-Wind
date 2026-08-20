"""Conservative passive-scalar transport for the staggered FV solver."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp

from jaxwind.domain.grid import UniformGrid

from .state import StaggeredVelocity, streamwise_is_periodic


@dataclass(frozen=True, slots=True)
class PassiveScalar:
    """Molecular/SGS diffusivity and prescribed vertical boundary fluxes."""

    diffusivity: float = 0.0
    turbulent_prandtl: float = 1.0
    lower_flux: float = 0.0
    upper_flux: float = 0.0

    def __post_init__(self) -> None:
        values = (self.diffusivity, self.lower_flux, self.upper_flux)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("passive-scalar parameters must be finite")
        if self.diffusivity < 0.0:
            raise ValueError("scalar diffusivity must be nonnegative")
        if not math.isfinite(self.turbulent_prandtl) or self.turbulent_prandtl <= 0.0:
            raise ValueError("the turbulent Prandtl number must be positive")


def scalar_tendency(
    scalar: jnp.ndarray,
    velocity: StaggeredVelocity,
    grid: UniformGrid,
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
    if periodic_x:
        x_scalar = 0.5 * (scalar + jnp.roll(scalar, 1, axis=2))
        x_diffusivity = 0.5 * (
            diffusivity + jnp.roll(diffusivity, 1, axis=2)
        )
        x_gradient = (scalar - jnp.roll(scalar, 1, axis=2)) / grid.dx
    else:
        x_scalar = jnp.concatenate(
            (
                scalar[..., :1],
                0.5 * (scalar[..., :-1] + scalar[..., 1:]),
                scalar[..., -1:],
            ),
            axis=2,
        )
        x_diffusivity = jnp.concatenate(
            (
                diffusivity[..., :1],
                0.5 * (diffusivity[..., :-1] + diffusivity[..., 1:]),
                diffusivity[..., -1:],
            ),
            axis=2,
        )
        zero = jnp.zeros_like(scalar[..., :1])
        x_gradient = jnp.concatenate(
            (zero, (scalar[..., 1:] - scalar[..., :-1]) / grid.dx, zero),
            axis=2,
        )
    y_scalar = 0.5 * (scalar + jnp.roll(scalar, 1, axis=1))
    y_diffusivity = 0.5 * (diffusivity + jnp.roll(diffusivity, 1, axis=1))
    x_flux = velocity.x * x_scalar - x_diffusivity * x_gradient
    y_flux = velocity.y * y_scalar - y_diffusivity * (
        scalar - jnp.roll(scalar, 1, axis=1)
    ) / grid.dy

    interior_diffusivity = 0.5 * (diffusivity[:-1] + diffusivity[1:])
    interior_scalar = 0.5 * (scalar[:-1] + scalar[1:])
    interior_flux = velocity.z[1:-1] * interior_scalar - interior_diffusivity * (
        scalar[1:] - scalar[:-1]
    ) / grid.dz
    lower_value = model.lower_flux if lower_flux is None else lower_flux
    upper_value = model.upper_flux if upper_flux is None else upper_flux
    lower = jnp.full_like(scalar[:1], lower_value)
    upper = jnp.full_like(scalar[:1], upper_value)
    z_flux = jnp.concatenate((lower, interior_flux, upper), axis=0)

    x_divergence = (
        (jnp.roll(x_flux, -1, axis=2) - x_flux) / grid.dx
        if periodic_x
        else (x_flux[..., 1:] - x_flux[..., :-1]) / grid.dx
    )
    return -(
        x_divergence
        + (jnp.roll(y_flux, -1, axis=1) - y_flux) / grid.dy
        + (z_flux[1:] - z_flux[:-1]) / grid.dz
    )


__all__ = ["PassiveScalar", "scalar_tendency"]
