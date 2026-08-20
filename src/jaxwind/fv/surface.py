"""Coupled Monin--Obukhov surface exchange for finite volumes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax.numpy as jnp

from jaxwind.domain.grid import UniformGrid

from .operators import cell_velocity
from .state import StaggeredVelocity


@dataclass(frozen=True, slots=True)
class MoninObukhovSurface:
    """Businger--Dyer exchange against an evolving surface scalar."""

    momentum_roughness: float
    scalar_roughness: float
    surface_scalar_initial: float
    surface_scalar_rate: float = 0.0
    x_velocity_offset: float = 0.0
    y_velocity_offset: float = 0.0
    buoyancy_coefficient: float = 0.0
    von_karman: float = 0.4
    positive_zeta_momentum_slope: float = 4.8
    positive_zeta_scalar_slope: float = 7.8
    negative_zeta_momentum_coefficient: float = 16.0
    negative_zeta_scalar_coefficient: float = 16.0
    iterations: int = 12
    relaxation: float = 0.5
    maximum_abs_zeta: float = 10.0

    def __post_init__(self) -> None:
        positive = (
            self.momentum_roughness,
            self.scalar_roughness,
            self.von_karman,
            self.positive_zeta_momentum_slope,
            self.positive_zeta_scalar_slope,
            self.negative_zeta_momentum_coefficient,
            self.negative_zeta_scalar_coefficient,
            self.maximum_abs_zeta,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("surface constants must be finite and positive")
        finite = (
            self.surface_scalar_initial,
            self.surface_scalar_rate,
            self.x_velocity_offset,
            self.y_velocity_offset,
            self.buoyancy_coefficient,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("surface evolution and coupling must be finite")
        if self.iterations <= 0:
            raise ValueError("surface iterations must be positive")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("surface relaxation must lie in (0, 1]")


class SurfaceExchange(NamedTuple):
    """Plane-mean exchange, with stress directed into the fluid."""

    stress_x: jnp.ndarray
    stress_y: jnp.ndarray
    scalar_flux: jnp.ndarray
    friction_velocity: jnp.ndarray
    scalar_scale: jnp.ndarray
    obukhov_length: jnp.ndarray
    surface_scalar: jnp.ndarray


def coupled_surface_exchange(
    velocity: StaggeredVelocity,
    scalar: jnp.ndarray,
    time: jnp.ndarray,
    grid: UniformGrid,
    model: MoninObukhovSurface,
) -> SurfaceExchange:
    """Evaluate stability-dependent plane-mean lower-boundary exchange."""

    u, v, _w = cell_velocity(velocity)
    measurement_height = 0.5 * grid.dz
    mean_u = jnp.mean(u[0]) + model.x_velocity_offset
    mean_v = jnp.mean(v[0]) + model.y_velocity_offset
    mean_scalar = jnp.mean(scalar[0])
    surface_scalar = (
        model.surface_scalar_initial + model.surface_scalar_rate * time
    )
    speed = jnp.hypot(mean_u, mean_v)
    scalar_difference = mean_scalar - surface_scalar
    limit = model.maximum_abs_zeta / measurement_height

    def bounded(zeta):
        return jnp.clip(
            zeta,
            -model.maximum_abs_zeta,
            model.maximum_abs_zeta,
        )

    def momentum_correction(zeta):
        zeta = bounded(zeta)
        positive = -model.positive_zeta_momentum_slope * zeta
        x = jnp.maximum(
            1.0 - model.negative_zeta_momentum_coefficient * zeta,
            1.0,
        ) ** 0.25
        negative = (
            2.0 * jnp.log(0.5 * (1.0 + x))
            + jnp.log(0.5 * (1.0 + x * x))
            - 2.0 * jnp.arctan(x)
            + 0.5 * jnp.pi
        )
        return jnp.where(zeta >= 0.0, positive, negative)

    def scalar_correction(zeta):
        zeta = bounded(zeta)
        positive = -model.positive_zeta_scalar_slope * zeta
        x = jnp.maximum(
            1.0 - model.negative_zeta_scalar_coefficient * zeta,
            1.0,
        ) ** 0.5
        negative = 2.0 * jnp.log(0.5 * (1.0 + x))
        return jnp.where(zeta >= 0.0, positive, negative)

    def denominator(roughness, inverse_obukhov, correction):
        return (
            jnp.log(measurement_height / roughness)
            - correction(measurement_height * inverse_obukhov)
            + correction(roughness * inverse_obukhov)
        )

    momentum_neutral = jnp.log(measurement_height / model.momentum_roughness)
    scalar_neutral = jnp.log(measurement_height / model.scalar_roughness)
    momentum_slope = model.positive_zeta_momentum_slope * (
        measurement_height - model.momentum_roughness
    )
    scalar_slope = model.positive_zeta_scalar_slope * (
        measurement_height - model.scalar_roughness
    )
    speed_squared = jnp.maximum(speed * speed, 1.0e-12)
    positive_c = (
        model.buoyancy_coefficient
        * jnp.maximum(scalar_difference, 0.0)
        / speed_squared
    )
    quadratic = scalar_slope - positive_c * momentum_slope**2
    linear = scalar_neutral - 2.0 * positive_c * momentum_neutral * momentum_slope
    constant = -positive_c * momentum_neutral**2
    discriminant = jnp.maximum(linear**2 - 4.0 * quadratic * constant, 0.0)
    positive_denominator = linear + jnp.sqrt(discriminant)
    positive_inverse_obukhov = jnp.where(
        positive_denominator > 1.0e-12,
        -2.0 * constant / positive_denominator,
        0.0,
    )
    positive_inverse_obukhov = jnp.clip(
        positive_inverse_obukhov,
        0.0,
        limit,
    )

    inverse_obukhov = jnp.zeros_like(speed)
    for _ in range(model.iterations):
        momentum_denominator = jnp.maximum(
            denominator(
                model.momentum_roughness,
                inverse_obukhov,
                momentum_correction,
            ),
            1.0e-6,
        )
        scalar_denominator = jnp.maximum(
            denominator(
                model.scalar_roughness,
                inverse_obukhov,
                scalar_correction,
            ),
            1.0e-6,
        )
        friction_velocity = model.von_karman * speed / momentum_denominator
        scalar_scale = (
            model.von_karman * scalar_difference / scalar_denominator
        )
        candidate = (
            model.von_karman
            * model.buoyancy_coefficient
            * scalar_scale
            / jnp.maximum(friction_velocity**2, 1.0e-12)
        )
        candidate = jnp.clip(candidate, -limit, limit)
        inverse_obukhov = (
            (1.0 - model.relaxation) * inverse_obukhov
            + model.relaxation * candidate
        )
    inverse_obukhov = jnp.where(
        scalar_difference >= 0.0,
        positive_inverse_obukhov,
        inverse_obukhov,
    )
    momentum_denominator = jnp.maximum(
        denominator(
            model.momentum_roughness,
            inverse_obukhov,
            momentum_correction,
        ),
        1.0e-6,
    )
    scalar_denominator = jnp.maximum(
        denominator(
            model.scalar_roughness,
            inverse_obukhov,
            scalar_correction,
        ),
        1.0e-6,
    )
    friction_velocity = model.von_karman * speed / momentum_denominator
    scalar_scale = model.von_karman * scalar_difference / scalar_denominator
    stress = friction_velocity**2
    direction = jnp.where(speed > 0.0, speed, 1.0)
    scalar_flux = -friction_velocity * scalar_scale
    obukhov_length = jnp.where(
        jnp.abs(inverse_obukhov) > 1.0e-12,
        1.0 / inverse_obukhov,
        jnp.inf,
    )
    return SurfaceExchange(
        stress * mean_u / direction,
        stress * mean_v / direction,
        scalar_flux,
        friction_velocity,
        scalar_scale,
        obukhov_length,
        surface_scalar,
    )


def surface_momentum_tendency(
    velocity: StaggeredVelocity,
    exchange: SurfaceExchange,
    grid: UniformGrid,
) -> StaggeredVelocity:
    """Apply plane-mean coupled stress to the first momentum cells."""

    x = jnp.zeros_like(velocity.x).at[0].set(-exchange.stress_x / grid.dz)
    y = jnp.zeros_like(velocity.y).at[0].set(-exchange.stress_y / grid.dz)
    return StaggeredVelocity(x, y, jnp.zeros_like(velocity.z))


__all__ = [
    "MoninObukhovSurface",
    "SurfaceExchange",
    "coupled_surface_exchange",
    "surface_momentum_tendency",
]
