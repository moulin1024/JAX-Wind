"""Compact Businger--Dyer MOST closure used by the GABLS1 runner."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax.numpy as jnp


class SurfaceFluxes(NamedTuple):
    """Fluxes use the physical convention: stress into fluid, heat upward."""

    stress_x: object
    stress_y: object
    heat_flux: object
    friction_velocity: object
    temperature_scale: object
    obukhov_length: object


@dataclass(frozen=True, slots=True)
class MoninObukhovWallLaw:
    momentum_roughness_length: float
    thermal_roughness_length: float
    reference_temperature: float
    gravity: float = 9.81
    von_karman: float = 0.4
    stable_momentum_beta: float = 4.8
    stable_heat_beta: float = 7.8
    unstable_momentum_gamma: float = 16.0
    unstable_heat_gamma: float = 16.0
    iterations: int = 12
    relaxation: float = 0.5
    maximum_abs_zeta: float = 10.0

    def __post_init__(self) -> None:
        positive = (
            self.momentum_roughness_length,
            self.thermal_roughness_length,
            self.reference_temperature,
            self.gravity,
            self.von_karman,
            self.stable_momentum_beta,
            self.stable_heat_beta,
            self.unstable_momentum_gamma,
            self.unstable_heat_gamma,
            self.maximum_abs_zeta,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("MOST constants must be finite and positive")
        if self.iterations <= 0:
            raise ValueError("MOST iterations must be positive")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("MOST relaxation must lie in (0, 1]")

    def _bounded_zeta(self, zeta):
        return jnp.clip(zeta, -self.maximum_abs_zeta, self.maximum_abs_zeta)

    def momentum_correction(self, zeta):
        zeta = self._bounded_zeta(zeta)
        stable = -self.stable_momentum_beta * zeta
        x = jnp.maximum(1.0 - self.unstable_momentum_gamma * zeta, 1.0) ** 0.25
        unstable = (
            2.0 * jnp.log(0.5 * (1.0 + x))
            + jnp.log(0.5 * (1.0 + x * x))
            - 2.0 * jnp.arctan(x)
            + 0.5 * jnp.pi
        )
        return jnp.where(zeta >= 0.0, stable, unstable)

    def heat_correction(self, zeta):
        zeta = self._bounded_zeta(zeta)
        stable = -self.stable_heat_beta * zeta
        x = jnp.maximum(1.0 - self.unstable_heat_gamma * zeta, 1.0) ** 0.5
        unstable = 2.0 * jnp.log(0.5 * (1.0 + x))
        return jnp.where(zeta >= 0.0, stable, unstable)

    def _denominator(self, height, roughness, inverse_obukhov, correction):
        return (
            jnp.log(height / roughness)
            - correction(height * inverse_obukhov)
            + correction(roughness * inverse_obukhov)
        )

    def surface_fluxes(
        self,
        u,
        v,
        potential_temperature,
        surface_temperature,
        measurement_height: float,
    ) -> SurfaceFluxes:
        """Solve the implicit MOST relation at the first cell centre.

        The stable branch uses the closed-form quadratic implied by the linear
        Businger--Dyer corrections.  This avoids the fixed-point oscillation
        that otherwise occurs when the surface first becomes colder than the
        air in the GABLS1 spin-up.
        """
        if measurement_height <= max(
            self.momentum_roughness_length,
            self.thermal_roughness_length,
        ):
            raise ValueError("MOST measurement height must exceed both roughnesses")
        speed = jnp.hypot(u, v)
        temperature_difference = potential_temperature - surface_temperature

        # With psi_m=-beta_m*z/L and psi_h=-beta_h*z/L, the implicit
        # stable-MOST equation is a quadratic in x=1/L:
        #   x (Ah + Bh*x) = C (Am + Bm*x)^2.
        momentum_neutral = jnp.log(
            measurement_height / self.momentum_roughness_length
        )
        heat_neutral = jnp.log(
            measurement_height / self.thermal_roughness_length
        )
        momentum_slope = self.stable_momentum_beta * (
            measurement_height - self.momentum_roughness_length
        )
        heat_slope = self.stable_heat_beta * (
            measurement_height - self.thermal_roughness_length
        )
        speed_squared = jnp.maximum(speed * speed, 1.0e-12)
        stable_c = (
            self.gravity
            * jnp.maximum(temperature_difference, 0.0)
            / (speed_squared * self.reference_temperature)
        )
        quadratic = heat_slope - stable_c * momentum_slope * momentum_slope
        linear = (
            heat_neutral
            - 2.0 * stable_c * momentum_neutral * momentum_slope
        )
        constant = -stable_c * momentum_neutral * momentum_neutral
        discriminant = jnp.maximum(
            linear * linear - 4.0 * quadratic * constant, 0.0
        )
        stable_denominator = linear + jnp.sqrt(discriminant)
        stable_inverse_obukhov = jnp.where(
            stable_denominator > 1.0e-12,
            -2.0 * constant / stable_denominator,
            0.0,
        )
        stable_inverse_obukhov = jnp.clip(
            stable_inverse_obukhov,
            0.0,
            self.maximum_abs_zeta / measurement_height,
        )

        # Retain the general fixed-point branch for callers that intentionally
        # supply an unstable state.  GABLS1 itself uses the stable branch.
        inverse_obukhov = jnp.zeros_like(speed)
        limit = self.maximum_abs_zeta / measurement_height
        for _ in range(self.iterations):
            momentum_denominator = jnp.maximum(
                self._denominator(
                    measurement_height,
                    self.momentum_roughness_length,
                    inverse_obukhov,
                    self.momentum_correction,
                ),
                1.0e-6,
            )
            heat_denominator = jnp.maximum(
                self._denominator(
                    measurement_height,
                    self.thermal_roughness_length,
                    inverse_obukhov,
                    self.heat_correction,
                ),
                1.0e-6,
            )
            friction_velocity = self.von_karman * speed / momentum_denominator
            temperature_scale = (
                self.von_karman * temperature_difference / heat_denominator
            )
            candidate = (
                self.von_karman
                * self.gravity
                * temperature_scale
                / (
                    jnp.maximum(friction_velocity * friction_velocity, 1.0e-12)
                    * self.reference_temperature
                )
            )
            candidate = jnp.clip(candidate, -limit, limit)
            inverse_obukhov = (
                (1.0 - self.relaxation) * inverse_obukhov
                + self.relaxation * candidate
            )
        inverse_obukhov = jnp.where(
            temperature_difference >= 0.0,
            stable_inverse_obukhov,
            inverse_obukhov,
        )
        momentum_denominator = jnp.maximum(
            self._denominator(
                measurement_height,
                self.momentum_roughness_length,
                inverse_obukhov,
                self.momentum_correction,
            ),
            1.0e-6,
        )
        heat_denominator = jnp.maximum(
            self._denominator(
                measurement_height,
                self.thermal_roughness_length,
                inverse_obukhov,
                self.heat_correction,
            ),
            1.0e-6,
        )
        friction_velocity = self.von_karman * speed / momentum_denominator
        temperature_scale = (
            self.von_karman * temperature_difference / heat_denominator
        )
        direction_denominator = jnp.where(speed > 0.0, speed, 1.0)
        stress = friction_velocity * friction_velocity
        obukhov_length = jnp.where(
            jnp.abs(inverse_obukhov) > 1.0e-12,
            1.0 / inverse_obukhov,
            jnp.inf,
        )
        return SurfaceFluxes(
            stress * u / direction_denominator,
            stress * v / direction_denominator,
            -friction_velocity * temperature_scale,
            friction_velocity,
            temperature_scale,
            obukhov_length,
        )
