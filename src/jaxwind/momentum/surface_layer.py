"""Surface-layer wall laws independent of the momentum discretization.

The returned momentum stress is positive into the fluid and the scalar flux is
positive upward.  Keeping these closures separate from the face-flux operator
lets neutral and thermally stratified cases use exactly the same conservative
wall coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class SurfaceLayerFluxes(NamedTuple):
    """Local lower-boundary fluxes and their diagnostic scales."""

    momentum_stress: Array
    heat_flux: Array
    friction_velocity: Array
    temperature_scale: Array
    obukhov_length: Array


def _validate_common(
    roughness_length: float,
    von_karman: float,
    displacement_height: float,
) -> None:
    if not math.isfinite(roughness_length) or roughness_length <= 0.0:
        raise ValueError("roughness length must be finite and positive")
    if not math.isfinite(von_karman) or von_karman <= 0.0:
        raise ValueError("von Karman constant must be finite and positive")
    if not math.isfinite(displacement_height) or displacement_height < 0.0:
        raise ValueError("displacement height must be finite and nonnegative")


def _stress_from_scale(horizontal_velocity: Array, ustar: Array) -> Array:
    speed = jnp.linalg.norm(horizontal_velocity, axis=-1)
    direction = horizontal_velocity / jnp.where(
        speed > 0.0,
        speed,
        1.0,
    )[..., None]
    return (ustar * ustar)[..., None] * direction


@dataclass(frozen=True, slots=True)
class NeutralLogWallLaw:
    """Rough-wall logarithmic law, the neutral ``L -> infinity`` limit."""

    roughness_length: float
    von_karman: float = 0.4
    displacement_height: float = 0.0

    def __post_init__(self) -> None:
        _validate_common(
            self.roughness_length,
            self.von_karman,
            self.displacement_height,
        )

    def _log_denominator(self, matching_height: float) -> float:
        effective_height = matching_height - self.displacement_height
        if effective_height <= self.roughness_length:
            raise ValueError(
                "matching height must exceed displacement plus roughness"
            )
        return math.log(effective_height / self.roughness_length)

    def friction_velocity(
        self,
        horizontal_velocity: Array,
        matching_height: float,
    ) -> Array:
        speed = jnp.linalg.norm(horizontal_velocity, axis=-1)
        return self.von_karman * speed / self._log_denominator(matching_height)

    def surface_fluxes(
        self,
        horizontal_velocity: Array,
        matching_height: float,
    ) -> SurfaceLayerFluxes:
        ustar = self.friction_velocity(horizontal_velocity, matching_height)
        zeros = jnp.zeros_like(ustar)
        return SurfaceLayerFluxes(
            _stress_from_scale(horizontal_velocity, ustar),
            zeros,
            ustar,
            zeros,
            jnp.full_like(ustar, jnp.inf),
        )


@dataclass(frozen=True, slots=True)
class MoninObukhovWallLaw:
    """Coupled Businger--Dyer MOST law for prescribed surface temperature.

    ``surface_fluxes`` solves the local implicit dependence of the momentum and
    heat fluxes on the Obukhov length.  Stable linear corrections and unstable
    Businger--Dyer corrections are used.  A zero temperature difference
    recovers :class:`NeutralLogWallLaw` to roundoff.
    """

    momentum_roughness_length: float
    thermal_roughness_length: float
    reference_potential_temperature: float
    von_karman: float = 0.4
    gravity: float = 9.81
    displacement_height: float = 0.0
    stable_momentum_beta: float = 4.8
    stable_heat_beta: float = 7.8
    unstable_momentum_gamma: float = 16.0
    unstable_heat_gamma: float = 16.0
    iterations: int = 12
    relaxation: float = 0.5
    maximum_abs_zeta: float = 10.0

    def __post_init__(self) -> None:
        _validate_common(
            self.momentum_roughness_length,
            self.von_karman,
            self.displacement_height,
        )
        if (
            not math.isfinite(self.thermal_roughness_length)
            or self.thermal_roughness_length <= 0.0
        ):
            raise ValueError("thermal roughness length must be finite and positive")
        positive = (
            self.reference_potential_temperature,
            self.gravity,
            self.stable_momentum_beta,
            self.stable_heat_beta,
            self.unstable_momentum_gamma,
            self.unstable_heat_gamma,
            self.maximum_abs_zeta,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("MOST physical and similarity constants must be positive")
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations <= 0
        ):
            raise ValueError("MOST iteration count must be a positive integer")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("MOST relaxation must lie in (0, 1]")

    def _validate_height(self, matching_height: float) -> float:
        effective_height = matching_height - self.displacement_height
        if effective_height <= max(
            self.momentum_roughness_length,
            self.thermal_roughness_length,
        ):
            raise ValueError(
                "matching height must exceed displacement plus both roughnesses"
            )
        return effective_height

    def _bounded_zeta(self, zeta: Array) -> Array:
        return jnp.clip(
            zeta,
            -self.maximum_abs_zeta,
            self.maximum_abs_zeta,
        )

    def momentum_stability_correction(self, zeta: Array) -> Array:
        zeta = self._bounded_zeta(zeta)
        stable = -self.stable_momentum_beta * zeta
        x = jnp.maximum(
            1.0 - self.unstable_momentum_gamma * zeta,
            1.0,
        ) ** 0.25
        unstable = (
            2.0 * jnp.log(0.5 * (1.0 + x))
            + jnp.log(0.5 * (1.0 + x * x))
            - 2.0 * jnp.arctan(x)
            + 0.5 * jnp.pi
        )
        return jnp.where(zeta >= 0.0, stable, unstable)

    def heat_stability_correction(self, zeta: Array) -> Array:
        zeta = self._bounded_zeta(zeta)
        stable = -self.stable_heat_beta * zeta
        x = jnp.maximum(1.0 - self.unstable_heat_gamma * zeta, 1.0) ** 0.5
        unstable = 2.0 * jnp.log(0.5 * (1.0 + x))
        return jnp.where(zeta >= 0.0, stable, unstable)

    def _stable_inverse_obukhov(
        self,
        speed: Array,
        temperature_difference: Array,
        height: float,
        neutral_momentum_log: float,
        neutral_heat_log: float,
    ) -> tuple[Array, Array]:
        """Return the physical closed-form stable MOST root and its validity.

        With the linear stable corrections, ``x = 1 / L`` obeys a quadratic:

        ``x (Ah + Bh x) = C (Am + Bm x)^2``.

        The cancellation-free expression below selects the root connected to
        the neutral limit.  Cases outside that branch (including clipped zeta
        or vanishing wind) are marked invalid and retain the iterative path.
        """
        speed_squared = speed * speed
        bulk_coefficient = (
            self.gravity
            * temperature_difference
            / (
                jnp.maximum(speed_squared, 1.0e-12)
                * self.reference_potential_temperature
            )
        )
        momentum_slope = self.stable_momentum_beta * (
            height - self.momentum_roughness_length
        )
        heat_slope = self.stable_heat_beta * (
            height - self.thermal_roughness_length
        )
        quadratic = heat_slope - bulk_coefficient * momentum_slope**2
        linear = (
            neutral_heat_log
            - 2.0
            * bulk_coefficient
            * neutral_momentum_log
            * momentum_slope
        )
        constant = -bulk_coefficient * neutral_momentum_log**2
        discriminant = linear * linear - 4.0 * quadratic * constant
        square_root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        inverse_obukhov = -2.0 * constant / jnp.maximum(
            linear + square_root,
            1.0e-12,
        )
        valid = (
            (temperature_difference >= 0.0)
            & (speed_squared > 1.0e-12)
            & (quadratic > 0.0)
            & (linear > 0.0)
            & (discriminant >= 0.0)
            & jnp.isfinite(inverse_obukhov)
            & (inverse_obukhov >= 0.0)
            & (height * inverse_obukhov <= self.maximum_abs_zeta)
        )
        return inverse_obukhov, valid

    def _iterative_inverse_obukhov(
        self,
        speed: Array,
        temperature_difference: Array,
        height: float,
        neutral_momentum_log: float,
        neutral_heat_log: float,
    ) -> Array:
        """Solve the general stable/unstable MOST relation by relaxation."""
        inverse_obukhov = jnp.zeros_like(speed)
        for _ in range(self.iterations):
            momentum_denominator = (
                neutral_momentum_log
                - self.momentum_stability_correction(height * inverse_obukhov)
                + self.momentum_stability_correction(
                    self.momentum_roughness_length * inverse_obukhov
                )
            )
            heat_denominator = (
                neutral_heat_log
                - self.heat_stability_correction(height * inverse_obukhov)
                + self.heat_stability_correction(
                    self.thermal_roughness_length * inverse_obukhov
                )
            )
            momentum_denominator = jnp.maximum(momentum_denominator, 1.0e-6)
            heat_denominator = jnp.maximum(heat_denominator, 1.0e-6)
            ustar = self.von_karman * speed / momentum_denominator
            temperature_scale = (
                self.von_karman * temperature_difference / heat_denominator
            )
            candidate = (
                self.von_karman
                * self.gravity
                * temperature_scale
                / (
                    jnp.maximum(ustar * ustar, 1.0e-12)
                    * self.reference_potential_temperature
                )
            )
            inverse_obukhov = (
                (1.0 - self.relaxation) * inverse_obukhov
                + self.relaxation * candidate
            )
        return inverse_obukhov

    def surface_fluxes(
        self,
        horizontal_velocity: Array,
        potential_temperature: Array,
        surface_potential_temperature: Array | float,
        matching_height: float,
    ) -> SurfaceLayerFluxes:
        """Return local stress and kinematic heat flux at the lower boundary."""
        height = self._validate_height(matching_height)
        speed = jnp.linalg.norm(horizontal_velocity, axis=-1)
        temperature = jnp.asarray(
            potential_temperature,
            dtype=horizontal_velocity.dtype,
        )
        surface_temperature = jnp.asarray(
            surface_potential_temperature,
            dtype=horizontal_velocity.dtype,
        )
        temperature_difference = temperature - surface_temperature
        neutral_momentum_log = math.log(
            height / self.momentum_roughness_length
        )
        neutral_heat_log = math.log(height / self.thermal_roughness_length)
        stable_inverse_obukhov, stable_valid = self._stable_inverse_obukhov(
            speed,
            temperature_difference,
            height,
            neutral_momentum_log,
            neutral_heat_log,
        )

        def closed_form(_: None) -> tuple[Array, Array, Array]:
            momentum_denominator = (
                neutral_momentum_log
                + self.stable_momentum_beta
                * (height - self.momentum_roughness_length)
                * stable_inverse_obukhov
            )
            heat_denominator = (
                neutral_heat_log
                + self.stable_heat_beta
                * (height - self.thermal_roughness_length)
                * stable_inverse_obukhov
            )
            return (
                stable_inverse_obukhov,
                momentum_denominator,
                heat_denominator,
            )

        def iterative(_: None) -> tuple[Array, Array, Array]:
            inverse_obukhov = self._iterative_inverse_obukhov(
                speed,
                temperature_difference,
                height,
                neutral_momentum_log,
                neutral_heat_log,
            )
            momentum_denominator = (
                neutral_momentum_log
                - self.momentum_stability_correction(height * inverse_obukhov)
                + self.momentum_stability_correction(
                    self.momentum_roughness_length * inverse_obukhov
                )
            )
            heat_denominator = (
                neutral_heat_log
                - self.heat_stability_correction(height * inverse_obukhov)
                + self.heat_stability_correction(
                    self.thermal_roughness_length * inverse_obukhov
                )
            )
            return inverse_obukhov, momentum_denominator, heat_denominator

        inverse_obukhov, momentum_denominator, heat_denominator = jax.lax.cond(
            jnp.all(stable_valid),
            closed_form,
            iterative,
            operand=None,
        )
        ustar = self.von_karman * speed / jnp.maximum(
            momentum_denominator,
            1.0e-6,
        )
        temperature_scale = (
            self.von_karman
            * temperature_difference
            / jnp.maximum(heat_denominator, 1.0e-6)
        )
        heat_flux = -ustar * temperature_scale
        obukhov_length = jnp.where(
            jnp.abs(inverse_obukhov) > 1.0e-12,
            1.0 / inverse_obukhov,
            jnp.inf,
        )
        return SurfaceLayerFluxes(
            _stress_from_scale(horizontal_velocity, ustar),
            heat_flux,
            ustar,
            temperature_scale,
            obukhov_length,
        )


__all__ = [
    "MoninObukhovWallLaw",
    "NeutralLogWallLaw",
    "SurfaceLayerFluxes",
]
