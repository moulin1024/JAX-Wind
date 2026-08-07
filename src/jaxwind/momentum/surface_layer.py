"""Surface-layer wall laws independent of the momentum discretization.

The returned momentum stress is positive into the fluid and the scalar flux is
positive upward.  Keeping these closures separate from the face-flux operator
lets neutral and thermally stratified cases use exactly the same conservative
wall coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


_GAUSS_LEGENDRE_NODES = (
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
    0.1834346424956498,
    0.5255324099163290,
    0.7966664774136267,
    0.9602898564975363,
)
_GAUSS_LEGENDRE_WEIGHTS = (
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
)


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


def _effective_cell_top(
    first_cell_height: float,
    roughness_length: float,
    displacement_height: float,
) -> float:
    if not math.isfinite(first_cell_height) or first_cell_height <= 0.0:
        raise ValueError("first-cell height must be positive and finite")
    effective_top = first_cell_height - displacement_height
    if effective_top <= roughness_length:
        raise ValueError(
            "first cell must extend above displacement plus roughness"
        )
    return effective_top


def _cell_average_log_denominator(
    first_cell_height: float,
    roughness_length: float,
    displacement_height: float,
) -> float:
    """Average the rough-wall log profile over the first control volume.

    The resolved profile is taken as zero below ``d + z0`` and logarithmic
    above it.  Dividing by the complete finite-volume height makes the transfer
    relation act on the prognostic cell average rather than on a point sample.
    """

    effective_top = _effective_cell_top(
        first_cell_height,
        roughness_length,
        displacement_height,
    )
    return (
        effective_top * math.log(effective_top / roughness_length)
        - effective_top
        + roughness_length
    ) / first_cell_height


def _cell_average_stable_slope(
    first_cell_height: float,
    roughness_length: float,
    displacement_height: float,
    beta: float,
) -> float:
    effective_top = _effective_cell_top(
        first_cell_height,
        roughness_length,
        displacement_height,
    )
    return (
        beta
        * (effective_top - roughness_length) ** 2
        / (2.0 * first_cell_height)
    )


@dataclass(frozen=True, slots=True)
class NeutralLogWallLaw:
    """First-control-volume filtered neutral rough-wall logarithmic law."""

    roughness_length: float
    von_karman: float = 0.4
    displacement_height: float = 0.0

    def __post_init__(self) -> None:
        _validate_common(
            self.roughness_length,
            self.von_karman,
            self.displacement_height,
        )

    def cell_average_log_denominator(self, first_cell_height: float) -> float:
        return _cell_average_log_denominator(
            first_cell_height,
            self.roughness_length,
            self.displacement_height,
        )

    def cell_average_log_denominators(
        self,
        lower_heights: Array,
        upper_heights: Array,
    ) -> Array:
        """Average the neutral profile independently over multiple z cells."""
        widths = upper_heights - lower_heights
        effective_lower = lower_heights - self.displacement_height
        effective_upper = upper_heights - self.displacement_height
        roughness = jnp.asarray(self.roughness_length, dtype=widths.dtype)
        integration_lower = jnp.maximum(effective_lower, roughness)
        integration_upper = jnp.maximum(effective_upper, roughness)

        def primitive(height: Array) -> Array:
            return height * jnp.log(height / roughness) - height

        integral = jnp.where(
            effective_upper > roughness,
            primitive(integration_upper) - primitive(integration_lower),
            0.0,
        )
        return integral / widths

    def point_log_denominator(self, height: float | Array) -> Array:
        """Return the integrated neutral similarity function at ``height``."""
        height = jnp.asarray(height)
        effective = height - self.displacement_height
        roughness = jnp.asarray(self.roughness_length, dtype=height.dtype)
        return jnp.where(
            effective > roughness,
            jnp.log(effective / roughness),
            0.0,
        )

    def first_internal_face_velocity(
        self,
        horizontal_cell_average: Array,
        first_cell_height: float,
    ) -> Array:
        """Reconstruct tangential velocity on top of the first control volume.

        The wall-model input is a finite-volume average.  Multiplying it by
        the ratio of the point and cell-averaged log functions produces the
        face value represented by the same neutral MOST profile.  This is a
        reconstruction only; the wall stress continues to be set by
        :meth:`surface_fluxes`.
        """
        average = self.cell_average_log_denominator(first_cell_height)
        face = self.point_log_denominator(first_cell_height)
        return horizontal_cell_average * (face / average)

    def friction_velocity(
        self,
        horizontal_velocity: Array,
        first_cell_height: float,
    ) -> Array:
        speed = jnp.linalg.norm(horizontal_velocity, axis=-1)
        return (
            self.von_karman
            * speed
            / self.cell_average_log_denominator(first_cell_height)
        )

    def friction_velocity_from_layer(
        self,
        horizontal_velocity: Array,
        lower_height: float,
        upper_height: float,
    ) -> Array:
        """Diagnose ``u_*`` from one actual finite-volume layer average."""
        if not 0.0 <= lower_height < upper_height:
            raise ValueError("MOST layer bounds must satisfy 0 <= lower < upper")
        average = self.cell_average_log_denominators(
            jnp.asarray(lower_height),
            jnp.asarray(upper_height),
        )
        speed = jnp.linalg.norm(horizontal_velocity, axis=-1)
        return self.von_karman * speed / jnp.maximum(average, 1.0e-6)

    def surface_fluxes_from_layer(
        self,
        horizontal_velocity: Array,
        lower_height: float,
        upper_height: float,
    ) -> SurfaceLayerFluxes:
        """Return neutral fluxes from an elevated matching control volume."""
        ustar = self.friction_velocity_from_layer(
            horizontal_velocity,
            lower_height,
            upper_height,
        )
        zeros = jnp.zeros_like(ustar)
        return SurfaceLayerFluxes(
            _stress_from_scale(horizontal_velocity, ustar),
            zeros,
            ustar,
            zeros,
            jnp.full_like(ustar, jnp.inf),
        )

    def internal_face_velocities(
        self,
        horizontal_velocity: Array,
        lower_height: float,
        upper_height: float,
        face_heights: Array,
    ) -> Array:
        """Reconstruct all wall-layer faces from one shared neutral MOST law."""
        average = self.cell_average_log_denominators(
            jnp.asarray(lower_height),
            jnp.asarray(upper_height),
        )
        face = self.point_log_denominator(face_heights)
        scale_shape = (face.shape[0],) + (1,) * horizontal_velocity.ndim
        return horizontal_velocity[None, ...] * jnp.reshape(
            face / jnp.maximum(average, 1.0e-6),
            scale_shape,
        )

    def wall_layer_eddy_viscosity(
        self,
        fluxes: SurfaceLayerFluxes,
        heights: Array,
    ) -> Array:
        """Return neutral mixing-length viscosity at wall-layer cell centres."""
        heights = jnp.asarray(heights, dtype=fluxes.friction_velocity.dtype)
        shape = (heights.shape[0],) + (1,) * fluxes.friction_velocity.ndim
        return (
            self.von_karman
            * jnp.reshape(heights, shape)
            * fluxes.friction_velocity[None, ...]
        )

    def surface_fluxes(
        self,
        horizontal_velocity: Array,
        first_cell_height: float,
    ) -> SurfaceLayerFluxes:
        ustar = self.friction_velocity(horizontal_velocity, first_cell_height)
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
    """First-control-volume filtered Businger--Dyer MOST wall law.

    ``surface_fluxes`` solves the local implicit dependence of the momentum and
    heat fluxes on the Obukhov length.  The logarithmic profile and its
    stability corrections are averaged over the actual first finite volume.
    Stable linear corrections and unstable Businger--Dyer corrections are
    used.  A zero temperature difference recovers
    :class:`NeutralLogWallLaw` to roundoff.
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

    def _validate_height(self, first_cell_height: float) -> float:
        _effective_cell_top(
            first_cell_height,
            self.momentum_roughness_length,
            self.displacement_height,
        )
        return _effective_cell_top(
            first_cell_height,
            self.thermal_roughness_length,
            self.displacement_height,
        )

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

    def momentum_gradient_function(self, zeta: Array) -> Array:
        """Return the Businger--Dyer local momentum gradient function."""
        zeta = self._bounded_zeta(zeta)
        stable = 1.0 + self.stable_momentum_beta * zeta
        unstable = jnp.maximum(
            1.0 - self.unstable_momentum_gamma * zeta,
            1.0,
        ) ** (-0.25)
        return jnp.where(zeta >= 0.0, stable, unstable)

    def heat_gradient_function(self, zeta: Array) -> Array:
        """Return the Businger--Dyer local heat gradient function."""
        zeta = self._bounded_zeta(zeta)
        stable = 1.0 + self.stable_heat_beta * zeta
        unstable = jnp.maximum(
            1.0 - self.unstable_heat_gamma * zeta,
            1.0,
        ) ** (-0.5)
        return jnp.where(zeta >= 0.0, stable, unstable)

    def _average_transfer_denominator(
        self,
        inverse_obukhov: Array,
        first_cell_height: float,
        roughness_length: float,
        correction: Callable[[Array], Array],
    ) -> Array:
        """Return the finite-volume average of one integrated MOST profile."""
        effective_top = _effective_cell_top(
            first_cell_height,
            roughness_length,
            self.displacement_height,
        )
        neutral = _cell_average_log_denominator(
            first_cell_height,
            roughness_length,
            self.displacement_height,
        )
        nodes = jnp.asarray(_GAUSS_LEGENDRE_NODES, dtype=inverse_obukhov.dtype)
        weights = jnp.asarray(_GAUSS_LEGENDRE_WEIGHTS, dtype=inverse_obukhov.dtype)
        half_span = 0.5 * (effective_top - roughness_length)
        midpoint = 0.5 * (effective_top + roughness_length)
        heights = midpoint + half_span * nodes
        quadrature_shape = (heights.shape[0],) + (1,) * inverse_obukhov.ndim
        heights = jnp.reshape(heights, quadrature_shape)
        weights = jnp.reshape(weights, quadrature_shape)
        reference = correction(roughness_length * inverse_obukhov)
        correction_integral = half_span * jnp.sum(
            weights
            * (
                -correction(heights * inverse_obukhov[None, ...])
                + reference[None, ...]
            ),
            axis=0,
        )
        return neutral + correction_integral / first_cell_height

    def _layer_average_transfer_denominator(
        self,
        inverse_obukhov: Array,
        lower_height: float,
        upper_height: float,
        roughness_length: float,
        correction: Callable[[Array], Array],
    ) -> Array:
        """Average one integrated MOST profile over an arbitrary FV cell."""
        if not 0.0 <= lower_height < upper_height:
            raise ValueError("MOST layer bounds must satisfy 0 <= lower < upper")
        effective_upper = upper_height - self.displacement_height
        if effective_upper <= roughness_length:
            raise ValueError("MOST matching cell must extend above roughness")
        nodes = jnp.asarray(_GAUSS_LEGENDRE_NODES, dtype=inverse_obukhov.dtype)
        weights = jnp.asarray(_GAUSS_LEGENDRE_WEIGHTS, dtype=inverse_obukhov.dtype)
        half_span = 0.5 * (upper_height - lower_height)
        midpoint = 0.5 * (upper_height + lower_height)
        heights = midpoint + half_span * nodes
        quadrature_shape = (heights.shape[0],) + (1,) * inverse_obukhov.ndim
        heights = jnp.reshape(heights, quadrature_shape)
        weights = jnp.reshape(weights, quadrature_shape)
        values = self.point_transfer_denominator(
            inverse_obukhov[None, ...],
            heights,
            roughness_length,
            correction,
        )
        return 0.5 * jnp.sum(weights * values, axis=0)

    def point_transfer_denominator(
        self,
        inverse_obukhov: Array,
        height: Array,
        roughness_length: float,
        correction: Callable[[Array], Array],
    ) -> Array:
        """Vectorized integrated MOST denominator, zero below roughness."""
        height = jnp.asarray(height, dtype=inverse_obukhov.dtype)
        effective = height - self.displacement_height
        roughness = jnp.asarray(roughness_length, dtype=inverse_obukhov.dtype)
        clipped = jnp.maximum(effective, roughness)
        value = (
            jnp.log(clipped / roughness)
            - correction(clipped * inverse_obukhov)
            + correction(roughness * inverse_obukhov)
        )
        return jnp.where(effective > roughness, value, 0.0)

    def _point_transfer_denominator(
        self,
        inverse_obukhov: Array,
        height: float,
        roughness_length: float,
        correction: Callable[[Array], Array],
    ) -> Array:
        """Return a pointwise integrated MOST transfer denominator."""
        effective_height = height - self.displacement_height
        if effective_height <= roughness_length:
            raise ValueError("MOST reconstruction height must exceed roughness")
        roughness = jnp.asarray(roughness_length, dtype=inverse_obukhov.dtype)
        height_array = jnp.asarray(effective_height, dtype=inverse_obukhov.dtype)
        return (
            jnp.log(height_array / roughness)
            - correction(height_array * inverse_obukhov)
            + correction(roughness * inverse_obukhov)
        )

    def _stable_inverse_obukhov(
        self,
        speed: Array,
        temperature_difference: Array,
        first_cell_height: float,
        neutral_momentum_transfer: float,
        neutral_heat_transfer: float,
        momentum_slope: float,
        heat_slope: float,
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
        quadratic = heat_slope - bulk_coefficient * momentum_slope**2
        linear = (
            neutral_heat_transfer
            - 2.0
            * bulk_coefficient
            * neutral_momentum_transfer
            * momentum_slope
        )
        constant = -bulk_coefficient * neutral_momentum_transfer**2
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
            & (
                (first_cell_height - self.displacement_height)
                * inverse_obukhov
                <= self.maximum_abs_zeta
            )
        )
        return inverse_obukhov, valid

    def _iterative_inverse_obukhov(
        self,
        speed: Array,
        temperature_difference: Array,
        first_cell_height: float,
    ) -> Array:
        """Solve the general stable/unstable MOST relation by relaxation."""
        inverse_obukhov = jnp.zeros_like(speed)
        for _ in range(self.iterations):
            momentum_denominator = self._average_transfer_denominator(
                inverse_obukhov,
                first_cell_height,
                self.momentum_roughness_length,
                self.momentum_stability_correction,
            )
            heat_denominator = self._average_transfer_denominator(
                inverse_obukhov,
                first_cell_height,
                self.thermal_roughness_length,
                self.heat_stability_correction,
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
        first_cell_height: float,
    ) -> SurfaceLayerFluxes:
        """Return local stress and kinematic heat flux at the lower boundary."""
        self._validate_height(first_cell_height)
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
        neutral_momentum_transfer = _cell_average_log_denominator(
            first_cell_height,
            self.momentum_roughness_length,
            self.displacement_height,
        )
        neutral_heat_transfer = _cell_average_log_denominator(
            first_cell_height,
            self.thermal_roughness_length,
            self.displacement_height,
        )
        momentum_slope = _cell_average_stable_slope(
            first_cell_height,
            self.momentum_roughness_length,
            self.displacement_height,
            self.stable_momentum_beta,
        )
        heat_slope = _cell_average_stable_slope(
            first_cell_height,
            self.thermal_roughness_length,
            self.displacement_height,
            self.stable_heat_beta,
        )
        stable_inverse_obukhov, stable_valid = self._stable_inverse_obukhov(
            speed,
            temperature_difference,
            first_cell_height,
            neutral_momentum_transfer,
            neutral_heat_transfer,
            momentum_slope,
            heat_slope,
        )

        def closed_form(_: None) -> tuple[Array, Array, Array]:
            momentum_denominator = (
                neutral_momentum_transfer
                + momentum_slope * stable_inverse_obukhov
            )
            heat_denominator = (
                neutral_heat_transfer + heat_slope * stable_inverse_obukhov
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
                first_cell_height,
            )
            momentum_denominator = self._average_transfer_denominator(
                inverse_obukhov,
                first_cell_height,
                self.momentum_roughness_length,
                self.momentum_stability_correction,
            )
            heat_denominator = self._average_transfer_denominator(
                inverse_obukhov,
                first_cell_height,
                self.thermal_roughness_length,
                self.heat_stability_correction,
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

    def surface_fluxes_from_heat_flux(
        self,
        horizontal_velocity: Array,
        heat_flux: Array | float,
        first_cell_height: float,
    ) -> SurfaceLayerFluxes:
        """Return MOST fluxes when kinematic heat flux is prescribed.

        This closes unstable and stable prescribed-flux surfaces without a
        benchmark-specific branch.  The implicit momentum relation is solved
        for ``u_*`` while ``1/L = -kappa g q/(theta_ref u_*^3)`` supplies the
        stability coupling.
        """
        self._validate_height(first_cell_height)
        speed = jnp.linalg.norm(horizontal_velocity, axis=-1)
        prescribed_flux = jnp.asarray(heat_flux, dtype=horizontal_velocity.dtype)
        neutral = _cell_average_log_denominator(
            first_cell_height,
            self.momentum_roughness_length,
            self.displacement_height,
        )
        ustar = self.von_karman * speed / neutral
        for _ in range(self.iterations):
            inverse_obukhov = -(
                self.von_karman * self.gravity * prescribed_flux
            ) / (
                self.reference_potential_temperature
                * jnp.maximum(ustar**3, 1.0e-12)
            )
            inverse_obukhov = jnp.clip(
                inverse_obukhov,
                -self.maximum_abs_zeta / first_cell_height,
                self.maximum_abs_zeta / first_cell_height,
            )
            denominator = self._average_transfer_denominator(
                inverse_obukhov,
                first_cell_height,
                self.momentum_roughness_length,
                self.momentum_stability_correction,
            )
            candidate = self.von_karman * speed / jnp.maximum(denominator, 1.0e-6)
            ustar = (1.0 - self.relaxation) * ustar + self.relaxation * candidate
        inverse_obukhov = -(
            self.von_karman * self.gravity * prescribed_flux
        ) / (
            self.reference_potential_temperature
            * jnp.maximum(ustar**3, 1.0e-12)
        )
        inverse_obukhov = jnp.clip(
            inverse_obukhov,
            -self.maximum_abs_zeta / first_cell_height,
            self.maximum_abs_zeta / first_cell_height,
        )
        temperature_scale = -prescribed_flux / jnp.maximum(ustar, 1.0e-12)
        obukhov_length = jnp.where(
            jnp.abs(inverse_obukhov) > 1.0e-12,
            1.0 / inverse_obukhov,
            jnp.inf,
        )
        return SurfaceLayerFluxes(
            _stress_from_scale(horizontal_velocity, ustar),
            jnp.broadcast_to(prescribed_flux, ustar.shape),
            ustar,
            temperature_scale,
            obukhov_length,
        )

    def surface_fluxes_from_layer(
        self,
        horizontal_velocity: Array,
        potential_temperature: Array,
        surface_potential_temperature: Array | float,
        lower_height: float,
        upper_height: float,
    ) -> SurfaceLayerFluxes:
        """Solve coupled momentum/heat MOST from one matching FV cell."""
        speed = jnp.linalg.norm(horizontal_velocity, axis=-1)
        surface = jnp.asarray(
            surface_potential_temperature,
            dtype=horizontal_velocity.dtype,
        )
        difference = jnp.asarray(
            potential_temperature,
            dtype=horizontal_velocity.dtype,
        ) - surface
        inverse_obukhov = jnp.zeros_like(speed)
        for _ in range(self.iterations):
            momentum = self._layer_average_transfer_denominator(
                inverse_obukhov,
                lower_height,
                upper_height,
                self.momentum_roughness_length,
                self.momentum_stability_correction,
            )
            heat = self._layer_average_transfer_denominator(
                inverse_obukhov,
                lower_height,
                upper_height,
                self.thermal_roughness_length,
                self.heat_stability_correction,
            )
            ustar = self.von_karman * speed / jnp.maximum(momentum, 1.0e-6)
            temperature_scale = (
                self.von_karman * difference / jnp.maximum(heat, 1.0e-6)
            )
            candidate = (
                self.von_karman
                * self.gravity
                * temperature_scale
                / (
                    self.reference_potential_temperature
                    * jnp.maximum(ustar * ustar, 1.0e-12)
                )
            )
            candidate = jnp.clip(
                candidate,
                -self.maximum_abs_zeta / upper_height,
                self.maximum_abs_zeta / upper_height,
            )
            inverse_obukhov = (
                (1.0 - self.relaxation) * inverse_obukhov
                + self.relaxation * candidate
            )
        momentum = self._layer_average_transfer_denominator(
            inverse_obukhov,
            lower_height,
            upper_height,
            self.momentum_roughness_length,
            self.momentum_stability_correction,
        )
        heat = self._layer_average_transfer_denominator(
            inverse_obukhov,
            lower_height,
            upper_height,
            self.thermal_roughness_length,
            self.heat_stability_correction,
        )
        ustar = self.von_karman * speed / jnp.maximum(momentum, 1.0e-6)
        temperature_scale = self.von_karman * difference / jnp.maximum(
            heat,
            1.0e-6,
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

    def surface_fluxes_from_layer_heat_flux(
        self,
        horizontal_velocity: Array,
        heat_flux: Array | float,
        lower_height: float,
        upper_height: float,
    ) -> SurfaceLayerFluxes:
        """Solve MOST at a matching FV cell for prescribed surface heat flux."""
        speed = jnp.linalg.norm(horizontal_velocity, axis=-1)
        prescribed = jnp.asarray(heat_flux, dtype=horizontal_velocity.dtype)
        neutral = self._layer_average_transfer_denominator(
            jnp.zeros_like(speed),
            lower_height,
            upper_height,
            self.momentum_roughness_length,
            self.momentum_stability_correction,
        )
        ustar = self.von_karman * speed / jnp.maximum(neutral, 1.0e-6)
        for _ in range(self.iterations):
            inverse_obukhov = -(
                self.von_karman * self.gravity * prescribed
            ) / (
                self.reference_potential_temperature
                * jnp.maximum(ustar**3, 1.0e-12)
            )
            inverse_obukhov = jnp.clip(
                inverse_obukhov,
                -self.maximum_abs_zeta / upper_height,
                self.maximum_abs_zeta / upper_height,
            )
            momentum = self._layer_average_transfer_denominator(
                inverse_obukhov,
                lower_height,
                upper_height,
                self.momentum_roughness_length,
                self.momentum_stability_correction,
            )
            candidate = self.von_karman * speed / jnp.maximum(momentum, 1.0e-6)
            ustar = (1.0 - self.relaxation) * ustar + self.relaxation * candidate
        inverse_obukhov = -(
            self.von_karman * self.gravity * prescribed
        ) / (
            self.reference_potential_temperature
            * jnp.maximum(ustar**3, 1.0e-12)
        )
        inverse_obukhov = jnp.clip(
            inverse_obukhov,
            -self.maximum_abs_zeta / upper_height,
            self.maximum_abs_zeta / upper_height,
        )
        temperature_scale = -prescribed / jnp.maximum(ustar, 1.0e-12)
        obukhov_length = jnp.where(
            jnp.abs(inverse_obukhov) > 1.0e-12,
            1.0 / inverse_obukhov,
            jnp.inf,
        )
        return SurfaceLayerFluxes(
            _stress_from_scale(horizontal_velocity, ustar),
            jnp.broadcast_to(prescribed, ustar.shape),
            ustar,
            temperature_scale,
            obukhov_length,
        )

    def internal_face_profiles(
        self,
        horizontal_velocity: Array,
        cell_average_temperature: Array,
        fluxes: SurfaceLayerFluxes,
        lower_height: float,
        upper_height: float,
        face_heights: Array,
        *,
        surface_temperature: Array | float | None,
    ) -> tuple[Array, Array]:
        """Return shared-MOST velocity and temperature on wall-layer faces."""
        inverse_obukhov = self._inverse_obukhov(fluxes)
        heights = jnp.asarray(face_heights, dtype=horizontal_velocity.dtype)
        expanded_inverse = inverse_obukhov[None, ...]
        height_shape = (heights.shape[0],) + (1,) * inverse_obukhov.ndim
        expanded_heights = jnp.reshape(heights, height_shape)
        momentum = self.point_transfer_denominator(
            expanded_inverse,
            expanded_heights,
            self.momentum_roughness_length,
            self.momentum_stability_correction,
        )
        speed = jnp.linalg.norm(horizontal_velocity, axis=-1)
        direction = horizontal_velocity / jnp.maximum(speed, 1.0e-12)[..., None]
        face_velocity = (
            fluxes.friction_velocity[None, ..., None]
            * direction[None, ...]
            * momentum[..., None]
            / self.von_karman
        )
        face_temperature = self.internal_face_temperatures(
            cell_average_temperature,
            fluxes,
            lower_height,
            upper_height,
            face_heights,
            surface_temperature=surface_temperature,
        )
        return face_velocity, face_temperature

    def internal_face_temperatures(
        self,
        cell_average_temperature: Array,
        fluxes: SurfaceLayerFluxes,
        lower_height: float,
        upper_height: float,
        face_heights: Array,
        *,
        surface_temperature: Array | float | None,
    ) -> Array:
        """Return the shared-MOST thermal profile on modeled z faces."""
        inverse_obukhov = self._inverse_obukhov(fluxes)
        heights = jnp.asarray(face_heights, dtype=cell_average_temperature.dtype)
        height_shape = (heights.shape[0],) + (1,) * inverse_obukhov.ndim
        heat = self.point_transfer_denominator(
            inverse_obukhov[None, ...],
            jnp.reshape(heights, height_shape),
            self.thermal_roughness_length,
            self.heat_stability_correction,
        )
        if surface_temperature is not None:
            surface = jnp.asarray(
                surface_temperature,
                dtype=cell_average_temperature.dtype,
            )
            face_temperature = surface + (
                fluxes.temperature_scale[None, ...] * heat / self.von_karman
            )
        else:
            average = self._layer_average_transfer_denominator(
                inverse_obukhov,
                lower_height,
                upper_height,
                self.thermal_roughness_length,
                self.heat_stability_correction,
            )
            intercept = cell_average_temperature - (
                fluxes.temperature_scale * average / self.von_karman
            )
            face_temperature = intercept[None, ...] + (
                fluxes.temperature_scale[None, ...] * heat / self.von_karman
            )
        return face_temperature

    def wall_layer_eddy_diffusivities(
        self,
        fluxes: SurfaceLayerFluxes,
        heights: Array,
    ) -> tuple[Array, Array]:
        """Return shared-MOST momentum and heat diffusivities at FV centres."""
        inverse_obukhov = self._inverse_obukhov(fluxes)
        heights = jnp.asarray(heights, dtype=fluxes.friction_velocity.dtype)
        shape = (heights.shape[0],) + (1,) * inverse_obukhov.ndim
        expanded_height = jnp.reshape(heights, shape)
        zeta = expanded_height * inverse_obukhov[None, ...]
        numerator = (
            self.von_karman
            * expanded_height
            * fluxes.friction_velocity[None, ...]
        )
        return (
            numerator / jnp.maximum(self.momentum_gradient_function(zeta), 1.0e-6),
            numerator / jnp.maximum(self.heat_gradient_function(zeta), 1.0e-6),
        )

    @staticmethod
    def _inverse_obukhov(fluxes: SurfaceLayerFluxes) -> Array:
        return jnp.where(
            jnp.isfinite(fluxes.obukhov_length),
            1.0 / fluxes.obukhov_length,
            0.0,
        )

    def first_internal_face_velocity(
        self,
        horizontal_cell_average: Array,
        fluxes: SurfaceLayerFluxes,
        first_cell_height: float,
    ) -> Array:
        """Return the MOST-consistent tangential velocity at the first face."""
        inverse_obukhov = self._inverse_obukhov(fluxes)
        average = self._average_transfer_denominator(
            inverse_obukhov,
            first_cell_height,
            self.momentum_roughness_length,
            self.momentum_stability_correction,
        )
        face = self._point_transfer_denominator(
            inverse_obukhov,
            first_cell_height,
            self.momentum_roughness_length,
            self.momentum_stability_correction,
        )
        return horizontal_cell_average * (face / jnp.maximum(average, 1.0e-6))[..., None]

    def first_internal_face_temperature(
        self,
        cell_average_temperature: Array,
        fluxes: SurfaceLayerFluxes,
        first_cell_height: float,
        *,
        surface_temperature: Array | float | None,
    ) -> Array:
        """Return a MOST-consistent scalar value at the first internal face.

        For a prescribed temperature, the face-to-cell ratio follows directly
        from the integrated similarity profile.  For a prescribed flux, the
        unknown surface intercept cancels and the reconstruction uses the
        diagnosed temperature scale.
        """
        inverse_obukhov = self._inverse_obukhov(fluxes)
        average = self._average_transfer_denominator(
            inverse_obukhov,
            first_cell_height,
            self.thermal_roughness_length,
            self.heat_stability_correction,
        )
        face = self._point_transfer_denominator(
            inverse_obukhov,
            first_cell_height,
            self.thermal_roughness_length,
            self.heat_stability_correction,
        )
        if surface_temperature is None:
            return cell_average_temperature + (
                fluxes.temperature_scale / self.von_karman
            ) * (face - average)
        surface = jnp.asarray(surface_temperature, dtype=cell_average_temperature.dtype)
        return surface + (cell_average_temperature - surface) * (
            face / jnp.maximum(average, 1.0e-6)
        )


__all__ = [
    "MoninObukhovWallLaw",
    "NeutralLogWallLaw",
    "SurfaceLayerFluxes",
]
