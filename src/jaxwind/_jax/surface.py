"""Compiled lower-boundary exchange kernels for the JAX solver."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from jaxwind.physics.surface_transfer import SurfaceTransferResult


def build_monin_obukhov_surface_transfer_kernel(axis_name: str):
    """Build a job-wide lower-surface law over the solver device axis."""

    def local(
        u_payload,
        v_payload,
        scalar_payload,
        execution_time,
        grid_spacing,
        momentum_roughness_length,
        scalar_roughness_length,
        surface_scalar_initial,
        surface_scalar_rate,
        x_velocity_offset,
        y_velocity_offset,
        buoyancy_coefficient,
        von_karman,
        positive_zeta_momentum_slope,
        positive_zeta_scalar_slope,
        negative_zeta_momentum_coefficient,
        negative_zeta_scalar_coefficient,
        relaxation,
        maximum_abs_zeta,
        iterations,
    ):
        index = jax.lax.axis_index(axis_name)
        is_bottom = index == 0

        def bottom_mean(values):
            local_mean = jnp.where(is_bottom, jnp.mean(values[0]), 0.0)
            return jax.lax.psum(local_mean, axis_name)

        shape = (1, 1, 1)
        return monin_obukhov_surface_transfer(
            bottom_mean(u_payload).reshape(shape),
            bottom_mean(v_payload).reshape(shape),
            bottom_mean(scalar_payload).reshape(shape),
            execution_time,
            grid_spacing,
            momentum_roughness_length,
            scalar_roughness_length,
            surface_scalar_initial,
            surface_scalar_rate,
            x_velocity_offset,
            y_velocity_offset,
            buoyancy_coefficient,
            von_karman,
            positive_zeta_momentum_slope,
            positive_zeta_scalar_slope,
            negative_zeta_momentum_coefficient,
            negative_zeta_scalar_coefficient,
            relaxation,
            maximum_abs_zeta,
            bottom=0,
            iterations=iterations,
        )

    return jax.pmap(
        local,
        axis_name=axis_name,
        in_axes=(0, 0, 0) + (None,) * 17,
        static_broadcasted_argnums=(19,),
    )


@partial(jax.jit, static_argnames=("bottom", "iterations"))
def monin_obukhov_surface_transfer(
    u_payload,
    v_payload,
    scalar_payload,
    execution_time,
    grid_spacing,
    momentum_roughness_length,
    scalar_roughness_length,
    surface_scalar_initial,
    surface_scalar_rate,
    x_velocity_offset,
    y_velocity_offset,
    buoyancy_coefficient,
    von_karman,
    positive_zeta_momentum_slope,
    positive_zeta_scalar_slope,
    negative_zeta_momentum_coefficient,
    negative_zeta_scalar_coefficient,
    relaxation,
    maximum_abs_zeta,
    *,
    bottom: int,
    iterations: int,
) -> SurfaceTransferResult:
    """Return coupled plane-mean exchange as one compiled GPU program."""

    measurement_height = 0.5 * grid_spacing
    u = jnp.mean(u_payload[bottom, 0]) + x_velocity_offset
    v = jnp.mean(v_payload[bottom, 0]) + y_velocity_offset
    scalar = jnp.mean(scalar_payload[bottom, 0])
    surface_scalar = surface_scalar_initial + surface_scalar_rate * execution_time
    speed = jnp.hypot(u, v)
    scalar_difference = scalar - surface_scalar

    def bounded_zeta(zeta):
        return jnp.clip(zeta, -maximum_abs_zeta, maximum_abs_zeta)

    def momentum_correction(zeta):
        zeta = bounded_zeta(zeta)
        positive = -positive_zeta_momentum_slope * zeta
        x = jnp.maximum(
            1.0 - negative_zeta_momentum_coefficient * zeta,
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
        zeta = bounded_zeta(zeta)
        positive = -positive_zeta_scalar_slope * zeta
        x = jnp.maximum(
            1.0 - negative_zeta_scalar_coefficient * zeta,
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

    momentum_neutral = jnp.log(
        measurement_height / momentum_roughness_length
    )
    scalar_neutral = jnp.log(measurement_height / scalar_roughness_length)
    momentum_slope = positive_zeta_momentum_slope * (
        measurement_height - momentum_roughness_length
    )
    scalar_slope = positive_zeta_scalar_slope * (
        measurement_height - scalar_roughness_length
    )
    speed_squared = jnp.maximum(speed * speed, 1.0e-12)
    positive_c = (
        buoyancy_coefficient
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
    limit = maximum_abs_zeta / measurement_height
    positive_inverse_obukhov = jnp.clip(
        positive_inverse_obukhov,
        0.0,
        limit,
    )

    inverse_obukhov = jnp.zeros_like(speed)
    for _ in range(iterations):
        momentum_denominator = jnp.maximum(
            denominator(
                momentum_roughness_length,
                inverse_obukhov,
                momentum_correction,
            ),
            1.0e-6,
        )
        scalar_denominator = jnp.maximum(
            denominator(
                scalar_roughness_length,
                inverse_obukhov,
                scalar_correction,
            ),
            1.0e-6,
        )
        friction_velocity = von_karman * speed / momentum_denominator
        scalar_scale = von_karman * scalar_difference / scalar_denominator
        candidate = (
            von_karman
            * buoyancy_coefficient
            * scalar_scale
            / jnp.maximum(friction_velocity**2, 1.0e-12)
        )
        candidate = jnp.clip(candidate, -limit, limit)
        inverse_obukhov = (
            (1.0 - relaxation) * inverse_obukhov + relaxation * candidate
        )
    inverse_obukhov = jnp.where(
        scalar_difference >= 0.0,
        positive_inverse_obukhov,
        inverse_obukhov,
    )
    momentum_denominator = jnp.maximum(
        denominator(
            momentum_roughness_length,
            inverse_obukhov,
            momentum_correction,
        ),
        1.0e-6,
    )
    scalar_denominator = jnp.maximum(
        denominator(
            scalar_roughness_length,
            inverse_obukhov,
            scalar_correction,
        ),
        1.0e-6,
    )
    friction_velocity = von_karman * speed / momentum_denominator
    scalar_scale = von_karman * scalar_difference / scalar_denominator
    stress = friction_velocity**2
    direction_denominator = jnp.where(speed > 0.0, speed, 1.0)
    stress_x = stress * u / direction_denominator
    stress_y = stress * v / direction_denominator
    scalar_flux = -friction_velocity * scalar_scale
    obukhov_length = jnp.where(
        jnp.abs(inverse_obukhov) > 1.0e-12,
        1.0 / inverse_obukhov,
        jnp.inf,
    )
    return SurfaceTransferResult(
        stress_x,
        stress_y,
        scalar_flux,
        friction_velocity,
        scalar_scale,
        obukhov_length,
        surface_scalar,
        -stress_x / grid_spacing,
        -stress_y / grid_spacing,
        scalar_flux / grid_spacing,
    )


__all__ = [
    "build_monin_obukhov_surface_transfer_kernel",
    "monin_obukhov_surface_transfer",
]
