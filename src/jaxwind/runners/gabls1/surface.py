"""Physical field conversion and surface diagnostics for stratified ABL runs."""

from __future__ import annotations

import math

from .most import SurfaceFluxes


def is_fixed_surface_flux(case) -> bool:
    return (
        getattr(
            case.thermal,
            "boundary_condition",
            "prescribed_surface_temperature",
        )
        == "fixed_surface_flux"
    )


def stability(case) -> str:
    configured = getattr(case, "stability", None)
    if configured is not None:
        return configured
    if is_fixed_surface_flux(case):
        return (
            "unstable"
            if case.thermal.surface_heat_flux_k_m_s > 0.0
            else "stable"
        )
    boundary = getattr(
        case.thermal,
        "boundary_condition",
        "prescribed_surface_temperature",
    )
    if boundary == "prescribed_surface_temperature":
        return "unstable" if case.thermal.surface_cooling_k_s > 0.0 else "stable"
    return "neutral"


def velocity_scale(case) -> float:
    if is_fixed_surface_flux(case):
        depth = max(case.thermal.inversion_height_m, case.domain.dz_m)
        return (
            case.thermal.gravity_m_s2
            * abs(case.thermal.surface_heat_flux_k_m_s)
            * depth
            / case.thermal.reference_temperature_k
        ) ** (1.0 / 3.0)
    return math.hypot(
        case.flow.geostrophic_u_m_s,
        case.flow.geostrophic_v_m_s,
    )


def physical_arrays(fields, case, mechanical_scales, thermal_scales, jnp):
    velocity = fields.velocity
    u = case.flow.geostrophic_u_m_s + mechanical_scales.from_execution_velocity(
        velocity.x.payload[0]
    )
    v = case.flow.geostrophic_v_m_s + mechanical_scales.from_execution_velocity(
        velocity.y.payload[0]
    )
    w_upper = mechanical_scales.from_execution_velocity(
        velocity.z.owned.payload[0]
    )
    w_lower = jnp.concatenate((jnp.zeros_like(w_upper[:1]), w_upper[:-1]), axis=0)
    w = 0.5 * (w_lower + w_upper)
    theta = case.thermal.initial_temperature_k + (
        thermal_scales.from_execution_potential_temperature(
            fields.potential_temperature.payload[0]
        )
    )
    return u, v, w, w_upper, theta


def surface_fluxes(
    fields,
    execution_time,
    *,
    case,
    mechanical_scales,
    thermal_scales,
    wall_law,
    jnp,
):
    u, v, _w, _w_upper, theta = physical_arrays(
        fields, case, mechanical_scales, thermal_scales, jnp
    )
    boundary = getattr(
        case.thermal,
        "boundary_condition",
        "prescribed_surface_temperature",
    )
    if boundary == "fixed_surface_flux":
        first_level_shape = u[0].shape
        mean_u = jnp.mean(u[0])
        mean_v = jnp.mean(v[0])
        mean_theta = jnp.mean(theta[0])
        speed = jnp.hypot(mean_u, mean_v)
        height = 0.5 * case.domain.dz_m
        drag_root = case.flow.von_karman / jnp.log(
            height / case.flow.roughness_length_m
        )
        friction_velocity = drag_root * speed
        safe_speed = jnp.maximum(speed, jnp.finfo(speed.dtype).tiny)
        stress = friction_velocity**2
        heat_flux = jnp.asarray(
            case.thermal.surface_heat_flux_k_m_s,
            dtype=mean_theta.dtype,
        )
        temperature_scale = -heat_flux / jnp.maximum(
            friction_velocity,
            jnp.finfo(speed.dtype).tiny,
        )
        obukhov = -(
            jnp.maximum(friction_velocity, jnp.finfo(speed.dtype).tiny) ** 3
            * mean_theta
            / (
                case.flow.von_karman
                * case.thermal.gravity_m_s2
                * heat_flux
            )
        )
        fluxes = SurfaceFluxes(
            stress * mean_u / safe_speed,
            stress * mean_v / safe_speed,
            heat_flux,
            friction_velocity,
            temperature_scale,
            obukhov,
        )
        return SurfaceFluxes(
            *(jnp.broadcast_to(value, first_level_shape) for value in fluxes)
        ), float(case.thermal.initial_temperature_k)

    physical_time = mechanical_scales.from_execution_time(execution_time)
    surface_temperature = (
        case.thermal.initial_temperature_k
        + case.thermal.surface_cooling_k_s * physical_time
    )
    first_level_shape = u[0].shape
    mean_fluxes = wall_law.surface_fluxes(
        jnp.mean(u[0]),
        jnp.mean(v[0]),
        jnp.mean(theta[0]),
        surface_temperature,
        0.5 * case.domain.dz_m,
    )
    fluxes = type(mean_fluxes)(
        *(jnp.broadcast_to(value, first_level_shape) for value in mean_fluxes)
    )
    return fluxes, surface_temperature


__all__ = [
    "is_fixed_surface_flux",
    "physical_arrays",
    "stability",
    "surface_fluxes",
    "velocity_scale",
]
