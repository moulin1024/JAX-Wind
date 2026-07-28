"""JAX building blocks for rigid blade-element actuator lines."""

from __future__ import annotations

import jax.numpy as jnp


def actuator_line_geometry(
    *,
    x,
    y,
    z,
    blade_count: int,
    element_radii,
    angular_velocity,
    time,
    yaw_degrees,
    tilt_degrees,
    precone_degrees,
    initial_azimuth_degrees,
    dtype,
):
    """Return flattened element positions, tangents, and the rotor normal."""
    yaw = jnp.deg2rad(jnp.asarray(yaw_degrees, dtype=dtype))
    tilt = jnp.deg2rad(jnp.asarray(tilt_degrees, dtype=dtype))
    precone = jnp.deg2rad(jnp.asarray(precone_degrees, dtype=dtype))
    normal = jnp.asarray(
        (
            jnp.cos(tilt) * jnp.cos(yaw),
            jnp.cos(tilt) * jnp.sin(yaw),
            jnp.sin(tilt),
        ),
        dtype=dtype,
    )
    horizontal = jnp.asarray(
        (-jnp.sin(yaw), jnp.cos(yaw), 0.0),
        dtype=dtype,
    )
    vertical = jnp.cross(normal, horizontal)

    radii = jnp.asarray(element_radii, dtype=dtype)
    blade_phase = (
        2.0
        * jnp.pi
        * jnp.arange(blade_count, dtype=dtype)
        / jnp.asarray(blade_count, dtype=dtype)
    )
    theta = (
        jnp.deg2rad(jnp.asarray(initial_azimuth_degrees, dtype=dtype))
        + jnp.asarray(angular_velocity, dtype=dtype) * jnp.asarray(time, dtype=dtype)
        + blade_phase[:, None]
        + jnp.zeros_like(radii)[None, :]
    )
    radial = (
        jnp.cos(theta)[..., None] * vertical
        + jnp.sin(theta)[..., None] * horizontal
    )
    tangent = (
        -jnp.sin(theta)[..., None] * vertical
        + jnp.cos(theta)[..., None] * horizontal
    )
    coned_radial = jnp.cos(precone) * radial + jnp.sin(precone) * normal
    hub = jnp.asarray((x, y, z), dtype=dtype)
    positions = hub + radii[None, :, None] * coned_radial
    tangential_speed = (
        jnp.asarray(angular_velocity, dtype=dtype)
        * radii[None, :]
        * jnp.cos(precone)
    )
    return (
        positions.reshape((-1, 3)),
        tangent.reshape((-1, 3)),
        jnp.broadcast_to(tangential_speed, theta.shape).reshape((-1,)),
        normal,
    )


def actuator_line_deformed_kinematics(
    *,
    x,
    y,
    z,
    blade_count: int,
    element_radii,
    angular_velocity,
    time,
    yaw_degrees,
    tilt_degrees,
    precone_degrees,
    initial_azimuth_degrees,
    flap_displacements,
    edge_displacements,
    flap_slopes,
    edge_slopes,
    flap_velocities,
    edge_velocities,
    dtype,
):
    """Return small-deflection actuator-line positions and section kinematics.

    Flap quantities are measured along the rigid rotor normal and edge
    quantities along the rotating in-plane tangent.  Slopes are derivatives
    with respect to undeformed blade span.  The resulting section basis is
    orthonormalized, while the point velocity includes both modal rates and
    rotation of an edgewise displacement.
    """

    yaw = jnp.deg2rad(jnp.asarray(yaw_degrees, dtype=dtype))
    tilt = jnp.deg2rad(jnp.asarray(tilt_degrees, dtype=dtype))
    precone = jnp.deg2rad(jnp.asarray(precone_degrees, dtype=dtype))
    normal = jnp.asarray(
        (
            jnp.cos(tilt) * jnp.cos(yaw),
            jnp.cos(tilt) * jnp.sin(yaw),
            jnp.sin(tilt),
        ),
        dtype=dtype,
    )
    horizontal = jnp.asarray(
        (-jnp.sin(yaw), jnp.cos(yaw), 0.0),
        dtype=dtype,
    )
    vertical = jnp.cross(normal, horizontal)

    radii = jnp.asarray(element_radii, dtype=dtype)
    element_count = radii.size
    blade_phase = (
        2.0
        * jnp.pi
        * jnp.arange(blade_count, dtype=dtype)
        / jnp.asarray(blade_count, dtype=dtype)
    )
    theta = (
        jnp.deg2rad(jnp.asarray(initial_azimuth_degrees, dtype=dtype))
        + jnp.asarray(angular_velocity, dtype=dtype) * jnp.asarray(time, dtype=dtype)
        + blade_phase[:, None]
        + jnp.zeros_like(radii)[None, :]
    )
    radial = (
        jnp.cos(theta)[..., None] * vertical
        + jnp.sin(theta)[..., None] * horizontal
    )
    tangent = (
        -jnp.sin(theta)[..., None] * vertical
        + jnp.cos(theta)[..., None] * horizontal
    )
    coned_radial = jnp.cos(precone) * radial + jnp.sin(precone) * normal

    shape = (blade_count, element_count)

    def modal(values):
        return jnp.asarray(values, dtype=dtype).reshape(shape)

    flap = modal(flap_displacements)
    edge = modal(edge_displacements)
    flap_slope = modal(flap_slopes)
    edge_slope = modal(edge_slopes)
    flap_rate = modal(flap_velocities)
    edge_rate = modal(edge_velocities)

    span_direction = (
        coned_radial
        + flap_slope[..., None] * normal
        + edge_slope[..., None] * tangent
    )
    span_direction = span_direction / jnp.maximum(
        jnp.linalg.norm(span_direction, axis=-1, keepdims=True),
        jnp.finfo(dtype).tiny,
    )
    section_tangent = tangent - jnp.sum(
        tangent * span_direction,
        axis=-1,
        keepdims=True,
    ) * span_direction
    section_tangent = section_tangent / jnp.maximum(
        jnp.linalg.norm(section_tangent, axis=-1, keepdims=True),
        jnp.finfo(dtype).tiny,
    )
    section_normal = jnp.cross(section_tangent, span_direction)
    section_normal = section_normal / jnp.maximum(
        jnp.linalg.norm(section_normal, axis=-1, keepdims=True),
        jnp.finfo(dtype).tiny,
    )

    hub = jnp.asarray((x, y, z), dtype=dtype)
    positions = (
        hub
        + radii[None, :, None] * coned_radial
        + flap[..., None] * normal
        + edge[..., None] * tangent
    )
    omega = jnp.asarray(angular_velocity, dtype=dtype)
    point_velocity = (
        omega * radii[None, :, None] * jnp.cos(precone) * tangent
        + flap_rate[..., None] * normal
        + edge_rate[..., None] * tangent
        - omega * edge[..., None] * radial
    )
    return (
        positions.reshape((-1, 3)),
        section_tangent.reshape((-1, 3)),
        point_velocity.reshape((-1, 3)),
        section_normal.reshape((-1, 3)),
        span_direction.reshape((-1, 3)),
    )


def gaussian_weights(points, coordinates, *, smoothing_width, period=None):
    """Return one discretely normalized one-dimensional Gaussian per point."""
    points = jnp.asarray(points)
    coordinates = jnp.asarray(coordinates, dtype=points.dtype)
    distance = coordinates[None, :] - points[:, None]
    if period is not None:
        period = jnp.asarray(period, dtype=points.dtype)
        distance = jnp.mod(distance + 0.5 * period, period) - 0.5 * period
    weights = jnp.exp(
        -(distance / jnp.asarray(smoothing_width, dtype=points.dtype)) ** 2
    )
    return weights / jnp.maximum(
        jnp.sum(weights, axis=1, keepdims=True),
        jnp.finfo(points.dtype).tiny,
    )


def interpolate_polar(
    alpha_degrees,
    airfoil_ids,
    polar_alpha_degrees,
    polar_coefficients,
):
    """Linearly interpolate one row-selected coefficient for every element."""
    alpha_degrees = jnp.asarray(alpha_degrees)
    alpha_grid = jnp.asarray(
        polar_alpha_degrees,
        dtype=alpha_degrees.dtype,
    )
    table = jnp.asarray(polar_coefficients, dtype=alpha_degrees.dtype)
    rows = table[jnp.asarray(airfoil_ids, dtype=jnp.int32)]
    lower = jnp.searchsorted(alpha_grid, alpha_degrees, side="right") - 1
    lower = jnp.clip(lower, 0, alpha_grid.size - 2)
    upper = lower + 1
    lower_alpha = alpha_grid[lower]
    fraction = (alpha_degrees - lower_alpha) / (
        alpha_grid[upper] - lower_alpha
    )
    fraction = jnp.clip(fraction, 0.0, 1.0)
    lower_value = jnp.take_along_axis(rows, lower[:, None], axis=1)[:, 0]
    upper_value = jnp.take_along_axis(rows, upper[:, None], axis=1)[:, 0]
    return lower_value + fraction * (upper_value - lower_value)


def blade_element_kinematic_forces(
    sampled_velocity,
    tangents,
    tangential_speed,
    normal,
    *,
    element_radii,
    element_widths,
    element_chords,
    element_twist_degrees,
    element_airfoil_ids,
    blade_count,
    hub_radius,
    tip_radius,
    pitch_degrees,
    polar_alpha_degrees,
    polar_lift_coefficients,
    polar_drag_coefficients,
    tip_loss,
    root_loss,
    blade_velocity=None,
):
    """Evaluate force-on-fluid divided by density for every blade element."""
    velocity = jnp.asarray(sampled_velocity)
    tangents = jnp.asarray(tangents, dtype=velocity.dtype)
    normal = jnp.asarray(normal, dtype=velocity.dtype)
    tangential_speed = jnp.asarray(tangential_speed, dtype=velocity.dtype)

    normals = jnp.asarray(normal, dtype=velocity.dtype)
    if normals.ndim == 1:
        normals = jnp.broadcast_to(normals, velocity.shape)
    if blade_velocity is None:
        relative_velocity = velocity
        relative_tangential = (
            jnp.sum(relative_velocity * tangents, axis=1)
            - tangential_speed
        )
    else:
        relative_velocity = velocity - jnp.asarray(
            blade_velocity,
            dtype=velocity.dtype,
        )
        relative_tangential = jnp.sum(
            relative_velocity * tangents,
            axis=1,
        )
    normal_velocity = jnp.sum(relative_velocity * normals, axis=1)
    relative_speed = jnp.sqrt(
        normal_velocity * normal_velocity
        + relative_tangential * relative_tangential
    )
    safe_speed = jnp.maximum(relative_speed, jnp.finfo(velocity.dtype).tiny)
    sine_phi = normal_velocity / safe_speed
    cosine_phi = -relative_tangential / safe_speed
    phi = jnp.arctan2(normal_velocity, -relative_tangential)

    alpha_degrees = (
        jnp.rad2deg(phi)
        - jnp.asarray(element_twist_degrees, dtype=velocity.dtype)
        - jnp.asarray(pitch_degrees, dtype=velocity.dtype)
    )
    lift = interpolate_polar(
        alpha_degrees,
        element_airfoil_ids,
        polar_alpha_degrees,
        polar_lift_coefficients,
    )
    drag = interpolate_polar(
        alpha_degrees,
        element_airfoil_ids,
        polar_alpha_degrees,
        polar_drag_coefficients,
    )

    radii = jnp.asarray(element_radii, dtype=velocity.dtype)
    widths = jnp.asarray(element_widths, dtype=velocity.dtype)
    chords = jnp.asarray(element_chords, dtype=velocity.dtype)
    sine_magnitude = jnp.maximum(
        jnp.abs(sine_phi),
        jnp.sqrt(jnp.finfo(velocity.dtype).eps),
    )
    blade_factor = 0.5 * jnp.asarray(blade_count, dtype=velocity.dtype)
    tip_exponent = (
        blade_factor
        * jnp.maximum(jnp.asarray(tip_radius, velocity.dtype) - radii, 0.0)
        / jnp.maximum(radii * sine_magnitude, jnp.finfo(velocity.dtype).tiny)
    )
    root_exponent = (
        blade_factor
        * jnp.maximum(radii - jnp.asarray(hub_radius, velocity.dtype), 0.0)
        / jnp.maximum(
            jnp.asarray(hub_radius, velocity.dtype) * sine_magnitude,
            jnp.finfo(velocity.dtype).tiny,
        )
    )
    prandtl_tip = 2.0 / jnp.pi * jnp.arccos(jnp.exp(-tip_exponent))
    prandtl_root = 2.0 / jnp.pi * jnp.arccos(jnp.exp(-root_exponent))
    loss = jnp.where(jnp.asarray(tip_loss), prandtl_tip, 1.0)
    loss = loss * jnp.where(jnp.asarray(root_loss), prandtl_root, 1.0)

    relative_direction = (
        sine_phi[:, None] * normals
        - cosine_phi[:, None] * tangents
    )
    lift_direction = (
        cosine_phi[:, None] * normals
        + sine_phi[:, None] * tangents
    )
    dynamic_line_force = (
        0.5 * relative_speed**2 * chords * widths * loss
    )
    force_on_fluid = -dynamic_line_force[:, None] * (
        lift[:, None] * lift_direction
        + drag[:, None] * relative_direction
    )
    return force_on_fluid, alpha_degrees, lift, drag, loss
