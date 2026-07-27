from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Params


def _cell_coordinates(params: Params) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Physical cell-centre coordinates, in metres."""
    x = (jnp.arange(params.nx, dtype=params.dtype) + 0.5) * params.dx * params.z_i
    y = (jnp.arange(params.ny, dtype=params.dtype) + 0.5) * params.dy * params.z_i
    z = (jnp.arange(params.nz, dtype=params.dtype) + 0.5) * params.dz * params.z_i
    return x[:, None, None], y[None, :, None], z[None, None, :]


def _periodic_offset(coordinate: jax.Array, centre: float, length: float) -> jax.Array:
    """Shortest signed offset from ``centre`` on a periodic axis."""
    period = jnp.asarray(length, dtype=coordinate.dtype)
    half_period = 0.5 * period
    return jnp.mod(coordinate - centre + half_period, period) - half_period


def _periodic_distance(coordinate: jax.Array, centre: float, length: float) -> jax.Array:
    return jnp.abs(_periodic_offset(coordinate, centre, length))


def _cinf_step(coordinate: jax.Array) -> jax.Array:
    """Compact C-infinity step from zero to one on ``0 < x < 1``."""
    one = jnp.asarray(1.0, dtype=coordinate.dtype)
    epsilon = jnp.asarray(jnp.finfo(coordinate.dtype).eps, dtype=coordinate.dtype)
    safe = jnp.clip(coordinate, epsilon, one - epsilon)
    interior = jax.nn.sigmoid(one / (one - safe) - one / safe)
    return jnp.where(
        coordinate <= 0.0,
        jnp.zeros_like(interior),
        jnp.where(coordinate >= 1.0, jnp.ones_like(interior), interior),
    )


def classic_fringe_window(
    x: jax.Array,
    start: float,
    end: float,
) -> jax.Array:
    """Nordstrom-style smooth rise/fall window on a periodic domain.

    The rise and fall each occupy half of the fringe interval.  The window and
    all of its derivatives vanish at both ends, so applying it next to the
    periodic seam does not introduce a forcing discontinuity.
    """
    start_array = jnp.asarray(start, dtype=x.dtype)
    end_array = jnp.asarray(end, dtype=x.dtype)
    half_width = 0.5 * (end_array - start_array)
    rise = _cinf_step((x - start_array) / half_width)
    fall = _cinf_step((end_array - x) / half_width)
    return rise * fall


def actuator_disk_kernel(params: Params) -> jax.Array:
    """Return the annular actuator indicator times a unit-integral x kernel.

    The x kernel has units 1/m.  Consequently the volume integral of this
    array is the effective loaded rotor area.
    """
    x, y, z = _cell_coordinates(params)
    lx = params.lx * params.z_i
    ly = params.ly * params.z_i
    dx = _periodic_offset(x, params.actuator_disk_x, lx)
    dy = _periodic_offset(y, params.actuator_disk_y, ly)
    yaw = jnp.deg2rad(
        jnp.asarray(params.actuator_disk_yaw_degrees, dtype=params.dtype)
    )
    cos_yaw = jnp.cos(yaw)
    sin_yaw = jnp.sin(yaw)
    normal_distance = dx * cos_yaw + dy * sin_yaw
    in_plane_distance = -dx * sin_yaw + dy * cos_yaw
    radius = jnp.sqrt(in_plane_distance**2 + (z - params.actuator_disk_z) ** 2)

    sigma_x = jnp.asarray(
        max(params.actuator_disk_thickness, 1.5 * params.dx * params.z_i),
        dtype=params.dtype,
    )
    streamwise = jnp.exp(-0.5 * (normal_distance / sigma_x) ** 2)
    cell_dx = params.dx * params.z_i
    streamwise = streamwise / jnp.maximum(
        jnp.sum(streamwise[:, 0, 0]) * cell_dx,
        jnp.asarray(jnp.finfo(params.dtype).tiny, dtype=params.dtype),
    )

    edge_width = max(
        0.5 * (params.dy * params.z_i + params.dz * params.z_i),
        0.25 * params.actuator_disk_thickness,
    )
    outer = 0.5 * (1.0 - jnp.tanh((radius - 0.5 * params.actuator_disk_diameter) / edge_width))
    if params.actuator_disk_hub_diameter > 0.0:
        inner = 0.5 * (
            1.0
            + jnp.tanh((radius - 0.5 * params.actuator_disk_hub_diameter) / edge_width)
        )
        outer = outer * inner
    return (streamwise * outer).astype(params.dtype)


def cold_source_kernel(params: Params) -> jax.Array:
    """Return a discrete unit-volume-integral Gaussian source kernel [1/m^3]."""
    x, y, z = _cell_coordinates(params)
    lx = params.lx * params.z_i
    ly = params.ly * params.z_i
    dx = _periodic_distance(x, params.cold_source_x, lx)
    dy = _periodic_distance(y, params.cold_source_y, ly)
    radial2 = dy * dy + (z - params.cold_source_z) ** 2
    sigma_x = max(params.cold_source_sigma_x, 1.5 * params.dx * params.z_i)
    sigma_r = max(
        params.cold_source_sigma_r,
        1.5 * max(params.dy, params.dz) * params.z_i,
    )
    kernel = jnp.exp(
        -0.5 * (dx / sigma_x) ** 2
        -0.5 * radial2 / (sigma_r**2)
    )
    cell_volume = (
        params.dx * params.dy * params.dz * params.z_i**3
    )
    normalization = jnp.sum(kernel) * cell_volume
    tiny = jnp.asarray(jnp.finfo(params.dtype).tiny, dtype=params.dtype)
    return (kernel / jnp.maximum(normalization, tiny)).astype(params.dtype)


def fringe_mask(params: Params) -> jax.Array:
    """Classic smooth downstream forcing window for a periodic spatial LES."""
    x, _, _ = _cell_coordinates(params)
    domain_x = params.lx * params.z_i
    return classic_fringe_window(x, params.fringe_start_x, domain_x)


def wind_tunnel_momentum_sources(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Actuator-disk, hub-jet and fringe accelerations in solver units."""
    source_u = jnp.zeros_like(u)
    source_v = jnp.zeros_like(v)
    source_w = jnp.zeros_like(w)

    if params.actuator_disk_enabled:
        disk = actuator_disk_kernel(params)
        yaw = jnp.deg2rad(
            jnp.asarray(params.actuator_disk_yaw_degrees, dtype=params.dtype)
        )
        normal_x = jnp.cos(yaw)
        normal_y = jnp.sin(yaw)
        weight_sum = jnp.sum(disk)
        normal_velocity = u * normal_x + v * normal_y
        disk_velocity = jnp.sum(normal_velocity * disk) / jnp.maximum(
            weight_sum, jnp.asarray(jnp.finfo(u.dtype).tiny, dtype=u.dtype)
        )
        disk_acceleration = (
            -0.5
            * params.actuator_disk_ct_prime
            * disk_velocity
            * jnp.abs(disk_velocity)
            * disk
        )
        source_u = source_u + params.z_i * disk_acceleration * normal_x
        source_v = source_v + params.z_i * disk_acceleration * normal_y

    if params.cold_source_enabled and params.cold_source_momentum_flux > 0.0:
        kernel = cold_source_kernel(params)
        acceleration = (
            params.cold_source_momentum_flux / params.cold_source_density
        ) * kernel
        source_u = source_u + params.z_i * acceleration

    if params.fringe_enabled:
        rate = params.z_i / params.fringe_timescale
        mask = fringe_mask(params)
        source_u = source_u + rate * mask * (params.fringe_target_u - u)
        source_v = source_v + rate * mask * (params.fringe_target_v - v)
        source_w = source_w - rate * mask * w
    return source_u, source_v, source_w


def wind_tunnel_scalar_sources(
    theta: jax.Array,
    qv: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array]:
    """Effective LN2 cooling and fringe scalar sources in solver units."""
    source_theta = jnp.zeros_like(theta)
    source_qv = jnp.zeros_like(qv)
    if params.cold_source_enabled and params.cold_source_cooling_power > 0.0:
        kernel = cold_source_kernel(params)
        cooling_rate = (
            params.cold_source_cooling_power
            / (params.cold_source_density * params.cold_source_heat_capacity)
        ) * kernel
        source_theta = source_theta - params.z_i * cooling_rate
    if params.fringe_enabled:
        rate = params.z_i / params.fringe_timescale
        mask = fringe_mask(params)
        target_theta = (
            params.theta0
            if params.fringe_target_theta is None
            else params.fringe_target_theta
        )
        source_theta = source_theta + rate * mask * (target_theta - theta)
        source_qv = source_qv + rate * mask * (params.qv0 - qv)
    return source_theta, source_qv
