"""Fully conservative Morinishi S4 transport on a uniform MAC grid.

This module implements the divergence form in Morinishi et al., JCP 143
(1998), Eq. (101).  The one- and three-mesh fluxes are combined with the
fourth-order weights 9/8 and -1/8.  Convecting velocities are interpolated to
the transported component's staggered control-volume faces before forming
the products; transported velocities retain the one- or three-mesh average
belonging to that flux.

Periodic directions use the uniform-grid identities exactly.  At the rigid
wall the tangential-velocity ghosts, normal-velocity ghosts, and three-mesh
momentum fluxes implement Eqs. (146)--(151), specialized to zero resolved wall
velocity and a uniform wall-normal mesh.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxwind.pressure.mac_projection import MACVelocity


Array = jax.Array
_ALPHA_ONE = 9.0 / 8.0
_ALPHA_THREE = -1.0 / 8.0


def _periodic_center_to_face_s4(field: Array, axis: int) -> Array:
    """Interpolate cell-centred values to periodic faces with S4 accuracy."""
    one = 0.5 * (jnp.roll(field, 1, axis=axis) + field)
    three = 0.5 * (jnp.roll(field, 2, axis=axis) + jnp.roll(field, -1, axis=axis))
    return _ALPHA_ONE * one + _ALPHA_THREE * three


def _periodic_face_divergence_s4(
    field: Array,
    spacing: float,
    axis: int,
) -> Array:
    """Apply the S4 face-to-cell continuity derivative."""
    return (
        jnp.roll(field, 1, axis=axis)
        - 27.0 * field
        + 27.0 * jnp.roll(field, -1, axis=axis)
        - jnp.roll(field, -2, axis=axis)
    ) / (24.0 * spacing)


def _tangential_wall_ghosts(field: Array) -> tuple[Array, Array, Array, Array]:
    """Return the two lower and upper ghosts from Morinishi Eq. (150)."""
    if field.shape[0] < 2:
        raise ValueError("Morinishi S4 requires at least two wall-normal cells")
    lower_adjacent = -2.0 * field[0] + field[1] / 3.0
    lower_outer = -9.0 * field[0] + 2.0 * field[1]
    upper_adjacent = -2.0 * field[-1] + field[-2] / 3.0
    upper_outer = -9.0 * field[-1] + 2.0 * field[-2]
    return lower_outer, lower_adjacent, upper_adjacent, upper_outer


def _pad_tangential_wall(field: Array) -> Array:
    lower_outer, lower_adjacent, upper_adjacent, upper_outer = _tangential_wall_ghosts(
        field
    )
    return jnp.concatenate(
        (
            lower_outer[None],
            lower_adjacent[None],
            field,
            upper_adjacent[None],
            upper_outer[None],
        ),
        axis=0,
    )


def _wall_center_to_face_s4(field: Array) -> Array:
    """Interpolate tangential cell values to wall-normal faces."""
    padded = _pad_tangential_wall(field)
    count = field.shape[0]
    one = 0.5 * (padded[1 : count + 2] + padded[2 : count + 3])
    three = 0.5 * (padded[: count + 1] + padded[3 : count + 4])
    return _ALPHA_ONE * one + _ALPHA_THREE * three


def _horizontal_self_transport(
    velocity: Array,
    spacing: float,
    axis: int,
) -> Array:
    """Return one S4 self-advection contribution on periodic faces."""
    transported_one = 0.5 * (velocity + jnp.roll(velocity, -1, axis=axis))
    transported_three = 0.5 * (
        jnp.roll(velocity, 1, axis=axis) + jnp.roll(velocity, -2, axis=axis)
    )
    convecting = _ALPHA_ONE * transported_one + _ALPHA_THREE * transported_three
    flux_one = convecting * transported_one
    flux_three = convecting * transported_three
    derivative_one = (flux_one - jnp.roll(flux_one, 1, axis=axis)) / spacing
    derivative_three = (
        jnp.roll(flux_three, -1, axis=axis) - jnp.roll(flux_three, 2, axis=axis)
    ) / (3.0 * spacing)
    return _ALPHA_ONE * derivative_one + _ALPHA_THREE * derivative_three


def _horizontal_cross_transport(
    transported: Array,
    convecting: Array,
    spacing: float,
    axis: int,
) -> Array:
    """Return one S4 cross-advection contribution on a periodic axis."""
    transported_one = 0.5 * (jnp.roll(transported, 1, axis=axis) + transported)
    transported_three = 0.5 * (
        jnp.roll(transported, 2, axis=axis) + jnp.roll(transported, -1, axis=axis)
    )
    flux_one = convecting * transported_one
    flux_three = convecting * transported_three
    derivative_one = (jnp.roll(flux_one, -1, axis=axis) - flux_one) / spacing
    derivative_three = (
        jnp.roll(flux_three, -2, axis=axis) - jnp.roll(flux_three, 1, axis=axis)
    ) / (3.0 * spacing)
    return _ALPHA_ONE * derivative_one + _ALPHA_THREE * derivative_three


def _vertical_tangential_transport(
    transported: Array,
    convecting: Array,
    spacing: float,
) -> Array:
    """Transport tangential momentum through wall-normal faces."""
    padded = _pad_tangential_wall(transported)
    count = transported.shape[0]
    transported_one = 0.5 * (padded[1 : count + 2] + padded[2 : count + 3])
    transported_three = 0.5 * (padded[: count + 1] + padded[3 : count + 4])
    flux_one = convecting * transported_one
    flux_three = convecting * transported_three

    # Eq. (147), with zero resolved normal and tangential wall velocities.
    lower_three = 27.0 * flux_one[0] - flux_three[0] - flux_three[1]
    upper_three = 27.0 * flux_one[-1] - flux_three[-2] - flux_three[-1]
    extended_three = jnp.concatenate(
        (lower_three[None], flux_three, upper_three[None]),
        axis=0,
    )
    derivative_one = (flux_one[1:] - flux_one[:-1]) / spacing
    derivative_three = (extended_three[3 : count + 3] - extended_three[:count]) / (
        3.0 * spacing
    )
    return _ALPHA_ONE * derivative_one + _ALPHA_THREE * derivative_three


def _normal_wall_extension(
    normal_velocity: Array,
    u: Array,
    v: Array,
    dx: float,
    dy: float,
    dz: float,
) -> Array:
    """Build normal-velocity ghosts satisfying Eqs. (146) and (151)."""
    normal = normal_velocity.at[0].set(0.0).at[-1].set(0.0)
    lower_u_outer, lower_u, upper_u, upper_u_outer = _tangential_wall_ghosts(u)
    lower_v_outer, lower_v, upper_v, upper_v_outer = _tangential_wall_ghosts(v)
    del lower_u_outer, upper_u_outer, lower_v_outer, upper_v_outer
    lower_horizontal_divergence = _periodic_face_divergence_s4(
        lower_u,
        dx,
        -1,
    ) + _periodic_face_divergence_s4(lower_v, dy, -2)
    upper_horizontal_divergence = _periodic_face_divergence_s4(
        upper_u,
        dx,
        -1,
    ) + _periodic_face_divergence_s4(upper_v, dy, -2)

    lower_adjacent = 2.0 * normal[0] - normal[1]
    upper_adjacent = 2.0 * normal[-1] - normal[-2]
    lower_outer = (
        27.0 * normal[0] - 26.0 * normal[1] - 24.0 * dz * lower_horizontal_divergence
    )
    upper_outer = (
        27.0 * normal[-1] - 26.0 * normal[-2] + 24.0 * dz * upper_horizontal_divergence
    )
    return jnp.concatenate(
        (
            lower_outer[None],
            lower_adjacent[None],
            normal,
            upper_adjacent[None],
            upper_outer[None],
        ),
        axis=0,
    )


def _vertical_self_transport(
    normal_velocity: Array,
    u: Array,
    v: Array,
    dx: float,
    dy: float,
    dz: float,
) -> Array:
    """Return the wall-normal S4 self-advection on physical MAC faces."""
    extended = _normal_wall_extension(normal_velocity, u, v, dx, dy, dz)
    count = normal_velocity.shape[0] - 1
    transported_one = 0.5 * (extended[1 : count + 3] + extended[2 : count + 4])
    transported_three = 0.5 * (extended[: count + 2] + extended[3 : count + 5])
    convecting = _ALPHA_ONE * transported_one + _ALPHA_THREE * transported_three
    flux_one = convecting * transported_one
    flux_three = convecting * transported_three
    derivative_one = (flux_one[2 : count + 1] - flux_one[1:count]) / dz
    derivative_three = (flux_three[3 : count + 2] - flux_three[: count - 1]) / (
        3.0 * dz
    )
    interior = _ALPHA_ONE * derivative_one + _ALPHA_THREE * derivative_three
    result = jnp.zeros_like(normal_velocity)
    return result.at[1:-1].set(interior)


def _unique_periodic_faces(velocity: MACVelocity) -> tuple[Array, Array]:
    # The final periodic face duplicates the first.  Merely drop it here:
    # averaging the two would corrupt the outermost exchanged halo when the
    # same kernel is evaluated on a padded y slab.
    return velocity.x[..., :-1], velocity.y[:, :-1, :]


def morinishi_s4_advection(
    velocity: MACVelocity,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> MACVelocity:
    """Return ``-Div-S4`` directly on the three native MAC velocity grids."""
    u, v = _unique_periodic_faces(velocity)
    w = velocity.z.at[0].set(0.0).at[-1].set(0.0)
    nz, ny, nx = u.shape
    if v.shape != (nz, ny, nx) or w.shape != (nz + 1, ny, nx):
        raise ValueError("velocity shapes are inconsistent with a MAC grid")
    if min(nx, ny) < 4 or nz < 2:
        raise ValueError(
            "Morinishi S4 requires four periodic cells and two wall-normal cells"
        )

    # u momentum: x self flux, y cross flux, and rigid-wall z flux.
    u_convection_y = _periodic_center_to_face_s4(v, -1)
    u_convection_z = _periodic_center_to_face_s4(w, -1)
    u_transport = _horizontal_self_transport(u, dx, -1)
    u_transport += _horizontal_cross_transport(u, u_convection_y, dy, -2)
    u_transport += _vertical_tangential_transport(u, u_convection_z, dz)

    # v momentum: x cross flux, y self flux, and rigid-wall z flux.
    v_convection_x = _periodic_center_to_face_s4(u, -2)
    v_convection_z = _periodic_center_to_face_s4(w, -2)
    v_transport = _horizontal_cross_transport(v, v_convection_x, dx, -1)
    v_transport += _horizontal_self_transport(v, dy, -2)
    v_transport += _vertical_tangential_transport(v, v_convection_z, dz)

    # w momentum: horizontal cross fluxes need the Eq. (150) z interpolation.
    w_convection_x = _wall_center_to_face_s4(u)
    w_convection_y = _wall_center_to_face_s4(v)
    w_transport = _horizontal_cross_transport(w, w_convection_x, dx, -1)
    w_transport += _horizontal_cross_transport(w, w_convection_y, dy, -2)
    w_transport += _vertical_self_transport(w, u, v, dx, dy, dz)

    u_tendency = -u_transport
    v_tendency = -v_transport
    w_tendency = -w_transport
    return MACVelocity(
        jnp.concatenate((u_tendency, u_tendency[..., :1]), axis=-1),
        jnp.concatenate((v_tendency, v_tendency[:, :1, :]), axis=1),
        w_tendency.at[0].set(0.0).at[-1].set(0.0),
    )


def staggered_kinetic_energy_work(
    velocity: MACVelocity,
    tendency: MACVelocity,
) -> Array:
    """Return the unweighted MAC-grid velocity/tendency inner product."""
    u, v = _unique_periodic_faces(velocity)
    tu, tv = _unique_periodic_faces(tendency)
    return jnp.sum(u * tu) + jnp.sum(v * tv) + jnp.sum(velocity.z * tendency.z)


def staggered_momentum(tendency: MACVelocity) -> Array:
    """Return componentwise sums over unique prognostic MAC locations."""
    u, v = _unique_periodic_faces(tendency)
    return jnp.stack((jnp.sum(u), jnp.sum(v), jnp.sum(tendency.z)))


__all__ = [
    "morinishi_s4_advection",
    "staggered_kinetic_energy_work",
    "staggered_momentum",
]
