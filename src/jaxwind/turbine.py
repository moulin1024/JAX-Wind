"""Adapters from the shared wind-turbine kernels to the FV MAC mesh."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jaxwind._jax.wind import (
    build_actuator_line_kernel,
    build_blade_element_disk_kernel,
    build_nacelle_tower_kernel,
)
from jaxwind.domain.grid import Grid
from jaxwind.physics import (
    BladeElementActuatorDisk,
    BladeElementActuatorLine,
    NacelleTowerDrag,
)

from jaxwind.numerics.discretization import cell_velocity
from .state import StaggeredVelocity


def _x_faces(values: jnp.ndarray, grid: Grid, *, periodic=False) -> jnp.ndarray:
    """Conservatively average cell forcing onto open or periodic x faces."""
    widths = jnp.asarray(grid.x_widths, dtype=values.dtype)
    if periodic:
        lower_widths = jnp.roll(widths, 1)
        return (
            jnp.roll(values, 1, axis=2) * widths[None, None, :]
            + values * lower_widths[None, None, :]
        ) / (lower_widths + widths)[None, None, :]
    total = widths[:-1] + widths[1:]
    interior = (
        values[..., :-1] * widths[:-1][None, None, :]
        + values[..., 1:] * widths[1:][None, None, :]
    ) / total[None, None, :]
    return jnp.concatenate((values[..., :1], interior, values[..., -1:]), axis=2)


def _y_faces(values: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Conservatively average cell forcing onto physical periodic-y faces."""
    upper_widths = jnp.asarray(grid.y_widths, dtype=values.dtype)
    lower_widths = jnp.roll(upper_widths, 1)
    total = lower_widths + upper_widths
    return (
        jnp.roll(values, 1, axis=1) * lower_widths[None, :, None]
        + values * upper_widths[None, :, None]
    ) / total[None, :, None]


def build_adbem_forcing(
    grid: Grid,
    disk: BladeElementActuatorDisk,
    body: NacelleTowerDrag | None = None,
) -> Callable[[StaggeredVelocity, jnp.ndarray], StaggeredVelocity]:
    """Build single-device DTU-style AD-BEM forcing for an open FV domain.

    The aerodynamic calculation reuses the shared annular turbine kernel.
    Velocities are sampled at cell centres and upper z faces;
    the resulting accelerations are conservatively centred back onto the MAC
    component faces before entering the FV momentum tendency.
    """
    disk_kernel = build_blade_element_disk_kernel(
        grid=grid,
        axis_name="fv_adbem",
        partition_count=1,
    )

    def disk_local(u, v, w_upper):
        dtype = u.dtype
        return disk_kernel(
            u,
            v,
            w_upper,
            disk.x,
            disk.y,
            disk.z,
            disk.blade_count,
            disk.hub_radius,
            disk.tip_radius,
            disk.angular_velocity,
            jnp.asarray(disk.element_smoothing_widths, dtype=dtype),
            jnp.asarray(disk.element_radii, dtype=dtype),
            jnp.asarray(disk.element_widths, dtype=dtype),
            jnp.asarray(disk.element_chords, dtype=dtype),
            jnp.asarray(disk.element_twist_degrees, dtype=dtype),
            jnp.asarray(disk.element_airfoil_ids, dtype=jnp.int32),
            jnp.asarray(disk.polar_alpha_degrees, dtype=dtype),
            jnp.asarray(disk.polar_lift_coefficients, dtype=dtype),
            jnp.asarray(disk.polar_drag_coefficients, dtype=dtype),
            disk.pitch_degrees,
            disk.tip_loss,
            disk.root_loss,
        )

    disk_mapped = jax.pmap(disk_local, axis_name="fv_adbem")
    body_mapped = None
    if body is not None:
        body_kernel = build_nacelle_tower_kernel(
            grid=grid,
            axis_name="fv_turbine_body",
        )

        def body_local(u, v):
            return body_kernel(
                u,
                v,
                body.x,
                body.y,
                body.hub_height,
                body.nacelle_length,
                body.nacelle_diameter,
                body.nacelle_drag_coefficient,
                body.tower_base_diameter,
                body.tower_top_diameter,
                body.tower_drag_coefficient,
                body.smoothing_width,
            )

        body_mapped = jax.pmap(body_local, axis_name="fv_turbine_body")

    def forcing(
        velocity: StaggeredVelocity,
        _time: jnp.ndarray,
    ) -> StaggeredVelocity:
        u, v, _w = cell_velocity(velocity)
        w_upper = velocity.z[1:]
        disk_values = disk_mapped(u[None], v[None], w_upper[None])
        source_x = disk_values[0][0]
        source_y = disk_values[1][0]
        source_z_upper = disk_values[2][0]
        if body_mapped is not None:
            body_values = body_mapped(u[None], v[None])
            source_x = source_x + body_values[0][0]
            source_y = source_y + body_values[1][0]
            source_z_upper = source_z_upper + body_values[2][0]
        wall = jnp.zeros_like(source_z_upper[:1])
        return StaggeredVelocity(
            _x_faces(source_x, grid, periodic=velocity.x.shape[-1] == grid.nx),
            _y_faces(source_y, grid),
            jnp.concatenate((wall, source_z_upper), axis=0),
        )

    return forcing


def build_actuator_line_forcing(
    grid: Grid,
    line: BladeElementActuatorLine,
    body: NacelleTowerDrag | None = None,
) -> Callable[[StaggeredVelocity, jnp.ndarray], StaggeredVelocity]:
    """Build rotating blade-element actuator-line forcing on an FV MAC mesh.

    The shared aerodynamic kernel samples the physical cell volumes and
    spreads every instantaneous blade-element load conservatively. The
    resulting cell-centred accelerations are then centred onto the MAC
    component faces, matching :func:`build_adbem_forcing`.
    """
    line_kernel = build_actuator_line_kernel(
        grid=grid,
        axis_name="fv_actuator_line",
        partition_count=1,
    )

    def line_local(u, v, w_upper, w_lower, time):
        dtype = u.dtype
        point_count = line.blade_count * len(line.element_radii)

        def deformation(values):
            return (
                jnp.asarray(values, dtype=dtype)
                if values
                else jnp.zeros((point_count,), dtype=dtype)
            )

        return line_kernel(
            u,
            v,
            w_upper,
            w_lower,
            time,
            line.x,
            line.y,
            line.z,
            line.blade_count,
            line.hub_radius,
            line.tip_radius,
            line.angular_velocity,
            jnp.asarray(line.point_smoothing_widths, dtype=dtype),
            jnp.asarray(line.element_radii, dtype=dtype),
            jnp.asarray(line.element_widths, dtype=dtype),
            jnp.asarray(line.element_chords, dtype=dtype),
            jnp.asarray(line.element_twist_degrees, dtype=dtype),
            jnp.asarray(line.element_airfoil_ids, dtype=jnp.int32),
            jnp.asarray(line.polar_alpha_degrees, dtype=dtype),
            jnp.asarray(line.polar_lift_coefficients, dtype=dtype),
            jnp.asarray(line.polar_drag_coefficients, dtype=dtype),
            line.pitch_degrees,
            line.yaw_degrees,
            line.tilt_degrees,
            line.precone_degrees,
            line.initial_azimuth_degrees,
            line.tip_loss,
            line.root_loss,
            deformation(line.element_flap_displacements),
            deformation(line.element_edge_displacements),
            deformation(line.element_flap_slopes),
            deformation(line.element_edge_slopes),
            deformation(line.element_flap_velocities),
            deformation(line.element_edge_velocities),
        )

    line_mapped = jax.pmap(line_local, axis_name="fv_actuator_line")
    body_mapped = None
    if body is not None:
        body_kernel = build_nacelle_tower_kernel(
            grid=grid,
            axis_name="fv_actuator_line_body",
        )

        def body_local(u, v):
            return body_kernel(
                u,
                v,
                body.x,
                body.y,
                body.hub_height,
                body.nacelle_length,
                body.nacelle_diameter,
                body.nacelle_drag_coefficient,
                body.tower_base_diameter,
                body.tower_top_diameter,
                body.tower_drag_coefficient,
                body.smoothing_width,
            )

        body_mapped = jax.pmap(body_local, axis_name="fv_actuator_line_body")

    def forcing(
        velocity: StaggeredVelocity,
        time: jnp.ndarray,
    ) -> StaggeredVelocity:
        u, v, _w = cell_velocity(velocity)
        values = line_mapped(
            u[None],
            v[None],
            velocity.z[1:][None],
            velocity.z[0][None],
            jnp.asarray(time)[None],
        )
        source_x = values[0][0]
        source_y = values[1][0]
        source_z_upper = values[2][0]
        if body_mapped is not None:
            body_values = body_mapped(u[None], v[None])
            source_x = source_x + body_values[0][0]
            source_y = source_y + body_values[1][0]
            source_z_upper = source_z_upper + body_values[2][0]
        wall = jnp.zeros_like(source_z_upper[:1])
        return StaggeredVelocity(
            _x_faces(source_x, grid),
            _y_faces(source_y, grid),
            jnp.concatenate((wall, source_z_upper), axis=0),
        )

    return forcing


__all__ = ["build_adbem_forcing", "build_actuator_line_forcing"]
