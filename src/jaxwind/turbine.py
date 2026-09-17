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
    PureThrustActuatorDisk,
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
            jnp.roll(values, 1, axis=2) * lower_widths[None, None, :]
            + values * widths[None, None, :]
        ) / (lower_widths + widths)[None, None, :]
    total = widths[:-1] + widths[1:]
    interior = (
        values[..., :-1] * widths[:-1][None, None, :]
        + values[..., 1:] * widths[1:][None, None, :]
    ) / total[None, None, :]
    return jnp.concatenate((values[..., :1], interior, values[..., -1:]), axis=2)


def _y_faces(values: jnp.ndarray, grid: Grid, *, periodic=True) -> jnp.ndarray:
    """Conservatively average cell forcing onto physical periodic-y faces."""
    upper_widths = jnp.asarray(grid.y_widths, dtype=values.dtype)
    if not periodic:
        interior = (values[:, :-1] * upper_widths[None, :-1, None]
                    + values[:, 1:] * upper_widths[None, 1:, None]) / (upper_widths[:-1] + upper_widths[1:])[None, :, None]
        return jnp.concatenate((values[:, :1], interior, values[:, -1:]), axis=1)
    lower_widths = jnp.roll(upper_widths, 1)
    total = lower_widths + upper_widths
    return (
        jnp.roll(values, 1, axis=1) * lower_widths[None, :, None]
        + values * upper_widths[None, :, None]
    ) / total[None, :, None]



def build_thrust_only_adm_forcing(grid: Grid, disk: PureThrustActuatorDisk, *, periodic_y: bool = True):
    """Prescribed freestream thrust-only ADM with discretely normalized Gaussian smearing."""
    if disk.prescribed_inflow_velocity <= 0.0 or disk.prescribed_thrust_coefficient <= 0.0:
        raise ValueError("thrust-only ADM requires prescribed inflow and thrust coefficient")
    x = jnp.asarray(grid.x_centers); y = jnp.asarray(grid.y_centers); z = jnp.asarray(grid.z_centers)
    zz, yy, xx = jnp.meshgrid(z, y, x, indexing="ij")
    epsn = disk.normal_smoothing_width; epst = disk.transverse_smoothing_width
    dx = xx - disk.x
    dy = yy - disk.y
    if periodic_y:
        dy = (dy + 0.5 * grid.ly) % grid.ly - 0.5 * grid.ly
    dz = zz - disk.z
    radial = jnp.sqrt(dy * dy + dz * dz)
    # Geometric disk convolved with a Gaussian transverse kernel.
    smooth_indicator = 0.5 * (1.0 - jnp.tanh((radial - 0.5 * disk.diameter) / (0.5 * epst)))
    weights = jnp.exp(-(dx / epsn) ** 2) * smooth_indicator
    volumes = (jnp.asarray(grid.z_widths)[:, None, None] * jnp.asarray(grid.y_widths)[None, :, None] * jnp.asarray(grid.x_widths)[None, None, :])
    weights = weights / jnp.sum(weights * volumes)
    integrated = 0.5 * disk.prescribed_thrust_coefficient * disk.prescribed_inflow_velocity ** 2 * jnp.pi * (0.5 * disk.diameter) ** 2
    cell_source = -integrated * weights
    def forcing(velocity, _time):
        source_x = _x_faces(cell_source, grid, periodic=False)
        return StaggeredVelocity(source_x, jnp.zeros_like(velocity.y), jnp.zeros_like(velocity.z))
    return forcing

def build_adbem_forcing(
    grid: Grid,
    disk: BladeElementActuatorDisk,
    body: NacelleTowerDrag | None = None,
    *, periodic_x: bool = True, periodic_y: bool = True,
    minimum_normal_smoothing_width: float = 0.0,
    momentum_stabilization_coefficient: float = 0.0,
) -> Callable[[StaggeredVelocity, jnp.ndarray], StaggeredVelocity]:
    """Build single-device AD-BEM forcing for an open or periodic FV domain.

    The aerodynamic calculation reuses the shared annular turbine kernel.
    Velocities are sampled at cell centres and upper z faces;
    the resulting accelerations are conservatively centred back onto the MAC
    component faces before entering the FV momentum tendency.
    """
    import math
    if not math.isfinite(minimum_normal_smoothing_width) or minimum_normal_smoothing_width < 0.:
        raise ValueError("minimum_normal_smoothing_width must be finite and nonnegative")
    if not math.isfinite(momentum_stabilization_coefficient) or not 0. <= momentum_stabilization_coefficient <= 1./16.:
        raise ValueError("momentum_stabilization_coefficient must be between 0 and 1/16")
    if momentum_stabilization_coefficient and not grid.is_uniform:
        raise ValueError("actuator momentum stabilization requires a uniform mesh")
    disk_kernel = build_blade_element_disk_kernel(
        grid=grid,
        minimum_normal_smoothing_width=minimum_normal_smoothing_width,
        axis_name="fv_adbem",
        partition_count=1,
        periodic_x=periodic_x,
        periodic_y=periodic_y,
    )

    def disk_local(u, v, w_upper, position, angular_velocity):
        dtype = u.dtype
        return disk_kernel(
            u,
            v,
            w_upper,
            position[0],
            position[1],
            position[2],
            disk.blade_count,
            disk.hub_radius,
            disk.tip_radius,
            angular_velocity,
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

    disk_mapped = jax.pmap(disk_local, axis_name="fv_adbem", in_axes=(0, 0, 0, None, None))
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
        *, position=None, angular_velocity=None,
    ) -> StaggeredVelocity:
        if position is not None and body_mapped is not None:
            raise ValueError("dynamic turbine positions require body drag to be disabled")
        u, v, _w = cell_velocity(velocity)
        w_upper = velocity.z[1:]
        if position is None:
            position = jnp.asarray((disk.x, disk.y, disk.z), u.dtype)
        if angular_velocity is None:
            angular_velocity = disk.angular_velocity
        disk_values = disk_mapped(u[None], v[None], w_upper[None], position, angular_velocity)
        source_x = disk_values[0][0]
        source_y = disk_values[1][0]
        source_z_upper = disk_values[2][0]
        if body_mapped is not None:
            body_values = body_mapped(u[None], v[None])
            source_x = source_x + body_values[0][0]
            source_y = source_y + body_values[1][0]
            source_z_upper = source_z_upper + body_values[2][0]
        wall = jnp.zeros_like(source_z_upper[:1])
        result = StaggeredVelocity(
            _x_faces(source_x, grid, periodic=velocity.x.shape[-1] == grid.nx),
            _y_faces(source_y, grid, periodic=velocity.y.shape[1] == grid.ny),
            jnp.concatenate((wall, source_z_upper), axis=0),
        )

        if momentum_stabilization_coefficient:
            from jaxwind.numerics.actuator_stabilization import actuator_momentum_stabilization
            damping = actuator_momentum_stabilization(
                velocity, grid, position, disk.tip_radius, minimum_normal_smoothing_width,
                momentum_stabilization_coefficient, periodic_x=periodic_x, periodic_y=periodic_y)
            result = StaggeredVelocity(*(a+b for a,b in zip(result,damping)))
        return result

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


__all__ = ["build_adbem_forcing", "build_actuator_line_forcing", "build_thrust_only_adm_forcing"]
