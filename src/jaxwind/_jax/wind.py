"""Blade-element actuator-disk, actuator-line, and body JAX kernels."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from jaxwind._jax.actuator_line import (
    actuator_line_deformed_kinematics,
    blade_element_kinematic_forces,
    gaussian_weights,
)


def _local_z_metrics(grid, dtype, local_nz, partition_index):
    """Return physical coordinates and control widths for one z slab."""
    start = partition_index * local_nz

    def local(values):
        return lax.dynamic_slice_in_dim(
            jnp.asarray(values, dtype=dtype), start, local_nz
        )

    z_widths = jnp.asarray(grid.z_widths, dtype=dtype)
    upper_face_widths = jnp.concatenate(
        (0.5 * (z_widths[:-1] + z_widths[1:]), 0.5 * z_widths[-1:])
    )
    return (
        local(grid.z_centers),
        local(grid.z_faces[1:]),
        local(grid.z_widths),
        local(upper_face_widths),
    )



def build_blade_element_disk_kernel(
    *,
    grid,
    axis_name: str,
    partition_count: int,
    periodic_x: bool = True,
    periodic_y: bool = True,
    minimum_normal_smoothing_width: float = 0.0,
):
    """Build an annular AD-BEM kernel for an upright, streamwise rotor."""

    def actuator_disk_bem_local(
        u,
        v,
        w_upper,
        disk_x,
        disk_y,
        disk_z,
        blade_count,
        hub_radius,
        tip_radius,
        angular_velocity,
        element_smoothing_widths,
        element_radii,
        element_widths,
        element_chords,
        element_twist_degrees,
        element_airfoil_ids,
        polar_alpha_degrees,
        polar_lift_coefficients,
        polar_drag_coefficients,
        pitch_degrees,
        tip_loss,
        root_loss,
    ):
        dtype = u.dtype
        local_nz = u.shape[0]
        partition_index = lax.axis_index(axis_name)
        tiny = jnp.finfo(dtype).tiny
        widths = jnp.asarray(element_smoothing_widths, dtype)

        x = jnp.asarray(grid.x_centers, dtype=dtype)
        y = jnp.asarray(grid.y_centers, dtype=dtype)
        z_cell, z_upper, z_widths, z_upper_widths = _local_z_metrics(
            grid, dtype, local_nz, partition_index
        )
        x_widths = jnp.asarray(grid.x_widths, dtype=dtype)
        y_widths = jnp.asarray(grid.y_widths, dtype=dtype)
        dx = jnp.mod(x - disk_x + 0.5 * grid.lx, grid.lx) - 0.5 * grid.lx
        dy = jnp.mod(y - disk_y + 0.5 * grid.ly, grid.ly) - 0.5 * grid.ly
        if not periodic_x:
            dx = x - disk_x
        if not periodic_y:
            dy = y - disk_y

        # Use the same normalized kernel for velocity sampling and spreading.
        # The optional x-only floor removes underresolved disk-normal forcing
        # without broadening annular loading in the rotor plane.
        normal_widths = (jnp.maximum(widths, minimum_normal_smoothing_width)
                         if minimum_normal_smoothing_width else widths)
        # A common log shift preserves the normalized Gaussian even when
        # every unshifted weight underflows on a coarse grid.
        log_x = -(dx[None, :] / normal_widths[:, None]) ** 2
        raw_x = jnp.exp(log_x - jnp.max(log_x, axis=1, keepdims=True))
        weighted_x = raw_x * x_widths[None, :]
        weights_x = weighted_x / jnp.maximum(
            jnp.sum(weighted_x, axis=1, keepdims=True), tiny
        )

        def ring_geometry(z_coordinates, transverse_areas):
            yy = dy[None, None, :]
            zz = z_coordinates[None, :, None] - jnp.asarray(disk_z, dtype)
            radius = jnp.sqrt(yy * yy + zz * zz)
            log_raw = -(
                (radius - element_radii[:, None, None])
                / widths[:, None, None]
            ) ** 2
            # Excluded faces have zero area and must not set the shift.
            # All z slabs must use the same shift before the global sum.
            log_raw = jnp.where(
                transverse_areas[None, :, :] > 0.0, log_raw, -jnp.inf
            )
            maximum = lax.pmax(
                jnp.max(log_raw, axis=(1, 2), keepdims=True), axis_name
            )
            maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
            raw = jnp.exp(log_raw - maximum)
            weighted = raw * transverse_areas[None, :, :]
            denominator = lax.psum(
                jnp.sum(weighted, axis=(1, 2)), axis_name
            )
            weights = weighted / jnp.maximum(
                denominator[:, None, None], tiny
            )
            # Positive rotation points along +y at the top of the rotor.
            tangent_y = zz / jnp.maximum(radius, tiny)
            tangent_z = -yy / jnp.maximum(radius, tiny)
            return weights, tangent_y[0], tangent_z[0]

        cell_areas = z_widths[:, None] * y_widths[None, :]
        upper_areas = z_upper_widths[:, None] * y_widths[None, :]
        global_upper = partition_index * local_nz + jnp.arange(local_nz) + 1
        spread_upper_widths = jnp.where(
            global_upper < grid.nz, z_upper_widths, 0.0
        )
        spread_upper_areas = spread_upper_widths[:, None] * y_widths[None, :]
        rings_cell, tangent_y_cell, _ = ring_geometry(z_cell, cell_areas)
        rings_upper, _, tangent_z_upper = ring_geometry(z_upper, upper_areas)
        rings_upper_force, _, _ = ring_geometry(
            z_upper, spread_upper_areas
        )

        sampled_u_local = jnp.einsum(
            "rzy,rx,zyx->r", rings_cell, weights_x, u, optimize="optimal"
        )
        sampled_vt_local = jnp.einsum(
            "rzy,zy,rx,zyx->r",
            rings_cell,
            tangent_y_cell,
            weights_x,
            v,
            optimize="optimal",
        )
        sampled_wt_local = jnp.einsum(
            "rzy,zy,rx,zyx->r",
            rings_upper,
            tangent_z_upper,
            weights_x,
            w_upper,
            optimize="optimal",
        )
        sampled_axial = lax.psum(sampled_u_local, axis_name)
        sampled_tangent = lax.psum(
            sampled_vt_local + sampled_wt_local, axis_name
        )

        count = element_radii.shape[0]
        normal = jnp.asarray((1.0, 0.0, 0.0), dtype=dtype)
        tangent = jnp.broadcast_to(
            jnp.asarray((0.0, 1.0, 0.0), dtype=dtype), (count, 3)
        )
        sampled = jnp.stack(
            (sampled_axial, sampled_tangent, jnp.zeros_like(sampled_axial)),
            axis=1,
        )
        forces, alpha, lift, drag, loss = blade_element_kinematic_forces(
            sampled,
            tangent,
            angular_velocity * element_radii,
            normal,
            element_radii=element_radii,
            element_widths=element_widths,
            element_chords=element_chords,
            element_twist_degrees=element_twist_degrees,
            element_airfoil_ids=element_airfoil_ids,
            blade_count=blade_count,
            hub_radius=hub_radius,
            tip_radius=tip_radius,
            pitch_degrees=pitch_degrees,
            polar_alpha_degrees=polar_alpha_degrees,
            polar_lift_coefficients=polar_lift_coefficients,
            polar_drag_coefficients=polar_drag_coefficients,
            tip_loss=tip_loss,
            root_loss=root_loss,
        )
        forces = forces * jnp.asarray(blade_count, dtype)
        axial_force = forces[:, 0]
        tangent_force = forces[:, 1]
        cell_volumes = (
            z_widths[:, None, None]
            * y_widths[None, :, None]
            * x_widths[None, None, :]
        )
        upper_face_volumes = (
            jnp.where(spread_upper_widths > 0.0, spread_upper_widths, 1.0)[
                :, None, None
            ]
            * y_widths[None, :, None]
            * x_widths[None, None, :]
        )
        source_x = jnp.einsum(
            "r,rzy,rx->zyx", axial_force, rings_cell, weights_x,
            optimize="optimal",
        ) / cell_volumes
        source_y = jnp.einsum(
            "r,rzy,zy,rx->zyx", tangent_force, rings_cell,
            tangent_y_cell, weights_x, optimize="optimal",
        ) / cell_volumes
        source_z = jnp.einsum(
            "r,rzy,zy,rx->zyx", tangent_force, rings_upper_force,
            tangent_z_upper, weights_x, optimize="optimal",
        ) / upper_face_volumes
        source_z = source_z.at[-1].set(
            jnp.where(partition_index == partition_count - 1, 0.0, source_z[-1])
        )
        return (
            source_x, source_y, source_z, forces, sampled,
            alpha, lift, drag, loss,
        )

    return actuator_disk_bem_local


def build_nacelle_tower_kernel(*, grid, axis_name: str):
    """Build local nacelle and tapered-tower drag forcing."""

    def nacelle_tower_local(
        u,
        v,
        body_x,
        body_y,
        hub_height,
        nacelle_length,
        nacelle_diameter,
        nacelle_drag_coefficient,
        tower_base_diameter,
        tower_top_diameter,
        tower_drag_coefficient,
        smoothing_width,
    ):
        dtype = u.dtype
        tiny = jnp.finfo(dtype).tiny
        local_nz = u.shape[0]
        partition_index = lax.axis_index(axis_name)
        x = jnp.asarray(grid.x_centers, dtype=dtype)
        y = jnp.asarray(grid.y_centers, dtype=dtype)
        z, _, z_widths, _ = _local_z_metrics(
            grid, dtype, local_nz, partition_index
        )
        x_widths = jnp.asarray(grid.x_widths, dtype=dtype)
        y_widths = jnp.asarray(grid.y_widths, dtype=dtype)
        cell_volumes = (
            z_widths[:, None, None]
            * y_widths[None, :, None]
            * x_widths[None, None, :]
        )
        horizontal_areas = y_widths[:, None] * x_widths[None, :]
        dx = jnp.mod(x - body_x + 0.5 * grid.lx, grid.lx) - 0.5 * grid.lx
        dy = jnp.mod(y - body_y + 0.5 * grid.ly, grid.ly) - 0.5 * grid.ly
        width = jnp.asarray(smoothing_width, dtype)

        # A Gaussian ellipsoid approximates the finite nacelle volume; its
        # normalized kernel conserves the requested projected drag force.
        axial_width = jnp.sqrt(width**2 + (0.5 * nacelle_length) ** 2)
        nacelle_raw = (
            jnp.exp(-(dx[None, None, :] / axial_width) ** 2)
            * jnp.exp(-(dy[None, :, None] / width) ** 2)
            * jnp.exp(-((z[:, None, None] - hub_height) / width) ** 2)
        )
        nacelle_mass = nacelle_raw * cell_volumes
        nacelle_sum = lax.psum(jnp.sum(nacelle_mass), axis_name)
        nacelle_weights = nacelle_mass / jnp.maximum(nacelle_sum, tiny)
        nacelle_velocity = lax.psum(
            jnp.sum(nacelle_weights * u), axis_name
        )
        nacelle_area = 0.25 * jnp.pi * nacelle_diameter**2
        nacelle_force = (
            -0.5 * nacelle_drag_coefficient * nacelle_area
            * nacelle_velocity * jnp.abs(nacelle_velocity)
        )
        source_x_nacelle = nacelle_force * nacelle_weights / cell_volumes

        # The tower is evaluated independently at every cell-centred height.
        # This gives local vector cross-flow drag and a linearly tapered width.
        tower_raw_xy = (
            jnp.exp(-(dx[None, :] / width) ** 2)
            * jnp.exp(-(dy[:, None] / width) ** 2)
        )
        tower_mass_xy = tower_raw_xy * horizontal_areas
        tower_weights_xy = tower_mass_xy / jnp.maximum(
            jnp.sum(tower_mass_xy), tiny
        )
        sampled_u = jnp.einsum("yx,zyx->z", tower_weights_xy, u)
        tower_top = hub_height - 0.5 * nacelle_diameter
        fraction = jnp.clip(z / jnp.maximum(tower_top, tiny), 0.0, 1.0)
        diameter = tower_base_diameter + fraction * (
            tower_top_diameter - tower_base_diameter
        )
        active = (z < tower_top).astype(dtype)
        force_per_length = (
            -0.5 * tower_drag_coefficient * diameter
            * sampled_u * jnp.abs(sampled_u)
        )
        source_x_tower = (
            active[:, None, None] * force_per_length[:, None, None]
            * tower_weights_xy[None, :, :]
            / horizontal_areas[None, :, :]
        )
        return (
            source_x_nacelle + source_x_tower,
            jnp.zeros_like(v),
            jnp.zeros_like(u),
        )

    return nacelle_tower_local


def build_actuator_line_kernel(
    *,
    grid,
    axis_name: str,
    partition_count: int,
):
    def actuator_line_local(
        u,
        v,
        w_upper,
        w_lower_boundary,
        time,
        line_x,
        line_y,
        line_z,
        blade_count,
        hub_radius,
        tip_radius,
        angular_velocity,
        smoothing_width,
        element_radii,
        element_widths,
        element_chords,
        element_twist_degrees,
        element_airfoil_ids,
        polar_alpha_degrees,
        polar_lift_coefficients,
        polar_drag_coefficients,
        pitch_degrees,
        yaw_degrees,
        tilt_degrees,
        precone_degrees,
        initial_azimuth_degrees,
        tip_loss,
        root_loss,
        flap_displacements,
        edge_displacements,
        flap_slopes,
        edge_slopes,
        flap_velocities,
        edge_velocities,
    ):
        """Sample and spread rigid or modal actuator lines across the z mesh."""

        dtype = u.dtype
        positions, tangents, blade_velocity, normal, span_directions = (
            actuator_line_deformed_kinematics(
                x=line_x,
                y=line_y,
                z=line_z,
                blade_count=blade_count,
                element_radii=element_radii,
                angular_velocity=angular_velocity,
                time=time,
                yaw_degrees=yaw_degrees,
                tilt_degrees=tilt_degrees,
                precone_degrees=precone_degrees,
                initial_azimuth_degrees=initial_azimuth_degrees,
                flap_displacements=flap_displacements,
                edge_displacements=edge_displacements,
                flap_slopes=flap_slopes,
                edge_slopes=edge_slopes,
                flap_velocities=flap_velocities,
                edge_velocities=edge_velocities,
                dtype=dtype,
            )
        )
        local_nz = u.shape[0]
        partition_index = lax.axis_index(axis_name)
        tiny = jnp.finfo(dtype).tiny
        x_coordinates = jnp.asarray(grid.x_centers, dtype=dtype)
        y_coordinates = jnp.asarray(grid.y_centers, dtype=dtype)
        x_widths = jnp.asarray(grid.x_widths, dtype=dtype)
        y_widths = jnp.asarray(grid.y_widths, dtype=dtype)
        (
            z_cell_coordinates,
            z_upper_coordinates,
            z_cell_widths,
            z_upper_widths,
        ) = _local_z_metrics(grid, dtype, local_nz, partition_index)

        weights_x = gaussian_weights(
            positions[:, 0],
            x_coordinates,
            smoothing_width=smoothing_width,
            period=grid.lx,
        )
        weights_x = weights_x * x_widths[None, :]
        weights_x = weights_x / jnp.maximum(
            jnp.sum(weights_x, axis=1, keepdims=True), tiny
        )
        weights_y = gaussian_weights(
            positions[:, 1],
            y_coordinates,
            smoothing_width=smoothing_width,
            period=grid.ly,
        )
        weights_y = weights_y * y_widths[None, :]
        weights_y = weights_y / jnp.maximum(
            jnp.sum(weights_y, axis=1, keepdims=True), tiny
        )
        width = jnp.asarray(smoothing_width, dtype=dtype)
        vertical_width = width if width.ndim == 0 else width[:, None]
        cell_exponent = (
            (z_cell_coordinates[None, :] - positions[:, 2, None])
            / vertical_width
        ) ** 2
        cell_minimum = lax.pmin(
            jnp.min(cell_exponent, axis=1),
            axis_name,
        )
        raw_z_cells = jnp.exp(
            -(cell_exponent - cell_minimum[:, None])
        )
        weighted_z_cells = raw_z_cells * z_cell_widths[None, :]
        cell_denominator = lax.psum(
            jnp.sum(weighted_z_cells, axis=1),
            axis_name,
        )
        weights_z_cells = weighted_z_cells / jnp.maximum(
            cell_denominator[:, None],
            tiny,
        )

        upper_exponent = (
            (z_upper_coordinates[None, :] - positions[:, 2, None])
            / vertical_width
        ) ** 2
        lower_exponent = (positions[:, 2] / width) ** 2
        local_face_minimum = jnp.minimum(
            jnp.min(upper_exponent, axis=1),
            jnp.where(
                partition_index == 0,
                lower_exponent,
                jnp.full_like(lower_exponent, jnp.inf),
            ),
        )
        face_minimum = lax.pmin(local_face_minimum, axis_name)
        raw_z_upper = jnp.exp(
            -(upper_exponent - face_minimum[:, None])
        )
        raw_lower_boundary = jnp.exp(
            -(lower_exponent - face_minimum)
        )
        weighted_z_upper = raw_z_upper * z_upper_widths[None, :]
        lower_width = 0.5 * jnp.asarray(grid.z_widths[0], dtype=dtype)
        weighted_lower_boundary = raw_lower_boundary * lower_width
        face_denominator = lax.psum(
            jnp.sum(weighted_z_upper, axis=1)
            + jnp.where(
                partition_index == 0,
                weighted_lower_boundary,
                jnp.zeros_like(raw_lower_boundary),
            ),
            axis_name,
        )
        weights_z_upper = weighted_z_upper / jnp.maximum(
            face_denominator[:, None],
            tiny,
        )
        weights_z_lower = weighted_lower_boundary / jnp.maximum(
            face_denominator,
            tiny,
        )

        global_upper = partition_index * local_nz + jnp.arange(local_nz) + 1
        spread_active = global_upper < grid.nz
        spread_z_widths = jnp.where(spread_active, z_upper_widths, 0.0)
        local_spread_minimum = jnp.min(
            jnp.where(
                spread_active[None, :],
                upper_exponent,
                jnp.inf,
            ),
            axis=1,
        )
        spread_minimum = lax.pmin(local_spread_minimum, axis_name)
        raw_z_spread = jnp.where(
            spread_active[None, :],
            jnp.exp(-(upper_exponent - spread_minimum[:, None])),
            0.0,
        )
        weighted_z_spread = raw_z_spread * spread_z_widths[None, :]
        spread_denominator = lax.psum(
            jnp.sum(weighted_z_spread, axis=1), axis_name
        )
        weights_z_spread = weighted_z_spread / jnp.maximum(
            spread_denominator[:, None], tiny
        )

        sampled_u_local = jnp.einsum(
            "pz,py,px,zyx->p",
            weights_z_cells,
            weights_y,
            weights_x,
            u,
            optimize="optimal",
        )
        sampled_v_local = jnp.einsum(
            "pz,py,px,zyx->p",
            weights_z_cells,
            weights_y,
            weights_x,
            v,
            optimize="optimal",
        )
        sampled_w_local = jnp.einsum(
            "pz,py,px,zyx->p",
            weights_z_upper,
            weights_y,
            weights_x,
            w_upper,
            optimize="optimal",
        )
        sampled_w_lower = jnp.einsum(
            "p,py,px,yx->p",
            weights_z_lower,
            weights_y,
            weights_x,
            w_lower_boundary,
            optimize="optimal",
        )
        sampled_local = jnp.stack(
            (
                sampled_u_local,
                sampled_v_local,
                sampled_w_local
                + jnp.where(
                    partition_index == 0,
                    sampled_w_lower,
                    jnp.zeros_like(sampled_w_lower),
                ),
            ),
            axis=1,
        )
        sampled = lax.psum(sampled_local, axis_name)
        repeat = blade_count
        forces, alpha, lift, drag, loss = blade_element_kinematic_forces(
            sampled,
            tangents,
            jnp.zeros((positions.shape[0],), dtype=dtype),
            normal,
            element_radii=jnp.tile(element_radii, repeat),
            element_widths=jnp.tile(element_widths, repeat),
            element_chords=jnp.tile(element_chords, repeat),
            element_twist_degrees=jnp.tile(
                element_twist_degrees,
                repeat,
            ),
            element_airfoil_ids=jnp.tile(
                element_airfoil_ids,
                repeat,
            ),
            blade_count=blade_count,
            hub_radius=hub_radius,
            tip_radius=tip_radius,
            pitch_degrees=pitch_degrees,
            polar_alpha_degrees=polar_alpha_degrees,
            polar_lift_coefficients=polar_lift_coefficients,
            polar_drag_coefficients=polar_drag_coefficients,
            tip_loss=tip_loss,
            root_loss=root_loss,
            blade_velocity=blade_velocity,
        )
        cell_volumes = (
            z_cell_widths[:, None, None]
            * y_widths[None, :, None]
            * x_widths[None, None, :]
        )
        upper_face_volumes = (
            jnp.where(spread_z_widths > 0.0, spread_z_widths, 1.0)[
                :, None, None
            ]
            * y_widths[None, :, None]
            * x_widths[None, None, :]
        )
        source_x = jnp.einsum(
            "p,pz,py,px->zyx",
            forces[:, 0],
            weights_z_cells,
            weights_y,
            weights_x,
            optimize="optimal",
        ) / cell_volumes
        source_y = jnp.einsum(
            "p,pz,py,px->zyx",
            forces[:, 1],
            weights_z_cells,
            weights_y,
            weights_x,
            optimize="optimal",
        ) / cell_volumes
        source_z = jnp.einsum(
            "p,pz,py,px->zyx",
            forces[:, 2],
            weights_z_spread,
            weights_y,
            weights_x,
            optimize="optimal",
        ) / upper_face_volumes
        source_z = source_z.at[-1].set(
            jnp.where(
                partition_index == partition_count - 1,
                0.0,
                source_z[-1],
            )
        )
        return (
            source_x,
            source_y,
            source_z,
            forces,
            positions,
            tangents,
            normal,
            span_directions,
            blade_velocity,
            sampled,
            alpha,
            lift,
            drag,
            loss,
        )


    return actuator_line_local
