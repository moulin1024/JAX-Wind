"""Actuator-disk, fringe, and actuator-line JAX kernels."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from jaxwind._jax.actuator_disk import (
    filtered_disk_velocity_correction,
    gaussian_convolved_annulus,
)
from jaxwind._jax.actuator_line import (
    actuator_line_deformed_kinematics,
    blade_element_kinematic_forces,
    gaussian_weights,
)
from jaxwind._jax.fringe import plateau_fringe_mask


def build_wind_tunnel_kernel(*, grid, axis_name: str):
    def wind_tunnel_local(
        u,
        v,
        w_upper,
        target_u,
        target_v,
        target_w_upper,
        disk_enabled,
        disk_x,
        disk_y,
        disk_z,
        disk_diameter,
        hub_diameter,
        thrust_coefficient_prime,
        normal_smoothing_width,
        transverse_smoothing_width,
        yaw_degrees,
        filtered_velocity_correction_enabled,
        prescribed_inflow_velocity,
        prescribed_thrust_coefficient,
        fringe_enabled,
        fringe_start_x,
        fringe_relaxation_time,
        fringe_rise_width,
        fringe_fall_width,
    ):
        dtype = u.dtype
        local_nz = u.shape[0]
        partition_index = lax.axis_index(axis_name)
        x = (jnp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
        y = (jnp.arange(grid.ny, dtype=dtype) + 0.5) * grid.dy
        z_index = partition_index * local_nz + jnp.arange(local_nz, dtype=dtype)
        z = (z_index + 0.5) * grid.dz

        periodic_x = (
            jnp.mod(x - jnp.asarray(disk_x, dtype) + 0.5 * grid.lx, grid.lx)
            - 0.5 * grid.lx
        )
        periodic_y = (
            jnp.mod(y - jnp.asarray(disk_y, dtype) + 0.5 * grid.ly, grid.ly)
            - 0.5 * grid.ly
        )
        yaw = jnp.deg2rad(jnp.asarray(yaw_degrees, dtype))
        normal_x = jnp.cos(yaw)
        normal_y = jnp.sin(yaw)
        normal_distance = (
            periodic_x[None, None, :] * normal_x
            + periodic_y[None, :, None] * normal_y
        )
        in_plane = (
            -periodic_x[None, None, :] * normal_y
            + periodic_y[None, :, None] * normal_x
        )
        radius = jnp.sqrt(
            in_plane**2 + (z[:, None, None] - jnp.asarray(disk_z, dtype)) ** 2
        )
        streamwise = jnp.exp(
            -(
                normal_distance
                / jnp.asarray(normal_smoothing_width, dtype)
            )
            ** 2
        )
        radial = gaussian_convolved_annulus(
            radius,
            outer_radius=0.5 * jnp.asarray(disk_diameter, dtype),
            inner_radius=0.5 * jnp.asarray(hub_diameter, dtype),
            smoothing_width=jnp.asarray(transverse_smoothing_width, dtype),
        )
        disk_kernel = radial * streamwise
        disk_area = 0.25 * jnp.pi * (
            jnp.asarray(disk_diameter, dtype) ** 2
            - jnp.asarray(hub_diameter, dtype) ** 2
        )
        kernel_integral = (
            lax.psum(jnp.sum(disk_kernel), axis_name)
            * grid.dx
            * grid.dy
            * grid.dz
        )
        disk_kernel = disk_kernel * disk_area / jnp.maximum(
            kernel_integral,
            jnp.finfo(dtype).tiny,
        )
        normal_velocity = u * normal_x + v * normal_y
        numerator = lax.psum(jnp.sum(normal_velocity * disk_kernel), axis_name)
        denominator = lax.psum(jnp.sum(disk_kernel), axis_name)
        disk_velocity = numerator / jnp.maximum(denominator, jnp.finfo(dtype).tiny)
        velocity_correction = jnp.where(
            jnp.asarray(filtered_velocity_correction_enabled),
            filtered_disk_velocity_correction(
                thrust_coefficient_prime,
                outer_radius=0.5 * jnp.asarray(disk_diameter, dtype),
                inner_radius=0.5 * jnp.asarray(hub_diameter, dtype),
                smoothing_width=jnp.asarray(transverse_smoothing_width, dtype),
                dtype=dtype,
            ),
            1.0,
        )
        disk_velocity = velocity_correction * disk_velocity
        prescribed = jnp.asarray(prescribed_inflow_velocity, dtype) > 0.0
        loading_velocity = jnp.where(
            prescribed,
            jnp.asarray(prescribed_inflow_velocity, dtype),
            disk_velocity,
        )
        loading_coefficient = jnp.where(
            prescribed,
            jnp.asarray(prescribed_thrust_coefficient, dtype),
            jnp.asarray(thrust_coefficient_prime, dtype),
        )
        disk_acceleration = (
            -0.5
            * loading_coefficient
            * loading_velocity
            * jnp.abs(loading_velocity)
            * disk_kernel
            * jnp.asarray(disk_enabled, dtype)
        )

        mask = plateau_fringe_mask(
            x,
            start_x=fringe_start_x,
            end_x=grid.lx,
            rise_width=fringe_rise_width,
            fall_width=fringe_fall_width,
        )
        rate = (
            mask
            / jnp.asarray(fringe_relaxation_time, dtype)
            * jnp.asarray(fringe_enabled, dtype)
        )
        source_u = disk_acceleration * normal_x + rate[None, None, :] * (
            target_u - u
        )
        source_v = disk_acceleration * normal_y + rate[None, None, :] * (
            target_v - v
        )
        source_w = rate[None, None, :] * (target_w_upper - w_upper)
        return source_u, source_v, source_w


    return wind_tunnel_local


def build_blade_element_disk_kernel(
    *,
    grid,
    axis_name: str,
    partition_count: int,
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

        x = (jnp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
        y = (jnp.arange(grid.ny, dtype=dtype) + 0.5) * grid.dy
        zi = partition_index * local_nz + jnp.arange(local_nz, dtype=dtype)
        z_cell = (zi + 0.5) * grid.dz
        z_upper = (zi + 1.0) * grid.dz
        dx = jnp.mod(x - disk_x + 0.5 * grid.lx, grid.lx) - 0.5 * grid.lx
        dy = jnp.mod(y - disk_y + 0.5 * grid.ly, grid.ly) - 0.5 * grid.ly

        raw_x = jnp.exp(-(dx[None, :] / widths[:, None]) ** 2)
        weights_x = raw_x / jnp.maximum(
            jnp.sum(raw_x, axis=1, keepdims=True), tiny
        )

        def ring_geometry(z_coordinates):
            yy = dy[None, None, :]
            zz = z_coordinates[None, :, None] - jnp.asarray(disk_z, dtype)
            radius = jnp.sqrt(yy * yy + zz * zz)
            raw = jnp.exp(
                -(
                    (radius - element_radii[:, None, None])
                    / widths[:, None, None]
                )
                ** 2
            )
            denominator = lax.psum(jnp.sum(raw, axis=(1, 2)), axis_name)
            weights = raw / jnp.maximum(denominator[:, None, None], tiny)
            # Positive rotation points along +y at the top of the rotor.
            tangent_y = zz / jnp.maximum(radius, tiny)
            tangent_z = -yy / jnp.maximum(radius, tiny)
            return weights, tangent_y[0], tangent_z[0]

        rings_cell, tangent_y_cell, _ = ring_geometry(z_cell)
        rings_upper, _, tangent_z_upper = ring_geometry(z_upper)

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
        inverse_volume = 1.0 / (grid.dx * grid.dy * grid.dz)
        source_x = inverse_volume * jnp.einsum(
            "r,rzy,rx->zyx", axial_force, rings_cell, weights_x,
            optimize="optimal",
        )
        source_y = inverse_volume * jnp.einsum(
            "r,rzy,zy,rx->zyx", tangent_force, rings_cell,
            tangent_y_cell, weights_x, optimize="optimal",
        )
        source_z = inverse_volume * jnp.einsum(
            "r,rzy,zy,rx->zyx", tangent_force, rings_upper,
            tangent_z_upper, weights_x, optimize="optimal",
        )
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
        x = (jnp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
        y = (jnp.arange(grid.ny, dtype=dtype) + 0.5) * grid.dy
        zi = partition_index * local_nz + jnp.arange(local_nz, dtype=dtype)
        z = (zi + 0.5) * grid.dz
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
        nacelle_sum = lax.psum(jnp.sum(nacelle_raw), axis_name)
        nacelle_weights = nacelle_raw / jnp.maximum(nacelle_sum, tiny)
        nacelle_velocity = lax.psum(
            jnp.sum(nacelle_weights * u), axis_name
        )
        nacelle_area = 0.25 * jnp.pi * nacelle_diameter**2
        nacelle_force = (
            -0.5 * nacelle_drag_coefficient * nacelle_area
            * nacelle_velocity * jnp.abs(nacelle_velocity)
        )
        source_x_nacelle = (
            nacelle_force * nacelle_weights / (grid.dx * grid.dy * grid.dz)
        )

        # The tower is evaluated independently at every cell-centred height.
        # This gives local vector cross-flow drag and a linearly tapered width.
        tower_raw_xy = (
            jnp.exp(-(dx[None, :] / width) ** 2)
            * jnp.exp(-(dy[:, None] / width) ** 2)
        )
        tower_weights_xy = tower_raw_xy / jnp.maximum(jnp.sum(tower_raw_xy), tiny)
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
        horizontal_scale = 1.0 / (grid.dx * grid.dy)
        source_x_tower = (
            active[:, None, None] * force_per_length[:, None, None]
            * tower_weights_xy[None, :, :]
            * horizontal_scale
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
        x_coordinates = (
            jnp.arange(grid.nx, dtype=dtype) + 0.5
        ) * grid.dx
        y_coordinates = (
            jnp.arange(grid.ny, dtype=dtype) + 0.5
        ) * grid.dy
        global_cell_index = (
            partition_index * local_nz
            + jnp.arange(local_nz, dtype=dtype)
        )
        z_cell_coordinates = (global_cell_index + 0.5) * grid.dz
        z_upper_coordinates = (global_cell_index + 1.0) * grid.dz

        weights_x = gaussian_weights(
            positions[:, 0],
            x_coordinates,
            smoothing_width=smoothing_width,
            period=grid.lx,
        )
        weights_y = gaussian_weights(
            positions[:, 1],
            y_coordinates,
            smoothing_width=smoothing_width,
            period=grid.ly,
        )
        width = jnp.asarray(smoothing_width, dtype=dtype)
        raw_z_cells = jnp.exp(
            -(
                (z_cell_coordinates[None, :] - positions[:, 2, None])
                / width
            )
            ** 2
        )
        cell_denominator = lax.psum(
            jnp.sum(raw_z_cells, axis=1),
            axis_name,
        )
        weights_z_cells = raw_z_cells / jnp.maximum(
            cell_denominator[:, None],
            jnp.finfo(dtype).tiny,
        )

        raw_z_upper = jnp.exp(
            -(
                (z_upper_coordinates[None, :] - positions[:, 2, None])
                / width
            )
            ** 2
        )
        raw_lower_boundary = jnp.exp(
            -(positions[:, 2] / width) ** 2
        )
        face_denominator = lax.psum(
            jnp.sum(raw_z_upper, axis=1)
            + jnp.where(
                partition_index == 0,
                raw_lower_boundary,
                jnp.zeros_like(raw_lower_boundary),
            ),
            axis_name,
        )
        weights_z_upper = raw_z_upper / jnp.maximum(
            face_denominator[:, None],
            jnp.finfo(dtype).tiny,
        )
        weights_z_lower = raw_lower_boundary / jnp.maximum(
            face_denominator,
            jnp.finfo(dtype).tiny,
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
        inverse_cell_volume = 1.0 / (grid.dx * grid.dy * grid.dz)
        source_x = inverse_cell_volume * jnp.einsum(
            "p,pz,py,px->zyx",
            forces[:, 0],
            weights_z_cells,
            weights_y,
            weights_x,
            optimize="optimal",
        )
        source_y = inverse_cell_volume * jnp.einsum(
            "p,pz,py,px->zyx",
            forces[:, 1],
            weights_z_cells,
            weights_y,
            weights_x,
            optimize="optimal",
        )
        source_z = inverse_cell_volume * jnp.einsum(
            "p,pz,py,px->zyx",
            forces[:, 2],
            weights_z_upper,
            weights_y,
            weights_x,
            optimize="optimal",
        )
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
