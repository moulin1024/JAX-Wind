"""Fused LASD Boussinesq right-hand side."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from .surface import monin_obukhov_surface_transfer


def build_fused_neutral_boussinesq_kernels(
    *,
    grid,
    axis_name,
    frozen_zero_scalar,
    scalar_context_local,
    scalar_advection_from_momentum_local,
    scalar_sgs_from_momentum_gradients_local,
    buoyancy_local,
    wall_filter_local,
    dry_flow_context_local,
    dry_sgs_from_gradients_local,
):
    if not isinstance(frozen_zero_scalar, bool):
        raise TypeError("frozen zero scalar flag must be boolean")

    def rotational_advection_local(
        momentum,
        momentum_fields,
        lower_fields,
        cell_gradients,
        face_gradients,
        wall_u,
        wall_v,
        wall_gradient_factor,
    ):
        """Evaluate legacy ``-omega x u`` on the base staggered layout."""

        u, v, w_upper = momentum_fields[:3]
        u_upper, v_upper = momentum_fields[3:5]
        w_lower = jnp.concatenate(
            (lower_fields[0][None], w_upper[:-1]),
            axis=0,
        )
        lower_dudz = 2.0 * cell_gradients[2] - face_gradients[2]
        lower_dvdz = 2.0 * cell_gradients[5] - face_gradients[5]
        lower_dwdx = 2.0 * cell_gradients[6] - face_gradients[6]
        lower_dwdy = 2.0 * cell_gradients[7] - face_gradients[7]
        bottom = lax.axis_index(axis_name) == 0
        lower_dudz = lower_dudz.at[0].set(
            jnp.where(
                bottom,
                wall_gradient_factor * wall_u,
                lower_dudz[0],
            )
        )
        lower_dvdz = lower_dvdz.at[0].set(
            jnp.where(
                bottom,
                wall_gradient_factor * wall_v,
                lower_dvdz[0],
            )
        )
        x = v * (cell_gradients[1] - cell_gradients[3])
        x += 0.5 * (
            w_upper * (face_gradients[2] - face_gradients[6])
            + w_lower * (lower_dudz - lower_dwdx)
        )
        y = u * (cell_gradients[3] - cell_gradients[1])
        y += 0.5 * (
            w_upper * (face_gradients[5] - face_gradients[7])
            + w_lower * (lower_dvdz - lower_dwdy)
        )
        z = u_upper * (face_gradients[6] - face_gradients[2])
        z += v_upper * (face_gradients[7] - face_gradients[5])
        x, y, z = -x, -y, -z
        z = z.at[-1].set(jnp.where(momentum.upper_is_physical, 0.0, z[-1]))
        return x, y, z

    def shared_momentum_from_context_local(
        momentum,
        wall_drag,
        wall_filtered,
        wall_filter_width,
        wall_gradient_factor,
    ):
        momentum_fields = jnp.stack(
            (
                momentum.u,
                momentum.v,
                momentum.w_upper,
                momentum.u_upper,
                momentum.v_upper,
                momentum.w_at_cells,
                momentum.w_next_cell,
            ),
            axis=0,
        )
        lower_fields = jnp.stack(
            (momentum.w_lower, momentum.u_lower, momentum.v_lower),
            axis=0,
        )
        cell_gradients = (
            momentum.dudx,
            momentum.dudy,
            momentum.dudz_at_cells,
            momentum.dvdx,
            momentum.dvdy,
            momentum.dvdz_at_cells,
            momentum.dwdx_at_cells,
            momentum.dwdy_at_cells,
            momentum.dwdz,
        )
        face_gradients = (
            momentum.dudx_upper,
            momentum.dudy_upper,
            momentum.dudz_upper,
            momentum.dvdx_upper,
            momentum.dvdy_upper,
            momentum.dvdz_upper,
            momentum.dwdx_upper,
            momentum.dwdy_upper,
            momentum.dwdz_upper,
        )
        wall_velocity = wall_filter_local(
            jnp.stack((momentum.u[0], momentum.v[0])),
            wall_filter_width,
        )
        wall_u = jnp.where(wall_filtered, wall_velocity[0], momentum.u[0])
        wall_v = jnp.where(wall_filtered, wall_velocity[1], momentum.v[0])
        advection = rotational_advection_local(
            momentum,
            momentum_fields,
            lower_fields,
            cell_gradients,
            face_gradients,
            wall_u,
            wall_v,
            wall_gradient_factor,
        )
        wall_speed = jnp.hypot(wall_u, wall_v)
        wall_x = -wall_drag * wall_speed * wall_u / grid.dz
        wall_y = -wall_drag * wall_speed * wall_v / grid.dz
        bottom = lax.axis_index(axis_name) == 0
        wall = (
            jnp.zeros_like(momentum.u).at[0].set(
                jnp.where(bottom, wall_x, 0.0)
            ),
            jnp.zeros_like(momentum.v).at[0].set(
                jnp.where(bottom, wall_y, 0.0)
            ),
            jnp.zeros_like(momentum.w_upper),
        )
        return (
            momentum,
            advection,
            wall,
            cell_gradients,
            face_gradients,
            momentum_fields,
            lower_fields,
        )

    def combine_local(advection, wall, sgs, pressure_x, pressure_y):
        return (
            advection[0] + wall[0] + sgs[0] + pressure_x,
            advection[1] + wall[1] + sgs[1] + pressure_y,
            advection[2] + wall[2] + sgs[2],
        )

    def add_coriolis_local(
        tendency,
        momentum,
        coriolis_parameter,
        geostrophic_x_velocity,
        geostrophic_y_velocity,
        horizontal_coriolis_parameter,
    ):
        local_f = jnp.asarray(coriolis_parameter, dtype=momentum.u.dtype)
        horizontal_f = jnp.asarray(
            horizontal_coriolis_parameter,
            dtype=momentum.u.dtype,
        )
        x = (
            tendency[0]
            + local_f * (momentum.v - geostrophic_y_velocity)
            - horizontal_f * momentum.w_at_cells
        )
        y = tendency[1] - local_f * (momentum.u - geostrophic_x_velocity)
        coriolis_z = horizontal_f.astype(momentum.w_upper.dtype) * (
            momentum.u_upper - geostrophic_x_velocity
        )
        coriolis_z = coriolis_z.at[-1].set(
            jnp.where(momentum.upper_is_physical, 0.0, coriolis_z[-1])
        )
        return x, y, tendency[2] + coriolis_z

    def fused_lasd_from_context_core_local(
        momentum,
        theta,
        coefficient,
        scalar_coefficient,
        pressure_x,
        pressure_y,
        coriolis_parameter,
        geostrophic_x_velocity,
        geostrophic_y_velocity,
        horizontal_coriolis_parameter,
        wall_drag,
        wall_filtered,
        wall_filter_width,
        wall_gradient_factor,
        minimum_coefficient,
        maximum_coefficient,
        minimum_scalar_coefficient,
        maximum_scalar_coefficient,
        lower_scalar_flux,
        upper_scalar_flux,
        stability_buoyancy_coefficient,
        stability_beta,
        stability_power,
        imposed_wall_x,
        imposed_wall_y,
        imposed_scalar_source,
        buoyancy_coefficient,
        execution_time,
        momentum_roughness_length,
        scalar_roughness_length,
        surface_scalar_initial,
        surface_scalar_rate,
        x_velocity_offset,
        y_velocity_offset,
        von_karman,
        positive_zeta_momentum_slope,
        positive_zeta_scalar_slope,
        negative_zeta_momentum_coefficient,
        negative_zeta_scalar_coefficient,
        surface_relaxation,
        maximum_abs_zeta,
        surface_iterations,
        source_mode,
    ):
        (
            momentum,
            advection,
            wall,
            cell_gradients,
            face_gradients,
            momentum_fields,
            lower_fields,
        ) = shared_momentum_from_context_local(
            momentum,
            wall_drag,
            wall_filtered,
            wall_filter_width,
            wall_gradient_factor,
        )
        u = momentum.u
        v = momentum.v
        w_upper = momentum.w_upper
        scalar_surface_source = imposed_scalar_source
        if source_mode == 2:
            index = lax.axis_index(axis_name)
            is_bottom = index == 0

            def bottom_mean(values):
                local_mean = jnp.where(is_bottom, jnp.mean(values[0]), 0.0)
                return lax.psum(local_mean, axis_name)

            singleton = (1, 1, 1)
            surface = monin_obukhov_surface_transfer(
                bottom_mean(u).reshape(singleton),
                bottom_mean(v).reshape(singleton),
                bottom_mean(theta).reshape(singleton),
                execution_time,
                grid.dz,
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
                surface_relaxation,
                maximum_abs_zeta,
                bottom=0,
                iterations=surface_iterations,
            )
            imposed_wall_x = surface.wall_x_acceleration
            imposed_wall_y = surface.wall_y_acceleration
            scalar_surface_source = surface.scalar_surface_source
        if source_mode:
            bottom = lax.axis_index(axis_name) == 0
            wall = (
                jnp.zeros_like(u).at[0].set(
                    jnp.where(bottom, imposed_wall_x, 0.0)
                ),
                jnp.zeros_like(v).at[0].set(
                    jnp.where(bottom, imposed_wall_y, 0.0)
                ),
                jnp.zeros_like(w_upper),
            )
        sgs = dry_sgs_from_gradients_local(
            momentum,
            cell_gradients,
            face_gradients,
            coefficient,
            minimum_coefficient,
            maximum_coefficient,
        )
        momentum_tendency = add_coriolis_local(
            combine_local(advection, wall, sgs, pressure_x, pressure_y),
            momentum,
            coriolis_parameter,
            geostrophic_x_velocity,
            geostrophic_y_velocity,
            horizontal_coriolis_parameter,
        )
        scalar_tendency = jnp.zeros_like(theta)
        if not frozen_zero_scalar:
            scalar = scalar_context_local(theta)
            scalar_tendency = scalar_advection_from_momentum_local(
                scalar,
                momentum_fields,
                lower_fields,
            )
            scalar_tendency = (
                scalar_tendency
                + scalar_sgs_from_momentum_gradients_local(
                    scalar,
                    momentum,
                    cell_gradients,
                    face_gradients,
                    scalar_coefficient,
                    minimum_scalar_coefficient,
                    maximum_scalar_coefficient,
                    lower_scalar_flux,
                    upper_scalar_flux,
                    stability_buoyancy_coefficient,
                    stability_beta,
                    stability_power,
                )
            )
            momentum_tendency = (
                momentum_tendency[0],
                momentum_tendency[1],
                momentum_tendency[2]
                + buoyancy_local(scalar, buoyancy_coefficient),
            )
        if source_mode:
            bottom = lax.axis_index(axis_name) == 0
            scalar_tendency = scalar_tendency.at[0].add(
                jnp.where(bottom, scalar_surface_source, 0.0)
            )
        return (
            *momentum_tendency,
            scalar_tendency,
        )

    def fused_lasd_local(
        u,
        v,
        w_upper,
        lower_boundary,
        *arguments,
    ):
        momentum = dry_flow_context_local(u, v, w_upper, lower_boundary)
        return fused_lasd_from_context_core_local(momentum, *arguments)

    def fused_lasd_from_context_local(momentum, *arguments):
        return fused_lasd_from_context_core_local(momentum, *arguments)

    return (
        fused_lasd_local,
        fused_lasd_from_context_local,
    )
