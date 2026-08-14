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
    scalar_advection_from_padded_momentum_local,
    scalar_sgs_from_padded_momentum_gradients_local,
    buoyancy_local,
    pad_horizontal_local,
    truncate_padded_local,
    wall_filter_local,
    exchange_local,
    dry_flow_context_local,
    dry_advection_from_padded_local,
    padded_momentum_gradients_local,
    dry_sgs_from_padded_gradients_local,
    reuse_state_filtered_context,
):
    if not isinstance(frozen_zero_scalar, bool):
        raise TypeError("frozen zero scalar flag must be boolean")
    if not isinstance(reuse_state_filtered_context, bool):
        raise TypeError("state-filtered context flag must be boolean")

    def dealiased_rotational_advection_local(
        momentum,
        padded_momentum,
        padded_lower,
        cell_gradients,
        face_gradients,
        padded_wall_u,
        padded_wall_v,
        wall_gradient_factor,
    ):
        """Evaluate ``-omega x u`` on the padded staggered layout."""

        padded_u, padded_v, padded_w_upper = padded_momentum[:3]
        padded_u_upper, padded_v_upper = padded_momentum[3:5]
        padded_lower_w = jnp.concatenate(
            (padded_lower[0][None], padded_w_upper[:-1]),
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
                wall_gradient_factor * padded_wall_u,
                lower_dudz[0],
            )
        )
        lower_dvdz = lower_dvdz.at[0].set(
            jnp.where(
                bottom,
                wall_gradient_factor * padded_wall_v,
                lower_dvdz[0],
            )
        )
        x = padded_v * (cell_gradients[1] - cell_gradients[3])
        x += 0.5 * (
            padded_w_upper * (face_gradients[2] - face_gradients[6])
            + padded_lower_w * (lower_dudz - lower_dwdx)
        )
        y = padded_u * (cell_gradients[3] - cell_gradients[1])
        y += 0.5 * (
            padded_w_upper * (face_gradients[5] - face_gradients[7])
            + padded_lower_w * (lower_dvdz - lower_dwdy)
        )
        z = padded_u_upper * (face_gradients[6] - face_gradients[2])
        z += padded_v_upper * (face_gradients[7] - face_gradients[5])
        x, y, z = truncate_padded_local(jnp.stack((-x, -y, -z), axis=0))
        z = z.at[-1].set(jnp.where(momentum.upper_is_physical, 0.0, z[-1]))
        return x, y, z

    def shared_momentum_from_context_local(
        momentum,
        wall_drag,
        wall_filtered,
        wall_filter_width,
        wall_gradient_factor,
        rotational_advection,
    ):
        # Horizontal padding is linear, so vertical shifts and averages commute
        # with it.  Pad only the three independent velocity fields and rebuild
        # the interpolated fields in padded space.  Besides avoiding seven
        # redundant base-to-padded transform batches, exchanging the padded
        # boundary planes keeps this construction valid across z partitions.
        padded_u, padded_v, padded_w_upper = pad_horizontal_local(
            jnp.stack((momentum.u, momentum.v, momentum.w_upper), axis=0)
        )
        padded_halo = exchange_local(
            jnp.stack((padded_u, padded_v, padded_w_upper), axis=0)
        )
        previous_u = jnp.where(
            padded_halo.lower_is_physical,
            padded_u[0],
            padded_halo.lower[0],
        )
        previous_v = jnp.where(
            padded_halo.lower_is_physical,
            padded_v[0],
            padded_halo.lower[1],
        )
        next_u_plane = jnp.where(
            padded_halo.upper_is_physical,
            padded_u[-1],
            padded_halo.upper[0],
        )
        next_v_plane = jnp.where(
            padded_halo.upper_is_physical,
            padded_v[-1],
            padded_halo.upper[1],
        )
        next_w_upper = jnp.where(
            padded_halo.upper_is_physical,
            padded_w_upper[-1],
            padded_halo.upper[2],
        )
        padded_lower_boundary = jnp.full_like(
            padded_w_upper[0],
            momentum.w_lower[0, 0],
        )
        padded_w_lower = jnp.where(
            padded_halo.lower_is_physical,
            padded_lower_boundary,
            padded_halo.lower[2],
        )
        padded_lower_faces = jnp.concatenate(
            (padded_w_lower[None], padded_w_upper[:-1]),
            axis=0,
        )
        padded_next_u = jnp.concatenate(
            (padded_u[1:], next_u_plane[None]),
            axis=0,
        )
        padded_next_v = jnp.concatenate(
            (padded_v[1:], next_v_plane[None]),
            axis=0,
        )
        padded_w_at_cells = 0.5 * (padded_lower_faces + padded_w_upper)
        padded_next_w_cell = jnp.concatenate(
            (
                padded_w_at_cells[1:],
                (0.5 * (padded_w_upper[-1] + next_w_upper))[None],
            ),
            axis=0,
        )
        padded_u_upper = 0.5 * (padded_u + padded_next_u)
        padded_v_upper = 0.5 * (padded_v + padded_next_v)
        padded_u_lower = 0.5 * (previous_u + padded_u[0])
        padded_v_lower = 0.5 * (previous_v + padded_v[0])
        padded_momentum = jnp.stack(
            (
                padded_u,
                padded_v,
                padded_w_upper,
                padded_u_upper,
                padded_v_upper,
                padded_w_at_cells,
                padded_next_w_cell,
            ),
            axis=0,
        )
        padded_lower = jnp.stack(
            (padded_w_lower, padded_u_lower, padded_v_lower),
            axis=0,
        )
        cell_gradients, face_gradients = padded_momentum_gradients_local(
            padded_momentum,
            padded_lower,
        )
        if reuse_state_filtered_context and rotational_advection:
            # The legacy path sharply filters every accepted velocity state.
            # Its existing interpolation and derivative context is therefore
            # already the exact base-grid context needed by rotational
            # advection and SGS; rebuilding it through pad/FFT/truncate is
            # redundant.  The generic values above become dead under this
            # static branch and XLA removes them.
            padded_momentum = jnp.stack(
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
            padded_lower = jnp.stack(
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
        padded_wall_u, padded_wall_v = pad_horizontal_local(
            jnp.stack((wall_u, wall_v), axis=0)
        )
        if rotational_advection:
            advection = dealiased_rotational_advection_local(
                momentum,
                padded_momentum,
                padded_lower,
                cell_gradients,
                face_gradients,
                padded_wall_u,
                padded_wall_v,
                wall_gradient_factor,
            )
        else:
            advection = dry_advection_from_padded_local(
                momentum,
                padded_momentum,
                padded_lower,
            )
        wall_speed = jnp.hypot(padded_wall_u, padded_wall_v)
        wall_x, wall_y = truncate_padded_local(
            jnp.stack(
                (
                    -wall_drag * wall_speed * padded_wall_u / grid.dz,
                    -wall_drag * wall_speed * padded_wall_v / grid.dz,
                ),
                axis=0,
            )
        )
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
            padded_momentum,
            padded_lower,
        )

    def shared_momentum_local(
        u,
        v,
        w_upper,
        lower_boundary,
        wall_drag,
        wall_filtered,
        wall_filter_width,
        wall_gradient_factor,
        rotational_advection,
    ):
        momentum = dry_flow_context_local(u, v, w_upper, lower_boundary)
        return shared_momentum_from_context_local(
            momentum,
            wall_drag,
            wall_filtered,
            wall_filter_width,
            wall_gradient_factor,
            rotational_advection,
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
        rotational_advection,
    ):
        (
            momentum,
            advection,
            wall,
            cell_gradients,
            face_gradients,
            padded_momentum,
            padded_lower,
        ) = shared_momentum_from_context_local(
            momentum,
            wall_drag,
            wall_filtered,
            wall_filter_width,
            wall_gradient_factor,
            rotational_advection,
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
        sgs = dry_sgs_from_padded_gradients_local(
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
            scalar_tendency = scalar_advection_from_padded_momentum_local(
                scalar,
                padded_momentum,
                padded_lower,
            )
            scalar_tendency = (
                scalar_tendency
                + scalar_sgs_from_padded_momentum_gradients_local(
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
        return fused_lasd_from_context_core_local(momentum, *arguments, False)

    def fused_lasd_from_context_local(momentum, *arguments):
        return fused_lasd_from_context_core_local(momentum, *arguments, False)

    def fused_rotational_lasd_local(
        u,
        v,
        w_upper,
        lower_boundary,
        *arguments,
    ):
        momentum = dry_flow_context_local(u, v, w_upper, lower_boundary)
        return fused_lasd_from_context_core_local(momentum, *arguments, True)

    def fused_rotational_lasd_from_context_local(momentum, *arguments):
        return fused_lasd_from_context_core_local(momentum, *arguments, True)

    return (
        fused_lasd_local,
        fused_lasd_from_context_local,
        fused_rotational_lasd_local,
        fused_rotational_lasd_from_context_local,
    )
