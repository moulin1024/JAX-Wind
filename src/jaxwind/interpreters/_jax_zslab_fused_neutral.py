"""Fused LASD Boussinesq right-hand side."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


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
    dry_flow_context_local,
    dry_advection_from_padded_local,
    padded_momentum_gradients_local,
    dry_sgs_from_padded_gradients_local,
):
    if not isinstance(frozen_zero_scalar, bool):
        raise TypeError("frozen zero scalar flag must be boolean")

    def shared_momentum_local(
        u,
        v,
        w_upper,
        lower_boundary,
        wall_drag,
        wall_filtered,
        wall_filter_width,
    ):
        momentum = dry_flow_context_local(u, v, w_upper, lower_boundary)
        padded_momentum = pad_horizontal_local(
            jnp.stack(
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
        )
        padded_lower = pad_horizontal_local(
            jnp.stack((momentum.w_lower, momentum.u_lower, momentum.v_lower), axis=0)
        )
        advection = dry_advection_from_padded_local(
            momentum,
            padded_momentum,
            padded_lower,
        )
        cell_gradients, face_gradients = padded_momentum_gradients_local(
            padded_momentum,
            padded_lower,
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
            jnp.zeros_like(u).at[0].set(jnp.where(bottom, wall_x, 0.0)),
            jnp.zeros_like(v).at[0].set(jnp.where(bottom, wall_y, 0.0)),
            jnp.zeros_like(w_upper),
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

    def fused_lasd_local(
        u,
        v,
        w_upper,
        lower_boundary,
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
        use_imposed_sources,
    ):
        (
            momentum,
            advection,
            wall,
            cell_gradients,
            face_gradients,
            padded_momentum,
            padded_lower,
        ) = shared_momentum_local(
            u,
            v,
            w_upper,
            lower_boundary,
            wall_drag,
            wall_filtered,
            wall_filter_width,
        )
        if use_imposed_sources:
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
        if use_imposed_sources:
            bottom = lax.axis_index(axis_name) == 0
            scalar_tendency = scalar_tendency.at[0].add(
                jnp.where(bottom, imposed_scalar_source, 0.0)
            )
        return (
            *momentum_tendency,
            scalar_tendency,
        )

    return fused_lasd_local
