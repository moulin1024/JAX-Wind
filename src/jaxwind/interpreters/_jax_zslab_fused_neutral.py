"""Fused neutral AMD and LASD Boussinesq right-hand sides."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def build_fused_neutral_boussinesq_kernels(
    *,
    grid,
    axis_name,
    frozen_zero_scalar,
    pad_horizontal_local,
    truncate_padded_local,
    wall_filter_local,
    dry_flow_context_local,
    dry_advection_from_padded_local,
    padded_momentum_gradients_local,
    dry_amd_from_padded_gradients_local,
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
        return momentum, advection, wall, cell_gradients, face_gradients

    def combine_local(advection, wall, sgs, pressure_x, pressure_y):
        return (
            advection[0] + wall[0] + sgs[0] + pressure_x,
            advection[1] + wall[1] + sgs[1] + pressure_y,
            advection[2] + wall[2] + sgs[2],
        )

    def fused_amd_local(
        u,
        v,
        w_upper,
        lower_boundary,
        theta,
        pressure_x,
        pressure_y,
        wall_drag,
        wall_filtered,
        wall_filter_width,
    ):
        momentum, advection, wall, cell_gradients, face_gradients = (
            shared_momentum_local(
                u,
                v,
                w_upper,
                lower_boundary,
                wall_drag,
                wall_filtered,
                wall_filter_width,
            )
        )
        sgs = dry_amd_from_padded_gradients_local(
            momentum,
            cell_gradients,
            face_gradients,
        )
        return (
            *combine_local(advection, wall, sgs, pressure_x, pressure_y),
            jnp.zeros_like(theta),
        )

    def fused_lasd_local(
        u,
        v,
        w_upper,
        lower_boundary,
        theta,
        coefficient,
        pressure_x,
        pressure_y,
        wall_drag,
        wall_filtered,
        wall_filter_width,
        minimum_coefficient,
        maximum_coefficient,
    ):
        momentum, advection, wall, cell_gradients, face_gradients = (
            shared_momentum_local(
                u,
                v,
                w_upper,
                lower_boundary,
                wall_drag,
                wall_filtered,
                wall_filter_width,
            )
        )
        sgs = dry_sgs_from_padded_gradients_local(
            momentum,
            cell_gradients,
            face_gradients,
            coefficient,
            minimum_coefficient,
            maximum_coefficient,
        )
        return (
            *combine_local(advection, wall, sgs, pressure_x, pressure_y),
            jnp.zeros_like(theta),
        )

    return fused_amd_local, fused_lasd_local
