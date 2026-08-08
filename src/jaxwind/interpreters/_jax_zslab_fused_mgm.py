"""Fused neutral Boussinesq MGM right-hand side."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def build_fused_mgm_boussinesq_kernel(
    *,
    grid,
    axis_name,
    frozen_zero_scalar,
    exchange_local,
    strain_magnitude_local,
    pad_horizontal_local,
    truncate_padded_spectrum_local,
    truncate_padded_local,
    padded_horizontal_gradient_pair_local,
    horizontal_spectral_flux_divergence_local,
    padded_horizontal_flux_divergence_local,
    wall_filter_local,
    dry_flow_context_local,
    scalar_context_local,
    dry_advection_from_padded_local,
    padded_momentum_gradients_local,
    dry_mgm_from_padded_gradients_local,
):
    def scalar_tendency_local(
        theta,
        padded_momentum,
        padded_lower,
        cell_gradients,
        face_gradients,
        scalar_coefficient,
        lower_scalar_flux,
        upper_scalar_flux,
    ):
        scalar = scalar_context_local(theta)
        padded_scalar = pad_horizontal_local(
            jnp.stack((scalar.theta, scalar.theta_upper, scalar.theta_lower), axis=0)
        )
        padded_theta, padded_theta_upper, padded_theta_lower = padded_scalar
        padded_u, padded_v, padded_w_upper = padded_momentum[:3]
        padded_w_lower = jnp.concatenate(
            (padded_lower[0][None], padded_w_upper[:-1]), axis=0
        )
        scalar_horizontal = padded_horizontal_flux_divergence_local(
            (padded_u * padded_theta)[None],
            (padded_v * padded_theta)[None],
        )[0]
        scalar_upper_flux, scalar_lower_flux = truncate_padded_local(
            jnp.stack(
                (
                    padded_w_upper * padded_theta_upper,
                    padded_w_lower * padded_theta_lower,
                ),
                axis=0,
            )
        )
        scalar_advection = -(
            scalar_horizontal + (scalar_upper_flux - scalar_lower_flux) / grid.dz
        )

        cell_magnitude = strain_magnitude_local(*cell_gradients)
        face_magnitude = strain_magnitude_local(*face_gradients)
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        cell_diffusivity = scalar_coefficient * delta**2 * cell_magnitude
        face_diffusivity = scalar_coefficient * delta**2 * face_magnitude
        dtheta_dx, dtheta_dy = padded_horizontal_gradient_pair_local(padded_theta)
        dtheta_dz_upper = 2.0 * (padded_theta_upper - padded_theta) / grid.dz
        qx_s, qy_s, qz_s = truncate_padded_spectrum_local(
            jnp.stack(
                (
                    -cell_diffusivity * dtheta_dx,
                    -cell_diffusivity * dtheta_dy,
                    -face_diffusivity * dtheta_dz_upper,
                ),
                axis=0,
            )
        )
        qz = jnp.fft.irfftn(
            qz_s,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(theta.dtype)
        qz = qz.at[-1].set(
            jnp.where(scalar.upper_is_physical, upper_scalar_flux, qz[-1])
        )
        flux_halo = exchange_local(qz[None, ...])
        lower_qz_plane = jnp.where(
            flux_halo.lower_is_physical,
            jnp.full_like(qz[0], lower_scalar_flux),
            flux_halo.lower[0],
        )
        lower_qz = jnp.concatenate((lower_qz_plane[None], qz[:-1]), axis=0)
        scalar_sgs_horizontal = horizontal_spectral_flux_divergence_local(
            qx_s[None], qy_s[None], theta.dtype
        )[0]
        scalar_sgs = -(scalar_sgs_horizontal + (qz - lower_qz) / grid.dz)
        return scalar_advection + scalar_sgs

    def fused_mgm_boussinesq_local(
        u,
        v,
        w_upper,
        lower_boundary,
        theta,
        pressure_x_acceleration,
        pressure_y_acceleration,
        wall_drag,
        wall_filtered,
        wall_filter_width,
        filter_grid_ratio,
        dissipation_coefficient,
        fallback_coefficient,
        gradient_norm_epsilon,
        kinematic_viscosity,
        wall_gradient_enabled,
        roughness_length,
        von_karman,
        scalar_coefficient,
        lower_scalar_flux,
        upper_scalar_flux,
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
        sgs = dry_mgm_from_padded_gradients_local(
            momentum,
            cell_gradients,
            face_gradients,
            filter_grid_ratio,
            dissipation_coefficient,
            fallback_coefficient,
            gradient_norm_epsilon,
            kinematic_viscosity,
            wall_gradient_enabled,
            roughness_length,
            von_karman,
            wall_filtered,
            wall_filter_width,
        )
        x = advection[0] + wall[0] + sgs[0] + pressure_x_acceleration
        y = advection[1] + wall[1] + sgs[1] + pressure_y_acceleration
        z = advection[2] + wall[2] + sgs[2]

        if frozen_zero_scalar:
            scalar_tendency = jnp.zeros_like(theta)
        else:
            scalar_tendency = scalar_tendency_local(
                theta,
                padded_momentum,
                padded_lower,
                cell_gradients,
                face_gradients,
                scalar_coefficient,
                lower_scalar_flux,
                upper_scalar_flux,
            )
        return x, y, z, scalar_tendency

    return fused_mgm_boussinesq_local
