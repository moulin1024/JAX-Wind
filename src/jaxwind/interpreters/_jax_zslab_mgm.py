"""Demo-aligned rotational convection and MGM kernels for the z-slab backend."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def mgm_clipping_coefficient(contraction, gkk, gradient_norm_epsilon):
    """Evaluate the Lu--Porte-Agel dynamic coefficient on horizontal planes."""
    valid = gkk > gradient_norm_epsilon
    safe_gkk = jnp.where(valid, gkk, 1.0)
    transfer = jnp.where(valid, -contraction / safe_gkk, 0.0)
    transfer_moment = transfer**3
    forward = valid & (contraction <= 0.0)
    valid_count = jnp.sum(valid, axis=(-2, -1))
    forward_count = jnp.sum(forward, axis=(-2, -1))
    conditional_mean = jnp.sum(
        jnp.where(forward, transfer_moment, 0.0), axis=(-2, -1)
    ) / jnp.maximum(forward_count, 1)
    unconditional_mean = jnp.sum(
        jnp.where(valid, transfer_moment, 0.0), axis=(-2, -1)
    ) / jnp.maximum(valid_count, 1)
    absolute_mean = jnp.sum(
        jnp.where(valid, jnp.abs(transfer_moment), 0.0), axis=(-2, -1)
    ) / jnp.maximum(valid_count, 1)
    denominator_floor = jnp.finfo(gkk.dtype).eps * jnp.maximum(
        absolute_mean,
        jnp.finfo(gkk.dtype).tiny,
    )
    coefficient_squared = conditional_mean / jnp.where(
        unconditional_mean > 0.0,
        unconditional_mean,
        1.0,
    )
    usable = (
        (valid_count > 0)
        & (forward_count > 0)
        & (unconditional_mean > denominator_floor)
        & jnp.isfinite(coefficient_squared)
        & (coefficient_squared > 0.0)
    )
    return jnp.sqrt(jnp.where(usable, coefficient_squared, 1.0))[:, None, None]


def build_mgm_kernels(
    *,
    grid,
    axis_name,
    exchange_local,
    wall_filter_local,
    strain_magnitude_local,
    pad_horizontal_local,
    truncate_padded_spectrum_local,
    horizontal_spectral_flux_divergence_local,
    horizontal_flux_divergence_local,
):
    """Build MGM kernels while retaining the factory's grid-specialized closures."""

    def wall_gradient_local(
        context,
        enabled,
        roughness_length,
        von_karman,
        filtered,
        filter_width,
    ):
        wall_velocity = wall_filter_local(
            jnp.stack((context.u[0], context.v[0])),
            filter_width,
        )
        wall_u = jnp.where(filtered, wall_velocity[0], context.u[0])
        wall_v = jnp.where(filtered, wall_velocity[1], context.v[0])
        speed = jnp.hypot(wall_u, wall_v)
        safe_speed = jnp.maximum(speed, jnp.finfo(speed.dtype).tiny)
        denominator = jnp.log(0.5 * grid.dz / roughness_length)
        friction_velocity = speed * von_karman / denominator
        gradient = (
            jnp.stack((wall_u, wall_v))
            * friction_velocity
            / (safe_speed * von_karman * 0.5 * grid.dz)
        )
        moving = speed > jnp.finfo(speed.dtype).tiny
        bottom = lax.axis_index(axis_name) == 0
        return jnp.where(enabled & bottom & moving[None], gradient, 0.0)

    def dry_rotational_advection_local(
        context,
        wall_gradient_enabled,
        roughness_length,
        von_karman,
        wall_filtered,
        wall_filter_width,
    ):
        """Reference rotational convection on the hybrid cell/face layout."""
        wall_gradient = wall_gradient_local(
            context,
            wall_gradient_enabled,
            roughness_length,
            von_karman,
            wall_filtered,
            wall_filter_width,
        )
        lower_w = 2.0 * context.w_at_cells - context.w_upper
        lower_dudz = 2.0 * context.dudz_at_cells - context.dudz_upper
        lower_dvdz = 2.0 * context.dvdz_at_cells - context.dvdz_upper
        lower_dwdx = 2.0 * context.dwdx_at_cells - context.dwdx_upper
        lower_dwdy = 2.0 * context.dwdy_at_cells - context.dwdy_upper
        bottom = lax.axis_index(axis_name) == 0
        lower_dudz = lower_dudz.at[0].set(
            jnp.where(bottom & wall_gradient_enabled, wall_gradient[0], lower_dudz[0])
        )
        lower_dvdz = lower_dvdz.at[0].set(
            jnp.where(bottom & wall_gradient_enabled, wall_gradient[1], lower_dvdz[0])
        )
        x = context.v * (context.dudy - context.dvdx)
        x += 0.5 * (
            context.w_upper * (context.dudz_upper - context.dwdx_upper)
            + lower_w * (lower_dudz - lower_dwdx)
        )
        y = context.u * (context.dvdx - context.dudy)
        y += 0.5 * (
            context.w_upper * (context.dvdz_upper - context.dwdy_upper)
            + lower_w * (lower_dvdz - lower_dwdy)
        )
        z = context.u_upper * (context.dwdx_upper - context.dudz_upper)
        z += context.v_upper * (context.dwdy_upper - context.dvdz_upper)
        z = z.at[-1].set(jnp.where(context.upper_is_physical, 0.0, z[-1]))
        return -x, -y, -z

    def mgm_stress_and_energy_local(
        dudx,
        dudy,
        dudz,
        dvdx,
        dvdy,
        dvdz,
        dwdx,
        dwdy,
        dwdz,
        filter_grid_ratio,
        dissipation_coefficient,
        fallback_coefficient,
        gradient_norm_epsilon,
        kinematic_viscosity,
    ):
        s11 = dudx
        s22 = dvdy
        s33 = dwdz
        s12 = 0.5 * (dudy + dvdx)
        s13 = 0.5 * (dudz + dwdx)
        s23 = 0.5 * (dvdz + dwdy)

        y_weight = (grid.dy / grid.dx) ** 2
        z_weight = (grid.dz / (grid.dx * filter_grid_ratio)) ** 2
        g11 = dudx**2 + y_weight * dudy**2 + z_weight * dudz**2
        g22 = dvdx**2 + y_weight * dvdy**2 + z_weight * dvdz**2
        g33 = dwdx**2 + y_weight * dwdy**2 + z_weight * dwdz**2
        g12 = dudx * dvdx + y_weight * dudy * dvdy + z_weight * dudz * dvdz
        g13 = dudx * dwdx + y_weight * dudy * dwdy + z_weight * dudz * dwdz
        g23 = dvdx * dwdx + y_weight * dvdy * dwdy + z_weight * dvdz * dwdz
        gkk = g11 + g22 + g33
        valid = gkk > gradient_norm_epsilon
        safe_gkk = jnp.where(valid, gkk, 1.0)
        raw_contraction = (
            g11 * s11
            + g22 * s22
            + g33 * s33
            + 2.0 * (g12 * s12 + g13 * s13 + g23 * s23)
        )
        diagnosed_coefficient = mgm_clipping_coefficient(
            raw_contraction,
            gkk,
            gradient_norm_epsilon,
        )
        contraction = jnp.minimum(raw_contraction, 0.0)
        delta = (
            filter_grid_ratio * grid.dx * filter_grid_ratio * grid.dy * grid.dz
        ) ** (1.0 / 3.0)
        ce = dissipation_coefficient * diagnosed_coefficient
        ksgs = (2.0 * delta / ce) ** 2 * (contraction / safe_gkk) ** 2
        modulation = 2.0 * ksgs / safe_gkk
        molecular = 2.0 * kinematic_viscosity
        magnitude = strain_magnitude_local(
            dudx,
            dudy,
            dudz,
            dvdx,
            dvdy,
            dvdz,
            dwdx,
            dwdy,
            dwdz,
        )
        fallback = -2.0 * (
            fallback_coefficient**2 * delta**2 * magnitude + kinematic_viscosity
        )

        def component(gradient, strain):
            modeled = modulation * gradient - molecular * strain
            return jnp.where(valid, modeled, fallback * strain)

        stresses = (
            component(g11, s11),
            component(g12, s12),
            component(g13, s13),
            component(g22, s22),
            component(g23, s23),
            component(g33, s33),
        )
        return stresses, jnp.where(valid, ksgs, 0.0)

    def mgm_stress_local(*args):
        stresses, _ = mgm_stress_and_energy_local(*args)
        return stresses

    def dealiased_mgm_stress_local(
        dudx,
        dudy,
        dudz,
        dvdx,
        dvdy,
        dvdz,
        dwdx,
        dwdy,
        dwdz,
        filter_grid_ratio,
        dissipation_coefficient,
        fallback_coefficient,
        gradient_norm_epsilon,
        kinematic_viscosity,
        component_indices,
    ):
        padded_gradients = tuple(
            pad_horizontal_local(
                jnp.stack(
                    (
                        dudx,
                        dudy,
                        dudz,
                        dvdx,
                        dvdy,
                        dvdz,
                        dwdx,
                        dwdy,
                        dwdz,
                    ),
                    axis=0,
                )
            )
        )
        return projected_mgm_stress_local(
            padded_gradients,
            filter_grid_ratio,
            dissipation_coefficient,
            fallback_coefficient,
            gradient_norm_epsilon,
            kinematic_viscosity,
            component_indices,
        )

    def projected_mgm_stress_local(
        padded_gradients,
        filter_grid_ratio,
        dissipation_coefficient,
        fallback_coefficient,
        gradient_norm_epsilon,
        kinematic_viscosity,
        component_indices,
    ):
        spectra = projected_mgm_stress_spectra_local(
            padded_gradients,
            filter_grid_ratio,
            dissipation_coefficient,
            fallback_coefficient,
            gradient_norm_epsilon,
            kinematic_viscosity,
            component_indices,
        )
        stresses = jnp.fft.irfftn(
            jnp.stack(spectra),
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(padded_gradients[0].dtype)
        return tuple(stresses)

    def projected_mgm_stress_spectra_local(
        padded_gradients,
        filter_grid_ratio,
        dissipation_coefficient,
        fallback_coefficient,
        gradient_norm_epsilon,
        kinematic_viscosity,
        component_indices,
    ):
        padded_stresses = mgm_stress_local(
            *padded_gradients,
            filter_grid_ratio,
            dissipation_coefficient,
            fallback_coefficient,
            gradient_norm_epsilon,
            kinematic_viscosity,
        )
        selected = jnp.stack(
            tuple(padded_stresses[index] for index in component_indices),
            axis=0,
        )
        return tuple(truncate_padded_spectrum_local(selected))

    def dry_mgm_from_padded_gradients_local(
        context,
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
    ):
        wall_gradient = pad_horizontal_local(
            wall_gradient_local(
                context,
                wall_gradient_enabled,
                roughness_length,
                von_karman,
                wall_filtered,
                wall_filter_width,
            )
        )
        bottom = lax.axis_index(axis_name) == 0
        cell_gradients = list(cell_gradients)
        cell_gradients[2] = (
            cell_gradients[2]
            .at[0]
            .set(
                jnp.where(
                    bottom & wall_gradient_enabled,
                    wall_gradient[0],
                    cell_gradients[2][0],
                )
            )
        )
        cell_gradients[5] = (
            cell_gradients[5]
            .at[0]
            .set(
                jnp.where(
                    bottom & wall_gradient_enabled,
                    wall_gradient[1],
                    cell_gradients[5][0],
                )
            )
        )
        cell_gradients[8] = -(cell_gradients[0] + cell_gradients[4])
        txx_s, txy_s, tyy_s, tzz_s = projected_mgm_stress_spectra_local(
            tuple(cell_gradients),
            filter_grid_ratio,
            dissipation_coefficient,
            fallback_coefficient,
            gradient_norm_epsilon,
            kinematic_viscosity,
            (0, 1, 3, 5),
        )
        txz_s, tyz_s = projected_mgm_stress_spectra_local(
            face_gradients,
            filter_grid_ratio,
            dissipation_coefficient,
            fallback_coefficient,
            gradient_norm_epsilon,
            kinematic_viscosity,
            (2, 4),
        )
        txz_s = txz_s.at[-1].set(
            jnp.where(context.upper_is_physical, 0.0, txz_s[-1])
        )
        tyz_s = tyz_s.at[-1].set(
            jnp.where(context.upper_is_physical, 0.0, tyz_s[-1])
        )
        vertical_stresses = jnp.fft.irfftn(
            jnp.stack((txz_s, tyz_s, tzz_s)),
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(context.u.dtype)
        txz, tyz, tzz = tuple(vertical_stresses)
        stress_halo = exchange_local(jnp.stack((txz, tyz, tzz), axis=0))
        lower_txz_plane = jnp.where(
            stress_halo.lower_is_physical,
            jnp.zeros_like(txz[0]),
            stress_halo.lower[0],
        )
        lower_tyz_plane = jnp.where(
            stress_halo.lower_is_physical,
            jnp.zeros_like(tyz[0]),
            stress_halo.lower[1],
        )
        lower_txz = jnp.concatenate((lower_txz_plane[None], txz[:-1]), axis=0)
        lower_tyz = jnp.concatenate((lower_tyz_plane[None], tyz[:-1]), axis=0)
        next_tzz_plane = jnp.where(
            stress_halo.upper_is_physical,
            tzz[-1],
            stress_halo.upper[2],
        )
        next_tzz = jnp.concatenate((tzz[1:], next_tzz_plane[None]), axis=0)
        horizontal = horizontal_spectral_flux_divergence_local(
            jnp.stack((txx_s, txy_s, txz_s)),
            jnp.stack((txy_s, tyy_s, tyz_s)),
            context.u.dtype,
        )
        x = -(horizontal[0] + (txz - lower_txz) / grid.dz)
        y = -(horizontal[1] + (tyz - lower_tyz) / grid.dz)
        z = -(horizontal[2] + (next_tzz - tzz) / grid.dz)
        z = z.at[-1].set(jnp.where(stress_halo.upper_is_physical, 0.0, z[-1]))
        return x, y, z

    def diagnostic_padded_gradients_local(
        context,
        wall_gradient_enabled,
        roughness_length,
        von_karman,
        wall_filtered,
        wall_filter_width,
    ):
        padded_gradients = list(
            pad_horizontal_local(
                jnp.stack(
                    (
                        context.dudx,
                        context.dudy,
                        context.dudz_at_cells,
                        context.dvdx,
                        context.dvdy,
                        context.dvdz_at_cells,
                        context.dwdx_at_cells,
                        context.dwdy_at_cells,
                        context.dwdz,
                    ),
                    axis=0,
                )
            )
        )
        wall_gradient = pad_horizontal_local(
            wall_gradient_local(
                context,
                wall_gradient_enabled,
                roughness_length,
                von_karman,
                wall_filtered,
                wall_filter_width,
            )
        )
        bottom = lax.axis_index(axis_name) == 0
        padded_gradients[2] = padded_gradients[2].at[0].set(
            jnp.where(
                bottom & wall_gradient_enabled,
                wall_gradient[0],
                padded_gradients[2][0],
            )
        )
        padded_gradients[5] = padded_gradients[5].at[0].set(
            jnp.where(
                bottom & wall_gradient_enabled,
                wall_gradient[1],
                padded_gradients[5][0],
            )
        )
        padded_gradients[8] = -(padded_gradients[0] + padded_gradients[4])
        return padded_gradients

    def dry_mgm_sgs_tke_local(
        context,
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
    ):
        """Return MGM's model-native diagnosed SGS kinetic energy."""
        padded_gradients = diagnostic_padded_gradients_local(
            context,
            wall_gradient_enabled,
            roughness_length,
            von_karman,
            wall_filtered,
            wall_filter_width,
        )
        _, padded_sgs_tke = mgm_stress_and_energy_local(
            *padded_gradients,
            filter_grid_ratio,
            dissipation_coefficient,
            fallback_coefficient,
            gradient_norm_epsilon,
            kinematic_viscosity,
        )
        spectrum = truncate_padded_spectrum_local(padded_sgs_tke[None])[0]
        sgs_tke = jnp.fft.irfftn(
            spectrum,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(context.u.dtype)
        return jnp.maximum(sgs_tke, 0.0)

    def dry_mgm_tke_transfer_local(
        context,
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
    ):
        """Return signed MGM resolved-TKE transfer from the padded stress."""
        gradients = diagnostic_padded_gradients_local(
            context,
            wall_gradient_enabled,
            roughness_length,
            von_karman,
            wall_filtered,
            wall_filter_width,
        )
        stresses, _ = mgm_stress_and_energy_local(
            *gradients,
            filter_grid_ratio,
            dissipation_coefficient,
            fallback_coefficient,
            gradient_norm_epsilon,
            kinematic_viscosity,
        )
        txx, txy, txz, tyy, tyz, tzz = stresses
        transfer = (
            txx * gradients[0]
            + txy * (gradients[1] + gradients[3])
            + txz * (gradients[2] + gradients[6])
            + tyy * gradients[4]
            + tyz * (gradients[5] + gradients[7])
            + tzz * gradients[8]
        )
        spectrum = truncate_padded_spectrum_local(transfer[None])[0]
        return jnp.fft.irfftn(
            spectrum,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(context.u.dtype)

    def dry_mgm_vertical_flux_local(
        context,
        filter_grid_ratio,
        dissipation_coefficient,
        fallback_coefficient,
        gradient_norm_epsilon,
        kinematic_viscosity,
    ):
        stresses = dealiased_mgm_stress_local(
            context.dudx_upper,
            context.dudy_upper,
            context.dudz_upper,
            context.dvdx_upper,
            context.dvdy_upper,
            context.dvdz_upper,
            context.dwdx_upper,
            context.dwdy_upper,
            context.dwdz_upper,
            filter_grid_ratio,
            dissipation_coefficient,
            fallback_coefficient,
            gradient_norm_epsilon,
            kinematic_viscosity,
            (2, 4),
        )
        txz, tyz = stresses
        txz = txz.at[-1].set(jnp.where(context.upper_is_physical, 0.0, txz[-1]))
        tyz = tyz.at[-1].set(jnp.where(context.upper_is_physical, 0.0, tyz[-1]))
        return txz, tyz

    def dry_mgm_local(
        context,
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
    ):
        wall_gradient = wall_gradient_local(
            context,
            wall_gradient_enabled,
            roughness_length,
            von_karman,
            wall_filtered,
            wall_filter_width,
        )
        bottom = lax.axis_index(axis_name) == 0
        dudz_at_cells = context.dudz_at_cells.at[0].set(
            jnp.where(
                bottom & wall_gradient_enabled,
                wall_gradient[0],
                context.dudz_at_cells[0],
            )
        )
        dvdz_at_cells = context.dvdz_at_cells.at[0].set(
            jnp.where(
                bottom & wall_gradient_enabled,
                wall_gradient[1],
                context.dvdz_at_cells[0],
            )
        )
        stresses = dealiased_mgm_stress_local(
            context.dudx,
            context.dudy,
            dudz_at_cells,
            context.dvdx,
            context.dvdy,
            dvdz_at_cells,
            context.dwdx_at_cells,
            context.dwdy_at_cells,
            -(context.dudx + context.dvdy),
            filter_grid_ratio,
            dissipation_coefficient,
            fallback_coefficient,
            gradient_norm_epsilon,
            kinematic_viscosity,
            (0, 1, 3, 5),
        )
        txx, txy, tyy, tzz = stresses
        txz, tyz = dry_mgm_vertical_flux_local(
            context,
            filter_grid_ratio,
            dissipation_coefficient,
            fallback_coefficient,
            gradient_norm_epsilon,
            kinematic_viscosity,
        )
        stress_halo = exchange_local(jnp.stack((txz, tyz, tzz), axis=0))
        lower_txz_plane = jnp.where(
            stress_halo.lower_is_physical,
            jnp.zeros_like(txz[0]),
            stress_halo.lower[0],
        )
        lower_tyz_plane = jnp.where(
            stress_halo.lower_is_physical,
            jnp.zeros_like(tyz[0]),
            stress_halo.lower[1],
        )
        lower_txz = jnp.concatenate((lower_txz_plane[None], txz[:-1]), axis=0)
        lower_tyz = jnp.concatenate((lower_tyz_plane[None], tyz[:-1]), axis=0)
        next_tzz_plane = jnp.where(
            stress_halo.upper_is_physical,
            tzz[-1],
            stress_halo.upper[2],
        )
        next_tzz = jnp.concatenate((tzz[1:], next_tzz_plane[None]), axis=0)
        horizontal = horizontal_flux_divergence_local(
            jnp.stack((txx, txy, txz), axis=0),
            jnp.stack((txy, tyy, tyz), axis=0),
        )
        x = -(horizontal[0] + (txz - lower_txz) / grid.dz)
        y = -(horizontal[1] + (tyz - lower_tyz) / grid.dz)
        z = -(horizontal[2] + (next_tzz - tzz) / grid.dz)
        z = z.at[-1].set(jnp.where(stress_halo.upper_is_physical, 0.0, z[-1]))
        return x, y, z

    return (
        dry_rotational_advection_local,
        dry_mgm_local,
        dry_mgm_vertical_flux_local,
        dry_mgm_from_padded_gradients_local,
        dry_mgm_sgs_tke_local,
        dry_mgm_tke_transfer_local,
    )
