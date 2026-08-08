"""Staggered anisotropic minimum-dissipation kernels for the z-slab backend."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import lax


def build_amd_kernels(
    *,
    grid,
    axis_name,
    exchange_local,
    strain_magnitude_local,
    pad_horizontal_local,
    truncate_padded_local,
    horizontal_derivative_local,
):
    """Build the local unaveraged AMD closure used by the legacy solver."""

    def pad_context_local(context):
        return context._replace(
            dudx=pad_horizontal_local(context.dudx),
            dudy=pad_horizontal_local(context.dudy),
            dudz_at_cells=pad_horizontal_local(context.dudz_at_cells),
            dvdx=pad_horizontal_local(context.dvdx),
            dvdy=pad_horizontal_local(context.dvdy),
            dvdz_at_cells=pad_horizontal_local(context.dvdz_at_cells),
            dwdx_at_cells=pad_horizontal_local(context.dwdx_at_cells),
            dwdy_at_cells=pad_horizontal_local(context.dwdy_at_cells),
            dwdz=pad_horizontal_local(context.dwdz),
            dudz_upper=pad_horizontal_local(context.dudz_upper),
            dvdz_upper=pad_horizontal_local(context.dvdz_upper),
            dwdx_upper=pad_horizontal_local(context.dwdx_upper),
            dwdy_upper=pad_horizontal_local(context.dwdy_upper),
        )

    def context_from_padded_gradients_local(context, cell_gradients, face_gradients):
        return context._replace(
            dudx=cell_gradients[0],
            dudy=cell_gradients[1],
            dudz_at_cells=cell_gradients[2],
            dvdx=cell_gradients[3],
            dvdy=cell_gradients[4],
            dvdz_at_cells=cell_gradients[5],
            dwdx_at_cells=cell_gradients[6],
            dwdy_at_cells=cell_gradients[7],
            dwdz=cell_gradients[8],
            dudz_upper=face_gradients[2],
            dvdz_upper=face_gradients[5],
            dwdx_upper=face_gradients[6],
            dwdy_upper=face_gradients[7],
        )

    def eddy_viscosity_local(context):
        lower_dudz = 2.0 * context.dudz_at_cells - context.dudz_upper
        lower_dvdz = 2.0 * context.dvdz_at_cells - context.dvdz_upper
        lower_dwdx = 2.0 * context.dwdx_at_cells - context.dwdx_upper
        lower_dwdy = 2.0 * context.dwdy_at_cells - context.dwdy_upper

        def face_mean(lower_a, upper_a, lower_b, upper_b):
            return 0.5 * (lower_a * lower_b + upper_a * upper_b)

        s11 = context.dudx
        s22 = context.dvdy
        s33 = context.dwdz
        s12 = 0.5 * (context.dudy + context.dvdx)
        lower_s13 = 0.5 * (lower_dudz + lower_dwdx)
        upper_s13 = 0.5 * (context.dudz_upper + context.dwdx_upper)
        lower_s23 = 0.5 * (lower_dvdz + lower_dwdy)
        upper_s23 = 0.5 * (context.dvdz_upper + context.dwdy_upper)

        face_wx_s13 = face_mean(
            lower_dwdx, context.dwdx_upper, lower_s13, upper_s13
        )
        face_wx_s23 = face_mean(
            lower_dwdx, context.dwdx_upper, lower_s23, upper_s23
        )
        face_wy_s13 = face_mean(
            lower_dwdy, context.dwdy_upper, lower_s13, upper_s13
        )
        face_wy_s23 = face_mean(
            lower_dwdy, context.dwdy_upper, lower_s23, upper_s23
        )
        face_uz_s13 = face_mean(
            lower_dudz, context.dudz_upper, lower_s13, upper_s13
        )
        face_vz_s23 = face_mean(
            lower_dvdz, context.dvdz_upper, lower_s23, upper_s23
        )

        def horizontal_contraction(
            u_direction,
            v_direction,
            lower_w_direction,
            upper_w_direction,
            w_s13,
            w_s23,
        ):
            w_squared = face_mean(
                lower_w_direction,
                upper_w_direction,
                lower_w_direction,
                upper_w_direction,
            )
            return (
                s11 * u_direction**2
                + s22 * v_direction**2
                + s33 * w_squared
                + 2.0 * s12 * u_direction * v_direction
                + 2.0 * u_direction * w_s13
                + 2.0 * v_direction * w_s23
            )

        horizontal_x = horizontal_contraction(
            context.dudx,
            context.dvdx,
            lower_dwdx,
            context.dwdx_upper,
            face_wx_s13,
            face_wx_s23,
        )
        horizontal_y = horizontal_contraction(
            context.dudy,
            context.dvdy,
            lower_dwdy,
            context.dwdy_upper,
            face_wy_s13,
            face_wy_s23,
        )
        vertical = (
            s11
            * face_mean(
                lower_dudz,
                context.dudz_upper,
                lower_dudz,
                context.dudz_upper,
            )
            + s22
            * face_mean(
                lower_dvdz,
                context.dvdz_upper,
                lower_dvdz,
                context.dvdz_upper,
            )
            + s33**3
            + 2.0
            * s12
            * face_mean(
                lower_dudz,
                context.dudz_upper,
                lower_dvdz,
                context.dvdz_upper,
            )
            + 2.0 * s33 * face_uz_s13
            + 2.0 * s33 * face_vz_s23
        )

        length_x = grid.dx / math.sqrt(12.0)
        length_y = grid.dy / math.sqrt(12.0)
        length_z = grid.dz / math.sqrt(3.0)
        numerator = -(
            length_x**2 * horizontal_x
            + length_y**2 * horizontal_y
            + length_z**2 * vertical
        )
        denominator = (
            context.dudx**2
            + context.dvdx**2
            + face_mean(
                lower_dwdx,
                context.dwdx_upper,
                lower_dwdx,
                context.dwdx_upper,
            )
            + context.dudy**2
            + context.dvdy**2
            + face_mean(
                lower_dwdy,
                context.dwdy_upper,
                lower_dwdy,
                context.dwdy_upper,
            )
            + face_mean(
                lower_dudz,
                context.dudz_upper,
                lower_dudz,
                context.dudz_upper,
            )
            + face_mean(
                lower_dvdz,
                context.dvdz_upper,
                lower_dvdz,
                context.dvdz_upper,
            )
            + context.dwdz**2
        )
        valid = (
            (denominator > 0.0)
            & jnp.isfinite(numerator)
            & jnp.isfinite(denominator)
        )
        safe_denominator = jnp.where(valid, denominator, 1.0)
        return jnp.where(
            valid,
            jnp.maximum(numerator, 0.0) / safe_denominator,
            0.0,
        )

    def face_viscosity_local(viscosity):
        halo = exchange_local(viscosity[None, ...])
        next_plane = jnp.where(
            halo.upper_is_physical,
            viscosity[-1],
            halo.upper[0],
        )
        next_viscosity = jnp.concatenate(
            (viscosity[1:], next_plane[None]),
            axis=0,
        )
        return 0.5 * (viscosity + next_viscosity)

    def vertical_flux_local(context, viscosity):
        face_viscosity = face_viscosity_local(viscosity)
        txz = -face_viscosity * (context.dudz_upper + context.dwdx_upper)
        tyz = -face_viscosity * (context.dvdz_upper + context.dwdy_upper)
        txz = truncate_padded_local(txz)
        tyz = truncate_padded_local(tyz)
        txz = txz.at[-1].set(jnp.where(context.upper_is_physical, 0.0, txz[-1]))
        tyz = tyz.at[-1].set(jnp.where(context.upper_is_physical, 0.0, tyz[-1]))
        return txz, tyz

    def dry_amd_vertical_flux_local(context):
        padded_context = pad_context_local(context)
        return vertical_flux_local(
            padded_context,
            eddy_viscosity_local(padded_context),
        )

    def amd_tendency_from_padded_context_local(padded_context):
        viscosity = eddy_viscosity_local(padded_context)
        txx = truncate_padded_local(-2.0 * viscosity * padded_context.dudx)
        txy = truncate_padded_local(
            -viscosity * (padded_context.dudy + padded_context.dvdx)
        )
        tyy = truncate_padded_local(-2.0 * viscosity * padded_context.dvdy)
        tzz = truncate_padded_local(-2.0 * viscosity * padded_context.dwdz)
        txz, tyz = vertical_flux_local(padded_context, viscosity)

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
        x = -(
            horizontal_derivative_local(txx, 0)
            + horizontal_derivative_local(txy, 1)
            + (txz - lower_txz) / grid.dz
        )
        y = -(
            horizontal_derivative_local(txy, 0)
            + horizontal_derivative_local(tyy, 1)
            + (tyz - lower_tyz) / grid.dz
        )
        z = -(
            horizontal_derivative_local(txz, 0)
            + horizontal_derivative_local(tyz, 1)
            + (next_tzz - tzz) / grid.dz
        )
        z = z.at[-1].set(jnp.where(stress_halo.upper_is_physical, 0.0, z[-1]))
        return x, y, z

    def dry_amd_from_padded_gradients_local(context, cell_gradients, face_gradients):
        return amd_tendency_from_padded_context_local(
            context_from_padded_gradients_local(
                context,
                cell_gradients,
                face_gradients,
            )
        )

    def dry_amd_local(context):
        return amd_tendency_from_padded_context_local(pad_context_local(context))

    def amd_diagnostics_local(
        context,
        dissipation_coefficient,
        wall_gradient_factor,
    ):
        """Diagnose AMD eddy viscosity and local-equilibrium SGS energy."""
        padded_context = pad_context_local(context)
        viscosity = eddy_viscosity_local(padded_context)
        diagnostic_dudz = padded_context.dudz_at_cells
        diagnostic_dvdz = padded_context.dvdz_at_cells
        bottom = lax.axis_index(axis_name) == 0
        wall_dudz = pad_horizontal_local(context.u[0] * wall_gradient_factor)
        wall_dvdz = pad_horizontal_local(context.v[0] * wall_gradient_factor)
        diagnostic_dudz = diagnostic_dudz.at[0].set(
            jnp.where(
                bottom & (wall_gradient_factor > 0.0),
                wall_dudz,
                diagnostic_dudz[0],
            )
        )
        diagnostic_dvdz = diagnostic_dvdz.at[0].set(
            jnp.where(
                bottom & (wall_gradient_factor > 0.0),
                wall_dvdz,
                diagnostic_dvdz[0],
            )
        )
        magnitude = strain_magnitude_local(
            padded_context.dudx,
            padded_context.dudy,
            diagnostic_dudz,
            padded_context.dvdx,
            padded_context.dvdy,
            diagnostic_dvdz,
            padded_context.dwdx_at_cells,
            padded_context.dwdy_at_cells,
            padded_context.dwdz,
        )
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        production = jnp.maximum(viscosity * magnitude**2, 0.0)
        sgs_tke = (
            production * delta / jnp.asarray(dissipation_coefficient, magnitude.dtype)
        ) ** (2.0 / 3.0)
        return (
            jnp.maximum(truncate_padded_local(viscosity), 0.0),
            jnp.maximum(truncate_padded_local(sgs_tke), 0.0),
        )

    def amd_tke_transfer_local(context, wall_gradient_factor):
        """Return signed resolved-TKE transfer with the diagnostic wall shear."""
        padded_context = pad_context_local(context)
        viscosity = eddy_viscosity_local(padded_context)
        dudz = padded_context.dudz_at_cells
        dvdz = padded_context.dvdz_at_cells
        bottom = lax.axis_index(axis_name) == 0
        wall_dudz = pad_horizontal_local(context.u[0] * wall_gradient_factor)
        wall_dvdz = pad_horizontal_local(context.v[0] * wall_gradient_factor)
        dudz = dudz.at[0].set(
            jnp.where(
                bottom & (wall_gradient_factor > 0.0),
                wall_dudz,
                dudz[0],
            )
        )
        dvdz = dvdz.at[0].set(
            jnp.where(
                bottom & (wall_gradient_factor > 0.0),
                wall_dvdz,
                dvdz[0],
            )
        )
        magnitude = strain_magnitude_local(
            padded_context.dudx,
            padded_context.dudy,
            dudz,
            padded_context.dvdx,
            padded_context.dvdy,
            dvdz,
            padded_context.dwdx_at_cells,
            padded_context.dwdy_at_cells,
            padded_context.dwdz,
        )
        return truncate_padded_local(-viscosity * magnitude**2)

    def scalar_amd_local(
        scalar,
        momentum,
        turbulent_prandtl,
        lower_boundary_flux,
        upper_boundary_flux,
    ):
        padded_momentum = pad_context_local(momentum)
        viscosity = eddy_viscosity_local(padded_momentum)
        diffusivity = viscosity / turbulent_prandtl
        face_diffusivity = face_viscosity_local(viscosity) / turbulent_prandtl
        qx = truncate_padded_local(
            -diffusivity * pad_horizontal_local(scalar.dtheta_dx)
        )
        qy = truncate_padded_local(
            -diffusivity * pad_horizontal_local(scalar.dtheta_dy)
        )
        qz = truncate_padded_local(
            -face_diffusivity * pad_horizontal_local(scalar.dtheta_dz_upper)
        )
        qz = qz.at[-1].set(
            jnp.where(scalar.upper_is_physical, upper_boundary_flux, qz[-1])
        )
        flux_halo = exchange_local(qz[None, ...])
        lower_plane = jnp.where(
            flux_halo.lower_is_physical,
            jnp.full_like(qz[0], lower_boundary_flux),
            flux_halo.lower[0],
        )
        lower_qz = jnp.concatenate((lower_plane[None], qz[:-1]), axis=0)
        return -(
            horizontal_derivative_local(qx, 0)
            + horizontal_derivative_local(qy, 1)
            + (qz - lower_qz) / grid.dz
        )

    return (
        dry_amd_local,
        dry_amd_vertical_flux_local,
        scalar_amd_local,
        dry_amd_from_padded_gradients_local,
        amd_diagnostics_local,
        amd_tke_transfer_local,
    )
