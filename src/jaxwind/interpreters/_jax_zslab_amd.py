"""Staggered anisotropic minimum-dissipation kernels for the z-slab backend."""

from __future__ import annotations

import math

import jax.numpy as jnp


def build_amd_kernels(
    *,
    grid,
    exchange_local,
    two_thirds_filter_local,
    truncated_derivative_local,
):
    """Build the local unaveraged AMD closure used by the legacy solver."""

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
        txz = txz.at[-1].set(jnp.where(context.upper_is_physical, 0.0, txz[-1]))
        tyz = tyz.at[-1].set(jnp.where(context.upper_is_physical, 0.0, tyz[-1]))
        return two_thirds_filter_local(txz), two_thirds_filter_local(tyz)

    def dry_amd_vertical_flux_local(context):
        return vertical_flux_local(context, eddy_viscosity_local(context))

    def dry_amd_local(context):
        viscosity = eddy_viscosity_local(context)
        txx = -2.0 * viscosity * context.dudx
        txy = -viscosity * (context.dudy + context.dvdx)
        tyy = -2.0 * viscosity * context.dvdy
        tzz = two_thirds_filter_local(-2.0 * viscosity * context.dwdz)
        txz, tyz = vertical_flux_local(context, viscosity)

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
            truncated_derivative_local(txx, 0)
            + truncated_derivative_local(txy, 1)
            + (txz - lower_txz) / grid.dz
        )
        y = -(
            truncated_derivative_local(txy, 0)
            + truncated_derivative_local(tyy, 1)
            + (tyz - lower_tyz) / grid.dz
        )
        z = -(
            truncated_derivative_local(txz, 0)
            + truncated_derivative_local(tyz, 1)
            + (next_tzz - tzz) / grid.dz
        )
        z = z.at[-1].set(jnp.where(stress_halo.upper_is_physical, 0.0, z[-1]))
        return x, y, z

    def scalar_amd_local(
        scalar,
        momentum,
        turbulent_prandtl,
        lower_boundary_flux,
        upper_boundary_flux,
    ):
        viscosity = eddy_viscosity_local(momentum)
        diffusivity = viscosity / turbulent_prandtl
        face_diffusivity = face_viscosity_local(viscosity) / turbulent_prandtl
        qx = -diffusivity * scalar.dtheta_dx
        qy = -diffusivity * scalar.dtheta_dy
        qz = -face_diffusivity * scalar.dtheta_dz_upper
        qz = qz.at[-1].set(
            jnp.where(scalar.upper_is_physical, upper_boundary_flux, qz[-1])
        )
        qz = two_thirds_filter_local(qz)
        flux_halo = exchange_local(qz[None, ...])
        lower_plane = jnp.where(
            flux_halo.lower_is_physical,
            jnp.full_like(qz[0], lower_boundary_flux),
            flux_halo.lower[0],
        )
        lower_qz = jnp.concatenate((lower_plane[None], qz[:-1]), axis=0)
        return -(
            truncated_derivative_local(qx, 0)
            + truncated_derivative_local(qy, 1)
            + (qz - lower_qz) / grid.dz
        )

    return dry_amd_local, dry_amd_vertical_flux_local, scalar_amd_local
