"""Static and LASD Smagorinsky stress kernels for the JAX solver."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def build_smagorinsky_kernels(
    *,
    grid,
    axis_name,
    exchange_local,
    strain_magnitude_local,
    pad_horizontal_local,
    truncate_padded_spectrum_local,
    truncate_padded_local,
    horizontal_spectral_flux_divergence_local,
):
    def padded_gradients_local(context, upper):
        if upper:
            gradients = (
                context.dudx_upper,
                context.dudy_upper,
                context.dudz_upper,
                context.dvdx_upper,
                context.dvdy_upper,
                context.dvdz_upper,
                context.dwdx_upper,
                context.dwdy_upper,
                context.dwdz_upper,
            )
        else:
            gradients = (
                context.dudx,
                context.dudy,
                context.dudz_at_cells,
                context.dvdx,
                context.dvdy,
                context.dvdz_at_cells,
                context.dwdx_at_cells,
                context.dwdy_at_cells,
                context.dwdz,
            )
        return tuple(pad_horizontal_local(jnp.stack(gradients, axis=0)))

    def face_coefficient_local(padded_coefficient):
        coefficient_halo = exchange_local(padded_coefficient[None, ...])
        next_coefficient_plane = jnp.where(
            coefficient_halo.upper_is_physical,
            padded_coefficient[-1],
            coefficient_halo.upper[0],
        )
        next_coefficient = jnp.concatenate(
            (padded_coefficient[1:], next_coefficient_plane[None]),
            axis=0,
        )
        return 0.5 * (padded_coefficient + next_coefficient)

    def viscosities_from_padded_gradients_local(
        cell_gradients,
        face_gradients,
        coefficient,
        minimum_coefficient,
        maximum_coefficient,
    ):
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        padded_coefficient = jnp.clip(
            pad_horizontal_local(coefficient),
            minimum_coefficient,
            maximum_coefficient,
        )
        cell_viscosity = (
            padded_coefficient
            * delta**2
            * strain_magnitude_local(*cell_gradients)
        )
        face_viscosity = (
            face_coefficient_local(padded_coefficient)
            * delta**2
            * strain_magnitude_local(*face_gradients)
        )
        return cell_viscosity, face_viscosity

    def vertical_flux_from_padded_local(context, face_gradients, face_viscosity):
        txz, tyz = truncate_padded_local(
            jnp.stack(
                (
                    -face_viscosity * (face_gradients[2] + face_gradients[6]),
                    -face_viscosity * (face_gradients[5] + face_gradients[7]),
                ),
                axis=0,
            )
        )
        txz = txz.at[-1].set(jnp.where(context.upper_is_physical, 0.0, txz[-1]))
        tyz = tyz.at[-1].set(jnp.where(context.upper_is_physical, 0.0, tyz[-1]))
        return txz, tyz

    def dry_sgs_from_padded_gradients_local(
        context,
        cell_gradients,
        face_gradients,
        coefficient,
        minimum_coefficient,
        maximum_coefficient,
    ):
        cell_viscosity, face_viscosity = viscosities_from_padded_gradients_local(
            cell_gradients,
            face_gradients,
            coefficient,
            minimum_coefficient,
            maximum_coefficient,
        )
        txx_s, txy_s, tyy_s, tzz_s = tuple(
            truncate_padded_spectrum_local(
                jnp.stack(
                    (
                        -2.0 * cell_viscosity * cell_gradients[0],
                        -cell_viscosity * (cell_gradients[1] + cell_gradients[3]),
                        -2.0 * cell_viscosity * cell_gradients[4],
                        -2.0 * cell_viscosity * cell_gradients[8],
                    ),
                    axis=0,
                )
            )
        )
        txz_s, tyz_s = tuple(
            truncate_padded_spectrum_local(
                jnp.stack(
                    (
                        -face_viscosity * (face_gradients[2] + face_gradients[6]),
                        -face_viscosity * (face_gradients[5] + face_gradients[7]),
                    ),
                    axis=0,
                )
            )
        )
        txz_s = txz_s.at[-1].set(
            jnp.where(context.upper_is_physical, 0.0, txz_s[-1])
        )
        tyz_s = tyz_s.at[-1].set(
            jnp.where(context.upper_is_physical, 0.0, tyz_s[-1])
        )
        txz, tyz, tzz = tuple(
            jnp.fft.irfftn(
                jnp.stack((txz_s, tyz_s, tzz_s)),
                s=(grid.ny, grid.nx),
                axes=(-2, -1),
            ).astype(coefficient.dtype)
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
        horizontal = horizontal_spectral_flux_divergence_local(
            jnp.stack((txx_s, txy_s, txz_s)),
            jnp.stack((txy_s, tyy_s, tyz_s)),
            coefficient.dtype,
        )
        x = -(horizontal[0] + (txz - lower_txz) / grid.dz)
        y = -(horizontal[1] + (tyz - lower_tyz) / grid.dz)
        z = -(horizontal[2] + (next_tzz - tzz) / grid.dz)
        z = z.at[-1].set(jnp.where(stress_halo.upper_is_physical, 0.0, z[-1]))
        return x, y, z

    def dry_sgs_vertical_flux_local(
        context,
        coefficient,
        minimum_coefficient,
        maximum_coefficient,
    ):
        face_gradients = padded_gradients_local(context, True)
        padded_coefficient = jnp.clip(
            pad_horizontal_local(coefficient),
            minimum_coefficient,
            maximum_coefficient,
        )
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        face_viscosity = (
            face_coefficient_local(padded_coefficient)
            * delta**2
            * strain_magnitude_local(*face_gradients)
        )
        return vertical_flux_from_padded_local(
            context,
            face_gradients,
            face_viscosity,
        )

    def dry_sgs_local(
        context,
        coefficient,
        minimum_coefficient,
        maximum_coefficient,
    ):
        return dry_sgs_from_padded_gradients_local(
            context,
            padded_gradients_local(context, False),
            padded_gradients_local(context, True),
            coefficient,
            minimum_coefficient,
            maximum_coefficient,
        )

    def dry_sgs_tke_transfer_local(
        context,
        coefficient,
        minimum_coefficient,
        maximum_coefficient,
        wall_gradient_factor,
    ):
        """Return signed resolved-TKE transfer, tau_ij * d_j u_i, at cells."""
        dudz = context.dudz_at_cells
        dvdz = context.dvdz_at_cells
        bottom = lax.axis_index(axis_name) == 0
        dudz = dudz.at[0].set(
            jnp.where(
                bottom & (wall_gradient_factor > 0.0),
                context.u[0] * wall_gradient_factor,
                dudz[0],
            )
        )
        dvdz = dvdz.at[0].set(
            jnp.where(
                bottom & (wall_gradient_factor > 0.0),
                context.v[0] * wall_gradient_factor,
                dvdz[0],
            )
        )
        gradients = tuple(
            pad_horizontal_local(
                jnp.stack(
                    (
                        context.dudx,
                        context.dudy,
                        dudz,
                        context.dvdx,
                        context.dvdy,
                        dvdz,
                        context.dwdx_at_cells,
                        context.dwdy_at_cells,
                        context.dwdz,
                    ),
                    axis=0,
                )
            )
        )
        padded_coefficient = jnp.clip(
            pad_horizontal_local(coefficient),
            minimum_coefficient,
            maximum_coefficient,
        )
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        magnitude = strain_magnitude_local(*gradients)
        forward_transfer = padded_coefficient * delta**2 * magnitude**3
        return truncate_padded_local(-forward_transfer)

    return (
        dry_sgs_local,
        dry_sgs_vertical_flux_local,
        dry_sgs_from_padded_gradients_local,
        dry_sgs_tke_transfer_local,
    )
