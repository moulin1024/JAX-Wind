"""Three-halves conservative momentum advection for the z-slab backend."""

from __future__ import annotations

import jax.numpy as jnp


def build_conservative_advection_kernels(
    *,
    grid,
    pad_horizontal_local,
    truncate_padded_local,
    padded_horizontal_flux_divergence_local,
):
    def dry_advection_from_padded_local(context, padded, padded_lower):
        padded_u, padded_v, padded_w_upper = padded[:3]
        padded_u_upper, padded_v_upper = padded[3:5]
        padded_w_at_cells, padded_w_next_cell = padded[5:]
        padded_uv = padded_u * padded_v
        horizontal = padded_horizontal_flux_divergence_local(
            jnp.stack(
                (
                    padded_u * padded_u,
                    padded_uv,
                    padded_u_upper * padded_w_upper,
                ),
                axis=0,
            ),
            jnp.stack(
                (
                    padded_uv,
                    padded_v * padded_v,
                    padded_v_upper * padded_w_upper,
                ),
                axis=0,
            ),
        )
        upper_u_flux, upper_v_flux, vertical_flux, next_vertical_flux = (
            truncate_padded_local(
                jnp.stack(
                    (
                        padded_w_upper * padded_u_upper,
                        padded_w_upper * padded_v_upper,
                        padded_w_at_cells * padded_w_at_cells,
                        padded_w_next_cell * padded_w_next_cell,
                    ),
                    axis=0,
                )
            )
        )
        lower_u_flux_plane, lower_v_flux_plane = truncate_padded_local(
            jnp.stack(
                (
                    padded_lower[0] * padded_lower[1],
                    padded_lower[0] * padded_lower[2],
                ),
                axis=0,
            )
        )
        lower_u_flux = jnp.concatenate(
            (lower_u_flux_plane[None], upper_u_flux[:-1]),
            axis=0,
        )
        lower_v_flux = jnp.concatenate(
            (lower_v_flux_plane[None], upper_v_flux[:-1]),
            axis=0,
        )
        x = -(horizontal[0] + (upper_u_flux - lower_u_flux) / grid.dz)
        y = -(horizontal[1] + (upper_v_flux - lower_v_flux) / grid.dz)
        z = -(horizontal[2] + (next_vertical_flux - vertical_flux) / grid.dz)
        z = z.at[-1].set(jnp.where(context.upper_is_physical, 0.0, z[-1]))
        return x, y, z

    def dry_advection_local(context):
        padded = pad_horizontal_local(
            jnp.stack(
                (
                    context.u,
                    context.v,
                    context.w_upper,
                    context.u_upper,
                    context.v_upper,
                    context.w_at_cells,
                    context.w_next_cell,
                ),
                axis=0,
            )
        )
        padded_lower = pad_horizontal_local(
            jnp.stack((context.w_lower, context.u_lower, context.v_lower), axis=0)
        )
        return dry_advection_from_padded_local(context, padded, padded_lower)

    return dry_advection_local, dry_advection_from_padded_local
