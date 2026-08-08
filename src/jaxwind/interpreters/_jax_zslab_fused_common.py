"""Shared padded state for fused neutral Boussinesq right-hand sides."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import lax


def build_padded_momentum_gradients_kernel(
    *,
    grid,
    axis_name,
    cells_per_shard,
    porte_agel_wall_correction,
    padded_horizontal_gradient_pair_local,
):
    """Build gradients shared by conservative advection and SGS closures."""

    def padded_momentum_gradients_local(padded, padded_lower):
        padded_u, padded_v, padded_w_upper = padded[:3]
        padded_u_upper, padded_v_upper = padded[3:5]
        padded_w_at_cells, padded_w_next_cell = padded[5:]
        horizontal_x, horizontal_y = padded_horizontal_gradient_pair_local(
            jnp.stack(
                (
                    padded_u,
                    padded_v,
                    padded_w_at_cells,
                    padded_u_upper,
                    padded_v_upper,
                    padded_w_upper,
                ),
                axis=0,
            )
        )
        lower_u_faces = jnp.concatenate(
            (padded_lower[1][None], padded_u_upper[:-1]), axis=0
        )
        lower_v_faces = jnp.concatenate(
            (padded_lower[2][None], padded_v_upper[:-1]), axis=0
        )
        dudz_at_cells = (padded_u_upper - lower_u_faces) / grid.dz
        dvdz_at_cells = (padded_v_upper - lower_v_faces) / grid.dz
        dudz_upper = 2.0 * (padded_u_upper - padded_u) / grid.dz
        dvdz_upper = 2.0 * (padded_v_upper - padded_v) / grid.dz
        bottom = lax.axis_index(axis_name) == 0
        porte_agel_factor = 1.0 / math.log(3.0) - 1.0
        dudz_correction = jnp.where(
            bottom & porte_agel_wall_correction,
            porte_agel_factor * jnp.mean(dudz_upper[0]),
            0.0,
        )
        dvdz_correction = jnp.where(
            bottom & porte_agel_wall_correction,
            porte_agel_factor * jnp.mean(dvdz_upper[0]),
            0.0,
        )
        dudz_upper = dudz_upper.at[0].add(dudz_correction)
        dvdz_upper = dvdz_upper.at[0].add(dvdz_correction)
        dudz_at_cells = dudz_at_cells.at[0].add(0.5 * dudz_correction)
        dvdz_at_cells = dvdz_at_cells.at[0].add(0.5 * dvdz_correction)
        if cells_per_shard > 1:
            dudz_at_cells = dudz_at_cells.at[1].add(0.5 * dudz_correction)
            dvdz_at_cells = dvdz_at_cells.at[1].add(0.5 * dvdz_correction)
        dwdz = 2.0 * (padded_w_upper - padded_w_at_cells) / grid.dz
        dwdz_upper = (padded_w_next_cell - padded_w_at_cells) / grid.dz
        return (
            (
                horizontal_x[0],
                horizontal_y[0],
                dudz_at_cells,
                horizontal_x[1],
                horizontal_y[1],
                dvdz_at_cells,
                horizontal_x[2],
                horizontal_y[2],
                dwdz,
            ),
            (
                horizontal_x[3],
                horizontal_y[3],
                dudz_upper,
                horizontal_x[4],
                horizontal_y[4],
                dvdz_upper,
                horizontal_x[5],
                horizontal_y[5],
                dwdz_upper,
            ),
        )

    return padded_momentum_gradients_local
