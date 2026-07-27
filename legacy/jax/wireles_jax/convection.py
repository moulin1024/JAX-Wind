from __future__ import annotations

import jax
import jax.numpy as jnp

from .grid import center_to_upper_faces, lower_from_upper


def convec(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    dudy: jax.Array,
    dudz_face: jax.Array,
    dvdx: jax.Array,
    dvdz_face: jax.Array,
    dwdx: jax.Array,
    dwdy: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    vertical_x_upper = w * (dudz_face - dwdx)
    vertical_y_upper = w * (dvdz_face - dwdy)
    cx = v * (dudy - dvdx) + 0.5 * (
        lower_from_upper(vertical_x_upper) + vertical_x_upper
    )
    cy = u * (dvdx - dudy) + 0.5 * (
        lower_from_upper(vertical_y_upper) + vertical_y_upper
    )

    u_face = center_to_upper_faces(u)
    v_face = center_to_upper_faces(v)
    cz = u_face * (dwdx - dudz_face) + v_face * (dwdy - dvdz_face)
    cz = cz.at[:, :, -1].set(0.0)
    return cx, cy, cz
