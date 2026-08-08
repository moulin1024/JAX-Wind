"""Shared strain and Boussinesq source kernels for the z-slab backend."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def strain_magnitude_local(
    dudx,
    dudy,
    dudz,
    dvdx,
    dvdy,
    dvdz,
    dwdx,
    dwdy,
    dwdz,
):
    sxy = 0.5 * (dudy + dvdx)
    sxz = 0.5 * (dudz + dwdx)
    syz = 0.5 * (dvdz + dwdy)
    symmetric_dot = (
        dudx * dudx
        + dvdy * dvdy
        + dwdz * dwdz
        + 2.0 * (sxy * sxy + sxz * sxz + syz * syz)
    )
    return jnp.sqrt(jnp.maximum(2.0 * symmetric_dot, 0.0))


def build_boussinesq_source_kernels(*, grid, axis_name):
    def buoyancy_local(scalar, coefficient):
        local_coefficient = jnp.asarray(coefficient, dtype=scalar.theta.dtype)
        hydrostatic_free_theta = scalar.theta_upper - jnp.mean(
            scalar.theta_upper,
            axis=(-2, -1),
            keepdims=True,
        )
        z = local_coefficient * hydrostatic_free_theta
        return z.at[-1].set(jnp.where(scalar.upper_is_physical, 0.0, z[-1]))

    def rayleigh_damping_local(
        u,
        v,
        w_upper,
        start_height,
        maximum_rate,
        target_u,
        target_v,
    ):
        index = lax.axis_index(axis_name)
        local_nz = u.shape[0]
        global_cell = index * local_nz + jnp.arange(local_nz, dtype=u.dtype)
        cell_height = (global_cell + 0.5) * grid.dz
        upper_face_height = (global_cell + 1.0) * grid.dz
        depth = grid.lz - jnp.asarray(start_height, dtype=u.dtype)
        cell_eta = jnp.clip((cell_height - start_height) / depth, 0.0, 1.0)
        face_eta = jnp.clip(
            (upper_face_height - start_height) / depth,
            0.0,
            1.0,
        )
        cell_rate = jnp.asarray(maximum_rate, dtype=u.dtype) * cell_eta**2
        face_rate = jnp.asarray(maximum_rate, dtype=w_upper.dtype) * (
            face_eta.astype(w_upper.dtype) ** 2
        )
        return (
            -cell_rate[:, None, None] * (u - target_u),
            -cell_rate[:, None, None] * (v - target_v),
            -face_rate[:, None, None] * w_upper,
        )

    return buoyancy_local, rayleigh_damping_local
