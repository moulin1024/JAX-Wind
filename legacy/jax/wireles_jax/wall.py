from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Params


def filter_2d_wall(q: jax.Array, params: Params) -> jax.Array:
    filtered = filter_2d_wall_many((q,), params)[0]
    return filtered.astype(params.dtype)


def filter_2d_wall_many(qs: tuple[jax.Array, ...], params: Params) -> tuple[jax.Array, ...]:
    q_stack = jnp.stack(qs, axis=0)
    q_hat = jnp.fft.rfft2(q_stack, axes=(1, 2))
    x_mode = jnp.fft.fftfreq(params.nx, d=1.0) * params.nx
    y_mode = jnp.fft.rfftfreq(params.ny, d=1.0) * params.ny
    cutoff_x = jnp.floor(params.nx / (2.0 * params.fgr * params.tfr))
    cutoff_y = jnp.floor(params.ny / (2.0 * params.fgr * params.tfr))
    keep = (jnp.abs(x_mode)[:, None] < cutoff_x) & (y_mode[None, :] < cutoff_y)
    filtered = jnp.fft.irfft2(
        jnp.where(keep[None, :, :], q_hat, 0.0),
        s=(params.nx, params.ny),
        axes=(1, 2),
    ).real
    return tuple(filtered[i].astype(params.dtype) for i in range(len(qs)))


def apply_porte_agel_wall_correction(
    dudz: jax.Array,
    dvdz: jax.Array,
    *,
    correction_index: int | None = None,
    horizontal_average: bool = True,
) -> tuple[jax.Array, jax.Array]:
    fr1 = 1.0 / jnp.log(jnp.asarray(3.0, dtype=dudz.dtype)) - 1.0
    if correction_index is None:
        correction_index = min(1, dudz.shape[2] - 1)
    correction_index = min(correction_index, dudz.shape[2] - 1)
    dudz_plane = dudz[:, :, correction_index]
    dvdz_plane = dvdz[:, :, correction_index]
    dudz_correction = jnp.mean(dudz_plane) if horizontal_average else dudz_plane
    dvdz_correction = jnp.mean(dvdz_plane) if horizontal_average else dvdz_plane
    dudz = dudz.at[:, :, correction_index].add(fr1 * dudz_correction)
    dvdz = dvdz.at[:, :, correction_index].add(fr1 * dvdz_correction)
    return dudz, dvdz


def wall_stress(
    u: jax.Array,
    v: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if params.momentum_wall_model != "abl":
        zero = jnp.zeros((params.nx, params.ny), dtype=params.dtype)
        return zero, zero, zero, zero, zero

    u0, v0 = filter_2d_wall_many((u[:, :, 0], v[:, :, 0]), params)
    eps = jnp.asarray(1.0e-12, dtype=params.dtype)
    speed = jnp.sqrt(u0 * u0 + v0 * v0)
    denom = jnp.log(params.wall_ref_height / params.zo)
    valid = (speed > eps) & (jnp.abs(denom) > eps)
    safe_speed = jnp.where(speed > eps, speed, 1.0)
    safe_denom = jnp.where(jnp.abs(denom) > eps, denom, 1.0)
    if params.wall_stress_model == "prescribed_ustar":
        ustar = jnp.where(valid, params.u_fric, 0.0)
    else:
        ustar = jnp.where(valid, speed * params.vonk / safe_denom, 0.0)
    tau = -(ustar * ustar)
    txz0 = jnp.where(valid, tau * u0 / safe_speed, 0.0)
    tyz0 = jnp.where(valid, tau * v0 / safe_speed, 0.0)
    shear_denom = safe_speed * params.vonk * 0.5 * params.dz
    dudz0 = jnp.where(valid, u0 * ustar / shear_denom, 0.0)
    dvdz0 = jnp.where(valid, v0 * ustar / shear_denom, 0.0)
    return (
        txz0.astype(params.dtype),
        tyz0.astype(params.dtype),
        dudz0.astype(params.dtype),
        dvdz0.astype(params.dtype),
        ustar.astype(params.dtype),
    )
