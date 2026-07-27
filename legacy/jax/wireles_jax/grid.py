from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .config import Params
from .state import Operators


def _numpy_dtype(dtype: jnp.dtype) -> type[np.floating]:
    return np.float64 if jnp.dtype(dtype) == jnp.float64 else np.float32


def _pressure_tridiag(
    params: Params,
    kx: jnp.ndarray,
    ky_half: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dtype = _numpy_dtype(params.dtype)
    n = params.nz
    dz2 = params.dz * params.dz
    kx_np = np.asarray(kx, dtype=dtype)
    ky_np = np.asarray(ky_half, dtype=dtype)
    k2 = kx_np[:, None] * kx_np[:, None] + ky_np[None, :] * ky_np[None, :]

    shape = (kx_np.shape[0], ky_np.shape[0], n)
    a = np.zeros(shape, dtype=dtype)
    b = np.zeros(shape, dtype=dtype)
    c = np.zeros(shape, dtype=dtype)

    zero_k2 = np.abs(k2) < np.finfo(dtype).eps * 128.0
    b[:, :, 0] = -k2 - 1.0 / dz2
    c[:, :, 0] = 1.0 / dz2
    b[:, :, 0] = np.where(zero_k2, 1.0, b[:, :, 0])
    c[:, :, 0] = np.where(zero_k2, 0.0, c[:, :, 0])

    if n > 2:
        a[:, :, 1:-1] = 1.0 / dz2
        b[:, :, 1:-1] = -k2[:, :, None] - 2.0 / dz2
        c[:, :, 1:-1] = 1.0 / dz2

    if params.top_boundary_condition == "klemp_durran":
        kh = np.sqrt(k2)
        coefficient = np.divide(
            params.radiation_brunt_vaisala_internal,
            kh,
            out=np.zeros_like(kh),
            where=~zero_k2,
        )
        half_dz = 0.5 * params.dz
        alpha = coefficient * params.dt / half_dz
        boundary_diagonal = 1.0 / (
            half_dz * params.dz * (1.0 + alpha)
        )
        a[:, :, -1] = 1.0 / dz2
        b[:, :, -1] = -k2 - 1.0 / dz2 - boundary_diagonal
        a[:, :, -1] = np.where(zero_k2, 1.0 / dz2, a[:, :, -1])
        b[:, :, -1] = np.where(
            zero_k2, -1.0 / dz2, b[:, :, -1]
        )
    else:
        a[:, :, -1] = 1.0 / dz2
        b[:, :, -1] = -k2 - 1.0 / dz2

    inv_bet = np.zeros(shape, dtype=dtype)
    gam = np.zeros(shape, dtype=dtype)
    bet = b[:, :, 0]
    inv_bet[:, :, 0] = 1.0 / bet
    for j in range(1, n):
        gam[:, :, j] = c[:, :, j - 1] * inv_bet[:, :, j - 1]
        bet = b[:, :, j] - a[:, :, j] * gam[:, :, j]
        inv_bet[:, :, j] = 1.0 / bet

    return (
        jnp.asarray(a, dtype=params.dtype),
        jnp.asarray(b, dtype=params.dtype),
        jnp.asarray(c, dtype=params.dtype),
        jnp.asarray(inv_bet, dtype=params.dtype),
        jnp.asarray(gam, dtype=params.dtype),
    )


def _horizontal_grid_filter(params: Params) -> jnp.ndarray:
    x_mode = jnp.abs(jnp.fft.fftfreq(params.nx, d=1.0) * params.nx)
    y_mode = jnp.fft.rfftfreq(params.ny, d=1.0) * params.ny
    cutoff_x = float(np.rint(params.nx / (2.0 * params.fgr)))
    cutoff_y = float(np.rint(params.ny / (2.0 * params.fgr)))
    keep = (x_mode[:, None] < cutoff_x) & (y_mode[None, :] < cutoff_y)
    return keep[:, :, None].astype(params.dtype)


def _pressure_mode_filter(params: Params) -> jnp.ndarray:
    if not params.pressure_filter_nyquist:
        return jnp.ones((params.nx, params.ny // 2 + 1, 1), dtype=params.dtype)
    keep = np.ones((params.nx, params.ny // 2 + 1), dtype=bool)
    if params.nx % 2 == 0:
        keep[params.nx // 2, :] = False
    if params.ny % 2 == 0:
        keep[:, -1] = False
    return jnp.asarray(keep[:, :, None], dtype=params.dtype)


def make_operators(params: Params) -> Operators:
    dtype = params.dtype
    kx = (2.0 * jnp.pi * jnp.fft.fftfreq(params.nx, d=params.dx)).astype(dtype)
    ky = (2.0 * jnp.pi * jnp.fft.fftfreq(params.ny, d=params.dy)).astype(dtype)
    if params.nx % 2 == 0:
        kx = kx.at[params.nx // 2].set(0.0)
    if params.ny % 2 == 0:
        ky = ky.at[params.ny // 2].set(0.0)
    kx_rfft = kx[:, None, None]
    ky_half = (2.0 * jnp.pi * jnp.fft.rfftfreq(params.ny, d=params.dy)).astype(dtype)
    if params.ny % 2 == 0:
        ky_half = ky_half.at[-1].set(0.0)
    ky_rfft = ky_half[None, :, None]
    horizontal_cutoff_rfft = _horizontal_grid_filter(params)
    pressure_a, pressure_b, pressure_c, pressure_inv_bet, pressure_gam = _pressure_tridiag(params, kx, ky_half)
    pressure_mode_keep = _pressure_mode_filter(params)
    return Operators(
        kx=kx[:, None, None],
        ky=ky[None, :, None],
        kx_rfft=kx_rfft,
        ky_rfft=ky_rfft,
        horizontal_cutoff_rfft=horizontal_cutoff_rfft,
        pressure_a=pressure_a,
        pressure_b=pressure_b,
        pressure_c=pressure_c,
        pressure_inv_bet=pressure_inv_bet,
        pressure_gam=pressure_gam,
        pressure_mode_keep=pressure_mode_keep,
    )


def interior(q: jnp.ndarray) -> jnp.ndarray:
    """Return a center or owned-face field unchanged.

    Kept as a compatibility name while callers migrate away from the old
    persistent-ghost layout.
    """
    return q


def with_interior(template: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    del template
    return values


def center_z(params: Params, dtype: jnp.dtype | None = None) -> jnp.ndarray:
    dtype = params.dtype if dtype is None else dtype
    return (jnp.arange(params.nz, dtype=dtype) + 0.5) * params.dz


def upper_face_z(params: Params, dtype: jnp.dtype | None = None) -> jnp.ndarray:
    """Coordinates of the upper face owned by each center cell."""
    dtype = params.dtype if dtype is None else dtype
    return (jnp.arange(params.nz, dtype=dtype) + 1.0) * params.dz


def lower_from_upper(
    upper: jnp.ndarray,
    bottom: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    plane_shape = upper.shape[:2] + upper.shape[3:]
    bottom_plane = jnp.broadcast_to(
        jnp.asarray(bottom, dtype=upper.dtype), plane_shape
    )
    bottom_plane = jnp.expand_dims(bottom_plane, axis=2)
    return jnp.concatenate((bottom_plane, upper[:, :, :-1, ...]), axis=2)


def upper_face_to_center(
    upper: jnp.ndarray,
    bottom: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    return 0.5 * (lower_from_upper(upper, bottom) + upper)


def face_gradient_to_center(
    upper: jnp.ndarray,
    bottom: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Average face gradients to centers while preserving the wall value.

    The old staggered-grid implementation used the wall-model gradient
    directly at the first cell center, then averaged the two surrounding
    face gradients at all higher centers.  This helper preserves that
    discrete semantic without storing a lower ghost plane.
    """
    centered = upper_face_to_center(upper, bottom)
    plane_shape = upper.shape[:2] + upper.shape[3:]
    bottom_plane = jnp.broadcast_to(
        jnp.asarray(bottom, dtype=upper.dtype), plane_shape
    )
    return centered.at[:, :, 0, ...].set(bottom_plane)


def divergence_upper_faces(
    upper_flux: jnp.ndarray,
    dz: float,
    bottom_flux: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    return (upper_flux - lower_from_upper(upper_flux, bottom_flux)) / dz


def center_to_upper_faces(q: jnp.ndarray) -> jnp.ndarray:
    """Interpolate centers to their owned upper faces.

    The last value is a boundary extrapolation and should be overwritten when
    a boundary condition supplies a more specific value.
    """
    inner = 0.5 * (q[:, :, :-1, ...] + q[:, :, 1:, ...])
    return jnp.concatenate((inner, q[:, :, -1:, ...]), axis=2)


def center_gradient(q: jnp.ndarray, dz: float) -> jnp.ndarray:
    if q.shape[2] == 1:
        return jnp.zeros_like(q)
    lower = (q[:, :, 1:2, ...] - q[:, :, :1, ...]) / dz
    upper = (q[:, :, -1:, ...] - q[:, :, -2:-1, ...]) / dz
    if q.shape[2] == 2:
        return jnp.concatenate((lower, upper), axis=2)
    middle = (q[:, :, 2:, ...] - q[:, :, :-2, ...]) / (2.0 * dz)
    return jnp.concatenate((lower, middle, upper), axis=2)


def gradient_to_upper_faces(
    q: jnp.ndarray,
    dz: float,
    top_gradient: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    inner = (q[:, :, 1:, ...] - q[:, :, :-1, ...]) / dz
    plane_shape = q.shape[:2] + q.shape[3:]
    top = jnp.broadcast_to(
        jnp.asarray(top_gradient, dtype=q.dtype), plane_shape
    )
    top = jnp.expand_dims(top, axis=2)
    return jnp.concatenate((inner, top), axis=2)
