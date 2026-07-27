from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Params
from .grid import divergence_upper_faces, gradient_to_upper_faces
from .state import Operators


def apply_pressure_bc(p: jax.Array) -> jax.Array:
    return p


def _move_z_to_scan_axis(q: jax.Array) -> jax.Array:
    return jnp.moveaxis(q, -1, 0)


def _move_scan_to_z_axis(q: jax.Array) -> jax.Array:
    return jnp.moveaxis(q, 0, -1)


def _solve_tridiag(a: jax.Array, inv_bet: jax.Array, gam: jax.Array, rhs: jax.Array) -> jax.Array:
    u0 = rhs[..., 0] * inv_bet[..., 0]

    def forward(carry, values):
        u_prev = carry
        a_j, inv_bet_j, rhs_j = values
        u_j = (rhs_j - a_j * u_prev) * inv_bet_j
        return u_j, u_j

    _, u_tail_z = jax.lax.scan(
        forward,
        u0,
        (
            _move_z_to_scan_axis(a[..., 1:]),
            _move_z_to_scan_axis(inv_bet[..., 1:]),
            _move_z_to_scan_axis(rhs[..., 1:]),
        ),
    )
    u_forward = jnp.concatenate(
        (u0[..., None], _move_scan_to_z_axis(u_tail_z)),
        axis=-1,
    )

    def backward(next_u, values):
        u_j_forward, gam_next = values
        u_j = u_j_forward - gam_next * next_u
        return u_j, u_j

    _, u_prefix_rev_z = jax.lax.scan(
        backward,
        u_forward[..., -1],
        (
            _move_z_to_scan_axis(u_forward[..., :-1])[::-1],
            _move_z_to_scan_axis(gam[..., 1:])[::-1],
        ),
    )
    u_prefix = _move_scan_to_z_axis(u_prefix_rev_z[::-1])
    return jnp.concatenate((u_prefix, u_forward[..., -1:]), axis=-1)


def _radiation_pressure_coefficient(params: Params, ops: Operators) -> jax.Array:
    kh = jnp.sqrt(ops.kx_rfft * ops.kx_rfft + ops.ky_rfft * ops.ky_rfft)
    safe_kh = jnp.where(kh > 0.0, kh, 1.0)
    return jnp.where(
        kh > 0.0,
        params.radiation_brunt_vaisala_internal / safe_kh,
        0.0,
    )[..., 0]


def _prepare_radiation_top(
    w: jax.Array,
    params: Params,
    ops: Operators,
) -> tuple[jax.Array, jax.Array | None]:
    """Restrict the radiating top face to modes represented by the projection.

    The real-valued Fourier first derivative deliberately sets the isolated
    Nyquist modes to zero.  A top-face velocity in that null space cannot be
    balanced by the Klemp--Durran pressure impedance, so remove it before
    forming the projection divergence.
    """
    if params.top_boundary_condition != "klemp_durran":
        return w, None
    top_w_hat = jnp.fft.rfft2(w[:, :, -1], axes=(0, 1))
    coefficient = _radiation_pressure_coefficient(params, ops)
    top_w_hat = jnp.where(coefficient > 0.0, top_w_hat, 0.0)
    top_w = jnp.fft.irfft2(
        top_w_hat, s=(params.nx, params.ny), axes=(0, 1)
    ).real.astype(w.dtype)
    return w.at[:, :, -1].set(top_w), top_w_hat


def _solve_pressure_hat(
    rhs_hat: jax.Array,
    params: Params,
    ops: Operators,
    top_w_hat: jax.Array | None = None,
) -> jax.Array:
    rhs_col = rhs_hat
    rhs_col = rhs_col.at[0, 0, 0].set(0.0)
    if params.top_boundary_condition == "klemp_durran" and top_w_hat is not None:
        coefficient = _radiation_pressure_coefficient(params, ops).astype(rhs_hat.real.dtype)
        half_dz = 0.5 * params.dz
        alpha = coefficient * params.dt / half_dz
        forcing = coefficient * top_w_hat / (
            half_dz * params.dz * (1.0 + alpha)
        )
        rhs_col = rhs_col.at[..., -1].add(-forcing)
    p_col = _solve_tridiag(ops.pressure_a, ops.pressure_inv_bet, ops.pressure_gam, rhs_col)
    return p_col * ops.pressure_mode_keep.astype(p_col.dtype)


def _pressure_from_hat(p_hat: jax.Array, template: jax.Array, params: Params) -> jax.Array:
    p_inner = jnp.fft.irfft2(p_hat, s=(params.nx, params.ny), axes=(0, 1)).real
    del template
    return p_inner.astype(params.dtype)


def solve_pressure(rhs_p: jax.Array, params: Params, ops: Operators) -> jax.Array:
    rhs_hat = jnp.fft.rfft2(rhs_p, axes=(0, 1))
    return _pressure_from_hat(_solve_pressure_hat(rhs_hat, params, ops), rhs_p, params)


def pressure_gradients(p: jax.Array, params: Params, ops: Operators) -> tuple[jax.Array, jax.Array, jax.Array]:
    p_hat = jnp.fft.rfft2(p, axes=(0, 1))
    dpdx = jnp.fft.irfft2(1j * ops.kx_rfft * p_hat, s=(params.nx, params.ny), axes=(0, 1)).real
    dpdy = jnp.fft.irfft2(1j * ops.ky_rfft * p_hat, s=(params.nx, params.ny), axes=(0, 1)).real
    dpdz = gradient_to_upper_faces(p, params.dz)
    return (
        dpdx.astype(params.dtype),
        dpdy.astype(params.dtype),
        dpdz,
    )


def _divergence_hat(u: jax.Array, v: jax.Array, w: jax.Array, params: Params, ops: Operators) -> jax.Array:
    dwdz_inner = divergence_upper_faces(w, params.dz)
    q_inner = jnp.stack((u, v, dwdz_inner), axis=0)
    q_hat = jnp.fft.rfft2(q_inner, axes=(1, 2))
    return 1j * ops.kx_rfft * q_hat[0] + 1j * ops.ky_rfft * q_hat[1] + q_hat[2]


def _pressure_and_horizontal_gradients_from_hat(
    p_hat: jax.Array,
    template: jax.Array,
    params: Params,
    ops: Operators,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    fields_hat = jnp.stack(
        (
            p_hat,
            1j * ops.kx_rfft * p_hat,
            1j * ops.ky_rfft * p_hat,
        ),
        axis=0,
    )
    fields_inner = jnp.fft.irfft2(fields_hat, s=(params.nx, params.ny), axes=(1, 2)).real
    del template
    p = fields_inner[0].astype(params.dtype)
    dpdx = fields_inner[1].astype(params.dtype)
    dpdy = fields_inner[2].astype(params.dtype)
    return p, dpdx, dpdy


def project_velocity(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    params: Params,
    ops: Operators,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    w, top_w_hat = _prepare_radiation_top(w, params, ops)
    div_hat = _divergence_hat(u, v, w, params, ops)
    p_hat = _solve_pressure_hat(div_hat / params.dt, params, ops, top_w_hat=top_w_hat)
    p, dpdx, dpdy = _pressure_and_horizontal_gradients_from_hat(p_hat, u, params, ops)
    dpdz = gradient_to_upper_faces(p, params.dz)
    if params.top_boundary_condition == "klemp_durran":
        coefficient = _radiation_pressure_coefficient(params, ops)
        half_dz = 0.5 * params.dz
        alpha = coefficient * params.dt / half_dz
        top_gradient_hat = (
            coefficient * top_w_hat - p_hat[..., -1]
        ) / (half_dz * (1.0 + alpha))
        top_gradient_hat = jnp.where(coefficient > 0.0, top_gradient_hat, 0.0)
        top_gradient = jnp.fft.irfft2(
            top_gradient_hat, s=(params.nx, params.ny), axes=(0, 1)
        ).real.astype(params.dtype)
        dpdz = dpdz.at[:, :, -1].set(top_gradient)
    w_new = w - params.dt * dpdz
    if params.top_boundary_condition == "rigid_lid":
        w_new = w_new.at[:, :, -1].set(0.0)
    return u - params.dt * dpdx, v - params.dt * dpdy, w_new, p


def project(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    params: Params,
    ops: Operators,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    w, top_w_hat = _prepare_radiation_top(w, params, ops)
    div_hat = _divergence_hat(u, v, w, params, ops)
    p_hat = _solve_pressure_hat(div_hat / params.dt, params, ops, top_w_hat=top_w_hat)
    p, dpdx, dpdy = _pressure_and_horizontal_gradients_from_hat(p_hat, u, params, ops)
    dpdz = gradient_to_upper_faces(p, params.dz)
    if params.top_boundary_condition == "klemp_durran":
        coefficient = _radiation_pressure_coefficient(params, ops)
        half_dz = 0.5 * params.dz
        alpha = coefficient * params.dt / half_dz
        top_gradient_hat = (
            coefficient * top_w_hat - p_hat[..., -1]
        ) / (half_dz * (1.0 + alpha))
        top_gradient_hat = jnp.where(coefficient > 0.0, top_gradient_hat, 0.0)
        top_gradient = jnp.fft.irfft2(
            top_gradient_hat, s=(params.nx, params.ny), axes=(0, 1)
        ).real.astype(params.dtype)
        dpdz = dpdz.at[:, :, -1].set(top_gradient)
    w_new = w - params.dt * dpdz
    if params.top_boundary_condition == "rigid_lid":
        w_new = w_new.at[:, :, -1].set(0.0)
    div = jnp.fft.irfft2(div_hat, s=(params.nx, params.ny), axes=(0, 1)).real.astype(params.dtype)
    return u - params.dt * dpdx, v - params.dt * dpdy, w_new, p, div
