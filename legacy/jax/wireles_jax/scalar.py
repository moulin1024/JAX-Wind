from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Params
from .derivative import ddx, ddy, ddxy_filter_many, ddz_uv, ddz_w
from .grid import (
    center_gradient,
    center_to_upper_faces,
    divergence_upper_faces,
    gradient_to_upper_faces,
    upper_face_to_center,
)
from .init import apply_scalar_bc, apply_theta_bc
from .lasd_polynomial import (
    largest_positive_real_polynomial_root as _largest_positive_real_polynomial_root,
    porte_agel_polynomial as _porte_agel_polynomial,
)
from .state import FlowState, Operators
from .sgs import (
    _lagrangian_interp,
    _scale_dependence_beta,
    _strain_magnitude,
    _strain_uv,
)
from .wind_tunnel import wind_tunnel_scalar_sources

THETA_INDEX = 0
QV_INDEX = 1
N_SCALARS = 2


def _physical_mask(q: jax.Array) -> jax.Array:
    return jnp.ones_like(q)


def _scalar_mask(q: jax.Array) -> jax.Array:
    return jnp.ones_like(q)


def _avg_next(q: jax.Array) -> jax.Array:
    return upper_face_to_center(q)


def _avg_prev(q: jax.Array) -> jax.Array:
    return center_to_upper_faces(q)


def _scalar_center_dz(q: jax.Array, params: Params) -> jax.Array:
    """Cell-centred z derivative with physical one-sided wall stencils.

    This matches the C++ ``ddz_center`` semantics.  In particular, the first
    centre uses the first two physical cells and is independent of the wall
    ghost used to impose a face flux.
    """
    return center_gradient(q, params.dz).astype(q.dtype)


def _weno3_face_value(w: jax.Array, phi: jax.Array) -> jax.Array:
    """Third-order WENO reconstruction of cell-centered scalars at w faces."""
    q_left = jnp.roll(phi, 1, axis=2)
    q_left2 = jnp.roll(phi, 2, axis=2)
    q_right = phi
    q_right2 = jnp.roll(phi, -1, axis=2)

    centered = 0.5 * (q_left + q_right)
    left_extrapolated = 1.5 * q_left - 0.5 * q_left2
    right_extrapolated = 1.5 * q_right - 0.5 * q_right2
    beta_left = (q_left - q_left2) ** 2
    beta_center = (q_right - q_left) ** 2
    beta_right = (q_right2 - q_right) ** 2
    epsilon = jnp.asarray(1.0e-12, dtype=phi.dtype)

    alpha_positive_0 = (1.0 / 3.0) / (epsilon + beta_left) ** 2
    alpha_positive_1 = (2.0 / 3.0) / (epsilon + beta_center) ** 2
    weight_positive_0 = alpha_positive_0 / (alpha_positive_0 + alpha_positive_1)
    positive = weight_positive_0 * left_extrapolated + (1.0 - weight_positive_0) * centered

    alpha_negative_0 = (1.0 / 3.0) / (epsilon + beta_right) ** 2
    alpha_negative_1 = (2.0 / 3.0) / (epsilon + beta_center) ** 2
    weight_negative_0 = alpha_negative_0 / (alpha_negative_0 + alpha_negative_1)
    negative = weight_negative_0 * right_extrapolated + (1.0 - weight_negative_0) * centered
    return jnp.where(w[..., None] >= 0.0, positive, negative)


def _shift_z_clamped(q: jax.Array, offset: int) -> jax.Array:
    """Shift in z with linear virtual cells instead of periodic wrapping.

    WENO5 needs two values outside the first transported cell at its first
    interior face, while the solver stores only one lower ghost plane.  A
    repeated/clamped ghost destroys linear exactness at that face even when
    the stored ghost has the correct Neumann slope.  Extend the slope between
    the stored ghost and its adjacent cell to supply the missing virtual
    values.  At the rigid top, where the boundary planes are equal, this
    reduces exactly to the previous constant extension.
    """
    if offset > 0:
        shape = (1, 1, offset) + (1,) * (q.ndim - 3)
        distance = jnp.arange(offset, 0, -1, dtype=q.dtype).reshape(shape)
        lower_slope = q[:, :, 1:2, ...] - q[:, :, :1, ...]
        lower_virtual = q[:, :, :1, ...] - distance * lower_slope
        return jnp.concatenate(
            (lower_virtual, q[:, :, :-offset, ...]),
            axis=2,
        )
    if offset < 0:
        count = -offset
        shape = (1, 1, count) + (1,) * (q.ndim - 3)
        distance = jnp.arange(1, count + 1, dtype=q.dtype).reshape(shape)
        upper_slope = q[:, :, -1:, ...] - q[:, :, -2:-1, ...]
        upper_virtual = q[:, :, -1:, ...] + distance * upper_slope
        return jnp.concatenate(
            (q[:, :, count:, ...], upper_virtual),
            axis=2,
        )
    return q


def _pad_scalar_for_advection(
    phi: jax.Array,
    params: Params,
    ghost_count: int,
    lower_ghost: jax.Array | None = None,
    upper_ghost: jax.Array | None = None,
) -> jax.Array:
    """Construct virtual WENO stencil cells without storing them in state."""
    if lower_ghost is None:
        lower_slope = (
            phi[:, :, 1, :] - phi[:, :, 0, :]
            if phi.shape[2] > 1
            else jnp.zeros_like(phi[:, :, 0, :])
        )
        lower_ghost = phi[:, :, 0, :] - lower_slope
    lower_slope = phi[:, :, 0, :] - lower_ghost
    lower_distance = jnp.arange(
        ghost_count, 0, -1, dtype=phi.dtype
    )[None, None, :, None]
    lower = phi[:, :, :1, :] - lower_distance * lower_slope[:, :, None, :]

    if upper_ghost is None:
        theta_increment = (
            0.0
            if params.theta_top_gradient is None
            else params.theta_top_gradient * params.z_i * params.dz
        )
        upper_slope = jnp.zeros_like(phi[:, :, -1, :])
        upper_slope = upper_slope.at[..., THETA_INDEX].set(theta_increment)
    else:
        upper_slope = upper_ghost - phi[:, :, -1, :]
    upper_distance = jnp.arange(
        1, ghost_count + 1, dtype=phi.dtype
    )[None, None, :, None]
    upper = phi[:, :, -1:, :] + upper_distance * upper_slope[:, :, None, :]
    return jnp.concatenate((lower, phi, upper), axis=2)


def _weno_z_weights(
    beta0: jax.Array,
    beta1: jax.Array,
    beta2: jax.Array,
    epsilon: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Borges et al. WENO-Z weights with the standard fifth-order p=2 form."""
    if epsilon is None:
        epsilon = jnp.asarray(jnp.finfo(beta0.dtype).eps, dtype=beta0.dtype)
    tau5 = jnp.abs(beta0 - beta2)

    # A prescribed wall flux can require a large ghost-cell jump where the
    # local dynamic diffusivity is small.  In float32, the standard WENO-Z
    # expression (tau / (beta + eps))**2 can then overflow even though the
    # normalized nonlinear weights are perfectly finite.  Divide alpha by
    # ``1 + (tau / min(beta + eps))**2`` and evaluate that common factor with
    # reciprocal branches.  This is algebraically identical after weight
    # normalization and keeps every intermediate ratio in [0, 1].
    tiny = jnp.asarray(jnp.finfo(beta0.dtype).tiny, dtype=beta0.dtype)
    denom0 = jnp.maximum(beta0 + epsilon, tiny)
    denom1 = jnp.maximum(beta1 + epsilon, tiny)
    denom2 = jnp.maximum(beta2 + epsilon, tiny)
    minimum_denom = jnp.minimum(denom0, jnp.minimum(denom1, denom2))
    tau_is_smaller = tau5 <= minimum_denom
    bounded_ratio = jnp.where(tau_is_smaller, tau5, minimum_denom) / jnp.where(
        tau_is_smaller,
        minimum_denom,
        jnp.maximum(tau5, tiny),
    )
    ratio2 = bounded_ratio * bounded_ratio
    inverse_sum = 1.0 / (1.0 + ratio2)
    common_part = jnp.where(tau_is_smaller, inverse_sum, ratio2 * inverse_sum)
    nonlinear_part = jnp.where(tau_is_smaller, ratio2 * inverse_sum, inverse_sum)

    alpha0 = 0.1 * (common_part + nonlinear_part * (minimum_denom / denom0) ** 2)
    alpha1 = 0.6 * (common_part + nonlinear_part * (minimum_denom / denom1) ** 2)
    alpha2 = 0.3 * (common_part + nonlinear_part * (minimum_denom / denom2) ** 2)
    alpha_sum = alpha0 + alpha1 + alpha2
    return alpha0 / alpha_sum, alpha1 / alpha_sum, alpha2 / alpha_sum


def _weno5z_face_value(w: jax.Array, phi: jax.Array) -> jax.Array:
    """Fifth-order low-dissipation WENO-Z scalar reconstruction at w faces."""
    qim2 = _shift_z_clamped(phi, 3)
    qim1 = _shift_z_clamped(phi, 2)
    qi = _shift_z_clamped(phi, 1)
    qip1 = phi
    qip2 = _shift_z_clamped(phi, -1)
    qip3 = _shift_z_clamped(phi, -2)

    # Smoothness indicators are invariant to a common translation and
    # homogeneous under a common local scale.  Form them from translated,
    # scaled stencil values so an exactly imposed flux ghost cannot overflow
    # the float32 difference squares.  The reconstruction candidates below
    # still use the original scalar values, so this changes neither the
    # conservative face value nor the prescribed wall flux.
    deviations = (
        qim2 - qi,
        qim1 - qi,
        jnp.zeros_like(qi),
        qip1 - qi,
        qip2 - qi,
        qip3 - qi,
    )
    smoothness_scale = jnp.ones_like(qi)
    for deviation in deviations:
        smoothness_scale = jnp.maximum(smoothness_scale, jnp.abs(deviation))
    inverse_smoothness_scale = 1.0 / smoothness_scale
    smoothness_epsilon = jnp.maximum(
        jnp.asarray(jnp.finfo(phi.dtype).tiny, dtype=phi.dtype),
        jnp.asarray(jnp.finfo(phi.dtype).eps, dtype=phi.dtype)
        * inverse_smoothness_scale
        * inverse_smoothness_scale,
    )
    sim2, sim1, si, sip1, sip2, sip3 = tuple(
        deviation * inverse_smoothness_scale for deviation in deviations
    )

    positive0 = (1.0 / 3.0) * qim2 - (7.0 / 6.0) * qim1 + (11.0 / 6.0) * qi
    positive1 = -(1.0 / 6.0) * qim1 + (5.0 / 6.0) * qi + (1.0 / 3.0) * qip1
    positive2 = (1.0 / 3.0) * qi + (5.0 / 6.0) * qip1 - (1.0 / 6.0) * qip2
    beta_positive0 = (13.0 / 12.0) * (sim2 - 2.0 * sim1 + si) ** 2 + 0.25 * (
        sim2 - 4.0 * sim1 + 3.0 * si
    ) ** 2
    beta_positive1 = (13.0 / 12.0) * (sim1 - 2.0 * si + sip1) ** 2 + 0.25 * (
        sim1 - sip1
    ) ** 2
    beta_positive2 = (13.0 / 12.0) * (si - 2.0 * sip1 + sip2) ** 2 + 0.25 * (
        3.0 * si - 4.0 * sip1 + sip2
    ) ** 2
    wp0, wp1, wp2 = _weno_z_weights(
        beta_positive0,
        beta_positive1,
        beta_positive2,
        smoothness_epsilon,
    )
    positive = wp0 * positive0 + wp1 * positive1 + wp2 * positive2

    negative0 = (1.0 / 3.0) * qip3 - (7.0 / 6.0) * qip2 + (11.0 / 6.0) * qip1
    negative1 = -(1.0 / 6.0) * qip2 + (5.0 / 6.0) * qip1 + (1.0 / 3.0) * qi
    negative2 = (1.0 / 3.0) * qip1 + (5.0 / 6.0) * qi - (1.0 / 6.0) * qim1
    beta_negative0 = (13.0 / 12.0) * (sip3 - 2.0 * sip2 + sip1) ** 2 + 0.25 * (
        sip3 - 4.0 * sip2 + 3.0 * sip1
    ) ** 2
    beta_negative1 = (13.0 / 12.0) * (sip2 - 2.0 * sip1 + si) ** 2 + 0.25 * (
        sip2 - si
    ) ** 2
    beta_negative2 = (13.0 / 12.0) * (sip1 - 2.0 * si + sim1) ** 2 + 0.25 * (
        3.0 * sip1 - 4.0 * si + sim1
    ) ** 2
    wn0, wn1, wn2 = _weno_z_weights(
        beta_negative0,
        beta_negative1,
        beta_negative2,
        smoothness_epsilon,
    )
    negative = wn0 * negative0 + wn1 * negative1 + wn2 * negative2
    return jnp.where(w[..., None] >= 0.0, positive, negative)


def _scalar_advection_divergence(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    phi: jax.Array,
    params: Params,
    ops: Operators,
    lower_ghost: jax.Array | None = None,
    upper_ghost: jax.Array | None = None,
) -> jax.Array:
    """Conservative divergence of scalar fluxes on the staggered grid."""
    horizontal_fluxes = jnp.concatenate((u[..., None] * phi, v[..., None] * phi), axis=-1)
    flux_hat = jnp.fft.rfft2(horizontal_fluxes, axes=(0, 1))
    if params.horizontal_dealias:
        keep = ops.horizontal_cutoff_rfft[..., None].astype(flux_hat.dtype)
        flux_hat = flux_hat * keep
    kx = ops.kx_rfft[..., None].astype(flux_hat.real.dtype)
    ky = ops.ky_rfft[..., None].astype(flux_hat.real.dtype)
    horizontal_divergence = jnp.fft.irfft2(
        1j * kx * flux_hat[..., :N_SCALARS]
        + 1j * ky * flux_hat[..., N_SCALARS:],
        s=(params.nx, params.ny),
        axes=(0, 1),
    ).real
    horizontal_divergence = horizontal_divergence.astype(phi.dtype)

    padded_phi = _pad_scalar_for_advection(
        phi,
        params,
        ghost_count=3,
        lower_ghost=lower_ghost,
        upper_ghost=upper_ghost,
    )
    w_eval = jnp.zeros(
        w.shape[:2] + (padded_phi.shape[2],), dtype=w.dtype
    ).at[:, :, 4 : 4 + params.nz].set(w)
    if params.scalar_vertical_scheme == "weno5z":
        phi_on_w = _weno5z_face_value(w_eval, padded_phi)[
            :, :, 4 : 4 + params.nz, :
        ]
    elif params.scalar_vertical_scheme == "weno3":
        phi_on_w = _weno3_face_value(w_eval, padded_phi)[
            :, :, 4 : 4 + params.nz, :
        ]
    else:
        phi_on_w = center_to_upper_faces(phi)
    vertical_flux = w[..., None] * phi_on_w
    vertical_divergence = ddz_w(vertical_flux, params)
    return horizontal_divergence + vertical_divergence


def _safe_divide(num: jax.Array, den: jax.Array) -> jax.Array:
    valid = jnp.abs(den) > 1.0e-30
    safe_den = jnp.where(valid, den, 1.0)
    return jnp.where(valid, num / safe_den, 0.0)


def _filter_width(params: Params, test_ratio: float) -> float:
    return params.fgr * test_ratio


def _spectral_box_filter_concat(q: jax.Array, params: Params, filter_width: float) -> jax.Array:
    q_hat = jnp.fft.rfft2(q, axes=(0, 1))
    x_mode = jnp.abs(jnp.fft.fftfreq(params.nx, d=1.0) * params.nx)
    y_mode = jnp.fft.rfftfreq(params.ny, d=1.0) * params.ny
    # The original volume LASD filter uses Fortran NINT; only the separate
    # wall-plane filter uses FLOOR.
    cutoff_x = jnp.floor(params.nx / (2.0 * filter_width) + 0.5)
    cutoff_y = jnp.floor(params.ny / (2.0 * filter_width) + 0.5)
    keep = (x_mode[:, None] < cutoff_x) & (y_mode[None, :] < cutoff_y)
    q_filtered = jnp.fft.irfft2(
        q_hat * keep[:, :, None, None].astype(q_hat.dtype),
        s=(params.nx, params.ny),
        axes=(0, 1),
    ).real
    return q_filtered.astype(q.dtype)


def _scalar_stack(theta: jax.Array, qv: jax.Array) -> jax.Array:
    return jnp.stack((theta, qv), axis=-1)


def _unstack_scalars(phi: jax.Array) -> tuple[jax.Array, jax.Array]:
    return phi[..., THETA_INDEX], phi[..., QV_INDEX]


def _fixed_scalar_coefficients(cs2: jax.Array, params: Params) -> jax.Array:
    theta_coeff = cs2.astype(params.sgs_dtype) / params.prandtl_t
    qv_coeff = cs2.astype(params.sgs_dtype) / params.schmidt_t
    coeff = jnp.stack((theta_coeff, qv_coeff), axis=-1)
    return coeff * _scalar_mask(coeff)


def _lagrangian_average(
    current_a: jax.Array,
    current_b: jax.Array,
    old_a: jax.Array,
    old_b: jax.Array,
    u_lag: jax.Array,
    v_lag: jax.Array,
    w_lag: jax.Array,
    params: Params,
    *,
    timescale_a: jax.Array | None = None,
    timescale_b: jax.Array | None = None,
    ramp_numerator: bool = False,
) -> tuple[jax.Array, jax.Array]:
    dt_lag = params.dt * params.cs_count
    time_a = old_a if timescale_a is None else timescale_a
    time_b = old_b if timescale_b is None else timescale_b
    while time_a.ndim < old_a.ndim:
        time_a = time_a[..., None]
        time_b = time_b[..., None]
    product = time_a * time_b
    valid = (time_a > 0.0) & (time_b >= 0.0) & (product > 0.0)
    tn = 1.5 * params.sgs_delta
    tn = tn * jnp.where(valid, product ** (-0.125), 1.0)
    eps = jnp.where(valid, (dt_lag / tn) / (1.0 + dt_lag / tn), 0.0)
    a_interp = _lagrangian_interp(old_a, u_lag, v_lag, w_lag, params)
    b_interp = _lagrangian_interp(old_b, u_lag, v_lag, w_lag, params)
    avg_a = eps * current_a + (1.0 - eps) * a_interp
    avg_b = jnp.maximum(eps * current_b + (1.0 - eps) * b_interp, 0.0)
    if ramp_numerator:
        avg_a = jnp.where(avg_a > 0.0, avg_a, jnp.asarray(1.0e-32, dtype=avg_a.dtype))
    return avg_a, avg_b


def _history_bc(q: jax.Array) -> jax.Array:
    if q.shape[2] < 2:
        return q
    return q.at[:, :, 0, ...].set(q[:, :, 1, ...]).at[:, :, -1, ...].set(
        q[:, :, -2, ...]
    )


def _scalar_lm_mm(
    vel: jax.Array,
    phi: jax.Array,
    grad_phi: jax.Array,
    sij: jax.Array,
    strain_mag: jax.Array,
    params: Params,
    test_ratio: float,
) -> tuple[jax.Array, jax.Array]:
    delta = params.sgs_delta
    vel_phi = vel[..., :, None] * phi[..., None, :]
    sgrad = strain_mag[..., None, None] * grad_phi
    q = jnp.concatenate(
        (
            vel,
            phi,
            vel_phi.reshape(phi.shape[:-1] + (3 * N_SCALARS,)),
            sgrad.reshape(phi.shape[:-1] + (3 * N_SCALARS,)),
            grad_phi.reshape(phi.shape[:-1] + (3 * N_SCALARS,)),
            sij,
        ),
        axis=-1,
    )
    filtered = _spectral_box_filter_concat(q.astype(params.sgs_dtype), params, _filter_width(params, test_ratio))
    cursor = 0
    vel_hat = filtered[..., cursor : cursor + 3]
    cursor += 3
    phi_hat = filtered[..., cursor : cursor + N_SCALARS]
    cursor += N_SCALARS
    vel_phi_hat = filtered[..., cursor : cursor + 3 * N_SCALARS].reshape(phi.shape[:-1] + (3, N_SCALARS))
    cursor += 3 * N_SCALARS
    sgrad_hat = filtered[..., cursor : cursor + 3 * N_SCALARS].reshape(phi.shape[:-1] + (3, N_SCALARS))
    cursor += 3 * N_SCALARS
    grad_hat = filtered[..., cursor : cursor + 3 * N_SCALARS].reshape(phi.shape[:-1] + (3, N_SCALARS))
    cursor += 3 * N_SCALARS
    sij_hat = filtered[..., cursor : cursor + 6]
    strain_hat = _strain_magnitude(sij_hat)

    l_i = vel_phi_hat - vel_hat[..., :, None] * phi_hat[..., None, :]
    m_i = delta * delta * (sgrad_hat - test_ratio * test_ratio * strain_hat[..., None, None] * grad_hat)
    lm = jnp.sum(l_i * m_i, axis=-2)
    mm = jnp.sum(m_i * m_i, axis=-2)
    return lm * _scalar_mask(lm), mm * _scalar_mask(mm)


def _scalar_sd_terms(
    vel: jax.Array,
    phi: jax.Array,
    grad_phi: jax.Array,
    sij: jax.Array,
    strain_mag: jax.Array,
    params: Params,
    test_ratio: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return Porte-Agel's resolved flux K and the two vectors in X(beta)."""
    vel_phi = vel[..., :, None] * phi[..., None, :]
    sgrad = strain_mag[..., None, None] * grad_phi
    q = jnp.concatenate(
        (
            vel,
            phi,
            vel_phi.reshape(phi.shape[:-1] + (3 * N_SCALARS,)),
            sgrad.reshape(phi.shape[:-1] + (3 * N_SCALARS,)),
            grad_phi.reshape(phi.shape[:-1] + (3 * N_SCALARS,)),
            sij,
        ),
        axis=-1,
    )
    filtered = _spectral_box_filter_concat(
        q.astype(params.sgs_dtype), params, _filter_width(params, test_ratio)
    )
    cursor = 0
    vel_hat = filtered[..., cursor : cursor + 3]
    cursor += 3
    phi_hat = filtered[..., cursor : cursor + N_SCALARS]
    cursor += N_SCALARS
    vel_phi_hat = filtered[..., cursor : cursor + 3 * N_SCALARS].reshape(
        phi.shape[:-1] + (3, N_SCALARS)
    )
    cursor += 3 * N_SCALARS
    b_vector = filtered[..., cursor : cursor + 3 * N_SCALARS].reshape(
        phi.shape[:-1] + (3, N_SCALARS)
    )
    cursor += 3 * N_SCALARS
    grad_hat = filtered[..., cursor : cursor + 3 * N_SCALARS].reshape(
        phi.shape[:-1] + (3, N_SCALARS)
    )
    cursor += 3 * N_SCALARS
    sij_hat = filtered[..., cursor : cursor + 6]
    a_vector = _strain_magnitude(sij_hat)[..., None, None] * grad_hat
    resolved_flux = vel_phi_hat - vel_hat[..., :, None] * phi_hat[..., None, :]
    return resolved_flux, b_vector, a_vector


def _porte_agel_plane_coefficients(
    vel: jax.Array,
    phi: jax.Array,
    grad_phi: jax.Array,
    sij: jax.Array,
    strain_mag: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Original plane-averaged scale-dependent scalar dynamic model."""
    k1, b1, a1 = _scalar_sd_terms(
        vel, phi, grad_phi, sij, strain_mag, params, params.tfr
    )
    k2, b2, a2 = _scalar_sd_terms(
        vel, phi, grad_phi, sij, strain_mag, params, params.tfr * params.tfr
    )

    def plane_dot(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.mean(jnp.sum(x * y, axis=-2), axis=(0, 1))

    p = plane_dot(k1, b1)
    q = plane_dot(k1, a1)
    r = plane_dot(b1, b1)
    s = plane_dot(a1, a1)
    t = plane_dot(b1, a1)
    p2 = plane_dot(k2, b2)
    q2 = plane_dot(k2, a2)
    r2 = plane_dot(b2, b2)
    s2 = plane_dot(a2, a2)
    t2 = plane_dot(b2, a2)

    # Expansion of N1(beta) D2(beta) - N2(beta) D1(beta) = 0, where
    # X_2D = Delta^2 (B1 - 4 beta A1) and
    # X_4D = Delta^2 (B2 - 16 beta^2 A2).
    polynomial = _porte_agel_polynomial(p, q, r, s, t, p2, q2, r2, s2, t2)
    beta = _largest_positive_real_polynomial_root(polynomial)

    numerator = p - 4.0 * beta * q
    denominator = r - 8.0 * beta * t + 16.0 * beta * beta * s
    coeff = _safe_divide(numerator, params.sgs_delta**2 * denominator)

    # If the fifth-order equation is degenerate or its largest positive root
    # does not yield a dissipative coefficient, use the standard dynamic
    # beta=1 result on that horizontal plane.
    fallback_denominator = r - 8.0 * t + 16.0 * s
    fallback = _safe_divide(p - 4.0 * q, params.sgs_delta**2 * fallback_denominator)
    valid = jnp.isfinite(beta) & jnp.isfinite(coeff) & (coeff > 0.0)
    coeff = jnp.where(valid, coeff, fallback)
    coeff = jnp.clip(coeff, params.scalar_lasd_min, params.scalar_lasd_max)
    beta = jnp.where(valid, beta, 1.0)
    return tuple(
        jnp.broadcast_to(value[None, None, ...], phi.shape)
        for value in (coeff, beta, valid.astype(coeff.dtype))
    )


def _update_scalar_lasd_coefficients(
    state: FlowState,
    vel: jax.Array,
    phi: jax.Array,
    grad_phi: jax.Array,
    sij: jax.Array,
    strain_mag: jax.Array,
    fixed_coeff: jax.Array,
    momentum_lasd_state: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None,
    params: Params,
    update: bool,
    force_update: bool | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if params.scalar_sgs_model == "fixed_prandtl":
        return (
            fixed_coeff.astype(params.sgs_dtype),
            state.scalar_lm_old,
            state.scalar_mm_old,
            state.scalar_qn_old,
            state.scalar_nn_old,
        )

    if params.scalar_sgs_model == "porte_agel_sd":
        current = (
            state.scalar_c.astype(params.sgs_dtype),
            state.scalar_lm_old,
            state.scalar_mm_old,
            state.scalar_qn_old,
            state.scalar_nn_old,
        )
        if not update:
            return current
        should_update = ((state.step + 1) % params.cs_count) == 0

        def update_plane_model(_: None) -> tuple[jax.Array, ...]:
            coeff, beta, root_valid = _porte_agel_plane_coefficients(
                vel, phi, grad_phi, sij, strain_mag, params
            )
            zeros = jnp.zeros_like(beta)
            # These history slots are unused by the plane model.  Store beta
            # and root validity there so benchmark diagnostics can report the
            # polynomial solution and scale-invariant fallback rate.
            return coeff, beta, root_valid, zeros, zeros

        if force_update is True:
            return update_plane_model(None)
        if force_update is False:
            return current
        return jax.lax.cond(should_update, update_plane_model, lambda _: current, None)

    coeff_state = state.scalar_c.astype(params.sgs_dtype)
    lm_state = state.scalar_lm_old.astype(params.sgs_dtype)
    mm_state = state.scalar_mm_old.astype(params.sgs_dtype)
    qn_state = state.scalar_qn_old.astype(params.sgs_dtype)
    nn_state = state.scalar_nn_old.astype(params.sgs_dtype)
    if not update:
        return coeff_state, lm_state, mm_state, qn_state, nn_state

    # Use the same trajectory velocity accumulated by momentum LASD on this
    # update step. ``state.u_lag`` contains the previous samples; the current
    # filtered velocity completes the cs_count-sample average.
    u_lag = state.u_lag + vel[..., 0] / params.cs_count
    v_lag = state.v_lag + vel[..., 1] / params.cs_count
    w_lag = state.w_lag + vel[..., 2] / params.cs_count

    should_update = ((state.step + 1) % params.cs_count) == 0

    def do_update(_: None) -> tuple[jax.Array, ...]:
        lm, mm = _scalar_lm_mm(vel, phi, grad_phi, sij, strain_mag, params, params.tfr)
        qn, nn = _scalar_lm_mm(
            vel,
            phi,
            grad_phi,
            sij,
            strain_mag,
            params,
            params.tfr * params.tfr,
        )

        first_update = state.step == params.cs_count - 1
        lm_old = jnp.where(first_update, 0.03 * mm, lm_state)
        mm_old = jnp.where(first_update, mm, mm_state)
        qn_old = jnp.where(first_update, 0.03 * nn, qn_state)
        nn_old = jnp.where(first_update, nn, nn_state)
        lm_old = _history_bc(lm_old)
        mm_old = _history_bc(mm_old)
        qn_old = _history_bc(qn_old)
        nn_old = _history_bc(nn_old)

        momentum_lm, momentum_mm, momentum_qn, momentum_nn = (
            (state.lm_old, state.mm_old, state.qn_old, state.nn_old)
            if momentum_lasd_state is None
            else momentum_lasd_state
        )
        lm_avg, mm_avg = _lagrangian_average(
            lm,
            mm,
            lm_old,
            mm_old,
            u_lag,
            v_lag,
            w_lag,
            params,
            timescale_a=momentum_lm,
            timescale_b=momentum_mm,
            ramp_numerator=True,
        )
        qn_avg, nn_avg = _lagrangian_average(
            qn,
            nn,
            qn_old,
            nn_old,
            u_lag,
            v_lag,
            w_lag,
            params,
            timescale_a=momentum_qn,
            timescale_b=momentum_nn,
            ramp_numerator=True,
        )

        c_2d = jnp.maximum(_safe_divide(lm_avg, mm_avg), 0.0)
        c_4d = jnp.maximum(_safe_divide(qn_avg, nn_avg), 0.0)
        scale_dependent = params.scalar_lasd_scale_dependent
        if scale_dependent is None:
            scale_dependent = params.lasd_scale_dependent
        beta = _scale_dependence_beta(c_2d, c_4d, params, scale_dependent)
        coeff = jnp.clip(_safe_divide(c_2d, beta), params.scalar_lasd_min, params.scalar_lasd_max)
        coeff = coeff * _scalar_mask(coeff)
        return coeff, lm_avg, mm_avg, qn_avg, nn_avg

    def skip_update(_: None) -> tuple[jax.Array, ...]:
        return coeff_state, lm_state, mm_state, qn_state, nn_state

    if force_update is True:
        return do_update(None)
    if force_update is False:
        return skip_update(None)
    return jax.lax.cond(should_update, do_update, skip_update, operand=None)


def _scalar_flux_divergence(
    phi: jax.Array,
    flux_grad_phi: jax.Array,
    center_grad_phi: jax.Array,
    coeff: jax.Array,
    strain_mag: jax.Array,
    params: Params,
    ops: Operators,
) -> jax.Array:
    kappa = _scalar_diffusivity_from_coeff(phi, center_grad_phi, coeff, strain_mag, params)
    return _scalar_flux_divergence_from_kappa(phi, flux_grad_phi, kappa, params, ops)


def _scalar_diffusivity_from_coeff(
    phi: jax.Array,
    grad_phi: jax.Array,
    coeff: jax.Array,
    strain_mag: jax.Array,
    params: Params,
) -> jax.Array:
    """Return the cell-centred effective scalar diffusivity in solver units."""
    delta = params.sgs_delta
    stability = _scalar_stability_factor(phi, grad_phi, strain_mag, params)
    return (
        coeff.astype(params.dtype) * delta * delta * strain_mag[..., None].astype(params.dtype)
        + params.molecular_diffusivity_internal
    ) * stability[..., None] * _scalar_mask(phi)


def _apply_theta_surface_flux_ghost(
    theta: jax.Array,
    kappa_theta: jax.Array,
    params: Params,
) -> jax.Array:
    """Return the virtual lower theta cell required by WENO reconstruction."""
    if params.theta_bc == "dirichlet":
        return 2.0 * jnp.asarray(params.theta_bottom, dtype=theta.dtype) - theta[:, :, 0]
    if params.theta_bc != "flux" or params.surface_theta_flux == 0.0:
        return theta[:, :, 0]
    kappa_wall = kappa_theta[:, :, 0].astype(theta.dtype)
    positive = kappa_wall > jnp.asarray(jnp.finfo(theta.dtype).tiny, dtype=theta.dtype)
    safe_kappa = jnp.where(positive, kappa_wall, 1.0)
    ghost = theta[:, :, 0] + params.surface_theta_flux * params.dz / safe_kappa
    return jnp.where(positive, ghost, theta[:, :, 0])


def _scalar_flux_divergence_from_kappa(
    phi: jax.Array,
    grad_phi: jax.Array,
    kappa: jax.Array,
    params: Params,
    ops: Operators,
) -> jax.Array:
    qx = -kappa * grad_phi[..., 0, :]
    qy = -kappa * grad_phi[..., 1, :]
    qz = -center_to_upper_faces(kappa) * grad_phi[..., 2, :]
    qv_flux = (
        params.transported_surface_qv_flux
        if params.moisture_enabled
        else 0.0
    )
    if params.theta_bc == "dirichlet":
        theta_bottom_gradient = (
            2.0
            * (
                phi[:, :, 0, THETA_INDEX]
                - jnp.asarray(params.theta_bottom, dtype=phi.dtype)
            )
            / params.dz
        )
        theta_bottom_flux = -kappa[:, :, 0, THETA_INDEX] * theta_bottom_gradient
    else:
        theta_bottom_flux = params.surface_theta_flux

    div_theta = (
        ddx(qx[..., THETA_INDEX], params, ops)
        + ddy(qy[..., THETA_INDEX], params, ops)
        + divergence_upper_faces(
            qz[..., THETA_INDEX],
            params.dz,
            theta_bottom_flux,
        )
    )
    div_qv = (
        ddx(qx[..., QV_INDEX], params, ops)
        + ddy(qy[..., QV_INDEX], params, ops)
        + divergence_upper_faces(qz[..., QV_INDEX], params.dz, qv_flux)
    )
    return jnp.stack((div_theta, div_qv), axis=-1)


def _scalar_stability_factor(
    phi: jax.Array,
    grad_phi: jax.Array,
    strain_mag: jax.Array,
    params: Params,
) -> jax.Array:
    if not params.scalar_stability_correction:
        return jnp.ones_like(strain_mag, dtype=params.dtype)
    dtheta_v_dz = grad_phi[..., 2, THETA_INDEX]
    if params.moisture_enabled:
        theta = phi[..., THETA_INDEX]
        qv = phi[..., QV_INDEX]
        dqv_dz = grad_phi[..., 2, QV_INDEX]
        dtheta_v_dz = (
            dtheta_v_dz * (1.0 + 0.61 * qv)
            + 0.61 * theta * dqv_dz
        )
    n2_scaled = (params.z_i * params.g / params.theta_v0) * dtheta_v_dz
    ri = jnp.maximum(n2_scaled, 0.0) / jnp.maximum(strain_mag * strain_mag, 1.0e-24)
    factor = (1.0 + params.scalar_stability_beta * ri) ** (-params.scalar_stability_power)
    return factor.astype(params.dtype) * _physical_mask(strain_mag)


def scalar_rhs(
    state: FlowState,
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    dudx: jax.Array,
    dudy: jax.Array,
    dudz: jax.Array,
    dvdx: jax.Array,
    dvdy: jax.Array,
    dvdz: jax.Array,
    dwdx: jax.Array,
    dwdy: jax.Array,
    dwdz: jax.Array,
    cs2: jax.Array,
    params: Params,
    ops: Operators,
    update_lasd: bool,
    force_lasd_update: bool | None = None,
    momentum_lasd_state: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, ...]:
    theta = apply_theta_bc(state.theta, params)
    qv = apply_moisture_bounds(state.qv, params)
    if not params.thermo_enabled:
        zeros = jnp.zeros_like(theta)
        return theta, qv, zeros, zeros, state.scalar_c, state.scalar_lm_old, state.scalar_mm_old, state.scalar_qn_old, state.scalar_nn_old

    (theta, qv), (dtheta_dx, dqv_dx), (dtheta_dy, dqv_dy) = ddxy_filter_many((theta, qv), params, ops)
    theta = apply_theta_bc(theta, params)
    qv = apply_moisture_bounds(qv, params)
    dtheta_dz_center = _scalar_center_dz(theta, params)
    dqv_dz_center = _scalar_center_dz(qv, params)
    phi = _scalar_stack(theta, qv)
    scalar_reference = jnp.asarray(
        (params.theta0, params.qv0), dtype=phi.dtype
    )
    # Transport and dynamic contractions use anomalies.  Subtracting a
    # constant is analytically neutral for an incompressible flow, but avoids
    # amplifying tiny float32 divergence/filter errors by an O(300 K)
    # potential-temperature offset and makes a uniform scalar a discrete
    # free stream.
    phi_anomaly = phi - scalar_reference
    center_grad_phi = jnp.stack(
        (
            _scalar_stack(dtheta_dx, dqv_dx),
            _scalar_stack(dtheta_dy, dqv_dy),
            _scalar_stack(dtheta_dz_center, dqv_dz_center),
        ),
        axis=-2,
    )
    sij = _strain_uv(
        dudx.astype(params.sgs_dtype),
        dudy.astype(params.sgs_dtype),
        dudz.astype(params.sgs_dtype),
        dvdx.astype(params.sgs_dtype),
        dvdy.astype(params.sgs_dtype),
        dvdz.astype(params.sgs_dtype),
        dwdx.astype(params.sgs_dtype),
        dwdy.astype(params.sgs_dtype),
        dwdz.astype(params.sgs_dtype),
    )
    strain_mag = _strain_magnitude(sij)
    fixed_coeff = _fixed_scalar_coefficients(cs2, params)
    vel = jnp.stack((u, v, _avg_next(w)), axis=-1).astype(params.sgs_dtype)
    # Match the C++ dynamic-model semantics: Germano contractions and
    # stability use physical cell-centred gradients, not the wall-face
    # gradient that contains the imposed Neumann flux.
    coeff, lm_old, mm_old, qn_old, nn_old = _update_scalar_lasd_coefficients(
        state,
        vel,
        phi_anomaly.astype(params.sgs_dtype),
        center_grad_phi.astype(params.sgs_dtype),
        sij,
        strain_mag.astype(params.sgs_dtype),
        fixed_coeff,
        momentum_lasd_state,
        params,
        update_lasd,
        force_lasd_update,
    )

    # Transport still needs the wall-face gradient.  Reconstruct its ghost
    # from the accepted diffusivity so WENO and the explicit SGS face flux
    # obey the same Neumann condition without contaminating dynamic LASD.
    kappa = _scalar_diffusivity_from_coeff(
        phi, center_grad_phi, coeff, strain_mag, params
    )
    if params.theta_bc == "dirichlet":
        theta_top_gradient = (
            2.0
            * (
                jnp.asarray(params.theta_top, dtype=theta.dtype)
                - theta[:, :, -1]
            )
            / params.dz
        )
    else:
        theta_top_gradient = (
            0.0
            if params.theta_top_gradient is None
            else params.theta_top_gradient * params.z_i
        )
    dtheta_dz_face = gradient_to_upper_faces(
        theta, params.dz, theta_top_gradient
    )
    dqv_dz_face = gradient_to_upper_faces(qv, params.dz, 0.0)
    flux_grad_phi = jnp.stack(
        (
            _scalar_stack(dtheta_dx, dqv_dx),
            _scalar_stack(dtheta_dy, dqv_dy),
            _scalar_stack(dtheta_dz_face, dqv_dz_face),
        ),
        axis=-2,
    )

    theta_ghost = _apply_theta_surface_flux_ghost(
        theta, kappa[..., THETA_INDEX], params
    )
    qv_wall_kappa = kappa[:, :, 0, QV_INDEX]
    qv_positive = qv_wall_kappa > jnp.asarray(
        jnp.finfo(qv.dtype).tiny, dtype=qv.dtype
    )
    qv_safe_kappa = jnp.where(qv_positive, qv_wall_kappa, 1.0)
    qv_ghost = qv[:, :, 0] + (
        params.transported_surface_qv_flux
        * params.dz
        / qv_safe_kappa
    )
    qv_ghost = jnp.where(qv_positive, qv_ghost, qv[:, :, 0])
    lower_ghost = _scalar_stack(theta_ghost, qv_ghost)
    theta_upper_ghost = (
        2.0 * jnp.asarray(params.theta_top, dtype=theta.dtype) - theta[:, :, -1]
        if params.theta_bc == "dirichlet"
        else theta[:, :, -1] + theta_top_gradient * params.dz
    )
    upper_ghost = _scalar_stack(theta_upper_ghost, qv[:, :, -1])
    convection = _scalar_advection_divergence(
        u,
        v,
        w,
        phi_anomaly,
        params,
        ops,
        lower_ghost=lower_ghost - scalar_reference,
        upper_ghost=upper_ghost - scalar_reference,
    )
    div_q = _scalar_flux_divergence(
        phi, flux_grad_phi, center_grad_phi, coeff, strain_mag, params, ops
    )
    rhs = -convection - div_q
    source_theta, source_qv = wind_tunnel_scalar_sources(theta, qv, params)
    rhs = rhs + _scalar_stack(source_theta, source_qv)
    if not params.moisture_enabled:
        rhs = rhs.at[..., QV_INDEX].set(0.0)
    rhs_theta, rhs_qv = _unstack_scalars(rhs)
    return theta, qv, rhs_theta, rhs_qv, coeff, lm_old, mm_old, qn_old, nn_old


def apply_moisture_bounds(qv: jax.Array, params: Params) -> jax.Array:
    floor = jnp.asarray(params.qv_floor, dtype=qv.dtype)
    shifted = qv - floor
    positive = jnp.maximum(shifted, 0.0)
    negative_mass = -jnp.sum(jnp.minimum(shifted, 0.0))
    positive_mass = jnp.sum(positive)
    scale = jnp.where(
        positive_mass > negative_mass,
        (positive_mass - negative_mass) / jnp.maximum(positive_mass, 1.0e-30),
        0.0,
    )
    corrected = floor + positive * scale
    return apply_scalar_bc(corrected)


def virtual_potential_temperature(theta: jax.Array, qv: jax.Array, params: Params) -> jax.Array:
    if not params.moisture_enabled:
        qv = jnp.zeros_like(qv)
    return theta * (1.0 + 0.61 * qv)


def buoyancy_from_theta_qv(theta: jax.Array, qv: jax.Array, params: Params) -> jax.Array:
    if not params.thermo_enabled:
        return jnp.zeros_like(theta)
    theta_v = virtual_potential_temperature(theta, qv, params)
    if params.buoyancy_reference == "plane_mean":
        # Average the anomaly, not an O(300 K) absolute field.  The direct
        # float32 plane mean loses enough low bits to create a spurious
        # domain-wide acceleration in otherwise uniform wind-tunnel flow.
        theta_v_base = jnp.asarray(params.theta_v0, dtype=theta_v.dtype)
        theta_v_anomaly = theta_v - theta_v_base
        theta_v_prime = theta_v_anomaly - jnp.mean(
            theta_v_anomaly, axis=(0, 1), keepdims=True
        )
    else:
        theta_v_prime = theta_v - jnp.asarray(params.theta_v0, dtype=theta_v.dtype)
    theta_v_prime_w = center_to_upper_faces(theta_v_prime)
    buoyancy = (params.z_i * params.g / params.theta_v0) * theta_v_prime_w
    buoyancy = buoyancy.at[:, :, -1].set(0.0)
    return buoyancy.astype(params.dtype)
