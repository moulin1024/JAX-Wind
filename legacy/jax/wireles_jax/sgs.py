from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Params
from .grid import center_to_upper_faces, upper_face_to_center
from .lasd_polynomial import (
    largest_positive_real_polynomial_root,
    porte_agel_polynomial,
)
from .state import FlowState


def zero_stress(template: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    z = jnp.zeros_like(template)
    return z, z, z, z, z, z


def _to_sgs(q: jax.Array, params: Params) -> jax.Array:
    return q.astype(params.sgs_dtype)


def _to_solver(q: jax.Array, params: Params) -> jax.Array:
    return q.astype(params.dtype)


def _cast_stress_to_solver(stress: tuple[jax.Array, ...], params: Params) -> tuple[jax.Array, ...]:
    return tuple(_to_solver(t, params) for t in stress)


def _cast_sgs_state(state: FlowState, params: Params) -> tuple[jax.Array, ...]:
    return (
        _to_sgs(state.cs2, params),
        _to_sgs(state.lm_old, params),
        _to_sgs(state.mm_old, params),
        _to_sgs(state.qn_old, params),
        _to_sgs(state.nn_old, params),
        _to_sgs(state.u_lag, params),
        _to_sgs(state.v_lag, params),
        _to_sgs(state.w_lag, params),
    )


def classic_smagorinsky(
    dudx: jax.Array,
    dudy: jax.Array,
    dudz: jax.Array,
    dvdx: jax.Array,
    dvdy: jax.Array,
    dvdz: jax.Array,
    dwdx: jax.Array,
    dwdy: jax.Array,
    dwdz: jax.Array,
    params: Params,
    *,
    dudz_face: jax.Array | None = None,
    dvdz_face: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    dudx = _to_sgs(dudx, params)
    dudy = _to_sgs(dudy, params)
    dudz = _to_sgs(dudz, params)
    dvdx = _to_sgs(dvdx, params)
    dvdy = _to_sgs(dvdy, params)
    dvdz = _to_sgs(dvdz, params)
    dwdx = _to_sgs(dwdx, params)
    dwdy = _to_sgs(dwdy, params)
    dwdz = _to_sgs(dwdz, params)

    sij_center = _strain_uv(
        dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz
    )
    sij_face = _strain_w(
        dudx,
        dudy,
        dudz,
        dvdx,
        dvdy,
        dvdz,
        dwdx,
        dwdy,
        dwdz,
        dudz_face=dudz_face,
        dvdz_face=dvdz_face,
    )
    cs2 = jnp.full_like(dudx, params.smagorinsky_cs**2)
    return tuple(
        t.astype(params.sgs_dtype)
        for t in _stress_from_cs2(cs2, sij_center, sij_face, params)
    )


def _physical_mask(q: jax.Array) -> jax.Array:
    return jnp.ones_like(q)


def _sym_dot(a: jax.Array, b: jax.Array) -> jax.Array:
    return (
        a[..., 0] * b[..., 0]
        + 2.0 * a[..., 1] * b[..., 1]
        + 2.0 * a[..., 2] * b[..., 2]
        + a[..., 3] * b[..., 3]
        + 2.0 * a[..., 4] * b[..., 4]
        + a[..., 5] * b[..., 5]
    )


def _strain_magnitude(sij: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.maximum(2.0 * _sym_dot(sij, sij), 0.0))


def _avg_next(q: jax.Array) -> jax.Array:
    return center_to_upper_faces(q)


def _avg_prev(q: jax.Array) -> jax.Array:
    return upper_face_to_center(q)


def _strain_uv(
    dudx: jax.Array,
    dudy: jax.Array,
    dudz: jax.Array,
    dvdx: jax.Array,
    dvdy: jax.Array,
    dvdz: jax.Array,
    dwdx: jax.Array,
    dwdy: jax.Array,
    dwdz: jax.Array,
) -> jax.Array:
    ux = dudx
    uy = dudy
    uz = dudz
    vx = dvdx
    vy = dvdy
    vz = dvdz
    wx = upper_face_to_center(dwdx)
    wy = upper_face_to_center(dwdy)
    wz = dwdz
    sij = jnp.stack(
        (
            ux,
            0.5 * (uy + vx),
            0.5 * (uz + wx),
            vy,
            0.5 * (vz + wy),
            wz,
        ),
        axis=-1,
    )
    return sij


def _strain_w(
    dudx: jax.Array,
    dudy: jax.Array,
    dudz: jax.Array,
    dvdx: jax.Array,
    dvdy: jax.Array,
    dvdz: jax.Array,
    dwdx: jax.Array,
    dwdy: jax.Array,
    dwdz: jax.Array,
    *,
    dudz_face: jax.Array | None = None,
    dvdz_face: jax.Array | None = None,
) -> jax.Array:
    ux = center_to_upper_faces(dudx)
    uy = center_to_upper_faces(dudy)
    uz = center_to_upper_faces(dudz) if dudz_face is None else dudz_face
    vx = center_to_upper_faces(dvdx)
    vy = center_to_upper_faces(dvdy)
    vz = center_to_upper_faces(dvdz) if dvdz_face is None else dvdz_face
    wx = dwdx
    wy = dwdy
    wz = center_to_upper_faces(dwdz)
    sij = jnp.stack(
        (
            ux,
            0.5 * (uy + vx),
            0.5 * (uz + wx),
            vy,
            0.5 * (vz + wy),
            wz,
        ),
        axis=-1,
    )
    return sij


def _spectral_box_filter(q: jax.Array, params: Params, filter_width: float) -> jax.Array:
    q_hat = jnp.fft.rfft2(q, axes=(0, 1))
    x_mode = jnp.abs(jnp.fft.fftfreq(params.nx, d=1.0) * params.nx)
    y_mode = jnp.fft.rfftfreq(params.ny, d=1.0) * params.ny
    # Match Fortran filter_4d_vector/tensor: NINT for the volume test
    # filters.  The wall-plane filter intentionally uses FLOOR instead.
    cutoff_x = jnp.floor(params.nx / (2.0 * filter_width) + 0.5)
    cutoff_y = jnp.floor(params.ny / (2.0 * filter_width) + 0.5)
    keep = (x_mode[:, None] < cutoff_x) & (y_mode[None, :] < cutoff_y)
    while keep.ndim < q_hat.ndim:
        keep = keep[..., None]
    q_filtered = jnp.fft.irfft2(q_hat * keep.astype(q_hat.dtype), s=(params.nx, params.ny), axes=(0, 1)).real
    return q_filtered.astype(q.dtype)


def _spectral_box_filter_concat(q: jax.Array, params: Params, filter_width: float) -> jax.Array:
    q_hat = jnp.fft.rfft2(q, axes=(0, 1))
    x_mode = jnp.abs(jnp.fft.fftfreq(params.nx, d=1.0) * params.nx)
    y_mode = jnp.fft.rfftfreq(params.ny, d=1.0) * params.ny
    cutoff_x = jnp.floor(params.nx / (2.0 * filter_width) + 0.5)
    cutoff_y = jnp.floor(params.ny / (2.0 * filter_width) + 0.5)
    keep = (x_mode[:, None] < cutoff_x) & (y_mode[None, :] < cutoff_y)
    q_filtered = jnp.fft.irfft2(
        q_hat * keep[:, :, None, None].astype(q_hat.dtype),
        s=(params.nx, params.ny),
        axes=(0, 1),
    ).real
    return q_filtered.astype(q.dtype)


def _spectral_box_filter_concat_hat(
    q_hat: jax.Array,
    template: jax.Array,
    params: Params,
    filter_width: float,
) -> jax.Array:
    x_mode = jnp.abs(jnp.fft.fftfreq(params.nx, d=1.0) * params.nx)
    y_mode = jnp.fft.rfftfreq(params.ny, d=1.0) * params.ny
    cutoff_x = jnp.floor(params.nx / (2.0 * filter_width) + 0.5)
    cutoff_y = jnp.floor(params.ny / (2.0 * filter_width) + 0.5)
    keep = (x_mode[:, None] < cutoff_x) & (y_mode[None, :] < cutoff_y)
    q_filtered = jnp.fft.irfft2(
        q_hat * keep[:, :, None, None].astype(q_hat.dtype),
        s=(params.nx, params.ny),
        axes=(0, 1),
    ).real
    return q_filtered.astype(template.dtype)


def _velocity_products(u: jax.Array, v: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
    w_uv = upper_face_to_center(w)
    vel = jnp.stack((u, v, w_uv), axis=-1)
    uu = jnp.stack(
        (
            vel[..., 0] * vel[..., 0],
            vel[..., 0] * vel[..., 1],
            vel[..., 0] * vel[..., 2],
            vel[..., 1] * vel[..., 1],
            vel[..., 1] * vel[..., 2],
            vel[..., 2] * vel[..., 2],
        ),
        axis=-1,
    )
    return vel, uu


def _lmqn(
    vel: jax.Array,
    uu: jax.Array,
    sij: jax.Array,
    params: Params,
    test_ratio: float,
) -> tuple[jax.Array, jax.Array]:
    filter_width = params.fgr * test_ratio
    ssij = _strain_magnitude(sij)[..., None] * sij
    filtered = _spectral_box_filter_concat(jnp.concatenate((vel, uu, sij, ssij), axis=-1), params, filter_width)
    return _lmqn_from_filtered(filtered, params, test_ratio)


def _lmqn_from_filtered(
    filtered: jax.Array,
    params: Params,
    test_ratio: float,
) -> tuple[jax.Array, jax.Array]:
    vel_hat = filtered[..., :3]
    uu_hat = filtered[..., 3:9]
    sij_hat = filtered[..., 9:15]
    ssij_hat = filtered[..., 15:21]

    l_ij = jnp.stack(
        (
            uu_hat[..., 0] - vel_hat[..., 0] * vel_hat[..., 0],
            uu_hat[..., 1] - vel_hat[..., 0] * vel_hat[..., 1],
            uu_hat[..., 2] - vel_hat[..., 0] * vel_hat[..., 2],
            uu_hat[..., 3] - vel_hat[..., 1] * vel_hat[..., 1],
            uu_hat[..., 4] - vel_hat[..., 1] * vel_hat[..., 2],
            uu_hat[..., 5] - vel_hat[..., 2] * vel_hat[..., 2],
        ),
        axis=-1,
    )
    delta = params.sgs_delta
    m_ij = 2.0 * delta * delta * (
        ssij_hat - test_ratio * test_ratio * _strain_magnitude(sij_hat)[..., None] * sij_hat
    )
    return _sym_dot(l_ij, m_ij), _sym_dot(m_ij, m_ij)


def _lmqn_pair(
    vel: jax.Array,
    uu: jax.Array,
    sij: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    ssij = _strain_magnitude(sij)[..., None] * sij
    q = jnp.concatenate((vel, uu, sij, ssij), axis=-1)
    q_hat = jnp.fft.rfft2(q, axes=(0, 1))
    filtered_2d = _spectral_box_filter_concat_hat(q_hat, q, params, params.fgr * params.tfr)
    filtered_4d = _spectral_box_filter_concat_hat(q_hat, q, params, params.fgr * params.tfr * params.tfr)
    lm, mm = _lmqn_from_filtered(filtered_2d, params, params.tfr)
    qn, nn = _lmqn_from_filtered(filtered_4d, params, params.tfr * params.tfr)
    return lm, mm, qn, nn


def _momentum_sd_terms(
    vel: jax.Array,
    uu: jax.Array,
    sij: jax.Array,
    params: Params,
    test_ratio: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Resolved stress and the two tensors in the original X_ij(beta)."""
    ssij = _strain_magnitude(sij)[..., None] * sij
    fields = jnp.concatenate((vel, uu, sij, ssij), axis=-1)
    filtered = _spectral_box_filter_concat(
        fields, params, params.fgr * test_ratio
    )
    vel_hat = filtered[..., :3]
    uu_hat = filtered[..., 3:9]
    sij_hat = filtered[..., 9:15]
    b_tensor = filtered[..., 15:21]
    l_tensor = jnp.stack(
        (
            uu_hat[..., 0] - vel_hat[..., 0] * vel_hat[..., 0],
            uu_hat[..., 1] - vel_hat[..., 0] * vel_hat[..., 1],
            uu_hat[..., 2] - vel_hat[..., 0] * vel_hat[..., 2],
            uu_hat[..., 3] - vel_hat[..., 1] * vel_hat[..., 1],
            uu_hat[..., 4] - vel_hat[..., 1] * vel_hat[..., 2],
            uu_hat[..., 5] - vel_hat[..., 2] * vel_hat[..., 2],
        ),
        axis=-1,
    )
    a_tensor = _strain_magnitude(sij_hat)[..., None] * sij_hat
    return l_tensor, b_tensor, a_tensor


def _porte_agel_plane_cs2(
    vel: jax.Array,
    uu: jax.Array,
    sij: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Plane-averaged Porté-Agel (2000) momentum coefficient."""
    l1, b1, a1 = _momentum_sd_terms(vel, uu, sij, params, params.tfr)
    l2, b2, a2 = _momentum_sd_terms(
        vel, uu, sij, params, params.tfr * params.tfr
    )

    def plane_dot(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.mean(_sym_dot(x, y), axis=(0, 1))

    p = plane_dot(l1, b1)
    q = plane_dot(l1, a1)
    r = plane_dot(b1, b1)
    s = plane_dot(a1, a1)
    t = plane_dot(b1, a1)
    p2 = plane_dot(l2, b2)
    q2 = plane_dot(l2, a2)
    r2 = plane_dot(b2, b2)
    s2 = plane_dot(a2, a2)
    t2 = plane_dot(b2, a2)

    polynomial = porte_agel_polynomial(p, q, r, s, t, p2, q2, r2, s2, t2)
    beta = largest_positive_real_polynomial_root(polynomial)
    numerator = p - 4.0 * beta * q
    denominator = 2.0 * params.sgs_delta**2 * (
        r - 8.0 * beta * t + 16.0 * beta * beta * s
    )
    cs2 = _safe_divide(numerator, denominator)
    fallback = _safe_divide(
        p - 4.0 * q,
        2.0 * params.sgs_delta**2 * (r - 8.0 * t + 16.0 * s),
    )
    valid = jnp.isfinite(beta) & jnp.isfinite(cs2) & (cs2 > 0.0)
    cs2 = jnp.where(valid, cs2, fallback)
    cs2 = jnp.clip(cs2, 1.0e-6, 0.81)
    beta = jnp.where(valid, beta, 1.0)
    shape = vel.shape[:-1]
    return tuple(
        jnp.broadcast_to(value[None, None, ...], shape)
        for value in (cs2, beta, valid.astype(cs2.dtype))
    )


def _history_bc(q: jax.Array) -> jax.Array:
    # LASD histories live at physical cell centers.  Match the original
    # Moeng/NCAR and retained C++ semantics by extending the nearest interior
    # history into the boundary centers.  Independently evolved endpoints
    # create artificial coefficient extrema where one-sided gradients apply.
    if q.shape[2] < 2:
        return q
    return q.at[:, :, 0, ...].set(q[:, :, 1, ...]).at[:, :, -1, ...].set(
        q[:, :, -2, ...]
    )


def _trilinear_departure_interp(
    q: jax.Array,
    displacement_x: jax.Array,
    displacement_y: jax.Array,
    displacement_z: jax.Array,
    params: Params,
) -> jax.Array:
    """Interpolate ``q`` once at each Lagrangian departure point.

    The three displacement components belong to the arrival cell.  Applying
    three one-dimensional interpolations in sequence is not equivalent when
    those displacements vary in space: later passes then sample neighboring
    cells' trajectory velocities.  Constructing all eight corners from one
    coordinate triplet preserves the intended NCAR/Fortran LASD semantics.
    Horizontal coordinates are periodic and the vertical coordinate is
    clamped to the available physical/halo planes.
    """
    shape = displacement_x.shape
    i = jnp.arange(q.shape[0], dtype=displacement_x.dtype)[:, None, None]
    j = jnp.arange(q.shape[1], dtype=displacement_y.dtype)[None, :, None]
    k = jnp.arange(q.shape[2], dtype=displacement_z.dtype)[None, None, :]

    xi = jnp.mod(i + displacement_x / params.dx, q.shape[0])
    eta = jnp.mod(j + displacement_y / params.dy, q.shape[1])
    zeta = jnp.clip(k + displacement_z / params.dz, 0.0, float(q.shape[2] - 1))
    xi = jnp.broadcast_to(xi, shape)
    eta = jnp.broadcast_to(eta, shape)
    zeta = jnp.broadcast_to(zeta, shape)

    i0 = jnp.floor(xi).astype(jnp.int32)
    j0 = jnp.floor(eta).astype(jnp.int32)
    k0 = jnp.floor(zeta).astype(jnp.int32)
    i1 = (i0 + 1) % q.shape[0]
    j1 = (j0 + 1) % q.shape[1]
    k1 = jnp.minimum(k0 + 1, q.shape[2] - 1)
    fx = xi - i0.astype(xi.dtype)
    fy = eta - j0.astype(eta.dtype)
    fz = zeta - k0.astype(zeta.dtype)
    while fx.ndim < q.ndim:
        fx = fx[..., None]
        fy = fy[..., None]
        fz = fz[..., None]

    q00 = (1.0 - fx) * q[i0, j0, k0, ...] + fx * q[i1, j0, k0, ...]
    q10 = (1.0 - fx) * q[i0, j1, k0, ...] + fx * q[i1, j1, k0, ...]
    q01 = (1.0 - fx) * q[i0, j0, k1, ...] + fx * q[i1, j0, k1, ...]
    q11 = (1.0 - fx) * q[i0, j1, k1, ...] + fx * q[i1, j1, k1, ...]
    q0 = (1.0 - fy) * q00 + fy * q10
    q1 = (1.0 - fy) * q01 + fy * q11
    return (1.0 - fz) * q0 + fz * q1


def _lagrangian_interp(q: jax.Array, u_lag: jax.Array, v_lag: jax.Array, w_lag: jax.Array, params: Params) -> jax.Array:
    dt_lag = params.dt * params.cs_count
    return _trilinear_departure_interp(
        q,
        -u_lag * dt_lag,
        -v_lag * dt_lag,
        -w_lag * dt_lag,
        params,
    )


def _lagrangian_average(
    current_a: jax.Array,
    current_b: jax.Array,
    old_a: jax.Array,
    old_b: jax.Array,
    u_lag: jax.Array,
    v_lag: jax.Array,
    w_lag: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array]:
    dt_lag = params.dt * params.cs_count
    product = old_a * old_b
    valid = (old_a > 0.0) & (old_b >= 0.0) & (product > 0.0)
    tn = 1.5 * params.sgs_delta
    tn = tn * jnp.where(valid, product ** (-0.125), 1.0)
    eps = jnp.where(valid, (dt_lag / tn) / (1.0 + dt_lag / tn), 0.0)
    a_interp = _lagrangian_interp(old_a, u_lag, v_lag, w_lag, params)
    b_interp = _lagrangian_interp(old_b, u_lag, v_lag, w_lag, params)
    return eps * current_a + (1.0 - eps) * a_interp, eps * current_b + (1.0 - eps) * b_interp


def _safe_divide(num: jax.Array, den: jax.Array) -> jax.Array:
    valid = jnp.abs(den) > 1.0e-30
    safe_den = jnp.where(valid, den, 1.0)
    return jnp.where(valid, num / safe_den, 0.0)


def _scale_dependence_beta(
    c_2d: jax.Array,
    c_4d: jax.Array,
    params: Params,
    scale_dependent: bool,
) -> jax.Array:
    """Return the LASD scale ratio with an optional undefined-ratio fallback."""
    exponent = jnp.log(jnp.asarray(params.tfr, dtype=params.sgs_dtype)) / (
        jnp.log(jnp.asarray(params.tfr * params.tfr, dtype=params.sgs_dtype))
        - jnp.log(jnp.asarray(params.tfr, dtype=params.sgs_dtype))
    )
    raw_beta = _safe_divide(c_4d, c_2d) ** exponent
    beta_floor = 1.0 / (params.tfr * params.tfr * params.tfr)
    beta = jnp.maximum(raw_beta, beta_floor)
    if params.lasd_clipped_beta_fallback:
        beta = jnp.where(raw_beta > beta_floor, beta, 1.0)
    elif params.lasd_invalid_beta_fallback:
        valid = (c_2d > 1.0e-30) & (c_4d > 1.0e-30)
        beta = jnp.where(valid, beta, 1.0)
    return jnp.where(scale_dependent, beta, 1.0)


def _update_lasd_coefficients(
    state: FlowState,
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    sij_uv: jax.Array,
    params: Params,
    update: bool,
    force_update: bool | None = None,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    u = _to_sgs(u, params)
    v = _to_sgs(v, params)
    w = _to_sgs(w, params)
    sij_uv = _to_sgs(sij_uv, params)
    cs2_state, lm_state, mm_state, qn_state, nn_state, u_lag_state, v_lag_state, w_lag_state = _cast_sgs_state(
        state,
        params,
    )

    if params.sgs_model == "porte_agel_sd":
        current = (
            cs2_state,
            (
                lm_state,
                mm_state,
                qn_state,
                nn_state,
                u_lag_state,
                v_lag_state,
                w_lag_state,
            ),
        )
        if not update:
            return current
        should_update = ((state.step + 1) % params.cs_count) == 0

        def update_plane_model(_: None):
            vel, uu = _velocity_products(u, v, w)
            cs2, beta, root_valid = _porte_agel_plane_cs2(
                vel, uu, sij_uv, params
            )
            zeros = jnp.zeros_like(beta)
            # As in the scalar plane model, retain beta and validity in the
            # otherwise-unused history slots for benchmark diagnostics.
            return cs2, (beta, root_valid, zeros, zeros, zeros, zeros, zeros)

        if force_update is True:
            return update_plane_model(None)
        if force_update is False:
            return current
        return jax.lax.cond(
            should_update, update_plane_model, lambda _: current, None
        )

    u_lag = u_lag_state + u / params.cs_count
    v_lag = v_lag_state + v / params.cs_count
    w_lag = w_lag_state + upper_face_to_center(w) / params.cs_count

    if not update:
        return cs2_state, (lm_state, mm_state, qn_state, nn_state, u_lag_state, v_lag_state, w_lag_state)

    should_update = ((state.step + 1) % params.cs_count) == 0

    def do_update(_: None) -> tuple[jax.Array, ...]:
        vel, uu = _velocity_products(u, v, w)
        lm, mm, qn, nn = _lmqn_pair(vel, uu, sij_uv, params)

        first_update = state.step == params.cs_count - 1
        lm_old = jnp.where(first_update, 0.03 * mm, lm_state)
        mm_old = jnp.where(first_update, mm, mm_state)
        qn_old = jnp.where(first_update, 0.03 * nn, qn_state)
        nn_old = jnp.where(first_update, nn, nn_state)
        lm_old = _history_bc(lm_old)
        mm_old = _history_bc(mm_old)
        qn_old = _history_bc(qn_old)
        nn_old = _history_bc(nn_old)

        lm_avg, mm_avg = _lagrangian_average(lm, mm, lm_old, mm_old, u_lag, v_lag, w_lag, params)
        qn_avg, nn_avg = _lagrangian_average(qn, nn, qn_old, nn_old, u_lag, v_lag, w_lag, params)

        cs2_2d = jnp.maximum(_safe_divide(lm_avg, mm_avg), 0.0)
        cs2_4d = jnp.maximum(_safe_divide(qn_avg, nn_avg), 0.0)
        scale_dependent = params.momentum_lasd_scale_dependent
        if scale_dependent is None:
            scale_dependent = params.lasd_scale_dependent
        beta = _scale_dependence_beta(cs2_2d, cs2_4d, params, scale_dependent)
        cs2_new = jnp.clip(_safe_divide(cs2_2d, beta), 1.0e-6, 0.81)
        cs2_new = cs2_new * _physical_mask(cs2_new)
        zero = jnp.zeros_like(u_lag)
        return cs2_new, lm_avg, mm_avg, qn_avg, nn_avg, zero, zero, zero

    def skip_update(_: None) -> tuple[jax.Array, ...]:
        return cs2_state, lm_state, mm_state, qn_state, nn_state, u_lag, v_lag, w_lag

    if force_update is True:
        cs2, lm_old_new, mm_old_new, qn_old_new, nn_old_new, u_lag_new, v_lag_new, w_lag_new = do_update(None)
        return cs2, (lm_old_new, mm_old_new, qn_old_new, nn_old_new, u_lag_new, v_lag_new, w_lag_new)
    if force_update is False:
        cs2, lm_old_new, mm_old_new, qn_old_new, nn_old_new, u_lag_new, v_lag_new, w_lag_new = skip_update(None)
        return cs2, (lm_old_new, mm_old_new, qn_old_new, nn_old_new, u_lag_new, v_lag_new, w_lag_new)

    cs2, lm_old_new, mm_old_new, qn_old_new, nn_old_new, u_lag_new, v_lag_new, w_lag_new = jax.lax.cond(
        should_update,
        do_update,
        skip_update,
        operand=None,
    )
    return cs2, (lm_old_new, mm_old_new, qn_old_new, nn_old_new, u_lag_new, v_lag_new, w_lag_new)


def _stress_from_cs2(cs2: jax.Array, sij_uv: jax.Array, sij_w: jax.Array, params: Params) -> tuple[jax.Array, ...]:
    delta = params.sgs_delta
    factor_uv = -2.0 * cs2 * delta * delta * _strain_magnitude(sij_uv)
    txx = factor_uv * sij_uv[..., 0]
    txy = factor_uv * sij_uv[..., 1]
    tyy = factor_uv * sij_uv[..., 3]
    tzz = factor_uv * sij_uv[..., 5]

    cs2_w = center_to_upper_faces(cs2)
    factor_w = -2.0 * cs2_w * delta * delta * _strain_magnitude(sij_w)
    txz = factor_w * sij_w[..., 2]
    tyz = factor_w * sij_w[..., 4]
    txz = txz.at[:, :, -1].set(0.0)
    tyz = tyz.at[:, :, -1].set(0.0)
    return txx, txy, txz, tyy, tyz, tzz


def lasd_sgs(
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
    params: Params,
    update: bool,
    force_update: bool | None = None,
    *,
    dudz_face: jax.Array | None = None,
    dvdz_face: jax.Array | None = None,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    u = _to_sgs(u, params)
    v = _to_sgs(v, params)
    w = _to_sgs(w, params)
    dudx = _to_sgs(dudx, params)
    dudy = _to_sgs(dudy, params)
    dudz = _to_sgs(dudz, params)
    dvdx = _to_sgs(dvdx, params)
    dvdy = _to_sgs(dvdy, params)
    dvdz = _to_sgs(dvdz, params)
    dwdx = _to_sgs(dwdx, params)
    dwdy = _to_sgs(dwdy, params)
    dwdz = _to_sgs(dwdz, params)
    sij_uv = _strain_uv(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
    sij_w = _strain_w(
        dudx,
        dudy,
        dudz,
        dvdx,
        dvdy,
        dvdz,
        dwdx,
        dwdy,
        dwdz,
        dudz_face=None if dudz_face is None else _to_sgs(dudz_face, params),
        dvdz_face=None if dvdz_face is None else _to_sgs(dvdz_face, params),
    )
    cs2, sgs_state = _update_lasd_coefficients(state, u, v, w, sij_uv, params, update, force_update)
    return _stress_from_cs2(cs2, sij_uv, sij_w, params), (cs2, *sgs_state)


def subgrid_stress(
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
    params: Params,
    update_lasd: bool,
    force_lasd_update: bool | None = None,
    *,
    dudz_face: jax.Array | None = None,
    dvdz_face: jax.Array | None = None,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    if params.sgs_model == "smagorinsky":
        stress = classic_smagorinsky(
            dudx,
            dudy,
            dudz,
            dvdx,
            dvdy,
            dvdz,
            dwdx,
            dwdy,
            dwdz,
            params,
            dudz_face=dudz_face,
            dvdz_face=dvdz_face,
        )
        sgs_state = _cast_sgs_state(state, params)
        return stress, sgs_state
    if params.sgs_model in {"lasd", "porte_agel_sd"}:
        return lasd_sgs(
            state,
            u,
            v,
            w,
            dudx,
            dudy,
            dudz,
            dvdx,
            dvdy,
            dvdz,
            dwdx,
            dwdy,
            dwdz,
            params,
            update_lasd,
            force_lasd_update,
            dudz_face=dudz_face,
            dvdz_face=dvdz_face,
        )
    raise ValueError(f"Unsupported sgs_model: {params.sgs_model}")
