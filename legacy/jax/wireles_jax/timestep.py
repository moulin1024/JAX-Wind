from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Params
from .convection import convec
from .derivative import (
    ddx,
    ddy,
    ddxy_filter_many,
    ddz_uv_face,
    ddz_w,
    gradxy,
    horizontal_filter_many,
)
from .init import apply_theta_bc, apply_velocity_bc
from .grid import (
    divergence_upper_faces,
    face_gradient_to_center,
    gradient_to_upper_faces,
)
from .pressure import project_velocity
from .rhs import add_coriolis_geostrophic_forcing, assemble_rhs
from .scalar import (
    apply_moisture_bounds,
    buoyancy_from_theta_qv,
    scalar_rhs,
)
from .sgs import subgrid_stress
from .sponge import apply_rayleigh_sponge
from .state import FlowState, Operators
from .wall import apply_porte_agel_wall_correction, wall_stress
from .wind_tunnel import wind_tunnel_momentum_sources


def _ab_update(q: jax.Array, rhs: jax.Array, rhs_prev: jax.Array, step: jax.Array, params: Params) -> jax.Array:
    euler = q + params.dt * rhs
    ab2 = q + params.dt * (1.5 * rhs - 0.5 * rhs_prev)
    return jnp.where(step == 0, euler, ab2)


def _add_molecular_stress(
    stress: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    dudx: jax.Array,
    dudy: jax.Array,
    dudz: jax.Array,
    dudz_face: jax.Array,
    dvdx: jax.Array,
    dvdy: jax.Array,
    dvdz: jax.Array,
    dvdz_face: jax.Array,
    dwdx: jax.Array,
    dwdy: jax.Array,
    dwdz: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    nu = params.molecular_viscosity_internal
    if nu <= 0.0:
        return stress
    txx, txy, txz, tyy, tyz, tzz = stress
    txx = txx - 2.0 * nu * dudx
    txy = txy - nu * (dudy + dvdx)
    txz = txz - nu * (dudz_face + dwdx)
    tyy = tyy - 2.0 * nu * dvdy
    tyz = tyz - nu * (dvdz_face + dwdy)
    tzz = tzz - 2.0 * nu * dwdz
    txz = txz.at[:, :, -1].set(0.0)
    tyz = tyz.at[:, :, -1].set(0.0)
    return txx, txy, txz, tyy, tyz, tzz


def _momentum_rhs(
    state: FlowState,
    params: Params,
    ops: Operators,
    update_lasd: bool,
    force_lasd_update: bool | None = None,
) -> tuple[jax.Array, ...]:
    u, v, w = state.u, state.v, state.w
    u, v, w = apply_velocity_bc(u, v, w, params)
    (u, v, w), (dudx, dvdx, dwdx), (dudy, dvdy, dwdy) = ddxy_filter_many((u, v, w), params, ops)
    u, v, w = apply_velocity_bc(u, v, w, params)
    dudz_face = ddz_uv_face(u, params)
    dvdz_face = ddz_uv_face(v, params)
    dwdz = ddz_w(w, params)
    if params.momentum_wall_model == "abl":
        dudz_face, dvdz_face = apply_porte_agel_wall_correction(
            dudz_face,
            dvdz_face,
            correction_index=0,
            horizontal_average=params.horizontal_homogeneous,
        )
        txz0, tyz0, dudz0, dvdz0, _ = wall_stress(u, v, params)
    else:
        txz0 = jnp.zeros((params.nx, params.ny), dtype=params.dtype)
        tyz0 = jnp.zeros((params.nx, params.ny), dtype=params.dtype)
        dudz0 = jnp.zeros_like(txz0)
        dvdz0 = jnp.zeros_like(tyz0)
    dudz = face_gradient_to_center(dudz_face, dudz0)
    dvdz = face_gradient_to_center(dvdz_face, dvdz0)

    cx, cy, cz = convec(
        u,
        v,
        w,
        dudy,
        dudz_face,
        dvdx,
        dvdz_face,
        dwdx,
        dwdy,
    )
    (txx, txy, txz, tyy, tyz, tzz), sgs_state = subgrid_stress(
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
    txx, txy, txz, tyy, tyz, tzz = _add_molecular_stress(
        (txx, txy, txz, tyy, tyz, tzz),
        dudx,
        dudy,
        dudz,
        dudz_face,
        dvdx,
        dvdy,
        dvdz,
        dvdz_face,
        dwdx,
        dwdy,
        dwdz,
        params,
    )
    txy_dx, txy_dy = gradxy(txy, params, ops)
    divtx = ddx(txx, params, ops) + txy_dy + divergence_upper_faces(
        txz, params.dz, txz0
    )
    divty = txy_dx + ddy(tyy, params, ops) + divergence_upper_faces(
        tyz, params.dz, tyz0
    )
    divtz = (
        ddx(txz, params, ops)
        + ddy(tyz, params, ops)
        + gradient_to_upper_faces(tzz, params.dz)
    )
    divtz = divtz.at[:, :, -1].set(0.0)

    rhs_u = assemble_rhs(cx, divtx, params, pressure_force=True)
    rhs_v = assemble_rhs(cy, divty, params)
    rhs_w = assemble_rhs(cz, divtz, params)
    rhs_u, rhs_v = add_coriolis_geostrophic_forcing(rhs_u, rhs_v, u, v, params)
    source_u, source_v, source_w = wind_tunnel_momentum_sources(u, v, w, params)
    rhs_u = rhs_u + source_u
    rhs_v = rhs_v + source_v
    rhs_w = rhs_w + source_w
    (
        theta,
        qv,
        rhs_theta,
        rhs_qv,
        scalar_c,
        scalar_lm_old,
        scalar_mm_old,
        scalar_qn_old,
        scalar_nn_old,
    ) = scalar_rhs(
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
        sgs_state[0],
        params,
        ops,
        update_lasd,
        force_lasd_update,
        momentum_lasd_state=(sgs_state[1], sgs_state[2], sgs_state[3], sgs_state[4]),
    )

    rhs_w = rhs_w + buoyancy_from_theta_qv(theta, qv, params)
    return (
        u,
        v,
        w,
        theta,
        qv,
        rhs_u,
        rhs_v,
        rhs_w,
        rhs_theta,
        rhs_qv,
        *sgs_state,
        scalar_c,
        scalar_lm_old,
        scalar_mm_old,
        scalar_qn_old,
        scalar_nn_old,
    )


def _project_velocity(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    params: Params,
    ops: Operators,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    u, v, w = apply_velocity_bc(u, v, w, params)
    u, v, w, p = project_velocity(u, v, w, params, ops)
    u, v, w = apply_velocity_bc(u, v, w, params)
    return u, v, w, p


def _post_update_filter(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    theta: jax.Array,
    qv: jax.Array,
    params: Params,
    ops: Operators,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Apply sponge, walls, and filtering before projection in C++ order."""
    u, v, w = apply_rayleigh_sponge(u, v, w, params)
    u, v, w = apply_velocity_bc(u, v, w, params)
    fields: tuple[jax.Array, ...] = (u, v, w)
    if params.thermo_enabled:
        fields += (theta,)
    if params.moisture_enabled:
        fields += (qv,)
    filtered = horizontal_filter_many(fields, params, ops)
    u, v, w = filtered[:3]
    cursor = 3
    if params.thermo_enabled:
        theta = filtered[cursor]
        cursor += 1
    if params.moisture_enabled:
        qv = filtered[cursor]
        cursor += 1
    u, v, w = apply_velocity_bc(u, v, w, params)
    return u, v, w, theta, qv


def _rk4_stage_velocity(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    params: Params,
    ops: Operators,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    u, v, w = apply_velocity_bc(u, v, w, params)
    u, v, w = horizontal_filter_many((u, v, w), params, ops)
    u, v, w = apply_velocity_bc(u, v, w, params)
    if params.projection_mode == "stage":
        u, v, w, _ = _project_velocity(u, v, w, params, ops)
        return u, v, w
    if params.projection_mode == "final":
        return u, v, w
    raise ValueError(f"Unsupported projection_mode: {params.projection_mode}")


def _step_ab2(
    state: FlowState,
    params: Params,
    ops: Operators,
    force_lasd_update: bool | None = None,
) -> FlowState:
    (
        u,
        v,
        w,
        theta,
        qv,
        rhs_u,
        rhs_v,
        rhs_w,
        rhs_theta,
        rhs_qv,
        cs2,
        lm_old,
        mm_old,
        qn_old,
        nn_old,
        u_lag,
        v_lag,
        w_lag,
        scalar_c,
        scalar_lm_old,
        scalar_mm_old,
        scalar_qn_old,
        scalar_nn_old,
    ) = _momentum_rhs(state, params, ops, update_lasd=True, force_lasd_update=force_lasd_update)
    u_star = _ab_update(u, rhs_u, state.rhs_u_prev, state.step, params)
    v_star = _ab_update(v, rhs_v, state.rhs_v_prev, state.step, params)
    w_star = _ab_update(w, rhs_w, state.rhs_w_prev, state.step, params)
    theta_star = _ab_update(theta, rhs_theta, state.rhs_theta_prev, state.step, params)
    qv_star = _ab_update(qv, rhs_qv, state.rhs_qv_prev, state.step, params)
    u_star, v_star, w_star, theta_star, qv_star = _post_update_filter(
        u_star, v_star, w_star, theta_star, qv_star, params, ops
    )
    theta_new = apply_theta_bc(theta_star, params)
    qv_new = apply_moisture_bounds(qv_star, params)
    u_new, v_new, w_new, p_new = _project_velocity(u_star, v_star, w_star, params, ops)

    new_state = FlowState(
        u=u_new,
        v=v_new,
        w=w_new,
        p=p_new,
        theta=theta_new,
        qv=qv_new,
        rhs_u_prev=rhs_u,
        rhs_v_prev=rhs_v,
        rhs_w_prev=rhs_w,
        rhs_theta_prev=rhs_theta,
        rhs_qv_prev=rhs_qv,
        lm_old=lm_old,
        mm_old=mm_old,
        qn_old=qn_old,
        nn_old=nn_old,
        cs2=cs2,
        scalar_c=scalar_c,
        scalar_lm_old=scalar_lm_old,
        scalar_mm_old=scalar_mm_old,
        scalar_qn_old=scalar_qn_old,
        scalar_nn_old=scalar_nn_old,
        u_lag=u_lag,
        v_lag=v_lag,
        w_lag=w_lag,
        step=state.step + 1,
    )
    return new_state


def step_ab2(state: FlowState, params: Params, ops: Operators) -> FlowState:
    return _step_ab2(state, params, ops, force_lasd_update=None)


def step_ab2_lasd_update(state: FlowState, params: Params, ops: Operators) -> FlowState:
    return _step_ab2(state, params, ops, force_lasd_update=True)


def step_ab2_lasd_skip(state: FlowState, params: Params, ops: Operators) -> FlowState:
    return _step_ab2(state, params, ops, force_lasd_update=False)


def step_rk4(state: FlowState, params: Params, ops: Operators) -> FlowState:
    (
        u0,
        v0,
        w0,
        theta0,
        qv0,
        k1u,
        k1v,
        k1w,
        k1theta,
        k1qv,
        cs2,
        lm_old,
        mm_old,
        qn_old,
        nn_old,
        u_lag,
        v_lag,
        w_lag,
        scalar_c,
        scalar_lm_old,
        scalar_mm_old,
        scalar_qn_old,
        scalar_nn_old,
    ) = _momentum_rhs(state, params, ops, update_lasd=True)
    state_sgs = state._replace(
        cs2=cs2,
        lm_old=lm_old,
        mm_old=mm_old,
        qn_old=qn_old,
        nn_old=nn_old,
        u_lag=u_lag,
        v_lag=v_lag,
        w_lag=w_lag,
        scalar_c=scalar_c,
        scalar_lm_old=scalar_lm_old,
        scalar_mm_old=scalar_mm_old,
        scalar_qn_old=scalar_qn_old,
        scalar_nn_old=scalar_nn_old,
    )
    u1, v1, w1 = _rk4_stage_velocity(
        u0 + 0.5 * params.dt * k1u,
        v0 + 0.5 * params.dt * k1v,
        w0 + 0.5 * params.dt * k1w,
        params,
        ops,
    )
    theta1 = apply_theta_bc(theta0 + 0.5 * params.dt * k1theta, params)
    qv1 = apply_moisture_bounds(qv0 + 0.5 * params.dt * k1qv, params)

    _, _, _, _, _, k2u, k2v, k2w, k2theta, k2qv, *_ = _momentum_rhs(
        state_sgs._replace(u=u1, v=v1, w=w1, theta=theta1, qv=qv1),
        params,
        ops,
        update_lasd=False,
    )
    u2, v2, w2 = _rk4_stage_velocity(
        u0 + 0.5 * params.dt * k2u,
        v0 + 0.5 * params.dt * k2v,
        w0 + 0.5 * params.dt * k2w,
        params,
        ops,
    )
    theta2 = apply_theta_bc(theta0 + 0.5 * params.dt * k2theta, params)
    qv2 = apply_moisture_bounds(qv0 + 0.5 * params.dt * k2qv, params)

    _, _, _, _, _, k3u, k3v, k3w, k3theta, k3qv, *_ = _momentum_rhs(
        state_sgs._replace(u=u2, v=v2, w=w2, theta=theta2, qv=qv2),
        params,
        ops,
        update_lasd=False,
    )
    u3, v3, w3 = _rk4_stage_velocity(
        u0 + params.dt * k3u,
        v0 + params.dt * k3v,
        w0 + params.dt * k3w,
        params,
        ops,
    )
    theta3 = apply_theta_bc(theta0 + params.dt * k3theta, params)
    qv3 = apply_moisture_bounds(qv0 + params.dt * k3qv, params)

    _, _, _, _, _, k4u, k4v, k4w, k4theta, k4qv, *_ = _momentum_rhs(
        state_sgs._replace(u=u3, v=v3, w=w3, theta=theta3, qv=qv3),
        params,
        ops,
        update_lasd=False,
    )
    u_star = u0 + (params.dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
    v_star = v0 + (params.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    w_star = w0 + (params.dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
    theta_star = theta0 + (params.dt / 6.0) * (k1theta + 2.0 * k2theta + 2.0 * k3theta + k4theta)
    qv_star = qv0 + (params.dt / 6.0) * (k1qv + 2.0 * k2qv + 2.0 * k3qv + k4qv)
    u_star, v_star, w_star, theta_star, qv_star = _post_update_filter(
        u_star, v_star, w_star, theta_star, qv_star, params, ops
    )
    theta_new = apply_theta_bc(theta_star, params)
    qv_new = apply_moisture_bounds(qv_star, params)
    u_new, v_new, w_new, p_new = _project_velocity(u_star, v_star, w_star, params, ops)

    new_state = FlowState(
        u=u_new,
        v=v_new,
        w=w_new,
        p=p_new,
        theta=theta_new,
        qv=qv_new,
        rhs_u_prev=k4u,
        rhs_v_prev=k4v,
        rhs_w_prev=k4w,
        rhs_theta_prev=k4theta,
        rhs_qv_prev=k4qv,
        lm_old=lm_old,
        mm_old=mm_old,
        qn_old=qn_old,
        nn_old=nn_old,
        cs2=cs2,
        scalar_c=scalar_c,
        scalar_lm_old=scalar_lm_old,
        scalar_mm_old=scalar_mm_old,
        scalar_qn_old=scalar_qn_old,
        scalar_nn_old=scalar_nn_old,
        u_lag=u_lag,
        v_lag=v_lag,
        w_lag=w_lag,
        step=state.step + 1,
    )
    return new_state


def step_rk3(state: FlowState, params: Params, ops: Operators) -> FlowState:
    """Advance one step with the three-stage, third-order SSP Runge--Kutta method."""
    (
        u0,
        v0,
        w0,
        theta0,
        qv0,
        k1u,
        k1v,
        k1w,
        k1theta,
        k1qv,
        cs2,
        lm_old,
        mm_old,
        qn_old,
        nn_old,
        u_lag,
        v_lag,
        w_lag,
        scalar_c,
        scalar_lm_old,
        scalar_mm_old,
        scalar_qn_old,
        scalar_nn_old,
    ) = _momentum_rhs(state, params, ops, update_lasd=True)
    state_sgs = state._replace(
        cs2=cs2,
        lm_old=lm_old,
        mm_old=mm_old,
        qn_old=qn_old,
        nn_old=nn_old,
        u_lag=u_lag,
        v_lag=v_lag,
        w_lag=w_lag,
        scalar_c=scalar_c,
        scalar_lm_old=scalar_lm_old,
        scalar_mm_old=scalar_mm_old,
        scalar_qn_old=scalar_qn_old,
        scalar_nn_old=scalar_nn_old,
    )

    u1, v1, w1 = _rk4_stage_velocity(
        u0 + params.dt * k1u,
        v0 + params.dt * k1v,
        w0 + params.dt * k1w,
        params,
        ops,
    )
    theta1 = apply_theta_bc(theta0 + params.dt * k1theta, params)
    qv1 = apply_moisture_bounds(qv0 + params.dt * k1qv, params)
    _, _, _, _, _, k2u, k2v, k2w, k2theta, k2qv, *_ = _momentum_rhs(
        state_sgs._replace(u=u1, v=v1, w=w1, theta=theta1, qv=qv1),
        params,
        ops,
        update_lasd=False,
    )

    u2, v2, w2 = _rk4_stage_velocity(
        0.75 * u0 + 0.25 * (u1 + params.dt * k2u),
        0.75 * v0 + 0.25 * (v1 + params.dt * k2v),
        0.75 * w0 + 0.25 * (w1 + params.dt * k2w),
        params,
        ops,
    )
    theta2 = apply_theta_bc(0.75 * theta0 + 0.25 * (theta1 + params.dt * k2theta), params)
    qv2 = apply_moisture_bounds(0.75 * qv0 + 0.25 * (qv1 + params.dt * k2qv), params)
    _, _, _, _, _, k3u, k3v, k3w, k3theta, k3qv, *_ = _momentum_rhs(
        state_sgs._replace(u=u2, v=v2, w=w2, theta=theta2, qv=qv2),
        params,
        ops,
        update_lasd=False,
    )

    u_star = (1.0 / 3.0) * u0 + (2.0 / 3.0) * (u2 + params.dt * k3u)
    v_star = (1.0 / 3.0) * v0 + (2.0 / 3.0) * (v2 + params.dt * k3v)
    w_star = (1.0 / 3.0) * w0 + (2.0 / 3.0) * (w2 + params.dt * k3w)
    theta_star = (1.0 / 3.0) * theta0 + (2.0 / 3.0) * (theta2 + params.dt * k3theta)
    qv_star = (1.0 / 3.0) * qv0 + (2.0 / 3.0) * (qv2 + params.dt * k3qv)
    u_star, v_star, w_star, theta_star, qv_star = _post_update_filter(
        u_star, v_star, w_star, theta_star, qv_star, params, ops
    )
    theta_new = apply_theta_bc(theta_star, params)
    qv_new = apply_moisture_bounds(qv_star, params)
    u_new, v_new, w_new, p_new = _project_velocity(u_star, v_star, w_star, params, ops)

    return FlowState(
        u=u_new,
        v=v_new,
        w=w_new,
        p=p_new,
        theta=theta_new,
        qv=qv_new,
        rhs_u_prev=k3u,
        rhs_v_prev=k3v,
        rhs_w_prev=k3w,
        rhs_theta_prev=k3theta,
        rhs_qv_prev=k3qv,
        lm_old=lm_old,
        mm_old=mm_old,
        qn_old=qn_old,
        nn_old=nn_old,
        cs2=cs2,
        scalar_c=scalar_c,
        scalar_lm_old=scalar_lm_old,
        scalar_mm_old=scalar_mm_old,
        scalar_qn_old=scalar_qn_old,
        scalar_nn_old=scalar_nn_old,
        u_lag=u_lag,
        v_lag=v_lag,
        w_lag=w_lag,
        step=state.step + 1,
    )


def step(state: FlowState, params: Params, ops: Operators) -> FlowState:
    if params.time_scheme == "rk3":
        return step_rk3(state, params, ops)
    if params.time_scheme == "rk4":
        return step_rk4(state, params, ops)
    if params.time_scheme == "ab2":
        return step_ab2(state, params, ops)
    raise ValueError(f"Unsupported time_scheme: {params.time_scheme}")
