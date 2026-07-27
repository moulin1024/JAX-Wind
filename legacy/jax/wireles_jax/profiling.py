from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .convection import convec
from .config import Params
from .derivative import ddx, ddy, ddxy_filter_many, ddz_uv_face, ddz_w, gradxy, horizontal_filter_many
from .diagnostics import diagnostics, validate_cfl, validate_lasd_cfl
from .grid import (
    divergence_upper_faces,
    face_gradient_to_center,
    gradient_to_upper_faces,
    make_operators,
)
from .init import apply_theta_bc, apply_velocity_bc, initial_state
from .pressure import (
    _divergence_hat,
    _prepare_radiation_top,
    _pressure_and_horizontal_gradients_from_hat,
    _radiation_pressure_coefficient,
    _solve_pressure_hat,
)
from .rhs import add_coriolis_geostrophic_forcing, assemble_rhs
from .scalar import apply_moisture_bounds, buoyancy_from_theta_qv, scalar_rhs
from .sgs import _strain_uv, _strain_w, _stress_from_cs2, _to_sgs, _update_lasd_coefficients, classic_smagorinsky
from .sponge import apply_rayleigh_sponge
from .state import Diagnostics, FlowState, Operators
from .timestep import _ab_update
from .wall import apply_porte_agel_wall_correction, wall_stress


@dataclass(frozen=True)
class ProfileRow:
    step: int
    velocity_xy_ms: float
    wall_z_ms: float
    convection_ms: float
    sgs_strain_ms: float
    lasd_coefficients_ms: float
    sgs_stress_ms: float
    sgs_ms: float
    stress_divergence_ms: float
    rhs_assembly_ms: float
    rhs_ms: float
    ab_update_ms: float
    projection_divergence_ms: float
    pressure_solve_ms: float
    pressure_gradient_ms: float
    projection_update_ms: float
    projection_ms: float
    state_pack_ms: float
    solver_ms: float
    diagnostics_ms: float
    total_ms: float
    div_max: float


def _block_until_ready(value):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        value,
    )


def _ab_update_velocities(state: FlowState, rhs_result: tuple, params: Params) -> tuple:
    u, v, w, theta, qv, rhs_u, rhs_v, rhs_w, rhs_theta, rhs_qv, *_ = rhs_result
    return (
        _ab_update(u, rhs_u, state.rhs_u_prev, state.step, params),
        _ab_update(v, rhs_v, state.rhs_v_prev, state.step, params),
        _ab_update(w, rhs_w, state.rhs_w_prev, state.step, params),
        apply_theta_bc(_ab_update(theta, rhs_theta, state.rhs_theta_prev, state.step, params), params),
        apply_moisture_bounds(_ab_update(qv, rhs_qv, state.rhs_qv_prev, state.step, params), params),
    )


def _pack_ab2_state(state: FlowState, rhs_result: tuple, projected: tuple) -> FlowState:
    (
        _,
        _,
        _,
        _,
        _,
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
    ) = rhs_result
    u_new, v_new, w_new, theta_new, qv_new, p_new = projected
    return FlowState(
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


def _velocity_xy_derivatives(state: FlowState, params: Params, ops: Operators) -> tuple:
    u, v, w = state.u, state.v, state.w
    u, v, w = apply_velocity_bc(u, v, w, params)
    (u, v, w), (dudx, dvdx, dwdx), (dudy, dvdy, dwdy) = ddxy_filter_many((u, v, w), params, ops)
    u, v, w = apply_velocity_bc(u, v, w, params)
    return u, v, w, dudx, dvdx, dwdx, dudy, dvdy, dwdy


def _wall_z_derivatives(xy_result: tuple, params: Params) -> tuple:
    u, v, w, dudx, dvdx, dwdx, dudy, dvdy, dwdy = xy_result
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
    return (
        u,
        v,
        w,
        dudx,
        dvdx,
        dwdx,
        dudy,
        dvdy,
        dwdy,
        dudz,
        dvdz,
        dudz_face,
        dvdz_face,
        dwdz,
        txz0,
        tyz0,
    )


def _convection_terms(wall_result: tuple) -> tuple:
    u, v, w, _, dvdx, dwdx, dudy, _, dwdy, _, _, dudz_face, dvdz_face, *_ = wall_result
    return convec(
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


def _sgs_strain_terms(wall_result: tuple, params: Params) -> tuple:
    (
        u,
        v,
        w,
        dudx,
        dvdx,
        dwdx,
        dudy,
        dvdy,
        dwdy,
        dudz,
        dvdz,
        dudz_face,
        dvdz_face,
        dwdz,
        txz0,
        tyz0,
    ) = wall_result
    u = _to_sgs(u, params)
    v = _to_sgs(v, params)
    w = _to_sgs(w, params)
    dudx = _to_sgs(dudx, params)
    dudy = _to_sgs(dudy, params)
    dudz = _to_sgs(dudz, params)
    dvdx = _to_sgs(dvdx, params)
    dvdy = _to_sgs(dvdy, params)
    dvdz = _to_sgs(dvdz, params)
    dudz_face = _to_sgs(dudz_face, params)
    dvdz_face = _to_sgs(dvdz_face, params)
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
        dudz_face=dudz_face,
        dvdz_face=dvdz_face,
    )
    return (
        u,
        v,
        w,
        txz0,
        tyz0,
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
        sij_uv,
        sij_w,
    )


def _sgs_coefficients(
    state: FlowState,
    strain_result: tuple,
    params: Params,
    force_lasd_update: bool | None = None,
) -> tuple:
    u, v, w = strain_result[:3]
    sij_uv = strain_result[-2]
    if params.sgs_model == "lasd":
        cs2, sgs_state = _update_lasd_coefficients(
            state,
            u,
            v,
            w,
            sij_uv,
            params,
            update=True,
            force_update=force_lasd_update,
        )
        return cs2, *sgs_state
    return (
        state.cs2,
        state.lm_old,
        state.mm_old,
        state.qn_old,
        state.nn_old,
        state.u_lag,
        state.v_lag,
        state.w_lag,
    )


def _sgs_stress_from_coefficients(strain_result: tuple, coeff_result: tuple, params: Params) -> tuple:
    (
        _u,
        _v,
        _w,
        txz0,
        tyz0,
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
        sij_uv,
        sij_w,
    ) = strain_result
    if params.sgs_model == "lasd":
        txx, txy, txz, tyy, tyz, tzz = _stress_from_cs2(coeff_result[0], sij_uv, sij_w, params)
    else:
        txx, txy, txz, tyy, tyz, tzz = classic_smagorinsky(
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
    return (txx, txy, txz, tyy, tyz, tzz), coeff_result, txz0, tyz0


def _stress_divergence(sgs_result: tuple, params: Params, ops: Operators) -> tuple:
    txx, txy, txz, tyy, tyz, tzz = sgs_result[0]
    txz0, tyz0 = sgs_result[2:]
    txy_dx, txy_dy = gradxy(txy, params, ops)
    divtx = ddx(txx, params, ops) + txy_dy + divergence_upper_faces(txz, params.dz, txz0)
    divty = txy_dx + ddy(tyy, params, ops) + divergence_upper_faces(tyz, params.dz, tyz0)
    divtz = ddx(txz, params, ops) + ddy(tyz, params, ops) + gradient_to_upper_faces(tzz, params.dz)
    return divtx, divty, divtz


def _rhs_assembly(
    state: FlowState,
    wall_result: tuple,
    conv_result: tuple,
    stress_div_result: tuple,
    sgs_result: tuple,
    params: Params,
    ops: Operators,
    force_lasd_update: bool | None = False,
) -> tuple:
    (
        u,
        v,
        w,
        dudx,
        dvdx,
        dwdx,
        dudy,
        dvdy,
        dwdy,
        dudz,
        dvdz,
        dwdz,
        *_,
    ) = wall_result
    cx, cy, cz = conv_result
    divtx, divty, divtz = stress_div_result
    sgs_state = sgs_result[1]
    rhs_u = assemble_rhs(cx, divtx, params, pressure_force=True)
    rhs_v = assemble_rhs(cy, divty, params)
    rhs_w = assemble_rhs(cz, divtz, params)
    rhs_u, rhs_v = add_coriolis_geostrophic_forcing(rhs_u, rhs_v, u, v, params)
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
        update_lasd=True,
        force_lasd_update=force_lasd_update,
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


def _projection_divergence(predictor: tuple, params: Params, ops: Operators) -> tuple:
    u, v, w, theta, qv = predictor
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
    u, v, w = apply_velocity_bc(u, v, w, params)
    w, top_w_hat = _prepare_radiation_top(w, params, ops)
    return u, v, w, theta, qv, _divergence_hat(u, v, w, params, ops), top_w_hat


def _projection_pressure_solve(proj_div_result: tuple, params: Params, ops: Operators) -> tuple:
    u, v, w, theta, qv, div_hat, top_w_hat = proj_div_result
    p_hat = _solve_pressure_hat(div_hat / params.dt, params, ops, top_w_hat=top_w_hat)
    return u, v, w, theta, qv, p_hat, top_w_hat


def _projection_gradient_ifft(proj_solve_result: tuple, params: Params, ops: Operators) -> tuple:
    u, v, w, theta, qv, p_hat, top_w_hat = proj_solve_result
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
    return u, v, w, theta, qv, p, dpdx, dpdy, dpdz


def _projection_velocity_update(proj_gradient_result: tuple, params: Params) -> tuple:
    u, v, w, theta, qv, p, dpdx, dpdy, dpdz = proj_gradient_result
    u, v, w = u - params.dt * dpdx, v - params.dt * dpdy, w - params.dt * dpdz
    u, v, w = apply_velocity_bc(u, v, w, params)
    return u, v, w, theta, qv, p


def _time_call(fn: Callable, *args):
    start = time.perf_counter()
    value = _block_until_ready(fn(*args))
    return value, (time.perf_counter() - start) * 1000.0


def _compile_component(name: str, lower_call: Callable, status_callback: Callable[[str], None] | None):
    start = time.perf_counter()
    if status_callback is not None:
        status_callback(f"[profile] compiling {name}")
    compiled = lower_call().compile()
    if status_callback is not None:
        status_callback(f"[profile] compiled {name} in {time.perf_counter() - start:.1f}s")
    return compiled


def _make_profile_functions(params: Params, ops: Operators, state: FlowState, status_callback: Callable[[str], None] | None):
    xy_raw = lambda s: _velocity_xy_derivatives(s, params, ops)
    wall_raw = lambda xy: _wall_z_derivatives(xy, params)
    convection_raw = _convection_terms
    sgs_strain_raw = lambda wall: _sgs_strain_terms(wall, params)
    sgs_coefficients_skip_raw = lambda s, strain: _sgs_coefficients(s, strain, params, force_lasd_update=False)
    sgs_coefficients_update_raw = lambda s, strain: _sgs_coefficients(s, strain, params, force_lasd_update=True)
    sgs_stress_raw = lambda strain, coeff: _sgs_stress_from_coefficients(strain, coeff, params)
    stress_raw = lambda sgs: _stress_divergence(sgs, params, ops)
    rhs_raw = lambda s, wall, conv, stress, sgs: _rhs_assembly(s, wall, conv, stress, sgs, params, ops)
    ab_raw = lambda s, rhs: _ab_update_velocities(s, rhs, params)
    proj_div_raw = lambda predictor: _projection_divergence(predictor, params, ops)
    proj_solve_raw = lambda proj_div: _projection_pressure_solve(proj_div, params, ops)
    proj_gradient_raw = lambda proj_solve: _projection_gradient_ifft(proj_solve, params, ops)
    proj_update_raw = lambda proj_gradient: _projection_velocity_update(proj_gradient, params)
    pack_raw = _pack_ab2_state
    diagnostics_raw = lambda s: diagnostics(s, params, ops)

    if not params.use_jit:
        return (
            xy_raw,
            wall_raw,
            convection_raw,
            sgs_strain_raw,
            sgs_coefficients_skip_raw,
            sgs_coefficients_update_raw,
            sgs_stress_raw,
            stress_raw,
            rhs_raw,
            ab_raw,
            proj_div_raw,
            proj_solve_raw,
            proj_gradient_raw,
            proj_update_raw,
            pack_raw,
            diagnostics_raw,
        )

    xy_jit = jax.jit(xy_raw)
    xy_fn = _compile_component("velocity_xy_derivatives", lambda: xy_jit.lower(state), status_callback)
    xy_sample = _block_until_ready(xy_fn(state))

    wall_jit = jax.jit(wall_raw)
    wall_fn = _compile_component("wall_z_derivatives", lambda: wall_jit.lower(xy_sample), status_callback)
    wall_sample = _block_until_ready(wall_fn(xy_sample))

    convection_jit = jax.jit(convection_raw)
    convection_fn = _compile_component("convection", lambda: convection_jit.lower(wall_sample), status_callback)
    convection_sample = _block_until_ready(convection_fn(wall_sample))

    sgs_strain_jit = jax.jit(sgs_strain_raw)
    sgs_strain_fn = _compile_component("sgs_strain", lambda: sgs_strain_jit.lower(wall_sample), status_callback)
    sgs_strain_sample = _block_until_ready(sgs_strain_fn(wall_sample))

    sgs_coefficients_skip_jit = jax.jit(sgs_coefficients_skip_raw)
    sgs_coefficients_skip_fn = _compile_component(
        "lasd_coefficients_skip",
        lambda: sgs_coefficients_skip_jit.lower(state, sgs_strain_sample),
        status_callback,
    )
    sgs_coefficients_sample = _block_until_ready(sgs_coefficients_skip_fn(state, sgs_strain_sample))

    sgs_coefficients_update_jit = jax.jit(sgs_coefficients_update_raw)
    sgs_coefficients_update_fn = _compile_component(
        "lasd_coefficients_update",
        lambda: sgs_coefficients_update_jit.lower(state, sgs_strain_sample),
        status_callback,
    )
    _block_until_ready(sgs_coefficients_update_fn(state, sgs_strain_sample))

    sgs_stress_jit = jax.jit(sgs_stress_raw)
    sgs_stress_fn = _compile_component(
        "sgs_stress",
        lambda: sgs_stress_jit.lower(sgs_strain_sample, sgs_coefficients_sample),
        status_callback,
    )
    sgs_sample = _block_until_ready(sgs_stress_fn(sgs_strain_sample, sgs_coefficients_sample))

    stress_jit = jax.jit(stress_raw)
    stress_fn = _compile_component("stress_divergence", lambda: stress_jit.lower(sgs_sample), status_callback)
    stress_sample = _block_until_ready(stress_fn(sgs_sample))

    rhs_jit = jax.jit(rhs_raw)
    rhs_fn = _compile_component(
        "rhs_assembly",
        lambda: rhs_jit.lower(state, wall_sample, convection_sample, stress_sample, sgs_sample),
        status_callback,
    )
    rhs_sample = _block_until_ready(rhs_fn(state, wall_sample, convection_sample, stress_sample, sgs_sample))

    ab_jit = jax.jit(ab_raw)
    ab_fn = _compile_component("ab_update", lambda: ab_jit.lower(state, rhs_sample), status_callback)
    ab_sample = _block_until_ready(ab_fn(state, rhs_sample))

    proj_div_jit = jax.jit(proj_div_raw)
    proj_div_fn = _compile_component("projection_divergence", lambda: proj_div_jit.lower(ab_sample), status_callback)
    proj_div_sample = _block_until_ready(proj_div_fn(ab_sample))

    proj_solve_jit = jax.jit(proj_solve_raw)
    proj_solve_fn = _compile_component("pressure_solve", lambda: proj_solve_jit.lower(proj_div_sample), status_callback)
    proj_solve_sample = _block_until_ready(proj_solve_fn(proj_div_sample))

    proj_gradient_jit = jax.jit(proj_gradient_raw)
    proj_gradient_fn = _compile_component("pressure_gradient_ifft", lambda: proj_gradient_jit.lower(proj_solve_sample), status_callback)
    proj_gradient_sample = _block_until_ready(proj_gradient_fn(proj_solve_sample))

    proj_update_jit = jax.jit(proj_update_raw)
    proj_update_fn = _compile_component("projection_velocity_update", lambda: proj_update_jit.lower(proj_gradient_sample), status_callback)
    project_sample = _block_until_ready(proj_update_fn(proj_gradient_sample))

    pack_jit = jax.jit(pack_raw)
    pack_fn = _compile_component("state_pack", lambda: pack_jit.lower(state, rhs_sample, project_sample), status_callback)
    state_sample = _block_until_ready(pack_fn(state, rhs_sample, project_sample))

    diagnostics_jit = jax.jit(diagnostics_raw)
    diagnostics_fn = _compile_component("diagnostics", lambda: diagnostics_jit.lower(state_sample), status_callback)
    return (
        xy_fn,
        wall_fn,
        convection_fn,
        sgs_strain_fn,
        sgs_coefficients_skip_fn,
        sgs_coefficients_update_fn,
        sgs_stress_fn,
        stress_fn,
        rhs_fn,
        ab_fn,
        proj_div_fn,
        proj_solve_fn,
        proj_gradient_fn,
        proj_update_fn,
        pack_fn,
        diagnostics_fn,
    )


def _profile_one_step(
    state: FlowState,
    params: Params,
    xy_fn: Callable,
    wall_fn: Callable,
    convection_fn: Callable,
    sgs_strain_fn: Callable,
    sgs_coefficients_skip_fn: Callable,
    sgs_coefficients_update_fn: Callable,
    sgs_stress_fn: Callable,
    stress_fn: Callable,
    rhs_fn: Callable,
    ab_fn: Callable,
    proj_div_fn: Callable,
    proj_solve_fn: Callable,
    proj_gradient_fn: Callable,
    proj_update_fn: Callable,
    pack_fn: Callable,
    diagnostics_fn: Callable,
) -> tuple[FlowState, ProfileRow]:
    total_start = time.perf_counter()
    xy_result, velocity_xy_ms = _time_call(xy_fn, state)
    wall_result, wall_z_ms = _time_call(wall_fn, xy_result)
    conv_result, convection_ms = _time_call(convection_fn, wall_result)
    sgs_strain_result, sgs_strain_ms = _time_call(sgs_strain_fn, wall_result)
    update_lasd = params.sgs_model == "lasd" and ((int(state.step) + 1) % params.cs_count) == 0
    if update_lasd:
        update_diag = jax.block_until_ready(diagnostics_fn(state))
        validate_lasd_cfl(update_diag, params)
    sgs_coefficients_fn = sgs_coefficients_update_fn if update_lasd else sgs_coefficients_skip_fn
    sgs_coefficients_result, lasd_coefficients_ms = _time_call(sgs_coefficients_fn, state, sgs_strain_result)
    sgs_result, sgs_stress_ms = _time_call(sgs_stress_fn, sgs_strain_result, sgs_coefficients_result)
    sgs_ms = sgs_strain_ms + lasd_coefficients_ms + sgs_stress_ms
    stress_div_result, stress_divergence_ms = _time_call(stress_fn, sgs_result)
    rhs_result, rhs_assembly_ms = _time_call(rhs_fn, state, wall_result, conv_result, stress_div_result, sgs_result)
    rhs_ms = velocity_xy_ms + wall_z_ms + convection_ms + sgs_ms + stress_divergence_ms + rhs_assembly_ms
    predictor, ab_update_ms = _time_call(ab_fn, state, rhs_result)
    proj_div_result, projection_divergence_ms = _time_call(proj_div_fn, predictor)
    proj_solve_result, pressure_solve_ms = _time_call(proj_solve_fn, proj_div_result)
    proj_gradient_result, pressure_gradient_ms = _time_call(proj_gradient_fn, proj_solve_result)
    projected, projection_update_ms = _time_call(proj_update_fn, proj_gradient_result)
    projection_ms = projection_divergence_ms + pressure_solve_ms + pressure_gradient_ms + projection_update_ms
    new_state, state_pack_ms = _time_call(pack_fn, state, rhs_result, projected)
    solver_ms = (time.perf_counter() - total_start) * 1000.0
    diag, diagnostics_ms = _time_call(diagnostics_fn, new_state)
    validate_cfl(diag)
    total_ms = (time.perf_counter() - total_start) * 1000.0
    row = ProfileRow(
        step=int(diag.step),
        velocity_xy_ms=velocity_xy_ms,
        wall_z_ms=wall_z_ms,
        convection_ms=convection_ms,
        sgs_strain_ms=sgs_strain_ms,
        lasd_coefficients_ms=lasd_coefficients_ms,
        sgs_stress_ms=sgs_stress_ms,
        sgs_ms=sgs_ms,
        stress_divergence_ms=stress_divergence_ms,
        rhs_assembly_ms=rhs_assembly_ms,
        rhs_ms=rhs_ms,
        ab_update_ms=ab_update_ms,
        projection_divergence_ms=projection_divergence_ms,
        pressure_solve_ms=pressure_solve_ms,
        pressure_gradient_ms=pressure_gradient_ms,
        projection_update_ms=projection_update_ms,
        projection_ms=projection_ms,
        state_pack_ms=state_pack_ms,
        solver_ms=solver_ms,
        diagnostics_ms=diagnostics_ms,
        total_ms=total_ms,
        div_max=float(diag.div_max),
    )
    return new_state, row


def profile_ab2(
    params: Params,
    warmup_steps: int = 2,
    profile_steps: int | None = None,
    seed: int = 0,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[FlowState, list[ProfileRow]]:
    if params.time_scheme != "ab2":
        raise ValueError("Profiling mode currently supports time_scheme='ab2' only.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    run_steps = params.nsteps if profile_steps is None else profile_steps
    if run_steps <= 0:
        raise ValueError("profile_steps must be positive.")
    if warmup_steps >= run_steps:
        raise ValueError("profile_warmup must be smaller than the profiled run step count.")

    if status_callback is not None:
        status_callback(
            "[profile] split-step mode synchronizes after each component; "
            "normal AB2+LASD runs use separate skip/update step kernels."
        )

    ops = make_operators(params)
    state = initial_state(params, seed)
    (
        xy_fn,
        wall_fn,
        convection_fn,
        sgs_strain_fn,
        sgs_coefficients_skip_fn,
        sgs_coefficients_update_fn,
        sgs_stress_fn,
        stress_fn,
        rhs_fn,
        ab_fn,
        proj_div_fn,
        proj_solve_fn,
        proj_gradient_fn,
        proj_update_fn,
        pack_fn,
        diagnostics_fn,
    ) = _make_profile_functions(
        params,
        ops,
        state,
        status_callback,
    )

    if status_callback is not None:
        status_callback(
            f"[profile] running {run_steps} step(s); "
            f"excluding first {warmup_steps} step(s) from averages"
    )
    rows: list[ProfileRow] = []
    for n in range(run_steps):
        state, row = _profile_one_step(
            state,
            params,
            xy_fn,
            wall_fn,
            convection_fn,
            sgs_strain_fn,
            sgs_coefficients_skip_fn,
            sgs_coefficients_update_fn,
            sgs_stress_fn,
            stress_fn,
            rhs_fn,
            ab_fn,
            proj_div_fn,
            proj_solve_fn,
            proj_gradient_fn,
            proj_update_fn,
            pack_fn,
            diagnostics_fn,
        )
        if n >= warmup_steps:
            rows.append(row)
    return state, rows
