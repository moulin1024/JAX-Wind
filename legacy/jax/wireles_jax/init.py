from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Params
from .state import FlowState


def apply_velocity_bc(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    params: Params | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Enforce physical conditions on the owned-upper-face velocity layout."""
    if params is not None and params.top_boundary_condition == "klemp_durran":
        # The Klemp--Durran condition radiates every non-zero horizontal
        # Fourier mode through the top.  The horizontally uniform mode cannot
        # radiate and is removed to preserve the closed-domain mass budget.
        w_top = w[:, :, -1] - jnp.mean(w[:, :, -1])
        w = w.at[:, :, -1].set(w_top)
    else:
        w = w.at[:, :, -1].set(0.0)
    return u, v, w


def apply_scalar_bc(q: jax.Array) -> jax.Array:
    return q


def apply_theta_bc(theta: jax.Array, params: Params) -> jax.Array:
    del params
    return theta


def log_law_profile(params: Params, ustar: float | None = None) -> jax.Array:
    zc = (jnp.arange(params.nz, dtype=params.dtype) + 0.5) * params.dz
    z_phys = jnp.maximum(zc * params.z_i, params.zo * 1.01)
    friction_velocity = params.u_fric if ustar is None else ustar
    profile = (friction_velocity / params.vonk) * jnp.log(z_phys / params.zo)
    cap_height = jnp.maximum(jnp.asarray(params.bl_height, dtype=params.dtype), params.zo * 1.01)
    cap_value = (friction_velocity / params.vonk) * jnp.log(cap_height / params.zo)
    profile = jnp.where(z_phys >= params.bl_height, cap_value, profile)
    return profile.astype(params.dtype)


def uniform_pressure_driven_profile(params: Params) -> jax.Array:
    """Uniform streamwise profile with the target log law's bulk momentum.

    This gives a pressure-driven case the target bulk momentum without
    prescribing either its wall stress or its final vertical profile.
    """
    target = log_law_profile(params, ustar=params.pressure_ustar)
    zc = (jnp.arange(params.nz, dtype=params.dtype) + 0.5) * params.dz
    forced = zc <= params.forcing_height
    bulk_speed = jnp.sum(jnp.where(forced, target, 0.0)) / jnp.maximum(jnp.sum(forced), 1)
    return jnp.full((params.nz,), bulk_speed, dtype=params.dtype)


def geostrophic_wind_profiles(params: Params) -> tuple[jax.Array, jax.Array]:
    shape = (params.nz,)
    u_profile = jnp.full(shape, params.geostrophic_u, dtype=params.dtype)
    v_profile = jnp.full(shape, params.geostrophic_v, dtype=params.dtype)
    return u_profile, v_profile


def theta_profile(params: Params) -> jax.Array:
    zc = (jnp.arange(params.nz, dtype=params.dtype) + 0.5) * params.dz
    z_phys = zc * params.z_i
    if params.theta_bc == "dirichlet":
        height = params.lz * params.z_i
        profile = params.theta_bottom + (params.theta_top - params.theta_bottom) * (z_phys / height)
    elif params.theta_profile == "deardorff_cbl":
        zi = jnp.asarray(params.cbl_mixed_layer_height, dtype=params.dtype)
        thickness = jnp.asarray(params.cbl_inversion_thickness, dtype=params.dtype)
        inversion_bottom = zi - 0.5 * thickness
        inversion_top = zi + 0.5 * thickness
        ramp_arg = (z_phys - inversion_bottom) / thickness
        ramp_arg = jnp.clip(ramp_arg, 0.0, 1.0)
        smooth_ramp = ramp_arg * ramp_arg * (3.0 - 2.0 * ramp_arg)
        free_z = jnp.maximum(z_phys - inversion_top, 0.0)
        profile = (
            params.theta0
            + params.cbl_inversion_strength * smooth_ramp
            + params.cbl_free_atmosphere_gradient * free_z
        )
    else:
        profile = params.theta0 + params.theta_initial_gradient * z_phys
    return profile.astype(params.dtype)


def qv_profile(params: Params) -> jax.Array:
    zc = (jnp.arange(params.nz, dtype=params.dtype) + 0.5) * params.dz
    z_phys = zc * params.z_i
    profile = params.qv0 + params.qv_initial_gradient * z_phys
    profile = jnp.maximum(profile, params.qv_floor)
    return profile.astype(params.dtype)


def initial_state(params: Params, seed: int = 0) -> FlowState:
    key = jax.random.PRNGKey(seed)
    key_u, key_v, key_w, key_theta = jax.random.split(key, 4)
    if params.initial_condition == "geostrophic":
        u_profile, v_profile = geostrophic_wind_profiles(params)
        u_inner = jnp.broadcast_to(u_profile, (params.nx, params.ny, params.nz)).astype(params.dtype)
        v_inner = jnp.broadcast_to(v_profile, (params.nx, params.ny, params.nz)).astype(params.dtype)
    elif params.initial_condition == "uniform_flow":
        u_inner = jnp.full(
            (params.nx, params.ny, params.nz), params.uniform_u, dtype=params.dtype
        )
        v_inner = jnp.full(
            (params.nx, params.ny, params.nz), params.uniform_v, dtype=params.dtype
        )
    else:
        profile = (
            uniform_pressure_driven_profile(params)
            if params.initial_condition == "uniform"
            else log_law_profile(params)
        )
        u_inner = jnp.broadcast_to(profile, (params.nx, params.ny, params.nz)).astype(params.dtype)
        v_inner = jnp.zeros((params.nx, params.ny, params.nz), dtype=params.dtype)
    if params.momentum_wall_model == "abl" and params.initial_velocity_noise > 0.0:
        noise = params.initial_velocity_noise * jax.random.normal(
            key_u,
            (params.nx, params.ny, min(4, params.nz)),
            dtype=params.dtype,
        )
        u_inner = u_inner.at[:, :, : noise.shape[2]].add(noise)
        if params.initial_condition == "geostrophic":
            v_noise = params.initial_velocity_noise * jax.random.normal(
                key_v,
                (params.nx, params.ny, min(4, params.nz)),
                dtype=params.dtype,
            )
            v_inner = v_inner.at[:, :, : v_noise.shape[2]].add(v_noise)
    w_inner = jnp.zeros((params.nx, params.ny, params.nz), dtype=params.dtype)
    if params.momentum_wall_model != "abl" and params.initial_velocity_noise > 0.0:
        shape = (params.nx, params.ny, params.nz)
        u_inner = u_inner + params.initial_velocity_noise * jax.random.normal(key_u, shape, dtype=params.dtype)
        v_inner = v_inner + params.initial_velocity_noise * jax.random.normal(key_v, shape, dtype=params.dtype)
        w_inner = w_inner + params.initial_velocity_noise * jax.random.normal(key_w, shape, dtype=params.dtype)
    theta_inner = jnp.broadcast_to(
        theta_profile(params), (params.nx, params.ny, params.nz)
    ).astype(params.dtype)
    if params.theta_perturbation_amplitude > 0.0:
        zc = (jnp.arange(params.nz, dtype=params.dtype) + 0.5) * params.dz
        if params.theta_perturbation_height is None:
            envelope = jnp.sin(jnp.pi * zc / params.lz)
        else:
            z_phys = zc * params.z_i
            perturbation_height = jnp.asarray(params.theta_perturbation_height, dtype=params.dtype)
            envelope = jnp.where(
                z_phys < perturbation_height,
                jnp.sin(jnp.pi * z_phys / perturbation_height),
                0.0,
            )
        perturb = params.theta_perturbation_amplitude * jax.random.normal(
            key_theta,
            (params.nx, params.ny, params.nz),
            dtype=params.dtype,
        )
        theta_inner = theta_inner + perturb * envelope[None, None, :]
    qv_inner = jnp.broadcast_to(
        qv_profile(params), (params.nx, params.ny, params.nz)
    ).astype(params.dtype)

    zeros = jnp.zeros((params.nx, params.ny, params.nz), dtype=params.dtype)
    sgs_zeros = jnp.zeros((params.nx, params.ny, params.nz), dtype=params.sgs_dtype)
    scalar_zeros = jnp.zeros((params.nx, params.ny, params.nz, 2), dtype=params.sgs_dtype)
    u = u_inner
    v = v_inner
    w = w_inner
    theta = theta_inner
    qv = qv_inner
    u, v, w = apply_velocity_bc(u, v, w, params)
    theta = apply_theta_bc(theta, params)
    qv = apply_scalar_bc(qv)
    base_cs2 = params.smagorinsky_cs * params.smagorinsky_cs
    scalar_c = jnp.empty((params.nx, params.ny, params.nz, 2), dtype=params.sgs_dtype)
    scalar_c = scalar_c.at[..., 0].set(base_cs2 / params.prandtl_t)
    scalar_c = scalar_c.at[..., 1].set(base_cs2 / params.schmidt_t)
    return FlowState(
        u=u,
        v=v,
        w=w,
        p=zeros,
        theta=theta,
        qv=qv,
        rhs_u_prev=zeros,
        rhs_v_prev=zeros,
        rhs_w_prev=zeros,
        rhs_theta_prev=zeros,
        rhs_qv_prev=zeros,
        lm_old=sgs_zeros,
        mm_old=sgs_zeros,
        qn_old=sgs_zeros,
        nn_old=sgs_zeros,
        cs2=jnp.full_like(sgs_zeros, params.smagorinsky_cs * params.smagorinsky_cs),
        scalar_c=scalar_c,
        scalar_lm_old=scalar_zeros,
        scalar_mm_old=scalar_zeros,
        scalar_qn_old=scalar_zeros,
        scalar_nn_old=scalar_zeros,
        u_lag=sgs_zeros,
        v_lag=sgs_zeros,
        w_lag=sgs_zeros,
        step=jnp.array(0, dtype=jnp.int32),
    )
