from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


JAX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JAX_ROOT))


def test_rigid_lid_only_constrains_the_owned_top_w_face() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.init import apply_velocity_bc

    params = Params(nx=4, ny=4, nz=5, lz=1.0, z_i=1.0, zo=0.01, momentum_wall_model="free_slip")
    shape = (params.nx, params.ny, params.nz)
    u = jnp.arange(np.prod(shape), dtype=params.dtype).reshape(shape)
    v = -u
    w = jnp.ones(shape, dtype=params.dtype)

    u_bc, v_bc, w_bc = jax.block_until_ready(apply_velocity_bc(u, v, w, params))

    np.testing.assert_array_equal(np.asarray(u_bc), np.asarray(u))
    np.testing.assert_array_equal(np.asarray(v_bc), np.asarray(v))
    np.testing.assert_allclose(np.asarray(w_bc[:, :, :-1]), 1.0)
    np.testing.assert_allclose(np.asarray(w_bc[:, :, -1]), 0.0)


def test_scalar_boundaries_do_not_require_persistent_ghost_cells() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.init import apply_scalar_bc, apply_theta_bc

    params = Params(
        nx=2,
        ny=2,
        nz=6,
        lz=1.0,
        z_i=1000.0,
        zo=0.01,
        thermo_enabled=True,
        theta_bc="dirichlet",
        theta_bottom=299.0,
        theta_top=305.0,
    )
    q = jnp.arange(params.nz, dtype=params.dtype)[None, None, :]
    q = jnp.broadcast_to(q, (params.nx, params.ny, params.nz))
    q_bc = apply_scalar_bc(q)
    np.testing.assert_array_equal(np.asarray(q_bc), np.asarray(q))

    theta_bc = apply_theta_bc(q, params)
    np.testing.assert_array_equal(np.asarray(theta_bc), np.asarray(q))


def test_ab2_filters_new_nyquist_content_before_projection() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params, initial_state
    from wireles_jax.grid import make_operators
    from wireles_jax.timestep import step_ab2

    params = Params(
        nx=8,
        ny=8,
        nz=5,
        lz=1.0,
        z_i=1.0,
        dt=0.01,
        u_fric=0.0,
        pressure_force=0.0,
        zo=0.01,
        momentum_wall_model="free_slip",
        initial_velocity_noise=0.0,
        sgs_model="smagorinsky",
        horizontal_dealias=True,
        pressure_filter_nyquist=True,
        dtype=jnp.float32,
    )
    state = initial_state(params)
    checkerboard = (1.0 - 2.0 * (jnp.arange(params.nx) % 2)).astype(params.dtype)
    checkerboard = jnp.broadcast_to(
        checkerboard[:, None, None],
        (params.nx, params.ny, params.nz),
    )
    rhs_u_prev = jnp.zeros_like(state.u).at[:, :, :].set(checkerboard)
    state = state._replace(step=jnp.asarray(1, dtype=state.step.dtype), rhs_u_prev=rhs_u_prev)

    updated = jax.block_until_ready(step_ab2(state, params, make_operators(params)))

    np.testing.assert_allclose(np.asarray(updated.u), 0.0, atol=2.0e-7)
    np.testing.assert_allclose(np.asarray(updated.v), 0.0, atol=2.0e-7)
    np.testing.assert_allclose(np.asarray(updated.w[:, :, -1]), 0.0, atol=0.0)
