from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


JAX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JAX_ROOT))


def radiation_params(jnp):
    from wireles_jax import Params

    return Params(
        nx=8,
        ny=6,
        nz=7,
        lx=4.0,
        ly=3.0,
        lz=1.5,
        z_i=1600.0,
        dt=1.25 / 1600.0,
        momentum_wall_model="free_slip",
        top_boundary_condition="klemp_durran",
        theta_top_gradient=0.003,
        theta0=300.0,
        g=9.81,
        dtype=jnp.float32,
    )


def test_radiation_frequency_is_derived_from_free_atmosphere_gradient() -> None:
    jnp = pytest.importorskip("jax.numpy")

    params = radiation_params(jnp)
    expected = np.sqrt(params.g * params.theta_top_gradient / params.theta0)
    assert params.radiation_brunt_vaisala_physical == pytest.approx(expected)
    assert params.radiation_brunt_vaisala_internal == pytest.approx(expected * params.z_i)


def test_radiation_velocity_boundary_preserves_only_nonzero_top_modes() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax.init import apply_velocity_bc

    params = radiation_params(jnp)
    shape = (params.nx, params.ny, params.nz)
    u = jnp.zeros(shape, dtype=params.dtype)
    v = jnp.zeros(shape, dtype=params.dtype)
    x = jnp.arange(params.nx, dtype=params.dtype)
    top = 2.5 + jnp.sin(2.0 * jnp.pi * x / params.nx)[:, None]
    top = jnp.broadcast_to(top, (params.nx, params.ny))
    w = jnp.zeros(shape, dtype=params.dtype).at[:, :, -1].set(top)

    _, _, w_bc = jax.block_until_ready(apply_velocity_bc(u, v, w, params))
    expected = np.sin(2.0 * np.pi * np.arange(params.nx) / params.nx)[:, None]
    expected = np.broadcast_to(expected, (params.nx, params.ny))
    np.testing.assert_allclose(np.asarray(w_bc[:, :, -1]), expected, atol=3.0e-7)
    np.testing.assert_allclose(np.asarray(w_bc[:, :, :-1]), 0.0, atol=0.0)
    assert float(jnp.mean(w_bc[:, :, -1])) == pytest.approx(0.0, abs=2.0e-7)


def test_pressure_top_row_satisfies_implicit_klemp_durran_impedance() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax.grid import make_operators
    from wireles_jax.pressure import _radiation_pressure_coefficient, _solve_pressure_hat

    params = radiation_params(jnp)
    ops = make_operators(params)
    rhs_hat = jnp.zeros(
        (params.nx, params.ny // 2 + 1, params.nz),
        dtype=jnp.complex64,
    )
    top_w_hat = jnp.zeros(rhs_hat.shape[:-1], dtype=rhs_hat.dtype).at[1, 0].set(0.7 - 0.2j)
    p_hat = jax.block_until_ready(
        _solve_pressure_hat(rhs_hat, params, ops, top_w_hat=top_w_hat)
    )

    coefficient = _radiation_pressure_coefficient(params, ops)
    half_dz = 0.5 * params.dz
    top_gradient_hat = (
        coefficient * top_w_hat - p_hat[..., -1]
    ) / (half_dz * (1.0 + coefficient * params.dt / half_dz))
    w_new_hat = top_w_hat - params.dt * top_gradient_hat
    boundary_pressure_hat = p_hat[..., -1] + half_dz * top_gradient_hat
    np.testing.assert_allclose(
        np.asarray(boundary_pressure_hat[1, 0]),
        np.asarray(coefficient[1, 0] * w_new_hat[1, 0]),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(np.asarray(p_hat[0, 0, -1]), 0.0, atol=2.0e-7)


def test_rigid_lid_remains_the_default() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params

    params = Params()
    assert params.top_boundary_condition == "rigid_lid"
