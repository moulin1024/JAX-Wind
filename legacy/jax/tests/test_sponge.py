from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


JAX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JAX_ROOT))


def test_plane_mean_sponge_preserves_mean_and_uses_exact_exponential_decay() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.sponge import apply_rayleigh_sponge

    params = Params(
        nx=4,
        ny=3,
        nz=5,
        lz=1.0,
        z_i=100.0,
        dt=0.05,
        zo=0.1,
        momentum_wall_model="free_slip",
        sponge_enabled=True,
        sponge_start_height=50.0,
        sponge_timescale=10.0,
        sponge_power=2.0,
        sponge_target="plane_mean",
        dtype=jnp.float32,
    )
    shape = (params.nx, params.ny, params.nz)
    x_pattern = jnp.asarray([-1.5, -0.5, 0.5, 1.5], dtype=params.dtype)[:, None, None]
    mean_u = jnp.arange(params.nz, dtype=params.dtype)[None, None, :] + 7.0
    mean_v = -0.5 * mean_u
    u = mean_u + x_pattern
    v = mean_v - 2.0 * x_pattern
    w = jnp.full(shape, 3.0, dtype=params.dtype)

    damped_u, damped_v, damped_w = jax.block_until_ready(
        apply_rayleigh_sponge(u, v, w, params)
    )

    u_inner = np.asarray(u)
    v_inner = np.asarray(v)
    damped_u_inner = np.asarray(damped_u)
    damped_v_inner = np.asarray(damped_v)
    np.testing.assert_allclose(damped_u_inner.mean(axis=(0, 1)), u_inner.mean(axis=(0, 1)), atol=5.0e-6)
    np.testing.assert_allclose(damped_v_inner.mean(axis=(0, 1)), v_inner.mean(axis=(0, 1)), atol=5.0e-6)

    center_z = (np.arange(params.nz) + 0.5) * params.dz * params.z_i
    center_eta = np.clip((center_z - 50.0) / 50.0, 0.0, 1.0)
    center_decay = np.exp(-(params.dt_physical / 10.0) * center_eta**2)
    original_u_prime = u_inner - u_inner.mean(axis=(0, 1), keepdims=True)
    damped_u_prime = damped_u_inner - damped_u_inner.mean(axis=(0, 1), keepdims=True)
    np.testing.assert_allclose(damped_u_prime, original_u_prime * center_decay[None, None, :], atol=4.0e-6)

    face_z = (np.arange(params.nz) + 1.0) * params.dz * params.z_i
    face_eta = np.clip((face_z - 50.0) / 50.0, 0.0, 1.0)
    face_decay = np.exp(-(params.dt_physical / 10.0) * face_eta**2)
    np.testing.assert_allclose(
        np.asarray(damped_w),
        np.broadcast_to(3.0 * face_decay[None, None, :], (params.nx, params.ny, params.nz)),
        atol=2.0e-6,
    )


def test_sponge_parameter_validation() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params

    with pytest.raises(ValueError, match="sponge_start_height"):
        Params(
            lz=1.0,
            z_i=100.0,
            zo=0.1,
            sponge_enabled=True,
            sponge_start_height=100.0,
            sponge_timescale=90.0,
            dtype=jnp.float32,
        )
    with pytest.raises(ValueError, match="sponge_timescale"):
        Params(
            lz=1.0,
            z_i=100.0,
            zo=0.1,
            sponge_enabled=True,
            sponge_start_height=75.0,
            sponge_timescale=0.0,
            dtype=jnp.float32,
        )
