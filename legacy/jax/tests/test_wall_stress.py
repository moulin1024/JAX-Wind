from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


JAX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JAX_ROOT))


def test_dynamic_neutral_wall_stress_uses_local_log_law_speed() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.wall import wall_stress

    params = Params(
        nx=8,
        ny=8,
        nz=5,
        lz=1.0,
        z_i=1000.0,
        u_fric=0.0,
        zo=0.1,
        wall_stress_model="dynamic_neutral",
        dtype=jnp.float32,
    )
    u = jnp.zeros((params.nx, params.ny, params.nz), dtype=params.dtype).at[:, :, 0].set(10.0)
    v = jnp.zeros_like(u)

    txz0, tyz0, dudz0, dvdz0, ustar = jax.block_until_ready(wall_stress(u, v, params))
    expected_ustar = params.vonk * 10.0 / np.log(params.wall_ref_height / params.zo)
    np.testing.assert_allclose(np.asarray(ustar), expected_ustar, rtol=2.0e-6)
    np.testing.assert_allclose(np.asarray(txz0), -(expected_ustar**2), rtol=2.0e-6)
    np.testing.assert_allclose(np.asarray(tyz0), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(dudz0), expected_ustar / (params.vonk * 0.5 * params.dz), rtol=2.0e-6)
    np.testing.assert_allclose(np.asarray(dvdz0), 0.0, atol=1.0e-7)


def test_prescribed_wall_stress_keeps_ustar_magnitude() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.wall import wall_stress

    params = Params(
        nx=8,
        ny=8,
        nz=5,
        lz=1.0,
        z_i=1000.0,
        u_fric=0.5,
        zo=0.1,
        wall_stress_model="prescribed_ustar",
        dtype=jnp.float32,
    )
    u = jnp.zeros((params.nx, params.ny, params.nz), dtype=params.dtype).at[:, :, 0].set(10.0)
    v = jnp.zeros_like(u)

    txz0, tyz0, _, _, ustar = jax.block_until_ready(wall_stress(u, v, params))
    np.testing.assert_allclose(np.asarray(ustar), 0.5, rtol=2.0e-6)
    np.testing.assert_allclose(np.asarray(txz0), -0.25, rtol=2.0e-6)
    np.testing.assert_allclose(np.asarray(tyz0), 0.0, atol=1.0e-7)
