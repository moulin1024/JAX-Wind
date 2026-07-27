from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


JAX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JAX_ROOT))


def test_coriolis_geostrophic_rhs_signs() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.rhs import add_coriolis_geostrophic_forcing

    params = Params(
        nx=2,
        ny=2,
        nz=3,
        lz=1.0,
        z_i=1000.0,
        coriolis_f=1.0e-4,
        geostrophic_u=10.0,
        geostrophic_v=-2.0,
        dtype=jnp.float32,
    )
    rhs_u = jnp.zeros((params.nx, params.ny, params.nz), dtype=params.dtype)
    rhs_v = jnp.zeros_like(rhs_u)
    u = jnp.ones_like(rhs_u) * 7.0
    v = jnp.ones_like(rhs_u) * 1.0

    rhs_u, rhs_v = jax.block_until_ready(add_coriolis_geostrophic_forcing(rhs_u, rhs_v, u, v, params))
    np.testing.assert_allclose(np.asarray(rhs_u), 0.3)
    np.testing.assert_allclose(np.asarray(rhs_v), 0.3)
