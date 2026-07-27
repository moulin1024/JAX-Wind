from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


JAX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JAX_ROOT))


def test_geostrophic_initial_condition_sets_uniform_geostrophic_wind() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params, initial_state

    params = Params(
        nx=4,
        ny=4,
        nz=6,
        lz=1.0,
        z_i=1000.0,
        zo=0.1,
        initial_condition="geostrophic",
        geostrophic_u=8.0,
        geostrophic_v=-1.5,
        initial_velocity_noise=0.0,
        dtype=jnp.float32,
    )
    state = jax.block_until_ready(initial_state(params))

    np.testing.assert_allclose(np.asarray(state.u), 8.0)
    np.testing.assert_allclose(np.asarray(state.v), -1.5)
    np.testing.assert_allclose(np.asarray(state.w), 0.0)
    np.testing.assert_allclose(np.asarray(state.u[:, :, 0]), 8.0)
    np.testing.assert_allclose(np.asarray(state.v[:, :, -1]), -1.5)
