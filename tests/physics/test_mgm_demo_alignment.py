from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from neutral_abl_mgm_jax import clipping_coefficient as demo_clipping_coefficient
from jaxwind.interpreters._jax_zslab_mgm import mgm_clipping_coefficient


def test_production_plane_coefficient_matches_neutral_abl_demo() -> None:
    generator = np.random.default_rng(2026)
    gkk = np.abs(generator.normal(size=(5, 8, 8))).astype(np.float32)
    contraction = generator.normal(size=gkk.shape).astype(np.float32) * gkk
    gkk[0] = 0.0
    gkk[1, :2, :2] = 1.0e-8

    expected = demo_clipping_coefficient(
        jnp.asarray(contraction),
        jnp.asarray(gkk),
    )
    actual = mgm_clipping_coefficient(
        jnp.asarray(contraction),
        jnp.asarray(gkk),
        1.0e-6,
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)
