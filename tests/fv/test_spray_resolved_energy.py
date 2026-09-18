"""Resolved shear energy should not be silently assigned to the SGS model."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind import StaggeredVelocity
from jaxwind.domain import UniformGrid
from jaxwind.numerics.discretization import advection, divergence
from jaxwind.numerics.momentum import muscl_advection


def test_resolved_transverse_wave_central_preserves_energy_mc_damps():
    with jax.enable_x64():
        grid = UniformGrid(32, 4, 4, 1.9, 0.585, 0.585)
        # Four cells per wavelength: representative of a marginally resolved
        # inlet fluctuation. This periodic solenoidal shear has no energy flux.
        transverse = jnp.broadcast_to(
            0.2 * jnp.sin(2 * jnp.pi * jnp.arange(32) / 4)[None, None, :],
            (4, 4, 32),
        )
        velocity = StaggeredVelocity(
            jnp.full_like(transverse, 3.0), transverse, jnp.zeros((5, 4, 32))
        )
        np.testing.assert_allclose(divergence(velocity, grid), 0, atol=1e-14)
        central = advection(velocity, grid)
        limited = muscl_advection(velocity, grid)
        central_work = sum(jnp.sum(u * a) for u, a in zip(velocity, central))
        limited_work = sum(jnp.sum(u * a) for u, a in zip(velocity, limited))
        np.testing.assert_allclose(central_work, 0, atol=1e-12)
        assert float(limited_work) < -1.0
        np.testing.assert_allclose(jnp.sum(limited.y), 0, atol=1e-12)
