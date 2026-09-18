"""Prescribed mixing must preserve the conservative stress operator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind import FREE_SLIP, Boundaries, StaggeredVelocity, Wall
from jaxwind.domain import UniformGrid
from jaxwind.sgs import ConstantEddyViscosity, subfilter_tendency

jax.config.update("jax_enable_x64", True)


def test_constant_viscosity_reproduces_discrete_shear_diffusion():
    grid = UniformGrid(8, 16, 4, 2 * np.pi, 2 * np.pi, 1.0)
    x = jnp.broadcast_to(
        jnp.sin(jnp.asarray(grid.y_centers))[None, :, None], (4, 16, 8)
    )
    u = StaggeredVelocity(x, jnp.zeros_like(x), jnp.zeros((5, 16, 8)))
    b = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP))
    model = ConstantEddyViscosity(0.0067)
    tendency, nu = jax.jit(lambda u: subfilter_tendency(u, grid, b, model))(u)
    expected = 0.0067 * (jnp.roll(x, 1, 1) - 2 * x + jnp.roll(x, -1, 1)) / grid.dy**2
    np.testing.assert_allclose(tendency.x, expected, atol=2e-15)
    np.testing.assert_allclose(tendency.y, 0.0, atol=2e-15)
    np.testing.assert_allclose(tendency.z, 0.0, atol=2e-15)
    np.testing.assert_allclose(jnp.sum(tendency.x), 0.0, atol=1e-14)
    assert float(jnp.sum(tendency.x * x)) < 0
    np.testing.assert_allclose(nu, 0.0067)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_prescribed_viscosity_is_rejected(value):
    with pytest.raises(ValueError):
        ConstantEddyViscosity(value)
