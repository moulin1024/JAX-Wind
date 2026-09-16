"""Conservation, dissipation, selectivity, and wall-shear preservation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxwind import UniformGrid, StaggeredVelocity
from jaxwind.numerics.discretization import advection
from jaxwind.numerics.momentum import _fourth_difference_damping, central_open_filter_advection


@pytest.mark.parametrize('periodic', [True, False])
def test_filter_conserves_sum_and_dissipates_energy(periodic):
    q = jax.random.normal(jax.random.PRNGKey(813), (5, 7, 32))
    d = _fourth_difference_damping(q, 2, periodic)
    np.testing.assert_allclose(d.sum(axis=2), 0., atol=1e-5)
    assert float(jnp.sum(q*d)) < 0.
    if not periodic:
        changed = q.at[..., -1].add(100.)
        np.testing.assert_array_equal(_fourth_difference_damping(changed,2,False)[...,:-3], d[...,:-3])
        linear = jnp.broadcast_to(jnp.arange(32.), q.shape)
        np.testing.assert_array_equal(_fourth_difference_damping(linear,2,False), 0.)


def test_short_waves_damped_more_than_long_waves():
    x = jnp.arange(64.)
    losses = []
    for k in (1, 16):
        q = jnp.cos(2*jnp.pi*k*x/64)
        losses.append(float(-jnp.vdot(q, _fourth_difference_damping(q,0,True))/jnp.vdot(q,q)))
    assert losses[1]/losses[0] > 10000


@pytest.mark.parametrize('periodic', [True, False])
def test_precursor_and_mean_shear_are_unchanged(periodic):
    g = UniformGrid(16,8,16,64.,32.,64.)
    if periodic:
        keys = jax.random.split(jax.random.PRNGKey(24), 3)
        velocity = StaggeredVelocity(*(jax.random.normal(k,s) for k,s in zip(keys,((16,8,16),(16,8,16),(17,8,16)))))
    else:
        u = jnp.broadcast_to(jnp.log(jnp.asarray(g.z_centers)/.001)[:,None,None],(16,8,17))
        v = jnp.zeros((16,8,16))
        w = jnp.broadcast_to(jnp.sin(jnp.linspace(0,jnp.pi,17))[:,None,None],(17,8,16))
        velocity = StaggeredVelocity(u,v,w)
    for before, after in zip(advection(velocity,g), central_open_filter_advection(velocity,g)):
        np.testing.assert_array_equal(before, after)
