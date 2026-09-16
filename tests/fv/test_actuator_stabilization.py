"""Local damping must dissipate energy without biasing mean shear or thrust."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxwind import UniformGrid, StaggeredVelocity
from jaxwind.numerics.actuator_stabilization import actuator_momentum_stabilization, _weighted_fourth_difference


@pytest.mark.parametrize('periodic',[False,True])
def test_weighted_operator_conserves_and_dissipates(periodic):
    q=jax.random.normal(jax.random.PRNGKey(41),(4,7,32))
    weight=jax.random.uniform(jax.random.PRNGKey(42),q.shape)
    result=_weighted_fourth_difference(q,weight,2,periodic)
    np.testing.assert_allclose(result.sum(axis=2),0.,atol=1e-5)
    assert float(jnp.vdot(q,result))<0.


def test_no_mean_shear_change_and_compact_support():
    grid=UniformGrid(64,32,32,1024.,512.,256.)
    position=jnp.asarray([512.,256.,100.])
    u=jnp.broadcast_to(jnp.log(jnp.asarray(grid.z_centers)/.001)[:,None,None],(32,32,65))
    velocity=StaggeredVelocity(u,jnp.zeros((32,33,64)),jnp.zeros((33,32,64)))
    apply=jax.jit(lambda v: actuator_momentum_stabilization(v,grid,position,40.,32.,1/16))
    for value in apply(velocity):
        np.testing.assert_array_equal(value,0.)
    keys=jax.random.split(jax.random.PRNGKey(43),3)
    turbulent=StaggeredVelocity(*(base+jax.random.normal(key,base.shape)*.1 for key,base in zip(keys,velocity)))
    result=apply(turbulent)
    for values,damping in zip(turbulent,result):
        assert float(jnp.vdot(values,damping))<0.
        np.testing.assert_allclose(damping.sum(),0.,atol=1e-5)
        np.testing.assert_array_equal(damping[...,:20],0.)
        np.testing.assert_array_equal(damping[...,45:],0.)
        # Outside rotor support (z < 28 m), x/y-only damping cannot touch the wall.
        np.testing.assert_array_equal(damping[:3],0.)
