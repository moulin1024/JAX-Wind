"""Energy, conservation, locality, and projection checks for an open-x buffer."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxwind import UniformGrid, StaggeredVelocity, FlowModel, Boundaries, OPEN, initial_atmospheric_solution, build_open_atmospheric_step, build_pressure_poisson, divergence
from jaxwind.open_boundary import InflowPlane
from jaxwind.numerics.mode_sponge import upstream_mode_sponge_tendency


def fields():
    g=UniformGrid(48,4,8,192.,16.,32.)
    u=jnp.log(jnp.asarray(g.z_centers)/.001)
    v=StaggeredVelocity(jnp.broadcast_to(u[:,None,None],(8,4,49)),jnp.zeros((8,4,48)),jnp.zeros((9,4,48)))
    return g,v


def test_arbitrary_vertical_profile_unchanged():
    g,v=fields()
    for force in upstream_mode_sponge_tendency(g,end_fraction=.5,timescale=1.)(v):
        np.testing.assert_array_equal(force,0.)


def test_energy_conservation_and_nonperiodic_support():
    g,v=fields()
    noise=StaggeredVelocity(*(jax.random.normal(jax.random.key(i),a.shape) for i,a in enumerate(v)))
    force=upstream_mode_sponge_tendency(g,end_fraction=.5,timescale=1.)(noise)
    for q,f in zip(noise,force):
        assert float(jnp.vdot(q,f))<0.
        np.testing.assert_allclose(np.asarray(f).sum(axis=2),0.,atol=4e-7)
        np.testing.assert_array_equal(f[...,:2],0.)
        np.testing.assert_array_equal(f[...,25:],0.)
    # Disturbance beyond the buffer cannot reappear through an opposite edge.
    outside=StaggeredVelocity(*(jnp.zeros_like(q).at[...,-3:].set(1.) for q in v))
    for f in upstream_mode_sponge_tendency(g,end_fraction=.5,timescale=1.)(outside):
        np.testing.assert_array_equal(f,0.)


def test_rk3_projection_retains_inlet_and_wall():
    g,v=fields()
    inflow=InflowPlane(v.x[...,0],v.y[...,0],v.z[...,0],jnp.zeros((8,4)))
    v=v._replace(y=v.y.at[:,:,5:20:2].set(.1))
    solver=build_pressure_poisson(g,backend='gmg',periodic_x=False,dtype='float32',config={'tolerance':1e-6})
    model=FlowModel(momentum_advection_scheme='central',upstream_mode_sponge_end_fraction=.5,upstream_mode_sponge_timescale_seconds=1.)
    step=build_open_atmospheric_step(g,Boundaries(streamwise=OPEN),solver,model,None,scheme='rk3')
    state=initial_atmospheric_solution(g,v,jnp.zeros((8,4,48)),dtype='float32')
    result=jax.jit(step)(state,.1,inflow)
    assert float(jnp.max(jnp.abs(divergence(result.velocity,g))))<1e-5
    np.testing.assert_array_equal(result.velocity.x[...,0],inflow.x_velocity)
    np.testing.assert_array_equal(result.velocity.z[0],0.)
    np.testing.assert_array_equal(result.velocity.z[-1],0.)


@pytest.mark.parametrize('end,tau',[(0.,1.),(1.,1.),(.1,1.),(.5,0.),(.5,float('nan'))])
def test_invalid_buffer(end,tau):
    g,_=fields()
    with pytest.raises(ValueError):upstream_mode_sponge_tendency(g,end_fraction=end,timescale=tau)
