"""Downstream damping preserves mean shear and remains pressure projected."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxwind import (UniformGrid, StaggeredVelocity, FlowModel, Boundaries, OPEN,
                     initial_atmospheric_solution, build_open_atmospheric_step,
                     build_pressure_poisson, divergence)
from jaxwind.open_boundary import InflowPlane
from jaxwind.sponge import outlet_sponge_tendency


def state_and_inflow():
    grid=UniformGrid(16,8,8,64.,32.,32.)
    u=jnp.log(jnp.asarray(grid.z_centers)/.001)
    velocity=StaggeredVelocity(jnp.broadcast_to(u[:,None,None],(8,8,17)),jnp.zeros((8,8,16)),jnp.zeros((9,8,16)))
    inflow=InflowPlane(velocity.x[...,0],velocity.y[...,0],velocity.z[...,0],jnp.zeros((8,8)))
    return grid,velocity,inflow


def test_log_profile_is_equilibrium_to_float32_roundoff():
    grid,v,inflow=state_and_inflow()
    sponge=outlet_sponge_tendency(grid,start_fraction=.75,timescale=5.)
    for force in sponge(v,inflow):
        np.testing.assert_allclose(force,0.,atol=3e-7)


def test_damping_is_local_and_removes_perturbation_energy():
    grid,v,inflow=state_and_inflow()
    perturbation=StaggeredVelocity(*(jnp.ones_like(a)*.1 for a in v))
    changed=StaggeredVelocity(*(a+b for a,b in zip(v,perturbation)))
    force=outlet_sponge_tendency(grid,start_fraction=.75,timescale=5.)(changed,inflow)
    for f,p in zip(force,perturbation):
        np.testing.assert_array_equal(f[...,:12],0.)
        assert float(jnp.vdot(f,p))<0.
    np.testing.assert_allclose(force.x[...,-1],-.1/5,atol=3e-7)


def test_sponge_step_is_projected_and_keeps_inlet():
    grid,v,inflow=state_and_inflow()
    v=v._replace(y=v.y.at[:,:,12:].set(.05))
    solver=build_pressure_poisson(grid,backend='gmg',periodic_x=False,dtype='float32',config={'tolerance':1e-6})
    model=FlowModel(momentum_advection_scheme='central',outlet_sponge_start_fraction=.75,outlet_sponge_timescale_seconds=5.)
    step=build_open_atmospheric_step(grid,Boundaries(streamwise=OPEN),solver,model,None,scheme='rk3')
    state=initial_atmospheric_solution(grid,v,jnp.zeros((8,8,16)),dtype='float32')
    out=jax.jit(step)(state,.1,inflow)
    assert float(jnp.max(jnp.abs(divergence(out.velocity,grid))))<1e-5
    np.testing.assert_array_equal(out.velocity.x[...,0],inflow.x_velocity)
    np.testing.assert_array_equal(out.velocity.z[0],0.)
    np.testing.assert_array_equal(out.velocity.z[-1],0.)


@pytest.mark.parametrize('start,tau',[(1.,5.),(0.,5.),(.75,0.),(.75,float('nan'))])
def test_invalid_zone_parameters(start,tau):
    grid,_,_=state_and_inflow()
    with pytest.raises(ValueError):
        outlet_sponge_tendency(grid,start_fraction=start,timescale=tau)
