"""Lateral outlet pressure lifting and uniform-throughflow preservation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxwind import (UniformGrid, StaggeredVelocity, Boundaries, OPEN, FlowModel,
    InflowPlane, build_pressure_poisson, pressure_gradient, project, divergence,
    initial_atmospheric_solution, build_open_atmospheric_step)
from jaxwind.open_boundary import backflow_lateral_pressures

@pytest.fixture(autouse=True)
def precision():
    old = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', old)


def test_lateral_pressure_manufactured_projection_with_corner_constraints():
    g=UniformGrid(8,6,4,16.,12.,4.)
    shape=(g.nz,g.ny,g.nx)
    p=jax.random.normal(jax.random.PRNGKey(1),shape)
    lo=jax.random.normal(jax.random.PRNGKey(2),(g.nz,g.nx)).at[:,0].set(0.).at[:,-1].set(0.)
    hi=jax.random.normal(jax.random.PRNGKey(3),(g.nz,g.nx)).at[:,0].set(0.).at[:,-1].set(0.)
    px=jax.random.normal(jax.random.PRNGKey(4),(g.nz,g.ny))
    gradient=pressure_gradient(p,g,periodic_x=False,periodic_y=False,open_y=True)
    gradient=gradient._replace(x=gradient.x.at[...,-1].add(px/(.5*g.dx)),
        y=gradient.y.at[:,0].add(-lo/(.5*g.dy)).at[:,-1].add(hi/(.5*g.dy)))
    base=StaggeredVelocity(jnp.full((g.nz,g.ny,g.nx+1),10.),jnp.zeros((g.nz,g.ny+1,g.nx)),jnp.zeros((g.nz+1,g.ny,g.nx)))
    candidate=StaggeredVelocity(*(a+.1*b for a,b in zip(base,gradient)))
    solver=build_pressure_poisson(g,backend='gmg',periodic_x=False,periodic_y=False,open_y=True,dtype='float64')
    final,observed=project(candidate,solver,.1,outlet_pressure=px,lateral_pressures=(lo,hi))
    for a,b in zip(final,base): np.testing.assert_allclose(a,b,atol=2e-8)
    np.testing.assert_allclose(observed,p,atol=2e-7)
    assert float(jnp.max(jnp.abs(divergence(final,g))))<1e-8


def test_lateral_pressure_dissipates_perturbation_energy():
    g=UniformGrid(8,6,4,16.,12.,4.)
    u=jnp.full((4,6,9),9.)
    v=jnp.zeros((4,7,8)).at[:,0].set(jnp.linspace(-1.,1.,8)).at[:,-1].set(jnp.linspace(1.,-1.,8))
    w=jnp.zeros((5,6,8))
    lo,hi=backflow_lateral_pressures(StaggeredVelocity(u,v,w),g,10.)
    for p,normal in ((lo,-v[:,0]),(hi,v[:,-1])):
        power=-(p+.5*(1.+normal**2))*normal
        assert bool(jnp.all(power<=0.))
        assert bool(jnp.all(p[normal>=0.] == 0.))


def test_energy_lateral_outlets_preserve_uniform_tangential_flow():
    g=UniformGrid(8,6,4,16.,12.,4.)
    v=StaggeredVelocity(jnp.full((4,6,9),10.),jnp.zeros((4,7,8)),jnp.zeros((5,6,8)))
    inlet=InflowPlane(*(f[...,0] for f in v),jnp.zeros((4,6)))
    poisson=build_pressure_poisson(g,backend='gmg',periodic_x=False,periodic_y=False,open_y=True,dtype='float64')
    step=build_open_atmospheric_step(g,Boundaries(streamwise=OPEN,spanwise=OPEN),poisson,
        FlowModel(momentum_advection_scheme='central',outlet_backflow='energy'),None,scheme='rk3')
    state=initial_atmospheric_solution(g,v,dtype='float64')
    final=jax.jit(lambda s:jax.lax.scan(lambda s,_:(step(s,.1,inlet),None),s,None,length=10)[0])(state)
    for a,b in zip(final.velocity,v):np.testing.assert_allclose(a,b,atol=1e-12)
    assert float(jnp.max(jnp.abs(divergence(final.velocity,g))))<1e-12
