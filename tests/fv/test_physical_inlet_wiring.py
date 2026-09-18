"""Physical-face inlet must survive enforcement and complete carrier stages."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from jaxwind import (FREE_SLIP, OPEN, Boundaries, FlowModel, InflowPlane,
    StaggeredVelocity, Wall, build_open_atmospheric_step, build_pressure_poisson,
    initial_atmospheric_solution, divergence)
from jaxwind.domain import UniformGrid
from jaxwind.open_boundary import enforce_open_velocity
from jaxwind.sgs import AnisotropicMinimumDissipation


def fixture():
    g=UniformGrid(8,6,4,2.,1.5,1.)
    v=StaggeredVelocity(jnp.full((g.nz,g.ny,g.nx+1),2.),
        jnp.zeros((g.nz,g.ny+1,g.nx)),jnp.zeros((g.nz+1,g.ny,g.nx)))
    p=InflowPlane(*(a[...,0] for a in v),jnp.zeros((g.nz,g.ny)))
    b=Boundaries(Wall(FREE_SLIP),Wall(FREE_SLIP),streamwise=OPEN,spanwise=FREE_SLIP)
    return g,v,p,b


def test_enforcement_preserves_interior_transverse_values_only_when_requested():
    g,v,p,_=fixture()
    v=v._replace(y=v.y.at[:,1:-1,0].set(.3),z=v.z.at[1:-1,:,0].set(-.2))
    physical=enforce_open_velocity(v,p,g,physical_transverse_inlet=True)
    legacy=enforce_open_velocity(v,p,g)
    np.testing.assert_array_equal(physical.y[...,0],v.y[...,0])
    np.testing.assert_array_equal(physical.z[...,0],v.z[...,0])
    np.testing.assert_array_equal(legacy.y[...,0],0.)
    np.testing.assert_array_equal(legacy.z[...,0],0.)
    np.testing.assert_array_equal(physical.x[...,0],p.x_velocity)


def test_physical_inlet_uniform_flow_survives_fast_rk3_stages():
    g,v,p,b=fixture()
    poisson=build_pressure_poisson(g,backend='gmg',periodic_x=False,periodic_y=False,
        dtype='float64',physical_transverse_inlet=True)
    step=build_open_atmospheric_step(g,b,poisson,
        FlowModel(viscosity=1.5e-5,subfilter=AnisotropicMinimumDissipation()),None,scheme='fast-rk3')
    state=initial_atmospheric_solution(g,v,dtype='float64')
    advance=jax.jit(step)
    for _ in range(3):state=advance(state,.001,p)
    for got,expected in zip(state.velocity,v):np.testing.assert_allclose(got,expected,rtol=0,atol=2e-13)
    assert float(jnp.max(jnp.abs(divergence(state.velocity,g))))<1e-12
    assert int(state.step)==3


def test_generic_step_rejects_missing_physical_inlet_flux_data():
    import pytest
    from jaxwind.numerics.integrate import build_step
    g,_,_,b=fixture()
    poisson=build_pressure_poisson(g,backend='gmg',periodic_x=False,periodic_y=False,
        dtype='float64',physical_transverse_inlet=True)
    with pytest.raises(ValueError,match='inlet fluxes'):
        build_step(g,b,poisson,FlowModel())
