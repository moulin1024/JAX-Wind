"""Preserve precursor and wall-normal transport while damping open-grid modes."""
import jax
import jax.numpy as jnp
import numpy as np
from jaxwind import UniformGrid, StaggeredVelocity
from jaxwind.numerics.discretization import advection
from jaxwind.numerics.momentum import central_open_upwind_advection, muscl_advection


def test_periodic_turbulent_transport_is_bitwise_central():
    g=UniformGrid(16,8,8,64.,32.,16.)
    keys=jax.random.split(jax.random.PRNGKey(84),3)
    v=StaggeredVelocity(*(jax.random.normal(k,s) for k,s in zip(keys,((8,8,16),(8,8,16),(9,8,16)))))
    for a,b in zip(advection(v,g),central_open_upwind_advection(v,g)):
        np.testing.assert_array_equal(a,b)


def test_all_centered_fluxes_recover_existing_open_operator():
    g=UniformGrid(16,8,8,64.,32.,16.)
    keys=jax.random.split(jax.random.PRNGKey(85),3)
    v=StaggeredVelocity(*(jax.random.normal(k,s) for k,s in zip(keys,((8,8,17),(8,9,16),(9,8,16)))))
    v=v._replace(z=v.z.at[0].set(0.).at[-1].set(0.))
    # x endpoint layers are prescribed/extrapolated by the open integrator.
    for a,b in zip(advection(v,g),muscl_advection(v,g,upwind_axes=())):
        np.testing.assert_allclose(a[...,1:-1],b[...,1:-1],atol=2e-6,rtol=2e-6)


def test_open_log_shear_retains_central_vertical_momentum_flux():
    g=UniformGrid(16,8,16,64.,32.,64.)
    z=jnp.asarray(g.z_centers)
    u=jnp.broadcast_to(jnp.log(z/.001)[:,None,None],(16,8,17))
    v=jnp.broadcast_to((.1*jnp.sin(z))[:,None,None],(16,9,16))
    w=jnp.broadcast_to((.03*jnp.sin(jnp.linspace(0,jnp.pi,17)))[:,None,None],(17,8,16))
    velocity=StaggeredVelocity(u,v,w)
    for a,b in zip(advection(velocity,g),central_open_upwind_advection(velocity,g)):
        np.testing.assert_allclose(a[...,1:-1],b[...,1:-1],atol=2e-6,rtol=2e-6)


def test_open_horizontal_checkerboard_loses_energy():
    g=UniformGrid(32,8,4,32.,8.,4.)
    u=jnp.full((4,8,33),10.)
    ripple=jnp.where(jnp.arange(32)%2==0,.1,-.1)
    v=jnp.broadcast_to(ripple,(4,9,32))
    w=jnp.zeros((5,8,32));velocity=StaggeredVelocity(u,v,w)
    central=advection(velocity,g).y[:,1:-1,2:-2]
    fixed=central_open_upwind_advection(velocity,g).y[:,1:-1,2:-2]
    values=v[:,1:-1,2:-2]
    assert abs(float(jnp.sum(values*central)))<1e-5
    assert float(jnp.sum(values*fixed))< -1.
