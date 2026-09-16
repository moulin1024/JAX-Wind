"""Conservative, energy-dissipating stabilization local to actuator rotors."""
from __future__ import annotations
import jax.numpy as jnp
from jaxwind.state import StaggeredVelocity
from .discretization import _axis_slice


def _weighted_fourth_difference(values, weight, axis, periodic):
    """-D2.T diag(weight) D2: nonpositive work and zero integrated force."""
    if periodic:
        curvature = jnp.roll(values,-1,axis=axis)-2*values+jnp.roll(values,1,axis=axis)
        weighted = weight*curvature
        return -(jnp.roll(weighted,-1,axis=axis)-2*weighted+jnp.roll(weighted,1,axis=axis))
    weighted = jnp.diff(values,n=2,axis=axis)*weight[_axis_slice(3,axis,1,-1)]
    def pad(low, high):
        widths=[(0,0)]*3; widths[axis]=(low,high)
        return jnp.pad(weighted,widths)
    return -pad(0,2)+2*pad(1,1)-pad(2,0)


def _window(distance, inner, outer):
    phase=jnp.clip((jnp.abs(distance)-inner)/(outer-inner),0.,1.)
    return .5*(1+jnp.cos(jnp.pi*phase))


def actuator_momentum_stabilization(velocity, grid, position, radius, normal_width,
                                    coefficient, *, periodic_x=False, periodic_y=False):
    """Damp short horizontal waves near a rotor, without vertical diffusion.

    The nonnegative window is one across the rotor and falls smoothly to zero
    outside it. Applying the window between D2 and its transpose retains the
    conservation/energy property; multiplying an already formed diffusion
    tendency by a mask would not. No stabilization is present without a rotor.
    Width includes two stencil cells when constructing local force patches.
    """
    if not grid.is_uniform:
        raise ValueError('actuator momentum stabilization requires a uniform mesh')
    normal_scale=max(normal_width,grid.dx)
    transverse_scale=max(grid.dy,grid.dz)
    speeds=(jnp.max(jnp.abs(velocity.x)),jnp.max(jnp.abs(velocity.y)))
    results=[]
    for component,values in zip((2,1,0),velocity):
        coordinates=[]
        for axis,dimension,count in ((0,'z',grid.nz),(1,'y',grid.ny),(2,'x',grid.nx)):
            face=axis==component and values.shape[axis]==count+1
            coordinates.append(jnp.asarray(getattr(grid,dimension+('_faces' if face else '_centers')),values.dtype))
        z,y,x=coordinates
        dx=x-position[0]; dy=y-position[1]
        if periodic_x: dx=(dx+.5*grid.lx)%grid.lx-.5*grid.lx
        if periodic_y: dy=(dy+.5*grid.ly)%grid.ly-.5*grid.ly
        radial=jnp.sqrt((z[:,None]-position[2])**2+dy[None,:]**2)
        weight=_window(radial,radius,radius+2*transverse_scale)[:,:,None]*_window(dx,2*normal_scale,4*normal_scale)[None,None,:]
        damp=coefficient*(speeds[0]/grid.dx*_weighted_fourth_difference(values,weight,2,periodic_x)
                          +speeds[1]/grid.dy*_weighted_fourth_difference(values,weight,1,periodic_y))
        results.append(damp)
    return StaggeredVelocity(*results)
