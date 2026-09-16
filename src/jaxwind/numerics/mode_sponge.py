"""Nonperiodic streamwise high-frequency damping in an upstream buffer."""
from __future__ import annotations
import math
import jax.numpy as jnp
from jaxwind.state import StaggeredVelocity


def upstream_mode_sponge_tendency(grid, *, end_fraction: float, timescale: float):
    """Return -D4.T W D4/(256*tau) on raw MAC components.

    D4 is the dimensionless fourth forward difference. All stencils are inside
    the buffer and avoid the first two inlet layers. W is a nonnegative smooth
    window. Thus the added tendency has nonpositive kinetic-energy work, zero
    integrated component force, and annihilates streamwise cubics (including
    arbitrary x-independent wall-normal shear). No Fourier transform, periodic
    extension, actuator sampling change, or vertical diffusion is involved.

    An interior alternating mode has peak decay rate 1/tau; longer wavelengths
    are attenuated in proportion to sin(k*dx/2)**8. This does not guarantee
    turbulent wall-stress preservation or stability of the complete RK solver.
    """
    if not grid.is_uniform:
        raise ValueError('upstream mode sponge requires a uniform grid')
    if not math.isfinite(end_fraction) or not 0. < end_fraction < 1.:
        raise ValueError('upstream mode sponge end_fraction must lie between 0 and 1')
    if not math.isfinite(timescale) or timescale <= 0.:
        raise ValueError('upstream mode sponge timescale must be finite and positive')
    end=end_fraction*grid.lx
    if end <= 8*grid.dx:
        raise ValueError('upstream mode sponge needs more than eight streamwise cells')

    def weights(x):
        x=jnp.asarray(x)
        # A D4 row touches five entries; its complete support stays inside
        # the buffer. Smoothly turn off before both stencil-support endpoints.
        low=x[2]+2*grid.dx
        high=end-2*grid.dx
        phase=(x[2:-2]-low)/(high-low)
        return jnp.where((phase>0)&(phase<1),jnp.sin(jnp.pi*jnp.clip(phase,0.,1.))**2,0.)/(256*timescale)
    face_weight=weights(grid.x_faces)
    center_weight=weights(grid.x_centers)

    def tendency(velocity):
        if velocity.x.shape[-1] != grid.nx+1:
            raise ValueError('upstream mode sponge requires open x')
        result=[]
        for index,q in enumerate(velocity):
            weight=face_weight if index==0 else center_weight
            weighted=jnp.diff(q,n=4,axis=2)*weight.astype(q.dtype)[None,None,:]
            force=jnp.zeros_like(q)
            for offset,coefficient in enumerate((1.,-4.,6.,-4.,1.)):
                force=force-coefficient*jnp.pad(weighted,((0,0),(0,0),(offset,4-offset)))
            result.append(force)
        return StaggeredVelocity(*result)
    return tendency
