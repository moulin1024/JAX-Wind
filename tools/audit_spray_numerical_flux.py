"""Instantaneous native MUSCL-minus-centred radial flux from a saved checkpoint.

No simulation advancement. This isolates spatial numerical transport, not the
complete RK/projection energy budget and not a temporal turbulent covariance.
"""
import argparse,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jaxwind import Boundaries,Wall,FREE_SLIP,OPEN,StaggeredVelocity
from jaxwind.domain import UniformGrid
from jaxwind.numerics.discretization import _centered_cells_to_faces
from jaxwind.numerics.momentum import _muscl_flux,muscl_advection
from jaxwind.sgs import (AnisotropicMinimumDissipation,edge_gradients,eddy_viscosity,
    _to_xy_edge_from_cell,_to_xz_edge_from_cell,_to_cell_from_xy_edge,_to_cell_from_xz_edge)


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    jax.config.update('jax_enable_x64',True)
    data=np.load(args.run/'checkpoint.npz');doc=json.loads(str(data['metadata']))['resolved_case']
    grid=UniformGrid(*doc['mesh']['cells'],*doc['mesh']['lengths_m'])
    velocity=StaggeredVelocity(*(jnp.asarray(data['state/velocity/'+k]) for k in 'xyz'))
    bc=Boundaries(Wall(FREE_SLIP),Wall(FREE_SLIP),streamwise=OPEN,spanwise=FREE_SLIP)
    rho=doc['physics']['moisture']['dry_air_density_kg_m3']
    @jax.jit
    def evaluate(v):
        u=v.x
        my=_centered_cells_to_faces(v.y,2,periodic=False,boundary='copy')
        mz=_centered_cells_to_faces(v.z,2,periodic=False,boundary='copy')
        lx=_muscl_flux(u,.5*(u[...,:-1]+u[...,1:]),2,False,internal=True)
        ly=_muscl_flux(u,my,1,False);lz=_muscl_flux(u,mz,0,False)
        cy=_muscl_flux(u,my,1,False,upwind=False);cz=_muscl_flux(u,mz,0,False,upwind=False)
        # Verify flux assembly against the production axial operator, including endpoints.
        dx=jnp.pad(jnp.diff(lx,axis=2)/grid.dx,((0,0),(0,0),(1,1)))
        assembled=-dx-jnp.diff(ly,axis=1)/grid.dy-jnp.diff(lz,axis=0)/grid.dz
        assembled=assembled.at[...,0].set(0.).at[...,-1].set(0.)
        error=jnp.max(jnp.abs(assembled-muscl_advection(v,grid).x))
        g=edge_gradients(v,grid,bc);nu=eddy_viscosity(v,grid,bc,AnisotropicMinimumDissipation(),gradients=g)
        sy=_to_xy_edge_from_cell(nu,open_x=True,wall_y=True)*(g['xy']+g['yx'])
        sz=_to_xz_edge_from_cell(nu,open_x=True)*(g['xz']+g['zx'])
        cc_y=lambda a:_to_cell_from_xy_edge(a,open_x=True,wall_y=True)
        cc_z=lambda a:_to_cell_from_xz_edge(a,open_x=True)
        return rho*cc_y(ly-cy),rho*cc_z(lz-cz),-rho*cc_y(sy),-rho*cc_z(sz),nu,error
    fy,fz,sy,sz,nu,error=map(np.asarray,evaluate(velocity));assert float(error)<1e-11,error
    yy=np.asarray(grid.y_centers)[None,:,None]-grid.ly/2;zz=np.asarray(grid.z_centers)[:,None,None]-grid.lz/2;r=np.hypot(yy,zz)
    numerical=(fy*yy+fz*zz)/r;modeled=(sy*yy+sz*zz)/r
    stations=np.array([.1,.4,.7,1.,1.3,1.6,1.9]);x=np.asarray(grid.x_centers);edges=np.linspace(0,.28,max(2,int(.28/max(grid.dy,grid.dz,.01)))+1);rc=.5*(edges[:-1]+edges[1:]);radius=r[:,:,0]
    def radial(a):
        prof=np.stack([np.mean(a[(radius>=lo)&(radius<hi)],axis=0) for lo,hi in zip(edges[:-1],edges[1:])])
        return np.stack([np.interp(stations,x,p) for p in prof])
    fn,fs,nut=map(radial,[numerical,modeled,nu]);pick=int(np.argmin(abs(rc-.05)))
    result={'time_seconds':float(data['state/time']),'maximum_native_axial_operator_reconstruction_error_m_s2':float(error),'radius_m':float(rc[pick]),'stations_m':stations.tolist(),
        'numerical_outward_flux_Pa':fn[pick].tolist(),'AMD_outward_flux_Pa':fs[pick].tolist(),'AMD_viscosity_m2_s':nut[pick].tolist(),
        'interpretation':'Instantaneous MUSCL-minus-centred spatial flux, not total numerical dissipation or time-mean covariance. Positive transports axial momentum outward. Native edge fluxes reconstructed to cell centres then annularly averaged.'}
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(args.output/'profiles.npz',radius_m=rc,stations_m=stations,numerical_flux_Pa=fn,AMD_flux_Pa=fs,AMD_viscosity_m2_s=nut)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
