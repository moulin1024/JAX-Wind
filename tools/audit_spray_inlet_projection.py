"""Frozen source projection sensitivity to transverse end-cell constraints.

This is not a complete boundary implementation: physical inlet advection and
viscous ghost conditions must also change before accepting a coupled run.
"""
import argparse,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg
from audit_spray_radial_momentum import make_sampler
from jaxwind.config.document import load_case
from jaxwind.io.checkpoint import load_checkpoint
from jaxwind.simulation.water_spray_benchmark import build_simulation
from jaxwind.spray_pressure import momentum_pressure_gradient
from jaxwind.numerics.discretization import divergence,cell_velocity
from jaxwind.numerics.poisson import build_pressure_poisson,project
from jaxwind.open_boundary import InflowPlane,enforce_open_velocity
from jaxwind.state import StaggeredVelocity


def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 jax.config.update('jax_enable_x64',True);assert jax.default_backend()=='gpu'
 case=load_case(args.run/'resolved_case.toml');sim=build_simulation(case);g=sim.grid;doc=case.document
 state,_,_=load_checkpoint(args.run/'checkpoint.npz',sim.initial_state,fingerprint=case.fingerprint)
 dt=doc['time']['dt_seconds'];rho=doc['physics']['moisture']['dry_air_density_kg_m3']
 raw=make_sampler(g,doc,source_impulse_only=True)[0](state)
 plane=InflowPlane(jnp.zeros((g.nz,g.ny)),jnp.zeros((g.nz,g.ny+1)),jnp.zeros((g.nz+1,g.ny)),jnp.zeros((g.nz,g.ny)))
 constrained=enforce_open_velocity(raw,plane,g)
 native=build_pressure_poisson(g,backend='gmg',periodic_x=False,periodic_y=False,dtype='float64',config={'tolerance':1e-9})
 reference=jax.jit(lambda v:project(v,native,dt)[0])(constrained)
 x=np.asarray(g.x_centers);y=np.asarray(g.y_centers)-g.ly/2;z=np.asarray(g.z_centers)-g.lz/2
 r=np.hypot(z[:,None],y[None,:]);vol=np.asarray(g.cell_volumes)
 masks={'all':np.ones_like(vol,dtype=bool),'near_nozzle':np.broadcast_to(x<.1,vol.shape),'downstream_core':(r[...,None]<.05)&(x>.4)}
 fields={};report={'scope':__doc__,'mesh':doc['mesh']['cells'],'time_s':float(state.time),'dt_s':dt,'controls':{}}
 refcell=np.asarray(jnp.stack(cell_velocity(reference)))/dt
 for mode in ['legacy','free_inlet_pressure_only','free_inlet','free_both']:
  def gradient(p):
   grad=momentum_pressure_gradient(p,g,periodic_x=False,periodic_y=False,open_x_low=False)
   v,w=grad.y,grad.z
   if mode=='legacy':v=v.at[...,0].set(0);w=w.at[...,0].set(0)
   if mode!='free_both':v=v.at[...,-1].set(0);w=w.at[...,-1].set(0)
   return grad._replace(y=v,z=w)
  v,w=constrained.y,constrained.z
  if mode in ['free_inlet','free_both']:v=v.at[...,0].set(raw.y[...,0]);w=w.at[...,0].set(raw.z[...,0])
  if mode=='free_both':v=v.at[...,-1].set(raw.y[...,-1]);w=w.at[...,-1].set(raw.z[...,-1])
  candidate=constrained._replace(y=v,z=w)
  @jax.jit
  def solve(v):
   rhs=-divergence(v,g)
   potential,_=cg(lambda p:-divergence(gradient(p),g),rhs,tol=1e-11,atol=1e-14,maxiter=2500)
   return StaggeredVelocity(*(a-b for a,b in zip(v,gradient(potential))))
  corrected=solve(candidate)
  residual=float(jnp.max(jnp.abs(divergence(corrected,g))));assert residual<1e-8,(mode,residual)
  cell=np.asarray(jnp.stack(cell_velocity(corrected)))/dt;before=np.asarray(jnp.stack(cell_velocity(candidate)))/dt
  fields[mode+'_acceleration']=cell;fields[mode+'_boundary_acceleration']=before
  centre=cell[0,g.nz//2-1:g.nz//2+1,g.ny//2-1:g.ny//2+1].mean((0,1))
  delta=cell-refcell
  report['controls'][mode]={'max_impulse_divergence_s-1':residual,'max_difference_from_native_acceleration_m_s2':float(abs(delta).max()),
    'regional_axial_force_N':{key:float(np.sum(cell[0]*vol*m)*rho) for key,m in masks.items()},
    'regional_radial_force_N':{key:float(np.sum((cell[1]*y[None,:,None]+cell[2]*z[:,None,None])/np.maximum(r[...,None],1e-30)*vol*m)*rho) for key,m in masks.items()},
    'stations':[{'x_m':s,'centre_acceleration_m_s2':float(np.interp(s,x,centre)),'centre_difference_m_s2':float(np.interp(s,x,delta[0,g.nz//2-1:g.nz//2+1,g.ny//2-1:g.ny//2+1].mean((0,1))))} for s in [.1,.4,.7,1.,1.3,1.6,1.9]]}
  if mode=='legacy':assert abs(delta).max()<1e-5
 args.output.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(args.output/'fields.npz',**fields,x=x,y=y,z=z)
 (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
