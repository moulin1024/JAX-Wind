"""Frozen full-section vapor flux/source audit on GPU; no accepted state changes.

Native instantaneous transport operator, not an RK-stage or time-averaged
budget. Cumulative source-minus-flux is predicted inventory rate, not a
conservation error or proof of steady state. Axisymmetric profiles are a
separate approximation used solely to assess the Figure 9 estimate.
"""
import argparse,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jaxwind.config.document import load_case
from jaxwind.config.moisture import load_moisture
from jaxwind.io.checkpoint import load_checkpoint
from jaxwind.simulation.water_spray_benchmark import build_simulation,build_benchmark_inlet
from jaxwind.open_boundary import InflowPlane
from jaxwind import Boundaries,Wall,FREE_SLIP,OPEN
from jaxwind.inlet_momentum import physical_inlet_gradients,physical_inlet_eddy_viscosity
from jaxwind.sgs import AnisotropicMinimumDissipation,eddy_viscosity
from jaxwind.rans_realizable import turbulent_viscosity
from jaxwind.scalar import PassiveScalar,open_scalar_tendency
from jaxwind.scalar_transport import _correction
from jaxwind.numerics.discretization import cell_velocity
from audit_spray_radial_momentum import make_sampler


def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('runs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 jax.config.update('jax_enable_x64',True);assert jax.default_backend()=='gpu'
 reports=[]
 for run in a.runs:
  case=load_case(run/'resolved_case.toml');doc=case.document;sim=build_simulation(case);g=sim.grid
  assert g.is_uniform
  state,_,_=load_checkpoint(run/'checkpoint.npz',sim.initial_state,fingerprint=case.fingerprint)
  cfg=load_moisture(doc['physics'])[0].thermodynamics;rho=cfg.dry_air_density
  b=Boundaries(Wall(FREE_SLIP),Wall(FREE_SLIP),streamwise=OPEN,spanwise=FREE_SLIP)
  initial=sim.initial_state;plane=InflowPlane(*(v[...,0] for v in initial.velocity),initial.scalar[...,0])
  model=doc['case'].get('carrier_turbulence_model','les');physical=doc['case'].get('physical_transverse_inlet',False)
  inlet=build_benchmark_inlet(g,doc['physics']['flow']['streamwise_velocity_m_s'],plane,None if model!='les' else doc['case'].get('inlet_turbulence'),doc['time']['steps']*doc['time']['dt_seconds'])(state.time)
  grad=physical_inlet_gradients(state.velocity,inlet,g,b) if physical else None
  if model=='realizable-k-epsilon':nu=turbulent_viscosity(state.turbulence,state.velocity,g,b,gradients=grad)
  elif model=='les':
   assert doc['case'].get('carrier_sgs_model','amd')=='amd'
   nu=physical_inlet_eddy_viscosity(state.velocity,inlet,g,b,AnisotropicMinimumDissipation()) if physical else eddy_viscosity(state.velocity,g,b,AnisotropicMinimumDissipation())
  else:raise ValueError(model)
  assert doc['case'].get('carrier_scalar_transport','shared-diffusivity')=='shared-diffusivity'
  D=cfg.vapor_diffusivity+nu/.7;q=state.moisture.vapor;q0=initial.moisture.vapor[...,0]
  assert float(jnp.max(q0)-jnp.min(q0))<1e-14
  scheme=doc['numerics'].get('scalar_advection_scheme','upwind');assert scheme in ('muscl-mc','upwind','upwind-ssprk3')
  rhsadv=open_scalar_tendency(q,state.velocity,g,PassiveScalar(advection_scheme='upwind'),q0)
  if scheme=='muscl-mc':rhsadv+=_correction(q,state.velocity,g)
  rhsall=open_scalar_tendency(q,state.velocity,g,PassiveScalar(advection_scheme='upwind'),q0,eddy_viscosity=D)
  if scheme=='muscl-mc':rhsall+=_correction(q,state.velocity,g)
  # Sum native conservative tendencies over impermeable y/z sections.
  inletflux=float(jnp.sum(state.velocity.x[...,0]*jnp.where(state.velocity.x[...,0]>=0,q0,q[...,0]))*rho*g.dy*g.dz)
  def recover(rhs,start):return np.r_[start,start-np.cumsum(np.asarray(rhs).sum((0,1))*rho*g.dx*g.dy*g.dz)]
  adv=recover(rhsadv,inletflux);total=recover(rhsall,inletflux);diff=total-adv
  expected=float(jnp.sum(state.velocity.x[...,-1]*jnp.where(state.velocity.x[...,-1]>=0,q[...,-1],q0))*rho*g.dy*g.dz)
  assert abs(total[-1]-expected)<1e-10,(total[-1],expected)
  volume_flow=np.asarray(state.velocity.x).sum((0,1))*g.dy*g.dz
  excess=total-rho*float(q0[0,0])*volume_flow
  dT,dq,evap,heat=make_sampler(g,doc,source_impulse_only=True,return_thermal_increments=True)[0](state)
  dt=doc['time']['dt_seconds'];source=np.r_[0,np.cumsum(np.asarray(dq).sum((0,1))*rho*g.dx*g.dy*g.dz/dt)]
  ledger=source[-1]*dt-float(evap);assert abs(ledger)<1e-12
  xf=np.arange(g.nx+1)*g.dx;xc=np.asarray(g.x_centers);z=np.asarray(g.z_centers)-g.lz/2
  u=np.asarray(cell_velocity(state.velocity)[0]);qq=np.asarray(q)
  stations=[]
  for x in [.1,.4,.7,1.,1.3,1.6,1.9]:
   up=np.array([np.interp(x,xc,row) for row in u[:,g.ny//2-1:g.ny//2+1].mean(1)])
   qp=np.array([np.interp(x,xc,row) for row in qq[:,g.ny//2-1:g.ny//2+1].mean(1)])
   axis=[];r=np.linspace(0,.25,2001)
   for sign in [-1,1]:axis.append(float(rho*2*np.pi*np.trapezoid(np.interp(sign*r,z,up)*(np.interp(sign*r,z,qp)-float(q0[0,0]))*r,r)))
   stations.append({'x_m':x,'native_advective_excess_kg_s':float(np.interp(x,xf,adv-rho*float(q0[0,0])*volume_flow)),'native_diffusive_kg_s':float(np.interp(x,xf,diff)),'native_total_excess_kg_s':float(np.interp(x,xf,excess)),'cumulative_source_kg_s':float(np.interp(x,xf,source)),'predicted_upstream_inventory_rate_kg_s':float(np.interp(x,xf,source-(total-inletflux))),'axisymmetric_vertical_estimate_kg_s':axis})
  reports.append({'run':str(run),'time_s':float(state.time),'scope':__doc__,'outlet_flux_identity_error_kg_s':total[-1]-expected,'source_ledger_error_kg':ledger,'inlet_q':float(q0[0,0]),'stations':stations})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2),flush=True)
if __name__=='__main__':main()
