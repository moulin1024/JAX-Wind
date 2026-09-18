"""GPU passive mass-quantile cohorts in an archived carrier field.

Unchanged native drag/thermal laws, no feedback or accepted-state advancement.
First forward crossings retain mass, sensible heat and thermal/residence state.
A frozen-field diagnostic, not a coupled solution or an experimental fit.
"""
import argparse,json,math,tomllib
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jaxwind.config.moisture import load_moisture
from jaxwind.cryogenic import _cic_coordinates,_cic_sample_many
from jaxwind.domain import UniformGrid
from jaxwind.physics.moisture import WaterDropletProperties,advance_water_droplet
from jaxwind.water_spray import advance_water_droplet_motion


def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--carrier-fields',type=Path,help='Diagnostic cell-centred [u,v,w,T,q] override, not a carrier solution');p.add_argument('--diameter-rule',choices=['mass-quantiles','linear-point-density'],default='mass-quantiles');a=p.parse_args()
 jax.config.update('jax_enable_x64',True);assert jax.default_backend()=='gpu'
 doc=tomllib.loads((a.run/'resolved_case.toml').read_text());g=UniformGrid(*doc['mesh']['cells'],*doc['mesh']['lengths_m'])
 cfg=load_moisture(doc['physics'])[0].thermodynamics;prop=WaterDropletProperties();ref=doc['case']['reference']
 with np.load(a.run/'checkpoint.npz') as f:
  uv=[]
  for c,axis in zip('xyz',[2,1,0]):
   v=f['state/velocity/'+c];uv.append((np.take(v,range(v.shape[axis]-1),axis)+np.take(v,range(1,v.shape[axis]),axis))/2)
  fields=jnp.asarray(np.stack([*uv,f['state/scalar']+doc['physics']['moisture']['temperature_offset_k'],f['state/moisture/vapor']]))
 if a.carrier_fields is not None:
  with np.load(a.carrier_fields) as override:replacement=np.asarray(override['fields'])
  assert replacement.shape==fields.shape and np.isfinite(replacement).all()
  fields=jnp.asarray(replacement)
 stations=jnp.array([.05,.1,.19,.38,.4,.475,.57,.76,.95]);flow=doc['physics']['water_spray']['mass_flow_rate_kg_s']
 speed=.9*math.sqrt(2*ref['water_gauge_pressure_pa']/cfg.water_density);angle=math.radians(ref['cone_half_angle_degrees'])
 results=[];a.output.mkdir(parents=True,exist_ok=True)
 controls=[(32,6.25e-5),(64,6.25e-5),(64,3.125e-5)] if a.diameter_rule=='mass-quantiles' else [(20,6.25e-5),(40,6.25e-5),(20,3.125e-5)]
 for nq,dt in controls:
  na=16;lo=np.exp(-(ref['minimum_diameter_m']/ref['rosin_rammler_scale_m'])**ref['rosin_rammler_spread']);hi=np.exp(-(ref['maximum_diameter_m']/ref['rosin_rammler_scale_m'])**ref['rosin_rammler_spread'])
  ds=ref['rosin_rammler_scale_m']*(-np.log(lo-(np.arange(nq)+.5)/nq*(lo-hi)))**(1/ref['rosin_rammler_spread'])
  if a.diameter_rule=='linear-point-density':
   ds=np.linspace(ref['minimum_diameter_m'],ref['maximum_diameter_m'],nq)
   size_weights=ds**(ref['rosin_rammler_spread']-1)*np.exp(-(ds/ref['rosin_rammler_scale_m'])**ref['rosin_rammler_spread']);size_weights/=size_weights.sum()
  else:size_weights=np.ones(nq)/nq
  weights=np.repeat(size_weights/na,na);assert np.isclose(weights.sum(),1)
  d0=jnp.asarray(np.repeat(ds,na));phi=jnp.tile(jnp.arange(na)*2*jnp.pi/na,nq)
  pos=jnp.stack([jnp.zeros_like(phi),g.ly/2+ref['nozzle_diameter_m']/2*jnp.cos(phi),g.lz/2+ref['nozzle_diameter_m']/2*jnp.sin(phi)])
  vel=speed*jnp.stack([jnp.full_like(phi,math.cos(angle)),math.sin(angle)*jnp.cos(phi),math.sin(angle)*jnp.sin(phi)])
  m0=cfg.water_density*jnp.pi*d0**3/6;t0=jnp.full_like(phi,ref['water_inlet_c']+273.15)
  names=['age_s','mass_loss_fraction','temperature_C','axial_velocity_m_s','slip_m_s','gas_temperature_C','gas_q','radius_m','wall_touched','gas_sensible_J_per_kg_birth','energy_identity_J_per_kg_birth']
  records=jnp.full((len(stations),len(names),len(phi)),jnp.nan)
  initial=(pos,vel,m0,t0,jnp.zeros_like(phi),jnp.zeros_like(phi,dtype=bool),records)
  def body(i,s):
   p,v,m,t,Q,wall,records=s;active=(p[0]<g.lx)&(m>0);gas=_cic_sample_many(fields,_cic_coordinates(*p,g));slip=jnp.linalg.norm(v-gas[:3],axis=0)
   h=dt*active;d=jnp.cbrt(6*jnp.maximum(m,1e-30)/(jnp.pi*cfg.water_density))
   vn,displacement,_=advance_water_droplet_motion(v,gas[:3],d,h,cfg,prop)
   up=advance_water_droplet(m,t,gas[3],gas[4],slip,h,cfg,prop);pn=p+displacement;Qn=Q+up.gas_sensible_energy_loss
   for axis,extent in [(1,g.ly),(2,g.lz)]:
    hit=(pn[axis]<0)|(pn[axis]>extent);wall=wall|hit;pn=pn.at[axis].set(jnp.clip(pn[axis],0,extent));vn=vn.at[axis].set(jnp.where(hit,0,vn[axis]))
   for k in range(len(stations)):
    cross=active&(p[0]<stations[k])&(pn[0]>=stations[k])&jnp.isnan(records[k,0])
    f=jnp.clip((stations[k]-p[0])/jnp.maximum(pn[0]-p[0],1e-30),0,1)
    # Interpolate conserved liquid enthalpy, not T, for crossing energy ledger.
    mass=m+f*(up.mass-m);H=m*prop.liquid_heat_capacity*(t-273.15);Hn=up.mass*prop.liquid_heat_capacity*(up.temperature-273.15);Hc=H+f*(Hn-H);Tc=Hc/(mass*prop.liquid_heat_capacity)
    Qc=Q+f*(Qn-Q);loss=1-mass/m0;pc=p+f[None]*(pn-p)
    residual=(Qc-Hc+m0*prop.liquid_heat_capacity*(t0-273.15)-cfg.water_vapor_latent_heat*(m0-mass))/m0
    sample=jnp.stack([(i+f)*dt,loss,Tc,v[0]+f*(vn[0]-v[0]),slip,gas[3]-273.15,gas[4],jnp.hypot(pc[1]-g.ly/2,pc[2]-g.lz/2),wall.astype(float),Qc/m0,residual])
    records=records.at[k].set(jnp.where(cross[None],sample,records[k]))
   return pn,vn,up.mass,up.temperature,Qn,wall,records
  out=jax.jit(lambda:jax.lax.fori_loop(0,round(.5/dt),body,initial))()
  raw=np.asarray(out[-1]);assert np.isfinite(raw).all(),'A cohort failed to cross within .5s'
  err=float(np.max(abs(raw[:,names.index('energy_identity_J_per_kg_birth')])));assert err<1e-5,err
  rows=[]
  for k,x in enumerate(np.asarray(stations)):
   row={'x_m':float(x),'evaporation_kg_s':float(flow*np.sum(weights*raw[k,1])),'gas_sensible_W':float(flow*np.sum(weights*raw[k,9])),'mean_age_s':float(np.sum(weights*raw[k,0])),'size_bins':[]}
   for low,high in [(74,150),(150,250),(250,350),(350,450),(450,519)]:
    mask=(np.asarray(d0)*1e6>=low)&(np.asarray(d0)*1e6<high)
    if not mask.any():continue
    entry={'birth_diameter_um':[low,high],'birth_mass_fraction':float(weights[mask].sum()),'evaporation_kg_s':float(flow*np.sum(weights[mask]*raw[k,1,mask]))}
    entry.update({name:float(np.average(raw[k,j,mask],weights=weights[mask])) for j,name in enumerate(names)})
    row['size_bins'].append(entry)
   rows.append(row)
  label=f'q{nq}_dt{dt:g}';np.savez_compressed(a.output/(label+'.npz'),records=raw,fields=np.array(names),initial_diameter_m=np.asarray(d0),initial_mass_weights=weights,stations_m=np.asarray(stations))
  results.append({'mass_quantiles':nq,'diameter_rule':a.diameter_rule,'initial_D10_um':float(np.sum(weights/np.asarray(d0)**2)/np.sum(weights/np.asarray(d0)**3)*1e6),'initial_D32_um':float(1/np.sum(weights/np.asarray(d0))*1e6),'azimuths':na,'dt_s':dt,'max_crossing_energy_identity_J_per_kg_birth':err,'stations':rows})
  print('Completed',label,'upstream evaporation g/s',[1000*r['evaporation_kg_s'] for r in rows],flush=True)
 report={'scope':__doc__,'carrier_fields_override':str(a.carrier_fields) if a.carrier_fields else None,'run':str(a.run),'injected_water_kg_s':flow,'sampling':a.diameter_rule+' with uniform azimuth. Linear point-density is a diagnostic discretization, not established paper stream weighting.','results':results}
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
