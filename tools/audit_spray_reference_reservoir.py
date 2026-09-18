"""Native droplet thermal replays on identical native trajectories.

Compare 3D gas, reconstructed native profiles, and digitized paper T/Y profiles.
Reference reconstruction is axisymmetric and bounded to x<=.95m; inlet bridging
and upper/lower profiles are explicit controls. No carrier or source update.
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
 ap=argparse.ArgumentParser(__doc__);ap.add_argument('run',type=Path);ap.add_argument('--profiles',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 jax.config.update('jax_enable_x64',True);assert jax.default_backend()=='gpu'
 doc=tomllib.loads((a.run/'resolved_case.toml').read_text());g=UniformGrid(*doc['mesh']['cells'],*doc['mesh']['lengths_m']);cfg=load_moisture(doc['physics'])[0].thermodynamics;prop=WaterDropletProperties();ref=doc['case']['reference'];flow=doc['physics']['water_spray']['mass_flow_rate_kg_s']
 with np.load(a.run/'checkpoint.npz') as f:
  uv=[]
  for c,axis in zip('xyz',[2,1,0]):
   v=f['state/velocity/'+c];uv.append((np.take(v,range(v.shape[axis]-1),axis)+np.take(v,range(1,v.shape[axis]),axis))/2)
  fields=jnp.asarray(np.stack([*uv,f['state/scalar']+doc['physics']['moisture']['temperature_offset_k'],f['state/moisture/vapor']]))
 with np.load(a.profiles) as f:
  native=np.asarray(f['means']).transpose(0,2,1)[:,:4];paper=np.asarray(f['paper_profiles'])[:,:4];zn=np.asarray(f['z_relative_m']);zp=np.asarray(f['paper_z_relative_m']);xs=np.r_[0,f['stations_m'][:4]]
 assert np.isfinite(native[1:]).all() and np.isfinite(paper[1:]).all()
 from jaxwind.simulation.water_spray_benchmark import inlet_mixing_ratio
 qin=float(inlet_mixing_ratio(ref,cfg));Yin=qin/(1+qin);pYin=float(np.median(paper[2,0,(abs(zp)>.20)&(abs(zp)<.26)]))/1000
 def table(values,z,Y0,flat=False):
  out=[]
  for j in [1,2]:
   v=values[j].copy();v=v+273.15 if j==1 else v/1000
   inlet=v[:1] if flat else np.full_like(v[:1],ref['dry_bulb_c']+273.15 if j==1 else Y0)
   order=np.argsort(z);out.append(jnp.asarray(np.concatenate([inlet,v])[:,order]))
  return jnp.asarray(np.sort(z)),out
 snapshot=np.zeros_like(native);host=np.asarray(fields);xc=(np.arange(g.nx)+.5)*g.dx
 for channel,data in [(1,host[3]-273.15),(2,1000*host[4]/(1+host[4]))]:
  centre=data[:,g.ny//2-1:g.ny//2+1,:].mean(1)
  snapshot[channel]=np.array([np.interp(xs[1:],xc,row) for row in centre]).T
 native_table=table(native,zn,Yin);native_flat=table(native,zn,Yin,True);snapshot_table=table(snapshot,zn,Yin);snapshot_flat=table(snapshot,zn,Yin,True);paper_table=table(paper,zp,pYin);paper_flat=table(paper,zp,pYin,True);xp=jnp.asarray(xs)
 profile_controls=[]
 for label,tab in [('native_mean',native_table),('native_mean_flat_inlet',native_flat),('native_snapshot',snapshot_table),('native_snapshot_flat_inlet',snapshot_flat),('paper',paper_table),('paper_flat_inlet',paper_flat)]:
  for branch,side in [('upper',1),('lower',-1)]:profile_controls.append((label+'_'+branch,tab,side))
 names=['native_3d']+[r[0] for r in profile_controls]+['paper_T_only','paper_q_only']
 ds=np.linspace(ref['minimum_diameter_m'],ref['maximum_diameter_m'],20);sw=ds**(ref['rosin_rammler_spread']-1)*np.exp(-(ds/ref['rosin_rammler_scale_m'])**ref['rosin_rammler_spread']);sw/=sw.sum();na=16;weights=np.repeat(sw/na,na);d0=jnp.asarray(np.repeat(ds,na));phi=jnp.tile(jnp.arange(na)*2*jnp.pi/na,20)
 p0=jnp.stack([jnp.zeros_like(phi),g.ly/2+ref['nozzle_diameter_m']/2*jnp.cos(phi),g.lz/2+ref['nozzle_diameter_m']/2*jnp.sin(phi)])
 speed=.9*math.sqrt(2*ref['water_gauge_pressure_pa']/cfg.water_density);angle=math.radians(ref['cone_half_angle_degrees']);v0=speed*jnp.stack([jnp.full_like(phi,math.cos(angle)),math.sin(angle)*jnp.cos(phi),math.sin(angle)*jnp.sin(phi)])
 m0=cfg.water_density*jnp.pi*d0**3/6;T0=ref['water_inlet_c']+273.15;stations=jnp.array([.1,.4,.475,.95]);all_results=[]
 def sample_profiles(p,tab,side):
  z,values=tab;r=jnp.hypot(p[1]-g.ly/2,p[2]-g.lz/2)*side
  idx=jnp.clip(jnp.searchsorted(xp,p[0]),1,len(xs)-1);frac=jnp.clip((p[0]-xp[idx-1])/(xp[idx]-xp[idx-1]),0,1)
  results=[]
  for val in values:
   radial=jax.vmap(lambda v:jnp.interp(r,z,v))(val);left=jnp.take_along_axis(radial,(idx-1)[None],axis=0)[0];right=jnp.take_along_axis(radial,idx[None],axis=0)[0];results.append(left+frac*(right-left))
  return results[0],results[1]/(1-results[1])
 for dt in [6.25e-5,3.125e-5]:
  initial=(p0,v0,jnp.broadcast_to(m0,(len(names),len(d0))),jnp.full((len(names),len(d0)),T0),jnp.zeros((len(names),len(d0))),jnp.full((4,5,len(names),len(d0)),jnp.nan))
  def body(i,state):
   p,v,m,t,Q,records=state;active=p[0]<1.0;gas=_cic_sample_many(fields,_cic_coordinates(*p,g));slip=jnp.linalg.norm(v-gas[:3],axis=0);Ts=[gas[3]];qs=[gas[4]]
   for _,tab,side in profile_controls:
    temp,q=sample_profiles(p,tab,side);Ts.append(temp);qs.append(q)
   jup=names.index('paper_upper');jdown=names.index('paper_lower');paperT=(Ts[jup]+Ts[jdown])/2;paperq=(qs[jup]+qs[jdown])/2;Ts.extend([paperT,gas[3]]);qs.extend([gas[4],paperq]);Ts=jnp.stack(Ts);qs=jnp.stack(qs)
   h=dt*active;d=jnp.cbrt(6*jnp.maximum(m[0],1e-30)/(jnp.pi*cfg.water_density));vn,dp,_=advance_water_droplet_motion(v,gas[:3],d,h,cfg,prop);pn=p+dp
   for axis,extent in [(1,g.ly),(2,g.lz)]:
    hit=(pn[axis]<0)|(pn[axis]>extent);pn=pn.at[axis].set(jnp.clip(pn[axis],0,extent));vn=vn.at[axis].set(jnp.where(hit,0,vn[axis]))
   up=advance_water_droplet(m,t,Ts,qs,slip[None],h[None],cfg,prop);Qn=Q+up.gas_sensible_energy_loss
   for k in range(4):
    cross=active&(p[0]<stations[k])&(pn[0]>=stations[k])&jnp.isnan(records[k,0,0]);f=jnp.clip((stations[k]-p[0])/jnp.maximum(pn[0]-p[0],1e-30),0,1)[None]
    mc=m+f*(up.mass-m);H=m*prop.liquid_heat_capacity*(t-273.15);Hn=up.mass*prop.liquid_heat_capacity*(up.temperature-273.15);Hc=H+f*(Hn-H);Qc=Q+f*(Qn-Q)
    residual=(Qc-Hc+m0*prop.liquid_heat_capacity*(T0-273.15)-cfg.water_vapor_latent_heat*(m0-mc))/m0
    record=jnp.stack([1-mc/m0,Hc/(mc*prop.liquid_heat_capacity),Qc/m0,residual,jnp.broadcast_to((i+f)*dt,mc.shape)])
    records=records.at[k].set(jnp.where(cross[None,None],record,records[k]))
   return pn,vn,up.mass,up.temperature,Qn,records
  raw=np.asarray(jax.jit(lambda:jax.lax.fori_loop(0,round(.5/dt),body,initial))()[-1]);assert np.isfinite(raw).all();assert np.max(abs(raw[:,3]))<1e-5
  rows=[]
  for k,x in enumerate(np.asarray(stations)):
   entries=[]
   for j,name in enumerate(names):entries.append({'name':name,'evaporation_g_s':float(flow*1000*np.sum(weights*raw[k,0,j])),'gas_sensible_W':float(flow*np.sum(weights*raw[k,2,j])),'remaining_liquid_bulk_C':float(np.sum(weights*(1-raw[k,0,j])*raw[k,1,j])/np.sum(weights*(1-raw[k,0,j])))})
   rows.append({'x_m':float(x),'variants':entries})
  all_results.append({'dt_s':dt,'max_energy_identity_J_kg':float(np.max(abs(raw[:,3]))),'stations':rows});a.output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output/f'cohort_dt{dt:g}.npz',records=raw,variants=np.array(names),initial_diameter_m=np.asarray(d0),initial_mass_weights=weights,stations_m=np.asarray(stations));print('completed',dt,json.dumps(rows[1]),flush=True)
 (a.output/'summary.json').write_text(json.dumps({'scope':__doc__,'native_inlet_q':qin,'paper_inferred_inlet_Y':pYin,'results':all_results},indent=2)+'\n')
if __name__=='__main__':main()
