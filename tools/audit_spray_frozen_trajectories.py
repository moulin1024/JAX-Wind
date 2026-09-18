"""GPU passive droplet trajectories through a saved steady carrier snapshot.

No feedback or carrier advancement. Prescribed diameter/azimuth rays diagnose
residence and thermal histories; they are not a new coupled validation result.
The shared production motion and evaporation updates are used unchanged.
"""
import argparse
import json
import math
import tomllib
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jaxwind.config.moisture import load_moisture
from jaxwind.cryogenic import _cic_coordinates, _cic_sample_many
from jaxwind.domain import UniformGrid
from jaxwind.physics.moisture import WaterDropletProperties, advance_water_droplet
from jaxwind.water_spray import advance_water_droplet_motion


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--save-history',action='store_true',help='Save imposed gas/slip history every 0.5 ms for a separate thermal-law diagnostic')
    args=ap.parse_args()
    assert jax.default_backend()=='gpu'
    jax.config.update('jax_enable_x64',True)
    doc=tomllib.loads((args.run/'resolved_case.toml').read_text())
    grid=UniformGrid(*doc['mesh']['cells'],*doc['mesh']['lengths_m'])
    config=load_moisture(doc['physics'])[0].thermodynamics
    properties=WaterDropletProperties()
    with np.load(args.run/'checkpoint.npz') as a:
        uv=[]
        for component,axis in zip('xyz',[2,1,0]):
            v=a['state/velocity/'+component]
            uv.append((np.take(v,range(v.shape[axis]-1),axis)+np.take(v,range(1,v.shape[axis]),axis))/2)
        fields=jnp.asarray(np.stack([*uv,a['state/scalar']+doc['physics']['moisture']['temperature_offset_k'],a['state/moisture/vapor']]))
    ref=doc['case']['reference']; count=32
    diameters=np.repeat(np.array([100,200,300,400,500])*1e-6,count)
    phi=jnp.tile(jnp.arange(count)*2*jnp.pi/count,5)
    radius=ref['nozzle_diameter_m']/2
    pos=jnp.stack([jnp.zeros_like(phi),grid.ly/2+radius*jnp.cos(phi),grid.lz/2+radius*jnp.sin(phi)])
    angle=math.radians(ref['cone_half_angle_degrees'])
    speed=.9*math.sqrt(2*ref['water_gauge_pressure_pa']/config.water_density)
    vel=speed*jnp.stack([jnp.full_like(phi,math.cos(angle)),math.sin(angle)*jnp.cos(phi),math.sin(angle)*jnp.sin(phi)])
    mass=jnp.asarray(config.water_density*np.pi*diameters**3/6)
    temp=jnp.full_like(phi,ref['water_inlet_c']+273.15)
    stations=jnp.asarray([.25,.5,.75,.99])*grid.lx
    def trace(dt):
        stride=round(.0005/dt)
        history=jnp.full((round(1.0/dt)//stride,7,len(phi)),jnp.nan) if args.save_history else jnp.zeros((0,))
        initial=(pos,vel,mass,temp,jnp.zeros_like(phi,dtype=bool),jnp.full((4,5,len(phi)),jnp.nan),history)
        def body(i,state):
            p,v,m,t,wall,records,history=state
            active=(p[0]<grid.lx)&(p[0]>=0)&(m>0)
            gas=_cic_sample_many(fields,_cic_coordinates(*p,grid))
            if args.save_history:
                sample=jnp.stack([jnp.full_like(t,i*dt),gas[3],gas[4],jnp.linalg.norm(v-gas[:3],axis=0),p[0],m,t])
                history=jax.lax.cond(i%stride==0,lambda h:h.at[i//stride].set(sample),lambda h:h,history)
            h=dt*active
            d=jnp.cbrt(6*jnp.maximum(m,1e-30)/(jnp.pi*config.water_density))
            vn,displacement,_=advance_water_droplet_motion(v,gas[:3],d,h,config,properties)
            update=advance_water_droplet(m,t,gas[3],gas[4],jnp.linalg.norm(v-gas[:3],axis=0),h,config,properties)
            pn=p+displacement
            hit=(pn[1]<0)|(pn[1]>grid.ly)|(pn[2]<0)|(pn[2]>grid.lz)
            wall=wall|hit
            for axis,extent in [(1,grid.ly),(2,grid.lz)]:
                collision=(pn[axis]<0)|(pn[axis]>extent)
                pn=pn.at[axis].set(jnp.clip(pn[axis],0,extent))
                vn=vn.at[axis].set(jnp.where(collision,0,vn[axis]))
            for k in range(4):
                crossing=active&(p[0]<stations[k])&(pn[0]>=stations[k])
                f=jnp.clip((stations[k]-p[0])/jnp.maximum(pn[0]-p[0],1e-30),0,1)
                sample=jnp.stack([(i+f)*dt,t+f*(update.temperature-t)-273.15,v[0]+f*(vn[0]-v[0]),wall.astype(float),gas[4]])
                records=records.at[k].set(jnp.where(crossing[None],sample,records[k]))
            return pn,vn,update.mass,update.temperature,wall,records,history
        return jax.lax.fori_loop(0,round(1.0/dt),body,initial)[-2:]
    results={}
    for dt in [6.25e-5,3.125e-5]:
        raw,history=jax.jit(lambda:trace(dt))()
        values=np.asarray(raw)
        if args.save_history:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.output.parent/f'history_dt_{dt:g}.npz',history=np.asarray(history),initial_diameter_m=diameters,fields=np.array(['time_s','gas_temperature_K','gas_vapor_kg_kg','slip_m_s','x_m','droplet_mass_kg','bulk_temperature_K']))
        np.testing.assert_equal(np.isfinite(values).all(),True)
        rows=[]
        for k,x in enumerate(np.asarray(stations)):
            for index,d in enumerate([100,200,300,400,500]):
                v=values[k,:,index*count:(index+1)*count]
                rows.append(dict(x_m=float(x),initial_diameter_um=d,mean_age_s=float(v[0].mean()),mean_temperature_C=float(v[1].mean()),min_temperature_C=float(v[1].min()),max_temperature_C=float(v[1].max()),mean_axial_velocity_m_s=float(v[2].mean()),wall_fraction=float(v[3].mean()),mean_sampled_vapor_kg_kg=float(v[4].mean())))
        results[str(dt)]=rows
        print('Finished dt',dt,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'run':str(args.run),'scope':__doc__,'results':results},indent=2)+'\n')


if __name__=='__main__':
    main()
