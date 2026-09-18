"""Finite-window face-native carrier velocity / parcel-source covariance.

Source-only replays do not mutate the accepted state. Raw combined drag and
vapour-momentum forcing, boundary restraint and source-pressure response are
separate. This measures resolved modulation, not absent subgrid wake energy.
It is an endpoint replay diagnostic, not an exact coupled RK-stage budget.
"""
import argparse
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from audit_spray_radial_momentum import make_sampler
from jaxwind.config.document import load_case
from jaxwind.io.checkpoint import load_checkpoint
from jaxwind.numerics.poisson import build_pressure_poisson, project
from jaxwind.open_boundary import InflowPlane, enforce_open_velocity
from jaxwind.simulation.api import RunControls
from jaxwind.simulation.water_spray_benchmark import build_simulation


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('run',type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--seconds',type=float,default=1.);ap.add_argument('--stride',type=int,default=4)
    args=ap.parse_args();jax.config.update('jax_enable_x64',True)
    assert jax.default_backend()=='gpu',jax.devices()
    args.output.mkdir(parents=True,exist_ok=True)
    case=load_case(args.run/'resolved_case.toml');doc=case.document
    sim=build_simulation(case);g=sim.grid;dt=doc['time']['dt_seconds']
    state,_,_=load_checkpoint(args.run/'checkpoint.npz',sim.initial_state,fingerprint=case.fingerprint)
    replay=make_sampler(g,doc,source_impulse_only=True)[0]
    zero=InflowPlane(jnp.zeros((g.nz,g.ny)),jnp.zeros((g.nz,g.ny+1)),jnp.zeros((g.nz+1,g.ny)),jnp.zeros((g.nz,g.ny)))
    poisson=build_pressure_poisson(g,backend='gmg',periodic_x=False,periodic_y=False,dtype='float64',config={'tolerance':1e-7})
    @jax.jit
    def sample(s):
        raw=replay(s);constrained=enforce_open_velocity(raw,zero,g)
        projected,_=project(constrained,poisson,dt)
        result=[]
        for u,a,b,c in zip(s.velocity,raw,constrained,projected):
            f0,f1,f2=a/dt,(b-a)/dt,(c-b)/dt
            result.append(jnp.stack([u,f0,f1,f2,u*f0,u*f1,u*f2,.5*c*c/dt]))
        return tuple(result)
    add=jax.jit(lambda a,b,w:jax.tree.map(lambda x,y:x+w*y,a,b))
    n=round(args.seconds/dt/args.stride);assert n>=4 and n%4==0
    total=sparse=None;blocks=[];half=None;start=float(state.time);began=time.monotonic()
    for i in range(n+1):
        values=sample(state)
        if total is None:
            total=tuple(jnp.zeros_like(v) for v in values);sparse=total;half=total
        weight=.5 if i in (0,n) else 1.
        total=add(total,values,weight)
        if i%2==0:sparse=add(sparse,values,.5 if i in (0,n) else 1.)
        half=add(half,values,.5 if i in (0,n//2,n) else 1.)
        if i in (n//2,n):
            blocks.append(tuple(np.asarray(v)/(n//2) for v in half))
            half=tuple(.5*v for v in values)
        if i%50==0:
            jax.block_until_ready(values)
            status={'sample':i,'total':n,'time_s':float(state.time),'elapsed_s':time.monotonic()-began,'CFL':float(sim.courant(state))}
            print(json.dumps(status),flush=True)
            (args.output/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
        if i<n:state=sim.advance(state,RunControls(count=args.stride,target_time=float(state.time)+args.stride*dt))
    means=tuple(np.asarray(v)/n for v in total)
    coarse=tuple(np.asarray(v)/(n/2) for v in sparse)
    rho=doc['physics']['moisture']['dry_air_density_kg_m3']
    weights=[];masks=[]
    for component,v in enumerate(means):
        axis=2-component;volume=np.full(v.shape[1:],g.dx*g.dy*g.dz)
        sl=[slice(None)]*3;sl[axis]=0;volume[tuple(sl)]*=.5;sl[axis]=-1;volume[tuple(sl)]*=.5
        x=np.asarray(g.x_faces if component==0 else g.x_centers)[None,None,:]
        y=np.asarray(g.y_faces if component==1 else g.y_centers)[None,:,None]
        z=np.asarray(g.z_faces if component==2 else g.z_centers)[:,None,None]
        r=np.hypot(y-g.ly/2,z-g.lz/2)
        weights.append(volume*rho)
        masks.append({'all':np.ones(v.shape[1:],bool),'near_nozzle':x<.1,'upstream':x<.4,'downstream_core':(x>.4)&(r<.05),'downstream_outer':(x>.4)&(r>=.05)})
    def reduce(moment):
        result={}
        for region in masks[0]:
            result[region]={}
            for j,name in enumerate(['raw_parcel','boundary_restraint','source_pressure']):
                total_work=mean_work=0.
                for v,w,m in zip(moment,weights,masks):
                    total_work+=float(np.sum(v[4+j]*w*m[region]))
                    mean_work+=float(np.sum(v[0]*v[1+j]*w*m[region]))
                result[region][name]={'total_linear_work_W':total_work,'mean_work_W':mean_work,'resolved_covariance_W':total_work-mean_work}
            result[region]['projected_quadratic_impulse_work_W']=sum(float(np.sum(v[7]*w*m[region])) for v,w,m in zip(moment,weights,masks))
        return result
    result={'scope':__doc__,'time_window_s':[start,float(state.time)],'samples':n+1,'sample_spacing_s':dt*args.stride,
            'regions':reduce(means),'coarsened_sampling':reduce(coarse),'half_windows':[reduce(v) for v in blocks],
            'device':str(jax.devices()),'elapsed_s':time.monotonic()-began}
    assert all(np.isfinite(v).all() for v in means)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(args.output/'moments.npz',**{f'mean_{i}':v for i,v in enumerate(means)},**{f'coarse_{i}':v for i,v in enumerate(coarse)})
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
