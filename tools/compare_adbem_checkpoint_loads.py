"""Compare actual applied AD-BEM loads at saved final states, not lookup power."""
from pathlib import Path
import argparse
import json
import numpy as np


def loads(run):
    import jax
    import jax.numpy as jnp
    from jaxwind.config.document import load_case
    from jaxwind.simulation.wind_farm import ControlledFarm
    from jaxwind.state import StaggeredVelocity
    case=load_case(run/'resolved_case.toml')
    farm=ControlledFarm(case)
    if len(farm.layout)!=1:
        raise ValueError('single-turbine diagnostic only')
    with np.load(run/'checkpoint.npz') as d:
        velocity=StaggeredVelocity(*(jnp.asarray(d['state/velocity/'+k]) for k in ('x','y','z')))
        omega=jnp.asarray(d['state/rotors/omega'])
        time=jnp.asarray(d['state/time'])
    force=jax.jit(farm.force)(velocity,time,omega)
    g=farm.grid;position=farm.layout[0]
    values=[np.asarray(q,dtype=np.float64) for q in force]
    volumes=[]
    for component,q in enumerate(values):
        vol=np.full(q.shape,g.dx*g.dy*g.dz)
        axis=2-component
        low=[slice(None)]*3;high=low.copy();low[axis]=0;high[axis]=-1
        vol[tuple(low)]*=.5;vol[tuple(high)]*=.5
        volumes.append(vol)
    thrust=-float(np.sum(values[0]*volumes[0]))
    torque=-float(np.sum((np.asarray(g.y_centers)-position['y_m'])[None,:,None]*values[2]*volumes[2])
                  -np.sum((np.asarray(g.z_centers)-position['hub_height_m'])[:,None,None]*values[1]*volumes[1]))
    return {'time_seconds':float(time),'thrust_per_density_m4_s2':thrust,
            'torque_per_density_m5_s2':torque,'signed_torque_times_omega_per_density_m5_s3':torque*float(omega[0]),
            'omega_rad_s':float(omega[0]),'probe_wind_m_s':float(farm.sample_wind(velocity)[0]),
            'minimum_normal_smoothing_width_m':case.document['physics']['turbine'].get('minimum_normal_smoothing_width_m',0.),
            'momentum_stabilization_coefficient':case.document['physics']['turbine'].get('momentum_stabilization_coefficient',0.)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result={'baseline':loads(args.baseline),'candidate':loads(args.candidate),
        'scope':'Instantaneous applied force and torque at final checkpoint, per unit air density; not a time mean or independent aerodynamic validation. Torque is signed about +x; torque times prescribed omega is a signed mechanical diagnostic, not electrical power. Lookup electrical power is not used.'}
    result['relative_changes']={k:result['candidate'][k]/result['baseline'][k]-1. for k in ('thrust_per_density_m4_s2','torque_per_density_m5_s2','signed_torque_times_omega_per_density_m5_s3','probe_wind_m_s')}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
