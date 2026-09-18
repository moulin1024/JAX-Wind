"""Read-only spatial drag-power inventory; lost slip energy is not a TKE source.

Uses each saved carrier/parcel state and native CIC gather and drag law.
Gravity, evaporation momentum and pressure work are deliberately separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jaxwind import StaggeredVelocity
from jaxwind.config.moisture import load_moisture
from jaxwind.cryogenic import _cic_coordinates, _cic_sample_many
from jaxwind.domain import UniformGrid
from jaxwind.numerics.discretization import cell_velocity
from jaxwind.physics.moisture import WaterDropletProperties
from jaxwind.water_spray import water_droplet_drag_rate


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    assert jax.default_backend()=='gpu',jax.devices()
    report={'scope':__doc__,'device':str(jax.devices()),'runs':{}}
    for run in args.runs:
        path=run/'checkpoint.npz'
        with np.load(path) as a:
            doc=json.loads(str(a['metadata']))['resolved_case']
            grid=UniformGrid(*doc['mesh']['cells'],*doc['mesh']['lengths_m'])
            config=load_moisture(doc['physics'])[0].thermodynamics
            v=StaggeredVelocity(*(jnp.asarray(a['state/velocity/'+c]) for c in 'xyz'))
            active=a['state/parcels/active']
            x=a['state/parcels/position'][:,active]
            vp=a['state/parcels/velocity'][:,active]
            m=a['state/parcels/mass'][active]
            number=a['state/parcels/multiplicity'][active]
            wall=a['state/parcels/touched_wall'][active]
            time=float(a['state/time'])
        props=WaterDropletProperties()
        ug=np.asarray(_cic_sample_many(jnp.stack(cell_velocity(v)),_cic_coordinates(*jnp.asarray(x),grid)))
        diameter=np.cbrt(6*m/(np.pi*config.water_density))
        slip=vp-ug; speed=np.linalg.norm(slip,axis=0)
        beta=np.asarray(water_droplet_drag_rate(jnp.asarray(diameter),jnp.asarray(speed),config,props))
        force=(m*number*beta)[None,:]*slip
        gas=np.sum(force*ug,axis=0)
        particle=-np.sum(force*vp,axis=0)
        lost=m*number*beta*speed**2
        error=np.max(np.abs(gas+particle+lost))
        assert error < 1e-10 and np.isfinite(lost).all() and np.all(lost>=0)
        re=config.dry_air_density*speed*diameter/props.air_dynamic_viscosity
        radius=np.hypot(x[1]-grid.ly/2,x[2]-grid.lz/2)
        masks={'all':np.ones_like(m,dtype=bool),'x_lt_0.1':x[0]<.1,'x_lt_0.4':x[0]<.4,
               'downstream_core':(x[0]>.4)&(radius<.05),'downstream_outer':(x[0]>.4)&(radius>=.05),
               'wall_touched':wall}
        rows={}
        for region,mask in masks.items():
            rows[region]={'drag_slip_loss_W':float(lost[mask].sum()),
                'carrier_drag_work_W':float(gas[mask].sum()),'particle_drag_work_W':float(particle[mask].sum()),
                'axial_force_N':float(force[0,mask].sum()),'liquid_inventory_kg':float((m*number)[mask].sum())}
        edges=np.array([0,1,10,100,200,500,1000,np.inf])
        re_power=[float(lost[(re>=lo)&(re<hi)].sum()) for lo,hi in zip(edges[:-1],edges[1:])]
        result={'time_s':time,'mesh':doc['mesh']['cells'],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'power_identity_max_error_W':float(error),'regions':rows,
                'reynolds_bin_edges':[0,1,10,100,200,500,1000,'infinity'], 'slip_power_by_reynolds_bin_W':re_power,
                'slip_power_weighted_Re':float(np.sum(lost*re)/lost.sum()),
                'max_diameter_over_min_cell':float(diameter.max()/min(grid.dx,grid.dy,grid.dz)),
                'nozzle_diameter_over_min_cell':doc['case']['reference']['nozzle_diameter_m']/min(grid.dx,grid.dy,grid.dz)}
        report['runs'][run.name]=result
        print(run.name,json.dumps(result),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
