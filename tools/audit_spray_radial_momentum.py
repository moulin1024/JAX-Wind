"""GPU continuation diagnostics; read-only source replay, no change to solver physics.

Sample Eulerian moments before advancement, with trapezoidal endpoint weights.
Replay the identical injection and four parcel substeps to measure their CIC
source impulses; discard replay state. SGS flux uses native staggered stresses.
The output is a radial transport decomposition, not a closed discrete budget.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind import Boundaries, FREE_SLIP, OPEN, Wall
from jaxwind.config.document import load_case
from jaxwind.cryogenic import _cic_coordinates, _cic_deposit_many, _cic_sample_many
from jaxwind.io.checkpoint import load_checkpoint, save_checkpoint
from jaxwind.moist_abl import MoistAtmosphericSolution
from jaxwind.numerics.discretization import cell_velocity
from jaxwind.physics.moisture import MoistureConfig, WaterDropletProperties, advance_water_droplet
from jaxwind.sgs import (AnisotropicMinimumDissipation, edge_gradients, eddy_viscosity,
                        _to_xy_edge_from_cell, _to_xz_edge_from_cell,
                        _to_cell_from_xy_edge, _to_cell_from_xz_edge)
from jaxwind.simulation.api import RunControls
from jaxwind.simulation.water_spray_benchmark import build_simulation
from jaxwind.water_parcels import WaterParcelSource, inject_water_parcels, exchange_water_parcels
from jaxwind.water_spray import advance_water_droplet_motion

NAMES=['u','v','w','uv','uw','u2','v2','w2','sgs_xy','sgs_xz',
       'molecular_xy','molecular_xz','nu_sgs','speed','T_C','Y_vapor',
       'drag_force_x','evap_momentum_x']


def make_sampler(grid,doc, *, source_impulse_only=False, substep_projector=None, return_source_totals=False, return_thermal_increments=False):
    if not source_impulse_only and (
        doc['case'].get('carrier_turbulence_model', 'les') != 'les'
        or doc['case'].get('carrier_sgs_model', 'amd') != 'amd'
        or doc['case'].get('physical_transverse_inlet', False)
    ):
        raise ValueError('This time-moment sampler implements legacy-inlet AMD only; use the closure-aware saved-state stress diagnostic for RANS')
    ref=doc['case']['reference'];dt=doc['time']['dt_seconds']
    from jaxwind.config.moisture import load_moisture
    cfg=load_moisture(doc['physics'])[0].thermodynamics
    prop=WaterDropletProperties();offset=doc['physics']['moisture']['temperature_offset_k']
    src=WaterParcelSource(center=(0,grid.ly/2,grid.lz/2),radius=ref['nozzle_diameter_m']/2,
        speed=.9*np.sqrt(2*ref['water_gauge_pressure_pa']/cfg.water_density),
        temperature=ref['water_inlet_c']+273.15,mass_flow=doc['physics']['water_spray']['mass_flow_rate_kg_s'],
        half_angle_degrees=ref['cone_half_angle_degrees'],diameter_scale=ref['rosin_rammler_scale_m'],
        diameter_spread=ref['rosin_rammler_spread'],diameter_minimum=ref['minimum_diameter_m'],
        diameter_maximum=ref['maximum_diameter_m'],count_per_step=doc['case']['parcels_per_step'],
        capacity=doc['case']['parcel_capacity'],substeps=doc['case']['parcel_substeps'],
        ramp_time=doc['physics']['water_spray']['ramp_time_s'])
    boundaries=Boundaries(Wall(FREE_SLIP),Wall(FREE_SLIP),streamwise=OPEN,spanwise=FREE_SLIP)
    stations=np.array([.1,.4,.7,1.,1.3,1.6,1.9])
    xc=np.asarray(grid.x_centers);inds=np.clip(np.searchsorted(xc,stations),1,grid.nx-1)
    alpha=np.clip((stations-xc[inds-1])/(xc[inds]-xc[inds-1]),0,1)
    # x=1.9 uses last cell; other profiles linearly interpolate to paper stations.
    def sample_x(a):return a[...,inds-1]*(1-alpha)+a[...,inds]*alpha
    rr=np.hypot(np.asarray(grid.y_centers)[None,:]-grid.ly/2,np.asarray(grid.z_centers)[:,None]-grid.lz/2)
    edges=np.linspace(0,.42,43);ids=jnp.asarray(np.clip(np.digitize(rr.ravel(),edges)-1,0,41))
    volume=grid.dx*grid.dy*grid.dz

    @jax.jit
    def sample(state):
        u,v,w=cell_velocity(state.velocity)
        g=edge_gradients(state.velocity,grid,boundaries)
        nu=eddy_viscosity(state.velocity,grid,boundaries,AnisotropicMinimumDissipation(),gradients=g)
        sy=g['xy']+g['yx'];sz=g['xz']+g['zx']
        xy=_to_cell_from_xy_edge(_to_xy_edge_from_cell(nu,open_x=True,wall_y=True)*sy,open_x=True,wall_y=True)
        xz=_to_cell_from_xz_edge(_to_xz_edge_from_cell(nu,open_x=True)*sz,open_x=True)
        mu=doc['physics']['flow']['kinematic_viscosity_m2_s']
        my=mu*_to_cell_from_xy_edge(sy,open_x=True,wall_y=True)
        mz=mu*_to_cell_from_xz_edge(sz,open_x=True)
        flow=MoistAtmosphericSolution(*state[:8])
        parcels=inject_water_parcels(state.parcels,state.time,state.step,dt,src,cfg)
        force=jnp.zeros((2,grid.nz,grid.ny,grid.nx),dtype=u.dtype)
        expected=jnp.zeros(2,dtype=u.dtype)
        def substep(_,carry):
            flow,p,total,expect=carry
            coords=_cic_coordinates(*p.position,grid)
            fields=jnp.stack((*cell_velocity(flow.velocity),flow.scalar+offset,flow.moisture.vapor))
            sampled=_cic_sample_many(fields,coords);gv,T,q=sampled[:3],sampled[3],sampled[4]
            diameter=jnp.cbrt(6*jnp.maximum(p.mass,1e-30)/(jnp.pi*cfg.water_density))
            h=dt/src.substeps*p.active
            vel,disp,reaction=advance_water_droplet_motion(p.velocity,gv,diameter,h,cfg,prop)
            update=advance_water_droplet(p.mass,p.temperature,T,q,jnp.linalg.norm(p.velocity-gv,axis=0),h,cfg,prop)
            number=p.multiplicity*p.active
            impulse=jnp.stack((p.mass*number*reaction[0],update.evaporated_mass*number*(vel[0]-gv[0])))
            candidate=p.position+disp
            candidate=candidate.at[1].set(jnp.clip(candidate[1],0,grid.ly))
            candidate=candidate.at[2].set(jnp.clip(candidate[2],0,grid.lz))
            midpoint=.5*(p.position+candidate)
            deposited=_cic_deposit_many(impulse,_cic_coordinates(*midpoint,grid),flow.scalar.shape)
            flow,p=exchange_water_parcels(flow,p,grid,dt/src.substeps,cfg,offset)
            if substep_projector is not None:
                flow=flow._replace(velocity=substep_projector(flow.velocity,dt/src.substeps))
            return flow,p,total+deposited,expect+jnp.sum(impulse,axis=1)
        replayed,replayed_parcels,force,expected=jax.lax.fori_loop(0,src.substeps,substep,(flow,parcels,force,expected))
        if return_thermal_increments:
            return (replayed.scalar-state.scalar, replayed.moisture.vapor-state.moisture.vapor,
                    replayed_parcels.evaporated_mass-state.parcels.evaporated_mass,
                    replayed_parcels.gas_sensible_energy_loss-state.parcels.gas_sensible_energy_loss)
        if source_impulse_only:
            delta=jax.tree.map(lambda after,before: after-before, replayed.velocity,state.velocity)
            return (delta,expected/dt) if return_source_totals else delta
        residual=jnp.sum(force,axis=(1,2,3))-expected
        rings=jnp.stack([jax.ops.segment_sum(f.reshape(-1,grid.nx),ids,num_segments=42) for f in force])/dt/grid.dx
        force=force/(dt*volume)
        fields=jnp.stack([sample_x(a) for a in (u,v,w,u*v,u*w,u*u,v*v,w*w,xy,xz,my,mz,nu,
                    jnp.sqrt(u*u+v*v+w*w),state.scalar+offset-273.15,state.moisture.vapor/(1+state.moisture.vapor),force[0],force[1])])
        # Form products AFTER spatial interpolation: otherwise steady streamwise
        # gradients masquerade as temporal Reynolds stresses at off-grid stations.
        fields=fields.at[3].set(fields[0]*fields[1]).at[4].set(fields[0]*fields[2])
        fields=fields.at[5:8].set(fields[:3]**2)
        fields=fields.at[13].set(jnp.sqrt(jnp.sum(fields[:3]**2,axis=0)))
        return fields,rings,residual,expected/dt
    return sample,stations,xc,edges


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seconds',type=float,default=1.);ap.add_argument('--stride',type=int,default=40)
    args=ap.parse_args();assert jax.default_backend()=='gpu','GPU required'
    jax.config.update('jax_enable_x64',True);args.output.mkdir(parents=True,exist_ok=True)
    case=load_case(args.run/'resolved_case.toml');sim=build_simulation(case)
    state,_,header=load_checkpoint(args.run/'checkpoint.npz',sim.initial_state,fingerprint=case.fingerprint)
    sampler,stations,xc,edges=make_sampler(sim.grid,case.document)
    dt=case.document['time']['dt_seconds'];n=round(args.seconds/dt/args.stride)
    assert n>=2 and n%2==0
    start=float(state.time);began=time.monotonic();sums=None;ringsums=None
    times=[];checks=[];totals=[];centre=[];block_fields=[];block_rings=[];prev=None
    for i in range(n+1):
        fields,rings,residual,total=jax.device_get(sampler(state))
        assert np.isfinite(fields).all() and np.isfinite(rings).all()
        assert np.max(np.abs(residual))<1e-10, residual
        if prev is not None:
            if sums is None:sums=np.zeros_like(fields);ringsums=np.zeros_like(rings)
            sums+=(fields+prev[0])*.5;ringsums+=(rings+prev[1])*.5
        if i==n//2 or i==n:
            block_fields.append(sums/(n//2));block_rings.append(ringsums/(n//2));sums=None;ringsums=None
            np.savez_compressed(args.output/'statistics.npz',block_fields=block_fields,block_rings=block_rings,
                names=NAMES,stations_m=stations,x_m=xc,y_m=sim.grid.y_centers,z_m=sim.grid.z_centers,radius_edges_m=edges,
                time=times+[float(state.time)],source_total_N=totals+[total],source_residual_kg_m_s=checks+[residual])
        prev=(fields,rings);times.append(float(state.time));checks.append(residual);totals.append(total)
        if i%10==0:
            status={'sample':i,'samples_target':n,'time_seconds':float(state.time),'elapsed_seconds':time.monotonic()-began,
                'cfl':float(sim.courant(state)),'max_source_deposition_residual':float(np.max(np.abs(checks)))}
            (args.output/'progress.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
        if i<n:state=sim.advance(state,RunControls(count=args.stride,target_time=float(state.time)+args.stride*dt))
    meta={'interval_seconds':[start,float(state.time)],'sample_spacing_seconds':args.stride*dt,'sample_count':n+1,
        'block_intervals_seconds':[[start,start+args.seconds/2],[start+args.seconds/2,start+args.seconds]],
        'devices':[str(d) for d in jax.devices()],'elapsed_seconds':time.monotonic()-began,'rho_kg_m3':case.document['physics']['moisture']['dry_air_density_kg_m3'],
        'notes':['Full pointwise temporal means precede azimuthal averaging.','Positive radial flux is outward axial momentum transport.',
        'SGS stress reconstructed from actual native edge stress and interpolated to cell centres.',
        'Source impulses replay exact source substeps at sampling times, independent of simulation update.',
        'Not a closed discrete momentum budget: axial flux divergence, pressure, storage and numerical flux residual are not balanced here.',
        'Outlet station uses last cell centre. One-second window does not establish statistical convergence.']}
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    save_checkpoint(args.output/'checkpoint.npz',state,metadata={k:v for k,v in header.items() if k not in ('state','observer','schema')})
    print('COMPLETE',flush=True)
    import subprocess,sys
    subprocess.run([sys.executable,str(Path(__file__).with_name('plot_spray_radial_momentum.py')),str(args.output.parent),'--run-directory',str(args.output)],check=True)

if __name__=='__main__':main()
