"""Frozen native thermal-source potential response; no accepted physical step."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg
from audit_spray_radial_momentum import make_sampler
from jaxwind.config.document import load_case
from jaxwind.config.moisture import load_moisture
from jaxwind.io.checkpoint import load_checkpoint
from jaxwind.simulation.water_spray_benchmark import build_simulation
from jaxwind.spray_pressure import momentum_pressure_gradient
from jaxwind.numerics.discretization import divergence, cell_velocity
from jaxwind.state import StaggeredVelocity


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    assert jax.default_backend()=='gpu'
    case=load_case(args.run/'resolved_case.toml'); doc=case.document
    sim=build_simulation(case); grid=sim.grid
    state,_,_=load_checkpoint(args.run/'checkpoint.npz',sim.initial_state,fingerprint=case.fingerprint)
    dt=doc['time']['dt_seconds']; moisture=doc['physics']['moisture']
    rho=moisture['dry_air_density_kg_m3']
    T=state.scalar+moisture['temperature_offset_k']; q=state.moisture.vapor
    dT,dq,evap_counter,heat_counter=make_sampler(grid,doc,source_impulse_only=True,return_thermal_increments=True)[0](state)
    cfg=load_moisture(doc["physics"])[0].thermodynamics
    a=cfg.water_vapor_gas_constant/cfg.dry_air_gas_constant
    heat=jnp.log1p(dT/T)/dt; vapor=jnp.log1p(a*dq/(1+a*q))/dt
    target=heat+vapor
    linear=dT/T/dt+a*dq/(1+a*q)/dt
    eos_rho=cfg.pressure/(cfg.dry_air_gas_constant*T*(1+a*q))
    eos_target=linear*rho/eos_rho
    evap_res=float(jnp.sum(dq)*rho*grid.dx*grid.dy*grid.dz-evap_counter)
    heat_res=float(-jnp.sum(dT)*rho*cfg.dry_air_heat_capacity*grid.dx*grid.dy*grid.dz-heat_counter)
    assert abs(evap_res)<1e-12,evap_res
    assert abs(heat_res)<1e-7,heat_res

    def gradient(p,closed):
        g=momentum_pressure_gradient(p,grid,periodic_x=False,periodic_y=False,open_x_low=False)
        return g._replace(x=g.x.at[...,-1].set(0)) if closed else g

    def solve(S,closed):
        mean=jnp.mean(S) if closed else 0.
        rhs=-(S-mean)
        # Gauge term makes the all-Neumann operator positive definite.
        op=lambda p:-divergence(gradient(p,closed),grid)+(jnp.mean(p) if closed else 0.)
        p,_=cg(op,rhs,tol=1e-11,atol=1e-13,maxiter=2000)
        g=gradient(p,closed)
        if closed:
            g=g._replace(x=g.x+mean*jnp.arange(grid.nx+1)[None,None,:]*grid.dx)
        return g
    mixed=jax.jit(lambda S:solve(S,False)); uniform=jax.jit(lambda S:solve(S,True))
    volume=np.asarray(grid.cell_volumes); x=np.asarray(grid.x_centers)
    y=np.asarray(grid.y_centers)-grid.ly/2; z=np.asarray(grid.z_centers)-grid.lz/2
    radius=np.hypot(z[:,None],y[None,:]); mask=radius<.05
    arrays={'heat_s-1':heat,'vapor_s-1':vapor,'total_s-1':target,'linear_s-1':linear}
    report={'scope':'Single frozen native parcel-source replay; excludes diffusion and coupled flow feedback.',
            'mesh':doc['mesh']['cells'],'time_s':float(state.time),'source_dt_s':dt,
            'evaporation_counter_residual_kg':evap_res,'heat_counter_residual_J':heat_res,
            'native_heat_W':float(heat_counter/dt),
            'projection_timestep':'No projection dt enters the potential equation; source dt defines the source rate only.',
            'source_integrals_m3_s':{name:float(np.sum(np.asarray(v)*volume)) for name,v in arrays.items()},
            'native_evaporation_kg_s':float(jnp.sum(dq)*rho*grid.dx*grid.dy*grid.dz/dt),
            'native_temperature_volume_rate_K_m3_s':float(jnp.sum(dT)*grid.dx*grid.dy*grid.dz/dt),
            'controls':{}}
    for name,S,solver in [('mixed_log',target,mixed),('uniform_outlet_log',target,uniform),('mixed_rate',linear,mixed),('mixed_eos_rate',eos_target,mixed),('mixed_half_secant',(jnp.log1p(dT/T/2)+jnp.log1p(a*dq/(1+a*q)/2))/(dt/2),mixed),('source_off',jnp.zeros_like(target),mixed)]:
        g=solver(S); cells=np.asarray(jnp.stack(cell_velocity(g)))
        residual=float(jnp.max(jnp.abs(divergence(g,grid)-S)))
        flux=float(jnp.sum(g.x[...,-1])*grid.dy*grid.dz)
        wanted=float(np.sum(np.asarray(S)*volume))
        assert residual<1e-7,(name,residual)
        assert abs(flux-wanted)<1e-9,(name,flux,wanted)
        assert float(jnp.max(jnp.abs(g.x[...,0])))==0
        assert float(jnp.max(jnp.abs(g.y[:,[0,-1],:])))==0
        assert float(jnp.max(jnp.abs(g.z[[0,-1],:,:])))==0
        stations=[]
        centre=cells[0,grid.nz//2-1:grid.nz//2+1,grid.ny//2-1:grid.ny//2+1].mean((0,1))
        radial=(cells[1]*y[None,:,None]+cells[2]*z[:,None,None])/np.maximum(radius[...,None],1e-30)
        for station in [.1,.4,.7,1.,1.3,1.6,1.9]:
            ix=np.argmin(abs(x-station))
            stations.append({'x_requested_m':station,'x_cell_m':float(x[ix]),'centre_du_m_s':float(np.interp(station,x,centre)),
                'core_mean_du_m_s':float(cells[0,mask,ix].mean()),'core_flux_change_m3_s':float(cells[0,mask,ix].sum()*grid.dy*grid.dz),
                'ring_mean_radial_du_m_s':float(radial[(radius>.04)&(radius<.06),ix].mean())})
        report['controls'][name]={'max_divergence_residual_s-1':residual,'outlet_flux_m3_s':flux,'flux_compatibility_error_m3_s':flux-wanted,'max_abs_du_m_s':float(np.max(abs(cells))),'stations':stations}
        arrays[name+'_velocity_m_s']=cells
    regions={'all':np.ones_like(volume,dtype=bool),'near_nozzle':np.broadcast_to(x<.1,volume.shape),'downstream_core':(radius[...,None]<.05)&(x>.4),'outer':np.broadcast_to(radius[...,None]>=.05,volume.shape)}
    report['regional_source_volume_rates_m3_s']={region:{name:{'signed':float(np.sum(np.asarray(field)*volume*m)),'absolute':float(np.sum(abs(np.asarray(field))*volume*m))} for name,field in [('heat',heat),('vapor',vapor),('total',target)]} for region,m in regions.items()}
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'fields.npz',**{k:np.asarray(v) for k,v in arrays.items()},x=x,y=y,z=z)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
