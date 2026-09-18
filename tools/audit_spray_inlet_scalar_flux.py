"""Frozen estimate of a Dirichlet diffusive inlet flux; no state is advanced.

Current simulations prescribe incoming advective scalar reservoirs and zero
inlet diffusion. This estimates an alternative with first-cell diffusivity
extrapolated to the boundary, not a demonstrated boundary defect or correction.
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
from jaxwind.rans_kepsilon import CMU,turbulent_viscosity
from jaxwind.rans_realizable import turbulent_viscosity as realizable_viscosity


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('runs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    jax.config.update('jax_enable_x64',True);assert jax.default_backend()=='gpu'
    reports=[]
    for run in args.runs:
        case=load_case(run/'resolved_case.toml');doc=case.document;sim=build_simulation(case);g=sim.grid
        if not g.is_uniform:raise ValueError('This diagnostic requires a uniform grid')
        state,_,_=load_checkpoint(run/'checkpoint.npz',sim.initial_state,fingerprint=case.fingerprint)
        cfg=load_moisture(doc['physics'])[0].thermodynamics
        b=Boundaries(Wall(FREE_SLIP),Wall(FREE_SLIP),streamwise=OPEN,spanwise=FREE_SLIP)
        initial=sim.initial_state;plane=InflowPlane(*(v[...,0] for v in initial.velocity),initial.scalar[...,0])
        model=doc['case'].get('carrier_turbulence_model','les');rans=model!='les';physical=doc['case'].get('physical_transverse_inlet',False)
        inlet=build_benchmark_inlet(g,doc['physics']['flow']['streamwise_velocity_m_s'],plane,None if rans else doc['case'].get('inlet_turbulence'),doc['time']['steps']*doc['time']['dt_seconds'])(state.time)
        if not rans and doc['case'].get('carrier_sgs_model','amd')!='amd':raise ValueError('LES diagnostic requires AMD')
        grad=physical_inlet_gradients(state.velocity,inlet,g,b) if physical else None
        if model=='realizable-k-epsilon':nu=realizable_viscosity(state.turbulence,state.velocity,g,b,gradients=grad)
        elif rans:nu=turbulent_viscosity(state.turbulence)
        elif physical:nu=physical_inlet_eddy_viscosity(state.velocity,inlet,g,b,AnisotropicMinimumDissipation())
        else:nu=eddy_viscosity(state.velocity,g,b,AnisotropicMinimumDissipation())
        if doc['case'].get('carrier_scalar_transport','shared-diffusivity')!='shared-diffusivity':raise ValueError('Audit currently requires shared scalar diffusivity')
        D=cfg.vapor_diffusivity+nu/.7
        fields={'heat_W':(state.scalar,inlet.scalar,D,cfg.dry_air_density*cfg.dry_air_heat_capacity),
                'vapor_kg_s':(state.moisture.vapor,initial.moisture.vapor[...,0],D,cfg.dry_air_density)}
        if rans:
            U=doc['physics']['flow']['streamwise_velocity_m_s'];t=doc['case']['inlet_turbulence'];k0=(U*t['intensity'])**2;eps0=CMU**.75*k0**1.5/t['length_scale_m'];mol=doc['physics']['flow']['kinematic_viscosity_m2_s']
            fields.update({'k_power_W':(state.turbulence.kinetic_energy,k0,mol+nu,cfg.dry_air_density),
                           'epsilon_W_s':(state.turbulence.dissipation,eps0,mol+nu/(1.2 if model=='realizable-k-epsilon' else 1.3),cfg.dry_air_density)})
        radius=np.hypot(np.asarray(g.z_centers)[:,None]-g.lz/2,np.asarray(g.y_centers)[None,:]-g.ly/2)
        report={'run':str(run),'time_s':float(state.time),'scope':__doc__,'physical_transverse_inlet':physical,'fluxes':{}}
        for name,(field,reservoir,diffusivity,factor) in fields.items():
            flux=np.asarray(-diffusivity[...,0]*(field[...,0]-reservoir)/(g.dx/2)*factor*g.dy*g.dz)
            report['fluxes'][name]={'signed_into_domain':float(flux.sum()),'absolute':float(abs(flux).sum()),'core_r_lt_005_signed':float(flux[radius<.05].sum())}
        report['heat_only_inlet_flux_W']=report['fluxes']['heat_W']['signed_into_domain']
        report['interpretation']='Heat-only and vapor-only are separate alternatives. Combined enthalpy below assumes both enabled; Fluent12 documented defaults are energy-on/species-off, with paper overrides unknown.'
        report['potential_dilute_gas_enthalpy_flux_W']=report['fluxes']['heat_W']['signed_into_domain']+cfg.water_vapor_latent_heat*report['fluxes']['vapor_kg_s']['signed_into_domain']
        reports.append(report)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2),flush=True)

if __name__=='__main__':main()
