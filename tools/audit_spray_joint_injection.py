"""Joint size/azimuth quadrature audit at actual parcel rates and indices.

Synthetic injection geometry/transfer moments only, no flow advancement.
Uses the exact production sequence and verifies representative native packets.
"""
import argparse,json,tomllib
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jaxwind.config.moisture import load_moisture
from jaxwind.physics.moisture import WaterDropletProperties,water_droplet_transfer_coefficients
from jaxwind.water_parcels import WaterParcelSource,initial_water_parcels,inject_water_parcels


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('runs',nargs='+',type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();jax.config.update('jax_enable_x64',True);assert jax.default_backend()=='gpu'
    reports={}
    for run in args.runs:
        doc=tomllib.loads((run/'resolved_case.toml').read_text());ref=doc['case']['reference']
        cfg=load_moisture(doc['physics'])[0].thermodynamics;prop=WaterDropletProperties()
        dt=doc['time']['dt_seconds'];count=doc['case']['parcels_per_step'];rate=count/dt
        src=WaterParcelSource(center=(0,.2925,.2925),radius=ref['nozzle_diameter_m']/2,
          speed=.9*np.sqrt(2*ref['water_gauge_pressure_pa']/cfg.water_density),temperature=ref['water_inlet_c']+273.15,
          mass_flow=doc['physics']['water_spray']['mass_flow_rate_kg_s'],half_angle_degrees=ref['cone_half_angle_degrees'],
          diameter_scale=ref['rosin_rammler_scale_m'],diameter_spread=ref['rosin_rammler_spread'],
          diameter_minimum=ref['minimum_diameter_m'],diameter_maximum=ref['maximum_diameter_m'],count_per_step=count,capacity=count)
        def sequence(indices):
            q=np.mod((indices+.5)*.7548776662466927,1)
            a=2*np.pi*np.mod((indices+.5)*.5698402909980532,1)
            lo=np.exp(-(src.diameter_minimum/src.diameter_scale)**src.diameter_spread)
            hi=np.exp(-(src.diameter_maximum/src.diameter_scale)**src.diameter_spread)
            d=src.diameter_scale*(-np.log(lo-q*(lo-hi)))**(1/src.diameter_spread)
            return q,a,d
        for step in [0,round(3/dt),round(4/dt)-1]:
            p=inject_water_parcels(initial_water_parcels(src,'float64'),jnp.asarray(step*dt),jnp.asarray(step),dt,src,cfg)
            q,a,d=sequence(np.arange(step*count,(step+1)*count))
            np.testing.assert_allclose(np.cbrt(6*np.asarray(p.mass)/(np.pi*cfg.water_density)),d,atol=1e-12,rtol=0)
            np.testing.assert_allclose(np.asarray(p.position)[1],src.center[1]+src.radius*np.cos(a),atol=1e-12,rtol=0)
        windows=[]
        for duration in [.005,.02,.1,1.]:
            rows=[]
            for start in np.linspace(3,4-duration,16) if duration<1 else [3.]:
                first=round(start/dt)*count;n=round(duration/dt)*count
                q,a,d=sequence(np.arange(first,first+n))
                slip=np.sqrt(src.speed**2+doc['physics']['flow']['streamwise_velocity_m_s']**2-2*src.speed*doc['physics']['flow']['streamwise_velocity_m_s']*np.cos(np.deg2rad(src.half_angle_degrees)))
                _,nu,sh=water_droplet_transfer_coefficients(jnp.asarray(d),slip,cfg,prop)
                # Initial heat/mass conductance per equal physical-mass parcel.
                # Common saturation/temperature driving factors cancel in angular moments.
                weights={'mass':np.ones_like(d),'heat_conductance':np.asarray(nu)/d**2,'mass_conductance':np.asarray(sh)/d**2}
                angular={key:{str(k):float(abs(np.sum(w*np.exp(1j*k*a))/w.sum())) for k in [1,2,4,8]} for key,w in weights.items()}
                sectors=np.floor(a/(2*np.pi)*36).astype(int)
                means=np.array([np.mean(q[sectors==i]) if np.any(sectors==i) else np.nan for i in range(36)])
                rows.append({'first_index':int(first),'parcels':int(n),'angular_harmonic_relative_amplitudes':angular,
                             'max_sector_mean_mass_quantile_deviation':float(np.nanmax(np.abs(means-.5)))})
            windows.append({'duration_s':duration,'phase_samples':rows,
                'worst_angular_harmonics':{key:{str(k):max(r['angular_harmonic_relative_amplitudes'][key][str(k)] for r in rows) for k in [1,2,4,8]} for key in weights},
                'worst_sector_quantile_deviation':max(r['max_sector_mean_mass_quantile_deviation'] for r in rows)})
        reports[run.name]={'parcel_rate_per_s':rate,'native_packet_checks':'passed','windows':windows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'scope':__doc__,'runs':reports},indent=2)+'\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
