"""Read-only estimate of omitted buoyant TKE production in saved RANS fields.

Uses the existing linear moist Boussinesq law and gradient-diffusion scalar
fluxes. This is an instantaneous constitutive diagnostic, not a coupled run
or evidence of the paper's undocumented density/material settings.
"""
import argparse
import json
from pathlib import Path
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind import FREE_SLIP, OPEN, Boundaries, StaggeredVelocity, Wall
from jaxwind.config.moisture import load_moisture
from jaxwind.domain import UniformGrid
from jaxwind.rans_kepsilon import CMU, KEpsilonState, production
from jaxwind.rans_realizable import turbulent_viscosity


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    assert jax.default_backend() == 'gpu', jax.devices()
    output = {'scope': __doc__, 'device': str(jax.devices()), 'runs': {}}
    for run in args.runs:
        file = run/'checkpoint.npz'
        a = np.load(file)
        doc = json.loads(str(a['metadata']))['resolved_case']
        assert doc['case']['carrier_turbulence_model'] == 'realizable-k-epsilon'
        grid = UniformGrid(*doc['mesh']['cells'], *doc['mesh']['lengths_m'])
        bc = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP)
        velocity = StaggeredVelocity(*(jnp.asarray(a['state/velocity/'+c]) for c in 'xyz'))
        state = KEpsilonState(*(jnp.asarray(a['state/turbulence/'+c]) for c in KEpsilonState._fields))
        nut = np.asarray(turbulent_viscosity(state, velocity, grid, bc))
        shear = np.asarray(production(velocity, grid, bc, jnp.asarray(nut)))
        config = load_moisture(doc['physics'])[0].thermodynamics
        tref = doc['physics']['moisture']['reference_temperature_k']
        epsilon = config.water_vapor_gas_constant/config.dry_air_gas_constant-1
        # Only interior cells enter reported integrals: no extrapolated wall gradient.
        dT = np.gradient(a['state/scalar'], grid.dz, axis=0)
        dq = np.gradient(a['state/moisture/vapor'], grid.dz, axis=0)
        condensate = sum(a['state/moisture/'+c] for c in ['cloud_liquid','cloud_ice','spray_liquid'])
        dc = np.gradient(condensate, grid.dz, axis=0)
        radius = np.hypot(np.asarray(grid.y_centers)[None,:]-grid.ly/2,
                          np.asarray(grid.z_centers)[:,None]-grid.lz/2)[...,None]
        x = np.asarray(grid.x_centers)
        interior = np.zeros_like(shear, dtype=bool); interior[1:-1,1:-1,1:-1] = True
        masks = {'interior':interior, 'near_nozzle':interior & (x<.1),
                 'downstream_core':interior & (x>.4) & (radius<.05),
                 'downstream_outer':interior & (x>.4) & (radius>=.05)}
        dv_rho = grid.dx*grid.dy*grid.dz*config.dry_air_density
        report = {'time_s':float(a['state/time']), 'checkpoint_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                  'closures':{}}
        for name, prandtl, schmidt in [('shared',.7,.7),('fluent_defaults',.85,.7)]:
            heat = -9.81/tref*nut*dT/prandtl
            vapor = -9.81*nut*(epsilon*dq-dc)/schmidt
            buoyant = heat+vapor
            rows = {}
            for region, mask in masks.items():
                ps = float(np.sum(shear[mask])*dv_rho)
                ab = float(np.sum(np.abs(buoyant[mask]))*dv_rho)
                rows[region] = {'shear_production_W':ps,
                    'thermal_buoyancy_W':float(np.sum(heat[mask])*dv_rho),
                    'moisture_buoyancy_W':float(np.sum(vapor[mask])*dv_rho),
                    'net_buoyancy_W':float(np.sum(buoyant[mask])*dv_rho),
                    'absolute_buoyancy_W':ab, 'absolute_buoyancy_over_shear':ab/ps,
                    'max_buoyancy_over_dissipation':float(np.max(np.abs(buoyant[mask])/np.asarray(state.dissipation)[mask]))}
            report['closures'][name] = rows
        k = np.asarray(state.kinetic_energy)
        viscosity = doc['physics']['flow']['kinematic_viscosity_m2_s']
        wall = np.concatenate([CMU**.25*np.sqrt(np.take(k,i,axis=axis)).ravel()*width/2/viscosity
                               for axis,width in [(0,grid.dz),(1,grid.dy)] for i in [0,-1]])
        report['wall_ystar'] = {'min':float(wall.min()), 'max':float(wall.max()),
                               'fraction_below_11_225':float(np.mean(wall<11.225))}
        output['runs'][run.name] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2),flush=True)


if __name__ == '__main__':
    main()
