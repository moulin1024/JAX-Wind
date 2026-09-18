"""Separate mean and fluctuating finite-impulse work in saved source moments.

The input stores E[u], E[f_i], E[u*f_i], and dt/2 E[(sum f_i)^2]
on native faces. No new simulation is run. Half-window moments were not saved.
"""
import argparse
import json
from pathlib import Path
import tomllib
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    d = json.loads((args.results/'summary.json').read_text())
    arrays = np.load(args.results/'moments.npz')
    case = tomllib.loads(args.case.read_text())
    nx, ny, nz = case['mesh']['cells']
    lx, ly, lz = case['mesh']['lengths_m']
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    dt = case['time']['dt_seconds']
    rho = case['physics']['moisture']['dry_air_density_kg_m3']
    for prefix, target in [('mean', 'regions'), ('coarse', 'coarsened_sampling')]:
        for region, row in d[target].items():
            mean_quadratic = 0.
            for component in range(3):
                v = arrays[f'{prefix}_{component}']
                volume = np.full(v.shape[1:], dx*dy*dz*rho)
                boundary = [slice(None)]*3
                boundary[2-component] = 0
                volume[tuple(boundary)] *= .5
                boundary[2-component] = -1
                volume[tuple(boundary)] *= .5
                np.testing.assert_allclose(volume.sum(), lx*ly*lz*rho, rtol=1e-12)
                x = (np.arange(nx+1)*dx if component == 0 else (np.arange(nx)+.5)*dx)[None,None,:]
                y = (np.arange(ny+1)*dy if component == 1 else (np.arange(ny)+.5)*dy)[None,:,None]
                z = (np.arange(nz+1)*dz if component == 2 else (np.arange(nz)+.5)*dz)[:,None,None]
                radius = np.hypot(y-ly/2, z-lz/2)
                masks = {'all':1, 'near_nozzle':x<.1, 'upstream':x<.4,
                         'downstream_core':(x>.4)&(radius<.05),
                         'downstream_outer':(x>.4)&(radius>=.05)}
                mean_quadratic += float(np.sum(.5*dt*(v[1]+v[2]+v[3])**2*volume*masks[region]))
            fluctuating = row['projected_quadratic_impulse_work_W']-mean_quadratic
            assert fluctuating >= -1e-12, (region, fluctuating)
            covariance = sum(row[k]['resolved_covariance_W'] for k in ['raw_parcel','boundary_restraint','source_pressure'])
            row['projected_mean_quadratic_impulse_work_W'] = mean_quadratic
            row['projected_fluctuating_quadratic_impulse_work_W'] = fluctuating
            row['projected_replay_fluctuation_energy_rate_W'] = covariance+fluctuating
    d['quadratic_decomposition_note'] = ('Full and coarsened moments postprocessed to separate mean and fluctuating quadratic source work; '
        'half-window arrays were not archived, so no corresponding half-window decomposition is asserted. '
        'This is exact for the sampled source-only finite impulses, not for the full coupled solver stages.')
    args.output.write_text(json.dumps(d, indent=2)+'\n')
    print(json.dumps(d['regions']['all'], indent=2))


if __name__ == '__main__':
    main()
