"""Compare native modeled radial momentum flux at saved spray checkpoints.

No advancement. A snapshot cannot supply temporal Reynolds stresses. The
RANS stress is evaluated from its saved k/epsilon; LES uses actual AMD.
"""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jaxwind import Boundaries, FREE_SLIP, OPEN, Wall, StaggeredVelocity
from jaxwind.domain import UniformGrid
from jaxwind.numerics.discretization import cell_velocity
from jaxwind.rans_kepsilon import KEpsilonState, turbulent_viscosity
from jaxwind.rans_realizable import turbulent_viscosity as realizable_viscosity
from jaxwind.sgs import (AnisotropicMinimumDissipation, eddy_viscosity, edge_gradients,
                        _to_xy_edge_from_cell, _to_xz_edge_from_cell,
                        _to_cell_from_xy_edge, _to_cell_from_xz_edge)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    assert jax.default_backend() == 'gpu', jax.devices()
    args.output.mkdir(parents=True, exist_ok=True)
    stations = np.array([.1,.4,.7,1.,1.3,1.6,1.9])
    reports = {}; profiles = {}
    for run in args.runs:
        checkpoint = run/'checkpoint.npz'
        a = np.load(checkpoint)
        doc = json.loads(str(a['metadata']))['resolved_case']
        grid = UniformGrid(*doc['mesh']['cells'], *doc['mesh']['lengths_m'])
        bc = Boundaries(Wall(FREE_SLIP),Wall(FREE_SLIP),streamwise=OPEN,spanwise=FREE_SLIP)
        velocity = StaggeredVelocity(*(jnp.asarray(a['state/velocity/'+c]) for c in 'xyz'))
        gradients = edge_gradients(velocity,grid,bc)
        model = doc['case'].get('carrier_turbulence_model','les')
        if model == 'les':
            assert doc['case'].get('carrier_sgs_model','amd') == 'amd'
            nut = eddy_viscosity(velocity,grid,bc,AnisotropicMinimumDissipation(),gradients=gradients)
        else:
            state = KEpsilonState(*(jnp.asarray(a['state/turbulence/'+c]) for c in KEpsilonState._fields))
            if model == 'realizable-k-epsilon': nut = realizable_viscosity(state,velocity,grid,bc)
            elif model == 'standard-k-epsilon': nut = turbulent_viscosity(state)
            else: raise ValueError(model)
        uy = _to_xy_edge_from_cell(nut,open_x=True,wall_y=True)
        uz = _to_xz_edge_from_cell(nut,open_x=True)
        # First part is radial derivative of u; second is axial derivative of v,w.
        stresses = []
        for gy,gz in [('xy','xz'),('yx','zx')]:
            sy = _to_cell_from_xy_edge(uy*gradients[gy],open_x=True,wall_y=True)
            sz = _to_cell_from_xz_edge(uz*gradients[gz],open_x=True)
            stresses.append((np.asarray(sy),np.asarray(sz)))
        xc = np.asarray(grid.x_centers)
        indices = np.clip(np.searchsorted(xc,stations),1,grid.nx-1)
        alpha = np.clip((stations-xc[indices-1])/(xc[indices]-xc[indices-1]),0,1)
        sample = lambda f: np.asarray(f)[...,indices-1]*(1-alpha)+np.asarray(f)[...,indices]*alpha
        u,v,w = [sample(f) for f in cell_velocity(velocity)]
        yy = np.asarray(grid.y_centers)[None,:,None]-grid.ly/2
        zz = np.asarray(grid.z_centers)[:,None,None]-grid.lz/2
        r = np.hypot(yy,zz); ny,nz = yy/r,zz/r
        rho = doc['physics']['moisture']['dry_air_density_kg_m3']
        parts = [-rho*(sample(sy)*ny+sample(sz)*nz) for sy,sz in stresses]
        radial_speed = v*ny+w*nz
        terms = {'mean_advection':rho*u*radial_speed,
                 'modeled_stress':sum(parts), 'radial_gradient_stress':parts[0],
                 'axial_gradient_stress':parts[1], 'eddy_viscosity':sample(nut),
                 'radial_velocity':radial_speed, 'axial_velocity':u}
        edges = np.linspace(0,.28, max(2,int(.28/max(grid.dy,grid.dz)))+1)
        radius = .5*(edges[1:]+edges[:-1])
        rr = r[...,0]
        counts = np.array([np.sum((rr>=lo)&(rr<hi)) for lo,hi in zip(edges[:-1],edges[1:])])
        assert np.all(counts>0),counts
        radial = {k:np.array([np.mean(f[(rr>=lo)&(rr<hi)],axis=0)
                              for lo,hi in zip(edges[:-1],edges[1:])]) for k,f in terms.items()}
        np.testing.assert_allclose(radial['modeled_stress'],radial['radial_gradient_stress']+radial['axial_gradient_stress'],atol=1e-12)
        assert all(np.isfinite(f).all() for f in radial.values())
        pick = np.abs(radius-.05).argmin()
        reports[run.name] = {'time_s':float(a['state/time']),'model':model,
            'mesh':doc['mesh']['cells'],'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'sample_radius_m':float(radius[pick]),
            'stations':[{'x_m':float(x),**{k:float(f[pick,j]) for k,f in radial.items()}} for j,x in enumerate(stations)]}
        profiles[run.name] = (radius,radial)
        np.savez_compressed(args.output/(run.name+'.npz'),radius_m=radius,stations_m=stations,**radial)
    fig,axes = plt.subplots(3,7,figsize=(19,9),layout='constrained')
    for name,(radius,radial) in profiles.items():
        for row,key in enumerate(['mean_advection','modeled_stress','eddy_viscosity']):
            for j,station in enumerate(stations):
                axes[row,j].plot(radius,radial[key][:,j],label=name)
                axes[row,j].set(xlabel='radius (m)',title=f'x={station:g} m')
                axes[row,j].axhline(0,color='k',lw=.5);axes[row,j].grid(alpha=.2)
                if j==0:axes[row,j].set_ylabel(key+(' (m²/s)' if row==2 else ' (Pa)'))
    axes[0,0].legend(fontsize=6)
    fig.suptitle('Saved-state radial momentum transport; positive outward; no temporal covariance')
    fig.savefig(args.output/'radial_transport.png',dpi=150);plt.close(fig)
    report = {'scope':__doc__, 'limitations':['Instantaneous checkpoints, not means or a closed momentum budget.',
        'No resolved temporal Reynolds covariance can be inferred from one checkpoint.',
        'Native stress is mapped to cells then interpolated in x; last station uses outlet cell.',
        'Complete annuli inside .28m; no wall traction or pressure budget included.'], 'runs':reports}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':main()
