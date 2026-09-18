"""Diagnostic invariants: no artificial Reynolds stress and outward flux sign."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind import StaggeredVelocity, initial_atmospheric_solution
from jaxwind.config.document import load_case
from jaxwind.domain import UniformGrid
from jaxwind.moist_abl import initialize_moisture
from jaxwind.physics.moisture import MoistureConfig
from jaxwind.water_parcels import InertialMoistAtmosphericSolution, WaterParcelSource, initial_water_parcels

ROOT = Path(__file__).resolve().parents[2]
jax.config.update('jax_enable_x64', True)


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_steady_spatial_gradients_do_not_become_reynolds_stress():
    audit = module('audit_spray_radial_momentum')
    doc = load_case(ROOT / 'cases/WaterSprayMontazeri2015/centre_investigation/uniform_256x128x128_case3.toml').document
    doc['case']['parcel_capacity'] = 64
    grid = UniformGrid(16, 8, 8, 1.9, .585, .585)
    u = jnp.broadcast_to(3 + 2 * jnp.asarray(grid.x_faces)[None, None, :], (8, 8, 17))
    v = jnp.broadcast_to(.1 * jnp.asarray(grid.x_centers)[None, None, :] *
        (1 - (2*jnp.asarray(grid.y_faces)[None, :, None]/grid.ly - 1)**2), (8, 9, 16))
    flow = initial_atmospheric_solution(grid, StaggeredVelocity(u, v, jnp.zeros((9,8,16))), dtype='float64')
    flow, _ = initialize_moisture(flow, 312.35, .1, MoistureConfig())
    src = WaterParcelSource(center=(0,.2925,.2925),radius=.002,speed=22,temperature=308.35,
        mass_flow=.2,half_angle_degrees=18,diameter_scale=.000369,diameter_spread=3.67,
        diameter_minimum=.000074,diameter_maximum=.000518,capacity=64)
    state = InertialMoistAtmosphericSolution(*flow, initial_water_parcels(src,'float64'))
    state = state._replace(time=jnp.array(4.), step=jnp.array(32000))
    fields, rings, residual, total = map(np.asarray, audit.make_sampler(grid,doc)[0](state))
    np.testing.assert_allclose(fields[3]-fields[0]*fields[1], 0, atol=1e-13)
    np.testing.assert_allclose(fields[4]-fields[0]*fields[2], 0, atol=1e-13)
    np.testing.assert_allclose(fields[5:8]-fields[:3]**2, 0, atol=1e-13)
    np.testing.assert_allclose(rings.sum(axis=(1,2))*grid.dx,total,rtol=1e-12,atol=1e-13)
    assert total[0] > 0
    assert np.max(np.abs(residual)) < 1e-15


def test_outward_stress_and_temporal_covariance_are_separate_from_mean_flow():
    plot = module('plot_spray_radial_momentum')
    y=z=np.linspace(.02,.565,8);yy=y[None,:,None]-.2925;zz=z[:,None,None]-.2925
    r=np.hypot(yy,zz);ny=yy/r;nz=zz/r;U=10-r**2;V=np.broadcast_to(.2*ny,U.shape);W=np.broadcast_to(.2*nz,U.shape)
    # u'=±1, radial v'=±0.3 gives <u'v_r'>=0.3 independently of mean radial flow.
    fields={'u':U,'v':V,'w':W,'uv':U*V+.3*ny,'uw':U*W+.3*nz,
        'sgs_xy':-.02*ny,'sgs_xz':-.02*nz,'molecular_xy':-.001*ny,'molecular_xz':-.001*nz}
    terms,_=plot.radial_terms(np.stack(list(fields.values())),list(fields),y,z,1.125)
    np.testing.assert_allclose(terms['resolved_turbulence'],1.125*.3,atol=1e-14)
    np.testing.assert_allclose(terms['mean_advection'],1.125*U*.2,atol=1e-14)
    np.testing.assert_allclose(terms['SGS'],1.125*.02,atol=1e-14)


def test_amd_time_sampler_rejects_mislabeled_rans_stresses():
    import pytest
    audit = module('audit_spray_radial_momentum')
    grid = UniformGrid(16, 8, 8, 1.9, .585, .585)
    for case in [{'carrier_turbulence_model': 'realizable-k-epsilon'},
                 {'carrier_sgs_model': 'smagorinsky'}]:
        with pytest.raises(ValueError, match='AMD only'):
            audit.make_sampler(grid, {'case':case})
