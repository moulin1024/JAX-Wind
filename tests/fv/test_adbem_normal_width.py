"""Streamwise-only disk regularization preserves load and transverse support."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxwind import UniformGrid, StaggeredVelocity, build_adbem_forcing
from jaxwind.domain import ScaleSystem
from jaxwind.windfarm import RigidBladeElementDisk, load_openfast_rigid_turbine
from jaxwind.config.document import load_case, native_document
from jaxwind.config.stages import _load_turbine

ROOT = Path(__file__).resolve().parents[2]


def test_normal_width_preserves_thrust_and_radial_loading():
    grid = UniformGrid(32, 8, 20, 640., 160., 240.)
    rotor = load_openfast_rigid_turbine(ROOT/'tests/fixtures/openfast/nrel5mw/NREL5MW_Rigid_Smoke.fst')
    disk = RigidBladeElementDisk(rotor=rotor, x_m=320., y_m=80., smoothing_width_m=20.,
                                hub_height_m=90., rotor_speed_rpm=8., pitch_degrees=0.).to_actuator_disk(scales=ScaleSystem(1.,1.))
    velocity = StaggeredVelocity(jnp.full((20,8,33),10.),jnp.zeros((20,9,32)),jnp.zeros((21,8,32)))
    result = []
    for width in (0.,30.):
        force = jax.jit(build_adbem_forcing(grid,disk,periodic_x=False,periodic_y=False,
                                          minimum_normal_smoothing_width=width))(velocity,jnp.asarray(0.))
        values = np.asarray(force.x)
        volumes = np.full_like(values,grid.dx*grid.dy*grid.dz)
        volumes[...,[0,-1]] *= .5
        loads = -values*volumes
        result.append(loads)
        assert np.isfinite(values).all()
    np.testing.assert_allclose(result[0].sum(axis=2), result[1].sum(axis=2), rtol=2e-5, atol=1e-3)
    for loads in result:
        centroid = (loads*np.asarray(grid.x_faces)[None,None,:]).sum()/loads.sum()
        assert abs(centroid-disk.x)<1e-3
    moment = [float((loads*(np.asarray(grid.x_faces)-disk.x)[None,None,:]**2).sum()/loads.sum()) for loads in result]
    assert moment[1]>moment[0]*1.5


@pytest.mark.parametrize('width', [-1., float('nan'), float('inf'), True])
def test_invalid_config_width_rejected(width):
    native = native_document(load_case(ROOT/'cases/HornsRev1/fv_v80_small_central_outlet_debug.toml'))
    native['finite_volume_turbine']['minimum_normal_smoothing_width_m'] = width
    with pytest.raises(ValueError,match='minimum_normal_smoothing_width_m'):
        _load_turbine(native)


def test_config_width_is_explicit_and_legacy_default_is_zero():
    native = native_document(load_case(ROOT/'cases/HornsRev1/fv_v80_small_central_outlet_debug.toml'))
    assert _load_turbine(native).minimum_normal_smoothing_width_m == 0.
    native['finite_volume_turbine']['minimum_normal_smoothing_width_m'] = 24.
    assert _load_turbine(native).minimum_normal_smoothing_width_m == 24.
