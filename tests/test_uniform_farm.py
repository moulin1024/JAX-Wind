from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind import (UniformGrid, StaggeredVelocity, Boundaries, Wall, FREE_SLIP,
                     OPEN, FlowModel, build_pressure_poisson, pressure_gradient,
                     divergence, project, initial_atmospheric_solution,
                     build_open_atmospheric_step)
from jaxwind.open_boundary import InflowPlane, enforce_open_velocity
from jaxwind.numerics.poisson import _diagonal_stencil
from jaxwind.config.document import load_case, ResolvedCase
from jaxwind.simulation.api import build_simulation, RunControls
from jaxwind.simulation.wind_farm import ControlledFarm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_jax_precision():
    original = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", original)


def test_lateral_outlet_pressure_projection_and_diagonal():
    jax.config.update("jax_enable_x64", True)
    grid = UniformGrid(8, 6, 4, 16., 12., 4.)
    shape = (grid.nz, grid.ny, grid.nx)
    def apply(p):
        return -divergence(pressure_gradient(p, grid, periodic_x=False, periodic_y=False, open_y=True), grid)
    # Explicit basis checks the mixed-boundary stencil and Jacobi diagonal.
    matrix = jax.jit(jax.vmap(lambda p: apply(p.reshape(shape)).reshape(-1)))(jnp.eye(grid.cell_count)).T
    np.testing.assert_allclose(matrix, matrix.T, atol=1.e-14)
    np.testing.assert_allclose(np.diag(matrix), np.asarray(_diagonal_stencil(grid, np.dtype("float64"), periodic_x=False, periodic_y=False, open_y=True)).reshape(-1))
    assert np.linalg.eigvalsh(matrix)[0] > 0
    keys = jax.random.split(jax.random.PRNGKey(11), 3)
    velocity = StaggeredVelocity(
        jax.random.normal(keys[0], (4, 6, 9)),
        jax.random.normal(keys[1], (4, 7, 8)),
        jax.random.normal(keys[2], (5, 6, 8)).at[0].set(0.).at[-1].set(0.),
    )
    solver = build_pressure_poisson(grid, backend="gmg", periodic_x=False, periodic_y=False, open_y=True, dtype="float64")
    corrected, _ = jax.jit(lambda v: project(v, solver, .1))(velocity)
    assert float(jnp.max(jnp.abs(divergence(corrected, grid)))) < 1.e-8
    np.testing.assert_array_equal(corrected.x[..., 0], velocity.x[..., 0])
    assert float(jnp.max(jnp.abs(corrected.y[:, 0]))) > .01


def test_lateral_outflow_and_uniform_open_step():
    jax.config.update("jax_enable_x64", True)
    grid = UniformGrid(8, 6, 4, 16., 12., 4.)
    velocity = StaggeredVelocity(jnp.full((4, 6, 9), 10.), jnp.full((4, 7, 8), .2), jnp.zeros((5, 6, 8)))
    plane = InflowPlane(jnp.full((4, 6), 10.), jnp.zeros((4, 7)), jnp.zeros((5, 6)), jnp.zeros((4, 6)))
    opened = enforce_open_velocity(velocity, plane, grid, open_y=True)
    np.testing.assert_allclose(opened.y[:, 0, 1:], .2)
    np.testing.assert_allclose(opened.y[:, -1, 1:], .2)
    walls = enforce_open_velocity(velocity, plane, grid)
    np.testing.assert_array_equal(walls.y[:, [0, -1]], 0.)
    solver = build_pressure_poisson(grid, backend="gmg", periodic_x=False, periodic_y=False, open_y=True, dtype="float64")
    boundaries = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=OPEN)
    step = build_open_atmospheric_step(grid, boundaries, solver, FlowModel(), None, scheme="fast-rk3")
    initial = initial_atmospheric_solution(grid, velocity._replace(y=jnp.zeros_like(velocity.y)), dtype="float64")
    final = jax.jit(lambda s: step(s, .05, plane))(initial)
    np.testing.assert_allclose(final.velocity.x, 10., atol=1.e-12)
    np.testing.assert_allclose(final.velocity.y, 0., atol=1.e-12)
    assert float(jnp.max(jnp.abs(divergence(final.velocity, grid)))) < 1.e-10


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv("JAXWIND_V80_FAST", str(ROOT / "cases/HornsRev1/turbines/V80/CustomRotor.fst"))
    return load_case(ROOT / "cases/HornsRev1/fv_hornsrev1_80_uniform10_open_gmg.toml")


def test_centered_layout_and_settings(case):
    doc = case.document
    layout = doc["physics"]["wind_farm"]["layout"]
    xy = np.array([[r["x_m"], r["y_m"]] for r in layout])
    assert len(layout) == 80
    np.testing.assert_allclose(xy.mean(axis=0), [4096., 4096.])
    np.testing.assert_allclose((xy.min(axis=0)+xy.max(axis=0))/2, [4096., 4096.])
    np.testing.assert_allclose(xy[0], [1337., 6041.5])
    np.testing.assert_allclose(xy[-1], [6855., 2150.5])
    assert doc["time"]["steps"]*doc["time"]["dt_seconds"] == 3600
    assert doc["time"]["frame_count"] == 100
    assert doc["numerics"]["pressure_backend"] == "gmg"
    assert doc["physics"]["inflow"]["speed_m_s"] == 10.


def test_open_farm_local_force_matches_full_and_runs(case):
    doc = deepcopy(case.document)
    doc["mesh"] = {"cells": [32, 24, 64], "lengths_m": [512., 384., 256.]}
    doc["physics"]["turbine"].update(x_m=256., y_m=192.)
    doc["physics"]["wind_farm"]["layout"] = [dict(id="T01", x_m=192., y_m=192., hub_height_m=70., initial_rpm=0.)]
    doc["time"].update(dt_seconds=.1, steps=2, frame_count=1)
    small = ResolvedCase(case.source, doc)
    simulation = build_simulation(small)
    farm = ControlledFarm(small)
    state = simulation.initial_state
    local = jax.jit(farm.force)(state.velocity, 0., state.rotors.omega)
    full = jax.jit(lambda v: farm.single_force(v, 0., position=farm.positions[0], angular_velocity=state.rotors.omega[0]))(state.velocity)
    for a, b in zip(local, full):
        np.testing.assert_allclose(a, b, rtol=1.e-4, atol=2.e-6)
    final = simulation.advance(state, RunControls(2, .2))
    assert all(np.isfinite(v).all() for v in final.velocity)
    np.testing.assert_allclose(final.velocity.x[..., 0], 10., atol=1.e-6)
    assert float(jnp.max(jnp.abs(divergence(final.velocity, simulation.grid)))) < 2.e-5
    diag = simulation.turbine_diagnostics(final)
    assert 0 < float(diag["farm_lookup_power_w"]) < 2.e6


def test_uniform_farm_rk3_backflow_and_diagnostics(case):
    doc = deepcopy(case.document)
    doc['mesh'] = {'cells': [8, 6, 8], 'lengths_m': [512., 384., 256.]}
    doc['physics']['turbine'].update(x_m=256., y_m=192.)
    doc['physics']['wind_farm']['layout'] = [dict(id='T01',x_m=256.,y_m=192.,hub_height_m=70.,initial_rpm=0.)]
    doc['numerics'].update(time_integration='rk3', momentum_advection_scheme='central', outlet_backflow='energy')
    doc['time'].update(dt_seconds=.1,steps=2,frame_count=1)
    simulation=build_simulation(ResolvedCase(case.source,doc))
    state=simulation.advance(simulation.initial_state,RunControls(2,.2))
    assert all(np.isfinite(v).all() for v in state.velocity)
    assert float(jnp.max(jnp.abs(divergence(state.velocity,simulation.grid))))<2e-5
    diagnostic=simulation.state_diagnostics(state)
    assert float(diagnostic['inlet_maximum_error_m_s'])==0.
    assert 0 <= float(diagnostic['outlet_x_backflow_area_fraction']) <= 1
    assert np.isfinite(float(diagnostic['boundary_net_volume_flux_m3_s']))
