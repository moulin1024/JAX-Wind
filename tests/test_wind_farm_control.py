from copy import deepcopy
from pathlib import Path
import tomllib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.config.document import ResolvedCase, load_case
from jaxwind.config.toml import dumps
from jaxwind.config.wind_farm import validate_wind_farm
from jaxwind.simulation.wind_farm import ControlledFarm, RotorState, advance_rotors, lookup_outputs
from jaxwind.simulation.api import RunControls, build_simulation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv("JAXWIND_V80_FAST", str(ROOT / "cases/HornsRev1/turbines/V80/CustomRotor.fst"))
    source = load_case(ROOT / "cases/HornsRev1/fv_v80_two_turbine_tsr.toml")
    doc = deepcopy(source.document)
    doc["mesh"] = {"cells": [32, 16, 256], "lengths_m": [512., 256., 1024.]}
    doc["physics"]["turbine"].update(x_m=256., y_m=128.)
    layout = doc["physics"]["wind_farm"]["layout"]
    for row, x, rpm in zip(layout, (160., 352.), (8., 14.)):
        row.update(x_m=x, y_m=128., initial_rpm=rpm)
    doc["time"].update(dt_seconds=.05, steps=4, chunk_steps=2, frame_count=2, checkpoint_every_steps=2)
    doc["diagnostics"].update(sample_start_step=0, sample_every_steps=1)
    return ResolvedCase(source.source, doc)


def test_tracking_is_independent_and_timestep_aware(case):
    control = case.document["physics"]["wind_farm"]["controller"]
    wind = jnp.array([8., 4.])
    initial = RotorState(jnp.zeros(2), wind)
    def evolve(dt, count):
        return jax.jit(lambda s: jax.lax.fori_loop(0, count, lambda _, v: advance_rotors(v, wind, dt, 40., control), s))(initial)
    a, b = evolve(.1, 2000), evolve(.5, 400)
    np.testing.assert_allclose(a.omega * 40. / wind, 7., atol=1.e-4)
    np.testing.assert_allclose(a.omega, b.omega, atol=1.e-5)
    assert a.omega[0] > a.omega[1]


def test_filter_slew_rpm_and_zero_wind(case):
    control = case.document["physics"]["wind_farm"]["controller"]
    initial = RotorState(jnp.array([0., 1.]), jnp.array([0., 8.]))
    result = advance_rotors(initial, jnp.array([100., -5.]), 1., 40., control)
    assert float(result.omega[0]) <= .5 * 2 * np.pi / 60 + 1.e-6
    assert 0 < result.filtered_wind[0] < 100
    assert 0 < result.filtered_wind[1] < 8
    high = advance_rotors(initial, jnp.ones(2)*100, 1.e4, 40., control)
    np.testing.assert_allclose(high.omega * 60 / (2*np.pi), control["maximum_rpm"], atol=1.e-5)
    stopped = advance_rotors(high, jnp.zeros(2), 1.e4, 40., control)
    np.testing.assert_allclose(stopped.omega, 0., atol=1.e-6)
    unchanged = advance_rotors(initial, jnp.zeros(2), 0., 40., control)
    np.testing.assert_array_equal(unchanged.omega, initial.omega)


def test_schema_roundtrip_and_errors(case):
    assert tomllib.loads(dumps(case.document)) == case.document
    for mutation in ("duplicate", "nan", "limits", "empty", "unknown"):
        doc = deepcopy(case.document)
        farm = doc["physics"]["wind_farm"]
        if mutation == "duplicate": farm["layout"][1]["id"] = "T01"
        if mutation == "nan": farm["controller"]["target_tsr"] = float("nan")
        if mutation == "limits": farm["controller"]["minimum_rpm"] = 20.
        if mutation == "empty": farm["layout"] = []
        if mutation == "unknown": farm["controller"]["typo"] = 1
        with pytest.raises(ValueError): validate_wind_farm(doc)


def test_farm_force_is_sum_and_dynamic_rpm(case):
    farm = ControlledFarm(case)
    simulation = build_simulation(case)
    velocity = simulation.initial_state.velocity
    omega = jnp.array([.8, 1.4])
    total = jax.jit(farm.force)(velocity, 0., omega)
    singles = [jax.jit(lambda v, p, w: farm.single_force(v, 0., position=p, angular_velocity=w))(velocity, farm.positions[i], omega[i]) for i in range(2)]
    for actual, a, b in zip(total, singles[0], singles[1]):
        np.testing.assert_allclose(actual, a+b, rtol=1.e-5, atol=1.e-7)
        assert np.isfinite(actual).all()
    changed = jax.jit(farm.force)(velocity, 0., omega*.8)
    assert not np.allclose(total.x, changed.x)
    stopped = jax.jit(farm.force)(velocity, 0., jnp.zeros_like(omega))
    assert all(np.isfinite(value).all() for value in stopped)
    # Probes sample their own local wind, not a common farm mean.
    from jaxwind.state import StaggeredVelocity
    u = jnp.broadcast_to(jnp.linspace(4., 10., farm.grid.nx), velocity.x.shape)
    measured = farm.sample_wind(StaggeredVelocity(u, velocity.y, velocity.z))
    assert abs(float(measured[0]-measured[1])) > .1


def test_coupled_checkpoint_resume(case, tmp_path):
    from jaxwind.io.checkpoint import save_checkpoint, load_checkpoint
    simulation = build_simulation(case)
    initial = simulation.initial_state
    first = simulation.advance(initial, RunControls(1, .05))
    jax.block_until_ready(first)
    assert first.rotors.omega[0] > initial.rotors.omega[0]
    assert first.rotors.omega[1] < initial.rotors.omega[1]
    assert float(first.time) == pytest.approx(.05)
    p = tmp_path / "checkpoint.npz"
    save_checkpoint(p, first, metadata={"fingerprint": case.fingerprint})
    restored, _, _ = load_checkpoint(p, initial, fingerprint=case.fingerprint)
    direct = simulation.advance(initial, RunControls(2, .1))
    resumed = simulation.advance(restored, RunControls(1, .1))
    for a, b in zip(jax.tree.leaves(direct), jax.tree.leaves(resumed)):
        np.testing.assert_allclose(a, b, rtol=1.e-6, atol=1.e-6)
    diagnostics = simulation.turbine_diagnostics(resumed)
    assert "turbine_T01_rpm" in diagnostics and "turbine_T02_tsr" in diagnostics
    assert np.isfinite(np.asarray(resumed.velocity.x)).all()


def test_runtime_resume_and_output(case, tmp_path):
    from jaxwind.runtime.engine import run, resume
    partial = run(case, output=tmp_path / "run", max_steps=1)
    assert partial.summary["status"] == "paused"
    final = resume(partial.output)
    assert final.summary["status"] == "complete"
    assert final.summary["time_seconds"] == pytest.approx(.2)
    history = np.genfromtxt(partial.output / "history.csv", delimiter=",", names=True)
    assert "turbine_T02_rpm" in history.dtype.names
    with np.load(partial.output / "flow_frames.npz") as frames:
        np.testing.assert_allclose(frames["time_seconds"], [.1, .2], atol=1.e-6)


def test_initialize_farm_from_ordinary_warmup(case, tmp_path):
    from jaxwind.abl import AtmosphericSolution
    from jaxwind.io.checkpoint import save_checkpoint
    simulation = build_simulation(case)
    flow = AtmosphericSolution(*simulation.initial_state[:7])._replace(time=jnp.asarray(12., jnp.float32))
    checkpoint = tmp_path / "warmup.npz"
    mesh = {name: np.asarray(getattr(simulation.grid, name)).tolist() for name in ("x_faces", "y_faces", "z_faces")}
    save_checkpoint(checkpoint, flow, metadata={"formulation": "boussinesq", "mesh": mesh})
    doc = deepcopy(case.document)
    doc.setdefault("initial_conditions", {})["checkpoint"] = str(checkpoint)
    initialized = build_simulation(ResolvedCase(case.source, doc)).initial_state
    np.testing.assert_array_equal(initialized.velocity.x, flow.velocity.x)
    assert float(initialized.time) == 12.
    np.testing.assert_allclose(initialized.rotors.omega * 60 / (2*np.pi), [8., 14.], atol=1.e-5)


def test_observation_boundary_does_not_repeat(case):
    from jaxwind.runtime.observers import Observer
    simulation = build_simulation(case)
    doc = case.document
    doc["time"].update(dt_seconds=.2, steps=600, frame_count=0, chunk_steps=20)
    doc["diagnostics"].update(sample_start_step=540, sample_every_steps=6)
    observer = Observer(simulation)
    state = simulation.initial_state._replace(time=jnp.asarray(109.2, jnp.float32))
    _, target = observer.next_block(state, 20, {"initial_time": 0., "initial_step": 0, "target_time": 120.})
    assert target == pytest.approx(110.4)


@pytest.fixture
def lookup_case(case):
    source = load_case(ROOT / "cases/HornsRev1/fv_v80_two_turbine_lookup.toml")
    doc = deepcopy(case.document)
    doc["physics"]["wind_farm"]["controller"] = source.document["physics"]["wind_farm"]["controller"]
    return ResolvedCase(source.source, doc)


def test_lookup_knots_midpoints_and_operating_limits(lookup_case):
    control = lookup_case.document["physics"]["wind_farm"]["controller"]
    evaluate = jax.jit(lambda u: lookup_outputs(u, control))
    rpm, power = evaluate(jnp.array([-5., 0., 3.99, 4., 8., 8.5, 10., 20., 24.99, 25., 30.]))
    np.testing.assert_allclose(rpm, [0, 0, 0, 12.5, 16.3, 16.95, 18., 18.1, 18.1, 0, 0], atol=1.e-5)
    np.testing.assert_allclose(power, [0, 0, 0, 66600, 696000, 846000, 1341000, 2000000, 2000000, 0, 0])
    # Every in-operation table knot is reproduced without rounding or units loss.
    rpm, power = evaluate(jnp.asarray(control["wind_speed_m_s"][:-1]))
    np.testing.assert_allclose(rpm, control["rpm"][:-1], rtol=1.e-6)
    np.testing.assert_allclose(power, control["power_w"][:-1])


def test_lookup_filter_and_direct_prescribed_rpm(lookup_case):
    control = lookup_case.document["physics"]["wind_farm"]["controller"]
    initial = RotorState(jnp.zeros(2), jnp.array([8., 6.]))
    wind = jnp.array([10., 8.])
    result = jax.jit(lambda s: advance_rotors(s, wind, 1., 40., control))(initial)
    np.testing.assert_allclose(result.filtered_wind, wind + (initial.filtered_wind-wind)*np.exp(-.2))
    expected, _ = lookup_outputs(result.filtered_wind, control)
    np.testing.assert_allclose(result.omega * 60 / (2*np.pi), expected, rtol=1.e-6)
    assert result.omega[0] > result.omega[1]
    same = advance_rotors(initial, wind, 0., 40., control)
    np.testing.assert_array_equal(same.omega, initial.omega)
    smaller = initial
    for _ in range(10):
        smaller = advance_rotors(smaller, wind, .1, 40., control)
    np.testing.assert_allclose(smaller.filtered_wind, result.filtered_wind, atol=2.e-6)
    np.testing.assert_allclose(smaller.omega, result.omega, atol=2.e-6)


@pytest.mark.parametrize("key,value", [
    ("wind_speed_m_s", [4., 4., 25.]),
    ("rpm", [12.]), ("power_w", [0., -1.]),
    ("wind_filter_seconds", 0.), ("cut_out_m_s", 26.),
    ("cut_in_m_s", 2.), ("cut_in_m_s", 25.),
    ("rpm_source", ""), ("rpm", [float("nan")]*22),
    ("power_w", [True]*22), ("power_w", [0., 1.]),
])
def test_lookup_schema_rejects_bad_data(lookup_case, key, value):
    doc = deepcopy(lookup_case.document)
    doc["physics"]["wind_farm"]["controller"][key] = value
    with pytest.raises(ValueError):
        validate_wind_farm(doc)


def test_lookup_schema_roundtrip_and_fingerprint(lookup_case):
    validate_wind_farm(lookup_case.document)
    assert tomllib.loads(dumps(lookup_case.document)) == lookup_case.document
    doc = deepcopy(lookup_case.document)
    doc["physics"]["wind_farm"]["controller"]["rpm"][0] += .1
    assert lookup_case.fingerprint != ResolvedCase(lookup_case.source, doc).fingerprint


def test_lookup_plane_probe_and_initialization(lookup_case):
    farm = ControlledFarm(lookup_case)
    simulation = build_simulation(lookup_case)
    velocity = simulation.initial_state.velocity
    np.testing.assert_allclose(np.sum(farm.wx, axis=1), 1.)
    assert np.all(np.sum(np.asarray(farm.wx) > 0, axis=1) <= 2)
    # Linear streamwise field: interpolated plane samples must be at x_i - D.
    faces = jnp.asarray(farm.grid.x_faces)
    u = jnp.broadcast_to(4. + .01*faces[:velocity.x.shape[-1]], velocity.x.shape)
    measured = farm.sample_wind(velocity._replace(x=u))
    expected = 4. + .01*(np.asarray(farm.positions)[:, 0]-80.)
    np.testing.assert_allclose(measured, expected, atol=2.e-5)
    # Probe crossing x=0 interpolates periodically between last and first cells.
    doc = deepcopy(lookup_case.document)
    doc["physics"]["wind_farm"]["layout"][0]["x_m"] = 80.
    wrapped = ControlledFarm(ResolvedCase(lookup_case.source, doc))
    np.testing.assert_allclose(np.asarray(wrapped.wx)[0, [0, -1]], [.5, .5])
    state = simulation.initial_state
    rpm, power = lookup_outputs(state.rotors.filtered_wind, farm.control)
    np.testing.assert_allclose(state.rotors.omega*60/(2*np.pi), rpm, rtol=1.e-6)
    diagnostics = farm.diagnostics(state)
    assert diagnostics["farm_lookup_power_w"] == pytest.approx(float(jnp.sum(power)))


def test_lookup_runtime_resume_and_power_history(lookup_case, tmp_path):
    from jaxwind.runtime.engine import run, resume
    partial = run(lookup_case, output=tmp_path / "lookup", max_steps=1)
    assert partial.summary["status"] == "paused"
    final = resume(partial.output)
    assert final.summary["status"] == "complete"
    history = np.atleast_1d(np.genfromtxt(partial.output / "history.csv", delimiter=",", names=True))
    control = lookup_case.document["physics"]["wind_farm"]["controller"]
    total = np.zeros(len(history))
    for name in ("T01", "T02"):
        prefix = f"turbine_{name}_"
        rpm, power = lookup_outputs(jnp.asarray(history[prefix+"filtered_wind_m_s"]), control)
        np.testing.assert_allclose(history[prefix+"rpm"], rpm, rtol=2.e-6)
        np.testing.assert_allclose(history[prefix+"lookup_power_w"], power, rtol=2.e-6)
        total += history[prefix+"lookup_power_w"]
    np.testing.assert_allclose(history["farm_lookup_power_w"], total, rtol=2.e-6)


def test_lookup_coupled_checkpoint_resume(lookup_case, tmp_path):
    from jaxwind.io.checkpoint import save_checkpoint, load_checkpoint
    simulation = build_simulation(lookup_case)
    initial = simulation.initial_state
    first = simulation.advance(initial, RunControls(1, .05))
    p = tmp_path / "lookup_checkpoint.npz"
    save_checkpoint(p, first, metadata={"fingerprint": lookup_case.fingerprint})
    restored, _, _ = load_checkpoint(p, initial, fingerprint=lookup_case.fingerprint)
    direct = simulation.advance(initial, RunControls(2, .1))
    resumed = simulation.advance(restored, RunControls(1, .1))
    for a, b in zip(jax.tree.leaves(direct), jax.tree.leaves(resumed)):
        np.testing.assert_allclose(a, b, rtol=1.e-6, atol=1.e-6)
