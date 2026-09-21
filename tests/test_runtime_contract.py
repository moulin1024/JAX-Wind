"""Small compute-node contracts for state, diagnostics, and stage resume."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from jaxwind.config.document import ResolvedCase, load_case
from jaxwind.config.toml import dumps
from jaxwind.io.checkpoint import checkpoint_metadata
from jaxwind.io.state_fields import state_fields
from jaxwind.runtime.engine import run, resume

ROOT = Path(__file__).resolve().parents[1]


def tiny_case(tmp_path, *, adaptive=False):
    base = load_case(ROOT / "cases/Andren1994/config.toml")
    document = deepcopy(base.document)
    document["mesh"]["cells"] = [8, 8, 8]
    document["case"]["profile_resampling"] = "linear"
    document["numerics"].update(pressure_backend="fft", time_integration="fast-rk3")
    document["time"].update(dt_seconds=.01, steps=6, chunk_steps=2, checkpoint_every_steps=2)
    document["diagnostics"].update(sample_start_step=0, sample_every_steps=2)
    document["output"]["directory"] = str(tmp_path / "run")
    if adaptive:
        document["time"]["cfl"] = .5
    return ResolvedCase(base.source, document)


@pytest.mark.parametrize("adaptive", [False, True])
def test_resume_preserves_complete_state_and_statistics(tmp_path, adaptive):
    case = tiny_case(tmp_path, adaptive=adaptive)
    whole = run(case, output=tmp_path / "whole")
    first = run(case, output=tmp_path / "split", max_steps=2)
    assert first.summary["status"] == "paused"
    final = resume(tmp_path / "split")
    assert final.summary["status"] == "complete"
    expected, _ = state_fields(whole.checkpoint)
    actual, _ = state_fields(final.checkpoint)
    assert expected.keys() == actual.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    assert (whole.output / "profiles.csv").read_text() == (final.output / "profiles.csv").read_text()


def test_existing_output_and_incompatible_resume_are_rejected(tmp_path):
    case = tiny_case(tmp_path)
    result = run(case, max_steps=2)
    with pytest.raises(FileExistsError):
        run(case)
    changed = deepcopy(case.document)
    changed["time"]["dt_seconds"] *= 2
    (result.output / "resolved_case.toml").write_text(dumps(changed))
    with pytest.raises(ValueError, match="configuration differs"):
        resume(result.output)


def test_historical_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, velocity_x=np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="historical"):
        checkpoint_metadata(path)


def test_cryogenic_resume_preserves_parcels_and_all_histories(tmp_path):
    base = load_case(ROOT / "cases/HITSZLiquidNitrogenJet/fv_256.toml")
    document = deepcopy(base.document)
    document["mesh"]["cells"] = [8, 8, 8]
    document["time"].update(steps=4, chunk_steps=2, checkpoint_every_steps=2)
    document["physics"]["jet"].update(maximum_parcels=16, parcels_per_step=2)
    document["output"]["directory"] = str(tmp_path / "jet")
    case = ResolvedCase(base.source, document)
    whole = run(case, output=tmp_path / "whole")
    run(case, output=tmp_path / "split", max_steps=2)
    split = resume(tmp_path / "split")
    expected, _ = state_fields(whole.checkpoint)
    actual, _ = state_fields(split.checkpoint)
    assert "parcels_mass" in actual and "temperature_tendency" in actual
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def test_differentiable_inlet_remains_an_explicit_jax_input():
    import jax
    import jax.numpy as jnp
    from dataclasses import replace
    from jaxwind.config.jet import load_case as load_jet
    from jaxwind.simulation.jet import build_simulation
    native = replace(load_jet(ROOT / "cases/HITSZLiquidNitrogenJet/fv_256_incompressible_rk3_inlet.toml"),
                     cells=(8, 8, 8), maximum_parcels=16, parcels_per_step=2)
    grid, jet, microphysics, initialize, advance, courant, control = build_simulation(native, differentiable_inlet=True)
    def objective(scale):
        current_control = control._replace(speed_scale=scale)
        initial = initialize(current_control)
        final = advance(initial, current_control, 1)
        return jnp.sum(final.velocity.x)
    derivative = jax.grad(objective)(control.speed_scale)
    assert np.isfinite(float(derivative))


def test_workflow_resume_and_coverage(tmp_path):
    from jaxwind.workflows.engine import execute
    case = tiny_case(tmp_path)
    case.document["workflow"].update(warmup_steps=4, precursor_steps=4, main_steps=4,
                                      record_plane=1, chunk_steps=2, output_directory=str(tmp_path / "workflow"))
    path = tmp_path / "case.toml"
    path.write_text(dumps(case.document))
    first = execute(path, max_steps=2)
    assert first["stages"]["warmup"]["status"] == "paused"
    final = execute(path, resume=True)
    assert all(value["status"] == "complete" for value in final["stages"].values())
    assert (tmp_path / "workflow/precursor/inflow/metadata.json").is_file()
    case.document["workflow"]["main_steps"] = 5
    case.document["workflow"]["output_directory"] = str(tmp_path / "invalid")
    path.write_text(dumps(case.document))
    with pytest.raises(ValueError):
        execute(path)


def test_low_mach_continuation_consumes_new_stage_checkpoint(tmp_path):
    from jaxwind.workflows.engine import execute
    path = ROOT / "cases/workflows/low_mach_continuation.toml"
    output = tmp_path / "continuation"
    paused = execute(path, output=output, max_steps=2)
    assert paused["stages"]["warmup"]["status"] == "paused"
    completed = execute(path, output=output, resume=True)
    assert completed["stages"]["warmup"]["step"] == 6
    assert completed["stages"]["continuation"]["step"] == 12
    assert completed["stages"]["continuation"]["status"] == "complete"
    assert checkpoint_metadata(output / "continuation/checkpoint.npz")["initial_step"] == 6


@pytest.mark.parametrize("adaptive", [False, True])
def test_physical_time_checkpoints_survive_mid_interval_resume(tmp_path, monkeypatch, adaptive):
    """Checkpoint clocks must not depend on actual step count or resume time."""
    from types import SimpleNamespace
    from collections import namedtuple
    from jaxwind.simulation.api import Simulation
    from jaxwind.runtime import engine
    from jaxwind.runtime.observers import Observer

    case = tiny_case(tmp_path, adaptive=adaptive)
    case.document["time"].update(dt_seconds=1. if adaptive else .25,
        steps=3 if adaptive else 12, chunk_steps=5, frame_count=0,
        checkpoint_every_steps=1, checkpoint_every_seconds=1.)
    State = namedtuple("ClockState", "step time")
    initial = State(np.asarray(0, dtype=np.int32), np.asarray(0., dtype=np.float64))
    def advance(state, controls):
        for _ in range(controls.count):
            dt = min(.3, controls.target_time - float(state.time)) if adaptive else .25
            if dt <= 1.e-12:
                break
            state = State(state.step + np.int32(1), state.time + dt)
        return state
    grid = SimpleNamespace(**{name: np.array([0., 1.]) for name in ("x_faces", "y_faces", "z_faces")})
    simulation = Simulation(case, grid, initial, advance, lambda state: 0., adaptive)
    monkeypatch.setattr(Observer, "sample", lambda *args: None)
    monkeypatch.setattr(Observer, "write", lambda *args: None)
    monkeypatch.setattr(Observer, "summary", lambda self: {})
    saved_times = []
    save = engine.save_checkpoint
    def capture(path, state, **kwargs):
        saved_times.append(float(state.time))
        save(path, state, **kwargs)
    monkeypatch.setattr(engine, "save_checkpoint", capture)
    first = run(case, max_steps=2, _simulation=simulation)
    assert first.summary["status"] == "paused"
    final = run(case, _resume=True, _simulation=simulation)
    assert final.summary["status"] == "complete"
    assert final.summary["time_seconds"] == pytest.approx(3.)
    np.testing.assert_allclose(saved_times, [0., .6 if adaptive else .5, 1., 2., 3., 3.])


def test_adaptive_frame_boundary_after_shortening_schedule(tmp_path):
    from types import SimpleNamespace
    from jaxwind.runtime.observers import Observer
    case = tiny_case(tmp_path, adaptive=True)
    case.document["time"].update(dt_seconds=6., steps=6000, frame_count=100,
        checkpoint_every_seconds=3600.)
    observer = Observer(SimpleNamespace(case=case, diagnostics=None, adaptive=True))
    observer.frames = [{"time_seconds": 720.}, {"time_seconds": 1440.}]
    state = SimpleNamespace(step=np.int32(7200), time=np.float32(1800.))
    metadata = {"initial_step": 0, "initial_time": 0., "target_time": 36000.}
    _, target = observer.next_block(state, 120, metadata)
    assert target == 2160.


@pytest.mark.parametrize("operation", ["periodic", "record-inflow"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_periodic_stages_average_volume_profiles_across_resume(tmp_path, operation, adaptive):
    """Stage profiles average evolved full-volume fields, including before resume."""
    from jaxwind.simulation.stages import build_stage

    case = tiny_case(tmp_path, adaptive=adaptive)
    case.document["time"]["steps"] = 4
    case.document["initial_conditions"] = {
        "operation": operation, "artifacts": {}, "stage_options": {},
    }
    simulation = build_stage(case, operation, {}, {})
    first = run(case, max_steps=2, _simulation=simulation)
    first_fields, _ = state_fields(first.checkpoint)
    final = resume(first.output)
    final_fields, _ = state_fields(final.checkpoint)
    profiles = np.genfromtxt(final.output / "profiles.csv", delimiter=",", names=True)
    expected = .5 * (
        first_fields["velocity_x"].mean(axis=(1, 2), dtype=np.float64)
        + final_fields["velocity_x"].mean(axis=(1, 2), dtype=np.float64)
    )
    np.testing.assert_allclose(profiles["mean_u_m_s"], expected, rtol=2.e-6, atol=2.e-6)
    assert final.summary["runtime"]["profile_samples"] == 2
    assert final.summary["runtime"]["ustar_m_s"] > 0.
    whole = run(case, output=tmp_path / "whole", _simulation=simulation)
    assert (whole.output / "profiles.csv").read_text() == (final.output / "profiles.csv").read_text()


@pytest.mark.parametrize("operation", ["periodic", "record-inflow"])
def test_adaptive_stage_reports_actual_cfl_and_lands_on_samples(tmp_path, operation):
    from jaxwind.simulation.stages import build_stage
    case = tiny_case(tmp_path, adaptive=True)
    case.document["time"].update(dt_seconds=100., steps=2, chunk_steps=40)
    case.document["time"]["cfl"] = .9
    case.document["diagnostics"].update(sample_every_steps=1)
    case.document["numerics"].update(momentum_advection_scheme="central", time_integration="rk3")
    result = run(case, _simulation=build_stage(case, operation, {}, {}))
    history = np.atleast_1d(np.genfromtxt(result.output / "history.csv", delimiter=",", names=True))
    assert result.summary["time_seconds"] == pytest.approx(200.)
    assert result.summary["runtime"]["profile_samples"] == 2
    assert np.max(history["block_maximum_cfl"]) <= .90001
    assert np.max(history["block_maximum_cfl"]) > .89
    assert np.max(history["dt_seconds"]) < 100.
    if operation == "record-inflow":
        import json
        meta = json.loads((result.output / "inflow/metadata.json").read_text())
        assert meta["duration_seconds"] == pytest.approx(200., abs=1.e-4)
        assert meta["samples"] > 2


@pytest.mark.parametrize("adaptive", [False, True])
def test_abl_cfl_callback_is_separate_from_state_diagnostics(tmp_path, adaptive):
    from jaxwind.simulation.api import build_simulation
    from jaxwind.runtime.observers import Observer

    case = tiny_case(tmp_path, adaptive=adaptive)
    simulation = build_simulation(case)
    state = simulation.initialize()
    assert simulation.state_diagnostics is None
    if adaptive:
        cfl = simulation.courant_with_timestep
        assert cfl is not None
        np.testing.assert_allclose(cfl(state, .02), 2 * cfl(state, .01))
    else:
        assert simulation.courant_with_timestep is None
    observer = Observer(simulation)
    observer.sample(state, {"initial_step": 0, "initial_time": 0., "target_time": .06})
    assert len(observer.history) == 1
    assert np.isfinite(observer.history[0]["maximum_cfl"])
