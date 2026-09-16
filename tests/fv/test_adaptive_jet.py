"""Adaptive jet timekeeping, CFL limiting, and physical-time snapshots."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.config.document import ResolvedCase, load_case
from jaxwind.simulation.api import RunControls, build_simulation
from jaxwind.runtime.observers import Observer
from tools.run_jet_interactive import with_donated_state

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases/HITSZWindTunnel/fv_512x128x256_l24_jet_only_two_outlets_adaptive_cfl0p6_1s.toml"


def small_case():
    case = load_case(CASE)
    document = deepcopy(case.document)
    document["mesh"]["cells"] = [16, 8, 8]
    return ResolvedCase(case.source, document)


def test_adaptive_jet_time_and_steps_are_independent():
    simulation = build_simulation(small_case())
    assert simulation.adaptive
    state = simulation.initialize()
    # The same physical time with different step numbers must give the same source.
    a = simulation.advance(state, RunControls(1, .00075))
    b = simulation.advance(state._replace(step=jnp.asarray(100, jnp.int32)), RunControls(1, .00075))
    np.testing.assert_allclose(a.nitrogen_density, b.nitrogen_density, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(a.temperature, b.temperature, atol=1e-5)
    assert abs(float(a.time) - .00075) < 1e-8
    assert int(a.step) == 1 and int(b.step) == 101
    assert float(a.last_cfl) <= .60001
    # Grow past the original 0.00025 s while landing exactly on the target.
    result = simulation.advance(a, RunControls(100, .025))
    assert abs(float(result.time) - .025) < 1e-7
    assert int(result.step) < 100
    assert float(a.last_dt) > .00025
    assert np.isfinite(np.asarray(result.temperature)).all()
    assert float(result.continuity_error) < 1e-3
    assert float(simulation.courant(result)) <= .60001


def test_adaptive_snapshots_follow_time_and_restart(tmp_path):
    from jaxwind.io.checkpoint import save_checkpoint, load_checkpoint
    case = small_case()
    simulation = with_donated_state(build_simulation(case))
    state = simulation.initialize()
    observer = Observer(simulation)
    metadata = {"initial_time": 0., "initial_step": 0, "target_time": 1., "target_step": 100,
                "fingerprint": case.fingerprint}
    for index in range(2):
        count, target = observer.next_block(state, 100, metadata)
        assert abs(target - (index + 1) * .05) < 1e-9
        state = simulation.advance(state, RunControls(count, target))
        jax.block_until_ready(state)
        observer.sample(state, metadata)
    assert len(observer.frames) == 2
    np.testing.assert_allclose([f["time_seconds"] for f in observer.frames], [.05, .10], atol=1e-7)
    assert observer.history[-1]["dt_seconds"] > .00025
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, state, metadata=metadata, observer=observer.snapshot())
    template = build_simulation(case).initialize()
    restored, saved_observer, _ = load_checkpoint(path, template, fingerprint=case.fingerprint)
    assert float(restored.last_dt) == float(state.last_dt)
    observer.restore(saved_observer)
    _, target = observer.next_block(restored, 100, metadata)
    assert abs(target - .15) < 1e-9


def test_accelerating_trial_is_retried_below_cfl_limit():
    from typing import NamedTuple
    from jaxwind import StaggeredVelocity
    from jaxwind.domain import UniformGrid
    from jaxwind.simulation.jet_adaptive import build_adaptive_advance

    class State(NamedTuple):
        velocity: object
        temperature: object
        density: object
        time: object
        step: object
        last_dt: object
        last_cfl: object
        rejected_steps: object

    grid = UniformGrid(4, 2, 2, 2., 1., 1.)
    velocity = StaggeredVelocity(jnp.ones((2, 2, 5)), jnp.zeros((2, 3, 4)), jnp.zeros((3, 2, 4)))
    state = State(velocity, jnp.full((2, 2, 4), 300.), jnp.ones((2, 2, 4)),
                  jnp.asarray(0.), jnp.asarray(0), jnp.asarray(1.), jnp.asarray(0.), jnp.asarray(0))

    def trial(current, dt):
        accelerated = current.velocity._replace(x=current.velocity.x + 100. * dt)
        return current._replace(velocity=accelerated, time=current.time+dt,
                                step=current.step+1, last_dt=dt,
                                last_cfl=dt*jnp.max(accelerated.x)/grid.dx)

    advance = build_adaptive_advance(trial, SimpleNamespace(cfl=.6, dt=1., ramp_time=0.),
                                     grid, lambda velocity: 0.)
    result = advance(state, 1., 1)
    assert int(result.rejected_steps) > 0
    assert float(result.last_cfl) <= .60001
    assert int(result.step) == 1
    assert float(result.time) == float(result.last_dt)
