"""Gravity work, residence-limited impulse and all-phase rollback."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_spray_cell_step import assert_tree_equal, budget
from test_spray_moving import coupled, moving_setup, source

from jaxwind.spray_body_force import kick_particles

GRAVITY = (0.0, 0.0, -9.81)


@pytest.mark.parametrize("dt", [0.0, 1e-5, 0.4])
def test_exact_external_kick_and_galilean_work(dt):
    _, _, _, _, q, _ = moving_setup()
    q = q._replace(multiplicity=q.multiplicity.at[-1].set(0))
    acceleration = jnp.array([2.0, -3.0, -9.81])
    r, impulse, work = jax.jit(kick_particles)(q, acceleration, dt)
    m = q.mass * q.multiplicity
    change = np.asarray(acceleration)[:, None] * dt * np.asarray(m > 0)[None]
    np.testing.assert_allclose(r.velocity, q.velocity + change, atol=1e-15)
    np.testing.assert_allclose(impulse, m * change, rtol=1e-14, atol=1e-20)
    dk = (
        0.5
        * np.asarray(m)
        * np.sum(np.asarray(r.velocity) ** 2 - np.asarray(q.velocity) ** 2, axis=0)
    )
    np.testing.assert_allclose(work, dk, rtol=1e-9, atol=2e-20)
    boost = jnp.array([8.0, -4.0, 2.0])
    boosted = q._replace(velocity=q.velocity + boost[:, None])
    _, impulse2, work2 = jax.jit(kick_particles)(boosted, acceleration, dt)
    np.testing.assert_allclose(impulse2, impulse, atol=1e-20)
    np.testing.assert_allclose(
        work2, work + jnp.sum(boost[:, None] * impulse, axis=0), atol=1e-20
    )


@pytest.mark.parametrize("exit_first", [False, True])
def test_full_source_budgets_include_external_force_and_work(exit_first):
    grid, gas, u, e, q, pos = moving_setup(exit_first)
    dt = 1e-4
    fn = source(grid, particle_acceleration=GRAVITY)
    r = fn(gas, u, e, q, dt, pos)
    assert bool(r.source.accepted)
    before = list(budget(grid, gas, u, e, q))
    before[2] += np.sum(r.external_impulse, axis=1)
    before[3] += np.sum(r.external_work)
    after = list(
        budget(
            grid,
            r.source.gas,
            r.source.velocity,
            r.source.unresolved_density,
            r.source.liquid,
        )
    )
    after[1] += np.sum(r.exited_mass)
    after[2] += np.sum(r.exited_momentum, axis=1)
    after[3] += np.sum(r.exited_energy)
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b, rtol=3e-13, atol=2e-13)
    assert np.all(np.asarray(r.external_impulse[2]) < 0)
    np.testing.assert_array_equal(r.external_impulse[:2], 0)
    if exit_first:
        residence = (grid.lx - pos[0, 0]) / q.velocity[0, 0]
        expected = (
            0.5
            * (q.mass[0] * q.multiplicity[0] + r.exited_mass[0])
            * GRAVITY[2]
            * residence
        )
        np.testing.assert_allclose(
            r.external_impulse[2, 0], expected, rtol=2e-13, atol=1e-24
        )
        assert residence < dt
        again = fn(
            r.source.gas,
            r.source.velocity,
            r.source.unresolved_density,
            r.source.liquid,
            dt,
            r.position,
        )
        assert bool(again.source.accepted)
        np.testing.assert_array_equal(again.external_impulse[:, 0], 0)
        assert again.external_work[0] == 0
    print(
        "gravity_source",
        exit_first,
        float(jnp.sum(r.external_work)),
        np.sum(r.external_impulse, axis=1),
    )


@pytest.mark.parametrize("rejection", ["source", "carrier"])
def test_rejection_cancels_body_ledgers_and_restores_all_states(rejection):
    grid, gas, u, e, q, pos = moving_setup(True)
    if rejection == "source":
        r = source(grid, particle_acceleration=GRAVITY, max_segments=1)(
            gas, u, e, q, 1e-4, pos
        )
        assert not bool(r.source.accepted)
        returned = (
            r.source.gas,
            r.source.velocity,
            r.source.unresolved_density,
            r.source.liquid,
            r.position,
        )
    else:
        r = coupled(grid, gas, particle_acceleration=GRAVITY, max_iterations=1)(
            gas, u, e, q, 1e-4, pos
        )
        assert bool(r.phase.source_accepted) and not bool(r.phase.accepted)
        returned = (
            r.phase.carrier.gas,
            r.phase.carrier.velocity,
            r.phase.unresolved_density,
            r.phase.liquid,
            r.position,
        )
    assert_tree_equal(returned, (gas, u, e, q, pos))
    np.testing.assert_array_equal(r.external_impulse, 0)
    np.testing.assert_array_equal(r.external_work, 0)
    np.testing.assert_array_equal(r.exited_mass, 0)


def test_carrier_preserves_accepted_external_ledgers():
    grid, gas, u, e, q, pos = moving_setup(True)
    raw = source(grid, particle_acceleration=GRAVITY)(gas, u, e, q, 1e-4, pos)
    r = coupled(grid, gas, particle_acceleration=GRAVITY)(gas, u, e, q, 1e-4, pos)
    assert bool(r.phase.accepted), r.phase.carrier[15:]
    assert_tree_equal(
        (r.external_impulse, r.external_work), (raw.external_impulse, raw.external_work)
    )
    print(
        "gravity_carrier",
        int(r.phase.carrier.iterations),
        float(r.phase.carrier.momentum_error),
    )
