"""Persistent DRW/analytic trajectory checks; not physical dispersion validation."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.fluent_dpm_dispersion import advance_drw, initial_drw

jax.config.update('jax_enable_x64', True)


def track(x, v, eddy, dt, **kwargs):
    return advance_drw(x, v, eddy, jnp.array([8.0, 0.0, 0.0]), 0.6, 0.3, 0.5, 0.05, dt, **kwargs)


def test_step_partition_does_not_resample_or_change_constant_coefficient_path():
    x, v, e = jnp.array([1.0, 8.0, 2.0]), jnp.array([12.0, 0.0, 0.0]), initial_drw(731)
    full = jax.jit(lambda x, v, e: track(x, v, e, 2.0, random_lifetime=True))(x, v, e)
    half = jax.jit(lambda x, v, e: track(x, v, e, 0.1, random_lifetime=True))
    for _ in range(20):
        result = half(x, v, e)
        assert bool(result.accepted)
        x, v, e = result.position, result.velocity, result.eddy
    assert bool(full.accepted) and int(e.draws) > 1
    np.testing.assert_allclose(x, full.position, atol=2e-13, rtol=0)
    np.testing.assert_allclose(v, full.velocity, atol=2e-13, rtol=0)
    np.testing.assert_allclose(e.remaining, full.eddy.remaining, atol=3e-14, rtol=0)
    np.testing.assert_array_equal(e.key, full.eddy.key)
    assert int(e.draws) == int(full.eddy.draws)


def test_no_renewal_before_eddy_expires_and_zero_step_preserves_rng():
    x = jnp.zeros(3)
    e = initial_drw(42)._replace(remaining=jnp.asarray(1.0), fluctuation=jnp.array([0.2, -0.1, 0.3]))
    zero = track(x, x, e, 0.0)
    for a, b in zip(jax.tree.leaves(e), jax.tree.leaves(zero.eddy)):
        np.testing.assert_array_equal(a, b)
    short = track(x, x, e, 0.01)
    np.testing.assert_array_equal(short.eddy.fluctuation, e.fluctuation)
    np.testing.assert_array_equal(short.eddy.key, e.key)
    assert int(short.eddy.draws) == 0
    np.testing.assert_allclose(short.eddy.remaining, 0.99)


def test_laminar_settling_matches_analytic_solution_and_turbulence_can_restart():
    x, v, e = jnp.zeros(3), jnp.array([12.0, 0.0, 0.0]), initial_drw(15)
    dt, tau = 2.0, 0.1
    u, a = jnp.array([8.0, 0.0, 0.0]), jnp.array([0.0, 0.0, -9.81])
    result = advance_drw(x, v, e, u, 0.0, 0.0, 4.0, tau, dt, acceleration=a)
    eq = u+tau*a
    np.testing.assert_allclose(result.velocity, eq+(v-eq)*np.exp(-dt/tau), atol=1e-14)
    np.testing.assert_allclose(result.position, eq*dt+(v-eq)*tau*(1-np.exp(-dt/tau)), atol=1e-14)
    assert int(result.eddy.draws) == 0
    restarted = track(result.position, result.velocity, result.eddy, 0.01)
    assert bool(restarted.accepted) and int(restarted.eddy.draws) > 0


def test_event_overflow_rolls_back_motion_and_rng():
    x, v, e = jnp.zeros(3), jnp.zeros(3), initial_drw(5)
    result = track(x, v, e, 10.0, max_intervals=1)
    assert not bool(result.accepted)
    for a, b in zip(jax.tree.leaves((x, v, e)), jax.tree.leaves((result.position, result.velocity, result.eddy))):
        np.testing.assert_array_equal(a, b)
