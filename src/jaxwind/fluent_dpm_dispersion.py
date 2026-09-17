"""Persistent, event-resolved isotropic Fluent-reference DRW tracking.

Local constant gas/drag/turbulence inputs per call. This module handles eddy
renewal and analytic particle motion; spatial interpolation, cell boundaries,
evaporation and equal/opposite source deposition belong to the caller.
Public Fluent 2026 R1 Theory Guide, sections 12.2.2 and 12.2.3.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .physics.fluent_dpm import sample_drw_eddy


class DRWState(NamedTuple):
    fluctuation: jax.Array
    remaining: jax.Array
    key: jax.Array
    draws: jax.Array


class DRWTrack(NamedTuple):
    position: jax.Array
    velocity: jax.Array
    eddy: DRWState
    accepted: jax.Array
    intervals: jax.Array


def initial_drw(seed, dtype='float64'):
    return DRWState(jnp.zeros(3, dtype), jnp.asarray(0.0, dtype),
                    jax.random.PRNGKey(seed), jnp.asarray(0, jnp.int32))


def advance_drw(position, velocity, eddy, gas_velocity, k, epsilon, eddy_length,
                relaxation_time, dt, *, acceleration=(0.0, 0.0, 0.0),
                random_lifetime=False, integral_time_constant=0.15, max_intervals=4096):
    """Advance one drop with frozen positive relaxation time, renewing at events.

    The caller supplies LES-only k, epsilon and eddy length. Resolved wake
    fluctuations are already in gas_velocity. Eddies and PRNG state survive CFD
    step boundaries and restarts. No new draw occurs at dt=0 or k=0. Turning
    turbulence off clears its residual; turning it back on starts a new eddy.
    Crossing uses instantaneous gas-minus-particle speed at renewal. No source
    term or turbulent-energy conservation is implied by this motion kernel.
    """
    if type(max_intervals) is not int or max_intervals < 1:
        raise ValueError('positive DRW event capacity required')
    position, velocity, gas_velocity = map(jnp.asarray, (position, velocity, gas_velocity))
    acceleration = jnp.asarray(acceleration, position.dtype)
    if any(a.shape != (3,) for a in (position, velocity, gas_velocity, acceleration, eddy.fluctuation)):
        raise ValueError('DRW position and velocities must have shape (3,)')
    valid = (jnp.isfinite(dt) & (dt >= 0) & jnp.isfinite(k) & (k >= 0)
             & jnp.isfinite(epsilon) & (epsilon >= 0)
             & ((k == 0) | (epsilon > 0)) & jnp.isfinite(eddy_length) & (eddy_length > 0)
             & jnp.isfinite(relaxation_time) & (relaxation_time > 0)
             & ~jnp.isnan(eddy.remaining) & (eddy.remaining >= 0))
    valid &= jnp.all(jnp.isfinite(jnp.stack((position, velocity, gas_velocity, acceleration, eddy.fluctuation))))

    def body(state):
        elapsed, x, v, e, intervals = state
        def renew(e):
            key, nk, uk = jax.random.split(e.key, 3)
            normal = jax.random.normal(nk, (3,), dtype=position.dtype)
            uniform = jax.random.uniform(uk, (), dtype=position.dtype,
                                        minval=jnp.finfo(position.dtype).eps, maxval=1.0)
            fluctuation = normal*jnp.sqrt(2*k/3)
            slip = jnp.linalg.norm(gas_velocity+fluctuation-v)
            fluctuation, duration = sample_drw_eddy(normal, uniform, k, epsilon,
                relaxation_time, slip, eddy_length, random_lifetime=random_lifetime,
                integral_time_constant=integral_time_constant)
            return DRWState(fluctuation, duration, key, e.draws+1)

        e = jax.lax.cond(k > 0,
            lambda e: jax.lax.cond((e.remaining <= 0) | jnp.isinf(e.remaining), renew, lambda e: e, e),
            lambda e: e._replace(fluctuation=jnp.zeros_like(e.fluctuation), remaining=jnp.asarray(jnp.inf, position.dtype)), e)
        h = jnp.minimum(dt-elapsed, e.remaining)
        u = gas_velocity+e.fluctuation
        response = -jnp.expm1(-h/relaxation_time)
        equilibrium = u+relaxation_time*acceleration
        dx = equilibrium*h+(v-equilibrium)*relaxation_time*response
        new_v = v+(equilibrium-v)*response
        return (elapsed+h, x+dx, new_v, e._replace(remaining=jnp.maximum(e.remaining-h, 0)), intervals+1)

    def condition(state):
        return valid & (state[0] < dt) & (state[-1] < max_intervals)

    elapsed, x, v, e, intervals = jax.lax.while_loop(condition, body,
        (jnp.asarray(0.0, position.dtype), position, velocity, eddy, jnp.asarray(0, jnp.int32)))
    accepted = valid & (elapsed >= dt) & jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(v))
    choose = lambda new, old: jnp.where(accepted, new, old)
    return DRWTrack(choose(x, position), choose(v, velocity), jax.tree.map(choose, e, eddy), accepted, intervals)
