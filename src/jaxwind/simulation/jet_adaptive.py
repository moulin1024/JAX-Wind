"""CFL-controlled RK3 for the fully vaporized volume-source jet."""
from typing import NamedTuple

from jaxwind.formulations.jet import CryogenicState

AdaptiveCryogenicState = NamedTuple(
    "AdaptiveCryogenicState",
    [(name, object) for name in CryogenicState._fields]
    + [("last_dt", object), ("last_cfl", object), ("rejected_steps", object)],
)


def build_adaptive_advance(step, case, grid, diffusivity):
    import jax
    import jax.numpy as jnp
    from jaxwind.numerics.discretization import stable_timestep

    def admissible(result):
        return (
            jnp.isfinite(result.last_cfl)
            & (result.last_cfl <= case.cfl * (1.0 + 1.0e-5))
            & jnp.isfinite(jnp.min(result.temperature))
            & jnp.isfinite(jnp.max(result.temperature))
            & jnp.isfinite(jnp.max(result.density))
            & (jnp.min(result.density) > 0.0)
        )

    def run(state, target_time, count):
        target = jnp.asarray(target_time, state.time.dtype)

        def body(_, current):
            remaining = target - current.time

            def advance(original):
                stable = stable_timestep(original.velocity, grid,
                                         diffusivity(original.velocity), courant=case.cfl)
                dt = jnp.minimum(stable, jnp.minimum(case.dt, 1.25 * original.last_dt))
                # Resolve the imposed source ramp even though initial velocity is zero.
                if case.ramp_time > 0.0:
                    dt = jnp.minimum(dt, jnp.where(original.time < case.ramp_time,
                                                   case.ramp_time / 20.0, jnp.inf))
                dt = jnp.minimum(dt, remaining)
                trial = step(original, dt)

                def retry_needed(carry):
                    result, _, attempts = carry
                    return (~admissible(result)) & (attempts < 8)

                def retry(carry):
                    result, old_dt, attempts = carry
                    ratio = jnp.where(jnp.isfinite(result.last_cfl),
                                      .98 * case.cfl / jnp.maximum(result.last_cfl, 1e-12), .5)
                    next_dt = old_dt * jnp.minimum(.95, ratio)
                    return step(original, next_dt), next_dt, attempts + 1

                result, _, attempts = jax.lax.while_loop(
                    retry_needed, retry, (trial, dt, jnp.asarray(0, jnp.int32)),
                )
                # A failed step must reach the runtime's finite-progress check.
                return result._replace(
                    time=jnp.where(admissible(result), result.time, jnp.nan),
                    rejected_steps=original.rejected_steps + attempts,
                )

            return jax.lax.cond(remaining > 0.0, advance, lambda value: value, current)

        return jax.lax.fori_loop(0, count, body, state)

    return jax.jit(run, static_argnums=2)
