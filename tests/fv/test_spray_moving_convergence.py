"""Fixed-duration timestep convergence for finite-inertia cell-crossing motion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_spray_cell_step import budget
from test_spray_moving import moving_setup, source


@pytest.mark.parametrize("acceleration", [(0.0, 0.0, 0.0), (0.0, 0.0, -9.81)])
def test_fixed_duration_trajectory_refinement_with_conservative_exchange(acceleration):
    grid, gas, velocity, unresolved, liquid, position = moving_setup()
    update = source(grid, particle_acceleration=acceleration)
    duration = 8e-4
    initial = (
        gas,
        velocity,
        unresolved,
        liquid,
        position,
        jnp.asarray(True),
        jnp.zeros(3),
        jnp.asarray(0.0),
    )

    @jax.jit
    def integrate(steps):
        dt = duration / steps

        def body(_, state):
            gas, velocity, unresolved, liquid, position, ok, impulse, work = state
            result = update(gas, velocity, unresolved, liquid, dt, position)
            s = result.source
            return (
                s.gas,
                s.velocity,
                s.unresolved_density,
                s.liquid,
                result.position,
                ok & s.accepted,
                impulse + jnp.sum(result.external_impulse, axis=1),
                work + jnp.sum(result.external_work),
            )

        return jax.lax.fori_loop(0, steps, body, initial)

    results = [integrate(jnp.asarray(n)) for n in (8, 16, 32, 64, 512, 1024)]
    before = budget(grid, gas, velocity, unresolved, liquid)
    for n, result in zip((8, 16, 32, 64, 512, 1024), results):
        g, u, e, q, pos, ok, impulse, work = result
        assert bool(ok), n
        assert np.all(np.asarray(pos[0]) > np.asarray(position[0]))
        expected = list(before)
        expected[2] = expected[2] + np.asarray(impulse)
        expected[3] = expected[3] + float(work)
        for a, b in zip(expected, budget(grid, g, u, e, q)):
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)
    reference = np.asarray(results[-1][4])
    errors = np.array(
        [np.linalg.norm(np.asarray(r[4]) - reference) for r in results[:4]]
    )
    ratios = errors[:-1] / errors[1:]
    reference_difference = np.linalg.norm(np.asarray(results[-2][4]) - reference)
    assert np.all((ratios > 1.7) & (ratios < 2.4)), (errors, ratios)
    assert reference_difference < 0.1 * errors[-1], (reference_difference, errors)
    print(
        "acceleration",
        acceleration,
        "trajectory_errors_m",
        errors,
        "ratios",
        ratios,
        "reference_difference_m",
        reference_difference,
    )
