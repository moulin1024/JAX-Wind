"""Repeated prescribed injection, slot reuse and global water accounting."""

import jax.numpy as jnp
import numpy as np
from test_spray_injection import C, coupled, injection_setup

from jaxwind.spray_injection import births_from_mass_flow


def test_continuous_injection_recycles_slots_without_losing_flow():
    grid, gas, velocity, unresolved, liquid, position = injection_setup()
    liquid = liquid._replace(multiplicity=jnp.zeros_like(liquid.mass))
    dt, flow, steps = 1e-4, 1e-4, 12
    births = births_from_mass_flow(
        flow,
        dt,
        jnp.array([70e-6]),
        jnp.array([1.0]),
        jnp.array([[18.0], [0.0], [0.0]]),
        jnp.array([300.0]),
        jnp.array([[0.5995], [0.15], [0.15]]),
        jnp.array([0.5]),
        C,
    )
    advance = coupled(grid, gas)
    volume = grid.dx * grid.dy * grid.dz
    initial_vapor = float(jnp.sum(gas.vapor_density) * volume)
    injected = exported = vapor_boundary = 0.0
    max_residual = 0.0
    for index in range(steps):
        result = advance(gas, velocity, unresolved, liquid, dt, position, births)
        moving = result.moving
        assert bool(moving.phase.accepted), moving.phase.carrier[15:]
        np.testing.assert_array_equal(result.slots, [0])
        assert int(result.available_slots) == liquid.mass.size
        carrier = moving.phase.carrier
        gas, velocity = carrier.gas, carrier.velocity
        unresolved, liquid = moving.phase.unresolved_density, moving.phase.liquid
        position = moving.position
        np.testing.assert_array_equal(liquid.multiplicity, 0)
        injected += float(result.injected_mass)
        exported += float(jnp.sum(moving.exited_mass))
        vapor_boundary += float(
            dt
            * grid.dy
            * grid.dz
            * jnp.sum(
                carrier.scalar_fluxes.x[1, ..., -1] - carrier.scalar_fluxes.x[1, ..., 0]
            )
        )
        np.testing.assert_allclose(injected, (index + 1) * flow * dt, rtol=3e-15)
        residual = (
            float(jnp.sum(gas.vapor_density) * volume)
            - initial_vapor
            + exported
            + vapor_boundary
            - injected
        )
        max_residual = max(max_residual, abs(residual))
        assert abs(residual) < 2e-16
    assert 0 < exported < injected
    print("continuous_injection", steps, injected, exported, max_residual)
