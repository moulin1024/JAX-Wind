"""Analytic limits, conservative exchange and independent ODE verification."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from jaxwind.physics.moisture import (
    MoistureConfig,
    WaterDropletProperties,
    advance_water_droplet,
    saturation_mixing_ratio,
)
from jaxwind.water_spray import advance_water_droplet_motion

jax.config.update("jax_enable_x64", True)
CONFIG = MoistureConfig(dry_air_density=1.125, pressure=101325)
PROPS = WaterDropletProperties()
DIAMETER = 293e-6
MASS = CONFIG.water_density * np.pi * DIAMETER**3 / 6


def test_no_evaporation_has_analytic_implicit_heat_relaxation():
    t, tg, dt = 290.0, 310.0, 0.1
    update = advance_water_droplet(jnp.asarray(MASS), t, tg, 0.1, 0.0, dt, CONFIG)
    conductance = 2 * np.pi * DIAMETER * PROPS.air_thermal_conductivity
    expected = (MASS * PROPS.liquid_heat_capacity * t + dt * conductance * tg) / (
        MASS * PROPS.liquid_heat_capacity + dt * conductance
    )
    assert float(update.mass) == MASS
    assert float(update.temperature) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_mass_and_enthalpy_exchange_close(dtype):
    t = jnp.asarray(308.35, dtype)
    m = jnp.asarray(MASS, dtype)
    update = jax.jit(
        lambda m, t: advance_water_droplet(m, t, 312.35, 0.005, 19.0, 0.002, CONFIG)
    )(m, t)
    liquid_before = m * PROPS.liquid_heat_capacity * (t - CONFIG.freezing_temperature)
    liquid_after = (
        update.mass
        * PROPS.liquid_heat_capacity
        * (update.temperature - CONFIG.freezing_temperature)
    )
    gas_change = (
        CONFIG.water_vapor_latent_heat * update.evaporated_mass
        - update.gas_sensible_energy_loss
    )
    np.testing.assert_allclose(update.mass + update.evaporated_mass, m, rtol=2e-7)
    np.testing.assert_allclose(liquid_after + gas_change, liquid_before, rtol=2e-7)
    assert 273.15 < float(update.temperature) < float(t)
    assert float(update.gas_sensible_energy_loss) > 0


def test_surface_saturation_uses_liquid_temperature():
    # Air is unsaturated at 312 K but saturated relative to the colder drop.
    q = saturation_mixing_ratio(jnp.asarray(290.0), CONFIG.pressure, CONFIG)
    update = advance_water_droplet(
        jnp.asarray(MASS), 290.0, 312.0, q, 0.0, 1e-8, CONFIG
    )
    assert float(update.evaporated_mass / MASS) < 1e-12


def test_zero_step_empty_and_saturated_equilibrium():
    q = saturation_mixing_ratio(jnp.asarray(300.0), CONFIG.pressure, CONFIG)
    for m, dt in [(MASS, 0.0), (0.0, 0.01), (MASS, 0.01)]:
        update = advance_water_droplet(jnp.asarray(m), 300.0, 300.0, q, 0.0, dt, CONFIG)
        assert float(update.mass) == pytest.approx(m, abs=1e-20)
        assert float(update.temperature) == pytest.approx(300, abs=1e-9)
        assert float(update.gas_sensible_energy_loss) == pytest.approx(0, abs=1e-12)


def test_warm_energy_limit_does_not_overevaporate():
    update = advance_water_droplet(
        jnp.asarray(MASS), 274.0, 274.0, 0.0, 0.0, 1e5, CONFIG
    )
    assert 0 <= float(update.mass) <= MASS
    assert float(update.temperature) >= CONFIG.freezing_temperature
    assert np.isfinite(float(update.gas_sensible_energy_loss))


def test_temperature_and_mass_converge_to_independent_ode():
    # Independent continuous equations integrated with adaptive DOP853. This
    # detects wrong enthalpy sign, surface temperature, or mass/diameter coupling.
    tg, qv, speed = 312.35, 0.005, 19.0
    cp, lv = PROPS.liquid_heat_capacity, CONFIG.water_vapor_latent_heat

    def rhs(_, state):
        m, t = state
        d = (6 * m / (np.pi * CONFIG.water_density)) ** (1 / 3)
        re = CONFIG.dry_air_density * speed * d / PROPS.air_dynamic_viscosity
        pr = (
            CONFIG.dry_air_heat_capacity
            * PROPS.air_dynamic_viscosity
            / PROPS.air_thermal_conductivity
        )
        sc = PROPS.air_dynamic_viscosity / (
            CONFIG.dry_air_density * CONFIG.vapor_diffusivity
        )
        nu, sh = 2 + 0.6 * re**0.5 * pr ** (1 / 3), 2 + 0.6 * re**0.5 * sc ** (1 / 3)
        # Murphy--Koop water saturation, independently evaluated with NumPy.
        logp = (
            54.842763
            - 6763.22 / t
            - 4.210 * np.log(t)
            + 0.000367 * t
            + np.tanh(0.0415 * (t - 218.8))
            * (53.878 - 1331.22 / t - 9.44523 * np.log(t) + 0.014025 * t)
        )
        p = np.exp(logp)
        qs = (
            CONFIG.dry_air_gas_constant
            / CONFIG.water_vapor_gas_constant
            * p
            / (CONFIG.pressure - p)
        )
        evaporation = (
            np.pi
            * d
            * CONFIG.dry_air_density
            * CONFIG.vapor_diffusivity
            * sh
            * max(np.log1p(qs) - np.log1p(qv), 0)
        )
        heat = np.pi * d * PROPS.air_thermal_conductivity * nu * (tg - t)
        return [
            -evaporation,
            (heat - (lv - cp * (t - CONFIG.freezing_temperature)) * evaporation)
            / (m * cp),
        ]

    target = solve_ivp(
        rhs, (0, 0.1), [MASS, 308.35], method="DOP853", rtol=1e-11, atol=[1e-20, 1e-10]
    ).y[:, -1]
    errors = []
    for count in [100, 200, 400]:

        def step(_, state):
            result = advance_water_droplet(*state, tg, qv, speed, 0.1 / count, CONFIG)
            return result.mass, result.temperature

        m, t = jax.jit(
            lambda: jax.lax.fori_loop(
                0, count, step, (jnp.asarray(MASS), jnp.asarray(308.35))
            )
        )()
        errors.append(
            np.linalg.norm([(float(m) - target[0]) / MASS, (float(t) - target[1]) / 20])
        )
    assert errors[1] < 0.6 * errors[0]
    assert errors[2] < 0.6 * errors[1]
    assert errors[2] < 0.001


def test_stokes_drag_motion_and_impulse_are_analytic():
    d, speed, dt = 1e-5, 0.001, 0.001
    velocity = jnp.asarray([speed, 0.0, 0.0])
    gas = jnp.zeros(3)
    updated, distance, impulse = advance_water_droplet_motion(
        velocity, gas, d, dt, CONFIG, PROPS, gravity=(0, 0, 0)
    )
    rate = 18 * PROPS.air_dynamic_viscosity / (CONFIG.water_density * d * d)
    expected = speed * np.exp(-rate * dt)
    assert float(updated[0]) == pytest.approx(expected)
    assert float(distance[0]) == pytest.approx((speed - expected) / rate)
    np.testing.assert_allclose(updated + impulse, velocity, atol=1e-15)


def test_terminal_settling_transfers_weight_to_gas():
    d, dt = 1e-5, 0.001
    rate = 18 * PROPS.air_dynamic_viscosity / (CONFIG.water_density * d * d)
    gravity = -9.81 * (1 - CONFIG.dry_air_density / CONFIG.water_density)
    v = jnp.asarray([0.0, 0.0, gravity / rate])
    updated, _, impulse = advance_water_droplet_motion(
        v, jnp.zeros(3), d, dt, CONFIG, PROPS
    )
    np.testing.assert_allclose(updated, v, atol=1e-14)
    assert float(impulse[2]) == pytest.approx(gravity * dt)
