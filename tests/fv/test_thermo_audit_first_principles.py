"""Independent continuum and constant-enthalpy checks for the spray audit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from jaxwind.domain import UniformGrid
from jaxwind.physics.moisture import MoistureConfig, advance_water_droplet
from jaxwind.scalar_transport import transport_scalars
from jaxwind.state import StaggeredVelocity

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("diameter", [100e-6, 300e-6, 500e-6])
def test_warm_droplet_converges_to_independent_mass_energy_ode(diameter):
    """Verify the mass derivative term in d(m cp T)/dt independently."""
    config = MoistureConfig(pressure=101325, dry_air_density=1.125)
    initial_mass = np.pi * config.water_density * diameter**3 / 6
    gas_t, gas_q, slip = 312.35, 0.0047, 19.0
    cp, latent, reference = 4182.0, 2.5e6, 273.15

    def rhs(_, state):
        mass, temperature = state
        size = (6 * mass / (np.pi * 997)) ** (1 / 3)
        reynolds = 1.125 * slip * size / 1.9e-5
        nusselt = 2 + 0.6 * np.sqrt(reynolds) * (1005 * 1.9e-5 / 0.0265) ** (1 / 3)
        sherwood = 2 + 0.6 * np.sqrt(reynolds) * (1.9e-5 / (1.125 * 2.5e-5)) ** (1 / 3)
        log_t = np.log(temperature)
        vapor_pressure = np.exp(
            54.842763
            - 6763.22 / temperature
            - 4.210 * log_t
            + 0.000367 * temperature
            + np.tanh(0.0415 * (temperature - 218.8))
            * (
                53.878
                - 1331.22 / temperature
                - 9.44523 * log_t
                + 0.014025 * temperature
            )
        )
        saturation = (287.05 / 461.5) * vapor_pressure / (101325 - vapor_pressure)
        evaporation = (
            np.pi
            * size
            * 1.125
            * 2.5e-5
            * sherwood
            * max(np.log1p(saturation) - np.log1p(gas_q), 0)
        )
        heat = np.pi * size * 0.0265 * nusselt * (gas_t - temperature)
        return [
            -evaporation,
            (heat - (latent - cp * (temperature - reference)) * evaporation)
            / (mass * cp),
        ]

    independent = solve_ivp(
        rhs, (0, 0.3), [initial_mass, 308.35], atol=[1e-20, 1e-9], rtol=2e-10
    ).y[:, -1]
    errors = []
    for dt in [1.25e-4, 6.25e-5]:

        def integrate(dt=dt):
            def step(_, state):
                result = advance_water_droplet(*state, gas_t, gas_q, slip, dt, config)
                return result.mass, result.temperature

            return jax.lax.fori_loop(
                0,
                round(0.3 / dt),
                step,
                (jnp.asarray(initial_mass), jnp.asarray(308.35)),
            )

        mass, temperature = jax.jit(integrate)()
        errors.append(abs(float(mass) - independent[0]))
        assert abs(float(temperature) - independent[1]) < 0.0021
        assert abs(float(mass) / independent[0] - 1) < 5e-6
    assert errors[1] < 0.55 * errors[0]


def test_transport_preserves_uniform_dilute_enthalpy_through_moisture_gradient():
    """Independent T and q limiting must not manufacture a hot/humid mode."""
    grid = UniformGrid(32, 2, 2, 1.0, 0.2, 0.2)
    shape = (2, 2, 32)
    x = np.asarray(grid.x_centers)
    humidity = np.broadcast_to(
        0.005 + 0.008 * np.exp(-(((x - 0.35) / 0.13) ** 2)), shape
    )
    enthalpy = 1005 * 312.35 + 2.5e6 * 0.005
    temperature = (enthalpy - 2.5e6 * humidity) / 1005
    fields = jnp.asarray(np.stack((temperature, humidity)))
    velocity = StaggeredVelocity(
        jnp.ones(shape), jnp.zeros(shape), jnp.zeros((3, 2, 32))
    )
    ambient = jnp.asarray(np.stack((temperature[..., 0], humidity[..., 0])))
    result = jax.jit(
        lambda f: transport_scalars(f, velocity, grid, 0.05, ambient, 2e-4)
    )(fields)
    np.testing.assert_allclose(
        1005 * result[0] + 2.5e6 * result[1], enthalpy, rtol=0, atol=3e-10
    )
