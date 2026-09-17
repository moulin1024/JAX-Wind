"""Equation/trajectory verification, not measurements or an ANSYS solver run."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from jaxwind.physics.fluent_dpm import (
    DPMWaterMaterial,
    advance_droplet,
    droplet_rates,
    gas_properties,
    les_dispersion_scales,
    liquid_enthalpy,
    sample_drw_eddy,
    vapor_enthalpy,
)
from jaxwind.physics.moisture import saturation_vapor_pressure_water

jax.config.update("jax_enable_x64", True)
P = DPMWaterMaterial()
PRESSURE = 101325.0


def mass(d):
    return np.pi/6*P.liquid_density*d**3


def saturated_y(t):
    x = float(saturation_vapor_pressure_water(jnp.asarray(t)))/PRESSURE
    eps = P.dry_air_gas_constant/P.vapor_gas_constant
    return eps*x/(1-(1-eps)*x)


def test_default_diffusion_law_uses_distinct_surface_and_bulk_temperatures():
    d, tp, tg, y = 80e-6, 295.0, 315.0, 0.005
    r = droplet_rates(mass(d), tp, tg, y, PRESSURE, 0.0)
    eps = P.dry_air_gas_constant/P.vapor_gas_constant
    pv = PRESSURE*y/(eps+(1-eps)*y)
    concentration = float(saturation_vapor_pressure_water(jnp.asarray(tp)))/(P.vapor_gas_constant*tp)
    concentration -= pv/(P.vapor_gas_constant*tg)
    expected = 2*np.pi*d*P.binary_diffusivity*concentration
    np.testing.assert_allclose(r.evaporation, expected, rtol=2e-14)
    assert bool(r.valid)
    assert float(r.nusselt) == 2 and float(r.sherwood) == 2


@pytest.mark.parametrize("model", ["diffusion-controlled", "convection-diffusion-controlled"])
def test_isothermal_saturation_is_equilibrium_and_supersaturation_does_not_condense(model):
    args = (mass(100e-6), 300.0, 300.0)
    r = droplet_rates(*args, saturated_y(300.0), PRESSURE, 0.0, vaporization=model)
    assert abs(float(r.evaporation)) < 1e-23
    r = droplet_rates(*args, 2*saturated_y(300.0), PRESSURE, 0.0, vaporization=model)
    assert float(r.evaporation) == 0 and float(r.convective_heat) == 0


def test_stefan_flow_changes_mass_and_heat_consistently():
    d, t, tg = 100e-6, 330.0, 340.0
    r = droplet_rates(mass(d), t, tg, 0.0, PRESSURE, 0.0,
                      vaporization="convection-diffusion-controlled")
    rho, _ = gas_properties(tg, 0, PRESSURE, P)
    bm = saturated_y(t)/(1-saturated_y(t))
    np.testing.assert_allclose(r.evaporation, 2*np.pi*d*rho*P.binary_diffusivity*np.log1p(bm), rtol=2e-14)
    np.testing.assert_allclose(r.nusselt, 2*np.log1p(bm)/bm, rtol=2e-14)
    assert float(r.nusselt) < 2


def test_inert_heating_matches_closed_form_and_conserves_enthalpy():
    m, tp, tg, dt = mass(100e-6), 290.0, 320.0, 0.05
    conductance = 2*np.pi*100e-6*P.conductivity
    exact = tg+(tp-tg)*np.exp(-conductance*dt/(m*P.liquid_cp))
    result = jax.jit(lambda: advance_droplet(m, tp, tg, 0.01, PRESSURE, 0, dt,
                                             evaporation_enabled=False))()
    assert bool(result.accepted)
    np.testing.assert_allclose(result.temperature, exact, atol=2e-7)
    assert float(result.vapor_mass) == 0
    np.testing.assert_allclose(result.gas_enthalpy_gain+m*liquid_enthalpy(result.temperature, P),
                               m*liquid_enthalpy(tp, P), rtol=3e-15)


@pytest.mark.parametrize("model", ["diffusion-controlled", "convection-diffusion-controlled"])
def test_coupled_trajectory_matches_independent_mass_temperature_ode(model):
    m, tp, tg, y, slip, dt = mass(150e-6), 307.0, 315.0, 0.006, 2.0, 0.15
    # Independent state variables (m,T), DOP853 integration and algebra.
    def ode(_time, state):
        fraction, t = state
        mm = fraction*m
        d = (6*mm/(np.pi*P.liquid_density))**(1/3)
        eps = P.dry_air_gas_constant/P.vapor_gas_constant
        rho = PRESSURE/(((1-y)*P.dry_air_gas_constant+y*P.vapor_gas_constant)*tg)
        re = rho*slip*d/P.viscosity
        cp = (1-y)*P.dry_air_cp+y*P.vapor_cp
        nu = 2+0.6*np.sqrt(re)*(cp*P.viscosity/P.conductivity)**(1/3)
        sh = 2+0.6*np.sqrt(re)*(P.viscosity/(rho*P.binary_diffusivity))**(1/3)
        ps = float(saturation_vapor_pressure_water(jnp.asarray(t)))
        x = y/(eps+(1-eps)*y)
        if model == "diffusion-controlled":
            md = np.pi*d*P.binary_diffusivity*sh*max(ps/(P.vapor_gas_constant*t)-x*PRESSURE/(P.vapor_gas_constant*tg), 0)
        else:
            xs = ps/PRESSURE
            ys = eps*xs/(1-(1-eps)*xs)
            bm = max((ys-y)/(1-ys), 0)
            md = np.pi*d*rho*P.binary_diffusivity*sh*np.log1p(bm)
            nu *= np.log1p(bm)/bm if bm else 1
        heat = np.pi*d*P.conductivity*nu*(tg-t)
        latent = P.latent_heat_reference+(P.vapor_cp-P.liquid_cp)*(t-P.reference_temperature)
        return [-md/m, (heat-md*latent)/(mm*P.liquid_cp)]
    ref = solve_ivp(ode, (0, dt), (1, tp), method="DOP853", rtol=1e-11, atol=1e-12)
    result = jax.jit(lambda: advance_droplet(m, tp, tg, y, PRESSURE, slip, dt,
                                             vaporization=model))()
    assert ref.success and bool(result.accepted)
    np.testing.assert_allclose(result.mass/m, ref.y[0, -1], rtol=3e-7)
    np.testing.assert_allclose(result.temperature, ref.y[1, -1], atol=3e-5)
    np.testing.assert_allclose(result.mass+result.vapor_mass, m, rtol=1e-15)
    np.testing.assert_allclose(result.gas_enthalpy_gain+result.mass*liquid_enthalpy(result.temperature, P),
                               m*liquid_enthalpy(tp, P), atol=1e-17)


def test_latent_heat_and_species_sensible_energy_use_one_reference():
    for t in (280.0, 300.0, 350.0):
        r = droplet_rates(mass(100e-6), t, 355.0, 0.001, PRESSURE, 1.0)
        lhs = mass(100e-6)*P.liquid_cp*r.temperature_rate-r.evaporation*liquid_enthalpy(t, P)
        np.testing.assert_allclose(lhs, r.liquid_enthalpy_rate, rtol=1e-14)
        np.testing.assert_allclose(r.liquid_enthalpy_rate+r.evaporation*vapor_enthalpy(t, P),
                                   r.convective_heat, rtol=1e-14,
                                   atol=8*np.finfo(float).eps*(abs(float(r.liquid_enthalpy_rate))
                                       + abs(float(r.evaporation*vapor_enthalpy(t, P)))))


def test_failed_integration_rolls_back_state_and_exchange():
    result = advance_droplet(mass(10e-6), 310.0, 340.0, 0.0, PRESSURE, 3.0, 1.0, max_steps=1)
    assert not bool(result.accepted)
    assert float(result.mass) == mass(10e-6) and float(result.temperature) == 310
    assert float(result.vapor_mass) == 0 and float(result.gas_enthalpy_gain) == 0


def test_zero_step_is_identity():
    m = mass(50e-6)
    r = advance_droplet(m, 300.0, 315.0, 0.005, PRESSURE, 0.0, 0.0)
    assert bool(r.accepted) and int(r.attempts) == 0
    assert float(r.mass) == m and float(r.temperature) == 300


@pytest.mark.parametrize("temperature", [270.0, 373.15, float('nan')])
def test_outside_warm_nonboiling_scope_is_rejected(temperature):
    assert not bool(droplet_rates(mass(50e-6), temperature, 300.0, 0.0, PRESSURE, 0).valid)


def test_disappearance_transfers_residual_mass_and_energy_once():
    m = mass(20e-6)
    r = jax.jit(lambda: advance_droplet(m, 310.0, 320.0, 0.001, PRESSURE, 0, 1.0))()
    assert bool(r.accepted) and float(r.mass) == 0
    assert float(r.vapor_mass) == m and 0 < float(r.terminal_mass) < 1e-8*m
    np.testing.assert_allclose(r.gas_enthalpy_gain, m*liquid_enthalpy(310.0, P), rtol=1e-14)


def test_drw_lifetime_crossing_and_zero_turbulence():
    normal = jnp.array([1.0, -2.0, 0.5])
    velocity, lifetime = sample_drw_eddy(normal, 0.5, 1.5, 0.3, 0.2, 0.0, 1.0)
    np.testing.assert_array_equal(velocity, normal)
    np.testing.assert_allclose(lifetime, 1.5)
    _, random = sample_drw_eddy(normal, 0.5, 1.5, 0.3, 0.2, 0.0, 1.0, random_lifetime=True)
    np.testing.assert_allclose(random, -0.75*np.log(0.5))
    _, crossing = sample_drw_eddy(normal, 0.5, 1.5, 0.3, 0.2, 10.0, 0.1)
    np.testing.assert_allclose(crossing, -0.2*np.log1p(-0.05))
    zero, duration = sample_drw_eddy(normal, 0.5, 0.0, 0.0, 0.2, 10.0, 0.1)
    np.testing.assert_array_equal(zero, 0)
    assert np.isinf(float(duration))


def test_drw_ensemble_has_specified_variance_and_isotropy():
    draws = jax.random.normal(jax.random.PRNGKey(17), (3, 131072))
    velocity, _ = sample_drw_eddy(draws, 0.5, 0.6, 0.1, 0.1, 0.0, 1.0)
    covariance = np.cov(np.asarray(velocity))
    np.testing.assert_allclose(covariance, np.eye(3)*0.4, atol=0.005)
    assert np.max(abs(np.asarray(velocity).mean(axis=1))) < 0.005


def test_les_mapping_requires_explicit_scale_and_has_correct_units():
    k, epsilon = les_dispersion_scales(2.0, length=4.0)
    assert float(k) == 4 and float(epsilon) == 2
    _, second = les_dispersion_scales(2.0, time=2.0)
    assert float(second) == 2
    with pytest.raises(ValueError):
        les_dispersion_scales(2.0)
    with pytest.raises(ValueError):
        les_dispersion_scales(2.0, length=1.0, time=1.0)


def test_material_validation():
    with pytest.raises(ValueError):
        replace(P, binary_diffusivity=-1)
