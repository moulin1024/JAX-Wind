from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


JAX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JAX_ROOT))


def _params(jnp, **overrides):
    from wireles_jax import Params

    values = dict(
        nx=4,
        ny=4,
        nz=4,
        lx=1.0,
        ly=1.0,
        lz=1.0,
        z_i=100.0,
        dt=0.1 / 100.0,
        nsteps=1,
        initial_condition="geostrophic",
        momentum_wall_model="free_slip",
        initial_velocity_noise=0.0,
        thermo_enabled=True,
        moisture_enabled=True,
        theta0=300.0,
        qv0=0.0,
        dtype=jnp.float32,
        use_jit=False,
    )
    values.update(overrides)
    return Params(**values)


def test_evaporated_liquid_is_conservatively_deposited_as_vapor() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initial_state, initialize_spray
    from wireles_jax.spray_dpm import spray_exchange

    params = _params(jnp)
    flow = initial_state(params)
    config = SprayDPMConfig(
        max_parcels=1,
        initial_parcels=1,
        parcel_weight=1.0e8,
        injection_x=37.0,
        injection_y=41.0,
        injection_z=50.0,
        initial_diameter=100.0e-6,
        initial_temperature=285.0,
        substeps=2,
        sky_temperature=285.0,
    )
    spray = initialize_spray(config, dtype=params.dtype)
    initial_mass = np.asarray(
        config.parcel_weight
        * (np.pi / 6.0)
        * config.liquid_density
        * config.initial_diameter**3
    )

    updated, increments, diagnostics = spray_exchange(flow, spray, params, config)
    final_mass = config.parcel_weight * float(np.asarray(updated.mass[0]))
    cell_volume = (
        params.dx * params.z_i
        * params.dy * params.z_i
        * params.dz * params.z_i
    )
    deposited_mass = (
        float(np.asarray(increments.qv).sum())
        * config.air_density
        * cell_volume
    )
    z_centers = (
        np.arange(params.nz, dtype=np.float64) + 0.5
    ) * params.dz * params.z_i
    kappa = config.dry_air_gas_constant / config.air_heat_capacity
    surface_exner = (params.surface_pressure / 100000.0) ** kappa
    exner = surface_exner - params.g * z_centers / (
        config.air_heat_capacity * params.theta0
    )
    deposited_air_energy_loss = -float(
        np.sum(
            np.asarray(increments.theta)
            * exner[None, None, :]
            * config.air_density
            * cell_volume
            * config.air_heat_capacity
        )
    )

    assert float(np.asarray(diagnostics.evaporated_mass)) > 0.0
    np.testing.assert_allclose(
        initial_mass - final_mass,
        float(np.asarray(diagnostics.evaporated_mass)),
        rtol=2.0e-5,
    )
    np.testing.assert_allclose(
        deposited_mass,
        float(np.asarray(diagnostics.evaporated_mass)),
        rtol=2.0e-5,
    )
    np.testing.assert_allclose(
        deposited_air_energy_loss,
        float(np.asarray(diagnostics.air_energy_loss)),
        rtol=2.0e-5,
    )


def test_drag_exchange_has_equal_and_opposite_streamwise_impulse() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initial_state, initialize_spray
    from wireles_jax.spray_dpm import saturation_vapor_pressure, spray_exchange

    temperature = 300.0
    vapor_pressure = float(np.asarray(saturation_vapor_pressure(jnp.asarray(temperature))))
    epsilon = 287.05 / 461.5
    saturated_qv = epsilon * vapor_pressure / (100000.0 - vapor_pressure)
    params = _params(jnp, geostrophic_u=10.0, qv0=saturated_qv)
    flow = initial_state(params)
    config = SprayDPMConfig(
        max_parcels=1,
        initial_parcels=1,
        parcel_weight=2.0e6,
        injection_x=50.0,
        injection_y=50.0,
        injection_z=1.0,
        injection_u=0.0,
        injection_w=1.0,
        initial_diameter=500.0e-6,
        initial_temperature=temperature,
        substeps=1,
        sky_temperature=temperature,
    )
    spray = initialize_spray(config, dtype=params.dtype)
    updated, increments, _ = spray_exchange(flow, spray, params, config)
    drop_mass = (
        (np.pi / 6.0) * config.liquid_density * config.initial_diameter**3
    )
    parcel_impulse = (
        config.parcel_weight
        * drop_mass
        * float(np.asarray(updated.u[0] - spray.u[0]))
    )
    cell_volume = (
        params.dx * params.z_i
        * params.dy * params.z_i
        * params.dz * params.z_i
    )
    gas_impulse = (
        float(np.asarray(increments.u).sum())
        * config.air_density
        * cell_volume
    )
    np.testing.assert_allclose(gas_impulse, -parcel_impulse, rtol=2.0e-5)
    gravity_delta_w = -params.g * (
        1.0 - config.air_density / config.liquid_density
    ) * params.dt_physical
    parcel_drag_impulse_w = (
        config.parcel_weight
        * drop_mass
        * float(np.asarray(updated.w[0] - spray.w[0] - gravity_delta_w))
    )
    gas_impulse_w = (
        float(np.asarray(increments.w).sum())
        * config.air_density
        * cell_volume
    )
    np.testing.assert_allclose(
        gas_impulse_w, -parcel_drag_impulse_w, rtol=3.0e-5, atol=1.0e-12
    )


def test_shortwave_heats_saturated_drop_without_directly_heating_air() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initial_state, initialize_spray
    from wireles_jax.spray_dpm import saturation_vapor_pressure, spray_exchange

    temperature = 300.0
    vapor_pressure = float(np.asarray(saturation_vapor_pressure(jnp.asarray(temperature))))
    epsilon = 287.05 / 461.5
    saturated_qv = epsilon * vapor_pressure / (100000.0 - vapor_pressure)
    params = _params(jnp, qv0=saturated_qv)
    flow = initial_state(params)
    config = SprayDPMConfig(
        max_parcels=1,
        initial_parcels=1,
        parcel_weight=1.0,
        injection_x=50.0,
        injection_y=50.0,
        injection_z=1.0,
        initial_diameter=1.0e-3,
        initial_temperature=temperature,
        substeps=1,
        shortwave_flux=800.0,
        shortwave_absorption_efficiency=1.0,
        sky_temperature=temperature,
    )
    spray = initialize_spray(config, dtype=params.dtype)
    updated, increments, diagnostics = spray_exchange(flow, spray, params, config)

    assert float(np.asarray(updated.temperature[0])) > temperature
    assert float(np.asarray(diagnostics.net_radiative_energy)) > 0.0
    np.testing.assert_allclose(np.asarray(increments.theta), 0.0, atol=1.0e-12)


def test_spray_requires_active_thermodynamics_and_moisture() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initial_state, initialize_spray
    from wireles_jax.spray_dpm import spray_exchange

    params = _params(jnp, moisture_enabled=False)
    flow = initial_state(params)
    config = SprayDPMConfig(max_parcels=1, initial_parcels=1)
    spray = initialize_spray(config, dtype=params.dtype)
    with pytest.raises(ValueError, match="thermo_enabled and moisture_enabled"):
        spray_exchange(flow, spray, params, config)


def test_sgs_velocity_seen_ou_has_target_one_step_variance_and_is_reproducible() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initialize_spray
    from wireles_jax.spray_dpm import _advance_sgs_velocity_seen

    count = 8192
    config = SprayDPMConfig(
        max_parcels=count,
        initial_parcels=count,
        turbulent_dispersion_enabled=True,
        random_seed=27,
    )
    spray = initialize_spray(config, dtype=jnp.float32)
    variances = (
        jnp.full((count,), 0.25, dtype=jnp.float32),
        jnp.full((count,), 0.16, dtype=jnp.float32),
        jnp.full((count,), 0.09, dtype=jnp.float32),
    )
    time_scale = jnp.full((count,), 0.1, dtype=jnp.float32)
    updated = _advance_sgs_velocity_seen(
        spray,
        *variances,
        time_scale,
        0.1,
        jnp.asarray(13, dtype=jnp.uint32),
        config,
    )
    repeated = _advance_sgs_velocity_seen(
        spray,
        *variances,
        time_scale,
        0.1,
        jnp.asarray(13, dtype=jnp.uint32),
        config,
    )
    expected_factor = 1.0 - np.exp(-2.0)

    np.testing.assert_array_equal(np.asarray(updated.sgs_u), np.asarray(repeated.sgs_u))
    for field, variance in zip(
        (updated.sgs_u, updated.sgs_v, updated.sgs_w),
        (0.25, 0.16, 0.09),
        strict=True,
    ):
        values = np.asarray(field)
        assert values.mean() == pytest.approx(0.0, abs=0.012)
        assert values.var() == pytest.approx(
            variance * expected_factor, rel=0.05
        )


def test_eddy_crossing_time_limits_only_fast_slip() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initialize_spray
    from wireles_jax.spray_dpm import _eddy_crossing_limited_time_scale

    params = _params(jnp)
    config = SprayDPMConfig(max_parcels=2, initial_parcels=2)
    spray = initialize_spray(config, dtype=params.dtype)._replace(
        u=jnp.asarray((0.0, 100.0), dtype=params.dtype)
    )
    gas = jnp.zeros((2,), dtype=params.dtype)
    base_time = jnp.full((2,), 10.0, dtype=params.dtype)

    limited = _eddy_crossing_limited_time_scale(
        spray, gas, gas, gas, base_time, params
    )
    delta = params.sgs_delta * params.z_i

    assert float(limited[0]) == pytest.approx(10.0)
    assert float(limited[1]) == pytest.approx(delta / 100.0)


def test_eddy_crossing_time_includes_correlated_sgs_velocity_seen() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initialize_spray
    from wireles_jax.spray_dpm import _eddy_crossing_limited_time_scale

    params = _params(jnp)
    config = SprayDPMConfig(max_parcels=1, initial_parcels=1)
    spray = initialize_spray(config, dtype=params.dtype)._replace(
        sgs_v=jnp.asarray((20.0,), dtype=params.dtype)
    )
    zero = jnp.zeros((1,), dtype=params.dtype)
    limited = _eddy_crossing_limited_time_scale(
        spray,
        zero,
        zero,
        zero,
        jnp.asarray((10.0,), dtype=params.dtype),
        params,
    )

    assert float(limited[0]) == pytest.approx(
        params.sgs_delta * params.z_i / 20.0
    )


def test_stiff_drag_uses_bounded_exact_relaxation() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initialize_spray
    from wireles_jax.spray_dpm import _ParcelRates, _advance_parcel_implicit

    params = _params(jnp)
    config = SprayDPMConfig(max_parcels=1, initial_parcels=1)
    spray = initialize_spray(config, dtype=params.dtype)
    one = jnp.ones((1,), dtype=params.dtype)
    zero = jnp.zeros_like(one)
    rates = _ParcelRates(
        drag_rate=1.0e5 * one,
        mass_transfer_coefficient=zero,
        pressure=1.0e5 * one,
        ambient_mass_fraction=zero,
        heat_conductance=zero,
        gas_temperature=300.0 * one,
        radiative_power=zero,
        reynolds=zero,
        relative_humidity=zero,
    )
    advanced = _advance_parcel_implicit(
        spray,
        15.0 * one,
        zero,
        zero,
        rates,
        zero,
        0.1,
        params,
        config,
    )

    assert float(advanced.u[0]) == pytest.approx(15.0)
    assert 0.0 <= float(advanced.u[0]) <= 15.0
    assert float(advanced.drag_delta_u[0]) == pytest.approx(15.0)
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(advanced))))


def test_semi_implicit_drop_temperature_preserves_exchange_energy() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initialize_spray
    from wireles_jax.spray_dpm import _ParcelRates, _advance_parcel_implicit

    params = _params(jnp)
    config = SprayDPMConfig(
        max_parcels=1,
        initial_parcels=1,
        initial_temperature=270.0,
    )
    spray = initialize_spray(config, dtype=params.dtype)
    one = jnp.ones((1,), dtype=params.dtype)
    zero = jnp.zeros_like(one)
    rates = _ParcelRates(
        drag_rate=zero,
        mass_transfer_coefficient=zero,
        pressure=1.0e5 * one,
        ambient_mass_fraction=zero,
        heat_conductance=2.0e-6 * one,
        gas_temperature=300.0 * one,
        radiative_power=zero,
        reynolds=zero,
        relative_humidity=zero,
    )
    advanced = _advance_parcel_implicit(
        spray, zero, zero, zero, rates, zero, 1.0, params, config
    )
    heat_capacity = spray.mass * config.liquid_heat_capacity

    assert 270.0 < float(advanced.temperature[0]) < 300.0
    np.testing.assert_allclose(
        np.asarray(heat_capacity * (advanced.temperature - spray.temperature)),
        np.asarray(advanced.convective_energy),
        rtol=2.0e-5,
    )


def test_cold_drop_condenses_vapor_conservatively() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initial_state, initialize_spray
    from wireles_jax.spray_dpm import spray_exchange

    params = _params(jnp, qv0=0.01, dt=0.01 / 100.0)
    flow = initial_state(params)
    config = SprayDPMConfig(
        max_parcels=1,
        initial_parcels=1,
        parcel_weight=1.0e8,
        injection_x=50.0,
        injection_y=50.0,
        injection_z=50.0,
        initial_diameter=100.0e-6,
        initial_temperature=250.0,
        liquid_emissivity=0.0,
        substeps=1,
    )
    spray = initialize_spray(config, dtype=params.dtype)
    initial_liquid = float(config.parcel_weight * spray.mass[0])
    updated, increments, diagnostics = spray_exchange(
        flow, spray, params, config
    )
    final_liquid = float(config.parcel_weight * updated.mass[0])
    cell_mass = (
        config.air_density
        * params.dx
        * params.z_i
        * params.dy
        * params.z_i
        * params.dz
        * params.z_i
    )
    vapor_change = float(jnp.sum(increments.qv) * cell_mass)

    assert final_liquid > initial_liquid
    assert float(diagnostics.evaporated_mass) < 0.0
    assert vapor_change < 0.0
    assert final_liquid - initial_liquid == pytest.approx(
        -vapor_change, rel=3.0e-5
    )


def test_condensation_cannot_consume_qv_floor() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initial_state, initialize_spray
    from wireles_jax.spray_dpm import spray_exchange

    params = _params(
        jnp,
        qv0=0.001,
        qv_floor=0.001,
        dt=0.01 / 100.0,
    )
    flow = initial_state(params)
    config = SprayDPMConfig(
        max_parcels=1,
        initial_parcels=1,
        injection_z=50.0,
        initial_temperature=240.0,
        liquid_emissivity=0.0,
        substeps=1,
    )
    spray = initialize_spray(config, dtype=params.dtype)
    updated, increments, diagnostics = spray_exchange(
        flow, spray, params, config
    )

    np.testing.assert_allclose(
        np.asarray(updated.mass), np.asarray(spray.mass), rtol=0.0, atol=0.0
    )
    assert float(jnp.sum(increments.qv)) == pytest.approx(0.0, abs=1.0e-12)
    assert float(diagnostics.evaporated_mass) == pytest.approx(0.0, abs=1.0e-12)


def test_condensation_limiter_is_local_to_parcel_cic_support() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initial_state, initialize_spray
    from wireles_jax.spray_dpm import spray_exchange

    params = _params(
        jnp,
        qv0=0.001,
        qv_floor=0.001,
        dt=0.01 / 100.0,
    )
    flow = initial_state(params)._replace(
        qv=initial_state(params).qv.at[2, :, :].set(0.01)
    )
    config = SprayDPMConfig(
        max_parcels=2,
        initial_parcels=2,
        initial_temperature=240.0,
        injection_z=12.5,
        liquid_emissivity=0.0,
        substeps=1,
    )
    spray = initialize_spray(config, dtype=params.dtype)._replace(
        x=jnp.asarray((0.0, 50.0), dtype=params.dtype),
        y=jnp.zeros((2,), dtype=params.dtype),
    )
    updated, _, _ = spray_exchange(flow, spray, params, config)

    assert float(updated.mass[0]) == pytest.approx(float(spray.mass[0]))
    assert float(updated.mass[1]) > float(spray.mass[1])


def test_seawater_initialization_preserves_nonvolatile_salt() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initialize_spray
    from wireles_jax.spray_dpm import (
        _ParcelRates,
        _advance_parcel_implicit,
    )

    params = _params(jnp)
    config = SprayDPMConfig(
        max_parcels=1,
        initial_parcels=1,
        initial_diameter=200.0e-6,
        liquid_density=1025.0,
        salinity_mass_fraction=0.035,
    )
    spray = initialize_spray(config, dtype=params.dtype)
    water_mass = spray.mass - spray.solute_mass
    zero = jnp.zeros((1,), dtype=params.dtype)
    rates = _ParcelRates(
        drag_rate=zero,
        mass_transfer_coefficient=zero,
        pressure=jnp.full((1,), 1.0e5, dtype=params.dtype),
        ambient_mass_fraction=zero,
        heat_conductance=zero,
        gas_temperature=spray.temperature,
        radiative_power=zero,
        reynolds=zero,
        relative_humidity=zero,
    )
    advanced = _advance_parcel_implicit(
        spray,
        zero,
        zero,
        zero,
        rates,
        water_mass,
        1.0,
        params,
        config,
    )

    assert float(spray.solute_mass[0] / spray.mass[0]) == pytest.approx(0.035)
    assert float(advanced.mass[0]) == pytest.approx(
        float(spray.solute_mass[0]), rel=2.0e-6
    )
    assert float(advanced.diameter[0]) > 0.0


def test_seawater_water_activity_predicts_75_percent_rh_equilibrium() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initialize_spray
    from wireles_jax.spray_dpm import (
        _ParcelRates,
        _phase_change_rate_at_temperature,
        saturation_vapor_pressure,
    )

    temperature = 273.15 + 18.0
    relative_humidity = 0.75
    pressure = 1.0e5
    config = SprayDPMConfig(
        max_parcels=1,
        initial_parcels=1,
        initial_diameter=200.0e-6,
        liquid_density=1025.0,
        salinity_mass_fraction=0.035,
        osmotic_coefficient_model="constant",
    )
    spray = initialize_spray(config, dtype=jnp.float32)
    initial_volume = (jnp.pi / 6.0) * spray.diameter**3
    salt = spray.solute_mass
    radius = 50.0e-6
    kelvin = jnp.exp(
        2.0
        * config.water_molar_mass
        * config.surface_tension
        / (
            config.universal_gas_constant
            * config.water_density
            * radius
            * temperature
        )
    )
    target_activity = relative_humidity / kelvin
    equilibrium_water = (
        config.salt_vant_hoff_factor
        * config.salt_osmotic_coefficient
        * salt
        * config.water_molar_mass
        / (config.salt_molar_mass * -jnp.log(target_activity))
    )
    removed_water = spray.mass - salt - equilibrium_water
    equilibrium_volume = initial_volume - removed_water / config.water_density
    equilibrium_diameter = (6.0 * equilibrium_volume / jnp.pi) ** (1.0 / 3.0)
    spray = spray._replace(
        mass=salt + equilibrium_water,
        diameter=equilibrium_diameter,
        temperature=jnp.full((1,), temperature, dtype=jnp.float32),
    )
    saturation = saturation_vapor_pressure(jnp.asarray(temperature))
    vapor_pressure = relative_humidity * saturation
    mixing_ratio = (
        config.dry_air_gas_constant
        / config.water_vapor_gas_constant
        * vapor_pressure
        / (pressure - vapor_pressure)
    )
    rates = _ParcelRates(
        drag_rate=jnp.zeros((1,), dtype=jnp.float32),
        mass_transfer_coefficient=jnp.ones((1,), dtype=jnp.float32),
        pressure=jnp.full((1,), pressure, dtype=jnp.float32),
        ambient_mass_fraction=jnp.full(
            (1,), mixing_ratio / (1.0 + mixing_ratio), dtype=jnp.float32
        ),
        heat_conductance=jnp.zeros((1,), dtype=jnp.float32),
        gas_temperature=jnp.full((1,), temperature, dtype=jnp.float32),
        radiative_power=jnp.zeros((1,), dtype=jnp.float32),
        reynolds=jnp.zeros((1,), dtype=jnp.float32),
        relative_humidity=jnp.full(
            (1,), relative_humidity, dtype=jnp.float32
        ),
    )
    rate = _phase_change_rate_at_temperature(
        spray, spray.temperature, rates, config
    )

    assert abs(float(rate[0])) < 2.0e-4
    assert 40.0 < 0.5e6 * float(equilibrium_diameter[0]) < 55.0


def test_andreas_osmotic_coefficient_uses_measured_molality_curve() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig
    from wireles_jax.spray_dpm import _salt_osmotic_coefficient

    config = SprayDPMConfig()
    molality = jnp.asarray((0.0, 0.62, 6.0, 8.0), dtype=jnp.float32)
    coefficient = _salt_osmotic_coefficient(molality, config)

    np.testing.assert_allclose(
        np.asarray(coefficient),
        np.asarray((0.9270, 0.9256217, 1.2724896, 1.2724896)),
        rtol=2.0e-6,
    )


def test_veron2020_transfer_rhs_matches_published_equations() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initialize_spray
    from wireles_jax.spray_dpm import (
        _parcel_rates_from_samples,
        _phase_change_rate_at_temperature,
        saturation_vapor_pressure,
    )

    params = _params(jnp)
    air_temperature = 291.15
    drop_temperature = 293.15
    relative_humidity = 0.75
    pressure = 1.0e5
    config = SprayDPMConfig(
        max_parcels=1,
        initial_parcels=1,
        injection_z=0.0,
        initial_diameter=200.0e-6,
        initial_temperature=drop_temperature,
        liquid_density=1025.0,
        salinity_mass_fraction=0.035,
        thermodynamic_transfer_model="veron2020",
        liquid_emissivity=0.0,
    )
    spray = initialize_spray(config, dtype=jnp.float32)
    saturation = float(saturation_vapor_pressure(jnp.asarray(air_temperature)))
    vapor_pressure = relative_humidity * saturation
    epsilon = config.dry_air_gas_constant / config.water_vapor_gas_constant
    qv = epsilon * vapor_pressure / (pressure - vapor_pressure)
    rates = _parcel_rates_from_samples(
        spray,
        jnp.asarray((15.0,), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.asarray((air_temperature,), dtype=jnp.float32),
        jnp.asarray((qv,), dtype=jnp.float32),
        params,
        config,
    )

    radius = 100.0e-6
    reynolds = (
        config.air_density
        * 15.0
        * (2.0 * radius)
        / config.air_dynamic_viscosity
    )
    ventilation = 1.0 + np.sqrt(reynolds) / 4.0
    vapor_denominator = (
        radius / (radius + config.vapor_jump_length)
        + config.vapor_diffusivity
        / (radius * config.vapor_accommodation_coefficient)
        * np.sqrt(
            2.0
            * np.pi
            * config.water_molar_mass
            / (config.universal_gas_constant * air_temperature)
        )
    )
    modified_diffusivity = (
        ventilation * config.vapor_diffusivity / vapor_denominator
    )
    expected_mass_coefficient = (
        4.0
        * np.pi
        * radius
        * modified_diffusivity
        * config.water_molar_mass
        * saturation
        / (config.universal_gas_constant * air_temperature)
    )
    thermal_denominator = (
        radius / (radius + config.thermal_jump_length)
        + config.air_thermal_conductivity
        / (
            radius
            * config.thermal_accommodation_coefficient
            * config.air_density
            * config.air_heat_capacity
        )
        * np.sqrt(
            2.0
            * np.pi
            * config.dry_air_molar_mass
            / (config.universal_gas_constant * air_temperature)
        )
    )
    expected_heat_conductance = (
        4.0
        * np.pi
        * radius
        * ventilation
        * config.air_thermal_conductivity
        / thermal_denominator
    )
    water_mass = float(spray.mass[0] - spray.solute_mass[0])
    molality = float(spray.solute_mass[0]) / (
        config.salt_molar_mass * water_mass
    )
    phi = (
        0.9270
        - 2.164e-2 * molality
        + 3.486e-2 * molality**2
        - 5.956e-3 * molality**3
        + 3.911e-4 * molality**4
    )
    solute = (
        config.salt_vant_hoff_factor
        * phi
        * molality
        * config.water_molar_mass
    )
    kelvin = (
        2.0
        * config.water_molar_mass
        * config.surface_tension
        / (
            config.universal_gas_constant
            * config.water_density
            * radius
            * drop_temperature
        )
    )
    thermal = (
        config.latent_heat
        * config.water_molar_mass
        / config.universal_gas_constant
        * (1.0 / air_temperature - 1.0 / drop_temperature)
    )
    surface_rh = (
        air_temperature
        / drop_temperature
        * np.exp(thermal + kelvin - solute)
    )
    expected_evaporation_rate = expected_mass_coefficient * (
        surface_rh - relative_humidity
    )
    evaporation_rate = _phase_change_rate_at_temperature(
        spray, spray.temperature, rates, config
    )

    assert float(rates.reynolds[0]) == pytest.approx(reynolds, rel=2.0e-6)
    assert float(rates.relative_humidity[0]) == pytest.approx(
        relative_humidity, rel=2.0e-6
    )
    assert float(rates.mass_transfer_coefficient[0]) == pytest.approx(
        expected_mass_coefficient, rel=3.0e-6
    )
    assert float(rates.heat_conductance[0]) == pytest.approx(
        expected_heat_conductance, rel=3.0e-6
    )
    assert float(evaporation_rate[0]) == pytest.approx(
        expected_evaporation_rate, rel=5.0e-6
    )


def test_terminal_settling_drag_is_a_separate_figure1_semantic() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initialize_spray
    from wireles_jax.spray_dpm import _parcel_rates_from_samples

    params = _params(jnp)
    config = SprayDPMConfig(
        max_parcels=1,
        initial_parcels=1,
        injection_z=0.0,
        initial_diameter=200.0e-6,
        liquid_density=1025.0,
        drag_correction_model="terminal_settling",
        ventilation_correction_enabled=False,
        thermodynamic_transfer_model="veron2020",
    )
    spray = initialize_spray(config, dtype=jnp.float32)
    zero = jnp.zeros((1,), dtype=jnp.float32)
    rates = _parcel_rates_from_samples(
        spray,
        jnp.asarray((15.0,), dtype=jnp.float32),
        zero,
        zero,
        jnp.asarray((291.15,), dtype=jnp.float32),
        zero,
        params,
        config,
    )

    diameter = config.initial_diameter
    stokes_time = (
        config.liquid_density
        * diameter**2
        / (18.0 * config.air_dynamic_viscosity)
    )
    gravity = params.g * (1.0 - config.air_density / config.liquid_density)
    settling_speed = gravity * stokes_time
    for _ in range(8):
        reynolds = (
            config.air_density
            * settling_speed
            * diameter
            / config.air_dynamic_viscosity
        )
        turbulent_drag = 0.42 / (
            1.0 + 42500.0 / max(reynolds, 1.0e-12) ** 1.16
        )
        correction = (
            1.0
            + 0.15 * max(reynolds, 1.0e-12) ** 0.687
            + turbulent_drag * reynolds / 24.0
        )
        settling_speed = gravity * stokes_time / correction
    reynolds = (
        config.air_density
        * settling_speed
        * diameter
        / config.air_dynamic_viscosity
    )
    turbulent_drag = 0.42 / (
        1.0 + 42500.0 / max(reynolds, 1.0e-12) ** 1.16
    )
    correction = (
        1.0
        + 0.15 * max(reynolds, 1.0e-12) ** 0.687
        + turbulent_drag * reynolds / 24.0
    )
    expected_rate = correction / stokes_time

    assert float(rates.drag_rate[0]) == pytest.approx(expected_rate, rel=3.0e-6)
    assert 0.07 < 1.0 / float(rates.drag_rate[0]) < 0.08


def test_test_filter_sgs_energy_vanishes_for_uniform_flow() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax.spray_dpm import _sgs_velocity_statistics

    params = _params(jnp, nx=8, ny=8, nz=8)
    shape = (params.nx, params.ny, params.nz)
    uniform = jnp.full(shape, 8.0, dtype=params.dtype)
    zero = jnp.zeros(shape, dtype=params.dtype)
    variance_u, variance_v, variance_w, time_scale = _sgs_velocity_statistics(
        uniform, zero, zero, params
    )

    assert float(jnp.max(variance_u)) == pytest.approx(0.0, abs=1.0e-7)
    assert float(jnp.max(variance_v)) == pytest.approx(0.0, abs=1.0e-7)
    assert float(jnp.max(variance_w)) == pytest.approx(0.0, abs=1.0e-7)
    assert bool(jnp.all(jnp.isfinite(time_scale)))


def test_turbulent_dispersion_changes_drag_reproducibly() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from dataclasses import replace

    from wireles_jax import SprayDPMConfig, initial_state, initialize_spray
    from wireles_jax.spray_dpm import spray_exchange

    params = _params(jnp, nx=8, ny=8, nz=8, dt=0.05 / 100.0)
    flow = initial_state(params)
    x = jnp.arange(params.nx, dtype=params.dtype)[:, None, None]
    resolved_u = 3.0 * jnp.sin(2.0 * jnp.pi * x / params.nx)
    flow = flow._replace(u=jnp.broadcast_to(resolved_u, flow.u.shape))
    config = SprayDPMConfig(
        max_parcels=128,
        initial_parcels=128,
        injection_x=50.0,
        injection_y=50.0,
        injection_z=50.0,
        injection_radius=35.0,
        initial_diameter=200.0e-6,
        initial_temperature=290.0,
        sky_temperature=290.0,
        substeps=1,
        turbulent_dispersion_enabled=True,
        random_seed=41,
    )
    spray = initialize_spray(config, dtype=params.dtype, seed=3)
    stochastic, _, _ = spray_exchange(flow, spray, params, config)
    repeated, _, _ = spray_exchange(flow, spray, params, config)
    deterministic, _, _ = spray_exchange(
        flow,
        spray,
        params,
        replace(config, turbulent_dispersion_enabled=False),
    )

    assert float(
        jnp.max(
            jnp.sqrt(
                stochastic.sgs_u**2
                + stochastic.sgs_v**2
                + stochastic.sgs_w**2
            )
        )
    ) > 0.0
    np.testing.assert_array_equal(
        np.asarray(stochastic.sgs_u), np.asarray(repeated.sgs_u)
    )
    assert not np.array_equal(
        np.asarray(stochastic.u), np.asarray(deterministic.u)
    )


def test_continuous_injection_matches_configured_mass_flow() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initialize_spray
    from wireles_jax.spray_dpm import inject_spray

    config = SprayDPMConfig(
        max_parcels=4,
        initial_parcels=0,
        parcels_per_step=2,
        mass_flow_rate=0.2,
        initial_diameter=1.0e-3,
        diameter_distribution="rosin_rammler",
        minimum_diameter=0.2e-3,
        maximum_diameter=2.0e-3,
        rosin_rammler_spread=2.5,
    )
    spray = initialize_spray(config, dtype=jnp.float32)
    injected = inject_spray(spray, jnp.asarray(0), 0.1, config)
    represented_mass = np.asarray(
        jnp.sum(injected.mass * injected.weight * injected.active)
    )
    assert int(np.asarray(injected.active).sum()) == 2
    assert np.unique(np.asarray(injected.parcel_id[injected.active])).size == 2
    np.testing.assert_allclose(represented_mass, 0.2 * 0.1, rtol=2.0e-6)
    assert np.unique(np.asarray(injected.diameter[injected.active])).size == 2


def test_streamwise_nozzle_uses_transverse_disk_and_finite_thickness() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initialize_spray

    config = SprayDPMConfig(
        max_parcels=4096,
        initial_parcels=4096,
        injection_x=1.0,
        injection_y=1.0,
        injection_z=1.0,
        injection_radius=0.15,
        injection_streamwise_thickness=0.0625,
        injection_u=8.0,
    )

    spray = initialize_spray(config, dtype=jnp.float32, seed=73)
    x = np.asarray(spray.x)
    y = np.asarray(spray.y)
    z = np.asarray(spray.z)
    transverse_radius = np.sqrt((y - 1.0) ** 2 + (z - 1.0) ** 2)

    assert np.max(np.abs(x - 1.0)) <= 0.03125 + 1.0e-7
    assert np.std(x) > 0.005
    assert np.std(y) > 0.05
    assert np.std(z) > 0.05
    assert np.max(transverse_radius) <= 0.15 + 2.0e-7


def test_continuous_injection_uses_half_cosine_startup_ramp() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, initialize_spray
    from wireles_jax.spray_dpm import inject_spray

    config = SprayDPMConfig(
        max_parcels=4,
        parcels_per_step=2,
        mass_flow_rate=0.2,
        injection_ramp_time=0.2,
        initial_diameter=1.0e-3,
    )
    initial = initialize_spray(config, dtype=jnp.float32)
    startup = inject_spray(initial, jnp.asarray(0), 0.1, config)
    established = inject_spray(initial, jnp.asarray(2), 0.1, config)

    startup_mass = float(
        jnp.sum(startup.mass * startup.weight * startup.active)
    )
    established_mass = float(
        jnp.sum(established.mass * established.weight * established.active)
    )
    expected_ramp = 0.5 * (1.0 - np.cos(np.pi * 0.25))
    assert startup_mass == pytest.approx(
        config.mass_flow_rate * 0.1 * expected_ramp,
        rel=2.0e-6,
    )
    assert established_mass == pytest.approx(
        config.mass_flow_rate * 0.1,
        rel=2.0e-6,
    )


def test_cic_deposition_has_eight_point_support_and_preserves_total() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.spray_dpm import (
        _cic_coordinates,
        _cic_deposit,
    )

    params = Params(
        nx=8,
        ny=8,
        nz=8,
        lx=1.0,
        ly=1.0,
        lz=1.0,
        z_i=1.0,
        dtype=jnp.float32,
    )
    x = jnp.asarray((0.43,), dtype=params.dtype)
    y = jnp.asarray((0.46,), dtype=params.dtype)
    z = jnp.asarray((0.49,), dtype=params.dtype)
    cic_coordinates = _cic_coordinates(x, y, z, params)
    deposited = np.asarray(
        _cic_deposit(
            jnp.asarray((3.5,), dtype=params.dtype),
            cic_coordinates,
            (params.nx, params.ny, params.nz),
            params.dtype,
        )
    )

    assert np.count_nonzero(deposited) == 2 * 2 * 2
    assert deposited.sum() == pytest.approx(3.5, rel=2.0e-6)


def test_rosin_rammler_sampler_matches_truncated_mass_cdf() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, sample_diameters

    config = SprayDPMConfig(
        initial_diameter=100.0e-6,
        diameter_distribution="rosin_rammler",
        minimum_diameter=20.0e-6,
        maximum_diameter=250.0e-6,
        rosin_rammler_spread=3.0,
    )
    samples = np.asarray(
        sample_diameters(config, jax.random.PRNGKey(91), 20000, jnp.float32)
    )
    scale = config.initial_diameter
    spread = config.rosin_rammler_spread
    fmin = 1.0 - np.exp(-((config.minimum_diameter / scale) ** spread))
    fmax = 1.0 - np.exp(-((config.maximum_diameter / scale) ** spread))
    median_probability = 0.5 * (fmin + fmax)
    expected_median = scale * (-np.log1p(-median_probability)) ** (1.0 / spread)
    assert samples.min() >= config.minimum_diameter
    assert samples.max() <= config.maximum_diameter
    np.testing.assert_allclose(np.median(samples), expected_median, rtol=1.5e-2)


def test_tabulated_sampler_uses_mass_fractions() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import SprayDPMConfig, sample_diameters

    config = SprayDPMConfig(
        diameter_distribution="tabulated",
        tabulated_diameters=(50.0e-6, 100.0e-6, 200.0e-6),
        tabulated_mass_fractions=(0.2, 0.3, 0.5),
    )
    samples = np.asarray(
        sample_diameters(config, jax.random.PRNGKey(17), 30000, jnp.float32)
    )
    fractions = np.asarray([(samples == value).mean() for value in config.tabulated_diameters])
    np.testing.assert_allclose(fractions, [0.2, 0.3, 0.5], atol=1.0e-2)
