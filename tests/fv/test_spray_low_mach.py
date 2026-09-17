"""Coupled EOS, finite-volume conservation, and pressure/source verification."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.domain import UniformGrid
from jaxwind.numerics.discretization import divergence
from jaxwind.numerics.poisson import build_pressure_poisson
from jaxwind.physics.moisture import MoistureConfig, WaterDropletProperties
from jaxwind.spray_core import WaterCoreBins, advance_water_core, core_gas_temperature
from jaxwind.spray_coupling import GasInventory
from jaxwind.spray_low_mach import (
    MoistGasFields,
    build_moist_gas_transport,
    moist_gas_eos_density,
    moist_gas_from_primitive,
    moist_gas_temperature,
)
from jaxwind.state import StaggeredVelocity

jax.config.update("jax_enable_x64", True)
CONFIG = MoistureConfig()


def setup(*, periodic=False, nx=8, dtype="float64"):
    grid = UniformGrid(nx, 4, 4, 1.0, 0.5, 0.5)
    shape = (grid.nz, grid.ny, grid.nx)
    gas = moist_gas_from_primitive(
        jnp.full(shape, 310.0, dtype=dtype), jnp.full(shape, 0.01, dtype=dtype), CONFIG
    )
    velocity = StaggeredVelocity(
        jnp.zeros(shape if periodic else (grid.nz, grid.ny, grid.nx + 1), dtype=dtype),
        jnp.zeros(shape if periodic else (grid.nz, grid.ny + 1, grid.nx), dtype=dtype),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=dtype),
    )
    poisson = build_pressure_poisson(
        grid,
        backend="fft" if periodic else "gmg",
        periodic_x=periodic,
        periodic_y=periodic,
        open_x_low=not periodic,
        dtype=dtype,
        config={} if periodic else {"tolerance": 1e-11 if dtype == "float64" else 1e-6},
    )
    ambient = jnp.stack(gas)[..., 0] / (gas.dry_density + gas.vapor_density)[..., 0]
    step = build_moist_gas_transport(poisson, CONFIG, ambient)
    zeros = jax.tree.map(jnp.zeros_like, gas)
    return grid, gas, velocity, step, zeros


def integrated(gas, grid):
    return np.sum(
        np.asarray(jnp.stack(gas)) * np.asarray(grid.cell_volumes), axis=(1, 2, 3)
    )


def outward(result, grid):
    # Direct face-surface quadrature, independent of the volume-divergence path.
    area = np.asarray(grid.z_widths)[:, None] * np.asarray(grid.y_widths)[None, :]
    return np.sum(
        np.asarray(result.fluxes.x[..., -1] - result.fluxes.x[..., 0]) * area,
        axis=(1, 2),
    )


def assert_eos(result):
    assert bool(result.accepted), (
        result.iterations,
        result.eos_error,
        result.continuity_error,
        result.outgoing_fraction,
    )
    rho = result.gas.dry_density + result.gas.vapor_density
    np.testing.assert_allclose(
        rho, moist_gas_eos_density(result.gas, CONFIG), rtol=2e-9
    )
    assert float(result.continuity_error) < 1e-9
    assert float(jnp.min(result.gas.vapor_density)) >= 0
    assert (
        float(jnp.min(moist_gas_temperature(result.gas, CONFIG)))
        >= CONFIG.freezing_temperature
    )


def assert_shared_flux(result, old, increments, grid, dt):
    rho = result.gas.dry_density + result.gas.vapor_density
    recovered = StaggeredVelocity(
        *(r * u for r, u in zip(result.transport_density, result.velocity))
    )
    for actual, transported in zip(recovered, result.fluxes):
        np.testing.assert_allclose(
            actual, transported[0] + transported[1], rtol=2e-13, atol=2e-14
        )
    residual = (
        (rho - old.dry_density - old.vapor_density) / dt
        + divergence(recovered, grid)
        - (increments.dry_density + increments.vapor_density) / dt
    )
    assert (
        float(jnp.max(jnp.abs(dt * residual / (old.dry_density + old.vapor_density))))
        < 1e-12
    )


def test_primitive_eos_and_core_enthalpy_reference_agree():
    temperature = jnp.array([274.0, 300.0, 330.0])
    gas = moist_gas_from_primitive(temperature, jnp.array([0.0, 0.01, 0.04]), CONFIG)
    np.testing.assert_allclose(
        moist_gas_temperature(gas, CONFIG), temperature, atol=1e-13
    )
    rho = gas.dry_density + gas.vapor_density
    np.testing.assert_allclose(moist_gas_eos_density(gas, CONFIG), rho, rtol=2e-15)
    inventory = GasInventory(
        rho,
        jnp.zeros((3, 3)),
        gas.enthalpy_density,
        jnp.stack((gas.dry_density, gas.vapor_density)),
        jnp.zeros(3),
    )
    np.testing.assert_allclose(
        core_gas_temperature(inventory, CONFIG), temperature, atol=1e-13
    )


@pytest.mark.parametrize("source", ["isothermal_vapor", "heat"])
def test_open_sources_use_one_flux_for_species_enthalpy_and_eos(
    source, record_testsuite_property
):
    grid, gas, velocity, step, zero = setup()
    dt = 0.02
    if source == "isothermal_vapor":
        vapor = jnp.full_like(gas.vapor_density, 0.001)
        increments = zero._replace(
            vapor_density=vapor, enthalpy_density=CONFIG.water_vapor_latent_heat * vapor
        )
        expected_temperature = 310.0
    else:
        increments = zero._replace(
            enthalpy_density=8 * CONFIG.dry_air_heat_capacity * gas.dry_density
        )
        expected_temperature = 318.0
    result = jax.jit(step)(gas, velocity, increments, dt)
    assert_eos(result)
    assert_shared_flux(result, gas, increments, grid, dt)
    record_testsuite_property(source + "_eos_error", float(result.eos_error))
    record_testsuite_property(source + "_iterations", int(result.iterations))
    np.testing.assert_allclose(
        moist_gas_temperature(result.gas, CONFIG), expected_temperature, atol=1e-8
    )
    np.testing.assert_allclose(
        integrated(result.gas, grid),
        integrated(gas, grid)
        + integrated(increments, grid)
        - dt * outward(result, grid),
        rtol=2e-12,
        atol=1e-10,
    )
    assert np.all(np.asarray(result.velocity.x[..., 0]) < 0)
    assert np.all(np.asarray(result.velocity.x[..., -1]) > 0)
    assert float(result.outgoing_fraction) < 1


def test_actual_core_evaporation_cools_and_draws_ambient_with_closed_phase_budget(
    record_testsuite_property,
):
    grid, gas, velocity, step, zero = setup(nx=12)
    index = (2, 2, 5)
    vol = float(np.asarray(grid.cell_volumes)[index])
    mass = (gas.dry_density[index] + gas.vapor_density[index]) * vol
    core = GasInventory(
        mass,
        jnp.zeros(3),
        gas.enthalpy_density[index] * vol,
        jnp.array([gas.dry_density[index], gas.vapor_density[index]]) * vol,
        jnp.array(0.0),
    )
    drop_mass = CONFIG.water_density * jnp.pi * (30e-6) ** 3 / 6
    liquid = WaterCoreBins(
        jnp.array([drop_mass]),
        jnp.array([0.01 * mass / drop_mass]),
        jnp.zeros((3, 1)),
        jnp.array([295.0]),
    )
    dt = 0.002
    exchange = advance_water_core(
        core, liquid, dt, CONFIG, WaterDropletProperties(), drag_heat_fraction=1.0
    )
    assert bool(exchange.accepted) and float(exchange.evaporated_mass) > 0
    increments = MoistGasFields(
        zero.dry_density.at[index].set(
            (exchange.gas.species[0] - core.species[0]) / vol
        ),
        zero.vapor_density.at[index].set(
            (exchange.gas.species[1] - core.species[1]) / vol
        ),
        zero.enthalpy_density.at[index].set(
            (exchange.gas.enthalpy - core.enthalpy) / vol
        ),
    )
    result = jax.jit(step)(gas, velocity, increments, dt)
    assert_eos(result)
    assert_shared_flux(result, gas, increments, grid, dt)
    record_testsuite_property("core_evaporation_eos_error", float(result.eos_error))
    record_testsuite_property("core_evaporation_iterations", int(result.iterations))
    record_testsuite_property(
        "core_evaporated_mass_kg", float(exchange.evaporated_mass)
    )
    record_testsuite_property(
        "core_min_temperature_K",
        float(jnp.min(moist_gas_temperature(result.gas, CONFIG))),
    )
    flux = outward(result, grid)
    record_testsuite_property("core_net_outward_gas_kg_s", float(np.sum(flux[:2])))
    np.testing.assert_allclose(
        integrated(result.gas, grid),
        integrated(gas, grid) + integrated(increments, grid) - dt * flux,
        rtol=2e-12,
        atol=1e-10,
    )
    # Evaporative cooling contracts this gas despite its positive vapor source.
    assert float(np.sum(flux[:2])) < 0
    props = WaterDropletProperties()
    liquid_h = lambda b: float(
        jnp.sum(
            b.mass
            * b.multiplicity
            * props.liquid_heat_capacity
            * (b.temperature - CONFIG.freezing_temperature)
        )
    )
    before = integrated(gas, grid)[2] + liquid_h(liquid)
    after = integrated(result.gas, grid)[2] + liquid_h(exchange.liquid) + dt * flux[2]
    np.testing.assert_allclose(after, before, rtol=2e-12)
    assert float(jnp.min(moist_gas_temperature(result.gas, CONFIG))) < 310.0


def test_periodic_mixture_transport_preserves_all_conserved_fields_without_eos_mass_replacement():
    grid, gas, velocity, step, zero = setup(periodic=True, nx=16)
    x = jnp.asarray(grid.x_centers)
    yv = jnp.broadcast_to(0.015 + 0.01 * jnp.sin(2 * jnp.pi * x), gas.dry_density.shape)
    gas = moist_gas_from_primitive(jnp.full_like(yv, 310.0), yv, CONFIG)
    velocity = velocity._replace(x=jnp.ones_like(velocity.x) * 0.5)
    result = jax.jit(step)(gas, velocity, zero, 0.01)
    assert_eos(result)
    assert_shared_flux(result, gas, zero, grid, 0.01)
    np.testing.assert_allclose(
        integrated(result.gas, grid), integrated(gas, grid), rtol=2e-13, atol=1e-12
    )


@pytest.mark.parametrize("failure", ["closed_heat", "negative_source", "cfl"])
def test_failed_stage_is_transactional_and_never_balances_mass_with_a_hidden_sink(
    failure,
):
    periodic = failure == "closed_heat"
    _grid, gas, velocity, step, zero = setup(periodic=periodic)
    if failure == "closed_heat":
        increments = zero._replace(
            enthalpy_density=jnp.ones_like(gas.dry_density) * 1000
        )
    elif failure == "negative_source":
        increments = zero._replace(dry_density=-2 * gas.dry_density)
    else:
        increments = zero
        velocity = velocity._replace(x=jnp.ones_like(velocity.x) * 100.0)
    result = jax.jit(step)(gas, velocity, increments, 0.1)
    assert not bool(result.accepted)
    for a, b in zip((*result.gas, *result.velocity), (*gas, *velocity)):
        np.testing.assert_array_equal(a, b)
    for f in result.fluxes:
        np.testing.assert_array_equal(f, jnp.zeros_like(f))
    if failure == "closed_heat":
        assert int(result.iterations) == 60
    if failure == "negative_source":
        assert int(result.iterations) == 0
    if failure == "cfl":
        assert float(result.outgoing_fraction) > 1


def test_uniform_isobaric_heating_converges_to_independent_open_reactor_solution(
    record_testsuite_property,
):
    _grid, gas, velocity, step, zero = setup()
    heating = 10000.0  # J/m3/s, zero added mass, constant-pressure outflow.

    @jax.jit
    def run(n):
        dt = 1.0 / n
        increments = zero._replace(
            enthalpy_density=jnp.ones_like(gas.dry_density) * heating * dt
        )

        def advance(_, state):
            g, v, ok = state
            result = step(g, v, increments, dt)
            return result.gas, result.velocity, ok & result.accepted

        return jax.lax.fori_loop(0, n, advance, (gas, velocity, jnp.array(True)))

    # M cp dT/dt=Q V and M R T=p V => T=T0 exp(Q R t/(p cp)).
    y = 0.01
    rmix = (1 - y) * CONFIG.dry_air_gas_constant + y * CONFIG.water_vapor_gas_constant
    cp = (1 - y) * CONFIG.dry_air_heat_capacity
    expected = 310 * np.exp(heating * rmix / (CONFIG.pressure * cp))
    errors = []
    for n in (4, 8, 16):
        final, _, ok = run(n)
        assert bool(ok)
        errors.append(
            abs(float(jnp.mean(moist_gas_temperature(final, CONFIG))) - expected)
        )
    record_testsuite_property("heating_exact_final_temperature_K", expected)
    for n, error in zip((4, 8, 16), errors):
        record_testsuite_property(f"heating_{n}_steps_error_K", error)
    assert min(errors[0] / errors[1], errors[1] / errors[2]) > 1.95


def test_float32_accepts_precision_appropriate_eos_and_preserves_budgets(
    record_testsuite_property,
):
    grid, gas, velocity, step, zero = setup(dtype="float32")
    increments = zero._replace(
        enthalpy_density=2 * CONFIG.dry_air_heat_capacity * gas.dry_density
    )
    result = jax.jit(step)(gas, velocity, increments, 0.02)
    assert bool(result.accepted)
    assert all(
        q.dtype == jnp.float32 for q in (*result.gas, *result.velocity, *result.fluxes)
    )
    assert float(result.eos_error) <= 50 * np.finfo(np.float32).eps
    np.testing.assert_allclose(
        integrated(result.gas, grid),
        integrated(gas, grid)
        + integrated(increments, grid)
        - 0.02 * outward(result, grid),
        rtol=2e-6,
    )
    for actual, transported in zip(
        (r * u for r, u in zip(result.transport_density, result.velocity)),
        result.fluxes,
    ):
        np.testing.assert_allclose(
            actual, transported[0] + transported[1], rtol=2e-6, atol=1e-8
        )
    record_testsuite_property("float32_eos_error", float(result.eos_error))


@pytest.mark.parametrize("kind", ["temperature", "composition"])
@pytest.mark.parametrize("speed", [-1.0, 1.0])
def test_isobaric_contact_preserves_uniform_translation(kind, speed):
    grid, _, velocity, _, _ = setup(periodic=True, nx=16)
    left = jnp.broadcast_to(jnp.arange(grid.nx) < grid.nx // 2, velocity.x.shape)
    temperature = (
        jnp.where(left, 290.0, 350.0)
        if kind == "temperature"
        else jnp.full(left.shape, 310.0)
    )
    vapor = (
        jnp.where(left, 0.005, 0.04)
        if kind == "composition"
        else jnp.full(left.shape, 0.01)
    )
    gas = moist_gas_from_primitive(temperature, vapor, CONFIG)
    velocity = velocity._replace(x=jnp.full_like(velocity.x, speed))
    poisson = build_pressure_poisson(
        grid, backend="fft", periodic_x=True, periodic_y=True, dtype="float64"
    )
    ambient = jnp.stack(gas)[..., 0] / (gas.dry_density + gas.vapor_density)[..., 0]
    step = jax.jit(build_moist_gas_transport(poisson, CONFIG, ambient, tolerance=1e-11))
    dt = 0.25 / grid.nx
    zero = jax.tree.map(jnp.zeros_like, gas)
    result = step(gas, velocity, zero, dt)
    assert_eos(result)
    assert_shared_flux(result, gas, zero, grid, dt)
    for a, b in zip(result.velocity, velocity):
        np.testing.assert_allclose(a, b, atol=1e-8, rtol=0)
    assert float(jnp.ptp(result.pressure)) < 1e-8
    q = np.asarray(jnp.stack(gas))
    expected = 0.75 * q + 0.25 * np.roll(q, int(speed), axis=-1)
    np.testing.assert_allclose(np.asarray(jnp.stack(result.gas)), expected, rtol=1e-9)
