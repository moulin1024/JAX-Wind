"""Shared-flux carrier verification including momentum and energy ledgers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.domain import UniformGrid
from jaxwind.numerics.poisson import build_pressure_poisson
from jaxwind.physics.moisture import MoistureConfig, WaterDropletProperties
from jaxwind.spray_carrier import build_carrier_step
from jaxwind.spray_core import WaterCoreBins, advance_water_core
from jaxwind.spray_coupling import GasInventory
from jaxwind.spray_low_mach import moist_gas_eos_density, moist_gas_from_primitive
from jaxwind.state import StaggeredVelocity

jax.config.update("jax_enable_x64", True)
CONFIG = MoistureConfig()


def setup(periodic=True, nx=8, **controls):
    grid = UniformGrid(nx, 4, 4, 1.0, 0.5, 0.5)
    shape = (grid.nz, grid.ny, grid.nx)
    gas = moist_gas_from_primitive(
        jnp.full(shape, 310.0), jnp.full(shape, 0.01), CONFIG
    )
    velocity = StaggeredVelocity(
        jnp.zeros(shape if periodic else (grid.nz, grid.ny, grid.nx + 1)),
        jnp.zeros(shape),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
    )
    poisson = build_pressure_poisson(
        grid,
        backend="fft" if periodic else "gmg",
        periodic_x=periodic,
        periodic_y=True,
        open_x_low=not periodic,
        dtype="float64",
    )
    ambient = jnp.stack(gas)[..., 0] / (gas.dry_density + gas.vapor_density)[..., 0]
    step = jax.jit(
        build_carrier_step(
            poisson, CONFIG, ambient, jnp.zeros((3, grid.nz, grid.ny)), **controls
        )
    )
    return (
        grid,
        gas,
        velocity,
        step,
        jax.tree.map(jnp.zeros_like, gas),
        jnp.zeros((3, *shape)),
    )


def geometry(grid, component, periodic):
    widths = [
        np.full(grid.nz, grid.dz),
        np.full(grid.ny, grid.dy),
        np.full(grid.nx, grid.dx),
    ]
    if not (False, True, periodic)[component]:
        h = widths[component][0]
        widths[component] = np.r_[h / 2, np.full(len(widths[component]) - 1, h), h / 2]
    volume = (
        widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
    )
    return widths, volume


def average(q, component, periodic):
    if periodic:
        return (q + np.roll(q, 1, axis=component)) / 2
    a = np.take(q, np.arange(q.shape[component] - 1), axis=component)
    b = np.take(q, np.arange(1, q.shape[component]), axis=component)
    return np.concatenate(
        (
            np.take(q, [0], axis=component),
            (a + b) / 2,
            np.take(q, [-1], axis=component),
        ),
        axis=component,
    )


def assert_ledgers(result, gas, velocity, increments, dp, grid, dt, periodic):
    assert bool(result.accepted), (
        result.iterations,
        result.eos_error,
        result.donor_error,
        result.momentum_error,
        result.flow_error,
        result.continuity_error,
        result.outgoing_fraction,
    )
    rho = np.asarray(result.gas.dry_density + result.gas.vapor_density)
    np.testing.assert_allclose(
        rho, moist_gas_eos_density(result.gas, CONFIG), rtol=1.01e-9
    )
    assert float(result.momentum_error) <= 1e-9 and float(result.flow_error) <= 1e-9
    before = (
        np.sum(np.asarray(jnp.stack(gas)), axis=(1, 2, 3)) * grid.dx * grid.dy * grid.dz
    )
    source = (
        np.sum(np.asarray(jnp.stack(increments)), axis=(1, 2, 3))
        * grid.dx
        * grid.dy
        * grid.dz
    )
    after = (
        np.sum(np.asarray(jnp.stack(result.gas)), axis=(1, 2, 3))
        * grid.dx
        * grid.dy
        * grid.dz
    )
    boundary = (
        0.0
        if periodic
        else np.sum(
            np.asarray(
                result.scalar_fluxes.x[..., -1] - result.scalar_fluxes.x[..., 0]
            ),
            axis=(1, 2),
        )
        * grid.dy
        * grid.dz
    )
    np.testing.assert_allclose(
        after, before + source - dt * boundary, rtol=3e-13, atol=5e-11
    )
    for f, s, t, u in zip(
        result.mass_flux,
        result.scalar_fluxes,
        result.transport_density,
        result.velocity,
    ):
        np.testing.assert_allclose(f, s[0] + s[1], rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(f, t * u, rtol=1e-13, atol=1e-14)
    old_rho = np.asarray(gas.dry_density + gas.vapor_density)
    dm = np.asarray(increments.dry_density + increments.vapor_density)
    for k, c in enumerate((2, 1, 0)):
        widths, volume = geometry(grid, c, periodic)
        old_inertia = average(old_rho, c, (False, True, periodic)[c])
        rho_star = average(old_rho + dm, c, (False, True, periodic)[c])
        old_p = old_inertia * np.asarray(velocity[k])
        pstar = old_p + average(np.asarray(dp[k]), c, (False, True, periodic)[c])
        new_p = np.asarray(result.inertia_density[k] * result.velocity[k])
        boundary_p = boundary_k = 0.0
        if not periodic:
            area = widths[0][:, None] * widths[1][None, :]
            pf = np.asarray(result.momentum_fluxes[k].x)
            kf = np.asarray(result.kinetic_fluxes[k].x)
            boundary_p = np.sum((pf[..., -1] - pf[..., 0]) * area)
            boundary_k = np.sum((kf[..., -1] - kf[..., 0]) * area)
        expected = (
            np.sum(old_p * volume)
            + np.sum(np.asarray(dp[k])) * grid.dx * grid.dy * grid.dz
            - dt * boundary_p
        )
        expected += np.sum(
            np.asarray(
                result.wall_impulse[k]
                + result.pressure_impulse[k]
                + result.momentum_residual[k]
            )
            * volume
        )
        np.testing.assert_allclose(
            np.sum(new_p * volume), expected, rtol=1e-11, atol=2e-14
        )
        expected_ke = np.sum(pstar**2 / (2 * rho_star) * volume) - dt * boundary_k
        expected_ke += np.sum(
            np.asarray(
                result.pressure_work[k]
                + result.iteration_work[k]
                - result.numerical_kinetic_loss[k]
            )
            * volume
        )
        actual_ke = np.sum(
            np.asarray(result.inertia_density[k] * result.velocity[k] ** 2 / 2) * volume
        )
        np.testing.assert_allclose(actual_ke, expected_ke, rtol=1e-11, atol=2e-14)


@pytest.mark.parametrize("kind", ["temperature", "humidity"])
@pytest.mark.parametrize("speed", [-1.0, 1.0])
def test_shared_iteration_preserves_uniform_moving_contact(
    kind, speed, record_testsuite_property
):
    grid, _, velocity, step, zero, dp = setup()
    left = jnp.broadcast_to(jnp.arange(grid.nx) < grid.nx // 2, zero.dry_density.shape)
    gas = moist_gas_from_primitive(
        jnp.where(left, 290.0, 350.0)
        if kind == "temperature"
        else jnp.full(left.shape, 310.0),
        jnp.where(left, 0.005, 0.04)
        if kind == "humidity"
        else jnp.full(left.shape, 0.01),
        CONFIG,
    )
    velocity = velocity._replace(x=jnp.full_like(velocity.x, speed))
    dt = 0.25 * grid.dx
    result = step(gas, velocity, zero, dp, dt)
    assert_ledgers(result, gas, velocity, zero, dp, grid, dt, True)
    for a, b in zip(result.velocity, velocity):
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=0)
    q = np.asarray(jnp.stack(gas))
    expected = 0.75 * q + 0.25 * np.roll(q, int(speed), axis=-1)
    np.testing.assert_allclose(jnp.stack(result.gas), expected, rtol=1e-10)
    record_testsuite_property(
        f"contact_iterations_{kind}_{speed}", int(result.iterations)
    )


def test_shear_wave_uses_same_final_flux_and_accounts_numerical_energy_loss(
    record_testsuite_property,
):
    grid, gas, velocity, step, zero, dp = setup(nx=16)
    x = jnp.asarray(grid.x_centers)
    shear = jnp.broadcast_to(jnp.sin(2 * jnp.pi * x), velocity.y.shape)
    velocity = velocity._replace(x=jnp.full_like(velocity.x, 0.4), y=shear)
    dt = 0.2 * grid.dx / 0.4
    result = step(gas, velocity, zero, dp, dt)
    assert_ledgers(result, gas, velocity, zero, dp, grid, dt, True)
    np.testing.assert_allclose(
        result.velocity.y,
        0.8 * shear + 0.2 * jnp.roll(shear, 1, axis=-1),
        atol=2e-10,
        rtol=0,
    )
    np.testing.assert_allclose(result.velocity.x, 0.4, atol=2e-10, rtol=0)
    assert float(jnp.max(jnp.abs(result.pressure))) < 1e-8
    _, vol = geometry(grid, 1, True)
    lost = float(np.sum(np.asarray(result.numerical_kinetic_loss.y) * vol))
    assert lost > 0
    ratio = float(
        np.sum(np.asarray(result.velocity.y) ** 2) / np.sum(np.asarray(shear) ** 2)
    )
    expected_ratio = abs(0.8 + 0.2 * np.exp(-2j * np.pi / grid.nx)) ** 2
    np.testing.assert_allclose(ratio, expected_ratio, atol=1e-11)
    record_testsuite_property("shear_numerical_loss_J", lost)
    record_testsuite_property("shear_iterations", int(result.iterations))


def test_open_heating_uses_consistent_momentum_pressure_and_boundary_budgets(
    record_testsuite_property,
):
    grid, gas, velocity, step, zero, dp = setup(periodic=False)
    increments = zero._replace(
        enthalpy_density=2 * CONFIG.dry_air_heat_capacity * gas.dry_density
    )
    result = step(gas, velocity, increments, dp, 0.02)
    assert_ledgers(result, gas, velocity, increments, dp, grid, 0.02, False)
    assert np.all(np.asarray(result.velocity.x[..., 0]) < 0)
    assert np.all(np.asarray(result.velocity.x[..., -1]) > 0)
    record_testsuite_property("heat_iterations", int(result.iterations))
    record_testsuite_property("heat_momentum_error", float(result.momentum_error))


def core_sources(grid, gas, zero, dp):
    index = (1, 2, 3)
    vol = grid.dx * grid.dy * grid.dz
    rho = gas.dry_density + gas.vapor_density
    mass = rho[index] * vol
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
        jnp.array([[8.0], [-3.0], [1.0]]),
        jnp.array([295.0]),
    )
    exchange = advance_water_core(
        core, liquid, 0.002, CONFIG, WaterDropletProperties(), drag_heat_fraction=1.0
    )
    assert bool(exchange.accepted) and float(exchange.evaporated_mass) > 0
    inc = zero._replace(
        dry_density=zero.dry_density.at[index].set(
            (exchange.gas.species[0] - core.species[0]) / vol
        ),
        vapor_density=zero.vapor_density.at[index].set(
            (exchange.gas.species[1] - core.species[1]) / vol
        ),
        enthalpy_density=zero.enthalpy_density.at[index].set(
            (exchange.gas.enthalpy - core.enthalpy) / vol
        ),
    )
    dp = dp.at[(slice(None), *index)].set((exchange.gas.momentum - core.momentum) / vol)
    return inc, dp, liquid, exchange


def test_local_evaporating_core_drives_coupled_flux_and_opposite_phase_impulse(
    record_testsuite_property,
):
    grid, gas, velocity, step, zero, dp = setup(periodic=False)
    increments, dp, liquid, exchange = core_sources(grid, gas, zero, dp)
    result = step(gas, velocity, increments, dp, 0.002)
    assert_ledgers(result, gas, velocity, increments, dp, grid, 0.002, False)
    liquid_momentum = lambda b: np.sum(
        np.asarray(b.velocity * (b.mass * b.multiplicity)[None]), axis=1
    )
    gas_source = np.sum(np.asarray(dp), axis=(1, 2, 3)) * grid.dx * grid.dy * grid.dz
    np.testing.assert_allclose(
        gas_source,
        liquid_momentum(liquid) - liquid_momentum(exchange.liquid),
        rtol=1e-12,
        atol=1e-17,
    )
    record_testsuite_property("evaporation_iterations", int(result.iterations))
    record_testsuite_property("evaporation_eos_error", float(result.eos_error))
    record_testsuite_property(
        "evaporation_momentum_error", float(result.momentum_error)
    )
    record_testsuite_property("evaporation_mass_kg", float(exchange.evaporated_mass))


@pytest.mark.parametrize(
    "failure",
    ["closed_heat", "nonlinear_limit", "pressure_limit", "negative_vapor", "cfl"],
)
def test_failed_coupled_step_rolls_back_every_carrier_inventory_and_ledger(failure):
    controls = (
        {"max_iterations": 1}
        if failure == "nonlinear_limit"
        else {"pressure_max_iterations": 1}
        if failure == "pressure_limit"
        else {}
    )
    periodic = failure in ("closed_heat", "negative_vapor", "cfl")
    grid, gas, velocity, step, zero, dp = setup(periodic=periodic, **controls)
    dt = 0.02
    increments = zero
    if failure in ("closed_heat", "nonlinear_limit"):
        increments = zero._replace(
            enthalpy_density=2 * CONFIG.dry_air_heat_capacity * gas.dry_density
        )
    elif failure == "pressure_limit":
        increments, dp, _, _ = core_sources(grid, gas, zero, dp)
        dt = 0.002
    elif failure == "negative_vapor":
        increments = zero._replace(vapor_density=-2 * gas.vapor_density)
    else:
        velocity = velocity._replace(x=jnp.full_like(velocity.x, 10.0))
        dt = 1.0
    result = step(gas, velocity, increments, dp, dt)
    assert not bool(result.accepted), (failure, result.iterations)
    for a, b in zip(result.gas, gas):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(result.velocity, velocity):
        np.testing.assert_array_equal(a, b)
    for name in result._fields[3:15]:
        for value in jax.tree.leaves(getattr(result, name)):
            assert np.all(np.asarray(value) == 0)


def test_temperature_contact_remains_consistent_over_one_domain_crossing(
    record_testsuite_property,
):
    grid, _, velocity, step, zero, dp = setup()
    left = jnp.broadcast_to(jnp.arange(grid.nx) < grid.nx // 2, zero.dry_density.shape)
    gas = moist_gas_from_primitive(
        jnp.where(left, 290.0, 350.0), jnp.full(left.shape, 0.01), CONFIG
    )
    velocity = velocity._replace(x=jnp.ones_like(velocity.x))
    dt = 0.25 * grid.dx
    count = 4 * grid.nx

    def advance(state, _):
        q, u = state
        result = step(q, u, zero, dp, dt)
        return (result.gas, result.velocity), (
            result.accepted,
            result.iterations,
            result.momentum_error,
        )

    (final, final_u), history = jax.jit(
        lambda: jax.lax.scan(advance, (gas, velocity), None, length=count)
    )()
    assert np.all(np.asarray(history[0]))
    q = np.asarray(jnp.stack(gas))
    for _ in range(count):
        q = 0.75 * q + 0.25 * np.roll(q, 1, axis=-1)
    np.testing.assert_allclose(jnp.stack(final), q, rtol=2e-9, atol=1e-10)
    for a, b in zip(final_u, velocity):
        np.testing.assert_allclose(a, b, rtol=0, atol=2e-9)
    record_testsuite_property("crossing_steps", count)
    record_testsuite_property("crossing_max_iterations", int(jnp.max(history[1])))
    record_testsuite_property("crossing_max_momentum_error", float(jnp.max(history[2])))


def test_variable_density_vortex_closes_accumulated_mechanical_and_scalar_budgets(
    record_testsuite_property,
):
    grid, _, velocity, step, zero, dp = setup()
    x = jnp.asarray(grid.x_centers)
    y = jnp.asarray(grid.y_centers)
    temperature = jnp.broadcast_to(
        310
        + 20
        * jnp.sin(2 * jnp.pi * x)[None, None, :]
        * jnp.cos(4 * jnp.pi * y)[None, :, None],
        zero.dry_density.shape,
    )
    gas = moist_gas_from_primitive(
        temperature, jnp.full_like(temperature, 0.01), CONFIG
    )
    # A corner streamfunction supplies exactly divergence-free staggered data.
    xf = jnp.arange(grid.nx) * grid.dx
    yf = jnp.arange(grid.ny) * grid.dy
    psi = 0.02 * jnp.sin(4 * jnp.pi * yf)[:, None] * jnp.sin(2 * jnp.pi * xf)[None, :]
    u = (jnp.roll(psi, -1, axis=0) - psi) / grid.dy
    v = -(jnp.roll(psi, -1, axis=1) - psi) / grid.dx
    velocity = velocity._replace(
        x=jnp.broadcast_to(u, velocity.x.shape), y=jnp.broadcast_to(v, velocity.y.shape)
    )
    volumes = StaggeredVelocity(
        *(jnp.asarray(geometry(grid, c, True)[1]) for c in (2, 1, 0))
    )
    integrate = lambda faces: sum(jnp.sum(q * w) for q, w in zip(faces, volumes))
    dt = 0.02
    count = 12

    def advance(state, _):
        q, u = state
        result = step(q, u, zero, dp, dt)
        balance = (
            integrate(result.pressure_work)
            + integrate(result.iteration_work)
            - integrate(result.numerical_kinetic_loss)
        )
        return (result.gas, result.velocity), (
            result.accepted,
            result.iterations,
            result.momentum_error,
            balance,
            result.eos_error,
            result.pressure_continuity_error,
            result.pressure_linear_error,
            result.flow_error,
            result.donor_error,
        )

    (final, final_u), history = jax.jit(
        lambda: jax.lax.scan(advance, (gas, velocity), None, length=count)
    )()
    for key, values in zip(
        (
            "accepted",
            "iterations",
            "momentum_error",
            "energy_change",
            "eos_error",
            "pressure_continuity",
            "pressure_linear",
            "flow_error",
            "donor_error",
        ),
        history,
    ):
        record_testsuite_property("vortex_first_" + key, str(np.asarray(values)[0]))
    assert np.all(np.asarray(history[0])), {
        key: np.asarray(values).tolist()
        for key, values in zip(
            (
                "accepted",
                "iterations",
                "momentum_error",
                "energy_change",
                "eos_error",
                "pressure_continuity",
                "pressure_linear",
                "flow_error",
                "donor_error",
            ),
            history,
        )
    }
    before = np.sum(np.asarray(jnp.stack(gas)), axis=(1, 2, 3))
    after = np.sum(np.asarray(jnp.stack(final)), axis=(1, 2, 3))
    np.testing.assert_allclose(after, before, rtol=3e-13, atol=1e-9)

    def kinetic(q, u):
        rho = np.asarray(q.dry_density + q.vapor_density)
        total = 0.0
        for c, component_u, volume in zip((2, 1, 0), u, volumes):
            density = average(rho, c, (False, True, True)[c])
            total += np.sum(
                0.5 * density * np.asarray(component_u) ** 2 * np.asarray(volume)
            )
        return total

    start_ke = kinetic(gas, velocity)
    end_ke = kinetic(final, final_u)
    np.testing.assert_allclose(
        end_ke, start_ke + np.sum(np.asarray(history[3])), rtol=3e-12, atol=3e-14
    )
    assert end_ke < start_ke
    record_testsuite_property("vortex_steps", count)
    record_testsuite_property("vortex_max_iterations", int(jnp.max(history[1])))
    record_testsuite_property("vortex_max_momentum_error", float(jnp.max(history[2])))
    record_testsuite_property("vortex_max_eos_error", float(jnp.max(history[4])))
    record_testsuite_property("vortex_initial_kinetic_J", start_ke)
    record_testsuite_property("vortex_final_kinetic_J", end_ke)
