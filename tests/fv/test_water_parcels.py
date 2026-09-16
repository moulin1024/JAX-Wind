"""Parcel injection and two-way carrier exchange conservation checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind import StaggeredVelocity, initial_atmospheric_solution
from jaxwind.domain import UniformGrid
from jaxwind.moist_abl import initialize_moisture
from jaxwind.physics.moisture import MoistureConfig, WaterDropletProperties
from jaxwind.water_parcels import (
    WaterParcelSource,
    exchange_water_parcels,
    initial_water_parcels,
    inject_water_parcels,
)

jax.config.update("jax_enable_x64", True)
CONFIG = MoistureConfig()
PROPS = WaterDropletProperties()


def source(**overrides):
    return WaterParcelSource(
        **{
            "center": (0.1, 0.2, 0.2),
            "radius": 0.002,
            "speed": 22.0,
            "temperature": 308.35,
            "mass_flow": 0.2,
            "half_angle_degrees": 18.0,
            "diameter_scale": 369e-6,
            "diameter_spread": 3.67,
            "diameter_minimum": 74e-6,
            "diameter_maximum": 518e-6,
            "count_per_step": 8,
            "capacity": 32,
            "ramp_time": 0.0,
            **overrides,
        }
    )


def test_equal_mass_injection_and_overflow_are_accounted():
    jet = source(capacity=8)
    empty = initial_water_parcels(jet, "float64")
    p = inject_water_parcels(empty, 0.0, jnp.asarray(0), 0.001, jet, CONFIG)
    np.testing.assert_allclose(p.mass * p.multiplicity, 0.2 * 0.001 / 8, rtol=1e-14)
    assert np.ptp(np.asarray(p.mass)) > 0
    assert float(p.injected_mass) == 0.0002
    full = inject_water_parcels(p, 0.001, jnp.asarray(1), 0.001, jet, CONFIG)
    np.testing.assert_array_equal(full.mass, p.mass)
    assert float(full.overflow_mass) == 0.0002
    assert float(full.injected_mass) == float(p.injected_mass)


def flow_and_grid():
    grid = UniformGrid(8, 8, 8, 0.4, 0.4, 0.4)
    velocity = StaggeredVelocity(
        jnp.full((8, 8, 9), 3.0), jnp.zeros((8, 9, 8)), jnp.zeros((9, 8, 8))
    )
    flow = initial_atmospheric_solution(grid, velocity, dtype="float64")
    state, _ = initialize_moisture(flow, 312.35, 0.2, CONFIG)
    return state, grid


def totals(flow, parcels, grid):
    dry_mass = CONFIG.dry_air_density * jnp.asarray(grid.cell_volumes)
    liquid_mass = parcels.mass * parcels.multiplicity * parcels.active
    water = (
        jnp.sum(flow.moisture.vapor * dry_mass)
        + jnp.sum(liquid_mass)
        + parcels.escaped_mass
    )
    enthalpy = jnp.sum(
        dry_mass
        * (
            CONFIG.dry_air_heat_capacity * flow.scalar
            + CONFIG.water_vapor_latent_heat * flow.moisture.vapor
        )
    )
    enthalpy += jnp.sum(
        liquid_mass
        * PROPS.liquid_heat_capacity
        * (parcels.temperature - CONFIG.freezing_temperature)
    )
    enthalpy += parcels.escaped_enthalpy
    return np.asarray([water, enthalpy])


def test_cell_deposition_conserves_water_and_thermal_enthalpy():
    flow, grid = flow_and_grid()
    jet = source()
    parcels = inject_water_parcels(
        initial_water_parcels(jet, "float64"), 0.0, jnp.asarray(0), 0.001, jet, CONFIG
    )
    before = totals(flow, parcels, grid)
    advance = jax.jit(
        lambda f, p: exchange_water_parcels(f, p, grid, 0.001, CONFIG, 312.35)
    )
    result, remaining = advance(flow, parcels)
    after = totals(result, remaining, grid)
    np.testing.assert_allclose(after, before, rtol=2e-14, atol=1e-12)
    assert float(remaining.evaporated_mass) > 0
    assert float(jnp.min(result.scalar)) < 0
    np.testing.assert_allclose(
        jnp.sum(remaining.mass * remaining.multiplicity) + remaining.evaporated_mass,
        remaining.injected_mass,
        rtol=1e-14,
    )


def test_escape_retains_mass_ledger():
    flow, grid = flow_and_grid()
    jet = source(center=(0.3999, 0.2, 0.2))
    parcels = inject_water_parcels(
        initial_water_parcels(jet, "float64"), 0.0, jnp.asarray(0), 0.001, jet, CONFIG
    )
    before = totals(flow, parcels, grid)
    result, remaining = exchange_water_parcels(
        flow, parcels, grid, 0.001, CONFIG, 312.35
    )
    assert not bool(jnp.any(remaining.active))
    assert float(remaining.escaped_mass) > 0
    np.testing.assert_allclose(totals(result, remaining, grid), before, rtol=1e-14)
    np.testing.assert_allclose(
        remaining.escaped_enthalpy
        + CONFIG.water_vapor_latent_heat * remaining.evaporated_mass,
        remaining.injected_enthalpy + remaining.gas_sensible_energy_loss,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        remaining.escaped_mass + remaining.evaporated_mass,
        remaining.injected_mass,
        rtol=1e-14,
    )


def test_finite_annular_injection_preserves_mass_mean_angle():
    jet = source(count_per_step=4096, capacity=4096, inner_outer_radius_ratio=0.475)
    parcels = inject_water_parcels(
        initial_water_parcels(jet, "float64"), 0.0, jnp.asarray(0), 0.001, jet, CONFIG
    )
    speed = np.linalg.norm(np.asarray(parcels.velocity), axis=0)
    theta = np.arccos(np.asarray(parcels.velocity[0]) / speed)
    np.testing.assert_allclose(speed, jet.speed, rtol=1e-14)
    assert abs(np.mean(theta) * 180 / np.pi - jet.half_angle_degrees) < 0.01
    tangent = np.tan(theta)
    assert tangent.min() >= 0.475 * jet.outer_cone_tangent()
    assert tangent.max() <= jet.outer_cone_tangent()
    np.testing.assert_allclose(
        jnp.sum(parcels.mass * parcels.multiplicity), jet.mass_flow * 0.001, rtol=1e-14
    )


def test_energy_ledgers_survive_checkpoint_roundtrip(tmp_path):
    from jaxwind.io.checkpoint import load_checkpoint, save_checkpoint

    flow, grid = flow_and_grid()
    jet = source(center=(0.3999, 0.2, 0.2))
    parcels = inject_water_parcels(
        initial_water_parcels(jet, "float64"), 0.0, jnp.asarray(0), 0.001, jet, CONFIG
    )
    _, parcels = exchange_water_parcels(flow, parcels, grid, 0.001, CONFIG, 312.35)
    path = tmp_path / "parcels.npz"
    save_checkpoint(path, parcels, metadata={"fingerprint": "energy-ledger-test"})
    restored, _, _ = load_checkpoint(
        path, initial_water_parcels(jet, "float64"), fingerprint="energy-ledger-test"
    )
    for actual, expected in zip(restored, parcels):
        np.testing.assert_array_equal(actual, expected)
    assert float(restored.escaped_enthalpy) > 0


def test_first_wall_contact_counted_once_and_population_budgets_close():
    flow, grid = flow_and_grid()
    jet = source()
    p = inject_water_parcels(
        initial_water_parcels(jet, "float64"), 0.0, jnp.asarray(0), 0.001, jet, CONFIG
    )
    p = p._replace(
        position=p.position.at[1].set(1e-6), velocity=p.velocity.at[1].set(-2.0)
    )
    flow, p = exchange_water_parcels(flow, p, grid, 0.001, CONFIG, 312.35)
    assert bool(jnp.all(p.touched_wall[p.active]))
    first_mass, first_h = float(p.first_wall_mass), float(p.first_wall_enthalpy)
    np.testing.assert_allclose(first_mass, jnp.sum(p.mass * p.multiplicity), rtol=1e-14)
    assert float(p.wall_evaporated_mass) == 0
    flow, p = exchange_water_parcels(flow, p, grid, 0.001, CONFIG, 312.35)
    assert float(p.first_wall_mass) == first_mass
    assert float(p.first_wall_enthalpy) == first_h
    p = p._replace(
        position=p.position.at[0].set(0.3999), velocity=p.velocity.at[0].set(22.0)
    )
    flow, p = exchange_water_parcels(flow, p, grid, 0.001, CONFIG, 312.35)
    assert not bool(jnp.any(p.active))
    np.testing.assert_allclose(
        p.escaped_wall_mass + p.wall_evaporated_mass, p.first_wall_mass, rtol=1e-14
    )
    np.testing.assert_allclose(
        p.escaped_wall_enthalpy
        + CONFIG.water_vapor_latent_heat * p.wall_evaporated_mass,
        p.first_wall_enthalpy + p.wall_gas_sensible_energy_loss,
        rtol=1e-14,
    )
    recycled = inject_water_parcels(p, 0.003, jnp.asarray(1), 0.001, jet, CONFIG)
    assert not bool(jnp.any(recycled.touched_wall[recycled.active]))
    assert float(recycled.first_wall_mass) == first_mass


def test_transport_only_preserves_thermodynamics_and_exchanges_momentum():
    from jaxwind.numerics.discretization import cell_velocity

    flow, grid = flow_and_grid()
    jet = source()
    p = inject_water_parcels(
        initial_water_parcels(jet, "float64"), 0.0, jnp.asarray(0), 0.001, jet, CONFIG
    )
    advance = jax.jit(
        lambda f, p: exchange_water_parcels(
            f, p, grid, 0.001, CONFIG, 312.35, thermal_exchange=False
        )
    )
    result, remaining = advance(flow, p)
    np.testing.assert_array_equal(remaining.mass, p.mass)
    np.testing.assert_array_equal(remaining.temperature, p.temperature)
    np.testing.assert_array_equal(result.scalar, flow.scalar)
    for actual, expected in zip(result.moisture, flow.moisture):
        np.testing.assert_array_equal(actual, expected)
    assert float(remaining.evaporated_mass) == 0
    assert float(remaining.gas_sensible_energy_loss) == 0
    assert np.all(np.asarray(remaining.position[0, p.active] > p.position[0, p.active]))
    assert np.all(np.asarray(remaining.velocity[0, p.active] < p.velocity[0, p.active]))
    # Interior axial momentum exchange excludes gravity (which acts vertically).
    parcel_change = jnp.sum(
        p.mass * p.multiplicity * (remaining.velocity[0] - p.velocity[0])
    )
    gas_change = jnp.sum(
        CONFIG.dry_air_density * jnp.asarray(grid.cell_volumes)
        * (cell_velocity(result.velocity)[0] - cell_velocity(flow.velocity)[0])
    )
    assert float(gas_change) > 0
    np.testing.assert_allclose(gas_change, -parcel_change, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(totals(result, remaining, grid), totals(flow, p, grid), rtol=1e-14)


def test_transport_only_escape_conserves_injected_liquid_and_enthalpy():
    flow, grid = flow_and_grid()
    jet = source(center=(0.3999, 0.2, 0.2))
    p = inject_water_parcels(
        initial_water_parcels(jet, "float64"), 0.0, jnp.asarray(0), 0.001, jet, CONFIG
    )
    result, remaining = exchange_water_parcels(
        flow, p, grid, 0.001, CONFIG, 312.35, thermal_exchange=False
    )
    assert not bool(jnp.any(remaining.active))
    np.testing.assert_allclose(remaining.escaped_mass, p.injected_mass, rtol=1e-14)
    np.testing.assert_allclose(remaining.escaped_enthalpy, p.injected_enthalpy, rtol=1e-14)
    np.testing.assert_allclose(totals(result, remaining, grid), totals(flow, p, grid), rtol=1e-14)


@pytest.mark.parametrize("axis", [1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
def test_open_side_escape_has_no_wall_contact_and_is_counted_once(axis, sign):
    flow, grid = flow_and_grid()
    jet = source()
    p = inject_water_parcels(
        initial_water_parcels(jet, "float64"), 0.0, jnp.asarray(0), 0.001, jet, CONFIG
    )
    edge = 1e-6 if sign < 0 else 0.4 - 1e-6
    p = p._replace(
        position=p.position.at[axis].set(edge),
        velocity=p.velocity.at[axis].set(sign * 2.0),
    )
    advance = jax.jit(
        lambda f, p: exchange_water_parcels(
            f, p, grid, 0.001, CONFIG, 312.35,
            thermal_exchange=False, side_boundary="escape",
        )
    )
    result, escaped = advance(flow, p)
    assert not bool(jnp.any(escaped.active))
    assert not bool(jnp.any(escaped.touched_wall))
    assert float(escaped.first_wall_mass) == 0
    assert float(escaped.escaped_wall_mass) == 0
    assert np.all(np.asarray(sign * escaped.velocity[axis, p.active] > 0))
    np.testing.assert_allclose(escaped.escaped_mass, p.injected_mass, rtol=1e-14)
    np.testing.assert_allclose(totals(result, escaped, grid), totals(flow, p, grid), rtol=1e-14)
    again, stopped = advance(result, escaped)
    np.testing.assert_array_equal(stopped.escaped_mass, escaped.escaped_mass)
    for actual, expected in zip(again.velocity, result.velocity):
        np.testing.assert_array_equal(actual, expected)


def test_unknown_side_boundary_is_rejected():
    flow, grid = flow_and_grid()
    with pytest.raises(ValueError, match="side_boundary"):
        exchange_water_parcels(
            flow, initial_water_parcels(source(), "float64"), grid,
            0.001, CONFIG, 312.35, side_boundary="open",
        )
