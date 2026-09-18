"""Tunnel conservation, angular source quadrature, separator events and sensors."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind import StaggeredVelocity
from jaxwind.config.fluent_dpm import DPMOptions
from jaxwind.config.moisture import WaterSprayOptions
from jaxwind.domain import UniformGrid
from jaxwind.fluent_dpm_atmosphere import specific_enthalpy
from jaxwind.fluent_dpm_les import DPMLESFields
from jaxwind.fluent_dpm_spatial import (
    build_spatial_step,
    initial_ledger,
    initial_parcels,
    inject_parcels,
    injection_velocities,
    liquid_inventory,
)
from jaxwind.physics.fluent_dpm import DPMWaterMaterial
from jaxwind.spray_low_mach import MoistGasFields
from jaxwind.waterjet_observation import WaterjetObservation, sample_plane, wet_bulb_c

jax.config.update("jax_enable_x64", True)
M = DPMWaterMaterial()


def fixture(**settings):
    grid = UniformGrid(4, 4, 4, 4.0, 4.0, 4.0)
    options = DPMOptions(
        (200e-6,),
        (1.0,),
        (2.0, 0.0, 0.0),
        300.0,
        capacity=4,
        dispersion="off",
        **settings,
    )
    source = WaterSprayOptions(
        1e-5, 0.5, (1.0, 1.0, 1.0), model="fluent-dpm", dpm=options
    )
    rho = jnp.ones((4, 4, 4))
    gas = MoistGasFields(rho, 0.005 * rho, specific_enthalpy(310.0, 0.005, M) * rho)
    u = StaggeredVelocity(
        jnp.ones((4, 4, 5)), jnp.zeros((4, 5, 4)), jnp.zeros((5, 4, 4))
    )
    zero = jnp.zeros_like(rho)
    les = DPMLESFields(zero, zero, zero, jnp.ones_like(rho))
    return grid, source, gas, u, les


def mac_inventory(grid, gas, velocity):
    rho = np.asarray(gas.dry_density + gas.vapor_density)
    momentum = []
    energy = float(jnp.sum(gas.enthalpy_density)) * grid.dx * grid.dy * grid.dz
    for axis, speed in zip((2, 1, 0), velocity):
        mass = (
            np.concatenate(
                (
                    np.take(rho, [0], axis),
                    np.take(rho, range(3), axis) + np.take(rho, range(1, 4), axis),
                    np.take(rho, [-1], axis),
                ),
                axis,
            )
            / 2
            * grid.dx
            * grid.dy
            * grid.dz
        )
        momentum.append(np.sum(mass * speed))
        energy += np.sum(0.5 * mass * speed**2)
    return np.asarray(momentum), energy


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (1.0, 2.0, 3.0)])
def test_hollow_cone_mass_momentum_and_solid_angle_quadrature(axis):
    o = DPMOptions(
        (100e-6, 300e-6),
        (0.3, 0.7),
        axis,
        305.0,
        capacity=128,
        injection_geometry="hollow-cone",
        cone_inner_half_angle_degrees=10.0,
        cone_outer_half_angle_degrees=30.0,
        cone_azimuthal_points=8,
        cone_polar_points=3,
    )
    source = WaterSprayOptions(0.2, 0.1, (1.0, 1.0, 1.0), model="fluent-dpm", dpm=o)
    p, ledger, ok = inject_parcels(
        initial_parcels(o, jnp.float64),
        initial_ledger(jnp.float64),
        source,
        (0.1, 1.0, 1.0),
        0.0,
        0.01,
        M,
    )
    assert ok
    count = o.injection_batch_size
    mass = np.asarray(p.mass * p.multiplicity)[:count].reshape(2, -1)
    np.testing.assert_allclose(
        mass.sum(axis=1), np.array([0.3, 0.7]) * 0.002, rtol=1e-14
    )
    speed = np.linalg.norm(axis)
    velocities = np.asarray(injection_velocities(o, jnp.float64))
    np.testing.assert_allclose(np.linalg.norm(velocities, axis=0), speed, rtol=1e-14)
    mean_cos = (np.cos(np.deg2rad(10)) + np.cos(np.deg2rad(30))) / 2
    np.testing.assert_allclose(
        ledger.injected[1:4], 0.002 * np.asarray(axis) * mean_cos, atol=1e-17
    )
    np.testing.assert_allclose(ledger.injected[5], 0.5 * 0.002 * speed**2, rtol=1e-14)
    # Capacity failure keeps all source/RNG state and inventories unchanged.
    p = p._replace(mass=jnp.ones_like(p.mass), multiplicity=jnp.ones_like(p.mass))
    rejected, unchanged, ok = inject_parcels(
        p, ledger, source, (0.1, 1.0, 1.0), 0.01, 0.01, M
    )
    assert not ok
    for a, b in zip(p, rejected):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(ledger, unchanged):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "position,velocity,kind",
    [
        ((1.0, 0.005, 2.0), (0.0, -2.0, 0.0), "trapped"),
        ((1.0, 3.995, 2.0), (0.0, 2.0, 0.0), "trapped"),
        ((1.0, 2.0, 0.005), (0.0, 0.0, -2.0), "trapped"),
        ((1.0, 2.0, 3.995), (0.0, 0.0, 2.0), "trapped"),
        ((3.995, 2.0, 2.0), (2.0, 0.0, 0.0), "escaped"),
        ((1.995, 2.0, 2.0), (2.0, 0.0, 0.0), "collected"),
    ],
)
def test_tunnel_and_separator_close_mass_momentum_energy(position, velocity, kind):
    grid, source, gas, u, les = fixture(eliminator_x_m=2.0)
    source = replace(source, dpm=replace(source.dpm, injection_velocity_m_s=velocity))
    p, l, ok = inject_parcels(
        initial_parcels(source.dpm, jnp.float64),
        initial_ledger(jnp.float64),
        source,
        position,
        0.0,
        0.01,
        M,
    )
    assert ok
    initial_momentum, initial_energy = mac_inventory(grid, gas, u)
    step = jax.jit(
        build_spatial_step(
            grid,
            source,
            M,
            101325.0,
            periodic_y=False,
            tunnel_walls=True,
            gravity=(0.0, 0.0, 0.0),
        )
    )
    r = step(gas, u, p, l, 0.01, les)
    assert r.accepted
    assert jnp.sum(r.parcels.mass * r.parcels.multiplicity) == 0
    removed = getattr(r.ledger, kind)
    assert removed[0] > 0
    np.testing.assert_allclose(
        removed[0] + r.ledger.evaporated_mass, l.injected[0], atol=1e-20
    )
    for other in {"trapped", "escaped", "collected"} - {kind}:
        np.testing.assert_array_equal(getattr(r.ledger, other), 0)
    momentum, energy = mac_inventory(grid, r.gas, r.velocity)
    np.testing.assert_allclose(
        momentum + removed[1:4] + r.ledger.wall_impulse,
        initial_momentum + l.injected[1:4],
        atol=2e-14,
    )
    np.testing.assert_allclose(
        energy + removed[4] + removed[5] + r.ledger.wall_energy,
        initial_energy + l.injected[4] + l.injected[5],
        atol=1e-8,
        rtol=2e-15,
    )
    np.testing.assert_array_equal(r.velocity.y[:, [0, -1], :], 0)
    np.testing.assert_array_equal(r.velocity.z[[0, -1], :, :], 0)


def test_separator_stops_exchange_at_crossing_and_handles_start_on_plane():
    grid, source, gas, u, les = fixture(eliminator_x_m=2.0)
    p, l, _ = inject_parcels(
        initial_parcels(source.dpm, jnp.float64),
        initial_ledger(jnp.float64),
        source,
        (1.99, 2.0, 2.0),
        0.0,
        0.01,
        M,
    )
    run = jax.jit(
        build_spatial_step(
            grid,
            source,
            M,
            101325.0,
            periodic_y=False,
            tunnel_walls=True,
            gravity=(0.0, 0.0, 0.0),
        )
    )
    r = run(gas, u, p, l, 0.02, les)
    control_source = replace(source, dpm=replace(source.dpm, eliminator_x_m=None))
    control = jax.jit(
        build_spatial_step(
            grid,
            control_source,
            M,
            101325.0,
            periodic_y=False,
            tunnel_walls=True,
            gravity=(0.0, 0.0, 0.0),
        )
    )
    c = control(gas, u, p, l, (2.0 - 1.99) / 2.0, les)
    assert r.accepted and c.accepted
    for a, b in zip(r.gas, c.gas):
        np.testing.assert_allclose(a, b, rtol=1e-14)
    np.testing.assert_allclose(
        r.ledger.collected,
        liquid_inventory(
            c.parcels.mass[0],
            c.parcels.multiplicity[0],
            c.parcels.velocity[:, 0],
            c.parcels.temperature[0],
            M,
        ),
        rtol=1e-14,
    )
    on_plane = p._replace(position=p.position.at[0, 0].set(2.0))
    at = run(gas, u, on_plane, l, 0.02, les)
    assert at.accepted
    for a, b in zip(at.gas, gas):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(at.ledger.collected, l.injected, rtol=1e-14)


def test_sensor_pairing_affine_interpolation_and_psychrometric_roundtrip():
    grid = UniformGrid(8, 6, 6, 2.4, 0.585, 0.585)
    dpm = DPMOptions((1e-4,), (1.0,), (1.0, 0.0, 0.0), 300.0, eliminator_x_m=1.5)
    o = WaterjetObservation.from_table({"sensor_x_m": 1.9}, grid, dpm)
    field = (
        2 * jnp.asarray(grid.x_centers)[None, None, :]
        + 3 * jnp.asarray(grid.y_centers)[None, :, None]
        + 5 * jnp.asarray(grid.z_centers)[:, None, None]
    )
    expected = [2 * 1.9 + 3 * y + 5 * z for z in o.sensor_z_m for y in o.sensor_y_m]
    np.testing.assert_allclose(sample_plane(field, grid, o), expected, atol=1e-14)
    from jaxwind.physics.moisture import saturation_vapor_pressure_water

    e = saturation_vapor_pressure_water(18.7 + 273.15) - 0.00066 * (
        1 + 0.00115 * 18.7
    ) * 101325 * (39.2 - 18.7)
    q = 0.622 * e / (101325 - e)
    np.testing.assert_allclose(
        wet_bulb_c(jnp.array([39.2]), jnp.array([q]), 101325.0, 0.622), 18.7, atol=1e-12
    )
    with pytest.raises(ValueError, match="downstream"):
        WaterjetObservation.from_table({"sensor_x_m": 1.55}, grid, dpm)


def test_legacy_ledger_checkpoint_adds_only_zero_separator_inventory():
    from jaxwind.io.checkpoint import _flatten, _restore

    l = initial_ledger(jnp.float64)._replace(injected=jnp.arange(6, dtype=jnp.float64))
    arrays = {}
    node = _flatten(l, "state", arrays)
    del node["fields"]["collected"]
    restored = _restore(node, arrays, l)
    for a, b in zip(restored, l):
        np.testing.assert_array_equal(a, b)


def test_coupled_tunnel_separator_restart_and_carrier_water_balance(tmp_path):
    """Actual tunnel driver, with a source near the plate to exercise collection."""
    from copy import deepcopy
    from pathlib import Path

    from jaxwind.config.document import load_case
    from jaxwind.io.checkpoint import load_checkpoint, save_checkpoint
    from jaxwind.simulation.api import RunControls
    from jaxwind.simulation.water_spray_benchmark import build_simulation

    path = (
        Path(__file__).resolve().parents[2]
        / "cases/FluentDPMWater/waterjet_tunnel_smoke.toml"
    )
    case = load_case(path)
    doc = deepcopy(case.document)
    source = doc["physics"]["water_spray"]
    source.update(streamwise_offset_m=1.899, mass_flow_rate_kg_s=1e-5, ramp_time_s=0.0)
    case = replace(case, document=doc)
    simulation = build_simulation(case)
    initial = simulation.initial_state
    once = simulation.advance(initial, RunControls(1, 0.001))
    assert once.accepted and once.dpm_ledger.collected[0] > 0
    assert once.dpm_ledger.escaped[0] == 0
    p = tmp_path / "tunnel.npz"
    save_checkpoint(p, once, metadata={"fingerprint": case.fingerprint})
    resumed, _, _ = load_checkpoint(p, initial, fingerprint=case.fingerprint)
    after = simulation.advance(once, RunControls(1, 0.002))
    restarted = simulation.advance(resumed, RunControls(1, 0.002))
    for a, b in zip(jax.tree.leaves(after), jax.tree.leaves(restarted)):
        np.testing.assert_array_equal(a, b)
    q0 = initial.moisture.vapor
    rho = doc["physics"]["moisture"]["dry_air_density_kg_m3"]
    gain = rho * jnp.sum((after.moisture.vapor - q0) * simulation.grid.cell_volumes)
    np.testing.assert_allclose(
        gain - after.transport_water, after.dpm_ledger.evaporated_mass, atol=2e-18
    )
    readings = simulation.state_diagnostics(after)
    assert abs(float(readings["dpm_water_budget_error_kg"])) < 1e-20
    assert readings["separator_collection_temperature_valid"]
    assert 0 < readings["separator_collection_temperature_c"] < 100
    for i in range(9):
        assert np.isfinite(readings[f"sensor_{i}_wbt_c"])
    np.testing.assert_array_equal(after.velocity.y[:, [0, -1], :], 0)
    np.testing.assert_array_equal(after.velocity.z[[0, -1], :, :], 0)


def test_tunnel_les_length_uses_all_four_walls():
    from jaxwind.sgs import FluentSmagorinsky

    grid = UniformGrid(2, 4, 4, 100.0, 1.0, 1.0)
    model = FluentSmagorinsky(coefficient=0.9, tunnel_walls=True)
    distance = np.minimum.outer(
        np.minimum(np.array(grid.z_centers), 1 - np.array(grid.z_centers)),
        np.minimum(np.array(grid.y_centers), 1 - np.array(grid.y_centers)),
    )
    expected = np.broadcast_to(0.41 * distance[:, :, None], (4, 4, 2))
    np.testing.assert_allclose(
        np.broadcast_to(model.length_scale(grid, jnp.float64), (4, 4, 2)),
        expected,
        rtol=1e-14,
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"eliminator_x_m": float("nan")},
        {"injection_geometry": "hollow-cone"},
        {"cone_outer_half_angle_degrees": 20.0},
        {
            "injection_geometry": "hollow-cone",
            "cone_outer_half_angle_degrees": 20.0,
            "capacity": 4,
        },
        {"cone_azimuthal_points": 3},
        {"cone_polar_points": True},
    ],
)
def test_invalid_tunnel_source_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        DPMOptions((1e-4,), (1.0,), (1.0, 0.0, 0.0), 300.0, **settings)


def test_spatial_prevalidation_rejects_bad_remote_cell_atomically():
    grid, source, gas, u, les = fixture()
    p, ledger, _ = inject_parcels(
        initial_parcels(source.dpm, jnp.float64),
        initial_ledger(jnp.float64),
        source,
        (1.0, 2.0, 2.0),
        0.0,
        0.01,
        M,
    )
    bad = gas._replace(vapor_density=gas.vapor_density.at[-1, -1, -1].set(-1.0))
    step = jax.jit(
        build_spatial_step(
            grid, source, M, 101325.0, periodic_y=False, tunnel_walls=True
        )
    )
    r = step(bad, u, p, ledger, 0.01, les)
    assert not r.accepted
    for a, b in zip(
        jax.tree.leaves((bad, u, p, ledger)),
        jax.tree.leaves((r.gas, r.velocity, r.parcels, r.ledger)),
    ):
        np.testing.assert_array_equal(a, b)


def test_local_exchange_validation_agrees_with_full_field_validation():
    from types import SimpleNamespace

    from jaxwind.fluent_dpm_source import build_dpm_cell_exchange, material_config
    from jaxwind.spray_core import WaterCoreBins

    grid, _, gas, u, _ = fixture()
    liquid = WaterCoreBins(
        jnp.array([M.liquid_density * jnp.pi / 6 * (200e-6) ** 3]),
        jnp.array([1e4]),
        jnp.array([[20.0], [2.0], [-1.0]]),
        jnp.array([300.0]),
    )
    outputs = []
    for local in (False, True):
        step = jax.jit(
            build_dpm_cell_exchange(
                grid,
                (0, 0, 1),
                material_config(M, 101325.0),
                SimpleNamespace(
                    liquid_heat_capacity=M.liquid_cp, air_dynamic_viscosity=M.viscosity
                ),
                periodic_x=False,
                periodic_y=False,
                drag_heat_fraction=1.0,
                prevalidated_fields=local,
            )
        )
        result = step(gas, u, jnp.zeros_like(gas.dry_density), liquid, 0.01)
        assert result.accepted
        outputs.append(result)
    for a, b in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1])):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)
