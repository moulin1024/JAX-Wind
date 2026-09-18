"""Spatial DPM budgets, resolved LES inputs, and coarse-cell carrier integration."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind import (
    FREE_SLIP,
    OPEN,
    Boundaries,
    FlowModel,
    InflowPlane,
    StaggeredVelocity,
    Wall,
    build_open_atmospheric_step,
    build_pressure_poisson,
    initial_atmospheric_solution,
)
from jaxwind.config.fluent_dpm import DPMOptions, load_dpm
from jaxwind.config.moisture import AtmosphericMoistureOptions, WaterSprayOptions
from jaxwind.domain import UniformGrid
from jaxwind.fluent_dpm_atmosphere import (
    build_dpm_atmospheric_step,
    initialize_dpm,
    specific_enthalpy,
    temperature_from_enthalpy,
)
from jaxwind.fluent_dpm_les import DPMLESFields, dpm_les_fields
from jaxwind.fluent_dpm_source import build_dpm_cell_exchange, material_config
from jaxwind.fluent_dpm_spatial import (
    build_spatial_step,
    initial_ledger,
    initial_parcels,
    inject_parcels,
)
from jaxwind.io.checkpoint import load_checkpoint, save_checkpoint
from jaxwind.numerics.poisson import project
from jaxwind.physics.fluent_dpm import DPMWaterMaterial
from jaxwind.scalar import PassiveScalar
from jaxwind.sgs import AnisotropicMinimumDissipation, FluentSmagorinsky, eddy_viscosity
from jaxwind.spray_core import WaterCoreBins
from jaxwind.spray_low_mach import MoistGasFields

jax.config.update("jax_enable_x64", True)
M = DPMWaterMaterial()
B = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN)


def setup():
    grid = UniformGrid(4, 2, 4, 64.0, 32.0, 16.0)  # EACH cell 16 x 16 x 4 metres
    shape = (4, 2, 4)
    dry = jnp.ones(shape)
    vapor = dry * 0.006
    gas = MoistGasFields(dry, vapor, dry * specific_enthalpy(310.0, vapor / dry, M))
    velocity = StaggeredVelocity(
        jnp.full((4, 2, 5), 8.0), jnp.zeros(shape), jnp.zeros((5, 2, 4))
    )
    options = DPMOptions(
        (200e-6,),
        (1.0,),
        (20.0, 0.0, 0.0),
        300.0,
        capacity=8,
        dispersion="off",
        tracking_substeps=2,
    )
    source = WaterSprayOptions(
        0.1, 1.0, (1.0, 1.0, 1.0), model="fluent-dpm", dpm=options
    )
    return grid, gas, velocity, source


def inventory(grid, gas, velocity, liquid):
    # Independent MAC half-volume quadrature, including boundary face volumes.
    rho = np.asarray(gas.dry_density + gas.vapor_density)
    volume = grid.dx * grid.dy * grid.dz
    momentum, kinetic = [], 0.0
    for axis, face in zip((2, 1, 0), velocity):
        if axis == 1:
            mass = (rho + np.roll(rho, 1, axis)) * volume / 2
        else:
            parts = (
                np.take(rho, [0], axis),
                np.take(rho, range(rho.shape[axis] - 1), axis)
                + np.take(rho, range(1, rho.shape[axis]), axis),
                np.take(rho, [-1], axis),
            )
            mass = np.concatenate(parts, axis) * volume / 2
        momentum.append(np.sum(mass * face))
        kinetic += 0.5 * np.sum(mass * face**2)
    m = np.asarray(liquid.mass * liquid.multiplicity)
    momentum = np.asarray(momentum) + np.sum(m * liquid.velocity, axis=1)
    kinetic += 0.5 * np.sum(m * np.sum(liquid.velocity**2, axis=0))
    energy = (
        np.sum(gas.enthalpy_density) * volume
        + np.sum(m * M.liquid_cp * (liquid.temperature - M.reference_temperature))
        + kinetic
    )
    return np.sum(gas.vapor_density) * volume + np.sum(m), momentum, energy


@pytest.mark.parametrize(
    "cell,noise", [((1, 0, 1), (0.0, 0.0, 0.0)), ((0, 0, 1), (1.0, 0.3, -0.2))]
)
def test_mac_source_conserves_actual_inventories_with_sgs_work_and_wall_reaction(
    cell, noise
):
    grid, gas, u, _ = setup()
    liquid = WaterCoreBins(
        jnp.array([M.liquid_density * jnp.pi / 6 * (200e-6) ** 3]),
        jnp.array([1e6]),
        jnp.array([[20.0], [2.0], [-1.0]]),
        jnp.array([300.0]),
    )
    fn = jax.jit(
        build_dpm_cell_exchange(
            grid,
            cell,
            material_config(M, 101325.0),
            SimpleNamespace(
                liquid_heat_capacity=M.liquid_cp, air_dynamic_viscosity=M.viscosity
            ),
            periodic_x=False,
            periodic_y=True,
            drag_heat_fraction=1.0,
        )
    )
    r = fn(
        gas,
        u,
        jnp.zeros_like(gas.dry_density),
        liquid,
        0.01,
        fluctuation=jnp.array(noise),
    )
    assert bool(r.accepted), (r.relative_energy_error, r.candidate_temperature)
    a, b = inventory(grid, gas, u, liquid), inventory(grid, r.gas, r.velocity, r.liquid)
    np.testing.assert_allclose(b[0], a[0], rtol=2e-14)
    np.testing.assert_allclose(b[1] + r.wall_impulse, a[1], atol=1e-9, rtol=2e-14)
    np.testing.assert_allclose(
        b[2] + r.wall_energy - r.stochastic_work, a[2], rtol=2e-14
    )
    assert r.evaporated_mass > 0


@pytest.mark.parametrize(
    "model,length",
    [(FluentSmagorinsky(), None), (AnisotropicMinimumDissipation(), 1.0)],
)
def test_les_uses_actual_carrier_viscosity_and_no_resolved_variance(model, length):
    grid, _, u, _ = setup()
    zero = dpm_les_fields(u, grid, B, model, amd_length=length)
    np.testing.assert_array_equal(zero.kinetic_energy, 0.0)
    # Synthetic 3D gradients exercise AMD as well as Smagorinsky.
    z, y, x = jnp.indices(u.x.shape)
    ux = u.x + 0.2 * jnp.sin(y + z) + 0.1 * jnp.cos(x)
    z, y, x = jnp.indices(u.y.shape)
    uy = 0.3 * jnp.sin(x + z)
    z, y, x = jnp.indices(u.z.shape)
    uz = 0.2 * jnp.cos(x + y) * jnp.sin(jnp.pi * z / grid.nz)
    u = StaggeredVelocity(ux, uy, uz)
    out = jax.jit(lambda u: dpm_les_fields(u, grid, B, model, amd_length=length))(u)
    np.testing.assert_allclose(
        out.viscosity, eddy_viscosity(u, grid, B, model), rtol=1e-13
    )
    assert jnp.max(out.kinetic_energy) > 0
    np.testing.assert_allclose(
        out.kinetic_energy, (out.viscosity / out.length) ** 2, rtol=1e-13
    )
    np.testing.assert_allclose(
        out.dissipation, out.kinetic_energy**1.5 / out.length, rtol=1e-13
    )
    if length is not None:
        with pytest.raises(ValueError, match="explicit"):
            dpm_les_fields(u, grid, B, model)


@pytest.mark.parametrize(
    "position,velocity,kind",
    [
        ((15.9, 8.0, 6.0), (20.0, 0.0, 0.0), "cross"),
        ((63.9, 8.0, 6.0), (20.0, 0.0, 0.0), "escape"),
        ((8.0, 8.0, 0.01), (0.0, 0.0, -2.0), "trap"),
    ],
)
def test_spatial_residence_evaporation_and_boundary_ledgers(position, velocity, kind):
    grid, gas, u, source = setup()
    source = replace(source, dpm=replace(source.dpm, injection_velocity_m_s=velocity))
    p, l, ok = inject_parcels(
        initial_parcels(source.dpm, jnp.float64),
        initial_ledger(jnp.float64),
        source,
        position,
        0.0,
        0.02,
        M,
    )
    assert ok
    zero = jnp.zeros_like(gas.dry_density)
    les = DPMLESFields(zero, zero, zero, jnp.asarray(1.0))
    fn = jax.jit(build_spatial_step(grid, source, M, 101325.0, gravity=(0.0, 0.0, 0.0)))
    r = fn(gas, u, p, l, 0.02, les)
    assert bool(r.accepted)
    assert r.ledger.evaporated_mass > 0
    np.testing.assert_allclose(
        jnp.sum(r.parcels.mass * r.parcels.multiplicity)
        + r.ledger.escaped[0]
        + r.ledger.trapped[0]
        + r.ledger.evaporated_mass,
        l.injected[0],
        rtol=2e-13,
    )
    if kind == "cross":
        assert jnp.sum(r.gas.vapor_density[..., 0] - gas.vapor_density[..., 0]) > 0
        assert jnp.sum(r.gas.vapor_density[..., 1] - gas.vapor_density[..., 1]) > 0
    else:
        ledger = r.ledger.escaped if kind == "escape" else r.ledger.trapped
        assert ledger[0] > 0
        again = fn(r.gas, r.velocity, r.parcels, r.ledger, 0.02, les)
        np.testing.assert_array_equal(again.ledger.escaped, r.ledger.escaped)
        np.testing.assert_array_equal(again.ledger.trapped, r.ledger.trapped)


def test_capacity_rejection_is_atomic_and_zero_flow_does_not_consume_slots():
    _, _, _, source = setup()
    source = replace(source, dpm=replace(source.dpm, capacity=1))
    p, l, _ = inject_parcels(
        initial_parcels(source.dpm, jnp.float64),
        initial_ledger(jnp.float64),
        source,
        (20.0, 8.0, 6.0),
        0.0,
        0.01,
        M,
    )
    p2, l2, ok = inject_parcels(p, l, source, (20.0, 8.0, 6.0), 0.01, 0.01, M)
    assert not ok
    for a, b in zip(jax.tree.leaves((p, l)), jax.tree.leaves((p2, l2))):
        np.testing.assert_array_equal(a, b)
    p2, l2, ok = inject_parcels(
        p, l, replace(source, mass_flow_rate_kg_s=0.0), (20.0, 8.0, 6.0), 0.01, 0.01, M
    )
    assert ok
    assert p2.next_id == p.next_id


def test_dpm_config_requires_explicit_amd_length():
    _, _, _, s = setup()
    with pytest.raises(ValueError, match="explicit"):
        replace(s.dpm, les_model="amd-inferred")
    assert (
        replace(
            s.dpm, les_model="amd-inferred", amd_length_scale_m=1.0
        ).amd_length_scale_m
        == 1.0
    )
    with pytest.raises(ValueError):
        load_dpm(
            {
                "diameters_m": [1e-4],
                "mass_fractions": [1.0],
                "injection_velocity_m_s": [8.0, 0.0, 0.0],
                "injection_temperature_k": 300.0,
                "unknown": 1,
            }
        )


def test_coarse_carrier_evaporates_cools_and_restarts_exactly(tmp_path):
    grid, _gas, u, source = setup()
    scalar = PassiveScalar()
    model = FlowModel(subfilter=FluentSmagorinsky())
    poisson = build_pressure_poisson(
        grid, backend="gmg", periodic_x=False, dtype="float64"
    )
    flow_step = build_open_atmospheric_step(
        grid,
        B,
        poisson,
        model,
        scalar,
        scalar_boundary="flux",
        transport_scalar=False,
        scheme="fast-rk3",
    )
    moisture = AtmosphericMoistureOptions(temperature_offset_k=310.0)
    flow = initial_atmospheric_solution(grid, u, dtype="float64")
    state = initialize_dpm(flow, moisture, source, M)
    plane = InflowPlane(u.x[..., 0], u.y[..., 0], u.z[..., 0], flow.scalar[..., 0])
    fn = jax.jit(
        build_dpm_atmospheric_step(
            flow_step,
            grid,
            B,
            model,
            scalar,
            moisture,
            source,
            (24.0, 8.0, 6.0),
            state.moisture.vapor[..., 0],
            lambda u, h, p: project(u, poisson, h)[0],
            gravity=(0.0, 0.0, 0.0),
        )
    )
    one = fn(state, 0.02, plane)
    assert one.accepted
    assert jnp.min(one.scalar) < 0
    assert one.dpm_ledger.evaporated_mass > 0
    expected_t = temperature_from_enthalpy(one.enthalpy, one.moisture.vapor, M)
    np.testing.assert_allclose(expected_t, one.scalar + 310.0, atol=1e-12)
    path = tmp_path / "restart.npz"
    save_checkpoint(path, one, metadata={"fingerprint": "test"})
    restored, _, _ = load_checkpoint(path, one, fingerprint="test")
    two, resumed = fn(one, 0.02, plane), fn(restored, 0.02, plane)
    assert two.accepted and resumed.accepted
    for a, b in zip(jax.tree.leaves(two), jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(a, b)
    rho = moisture.thermodynamics.dry_air_density
    gained = (
        rho
        * grid.dx
        * grid.dy
        * grid.dz
        * jnp.sum(two.moisture.vapor - state.moisture.vapor)
    )
    np.testing.assert_allclose(
        gained - two.transport_water,
        two.dpm_ledger.evaporated_mass,
        rtol=1e-6,
        atol=2e-10,
    )
    print(
        "coarse_carrier",
        float(jnp.min(two.scalar)),
        float(two.dpm_ledger.evaporated_mass),
    )


@pytest.mark.parametrize("change", ["open_sides", "lower_flux", "upper_flux", "no_scalar"])
def test_unsupported_atmospheric_boundaries_fail_before_compilation(change):
    grid, _gas, _u, source = setup()
    boundary = replace(B, spanwise=OPEN) if change == "open_sides" else B
    scalar = PassiveScalar(
        lower_flux=1.0 if change == "lower_flux" else 0.0,
        upper_flux=1.0 if change == "upper_flux" else 0.0,
    )
    if change == "no_scalar":
        scalar = None
    with pytest.raises(ValueError):
        build_dpm_atmospheric_step(
            None,
            grid,
            boundary,
            None,
            scalar,
            AtmosphericMoistureOptions(),
            source,
            (24.0, 8.0, 6.0),
            None,
            None,
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("injection_temperature_k", "300"),
        ("smagorinsky_constant", True),
        ("von_karman", float("nan")),
    ],
)
def test_non_numeric_or_nonfinite_dpm_settings_fail_cleanly(key, value):
    table = {
        "diameters_m": [1e-4],
        "mass_fractions": [1.0],
        "injection_velocity_m_s": [8.0, 0.0, 0.0],
        "injection_temperature_k": 300.0,
    }
    table[key] = value
    with pytest.raises(ValueError, match="finite number"):
        load_dpm(table)
