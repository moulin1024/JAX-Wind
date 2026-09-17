"""Actual MAC-inventory source conservation and cross-phase atomic rollback."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.domain import UniformGrid
from jaxwind.numerics.poisson import build_pressure_poisson
from jaxwind.physics.moisture import MoistureConfig, WaterDropletProperties
from jaxwind.spray_cell_exchange import build_cell_exchange, relax_staggered_drag
from jaxwind.spray_cell_step import build_cell_spray_step
from jaxwind.spray_core import WaterCoreBins
from jaxwind.spray_low_mach import moist_gas_from_primitive
from jaxwind.state import StaggeredVelocity

jax.config.update("jax_enable_x64", True)
C, P = MoistureConfig(), WaterDropletProperties()


def setup(nonuniform=False):
    grid = UniformGrid(6, 3, 4, 0.6, 0.3, 0.4)
    shape = (4, 3, 6)
    temperature = jnp.full(shape, 310.0)
    if nonuniform:
        temperature += jnp.arange(6)[None, None, :] * 3
    gas = moist_gas_from_primitive(temperature, jnp.full(shape, 0.01), C)
    u = StaggeredVelocity(
        jnp.full((4, 3, 7), 0.2), jnp.zeros(shape), jnp.zeros((5, 3, 6))
    )
    if nonuniform:
        u = u._replace(x=u.x + jnp.arange(7)[None, None, :] * 0.03)
    diam = jnp.array([20e-6, 70e-6, 180e-6])
    liquid = WaterCoreBins(
        C.water_density * jnp.pi * diam**3 / 6,
        jnp.array([1e4, 3e3, 100.0]),
        jnp.array([[18.0, 7.0, 11.0], [2.0, -1.0, 3.0], [-2.0, 2.0, 0.0]]),
        jnp.array([295.0, 298.0, 300.0]),
    )
    return grid, gas, u, jnp.full(shape, 0.4), liquid


def budget(grid, gas, u, e, liquid):
    # Independent half-volume quadrature and adjacent-cell density construction.
    rho = np.asarray(gas.dry_density + gas.vapor_density)
    volume = grid.dx * grid.dy * grid.dz
    momentum = []
    kinetic = 0.0
    for axis, face in zip((2, 1, 0), u):
        if axis == 1:
            mass = (rho + np.roll(rho, 1, axis)) * volume / 2
        else:
            low = np.take(rho, [0], axis=axis)
            high = np.take(rho, [-1], axis=axis)
            a = np.take(rho, np.arange(rho.shape[axis] - 1), axis=axis)
            b = np.take(rho, np.arange(1, rho.shape[axis]), axis=axis)
            mass = np.concatenate((low, a + b, high), axis=axis) * volume / 2
        momentum.append(np.sum(mass * np.asarray(face)))
        kinetic += 0.5 * np.sum(mass * np.asarray(face) ** 2)
    m = np.asarray(liquid.mass * liquid.multiplicity)
    momentum = np.asarray(momentum) + np.sum(m * np.asarray(liquid.velocity), axis=1)
    kinetic += 0.5 * np.sum(m * np.sum(np.asarray(liquid.velocity) ** 2, axis=0))
    heat = np.sum(
        m
        * P.liquid_heat_capacity
        * (np.asarray(liquid.temperature) - C.freezing_temperature)
    )
    return (
        np.sum(gas.dry_density) * volume,
        np.sum(gas.vapor_density) * volume + np.sum(m),
        momentum,
        np.sum(gas.enthalpy_density + e) * volume + heat + kinetic,
    )


@pytest.mark.parametrize("dt", [1e-7, 0.1, 100.0])
def test_exact_single_bin_staggered_drag(dt):
    masses = jnp.array([[0.01, 0.03], [0.02, 0.007], [0.04, 0.012]])
    faces = jnp.array([[1.0, 3.0], [-2.0, 1.0], [0.5, -0.5]])
    liquid = WaterCoreBins(
        jnp.array([0.05]),
        jnp.ones(1),
        jnp.array([[20.0], [4.0], [-3.0]]),
        jnp.array([300.0]),
    )
    fn = jax.jit(relax_staggered_drag)
    out, drops, impulse, loss = fn(masses, faces, liquid, jnp.array([17.0]), dt)
    effective = 1 / np.sum(0.25 / np.asarray(masses), axis=1)
    slip = np.asarray(liquid.velocity[:, 0]) - np.mean(faces, axis=1)
    change = (
        effective
        / (effective + 0.05)
        * slip
        * (-np.expm1(-17 * (1 + 0.05 / effective) * dt))
    )
    np.testing.assert_allclose(drops[:, 0], liquid.velocity[:, 0] - change, atol=2e-14)
    np.testing.assert_allclose(
        out, faces + 0.5 * (0.05 * change)[:, None] / masses, atol=2e-14
    )
    np.testing.assert_allclose(impulse, 0.05 * change, atol=2e-15)
    before = 0.5 * jnp.sum(masses * faces**2) + 0.025 * jnp.sum(liquid.velocity**2)
    after = 0.5 * jnp.sum(masses * out**2) + 0.025 * jnp.sum(drops**2)
    np.testing.assert_allclose(before - after, loss, atol=1e-14)
    boost = jnp.array([10.0, -2.0, 3.0])
    out2, drop2, impulse2, loss2 = fn(
        masses,
        faces + boost[:, None],
        liquid._replace(velocity=liquid.velocity + boost[:, None]),
        jnp.array([17.0]),
        dt,
    )
    np.testing.assert_allclose(out2, out + boost[:, None], atol=2e-14)
    np.testing.assert_allclose(drop2, drops + boost[:, None], atol=2e-14)
    np.testing.assert_allclose(impulse2, impulse, atol=1e-14)
    np.testing.assert_allclose(loss2, loss, atol=1e-14)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("x", [0, 3])
def test_source_actual_global_inventory_budget(fraction, x):
    grid, gas, u, e, liquid = setup(True)
    fn = jax.jit(
        build_cell_exchange(
            grid,
            (1, 1, x),
            C,
            P,
            periodic_x=False,
            periodic_y=True,
            drag_heat_fraction=fraction,
        )
    )
    r = fn(gas, u, e, liquid, 1e-4)
    assert bool(r.accepted), (r.relative_energy_error, r.candidate_temperature)
    assert r.evaporated_mass > 0 and r.drag_energy > 0 and r.vapor_mixing_energy > 0
    for a, b in zip(
        budget(grid, gas, u, e, liquid),
        budget(grid, r.gas, r.velocity, r.unresolved_density, r.liquid),
    ):
        np.testing.assert_allclose(a, b, rtol=3e-13, atol=1e-13)
    np.testing.assert_allclose(
        jnp.sum(r.unresolved_increment) * grid.dx * grid.dy * grid.dz,
        (1 - fraction) * (r.drag_energy + r.vapor_mixing_energy),
        rtol=1e-13,
    )
    print(
        "source", x, fraction, float(r.relative_energy_error), float(r.evaporated_mass)
    )


def coupled(grid, gas, **controls):
    poisson = build_pressure_poisson(
        grid,
        backend="gmg",
        periodic_x=False,
        periodic_y=True,
        open_x_low=True,
        dtype="float64",
    )
    ambient = jnp.stack(gas)[..., 0] / (gas.dry_density + gas.vapor_density)[..., 0]
    return jax.jit(
        build_cell_spray_step(
            poisson,
            (1, 1, 3),
            C,
            P,
            ambient,
            jnp.zeros((3, grid.nz, grid.ny)),
            jnp.zeros((grid.nz, grid.ny)),
            drag_heat_fraction=0.5,
            **controls,
        )
    )


def assert_tree_equal(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_atomic_acceptance_and_reservoir_boundary_budget():
    grid, gas, u, e, liquid = setup()
    dt = 1e-4
    r = coupled(grid, gas)(gas, u, e, liquid, dt)
    assert bool(r.accepted), r.carrier[15:]
    assert bool(r.source_accepted) and r.evaporated_mass > 0
    volume = grid.dx * grid.dy * grid.dz
    outflow = (
        jnp.sum(r.unresolved_flux.x[..., -1] - r.unresolved_flux.x[..., 0])
        * grid.dy
        * grid.dz
    )
    np.testing.assert_allclose(
        jnp.sum(r.unresolved_density - e) * volume + dt * outflow,
        0.5 * (r.drag_energy + r.vapor_mixing_energy),
        rtol=1e-10,
        atol=1e-14,
    )
    vapor_boundary = (
        jnp.sum(
            r.carrier.scalar_fluxes.x[1, ..., -1] - r.carrier.scalar_fluxes.x[1, ..., 0]
        )
        * grid.dy
        * grid.dz
    )
    np.testing.assert_allclose(
        jnp.sum(r.carrier.gas.vapor_density - gas.vapor_density) * volume
        + dt * vapor_boundary,
        r.evaporated_mass,
        rtol=1e-8,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        jnp.sum((liquid.mass - r.liquid.mass) * liquid.multiplicity),
        r.evaporated_mass,
        rtol=1e-12,
    )
    print(
        "atomic",
        int(r.carrier.iterations),
        float(r.source_relative_energy_error),
        float(r.carrier.momentum_error),
    )


@pytest.mark.parametrize(
    "controls", [{"max_iterations": 1}, {"pressure_max_iterations": 1}]
)
def test_carrier_rejection_restores_liquid_and_all_committed_ledgers(controls):
    grid, gas, u, e, liquid = setup()
    fn = coupled(grid, gas, **controls)
    r = fn(gas, u, e, liquid, 1e-4)
    assert bool(r.source_accepted) and not bool(r.accepted)
    assert not bool(r.carrier.accepted)
    assert_tree_equal(r.carrier.gas, gas)
    assert_tree_equal(r.carrier.velocity, u)
    assert_tree_equal(r.liquid, liquid)
    assert_tree_equal(r.unresolved_density, e)
    for a in jax.tree.leaves(
        (
            r.carrier[3:15],
            r.unresolved_flux,
            r.evaporated_mass,
            r.drag_energy,
            r.vapor_mixing_energy,
        )
    ):
        np.testing.assert_array_equal(a, np.zeros_like(a))
    # A retry from the restored state must not apply the attempted source twice.
    retry = coupled(grid, gas)(
        r.carrier.gas, r.carrier.velocity, r.unresolved_density, r.liquid, 1e-4
    )
    fresh = coupled(grid, gas)(gas, u, e, liquid, 1e-4)
    assert bool(retry.accepted)
    assert_tree_equal(retry, fresh)


def test_source_rejection_restores_all_phases():
    grid, gas, u, e, liquid = setup()
    liquid = liquid._replace(temperature=liquid.temperature.at[0].set(270.0))
    r = coupled(grid, gas)(gas, u, e, liquid, 1e-4)
    assert not bool(r.source_accepted) and not bool(r.accepted)
    assert_tree_equal(r.carrier.gas, gas)
    assert_tree_equal(r.carrier.velocity, u)
    assert_tree_equal(r.liquid, liquid)
    assert_tree_equal(r.unresolved_density, e)
