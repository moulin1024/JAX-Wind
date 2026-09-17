"""Finite-gas phase conservation, stiff relaxation, and convergence."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from jaxwind.physics.moisture import MoistureConfig, WaterDropletProperties
from jaxwind.spray_core import (
    WaterCoreBins,
    advance_water_core,
    core_gas_temperature,
    relax_core_drag,
)
from jaxwind.spray_coupling import (
    GasInventory,
    mean_kinetic_energy,
    transfer_gas_mass,
)

jax.config.update("jax_enable_x64", True)
CONFIG = MoistureConfig()
PROPS = WaterDropletProperties()


def gas(mass=0.001, temperature=310.0, velocity=(1.0, -2.0, 0.5), vapor=0.01):
    dry = mass / (1 + vapor)
    species = jnp.array([dry, dry * vapor])
    enthalpy = (
        dry * CONFIG.dry_air_heat_capacity * (temperature - CONFIG.freezing_temperature)
    )
    enthalpy += species[1] * CONFIG.water_vapor_latent_heat
    return GasInventory(
        jnp.array(mass),
        mass * jnp.array(velocity),
        jnp.array(enthalpy),
        species,
        jnp.array(0.003),
    )


def bins():
    diameter = jnp.array([20e-6, 70e-6, 180e-6])
    mass = CONFIG.water_density * jnp.pi * diameter**3 / 6
    return WaterCoreBins(
        mass,
        jnp.array([1e6, 3e5, 1e4]),
        jnp.array([[18.0, 7.0, 11.0], [2.0, -1.0, 3.0], [-2.0, 2.0, 0.0]]),
        jnp.array([295.0, 298.0, 300.0]),
    )


def budget(g, liquid):
    masses = liquid.mass * liquid.multiplicity
    heat = jnp.sum(
        masses
        * PROPS.liquid_heat_capacity
        * (liquid.temperature - CONFIG.freezing_temperature)
    )
    kinetic = 0.5 * jnp.sum(masses * jnp.sum(liquid.velocity**2, axis=0))
    return (
        g.species[0],
        g.species[1] + jnp.sum(masses),
        g.momentum + jnp.sum(masses * liquid.velocity, axis=1),
        g.enthalpy + heat + mean_kinetic_energy(g) + kinetic + g.unresolved_energy,
    )


def assert_budget(before, after):
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b, rtol=3e-12, atol=2e-13)


def test_stiff_heavily_loaded_single_bin_exact_relaxation_and_energy():
    g = gas(mass=0.01)
    b = WaterCoreBins(
        jnp.array([0.1]),
        jnp.ones(1),
        jnp.array([[21.0], [3.0], [-1.0]]),
        jnp.array([300.0]),
    )
    for dt in (0.0, 1e-7, 0.01, 1e3):
        after_g, after_b, loss = jax.jit(relax_core_drag)(g, b, jnp.array([17.0]), dt)
        u, v = np.asarray(g.momentum / g.mass), np.asarray(b.velocity[:, 0])
        center = (0.01 * u + 0.1 * v) / 0.11
        slip = (v - u) * np.exp(-17 * 11 * dt)
        np.testing.assert_allclose(
            after_g.momentum / g.mass, center - 0.1 / 0.11 * slip, atol=1e-13
        )
        np.testing.assert_allclose(
            after_b.velocity[:, 0], center + 0.01 / 0.11 * slip, atol=1e-13
        )
        expected_loss = (
            0.5
            * 0.01
            * 0.1
            / 0.11
            * np.sum((v - u) ** 2)
            * (-np.expm1(-2 * 17 * 11 * dt))
        )
        np.testing.assert_allclose(loss, expected_loss, atol=2e-14)
        assert_budget(
            budget(g, b),
            budget(
                after_g._replace(unresolved_energy=g.unresolved_energy + loss), after_b
            ),
        )


def test_multibin_drag_converges_to_independent_matrix_exponential():
    g, b = gas(), bins()
    rates = jnp.array([13.0, 4.0, 1.0])
    masses = np.asarray(b.mass * b.multiplicity)
    matrix = np.zeros((4, 4))
    matrix[0, 0] = -np.dot(masses, rates) / float(g.mass)
    matrix[0, 1:] = masses * rates / float(g.mass)
    matrix[1:, 0] = rates
    matrix[1:, 1:] = -np.diag(rates)
    initial = np.vstack((np.asarray(g.momentum / g.mass), np.asarray(b.velocity).T))
    reference = expm(0.2 * matrix) @ initial

    @jax.jit
    def evolve(n):
        def step(_, state):
            a, c, energy = state
            a, c, lost = relax_core_drag(a, c, rates, 0.2 / n)
            return a, c, energy + lost

        return jax.lax.fori_loop(0, n, step, (g, b, jnp.array(0.0)))

    errors = []
    for n in (8, 16, 32):
        a, c, energy = evolve(n)
        value = np.vstack((np.asarray(a.momentum / a.mass), np.asarray(c.velocity).T))
        errors.append(np.max(np.abs(value - reference)))
        assert_budget(
            budget(g, b),
            budget(a._replace(unresolved_energy=g.unresolved_energy + energy), c),
        )
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8


def test_drag_galilean_covariance_permutation_convergence_and_empty_bin():
    g, b = gas(), bins()
    b = b._replace(multiplicity=b.multiplicity.at[1].set(0))
    rate, dt = jnp.array([5.0, 20.0, 1.0]), 0.001
    a, c, loss = relax_core_drag(g, b, rate, dt)
    boost = jnp.array([1000.0, -2000.0, 3000.0])
    ag, bc, boosted_loss = relax_core_drag(
        g._replace(momentum=g.momentum + g.mass * boost),
        b._replace(velocity=b.velocity + boost[:, None]),
        rate,
        dt,
    )
    np.testing.assert_allclose(ag.momentum, a.momentum + a.mass * boost, atol=1e-14)
    np.testing.assert_allclose(bc.velocity, c.velocity + boost[:, None], atol=1e-12)
    np.testing.assert_allclose(boosted_loss, loss, rtol=3e-13)
    np.testing.assert_array_equal(c.velocity[:, 1], b.velocity[:, 1])
    # Bin ordering is a splitting error, not a source of lost conservation.
    reverse = jnp.array([2, 1, 0])
    reversed_b = jax.tree.map(lambda x: x[..., reverse], b)
    ar, cr, lr = relax_core_drag(g, reversed_b, rate[reverse], dt / 4)
    _af, cf, lf = relax_core_drag(g, b, rate, dt / 4)
    assert_budget(
        budget(g, reversed_b),
        budget(ar._replace(unresolved_energy=g.unresolved_energy + lr), cr),
    )
    assert np.max(np.abs(cr.velocity[:, reverse] - cf.velocity)) < 1e-6
    assert float(lf) >= 0


def test_evaporating_core_preserves_species_momentum_and_total_energy_for_all_partitions():
    g, b = gas(), bins()
    g = g._replace(unresolved_energy=jnp.array(0.0))
    initial = budget(g, b)
    for fraction in (0.0, 0.4, 1.0):
        advance = jax.jit(
            lambda a, c, fraction=fraction: advance_water_core(
                a, c, 0.0001, CONFIG, PROPS, drag_heat_fraction=fraction
            )
        )
        result = advance(g, b)
        assert bool(result.accepted)
        assert float(result.evaporated_mass) > 0
        assert float(result.drag_energy) > 0
        assert float(result.vapor_mixing_energy) > 0
        assert float(result.gas.unresolved_energy) >= 0
        np.testing.assert_allclose(
            result.gas.mass - g.mass, result.evaporated_mass, rtol=1e-10
        )
        np.testing.assert_allclose(
            jnp.sum(result.gas.species), result.gas.mass, rtol=2e-15
        )
        np.testing.assert_allclose(
            result.gas.unresolved_energy - g.unresolved_energy,
            (1 - fraction) * (result.drag_energy + result.vapor_mixing_energy),
            atol=1e-16,
        )
        assert_budget(initial, budget(result.gas, result.liquid))


def test_thermal_inventory_exhaustion_rejects_entire_exchange_without_clipping():
    g = gas(mass=1e-10, temperature=274.0, velocity=(0.0, 0.0, 0.0), vapor=0.0)
    b = bins()._replace(temperature=jnp.full(3, 273.15), velocity=jnp.zeros((3, 3)))
    result = jax.jit(
        lambda: advance_water_core(g, b, 0.1, CONFIG, PROPS, drag_heat_fraction=0.0)
    )()
    assert not bool(result.accepted)
    assert float(result.candidate_gas_temperature) < CONFIG.freezing_temperature
    assert float(result.evaporated_mass) > 0
    for a, c in zip((*result.gas, *result.liquid), (*g, *b)):
        np.testing.assert_array_equal(a, c)


def test_withdraw_exchange_handoff_cycle_conserves_and_handoff_cannot_repeat():
    ambient, b = gas(mass=0.01), bins()
    empty = GasInventory(*(jnp.zeros_like(q) for q in ambient))
    withdrawal = transfer_gas_mass(ambient, empty, jnp.array(0.001))
    result = advance_water_core(
        withdrawal.receiver, b, 0.0001, CONFIG, PROPS, drag_heat_fraction=0.0
    )
    assert bool(result.accepted)
    handoff = transfer_gas_mass(result.gas, withdrawal.donor, result.gas.mass)
    assert_budget(budget(ambient, b), budget(handoff.receiver, result.liquid))
    repeat = transfer_gas_mass(handoff.donor, handoff.receiver, handoff.donor.mass)
    for a, c in zip(repeat.receiver, handoff.receiver):
        np.testing.assert_array_equal(a, c)
    assert float(repeat.transferred_mass) == 0.0


def test_coupled_evaporation_drag_timestep_refinement_and_zero_step():
    g, b = gas(), bins()

    @jax.jit
    def run(n):
        def step(_, state):
            a, c, ok = state
            r = advance_water_core(
                a, c, 0.002 / n, CONFIG, PROPS, drag_heat_fraction=0.0
            )
            return r.gas, r.liquid, ok & r.accepted

        return jax.lax.fori_loop(0, n, step, (g, b, jnp.array(True)))

    reference_g, reference_b, ok = run(2048)
    assert bool(ok)

    def observable(a, c):
        return np.r_[
            float(core_gas_temperature(a, CONFIG)),
            np.asarray(c.temperature),
            np.asarray(c.velocity).ravel(),
            np.asarray(c.mass / b.mass),
        ]

    reference = observable(reference_g, reference_b)
    errors = []
    for n in (32, 64, 128):
        a, c, ok = run(n)
        assert bool(ok)
        assert_budget(budget(g, b), budget(a, c))
        errors.append(np.linalg.norm(observable(a, c) - reference))
    assert min(errors[0] / errors[1], errors[1] / errors[2]) > 1.8
    zero = advance_water_core(g, b, 0.0, CONFIG, PROPS, drag_heat_fraction=0.5)
    assert bool(zero.accepted)
    for a, c in zip((*zero.gas, *zero.liquid), (*g, *b)):
        np.testing.assert_array_equal(a, c)
