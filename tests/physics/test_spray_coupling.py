"""Conservation of disjoint gas inventories, including mixing kinetic energy."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.spray_coupling import (
    GasInventory,
    mean_kinetic_energy,
    merge_gas_inventories,
    partition_gas_inventory,
    transfer_gas_mass,
)

jax.config.update("jax_enable_x64", True)


def gas(mass, velocity, h=300000.0, y=(0.98, 0.02), k=0.7):
    mass = jnp.asarray(mass)
    velocity = jnp.asarray(velocity)
    if mass.ndim:
        velocity = velocity[:, None] * jnp.ones_like(mass)
    composition = jnp.asarray(y).reshape((len(y),) + (1,) * mass.ndim)
    return GasInventory(mass, mass * velocity, mass * h, mass * composition, mass * k)


def totals(*inventories):
    return (
        sum(q.mass for q in inventories),
        sum(q.momentum for q in inventories),
        sum(q.species for q in inventories),
        sum(q.enthalpy for q in inventories),
        sum(q.unresolved_energy + mean_kinetic_energy(q) for q in inventories),
    )


def assert_balance(before, after):
    for a, b in zip(totals(*before), totals(*after)):
        np.testing.assert_allclose(a, b, rtol=3e-14, atol=1e-12)


def test_closed_form_mixing_energy_and_species_not_arithmetic_velocity_average():
    a = gas(2.0, [3.0, -1.0, 0.0], y=(0.8, 0.2), k=1.0)
    b = gas(1.0, [-3.0, 2.0, 0.0], h=250000.0, y=(0.4, 0.6), k=2.0)
    c, loss = jax.jit(merge_gas_inventories)(a, b)
    np.testing.assert_allclose(c.momentum / c.mass, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(loss, 15.0)
    np.testing.assert_allclose(c.unresolved_energy, 19.0)
    np.testing.assert_allclose(c.species, [2.0, 1.0])
    assert_balance((a, b), (c,))


def test_entrainment_exhaustion_is_reported_cellwise_without_negative_inventory():
    ambient = gas([0.0, 1.0, 2.0, 3.0], [2.0, 1.0, -3.0])
    jet = gas([1.0, 0.0, 4.0, 2.0], [15.0, 0.0, 1.0], y=(0.5, 0.5))
    result = jax.jit(transfer_gas_mass)(ambient, jet, jnp.array([2.0, 0.25, 2.0, 10.0]))
    np.testing.assert_allclose(result.transferred_mass, [0.0, 0.25, 2.0, 3.0])
    np.testing.assert_allclose(result.unmet_mass, [2.0, 0.0, 0.0, 7.0])
    assert_balance((ambient, jet), (result.donor, result.receiver))
    for q in (result.donor, result.receiver):
        assert np.all(q.mass >= 0) and np.all(q.species >= 0)
        assert np.all(q.unresolved_energy >= 0)
        np.testing.assert_allclose(jnp.sum(q.species, axis=0), q.mass)
    np.testing.assert_allclose(
        result.donor.momentum[:, 1] / result.donor.mass[1], [2.0, 1.0, -3.0]
    )


def test_partition_repeated_handoff_and_empty_inventory_do_not_double_count():
    original = gas([1.0, 2.0, 3.0, 4.0], [8.0, -2.0, 1.0])
    ambient, jet = partition_gas_inventory(original, jnp.array([0.0, 0.1, 0.8, 1.0]))
    assert_balance((original,), (ambient, jet))
    first = transfer_gas_mass(jet, ambient, jet.mass)
    second = transfer_gas_mass(first.donor, first.receiver, first.donor.mass)
    for q in (first, second):
        for actual, expected in zip(q.receiver, original):
            np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-12)
        for field in q.donor:
            np.testing.assert_array_equal(field, jnp.zeros_like(field))
        np.testing.assert_allclose(q.mixing_energy, 0, atol=1e-28)
    empty = gas(0.0, [0.0, 0.0, 0.0])
    merged, energy = merge_gas_inventories(empty, empty)
    assert float(mean_kinetic_energy(merged)) == float(energy) == 0.0
    assert all(np.all(np.isfinite(q)) for q in merged)


def test_closed_transfer_cycle_conserves_total_energy_while_mean_motion_dissipates():
    ambient = gas([1.0, 2.0, 3.0], [0.0, 1.0, 0.0])
    jet = gas([0.1, 0.2, 0.3], [20.0, -2.0, 4.0], h=260000.0, y=(0.7, 0.3))
    original = (ambient, jet)
    for _ in range(10):
        e = transfer_gas_mass(ambient, jet, 0.1 * ambient.mass)
        h = transfer_gas_mass(e.receiver, e.donor, 0.15 * e.receiver.mass)
        ambient, jet = h.receiver, h.donor
        assert_balance(original, (ambient, jet))
    final, _ = merge_gas_inventories(ambient, jet)
    direct, _ = merge_gas_inventories(*original)
    for a, b in zip(final, direct):
        np.testing.assert_allclose(a, b, rtol=3e-14, atol=1e-12)


def test_mixing_energy_is_galilean_invariant_and_rotation_covariant():
    a = gas([1.0, 2.0], [3.0, 4.0, 5.0])
    b = gas([2.0, 1.0], [-1.0, 2.0, 8.0])
    c, loss = merge_gas_inventories(a, b)
    boost = jnp.array([1000.0, -2000.0, 3000.0])[:, None]
    boosted = [q._replace(momentum=q.momentum + q.mass * boost) for q in (a, b)]
    bc, bloss = merge_gas_inventories(*boosted)
    np.testing.assert_allclose(bloss, loss, rtol=1e-14)
    assert_balance(boosted, (bc,))
    rotation = jnp.asarray(
        np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))[0]
    )
    rc, rloss = merge_gas_inventories(
        *(q._replace(momentum=rotation @ q.momentum) for q in (a, b))
    )
    np.testing.assert_allclose(rloss, loss, rtol=1e-14)
    np.testing.assert_allclose(rc.momentum, rotation @ c.momentum, atol=1e-14)


def test_nonlocal_shell_withdrawal_conserves_global_inventory_in_sheared_gas():
    from jaxwind.spray_coupling import gather_gas_inventory, withdraw_gas_to_core

    rng = np.random.default_rng(21)
    mass = jnp.asarray(rng.uniform(0.2, 2.0, size=(2, 3, 4)))
    velocity = (
        jnp.asarray(rng.normal(size=(3, 2, 3, 4)))
        + jnp.array([100.0, -200.0, 300.0])[:, None, None, None]
    )
    ambient = GasInventory(
        mass,
        mass * velocity,
        mass * 300000.0,
        jnp.stack((0.9 * mass, 0.1 * mass)),
        0.2 * mass,
    )
    core = gas(0.05, [120.0, -195.0, 290.0], y=(0.7, 0.3))
    requests = 0.3 * mass
    result = jax.jit(withdraw_gas_to_core)(ambient, core, requests)
    before, _ = gather_gas_inventory(ambient)
    after, _ = gather_gas_inventory(result.donor)
    assert_balance((before, core), (after, result.receiver))
    np.testing.assert_allclose(result.receiver.mass, core.mass + jnp.sum(requests))
    np.testing.assert_allclose(result.donor.momentum, result.donor.mass * velocity)
    np.testing.assert_array_equal(result.unmet_mass, 0.0)
    assert float(result.mixing_energy) > 0
    assert float(result.receiver.unresolved_energy) > float(core.unresolved_energy)


def test_gather_matches_successive_merges_without_double_counting_variance():
    from jaxwind.spray_coupling import gather_gas_inventory

    samples = [
        gas(1.0, [1.0, 2.0, 3.0]),
        gas(2.0, [-2.0, 1.0, 0.0]),
        gas(3.0, [2.0, 4.0, -1.0]),
    ]
    stacked = GasInventory(
        *(jnp.stack([q[i] for q in samples], axis=-1) for i in range(5))
    )
    collected, _ = jax.jit(gather_gas_inventory)(stacked)
    sequential = samples[0]
    for q in samples[1:]:
        sequential, _ = merge_gas_inventories(sequential, q)
    for a, b in zip(collected, sequential):
        np.testing.assert_allclose(a, b, rtol=2e-14)
