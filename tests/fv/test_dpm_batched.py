"""Batched GPU DPM conservation, events, reference convergence and RNG state."""

from dataclasses import replace
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_dpm_tunnel import M, fixture, mac_inventory

from jaxwind.fluent_dpm_spatial import (
    build_spatial_step,
    initial_ledger,
    initial_parcels,
    inject_parcels,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "position,velocity,kind",
    [
        ((1.0, 0.005, 2.0), (0.0, -2.0, 0.0), "trapped"),
        ((1.0, 3.995, 2.0), (0.0, 2.0, 0.0), "trapped"),
        ((1.0, 2.0, 0.005), (0.0, 0.0, -2.0), "trapped"),
        ((1.0, 2.0, 3.995), (0.0, 0.0, 2.0), "trapped"),
        ((3.995, 2.0, 2.0), (2.0, 0.0, 0.0), "escaped"),
        ((1.995, 2.0, 2.0), (2.0, 0.0, 0.0), "collected"),
        ((2.0, 2.0, 2.0), (2.0, 0.0, 0.0), "collected"),
    ],
)
def test_batched_boundary_inventory_and_mechanical_energy(position, velocity, kind):
    grid, source, gas, u, les = fixture(execution="batched", eliminator_x_m=2.0)
    source = replace(source, dpm=replace(source.dpm, injection_velocity_m_s=velocity))
    p, l, _ = inject_parcels(
        initial_parcels(source.dpm, jnp.float64),
        initial_ledger(jnp.float64),
        source,
        position,
        0.0,
        0.01,
        M,
    )
    p0, e0 = mac_inventory(grid, gas, u)
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
    r = run(gas, u, p, l, 0.01, les)
    assert r.accepted
    removed = getattr(r.ledger, kind)
    assert removed[0] > 0
    np.testing.assert_allclose(
        removed[0] + r.ledger.evaporated_mass, l.injected[0], atol=1e-20
    )
    momentum, energy = mac_inventory(grid, r.gas, r.velocity)
    np.testing.assert_allclose(
        momentum + removed[1:4] + r.ledger.wall_impulse,
        p0 + l.injected[1:4],
        atol=3e-14,
    )
    np.testing.assert_allclose(
        energy + removed[4] + removed[5] + r.ledger.wall_energy,
        e0 + l.injected[4] + l.injected[5],
        rtol=3e-15,
        atol=1e-8,
    )
    assert jnp.sum(r.parcels.mass * r.parcels.multiplicity) == 0
    np.testing.assert_array_equal(r.velocity.y[:, [0, -1]], 0.0)
    np.testing.assert_array_equal(np.asarray(r.velocity.z)[[0, -1]], 0.0)


def setup_many():
    grid, source, gas, u, les = fixture(execution="batched")
    source = replace(
        source,
        mass_flow_rate_kg_s=0.1,
        dpm=replace(
            source.dpm, diameters_m=(100e-6, 300e-6), mass_fractions=(0.3, 0.7)
        ),
    )
    p, l, _ = inject_parcels(
        initial_parcels(source.dpm, jnp.float64),
        initial_ledger(jnp.float64),
        source,
        (1.5, 1.5, 1.5),
        0.0,
        0.01,
        M,
    )
    p, l, _ = inject_parcels(p, l, source, (2.5, 1.5, 1.5), 0.0, 0.01, M)
    return grid, source, gas, u, les, p, l


def test_shared_faces_and_same_cell_exchange_conserve_every_inventory():
    grid, source, gas, u, les, p, l = setup_many()
    run = jax.jit(
        build_spatial_step(
            grid, source, M, 101325.0, periodic_y=False, tunnel_walls=True
        )
    )
    result = run(gas, u, p, l, 0.01, les)
    assert result.accepted
    before_p, before_e = mac_inventory(grid, gas, u)
    after_p, after_e = mac_inventory(grid, result.gas, result.velocity)
    physical = result.parcels.mass * result.parcels.multiplicity
    p_final = jnp.sum(physical * result.parcels.velocity, axis=1)
    e_final = jnp.sum(
        physical
        * (
            M.liquid_cp * (result.parcels.temperature - M.reference_temperature)
            + 0.5 * jnp.sum(result.parcels.velocity**2, axis=0)
        )
    )
    np.testing.assert_allclose(
        after_p + p_final + result.ledger.wall_impulse,
        before_p + l.injected[1:4] + result.ledger.gravity_impulse,
        atol=3e-14,
    )
    np.testing.assert_allclose(
        after_e + e_final + result.ledger.wall_energy,
        before_e
        + l.injected[4]
        + l.injected[5]
        + result.ledger.gravity_work
        + result.ledger.stochastic_work,
        rtol=3e-15,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        jnp.sum(physical) + result.ledger.evaporated_mass, l.injected[0], atol=1e-18
    )
    # Permuting parcel slots must not introduce the serial update-order bias.
    order = jnp.array([3, 1, 0, 2])
    shuffled = p._replace(
        position=p.position[:, order],
        velocity=p.velocity[:, order],
        mass=p.mass[order],
        temperature=p.temperature[order],
        multiplicity=p.multiplicity[order],
        fluctuation=p.fluctuation[:, order],
        eddy_remaining=p.eddy_remaining[order],
        keys=p.keys[order],
        draws=p.draws[order],
    )
    permuted = run(gas, u, shuffled, l, 0.01, les)
    assert permuted.accepted
    for a, b in zip(jax.tree.leaves(result.gas), jax.tree.leaves(permuted.gas)):
        np.testing.assert_allclose(a, b, rtol=2e-14, atol=1e-12)


def test_batched_timestep_refinement_approaches_serial_reference():
    grid, source, gas, u, les, p, l = setup_many()
    serial_source = replace(source, dpm=replace(source.dpm, execution="serial"))
    baseline = jax.jit(
        build_spatial_step(
            grid,
            serial_source,
            M,
            101325.0,
            periodic_y=False,
            tunnel_walls=True,
            gravity=(0.0, 0.0, 0.0),
        )
    )
    batch = jax.jit(
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

    def evolve(fn, n):
        state = (gas, u, p, l)
        for _ in range(n):
            r = fn(*state, 0.02 / n, les)
            assert r.accepted
            state = (r.gas, r.velocity, r.parcels, r.ledger)
        return r

    reference = evolve(baseline, 64)
    errors = []
    for n in (1, 2, 4, 8):
        r = evolve(batch, n)
        errors.append(
            float(jnp.max(jnp.abs(r.parcels.velocity - reference.parcels.velocity)))
        )
    print("batched_velocity_refinement_errors", errors)
    assert all(a > b for a, b in pairwise(errors))
    assert errors[-1] < 0.3 * errors[0]


def test_batched_failure_rolls_back_rng_ledgers_and_fields():
    grid, source, gas, u, les, p, l = setup_many()
    bad = gas._replace(vapor_density=gas.vapor_density.at[-1, -1, -1].set(-1.0))
    run = jax.jit(
        build_spatial_step(
            grid, source, M, 101325.0, periodic_y=False, tunnel_walls=True
        )
    )
    r = run(bad, u, p, l, 0.01, les)
    assert not r.accepted
    for a, b in zip(
        jax.tree.leaves((bad, u, p, l)),
        jax.tree.leaves((r.gas, r.velocity, r.parcels, r.ledger)),
    ):
        np.testing.assert_array_equal(a, b)


def test_batched_drw_persists_and_restarts_exactly(tmp_path):
    from jaxwind.io.checkpoint import load_checkpoint, save_checkpoint

    grid, source, gas, u, les, p, l = setup_many()
    source = replace(source, dpm=replace(source.dpm, dispersion="drw"))
    les = les._replace(
        kinetic_energy=jnp.full_like(les.kinetic_energy, 0.1),
        dissipation=jnp.full_like(les.dissipation, 0.01),
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
    one = run(gas, u, p, l, 1e-5, les)
    assert one.accepted
    np.testing.assert_array_equal(one.parcels.draws, 1)
    path = tmp_path / "batched.npz"
    save_checkpoint(path, one, metadata={"fingerprint": "batched-test"})
    restored, _, _ = load_checkpoint(path, one, fingerprint="batched-test")
    two = run(one.gas, one.velocity, one.parcels, one.ledger, 1e-5, les)
    resumed = run(
        restored.gas, restored.velocity, restored.parcels, restored.ledger, 1e-5, les
    )
    assert two.accepted and resumed.accepted
    np.testing.assert_array_equal(two.parcels.draws, one.parcels.draws)
    np.testing.assert_array_equal(two.parcels.fluctuation, one.parcels.fluctuation)
    for a, b in zip(jax.tree.leaves(two), jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(a, b)


def test_batched_event_capacity_failure_rolls_back_after_partial_work():
    grid, source, gas, u, les, p, l = setup_many()
    source = replace(
        source, dpm=replace(source.dpm, dispersion="drw", max_eddy_intervals=1)
    )
    les = les._replace(
        kinetic_energy=jnp.full_like(les.kinetic_energy, 0.1),
        dissipation=jnp.full_like(les.dissipation, 0.01),
    )
    p = p._replace(eddy_remaining=jnp.full_like(p.mass, 1e-5))
    run = jax.jit(
        build_spatial_step(
            grid,
            source,
            M,
            101325.0,
            periodic_y=False,
            tunnel_walls=True,
        )
    )
    result = run(gas, u, p, l, 0.01, les)
    assert not result.accepted
    for a, b in zip(
        jax.tree.leaves((gas, u, p, l)),
        jax.tree.leaves((result.gas, result.velocity, result.parcels, result.ledger)),
    ):
        np.testing.assert_array_equal(a, b)
