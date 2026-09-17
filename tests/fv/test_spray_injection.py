"""Continuous-flow quadrature, birth ages, capacity and coupled rollback."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_spray_cell_step import C, P, assert_tree_equal, budget, setup

from jaxwind.numerics.poisson import build_pressure_poisson
from jaxwind.spray_core import WaterCoreBins
from jaxwind.spray_injection import (
    births_from_mass_flow,
    build_injected_spray_step,
    stage_parcel_births,
)
from jaxwind.spray_moving import build_moving_source


def injection_setup():
    (
        grid,
        gas,
        u,
        e,
        _,
    ) = setup()
    n = 6
    mass = C.water_density * jnp.pi * (70e-6) ** 3 / 6
    liquid = WaterCoreBins(
        jnp.full(n, mass),
        jnp.zeros(n).at[2].set(3000.0),
        jnp.zeros((3, n)).at[0, 2].set(3.0),
        jnp.full(n, 300.0),
    )
    position = jnp.full((3, n), 0.15)
    return grid, gas, u, e, liquid, position


def births(dt=1e-4, mass_flow=1e-4):
    return births_from_mass_flow(
        mass_flow,
        dt,
        jnp.array([20e-6, 70e-6, 180e-6]),
        jnp.array([0.2, 0.3, 0.5]),
        jnp.array([[18.0, 7.0, 11.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        jnp.full(3, 300.0),
        jnp.array([[0.1998, 0.1998, 0.2998], [0.15, 0.15, 0.15], [0.15, 0.15, 0.15]]),
        jnp.array([0.25, 0.5, 0.75]),
        C,
    )


def stage(liquid, position, request, dt):
    return jax.jit(lambda q, p, b, t: stage_parcel_births(q, p, b, t, C, P))(
        liquid, position, request, dt
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
        build_injected_spray_step(
            poisson,
            C,
            P,
            ambient,
            jnp.zeros((3, grid.nz, grid.ny)),
            jnp.zeros((grid.nz, grid.ny)),
            drag_heat_fraction=0.5,
            particle_acceleration=(0.0, 0.0, -9.81),
            max_segments=8,
            **controls,
        )
    )


def test_mass_flow_quadrature_and_birth_ages_preserve_existing_parcels():
    _, _, _, _, liquid, position = injection_setup()
    b = births()
    s = stage(liquid, position, b, 1e-4)
    assert bool(s.accepted)
    np.testing.assert_array_equal(s.slots, [0, 1, 3])
    np.testing.assert_allclose(s.injected_mass, 1e-8, rtol=2e-15)
    np.testing.assert_allclose(
        s.residence_times[jnp.array([0, 1, 3])],
        np.array([0.75, 0.5, 0.25]) * 1e-4,
        rtol=2e-15,
    )
    assert s.residence_times[2] == 1e-4
    assert_tree_equal(
        jax.tree.map(lambda q: q[..., 2], s.liquid),
        jax.tree.map(lambda q: q[..., 2], liquid),
    )
    m = np.array([0.2, 0.3, 0.5]) * 1e-8
    np.testing.assert_allclose(
        s.injected_momentum,
        np.sum(m * np.asarray(b.liquid.velocity), axis=1),
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        s.injected_enthalpy,
        1e-8 * P.liquid_heat_capacity * (300 - C.freezing_temperature),
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        s.injected_kinetic_energy,
        0.5 * np.sum(m * np.sum(np.asarray(b.liquid.velocity) ** 2, axis=0)),
        rtol=2e-15,
    )


def test_zero_flow_needs_no_slots_and_overflow_never_reduces_flow():
    _, _, _, _, q, p = injection_setup()
    full = q._replace(multiplicity=jnp.ones_like(q.mass))
    zero = stage(full, p, births(mass_flow=0.0), 1e-4)
    assert (
        bool(zero.accepted) and zero.requested_slots == 0 and zero.available_slots == 0
    )
    assert_tree_equal(zero.liquid, full)
    assert zero.injected_mass == 0
    overflow = stage(full, p, births(), 1e-4)
    assert not bool(overflow.accepted) and overflow.requested_slots == 3
    assert_tree_equal((overflow.liquid, overflow.position), (full, p))
    for a in jax.tree.leaves(overflow[3:7]):
        np.testing.assert_array_equal(a, 0)
    np.testing.assert_array_equal(overflow.slots, -1)


def test_bad_birth_time_and_unnormalized_weights_reject():
    _, _, _, _, q, p = injection_setup()
    b = births()
    bad = b._replace(time_offset=b.time_offset.at[0].set(1e-4))
    assert not bool(stage(q, p, bad, 1e-4).accepted)
    wrong = births_from_mass_flow(
        1e-4,
        1e-4,
        jnp.array([70e-6]),
        jnp.array([0.5]),
        jnp.zeros((3, 1)),
        jnp.array([300.0]),
        jnp.full((3, 1), 0.15),
        jnp.array([0.5]),
        C,
    )
    assert not bool(wrong.valid) and not bool(stage(q, p, wrong, 1e-4).accepted)


def test_injected_source_budget_and_actual_newborn_residence():
    grid, gas, u, e, q, p = injection_setup()
    dt = 1e-4
    b = births(dt)
    s = stage(q, p, b, dt)
    fn = jax.jit(
        build_moving_source(
            grid,
            C,
            P,
            periodic_x=False,
            periodic_y=True,
            drag_heat_fraction=0.5,
            particle_acceleration=(0.0, 0.0, -9.81),
            max_segments=8,
        )
    )
    r = fn(gas, u, e, s.liquid, dt, s.position, s.residence_times)
    assert bool(r.source.accepted)
    expected = list(budget(grid, gas, u, e, q))
    expected[1] += float(s.injected_mass)
    expected[2] += np.asarray(s.injected_momentum) + np.sum(r.external_impulse, axis=1)
    expected[3] += float(
        s.injected_enthalpy + s.injected_kinetic_energy + jnp.sum(r.external_work)
    )
    after = list(
        budget(
            grid,
            r.source.gas,
            r.source.velocity,
            r.source.unresolved_density,
            r.source.liquid,
        )
    )
    after[1] += np.sum(r.exited_mass)
    after[2] += np.sum(r.exited_momentum, axis=1)
    after[3] += np.sum(r.exited_energy)
    for a, z in zip(expected, after):
        np.testing.assert_allclose(a, z, rtol=3e-13, atol=2e-13)
    np.testing.assert_allclose(
        r.position[:, s.slots],
        b.position + b.liquid.velocity * (dt - b.time_offset),
        atol=2e-15,
    )


@pytest.mark.parametrize("failure", ["capacity", "source", "carrier"])
def test_failed_transaction_restores_preinjection_state(failure):
    grid, gas, u, e, q, p = injection_setup()
    b = births()
    options = {}
    if failure == "capacity":
        q = q._replace(multiplicity=jnp.ones_like(q.mass))
    elif failure == "source":
        b = b._replace(position=b.position.at[2, 0].set(0.01))
    else:
        options = {"max_iterations": 1}
    r = coupled(grid, gas, **options)(gas, u, e, q, 1e-4, p, b)
    m = r.moving
    assert not bool(m.phase.accepted)
    assert_tree_equal(
        (
            m.phase.carrier.gas,
            m.phase.carrier.velocity,
            m.phase.unresolved_density,
            m.phase.liquid,
            m.position,
        ),
        (gas, u, e, q, p),
    )
    for a in jax.tree.leaves(
        (
            r[1:5],
            m.phase.carrier[3:15],
            m.exited_mass,
            m.external_impulse,
            m.external_work,
        )
    ):
        np.testing.assert_array_equal(a, np.zeros_like(a))
    np.testing.assert_array_equal(r.slots, -1)


def test_accepted_coupled_births_and_retry_from_original_state():
    grid, gas, u, e, q, p = injection_setup()
    b = births()
    failed = coupled(grid, gas, max_iterations=1)(gas, u, e, q, 1e-4, p, b)
    f = failed.moving
    fn = coupled(grid, gas)
    r = fn(
        f.phase.carrier.gas,
        f.phase.carrier.velocity,
        f.phase.unresolved_density,
        f.phase.liquid,
        1e-4,
        f.position,
        b,
    )
    fresh = fn(gas, u, e, q, 1e-4, p, b)
    assert bool(r.moving.phase.accepted), r.moving.phase.carrier[15:]
    assert_tree_equal(r, fresh)
    c = r.moving.phase.carrier
    volume = grid.dx * grid.dy * grid.dz
    boundary = (
        jnp.sum(c.scalar_fluxes.x[1, ..., -1] - c.scalar_fluxes.x[1, ..., 0])
        * grid.dy
        * grid.dz
    )
    before = (
        jnp.sum(gas.vapor_density) * volume
        + jnp.sum(q.mass * q.multiplicity)
        + r.injected_mass
    )
    after = (
        jnp.sum(c.gas.vapor_density) * volume
        + jnp.sum(r.moving.phase.liquid.mass * r.moving.phase.liquid.multiplicity)
        + jnp.sum(r.moving.exited_mass)
        + 1e-4 * boundary
    )
    np.testing.assert_allclose(before, after, rtol=3e-13, atol=2e-16)
    print(
        "injected_carrier",
        int(c.iterations),
        float(c.momentum_error),
        float(r.injected_mass),
    )
