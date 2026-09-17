"""Shared-face source ownership, order convergence, and all-group rollback."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_spray_cell_step import C, P, assert_tree_equal, budget, setup

from jaxwind.numerics.poisson import build_pressure_poisson
from jaxwind.spray_cell_exchange import build_cell_exchange
from jaxwind.spray_core import WaterCoreBins
from jaxwind.spray_group_exchange import build_group_exchange, build_group_spray_step


def group_setup():
    grid, gas, u, e, bins = setup(True)
    other = bins._replace(
        velocity=-0.6 * bins.velocity, multiplicity=1.4 * bins.multiplicity
    )
    groups = jax.tree.map(lambda a, b: jnp.stack((a, b)), bins, other)
    groups = groups._replace(multiplicity=20 * groups.multiplicity)
    return grid, gas, u, e, groups


def flatten(groups):
    return WaterCoreBins(
        groups.mass.reshape(-1),
        groups.multiplicity.reshape(-1),
        groups.velocity.transpose(1, 0, 2).reshape(3, -1),
        groups.temperature.reshape(-1),
    )


def exchange(grid):
    return jax.jit(
        build_group_exchange(
            grid, C, P, periodic_x=False, periodic_y=True, drag_heat_fraction=0.5
        )
    )


@pytest.mark.parametrize("recipients", [[[1, 1, 2], [1, 1, 3]], [[1, 1, 2], [1, 1, 2]]])
def test_adjacent_and_same_cell_global_conservation(recipients):
    grid, gas, u, e, groups = group_setup()
    r = exchange(grid)(gas, u, e, groups, 1e-4, jnp.array(recipients))
    assert bool(r.accepted), r.relative_energy_error
    for a, b in zip(
        budget(grid, gas, u, e, flatten(groups)),
        budget(grid, r.gas, r.velocity, r.unresolved_density, flatten(r.liquid)),
    ):
        np.testing.assert_allclose(a, b, rtol=3e-13, atol=2e-13)
    # Explicit sequential physical reference, including the changed shared face.
    state = (gas, u, e)
    outputs = []
    for i, cell in enumerate(recipients):
        fn = jax.jit(
            build_cell_exchange(
                grid,
                cell,
                C,
                P,
                periodic_x=False,
                periodic_y=True,
                drag_heat_fraction=0.5,
            )
        )
        single = fn(*state, jax.tree.map(lambda q, i=i: q[i], groups), 1e-4)
        assert bool(single.accepted)
        state = single.gas, single.velocity, single.unresolved_density
        outputs.append(single.liquid)
    for a, b in zip(
        jax.tree.leaves((r.gas, r.velocity, r.unresolved_density)),
        jax.tree.leaves(state),
    ):
        np.testing.assert_allclose(a, b, rtol=3e-14, atol=2e-14)
    expected = jax.tree.map(lambda a, b: jnp.stack((a, b)), *outputs)
    for a, b in zip(r.liquid, expected):
        np.testing.assert_allclose(a, b, rtol=3e-14, atol=2e-14)
    print(
        "groups", recipients, float(r.relative_energy_error), float(r.evaporated_mass)
    )


def test_group_order_dependence_converges_with_timestep():
    grid, gas, u, e, groups = group_setup()
    fn = exchange(grid)
    recipients = jnp.array([[1, 1, 2], [1, 1, 3]])
    reverse = jax.tree.map(lambda q: q[::-1], groups)
    errors = []
    for dt in (2e-5, 1e-5, 5e-6):
        f = fn(gas, u, e, groups, dt, recipients)
        b = fn(gas, u, e, reverse, dt, recipients[::-1])
        assert bool(f.accepted & b.accepted)
        errors.append(
            float(jnp.linalg.norm(f.liquid.velocity - b.liquid.velocity[::-1]))
        )
    ratios = np.array(errors[:-1]) / errors[1:]
    assert np.all((ratios > 3.8) & (ratios < 4.2)), (errors, ratios)
    print("order_errors", errors, "ratios", ratios)


@pytest.mark.parametrize("bad_cell", [[0, 1, 3], [-1, 1, 3], [1, 1, 6]])
def test_later_invalid_group_restores_earlier_sources(bad_cell):
    grid, gas, u, e, groups = group_setup()
    r = exchange(grid)(gas, u, e, groups, 1e-4, jnp.array([[1, 1, 2], bad_cell]))
    assert not bool(r.accepted)
    assert_tree_equal(
        (r.gas, r.velocity, r.unresolved_density, r.liquid), (gas, u, e, groups)
    )
    for a in jax.tree.leaves((r[4:7], r[8:11])):
        np.testing.assert_array_equal(a, np.zeros_like(a))


def test_recipient_reassignment_preserves_one_liquid_owner():
    grid, gas, u, e, groups = group_setup()
    initial = budget(grid, gas, u, e, flatten(groups))
    fn = exchange(grid)
    for x in (1, 2, 3):
        r = fn(gas, u, e, groups, 1e-4, jnp.array([[1, 1, x], [1, 1, x + 1]]))
        assert bool(r.accepted)
        gas, u, e, groups = r.gas, r.velocity, r.unresolved_density, r.liquid
    for a, b in zip(initial, budget(grid, gas, u, e, flatten(groups))):
        np.testing.assert_allclose(a, b, rtol=3e-13, atol=2e-13)


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
        build_group_spray_step(
            poisson,
            C,
            P,
            ambient,
            jnp.zeros((3, grid.nz, grid.ny)),
            jnp.zeros((grid.nz, grid.ny)),
            drag_heat_fraction=0.5,
            **controls,
        )
    )


def test_multiple_groups_one_carrier_and_boundary_budgets():
    grid, gas, u, e, groups = group_setup()
    dt = 1e-4
    cells = jnp.array([[1, 1, 2], [1, 1, 3]])
    r = coupled(grid, gas)(gas, u, e, groups, dt, cells)
    assert bool(r.accepted), r.carrier[15:]
    volume = grid.dx * grid.dy * grid.dz
    boundary = (
        jnp.sum(
            r.carrier.scalar_fluxes.x[1, ..., -1] - r.carrier.scalar_fluxes.x[1, ..., 0]
        )
        * grid.dy
        * grid.dz
    )
    np.testing.assert_allclose(
        jnp.sum(r.carrier.gas.vapor_density - gas.vapor_density) * volume
        + dt * boundary,
        jnp.sum((groups.mass - r.liquid.mass) * groups.multiplicity),
        rtol=1e-9,
        atol=2e-16,
    )
    eb = (
        jnp.sum(r.unresolved_flux.x[..., -1] - r.unresolved_flux.x[..., 0])
        * grid.dy
        * grid.dz
    )
    np.testing.assert_allclose(
        jnp.sum(r.unresolved_density - e) * volume + dt * eb,
        0.5 * (r.drag_energy + r.vapor_mixing_energy),
        rtol=1e-11,
        atol=1e-14,
    )
    print(
        "coupled_groups",
        int(r.carrier.iterations),
        float(r.carrier.momentum_error),
        float(r.source_relative_energy_error),
    )


def test_carrier_rejection_restores_all_groups():
    grid, gas, u, e, groups = group_setup()
    r = coupled(grid, gas, max_iterations=1)(
        gas, u, e, groups, 1e-4, jnp.array([[1, 1, 2], [1, 1, 3]])
    )
    assert bool(r.source_accepted) and not bool(r.accepted)
    assert_tree_equal(
        (r.carrier.gas, r.carrier.velocity, r.unresolved_density, r.liquid),
        (gas, u, e, groups),
    )
    for a in jax.tree.leaves((r.carrier[3:15], r.unresolved_flux, r[6:9])):
        np.testing.assert_array_equal(a, np.zeros_like(a))
