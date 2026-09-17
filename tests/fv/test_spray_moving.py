"""Analytic path residence, moving phase budgets, and one-time outflow."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_spray_cell_step import C, P, assert_tree_equal, budget, setup

from jaxwind.domain import UniformGrid
from jaxwind.numerics.poisson import build_pressure_poisson
from jaxwind.spray_moving import build_moving_source, build_moving_spray_step
from jaxwind.spray_paths import build_parcel_path


def path(**kwargs):
    grid = UniformGrid(4, 4, 4, 1.0, 1.0, 1.0)
    return jax.jit(
        build_parcel_path(
            grid, periodic_x=kwargs.pop("periodic_x", False), periodic_y=True, **kwargs
        )
    )


@pytest.mark.parametrize(
    "speed,start,expected", [(1.0, 0.125, [0, 1, 2, 3]), (-1.0, 0.875, [3, 2, 1, 0])]
)
def test_multicell_residence_and_open_exit(speed, start, expected):
    r = path()(jnp.array([start, 0.375, 0.375]), jnp.array([speed, 0.0, 0.0]), 1.0)
    assert bool(r.accepted & r.exited)
    assert int(r.segments) == 4
    np.testing.assert_array_equal(r.cells[:4, 2], expected)
    np.testing.assert_allclose(r.durations[:4], [0.125, 0.25, 0.25, 0.25], atol=1e-15)
    np.testing.assert_allclose(jnp.sum(r.durations), 0.875, atol=1e-15)
    np.testing.assert_allclose(
        r.position, [1.0 if speed > 0 else 0.0, 0.375, 0.375], atol=1e-15
    )


def test_negative_face_corner_and_periodic_repeated_wraps():
    r = path()(jnp.array([0.5, 0.5, 0.375]), jnp.array([-1.0, -1.0, 0.0]), 0.375)
    assert bool(r.accepted) and not bool(r.exited)
    np.testing.assert_array_equal(r.cells[:2], [[1, 1, 1], [1, 0, 0]])
    np.testing.assert_allclose(r.durations[:2], [0.25, 0.125], atol=1e-15)
    p = path(periodic_x=True)(
        jnp.array([0.125, 0.375, 0.375]), jnp.array([2.0, 0.0, 0.0]), 1.25
    )
    assert bool(p.accepted) and not bool(p.exited)
    np.testing.assert_allclose(p.position, [0.625, 0.375, 0.375], atol=1e-15)
    np.testing.assert_allclose(jnp.sum(p.durations), 1.25, atol=1e-15)
    np.testing.assert_array_equal(p.cells[:11, 2], np.arange(11) % 4)


def test_stationary_immediate_exit_wall_and_capacity():
    pos = jnp.array([0.125, 0.375, 0.375])
    still = path()(pos, jnp.zeros(3), 0.5)
    assert bool(still.accepted) and int(still.segments) == 1
    np.testing.assert_allclose(still.durations[0], 0.5)
    immediate = path()(pos.at[0].set(0.0), jnp.array([-1.0, 0.0, 0.0]), 0.5)
    assert bool(immediate.accepted & immediate.exited) and int(immediate.segments) == 0
    for r in (
        path()(pos, jnp.array([0.0, 0.0, -1.0]), 0.5),
        path(max_segments=1)(pos, jnp.array([1.0, 0.0, 0.0]), 0.5),
    ):
        assert not bool(r.accepted)
        assert_tree_equal(r.position, pos)
        np.testing.assert_array_equal(r.durations, 0.0)


def moving_setup(exit_first=False):
    grid, gas, u, e, liquid = setup(True)
    position = jnp.array(
        [
            [0.599 if exit_first else 0.1998, 0.1998, 0.2998],
            [0.15, 0.15, 0.15],
            [0.15, 0.15, 0.15],
        ]
    )
    return grid, gas, u, e, liquid, position


def source(grid, **kwargs):
    return jax.jit(
        build_moving_source(
            grid,
            C,
            P,
            periodic_x=False,
            periodic_y=True,
            drag_heat_fraction=0.5,
            max_segments=kwargs.pop("max_segments", 8),
            **kwargs,
        )
    )


@pytest.mark.parametrize("exit_first", [False, True])
def test_moving_source_global_budgets_and_no_repeated_export(exit_first):
    grid, gas, u, e, liquid, pos = moving_setup(exit_first)
    dt = 1e-4
    fn = source(grid)
    r = fn(gas, u, e, liquid, dt, pos)
    assert bool(r.source.accepted), (r.path_accepted, r.source.relative_energy_error)
    before = budget(grid, gas, u, e, liquid)
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
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b, rtol=4e-13, atol=2e-13)
    if exit_first:
        assert r.exited_mass[0] > 0 and r.source.liquid.multiplicity[0] == 0
        again = fn(
            r.source.gas,
            r.source.velocity,
            r.source.unresolved_density,
            r.source.liquid,
            dt,
            r.position,
        )
        assert bool(again.source.accepted)
        assert again.exited_mass[0] == 0 and again.exited_energy[0] == 0
        assert_tree_equal(again.position[:, 0], r.position[:, 0])
    else:
        np.testing.assert_allclose(r.position, pos + dt * liquid.velocity, atol=1e-15)
    # At least one source is deposited on each side of a crossed x face.
    assert jnp.sum(r.source.gas_increments.vapor_density[..., 1]) > 0
    assert jnp.sum(r.source.gas_increments.vapor_density[..., 2]) > 0
    print(
        "moving_source",
        exit_first,
        r.path_segments,
        float(r.source.relative_energy_error),
        float(jnp.sum(r.exited_mass)),
    )


def test_capacity_rejection_restores_positions_and_every_inventory():
    grid, gas, u, e, liquid, pos = moving_setup()
    r = source(grid, max_segments=1)(gas, u, e, liquid, 1e-4, pos)
    assert not bool(r.source.accepted)
    assert_tree_equal(
        (
            r.source.gas,
            r.source.velocity,
            r.source.unresolved_density,
            r.source.liquid,
            r.position,
        ),
        (gas, u, e, liquid, pos),
    )
    for q in jax.tree.leaves((r.source[4:7], r.source[8:11], r[2:5])):
        np.testing.assert_array_equal(q, np.zeros_like(q))


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
        build_moving_spray_step(
            poisson,
            C,
            P,
            ambient,
            jnp.zeros((3, grid.nz, grid.ny)),
            jnp.zeros((grid.nz, grid.ny)),
            drag_heat_fraction=0.5,
            max_segments=8,
            **controls,
        )
    )


def test_moving_carrier_includes_liquid_outflow_and_boundary_vapor():
    grid, gas, u, e, liquid, pos = moving_setup(True)
    dt = 1e-4
    r = coupled(grid, gas)(gas, u, e, liquid, dt, pos)
    assert bool(r.phase.accepted), r.phase.carrier[15:]
    c = r.phase.carrier
    volume = grid.dx * grid.dy * grid.dz
    boundary = (
        jnp.sum(c.scalar_fluxes.x[1, ..., -1] - c.scalar_fluxes.x[1, ..., 0])
        * grid.dy
        * grid.dz
    )
    initial = jnp.sum(gas.vapor_density) * volume + jnp.sum(
        liquid.mass * liquid.multiplicity
    )
    final = (
        jnp.sum(c.gas.vapor_density) * volume
        + jnp.sum(r.phase.liquid.mass * r.phase.liquid.multiplicity)
        + jnp.sum(r.exited_mass)
        + dt * boundary
    )
    np.testing.assert_allclose(initial, final, rtol=3e-13, atol=1e-16)
    assert r.exited_mass[0] > 0 and r.phase.liquid.multiplicity[0] == 0
    print("moving_carrier", int(c.iterations), float(c.momentum_error))


def test_carrier_failure_restores_positions_and_cancels_outflow():
    grid, gas, u, e, liquid, pos = moving_setup(True)
    r = coupled(grid, gas, max_iterations=1)(gas, u, e, liquid, 1e-4, pos)
    assert bool(r.phase.source_accepted) and not bool(r.phase.accepted)
    assert_tree_equal(
        (
            r.phase.carrier.gas,
            r.phase.carrier.velocity,
            r.phase.unresolved_density,
            r.phase.liquid,
            r.position,
        ),
        (gas, u, e, liquid, pos),
    )
    for q in jax.tree.leaves((r.phase.carrier[3:15], r[2:5])):
        np.testing.assert_array_equal(q, np.zeros_like(q))
