"""Spatial tracking refinement and actual stochastic parcel restart."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from test_fluent_dpm_spatial import M, setup

from jaxwind.fluent_dpm_les import DPMLESFields
from jaxwind.fluent_dpm_spatial import (
    build_spatial_step,
    initial_ledger,
    initial_parcels,
    inject_parcels,
)
from jaxwind.io.checkpoint import load_checkpoint, save_checkpoint


def initial(source, center=(24.0, 8.0, 6.0)):
    return inject_parcels(
        initial_parcels(source.dpm, jnp.float64),
        initial_ledger(jnp.float64),
        source,
        center,
        0.0,
        0.02,
        M,
    )[:2]


def test_tracking_refinement_reduces_split_trajectory_error():
    grid, gas, u, source = setup()
    source = replace(source, dpm=replace(source.dpm, capacity=1))
    p, l = initial(source)
    zero = jnp.zeros_like(gas.dry_density)
    les = DPMLESFields(zero, zero, zero, jnp.asarray(1.0))
    fn = jax.jit(build_spatial_step(grid, source, M, 101325.0, gravity=(0.0, 0.0, 0.0)))
    endpoints = []
    for count in (1, 2, 4, 16):
        args = gas, u, p, l
        for _ in range(count):
            r = fn(*args, 0.02 / count, les)
            assert r.accepted
            args = r.gas, r.velocity, r.parcels, r.ledger
        endpoints.append(float(r.parcels.position[0, 0]))
    errors = np.abs(np.array(endpoints[:-1]) - endpoints[-1])
    assert errors[1] < 0.7 * errors[0]
    assert errors[2] < 0.7 * errors[1]
    print("tracking_refinement_position_errors_m", errors)


def test_spatial_drw_checkpoint_retains_eddy_and_key(tmp_path):
    grid, gas, u, source = setup()
    source = replace(source, dpm=replace(source.dpm, capacity=1, dispersion="drw"))
    p, l = initial(source)
    ones = jnp.ones_like(gas.dry_density)
    les = DPMLESFields(0.1 * ones, 0.2 * ones, 0.1 * ones, jnp.asarray(1.0))
    fn = jax.jit(build_spatial_step(grid, source, M, 101325.0, gravity=(0.0, 0.0, 0.0)))
    one = fn(gas, u, p, l, 0.01, les)
    assert one.accepted
    assert one.parcels.draws[0] == 1
    assert jnp.linalg.norm(one.parcels.fluctuation[:, 0]) > 0
    path = tmp_path / "eddy.npz"
    save_checkpoint(path, one, metadata={"fingerprint": "eddy"})
    restored, _, _ = load_checkpoint(path, one, fingerprint="eddy")
    a = fn(one.gas, one.velocity, one.parcels, one.ledger, 0.01, les)
    b = fn(
        restored.gas, restored.velocity, restored.parcels, restored.ledger, 0.01, les
    )
    assert a.accepted and b.accepted
    assert a.parcels.draws[0] == 1
    np.testing.assert_array_equal(a.parcels.fluctuation, one.parcels.fluctuation)
    assert a.parcels.eddy_remaining[0] < one.parcels.eddy_remaining[0]
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_failed_cell_path_rolls_back_all_parcel_sources_and_random_state():
    grid, gas, u, source = setup()
    source = replace(
        source,
        dpm=replace(source.dpm, capacity=1, max_path_segments=1, dispersion="drw"),
    )
    p, l = initial(source, center=(15.9, 8.0, 6.0))
    ones = jnp.ones_like(gas.dry_density)
    les = DPMLESFields(0.1 * ones, 0.2 * ones, 0.1 * ones, jnp.asarray(1.0))
    fn = jax.jit(build_spatial_step(grid, source, M, 101325.0, gravity=(0.0, 0.0, 0.0)))
    r = fn(gas, u, p, l, 0.02, les)
    assert not r.accepted
    for a, b in zip(
        jax.tree.leaves((gas, u, p, l)),
        jax.tree.leaves((r.gas, r.velocity, r.parcels, r.ledger)),
    ):
        np.testing.assert_array_equal(a, b)
