"""Conservation and stochastic verification, distinct from experimental validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from jaxwind.spray_closure import (
    RoundJetFlux,
    advance_round_jet,
    homogeneous_sgs_velocity_step,
    round_jet_gaussian,
    round_jet_plane_fluxes,
)

jax.config.update("jax_enable_x64", True)


def source():
    return RoundJetFlux(0.2, 1.5, 60000.0, jnp.array([0.198, 0.002]))


def test_entrainment_balances_mass_momentum_species_and_enthalpy():
    q = source()
    ambient_y = jnp.array([0.98, 0.02])
    advance = jax.jit(lambda f, ds: advance_round_jet(f, ds, 1.2, 310000.0, ambient_y))
    whole = advance(q, 2.0)
    split = q
    for _ in range(32):
        split = advance(split, 2.0 / 32)
    for a, b in zip(whole, split):
        np.testing.assert_allclose(a, b, rtol=2e-14)
    intake = whole.mass - q.mass
    assert intake > 0
    np.testing.assert_allclose(whole.momentum, q.momentum, rtol=0, atol=0)
    np.testing.assert_allclose(whole.enthalpy - q.enthalpy, intake * 310000.0)
    np.testing.assert_allclose(whole.species - q.species, intake * ambient_y)
    np.testing.assert_allclose(jnp.sum(whole.species), whole.mass)
    # Dilution is a convex mixture for each mass fraction and specific enthalpy.
    y = whole.species / whole.mass
    assert np.all(y >= np.minimum(q.species / q.mass, ambient_y))
    assert np.all(y <= np.maximum(q.species / q.mass, ambient_y))
    assert 300000.0 < whole.enthalpy / whole.mass < 310000.0


def test_gaussian_profile_integrates_to_both_independent_fluxes():
    q = source()
    rho = 1.2
    u, b = map(float, round_jet_gaussian(q, rho))
    mass = quad(lambda r: 2 * np.pi * r * rho * u * np.exp(-((r / b) ** 2)), 0, 10 * b)[
        0
    ]
    momentum = quad(
        lambda r: 2 * np.pi * r * rho * u**2 * np.exp(-2 * (r / b) ** 2), 0, 10 * b
    )[0]
    np.testing.assert_allclose([mass, momentum], [q.mass, q.momentum], rtol=1e-12)


@pytest.mark.parametrize("cells_per_radius", [0.25, 0.5, 1, 2, 4])
def test_subcell_plane_integrals_conserve_without_mesh_center_sampling(
    cells_per_radius,
):
    q = source()
    _, b = round_jet_gaussian(q, 1.2)
    ds = float(b) / cells_per_radius
    edges = jnp.arange(-32, 33) * ds
    plane = round_jet_plane_fluxes(q, 1.2, edges, edges, center=(0.37 * ds, -0.21 * ds))
    for projected, original in zip(plane[:3], q[:3]):
        assert np.min(projected) >= 0
        np.testing.assert_allclose(jnp.sum(projected), original, rtol=2e-14)
    np.testing.assert_allclose(
        jnp.sum(plane.species, axis=(1, 2)), q.species, rtol=2e-14
    )


def test_cropped_plane_reports_loss_instead_of_renormalizing():
    q = source()
    _, b = round_jet_gaussian(q, 1.2)
    # A plane with only the positive quadrant contains exactly a quarter of flux.
    edges = jnp.array([0.0, 10 * b])
    plane = round_jet_plane_fluxes(q, 1.2, edges, edges)
    for projected, original in zip(plane[:3], q[:3]):
        np.testing.assert_allclose(jnp.sum(projected), 0.25 * original, rtol=1e-14)


def test_ou_update_is_galilean_and_rotation_covariant():
    rng = np.random.default_rng(5)
    residual, gas, particle, normal = (
        jnp.asarray(rng.normal(size=(3, 37))) for _ in range(4)
    )
    rotation = jnp.asarray(np.linalg.qr(rng.normal(size=(3, 3)))[0])
    update = jax.jit(
        lambda r, u, p, n: homogeneous_sgs_velocity_step(r, u, p, 0.7, 0.1, 0.04, n)
    )
    expected = update(residual, gas, particle, normal)
    actual = update(
        rotation @ residual, rotation @ gas, rotation @ particle, rotation @ normal
    )
    np.testing.assert_allclose(actual, rotation @ expected, atol=1e-14)
    boost = jnp.array([20.0, -30.0, 10.0])[:, None]
    np.testing.assert_allclose(
        update(residual, gas + boost, particle + boost, normal), expected, atol=1e-14
    )


@pytest.mark.parametrize("dt", [1e-8, 0.03, 3.0])
def test_ou_stationary_variance_and_crossing_correlation(dt):
    n = 131072
    residual = jax.random.normal(jax.random.key(21), (3, n), dtype=jnp.float64)
    normal = jax.random.normal(jax.random.key(22), (3, n), dtype=jnp.float64)
    gas = jnp.zeros((3, 1))
    particle = jnp.array([[2.0], [0.0], [0.0]])
    updated = jax.jit(homogeneous_sgs_velocity_step)(
        residual, gas, particle, 1.5, 0.2, dt, normal
    )
    # Independent OU covariance result: variance=1 and lag covariance=e^(-dt/tau).
    a = np.exp(-dt * np.sqrt([5.0, 17.0, 17.0]) / 0.2)
    np.testing.assert_allclose(jnp.mean(updated, axis=1), 0, atol=0.012)
    np.testing.assert_allclose(jnp.var(updated, axis=1), 1, atol=0.018)
    np.testing.assert_allclose(jnp.mean(updated * residual, axis=1), a, atol=0.018)


def test_zero_energy_and_zero_time_are_explicit_limits():
    one = jnp.ones((3, 8))
    for velocity in (one, 2 * one):
        zero = homogeneous_sgs_velocity_step(one, one, velocity, 0.0, 0.1, 0.5, one)
        np.testing.assert_array_equal(zero, 0.0)
        same = homogeneous_sgs_velocity_step(one, one, velocity, 0.0, 0.1, 0.0, one)
        np.testing.assert_array_equal(same, one)
