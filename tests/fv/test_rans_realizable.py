"""Analytic invariants and ODE convergence for the published realizable model."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from jaxwind.rans_kepsilon import KEpsilonState
from jaxwind.rans_realizable import (
    invariants_from_gradient,
    local_sources,
    viscosity_from_invariants,
)

jax.config.update("jax_enable_x64", True)


def test_simple_shear_and_zero_strain_limits():
    gamma = 2.3
    g = jnp.array([[0, gamma, 0], [0, 0, 0], [0, 0, 0]], dtype=jnp.float64)
    s, u, a = invariants_from_gradient(g)
    np.testing.assert_allclose([s, u, a], [gamma, gamma, 3 / np.sqrt(2)], rtol=1e-14)
    state = KEpsilonState(jnp.array(0.09), jnp.array(0.1))
    expected = 0.09**2 / 0.1 / (4.04 + 3 / np.sqrt(2) * gamma * 0.09 / 0.1)
    np.testing.assert_allclose(
        viscosity_from_invariants(state, u, a), expected, rtol=1e-14
    )
    s, u, a = invariants_from_gradient(jnp.zeros((3, 3)))
    np.testing.assert_allclose(
        viscosity_from_invariants(state, u, a), 0.09**2 / 0.1 / 4.04, rtol=1e-14
    )
    assert float(s) == 0 and float(u) == 0


def test_rotation_invariance_and_nonnegative_normal_stresses():
    rng = np.random.default_rng(42)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    state = KEpsilonState(jnp.array(0.09), jnp.array(0.1))
    for scale in [1.0, 100.0, 1e5]:
        g = np.diag([2.0, -1.0, -1.0]) * scale
        inv = invariants_from_gradient(jnp.asarray(g))
        rot = invariants_from_gradient(jnp.asarray(q @ g @ q.T))
        np.testing.assert_allclose(inv, rot, rtol=1e-8)
        nu = float(viscosity_from_invariants(state, inv[1], inv[2]))
        stress = 2 / 3 * 0.09 * np.eye(3) - 2 * nu * g
        assert np.linalg.eigvalsh(stress).min() >= -1e-12
    s, u, _ = invariants_from_gradient(
        jnp.array([[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    assert float(s) == 0
    np.testing.assert_allclose(u, np.sqrt(8))


def test_source_derivative_matches_realizable_equations():
    k = 0.09
    eps = 0.1083
    p = 0.8
    s = 3.0
    nu = 1.7e-5
    dt = 1e-8
    initial = KEpsilonState(jnp.array(k), jnp.array(eps))
    result = local_sources(initial, p, s, nu, dt)
    c1 = max(0.43, (s * k / eps) / (s * k / eps + 5))
    expected = [p - eps, c1 * s * eps - 1.9 * eps**2 / (k + np.sqrt(nu * eps))]
    np.testing.assert_allclose(
        (np.asarray(result) - np.asarray(initial)) / dt, expected, rtol=1e-6
    )
    # Dissipation production must not inherit the standard model's dependence on P.
    changed = local_sources(initial, 20 * p, s, nu, dt)
    np.testing.assert_array_equal(result.dissipation, changed.dissipation)


def test_source_converges_to_independent_ode_and_remains_positive():
    initial = KEpsilonState(jnp.array(0.09), jnp.array(0.1083))
    nu = 1.7e-5
    duration = 0.2
    p = 0.2
    s = 3.0

    def rhs(t, a):
        k, e = a
        eta = s * k / e
        c1 = max(0.43, eta / (eta + 5))
        return [p - e, c1 * s * e - 1.9 * e * e / (k + np.sqrt(nu * e))]

    exact = solve_ivp(
        rhs, (0, duration), np.asarray(initial), rtol=1e-11, atol=1e-13
    ).y[:, -1]
    errors = []
    for n in [64, 128]:
        h = duration / n
        final = jax.lax.fori_loop(
            0, n, lambda _, a, h=h: local_sources(a, p, s, nu, h), initial
        )
        errors.append(np.linalg.norm(np.asarray(final) - exact))
    assert 1.8 < errors[0] / errors[1] < 2.2
    stiff = local_sources(initial, 0.0, 0.0, nu, 1e6)
    assert np.all(np.asarray(stiff) > 0) and np.all(
        np.asarray(stiff) < np.asarray(initial)
    )
