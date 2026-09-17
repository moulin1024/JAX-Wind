"""Well-mixed PDF identity and integration checks, not spray validation."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.inhomogeneous_dispersion import (
    isotropic_well_mixed_drift,
    normalized_transport_midpoint,
    well_mixed_tracer_step,
)

jax.config.update("jax_enable_x64", True)


def variance(x):
    return (
        1.2 + 0.25 * jnp.sin(x[0]) + 0.2 * jnp.cos(2 * x[1]) + 0.15 * jnp.sin(3 * x[2])
    )


def fields(x):
    gradient = jnp.stack(
        (0.25 * jnp.cos(x[0]), -0.4 * jnp.sin(2 * x[1]), 0.45 * jnp.cos(3 * x[2]))
    )
    return variance(x), gradient, 0.3 + 0.05 * jnp.cos(x[0])


def test_stationary_fokker_planck_identity_in_three_dimensions():
    # Independent differentiation of the physical phase-space PDF, including
    # spatial transport, velocity drift divergence and velocity diffusion.
    def density(x, v):
        q = variance(x)
        return jnp.exp(-jnp.sum(v * v) / (2 * q)) / (2 * jnp.pi * q) ** 1.5

    def residual(x, v, corrected):
        q, g, tau = fields(x)
        drift = lambda vv: (
            isotropic_well_mixed_drift(vv, q, g, tau) if corrected else -vv / tau
        )
        transport = -jnp.dot(v, jax.grad(density, argnums=0)(x, v))
        force = -jnp.trace(jax.jacfwd(lambda vv: drift(vv) * density(x, vv))(v))
        diffusion = q / tau * jnp.trace(jax.hessian(lambda vv: density(x, vv))(v))
        return (transport + force + diffusion) / density(x, v)

    rng = np.random.default_rng(18)
    points = jnp.asarray(rng.normal(size=(24, 3)))
    velocities = jnp.asarray(rng.normal(size=(24, 3)))
    evaluate = jax.jit(jax.vmap(residual, in_axes=(0, 0, None)), static_argnums=2)
    np.testing.assert_allclose(evaluate(points, velocities, True), 0, atol=2e-14)
    assert np.max(np.abs(evaluate(points, velocities, False))) > 0.1


def test_homogeneous_step_recovers_unit_variance_and_ou_lag_correlation():
    count = 131072
    w = jax.random.normal(jax.random.key(1), (3, count), dtype=jnp.float64)
    noise = jax.random.normal(jax.random.key(2), (2, 3, count), dtype=jnp.float64)
    const = lambda x: (jnp.ones(x.shape[1:]) * 2.0, jnp.zeros_like(x), 0.3)
    step = jax.jit(lambda x, w, n: well_mixed_tracer_step(x, w, 0.1, n, const))
    _, updated = step(jnp.zeros_like(w), w, noise)
    np.testing.assert_allclose(jnp.mean(updated, axis=1), 0, atol=0.012)
    np.testing.assert_allclose(jnp.var(updated, axis=1), 1, atol=0.018)
    np.testing.assert_allclose(
        jnp.mean(w * updated, axis=1), np.exp(-1 / 3), atol=0.015
    )


def test_zero_timestep_preserves_position_and_normalized_velocity():
    x = jnp.array([[0.1], [0.2], [0.3]])
    w = jnp.array([[0.4], [-0.2], [0.5]])
    actual = well_mixed_tracer_step(x, w, 0.0, jnp.ones((2, 3, 1)), fields)
    for a, b in zip(actual, (x, w)):
        np.testing.assert_array_equal(a, b)


def test_deterministic_transport_has_second_order_timestep_convergence():
    x = jnp.array([[0.1], [0.2], [0.3]])
    w = jnp.array([[0.4], [-0.2], [0.5]])

    def trajectory(count):
        step = lambda _, state: normalized_transport_midpoint(
            *state, 0.2 / count, fields
        )
        return jax.jit(lambda: jax.lax.fori_loop(0, count, step, (x, w)))()

    reference = trajectory(320)
    errors = []
    for count in (10, 20, 40):
        value = trajectory(count)
        errors.append(
            np.linalg.norm(
                np.concatenate(
                    [np.asarray(a - b).ravel() for a, b in zip(value, reference)]
                )
            )
        )
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8
    assert errors[-1] < 1e-5
