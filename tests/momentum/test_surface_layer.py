from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxwind.momentum import MoninObukhovWallLaw, NeutralLogWallLaw


def test_neutral_wall_law_returns_expected_vector_stress() -> None:
    law = NeutralLogWallLaw(roughness_length=0.1)
    velocity = jnp.asarray([[[3.0, 4.0]]], dtype=jnp.float32)
    fluxes = law.surface_fluxes(velocity, matching_height=10.0)
    expected_ustar = 0.4 * 5.0 / math.log(100.0)

    assert jnp.allclose(fluxes.friction_velocity, expected_ustar)
    assert jnp.allclose(
        fluxes.momentum_stress,
        expected_ustar**2 * jnp.asarray([[[0.6, 0.8]]]),
    )
    assert jnp.all(fluxes.heat_flux == 0.0)
    assert jnp.all(jnp.isinf(fluxes.obukhov_length))


def test_most_zero_temperature_difference_is_exact_neutral_limit() -> None:
    neutral = NeutralLogWallLaw(roughness_length=0.1)
    most = MoninObukhovWallLaw(
        momentum_roughness_length=0.1,
        thermal_roughness_length=0.01,
        reference_potential_temperature=300.0,
    )
    velocity = jnp.asarray([[[7.0, -2.0]]], dtype=jnp.float32)
    expected = neutral.surface_fluxes(velocity, matching_height=10.0)
    actual = most.surface_fluxes(
        velocity,
        jnp.asarray([[300.0]], dtype=jnp.float32),
        300.0,
        matching_height=10.0,
    )

    assert jnp.allclose(actual.friction_velocity, expected.friction_velocity)
    assert jnp.allclose(actual.momentum_stress, expected.momentum_stress)
    assert jnp.all(actual.heat_flux == 0.0)
    assert jnp.all(jnp.isinf(actual.obukhov_length))


def test_most_flux_signs_and_stability_effect_are_physical() -> None:
    law = MoninObukhovWallLaw(
        momentum_roughness_length=0.1,
        thermal_roughness_length=0.01,
        reference_potential_temperature=300.0,
    )
    velocity = jnp.asarray([[[8.0, 0.0]]], dtype=jnp.float32)
    temperature = jnp.asarray([[300.0]], dtype=jnp.float32)
    neutral = NeutralLogWallLaw(0.1).surface_fluxes(velocity, 10.0)
    stable = law.surface_fluxes(velocity, temperature, 299.5, 10.0)
    unstable = law.surface_fluxes(velocity, temperature, 300.5, 10.0)

    assert jnp.all(stable.heat_flux < 0.0)
    assert jnp.all(stable.obukhov_length > 0.0)
    assert jnp.all(stable.friction_velocity < neutral.friction_velocity)
    assert jnp.all(unstable.heat_flux > 0.0)
    assert jnp.all(unstable.obukhov_length < 0.0)
    assert jnp.all(unstable.friction_velocity > neutral.friction_velocity)


def test_stable_most_closed_form_satisfies_implicit_relation() -> None:
    law = MoninObukhovWallLaw(
        momentum_roughness_length=0.1,
        thermal_roughness_length=0.01,
        reference_potential_temperature=300.0,
        iterations=1,
    )
    velocity = jnp.asarray(
        [[[8.0, 0.0], [6.0, 1.0]]],
        dtype=jnp.float32,
    )
    temperature = jnp.asarray([[300.4, 300.2]], dtype=jnp.float32)
    height = 10.0
    fluxes = law.surface_fluxes(velocity, temperature, 300.0, height)
    inverse_obukhov = 1.0 / fluxes.obukhov_length
    speed = jnp.linalg.norm(velocity, axis=-1)
    momentum_denominator = (
        math.log(height / law.momentum_roughness_length)
        + law.stable_momentum_beta
        * (height - law.momentum_roughness_length)
        * inverse_obukhov
    )
    heat_denominator = (
        math.log(height / law.thermal_roughness_length)
        + law.stable_heat_beta
        * (height - law.thermal_roughness_length)
        * inverse_obukhov
    )
    expected_inverse_obukhov = (
        law.gravity
        * (temperature - 300.0)
        * momentum_denominator**2
        / (
            speed**2
            * law.reference_potential_temperature
            * heat_denominator
        )
    )

    assert jnp.allclose(
        inverse_obukhov,
        expected_inverse_obukhov,
        rtol=2.0e-6,
        atol=1.0e-8,
    )


def test_unstable_most_retains_iterative_fallback() -> None:
    velocity = jnp.asarray([[[8.0, 0.0]]], dtype=jnp.float32)
    temperature = jnp.asarray([[299.5]], dtype=jnp.float32)
    one_iteration = MoninObukhovWallLaw(
        momentum_roughness_length=0.1,
        thermal_roughness_length=0.01,
        reference_potential_temperature=300.0,
        iterations=1,
    ).surface_fluxes(velocity, temperature, 300.0, 10.0)
    converged = MoninObukhovWallLaw(
        momentum_roughness_length=0.1,
        thermal_roughness_length=0.01,
        reference_potential_temperature=300.0,
        iterations=12,
    ).surface_fluxes(velocity, temperature, 300.0, 10.0)

    assert not jnp.allclose(
        one_iteration.obukhov_length,
        converged.obukhov_length,
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_most_surface_fluxes_are_jittable_and_finite() -> None:
    law = MoninObukhovWallLaw(
        momentum_roughness_length=0.1,
        thermal_roughness_length=0.01,
        reference_potential_temperature=300.0,
    )

    @jax.jit
    def evaluate(velocity, temperature):
        return law.surface_fluxes(velocity, temperature, 300.2, 10.0)

    fluxes = evaluate(
        jnp.ones((4, 8, 2), dtype=jnp.float32),
        jnp.full((4, 8), 300.0, dtype=jnp.float32),
    )

    for value in fluxes[:-1]:
        assert jnp.all(jnp.isfinite(value))
    assert jnp.all(jnp.isfinite(fluxes.obukhov_length))
