"""Independent first-principles identities used to audit the centre outlier."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.cryogenic import _cic_coordinates, _cic_deposit_many, _cic_sample_many
from jaxwind.domain import UniformGrid
from jaxwind.physics.moisture import MoistureConfig, WaterDropletProperties
from jaxwind.water_spray import advance_water_droplet_motion

jax.config.update("jax_enable_x64", True)


def test_cic_conserves_and_is_adjoint_at_clipped_open_and_wall_boundaries():
    """Boundary clipping must neither duplicate source mass nor amplify work."""
    grid = UniformGrid(8, 6, 4, 0.8, 0.6, 0.4)
    rng = np.random.default_rng(47)
    position = rng.uniform(-0.05, 1.05, (3, 64)) * np.array([0.8, 0.6, 0.4])[:, None]
    position[:, :4] = np.array([[0, 0.8, 0.2, 0.7], [0, 0.6, 0, 0.6], [0, 0.4, 0.4, 0]])
    coordinates = _cic_coordinates(*jnp.asarray(position), grid)
    field = jnp.asarray(rng.normal(size=(3, 4, 6, 8)))
    source = jnp.asarray(rng.normal(size=(3, 64)))
    sampled = _cic_sample_many(field, coordinates)
    deposited = _cic_deposit_many(source, coordinates, (4, 6, 8))
    np.testing.assert_allclose(
        jnp.sum(deposited, axis=(1, 2, 3)),
        jnp.sum(source, axis=1),
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        jnp.sum(field * deposited), jnp.sum(sampled * source), rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        _cic_sample_many(jnp.ones_like(field), coordinates), 1, atol=3e-16
    )


def test_drag_reaction_and_evaporated_momentum_match_open_system_balance():
    """Reynolds transport theorem determines the vapor velocity correction."""
    config = MoistureConfig()
    properties = WaterDropletProperties()
    velocity = jnp.array([[22.0, 13.0], [4.0, -2.0], [0.0, 0.3]])
    gas_velocity = jnp.array([[3.0, 2.0], [0.1, -0.1], [0.05, -0.02]])
    dt = 1.25e-4
    initial_mass = jnp.array([2e-8, 5e-9])
    evaporated = jnp.array([3e-11, 1e-11])
    diameter = jnp.cbrt(6 * initial_mass / (jnp.pi * config.water_density))
    final_velocity, _, reaction = advance_water_droplet_motion(
        velocity, gas_velocity, diameter, dt, config, properties
    )
    gas_acceleration_impulse = initial_mass * reaction + evaporated * (
        final_velocity - gas_velocity
    )
    # Adding vapor at the carrier velocity accounts for carrier mass growth.
    carrier_total_momentum_gain = gas_acceleration_impulse + evaporated * gas_velocity
    parcel_momentum_change = (
        initial_mass - evaporated
    ) * final_velocity - initial_mass * velocity
    external_gravity_impulse = (
        initial_mass
        * jnp.array([0.0, 0.0, -9.81])[:, None]
        * (1 - config.dry_air_density / config.water_density)
        * dt
    )
    np.testing.assert_allclose(
        carrier_total_momentum_gain + parcel_momentum_change,
        external_gravity_impulse,
        atol=2e-22,
        rtol=2e-12,
    )


def test_disabled_source_is_exact_identity_without_zero_mass_parcel_activation():
    from jaxwind.water_parcels import (
        WaterParcelSource,
        initial_water_parcels,
        inject_water_parcels,
    )

    source = WaterParcelSource(
        center=(0.0, 0.2, 0.2),
        radius=0.002,
        speed=22.0,
        temperature=308.35,
        mass_flow=0.0,
        half_angle_degrees=18.0,
        diameter_scale=369e-6,
        diameter_spread=3.67,
        diameter_minimum=74e-6,
        diameter_maximum=518e-6,
        count_per_step=8,
        capacity=16,
    )
    initial = initial_water_parcels(source, "float64")
    inject = jax.jit(
        lambda p: inject_water_parcels(
            p, 1.0, jnp.asarray(10), 0.0005, source, MoistureConfig()
        )
    )
    result = inject(initial)
    for before, after in zip(initial, result):
        np.testing.assert_array_equal(before, after)
    assert not bool(jnp.any(result.active))


def test_disabled_source_does_not_allow_negative_or_nonfinite_flow():
    from dataclasses import replace

    import pytest

    from jaxwind.water_parcels import WaterParcelSource

    source = WaterParcelSource(
        center=(0.0, 0.2, 0.2),
        radius=0.002,
        speed=22.0,
        temperature=308.35,
        mass_flow=0.0,
        half_angle_degrees=18.0,
        diameter_scale=369e-6,
        diameter_spread=3.67,
        diameter_minimum=74e-6,
        diameter_maximum=518e-6,
    )
    for invalid in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="mass flow"):
            replace(source, mass_flow=invalid)
