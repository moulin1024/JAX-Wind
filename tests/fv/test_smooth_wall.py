"""Smooth-wall closure inversion, laminar limit and four-wall momentum flux."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind import StaggeredVelocity
from jaxwind.domain import UniformGrid
from jaxwind.smooth_wall import smooth_duct_tendency, smooth_friction_velocity

jax.config.update("jax_enable_x64", True)


def test_spalding_inverse_matches_prescribed_profile_and_viscous_limit():
    up = np.array([0.001, 0.1, 1.0, 5.0, 15.0, 25.0])
    a = 0.41 * up
    yp = up + (np.expm1(a) - a - a * a / 2 - a**3 / 6) / 9.8
    friction, nu = 0.2, 1.7e-5
    actual = smooth_friction_velocity(
        jnp.asarray(up * friction), yp * nu / friction, nu
    )
    np.testing.assert_allclose(actual, friction, rtol=1e-10)
    speed, y = 1e-5, 0.001
    np.testing.assert_allclose(
        smooth_friction_velocity(speed, y, nu), np.sqrt(nu * speed / y), rtol=1e-8
    )
    assert float(smooth_friction_velocity(0.0, y, nu)) == 0


def test_four_wall_integrated_streamwise_force_and_opposition():
    grid = UniformGrid(8, 6, 4, 1.9, 0.585, 0.4)
    velocity = StaggeredVelocity(
        jnp.full((4, 6, 9), 3.0), jnp.zeros((4, 7, 8)), jnp.zeros((5, 6, 8))
    )
    force = smooth_duct_tendency(velocity, grid, 1.7e-5)
    # Constant in x, so face and centre integrals coincide exactly.
    actual = float(jnp.sum(force.x[..., :-1] * jnp.asarray(grid.cell_volumes)))
    tz = float(smooth_friction_velocity(3.0, 0.05, 1.7e-5)) ** 2
    ty = float(smooth_friction_velocity(3.0, 0.585 / 12, 1.7e-5)) ** 2
    expected = -2 * grid.lx * (grid.ly * tz + grid.lz * ty)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert np.all(np.asarray(force.x) <= 0)
    np.testing.assert_array_equal(force.y, 0)
    np.testing.assert_array_equal(force.z, 0)
    np.testing.assert_array_equal(force.x[1:-1, 1:-1], 0)
    reverse = smooth_duct_tendency(
        StaggeredVelocity(-velocity.x, velocity.y, velocity.z), grid, 1.7e-5
    )
    np.testing.assert_allclose(reverse.x, -force.x, rtol=1e-14)


def test_four_wall_stress_dissipates_staggered_kinetic_energy():
    grid = UniformGrid(8, 6, 4, 1.9, 0.585, 0.4)
    rng = np.random.default_rng(21)
    u = rng.normal(size=(4, 6, 9))
    v = rng.normal(size=(4, 7, 8))
    w = rng.normal(size=(5, 6, 8))
    v[:, [0, -1]] = 0
    w[[0, -1]] = 0
    velocity = StaggeredVelocity(*(jnp.asarray(a) for a in (u, v, w)))
    force = smooth_duct_tendency(velocity, grid, 1.7e-5)
    power = 0.0
    for component, (speed, acceleration) in enumerate(zip(velocity, force)):
        weights = np.ones(speed.shape)
        slices = [slice(None)] * 3
        slices[2 - component] = [0, -1]
        weights[tuple(slices)] = 0.5
        power += float(np.sum(weights * np.asarray(speed * acceleration)))
    assert power < 0
    np.testing.assert_array_equal(force.y[:, [0, -1]], 0)
    np.testing.assert_array_equal(np.asarray(force.z)[[0, -1]], 0)
