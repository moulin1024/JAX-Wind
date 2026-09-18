"""Independent limiting cases for the experimental transported RANS closure."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind import FREE_SLIP, OPEN, Boundaries, StaggeredVelocity, Wall
from jaxwind.domain import UniformGrid
from jaxwind.rans_kepsilon import (
    C1,
    C2,
    CMU,
    E_WALL,
    KAPPA,
    KEpsilonState,
    advance_turbulence,
    local_sources,
    production,
    turbulent_viscosity,
    wall_terms,
)

jax.config.update("jax_enable_x64", True)


def _tunnel(speed=3.0, kinetic_energy=0.09):
    grid = UniformGrid(8, 8, 8, 1.0, 0.585, 0.585)
    shape = (grid.nz, grid.ny, grid.nx)
    velocity = StaggeredVelocity(
        jnp.full((grid.nz, grid.ny, grid.nx + 1), speed),
        jnp.zeros((grid.nz, grid.ny + 1, grid.nx)),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
    )
    state = KEpsilonState(jnp.full(shape, kinetic_energy), jnp.full(shape, 0.1))
    boundaries = Boundaries(
        Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP
    )
    return grid, velocity, state, boundaries


def test_bulk_production_matches_simple_shear_and_rigid_rotation():
    grid, velocity, _, boundaries = _tunnel()
    nu = 0.0067
    shear = 2.3
    y = jnp.asarray(grid.y_centers)[None, :, None]
    x = jnp.broadcast_to(shear * y, velocity.x.shape)
    velocity = velocity._replace(x=x)
    actual = production(velocity, grid, boundaries, nu)
    np.testing.assert_allclose(actual[:, 1:-1, :], nu * shear**2, atol=1e-14)
    # Solid-body rotation has vorticity but no strain or turbulent production.
    x_centres = jnp.asarray(grid.x_centers)[None, None, :]
    rotation = velocity._replace(
        y=jnp.broadcast_to(-shear * x_centres, velocity.y.shape)
    )
    actual = production(rotation, grid, boundaries, nu)
    np.testing.assert_allclose(actual[:, 1:-1, 1:-1], 0.0, atol=1e-14)


def test_local_sources_have_correct_first_order_derivative():
    state = KEpsilonState(jnp.array([0.09, 2.0]), jnp.array([0.1083, 0.4]))
    generation = jnp.array([0.8, 0.1])
    dt = 1e-7
    updated = local_sources(state, generation, dt)
    k, eps = state
    np.testing.assert_allclose(
        (updated.kinetic_energy - k) / dt, generation - eps, rtol=1e-6
    )
    expected = (C1 * generation - C2 * eps) * eps / k
    np.testing.assert_allclose((updated.dissipation - eps) / dt, expected, rtol=1e-6)


def test_local_sources_remain_positive_for_stiff_destruction():
    state = KEpsilonState(jnp.array([0.09, 1e-6]), jnp.array([0.1083, 4.0]))
    updated = local_sources(state, jnp.array([0.0, 0.0]), 1e6)
    assert np.all(np.asarray(updated) > 0)
    assert np.all(np.asarray(updated) < np.asarray(state))
    assert np.all(np.isfinite(np.asarray(turbulent_viscosity(updated))))


def test_homogeneous_decay_converges_to_analytic_solution():
    initial = KEpsilonState(jnp.array(0.09), jnp.array(0.1083))
    duration = 0.5
    factor = 1 + (C2 - 1) * initial.dissipation / initial.kinetic_energy * duration
    exact = np.array(
        [
            initial.kinetic_energy * factor ** (-1 / (C2 - 1)),
            initial.dissipation * factor ** (-C2 / (C2 - 1)),
        ]
    )
    errors = []
    for count in (64, 128):
        dt = duration / count
        result = jax.lax.fori_loop(
            0, count, lambda _, state, dt=dt: local_sources(state, 0.0, dt), initial
        )
        errors.append(np.linalg.norm((np.asarray(result) - exact) / exact))
    assert 1.9 < errors[0] / errors[1] < 2.1
    assert errors[1] < 0.002


def test_log_wall_equilibrium_and_total_wall_traction():
    grid, velocity, state, _ = _tunnel()
    molecular = 1.5e-5
    distance = grid.dy / 2
    scale = CMU**0.25 * np.sqrt(0.09)
    ystar = scale * distance / molecular
    speed = scale * np.log(E_WALL * ystar) / KAPPA
    velocity = velocity._replace(x=jnp.full_like(velocity.x, speed))
    force, generation, epsilon, mask = wall_terms(velocity, state, grid, molecular)
    expected = scale**3 / (KAPPA * distance)
    np.testing.assert_allclose(np.asarray(generation)[mask], expected, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(epsilon)[mask], expected, rtol=1e-14)
    # Integrate the staggered force over its dual volumes, including half
    # volumes at the two open x faces. Four wall tractions must add at corners.
    x_weights = np.ones(grid.nx + 1)
    x_weights[[0, -1]] = 0.5
    integrated_force = (
        np.sum(np.asarray(force.x) * x_weights[None, None, :])
        * grid.dx
        * grid.dy
        * grid.dz
    )
    expected_force = -(scale**2) * (2 * grid.lx * grid.ly + 2 * grid.lx * grid.lz)
    np.testing.assert_allclose(integrated_force, expected_force, rtol=1e-14)
    assert float(jnp.sum(force.x * velocity.x)) < 0
    np.testing.assert_allclose(force.y, 0.0)
    np.testing.assert_allclose(force.z, 0.0)


@pytest.mark.parametrize("kinetic_energy", [1e-8, 1e-5])
def test_viscous_wall_has_finite_decay_without_false_production(kinetic_energy):
    grid, velocity, state, _ = _tunnel(kinetic_energy=kinetic_energy)
    molecular = 1.5e-5
    force, generation, epsilon, mask = wall_terms(velocity, state, grid, molecular)
    np.testing.assert_allclose(generation, 0.0)
    expected = 2 * molecular * kinetic_energy / (grid.dy / 2) ** 2
    np.testing.assert_allclose(np.asarray(epsilon)[mask], expected, rtol=1e-14)
    assert np.all(np.isfinite(np.asarray(force.x)))
    assert float(jnp.sum(force.x * velocity.x)) < 0


def test_traction_switch_is_continuous_to_rounding_of_yplus_threshold():
    grid, _, _, _ = _tunnel()
    molecular = 1.5e-5
    distance = grid.dy / 2
    integrated = []
    for ystar in (11.225 * (1 - 1e-8), 11.225 * (1 + 1e-8)):
        kinetic_energy = (ystar * molecular / distance) ** 2 / CMU**0.5
        grid, velocity, state, _ = _tunnel(kinetic_energy=kinetic_energy)
        force, _, _, _ = wall_terms(velocity, state, grid, molecular)
        integrated.append(float(jnp.sum(force.x)))
    np.testing.assert_allclose(integrated[0], integrated[1], rtol=3e-5)


def test_transported_turbulence_is_positive_under_a_stiff_diffusion_step():
    grid, velocity, state, boundaries = _tunnel()
    x = jnp.asarray(grid.x_centers)[None, None, :]
    state = KEpsilonState(
        state.kinetic_energy * (1 + 0.9 * jnp.cos(2 * jnp.pi * x / grid.lx)),
        state.dissipation,
    )
    step = jax.jit(
        lambda s: advance_turbulence(
            s, velocity, grid, boundaries, 1.5e-5, (0.09, 0.1083), 0.1
        )
    )
    updated = step(state)
    assert np.all(np.isfinite(np.asarray(updated)))
    assert np.all(np.asarray(updated) > 0)


@pytest.mark.parametrize("solenoidal", [False, True])
def test_bulk_production_equals_native_stress_work_for_variable_viscosity(solenoidal):
    """No energy flux crosses the compact-support field's boundaries.

    This fails with collocated-gradient squaring by about 43%, independently
    of the implementation's interpolation formulas.
    """
    from jaxwind.sgs import stress_divergence

    grid = UniformGrid(12, 10, 8, 1.9, 0.585, 0.585)
    boundaries = Boundaries(
        Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP
    )
    rng = np.random.default_rng(7383)
    fields = []
    for shape in ((8, 10, 13), (8, 11, 12), (9, 10, 12)):
        field = np.zeros(shape)
        field[2:-2, 2:-2, 2:-2] = rng.normal(size=field[2:-2, 2:-2, 2:-2].shape)
        fields.append(jnp.asarray(field))
    velocity = StaggeredVelocity(*fields)
    if solenoidal:
        # Discrete curl of a compact-support streamfunction is exactly
        # divergence-free on this MAC grid, independent of the stress code.
        from jaxwind.numerics.discretization import divergence

        psi = np.zeros((grid.nz, grid.ny + 1, grid.nx + 1))
        psi[2:-2, 2:-2, 2:-2] = rng.normal(size=psi[2:-2, 2:-2, 2:-2].shape)
        velocity = StaggeredVelocity(
            jnp.asarray(np.diff(psi, axis=1) / grid.dy),
            jnp.asarray(-np.diff(psi, axis=2) / grid.dx),
            jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
        )
        assert float(jnp.max(jnp.abs(divergence(velocity, grid)))) < 1e-12
    viscosity = jnp.asarray(0.002 + 0.01 * rng.random((8, 10, 12)))
    force = stress_divergence(velocity, viscosity, grid, boundaries)
    # Boundary face velocities vanish, so their half-dual weights contribute zero.
    removed = -sum(jnp.sum(u * f) for u, f in zip(velocity, force))
    generated = production(velocity, grid, boundaries, viscosity)
    assert bool(jnp.all(generated >= 0))
    np.testing.assert_allclose(jnp.sum(generated), removed, rtol=2e-14, atol=1e-12)
