"""Supplied inlet gradients must close RANS stress work and source equations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind import FREE_SLIP, OPEN, Boundaries, InflowPlane, StaggeredVelocity, Wall
from jaxwind.domain import UniformGrid
from jaxwind.inlet_momentum import physical_inlet_gradients
from jaxwind.rans_kepsilon import (
    C1,
    C2,
    CMU,
    KEpsilonState,
    advance_turbulence,
    production,
)
from jaxwind.rans_realizable import turbulent_viscosity, velocity_invariants
from jaxwind.sgs import edge_gradients, stress_divergence

jax.config.update("jax_enable_x64", True)


def _boundaries():
    return Boundaries(
        Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP
    )


def _boundary_shear(gamma=3.25):
    # A resting interior with a prescribed transverse inlet isolates the
    # supplied boundary derivative. Test bulk cells are >3 cells from walls.
    grid = UniformGrid(8, 12, 12, 1.6, 1.2, 1.2)
    velocity = StaggeredVelocity(
        jnp.zeros((grid.nz, grid.ny, grid.nx + 1)),
        jnp.zeros((grid.nz, grid.ny + 1, grid.nx)),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
    )
    v_inlet = jnp.full((grid.nz, grid.ny + 1), -gamma * grid.dx)
    v_inlet = v_inlet.at[:, 0].set(0).at[:, -1].set(0)
    plane = InflowPlane(
        velocity.x[..., 0],
        v_inlet,
        velocity.z[..., 0],
        jnp.zeros((grid.nz, grid.ny)),
    )
    shape = (grid.nz, grid.ny, grid.nx)
    state = KEpsilonState(jnp.full(shape, 0.09), jnp.full(shape, 0.11))
    boundaries = _boundaries()
    gradients = physical_inlet_gradients(velocity, plane, grid, boundaries)
    return grid, velocity, state, boundaries, gradients


def test_production_balances_variable_viscosity_stress_work_and_inlet_traction():
    grid = UniformGrid(8, 8, 6, 1.6, 1.2, 0.9)
    boundaries = _boundaries()
    rng = np.random.default_rng(5127)
    shapes = (
        (grid.nz, grid.ny, grid.nx + 1),
        (grid.nz, grid.ny + 1, grid.nx),
        (grid.nz + 1, grid.ny, grid.nx),
    )
    fields = []
    for component, shape in enumerate(shapes):
        field = np.zeros(shape)
        selection = (slice(1, -1), slice(1, -1), slice(1 if component == 0 else 0, 4))
        field[selection] = rng.normal(size=field[selection].shape)
        fields.append(jnp.asarray(field))
    velocity = StaggeredVelocity(*fields)
    by = np.zeros((grid.nz, grid.ny + 1))
    bz = np.zeros((grid.nz + 1, grid.ny))
    by[1:-1, 1:-1] = rng.normal(size=by[1:-1, 1:-1].shape)
    bz[1:-1, 1:-1] = rng.normal(size=bz[1:-1, 1:-1].shape)
    plane = InflowPlane(
        velocity.x[..., 0],
        jnp.asarray(by),
        jnp.asarray(bz),
        jnp.zeros((grid.nz, grid.ny)),
    )
    gradients = physical_inlet_gradients(velocity, plane, grid, boundaries)
    nu = 0.002 + 0.01 * rng.random((grid.nz, grid.ny, grid.nx))
    force = stress_divergence(
        velocity, jnp.asarray(nu), grid, boundaries, gradients=gradients
    )
    generated = production(
        velocity, grid, boundaries, jnp.asarray(nu), gradients=gradients
    )
    volume = grid.dx * grid.dy * grid.dz
    # Normal boundary velocities vanish, so half-dual boundary weights add zero.
    work = sum(float(jnp.sum(u * f)) for u, f in zip(velocity, force)) * volume
    # Independently assemble inlet traction using physical face values, the
    # half-cell derivative and arithmetic native face viscosity. u_inlet=0,
    # so cross derivatives du/dy and du/dz vanish on this inlet plane.
    nu_y = 0.5 * (nu[:, :-1, 0] + nu[:, 1:, 0])
    nu_z = 0.5 * (nu[:-1, :, 0] + nu[1:, :, 0])
    tau_y = nu_y * (np.asarray(velocity.y)[:, 1:-1, 0] - by[:, 1:-1]) / (0.5 * grid.dx)
    tau_z = nu_z * (np.asarray(velocity.z)[1:-1, :, 0] - bz[1:-1]) / (0.5 * grid.dx)
    boundary_work = (
        -float(np.sum(by[:, 1:-1] * tau_y) + np.sum(bz[1:-1] * tau_z))
        * grid.dy
        * grid.dz
    )
    generated_work = float(jnp.sum(generated)) * volume
    assert np.asarray(generated).min() >= 0
    np.testing.assert_allclose(
        generated_work, boundary_work - work, rtol=3e-14, atol=1e-12
    )
    legacy = (
        float(jnp.sum(production(velocity, grid, boundaries, jnp.asarray(nu)))) * volume
    )
    assert abs(legacy - generated_work) > 1e-5


def test_realizable_invariants_viscosity_and_native_power_use_supplied_shear():
    gamma = 3.25
    grid, velocity, state, boundaries, gradients = _boundary_shear(gamma)
    point = (5, 5, 0)
    s, ustar, a_s = velocity_invariants(velocity, grid, boundaries, gradients=gradients)
    np.testing.assert_allclose(
        [s[point], ustar[point], a_s[point]], [gamma, gamma, 3 / np.sqrt(2)], rtol=1e-14
    )
    nu = turbulent_viscosity(state, velocity, grid, boundaries, gradients=gradients)
    expected_nu = 0.09**2 / 0.11 / (4.04 + (3 / np.sqrt(2)) * gamma * 0.09 / 0.11)
    np.testing.assert_allclose(nu[point], expected_nu, rtol=2e-14)
    power = production(velocity, grid, boundaries, nu, gradients=gradients)
    # At x=0 the edge derivative is 2*gamma, and the next edge is zero.
    # Native edge-power averaging therefore gives 2*nu*gamma^2, not the
    # nu*gamma^2 obtained by squaring the averaged cell gradient.
    np.testing.assert_allclose(power[point], 2 * expected_nu * gamma**2, rtol=2e-14)
    assert float(velocity_invariants(velocity, grid, boundaries)[0][point]) == 0


@pytest.mark.parametrize("model", ["standard-k-epsilon", "realizable-k-epsilon"])
def test_advance_forwards_supplied_gradients_to_all_local_source_paths(model):
    gamma, dt, molecular = 3.25, 0.0005, 1.7e-5
    grid, velocity, state, boundaries, gradients = _boundary_shear(gamma)
    result = jax.jit(
        lambda initial: advance_turbulence(
            initial,
            velocity,
            grid,
            boundaries,
            molecular,
            (0.09, 0.11),
            dt,
            model=model,
            gradients=gradients,
        )
    )(state)
    k, eps = 0.09, 0.11
    if model == "standard-k-epsilon":
        nu = CMU * k**2 / eps
    else:
        nu = k**2 / eps / (4.04 + (3 / np.sqrt(2)) * gamma * k / eps)
    generation = 2 * nu * gamma**2
    expected_k = (k + dt * generation) / (1 + dt * eps / k)
    if model == "standard-k-epsilon":
        expected_eps = (eps + dt * C1 * eps / k * generation) / (1 + dt * C2 * eps / k)
    else:
        eta = gamma * k / eps
        c1 = max(0.43, eta / (eta + 5))
        expected_eps = (eps + dt * c1 * gamma * eps) / (
            1 + dt * 1.9 * eps / (k + np.sqrt(molecular * eps))
        )
    # Scalar fields are constant in this interior neighborhood; zero velocity
    # and sufficient wall separation isolate the first-step source equations.
    point = (5, 5, 0)
    np.testing.assert_allclose(
        [result.kinetic_energy[point], result.dissipation[point]],
        [expected_k, expected_eps],
        rtol=3e-13,
        atol=1e-14,
    )
    assert float(result.kinetic_energy[point]) > k / (1 + dt * eps / k) + 1e-5


def test_explicit_native_gradients_preserve_default_closures():
    grid, velocity, state, boundaries, _ = _boundary_shear()
    native = edge_gradients(velocity, grid, boundaries)
    implicit = velocity_invariants(velocity, grid, boundaries)
    explicit = velocity_invariants(velocity, grid, boundaries, gradients=native)
    for a, b in zip(implicit, explicit):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        turbulent_viscosity(state, velocity, grid, boundaries),
        turbulent_viscosity(state, velocity, grid, boundaries, gradients=native),
    )
    np.testing.assert_array_equal(
        production(velocity, grid, boundaries, 0.01),
        production(velocity, grid, boundaries, 0.01, gradients=native),
    )
