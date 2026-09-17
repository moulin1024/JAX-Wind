"""Dense-matrix, analytic and mechanical-ledger pressure verification."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.domain import UniformGrid
from jaxwind.spray_pressure import build_momentum_projection
from jaxwind.state import StaggeredVelocity

jax.config.update("jax_enable_x64", True)


def np_divergence(flow, grid, px, py):
    answer = 0.0
    for axis, f, h, periodic in zip(
        (2, 1, 0), flow, (grid.dx, grid.dy, grid.dz), (px, py, False)
    ):
        a = np.asarray(f)
        answer = (
            answer
            + ((np.roll(a, -1, axis) - a) if periodic else np.diff(a, axis=axis)) / h
        )
    return answer


def reference_matrix(beta, grid, px, py, open_low):
    shape = (grid.nz, grid.ny, grid.nx)
    matrix = np.zeros((np.prod(shape), np.prod(shape)))
    for cell in np.ndindex(shape):
        row = np.ravel_multi_index(cell, shape)
        for axis, face, h, periodic in zip(
            (2, 1, 0), beta, (grid.dx, grid.dy, grid.dz), (px, py, False)
        ):
            for sign in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += sign
                face_index = list(cell)
                face_index[axis] += int(sign > 0)
                if periodic:
                    face_index[axis] %= shape[axis]
                coefficient = float(face[tuple(face_index)]) / h**2
                if not 0 <= neighbor[axis] < shape[axis] and not periodic:
                    if axis == 2 and (sign == 1 or open_low):
                        matrix[row, row] += 2 * coefficient
                    continue
                neighbor[axis] %= shape[axis]
                column = np.ravel_multi_index(tuple(neighbor), shape)
                matrix[row, row] += coefficient
                matrix[row, column] -= coefficient
    return matrix


def reference_gradient(p, grid, px, py, open_low):
    result = []
    for axis, h, periodic in zip(
        (2, 1, 0), (grid.dx, grid.dy, grid.dz), (px, py, False)
    ):
        if periodic:
            a = (p - np.roll(p, 1, axis)) / h
        else:
            shape = list(p.shape)
            shape[axis] += 1
            a = np.zeros(shape)
            for face in np.ndindex(tuple(shape)):
                k = face[axis]
                if k == 0:
                    if axis == 2 and open_low:
                        a[face] = 2 * p[face] / h
                elif k == p.shape[axis]:
                    cell = list(face)
                    cell[axis] -= 1
                    if axis == 2:
                        a[face] = -2 * p[tuple(cell)] / h
                else:
                    left = list(face)
                    left[axis] -= 1
                    a[face] = (p[face] - p[tuple(left)]) / h
        result.append(a)
    return result


def setup(px, py):
    grid = UniformGrid(5, 4, 3, 1.0, 0.8, 0.6)
    shape = (grid.nz, grid.ny, grid.nx)
    rng = np.random.default_rng(791)
    u = StaggeredVelocity(
        jnp.asarray(
            rng.normal(size=(grid.nz, grid.ny, grid.nx if px else grid.nx + 1))
        ),
        jnp.asarray(
            rng.normal(size=(grid.nz, grid.ny if py else grid.ny + 1, grid.nx))
        ),
        jnp.asarray(rng.normal(size=(grid.nz + 1, grid.ny, grid.nx))),
    )
    u = u._replace(z=u.z.at[0].set(0).at[-1].set(0))
    if not py:
        u = u._replace(y=u.y.at[:, 0].set(0).at[:, -1].set(0))
    inertia = jax.tree.map(lambda v: jnp.asarray(0.5 + rng.random(v.shape)), u)
    transport = jax.tree.map(lambda v: jnp.asarray(0.3 + 2 * rng.random(v.shape)), u)
    old = jnp.asarray(1 + 0.1 * rng.random(shape))
    delta = rng.normal(size=shape)
    delta -= delta.mean()
    source = rng.normal(size=shape)
    source -= source.mean()
    return (
        grid,
        u,
        inertia,
        transport,
        old,
        old + 0.005 * delta,
        jnp.asarray(0.001 * source),
    )


@pytest.mark.parametrize(
    "px,py,open_low",
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, True),
        (False, True, False),
    ],
)
def test_dense_reference_and_pressure_impulse_work(
    px, py, open_low, record_testsuite_property
):
    grid, u, inertia, transport, old, new, increment = setup(px, py)
    dt = 0.01
    projection = jax.jit(
        build_momentum_projection(
            grid, periodic_x=px, periodic_y=py, open_x_low=open_low
        )
    )
    result = projection(u, old, new, transport, inertia, increment, dt)
    assert bool(result.accepted), (
        result.continuity_error,
        result.linear_error,
        result.iterations,
    )
    beta = [np.asarray(t / r) for t, r in zip(transport, inertia)]
    matrix = reference_matrix(beta, grid, px, py, open_low)
    flow = [np.asarray(t * v) for t, v in zip(transport, u)]
    b = (
        -((np.asarray(new - old - increment)) / dt + np_divergence(flow, grid, px, py))
        / dt
    )
    if px:
        matrix += np.ones_like(matrix) / matrix.shape[0]
        b -= b.mean()
    reference = np.linalg.solve(matrix, b.ravel()).reshape(old.shape)
    np.testing.assert_allclose(result.pressure, reference, atol=3e-9, rtol=1e-9)
    grad = reference_gradient(np.asarray(result.pressure), grid, px, py, open_low)
    for k in range(3):
        actual_impulse = np.asarray(inertia[k] * (result.velocity[k] - u[k]))
        np.testing.assert_allclose(
            actual_impulse, -dt * grad[k], atol=2e-15, rtol=3e-13
        )
        np.testing.assert_allclose(
            result.impulse[k], actual_impulse, atol=2e-15, rtol=3e-13
        )
        delta_ke = np.asarray(inertia[k] * (result.velocity[k] ** 2 - u[k] ** 2) / 2)
        np.testing.assert_allclose(
            result.kinetic_work[k], delta_ke, atol=3e-15, rtol=5e-13
        )
        np.testing.assert_allclose(
            result.mass_flux[k], transport[k] * result.velocity[k], atol=1e-15
        )
    # Independent primary/dual quadrature checks the pressure work by discrete
    # integration by parts, including fixed-inlet pressure work when present.
    midpoint = [(np.asarray(a) + np.asarray(b)) / 2 for a, b in zip(u, result.velocity)]
    integrated_work = 0.0
    for component, work in zip((2, 1, 0), result.kinetic_work):
        widths = [
            np.full(grid.nz, grid.dz),
            np.full(grid.ny, grid.dy),
            np.full(grid.nx, grid.dx),
        ]
        if not (False, py, px)[component]:
            h = widths[component][0]
            widths[component] = np.r_[
                h / 2, np.full(len(widths[component]) - 1, h), h / 2
            ]
        volumes = (
            widths[0][:, None, None]
            * widths[1][None, :, None]
            * widths[2][None, None, :]
        )
        integrated_work += np.sum(np.asarray(work) * volumes)
    dilation_work = (
        dt
        * np.sum(np.asarray(result.pressure) * np_divergence(midpoint, grid, px, py))
        * grid.dx
        * grid.dy
        * grid.dz
    )
    if not px and not open_low:
        dilation_work += (
            dt
            * np.sum(np.asarray(result.pressure)[..., 0] * midpoint[0][..., 0])
            * grid.dy
            * grid.dz
        )
    np.testing.assert_allclose(integrated_work, dilation_work, atol=5e-15, rtol=1e-12)
    residual = np.asarray(new - old - increment) + dt * np_divergence(
        result.mass_flux, grid, px, py
    )
    assert np.max(np.abs(residual) / np.asarray(old)) < 1e-10
    record_testsuite_property(
        f"pressure_continuity_{px}_{py}_{open_low}", float(result.continuity_error)
    )
    record_testsuite_property(
        f"pressure_iterations_{px}_{py}_{open_low}", int(result.iterations)
    )


@pytest.mark.parametrize("px", [True, False])
def test_independent_continuum_pressure_has_second_order_grid_convergence(
    px, record_testsuite_property
):
    errors = []
    for n in (8, 16, 32):
        grid = UniformGrid(n, n, n // 2, 1.0, 1.0, 1.0)
        x = (np.arange(n) + 0.5) / n
        y = (np.arange(n) + 0.5) / n
        z = (np.arange(n // 2) + 0.5) / (n // 2)
        kz = np.pi
        ky = 2 * np.pi
        kx = 2 * np.pi if px else np.pi
        exact = (
            np.sin(kx * x)[None, None, :]
            * np.cos(ky * y)[None, :, None]
            * np.cos(kz * z)[:, None, None]
        )
        shape = exact.shape
        zero = StaggeredVelocity(
            jnp.zeros(shape if px else (grid.nz, grid.ny, n + 1)),
            jnp.zeros(shape),
            jnp.zeros((grid.nz + 1, grid.ny, n)),
        )
        inertia = jax.tree.map(jnp.ones_like, zero)
        # Smooth nonconstant coefficient beta(x) for the periodic case; the
        # pressure-open case independently checks half-cell Dirichlet geometry.
        amplitude = 0.3 if px else 0.0
        xf = np.arange(n if px else n + 1) / n
        transport = StaggeredVelocity(
            jnp.broadcast_to(1 + amplitude * np.sin(2 * np.pi * xf), zero.x.shape),
            jnp.broadcast_to(1 + amplitude * np.sin(2 * np.pi * x), zero.y.shape),
            jnp.broadcast_to(1 + amplitude * np.sin(2 * np.pi * x), zero.z.shape),
        )
        beta = 1 + amplitude * np.sin(2 * np.pi * x)[None, None, :]
        derivative = amplitude * 2 * np.pi * np.cos(2 * np.pi * x)[None, None, :]
        px_exact = (
            kx
            * np.cos(kx * x)[None, None, :]
            * np.cos(ky * y)[None, :, None]
            * np.cos(kz * z)[:, None, None]
        )
        laplacian = -(kx * kx + ky * ky + kz * kz) * exact
        continuous_operator = beta * laplacian + derivative * px_exact
        dt = 0.001
        # rho_new=rho_old, predictor=0 => div(beta grad p)=-Delta_rho/dt^2.
        increment = jnp.asarray(-dt * dt * continuous_operator)
        density = jnp.ones(shape)
        result = jax.jit(
            build_momentum_projection(
                grid, periodic_x=px, periodic_y=True, max_iterations=800
            )
        )(zero, density, density, transport, inertia, increment, dt)
        assert bool(result.accepted), (
            n,
            result.continuity_error,
            result.linear_error,
            result.iterations,
        )
        error = float(np.sqrt(np.mean((np.asarray(result.pressure) - exact) ** 2)))
        errors.append(error)
    assert errors[0] / errors[1] > 3.7 and errors[1] / errors[2] > 3.7, errors
    record_testsuite_property(f"analytic_pressure_rms_{px}", str(errors))


def test_unit_coefficient_flux_projection_is_not_a_valid_momentum_force_control():
    grid, u, inertia, transport, old, new, increment = setup(True, True)
    dt = 0.01
    # Independent dense solution for the old unit-coefficient mass-flux rule.
    one = [np.ones(v.shape) for v in u]
    matrix = reference_matrix(one, grid, True, True, True)
    matrix += np.ones_like(matrix) / matrix.shape[0]
    flow = [np.asarray(t * v) for t, v in zip(transport, u)]
    b = (
        -(
            (np.asarray(new - old - increment)) / dt
            + np_divergence(flow, grid, True, True)
        )
        / dt
    )
    b -= b.mean()
    p = np.linalg.solve(matrix, b.ravel()).reshape(old.shape)
    gradients = reference_gradient(p, grid, True, True, True)
    errors = []
    for t, r, g in zip(transport, inertia, gradients):
        # Recovering velocity after F <- F-dt grad(p) gives the wrong force
        # whenever donor transport density differs from momentum inertia.
        actual = np.asarray(r / t) * (-dt * g)
        errors.append(np.linalg.norm(actual + dt * g) / np.linalg.norm(dt * g))
    assert max(errors) > 0.1


@pytest.mark.parametrize("failure", ["closed_source", "iterations", "negative_density"])
def test_projection_rejection_commits_no_flux_or_pressure_work(failure):
    grid, u, inertia, transport, old, new, increment = setup(True, True)
    if failure == "closed_source":
        increment = increment + 0.01
    if failure == "negative_density":
        transport = transport._replace(x=-transport.x)
    result = jax.jit(
        build_momentum_projection(
            grid,
            periodic_x=True,
            periodic_y=True,
            max_iterations=1 if failure == "iterations" else 400,
        )
    )(u, old, new, transport, inertia, increment, 0.01)
    assert not bool(result.accepted)
    for a, b in zip(result.velocity, u):
        np.testing.assert_array_equal(a, b)
    for v in jax.tree.leaves(
        (result.mass_flux, result.pressure, result.impulse, result.kinetic_work)
    ):
        assert np.all(np.asarray(v) == 0)


def test_zero_rhs_and_uniform_translation_require_no_pressure_iterations():
    grid, u, _, _, old, _, _ = setup(True, True)
    u = StaggeredVelocity(jnp.ones_like(u.x), jnp.zeros_like(u.y), jnp.zeros_like(u.z))
    one = jax.tree.map(jnp.ones_like, u)
    result = jax.jit(build_momentum_projection(grid, periodic_x=True, periodic_y=True))(
        u, old, old, one, one, jnp.zeros_like(old), 0.01
    )
    assert bool(result.accepted) and int(result.iterations) == 0
    for a, b in zip(result.velocity, u):
        np.testing.assert_array_equal(a, b)
