"""Pressure correction consistent with distinct transport and inertia densities.

Uniform MAC grids; x periodic or pressure-open (optionally fixed-flux inlet),
y periodic or impermeable, z impermeable. This is an opt-in pressure primitive,
not a replacement of production Poisson or a completed spray timestep.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .numerics.discretization import divergence
from .state import StaggeredVelocity


class MomentumProjection(NamedTuple):
    velocity: StaggeredVelocity
    mass_flux: StaggeredVelocity
    pressure: jax.Array
    impulse: StaggeredVelocity
    kinetic_work: StaggeredVelocity
    accepted: jax.Array
    iterations: jax.Array
    continuity_error: jax.Array
    linear_error: jax.Array


def momentum_pressure_gradient(p, grid, *, periodic_x, periodic_y, open_x_low=True):
    """Full FV gradient, including transverse gradients at open end columns.

    Pressure is zero on open x faces; other nonperiodic faces have zero normal
    gradient. The half-cell distance is used at pressure-open boundary faces.
    Unlike the legacy open solver, transverse end-column gradients are retained.
    """
    parts = []
    for axis, periodic, h in (
        (2, periodic_x, grid.dx),
        (1, periodic_y, grid.dy),
        (0, False, grid.dz),
    ):
        if periodic:
            parts.append((p - jnp.roll(p, 1, axis)) / h)
        else:
            low = jnp.take(p, jnp.array([0]), axis=axis)
            high = jnp.take(p, jnp.array([-1]), axis=axis)
            low = 2 * low / h if axis == 2 and open_x_low else jnp.zeros_like(low)
            high = -2 * high / h if axis == 2 else jnp.zeros_like(high)
            parts.append(
                jnp.concatenate((low, jnp.diff(p, axis=axis) / h, high), axis=axis)
            )
    return StaggeredVelocity(*parts)


def _diagonal(beta, grid, periodic, open_x_low):
    diagonal = 0.0
    for axis, face, h in zip((2, 1, 0), beta, (grid.dx, grid.dy, grid.dz)):
        if periodic[axis]:
            term = face + jnp.roll(face, -1, axis)
        else:
            low = jnp.take(face, jnp.arange(face.shape[axis] - 1), axis=axis)
            high = jnp.take(face, jnp.arange(1, face.shape[axis]), axis=axis)
            idx = [slice(None)] * 3
            idx[axis] = 0
            low = low.at[tuple(idx)].multiply(2 if axis == 2 and open_x_low else 0)
            idx[axis] = -1
            high = high.at[tuple(idx)].multiply(2 if axis == 2 else 0)
            term = low + high
        diagonal = diagonal + term / h**2
    return diagonal


def build_momentum_projection(
    grid,
    *,
    periodic_x,
    periodic_y,
    open_x_low=True,
    tolerance=1e-10,
    linear_tolerance=1e-11,
    max_iterations=400,
):
    """Project velocity with pressure force based on momentum inertia.

    step(predictor, rho_old, rho_new, transport_density, inertia_density,
         mass_increment, dt)

    F*=rho_transport*u*; beta=rho_transport/rho_inertia.
    Solve div(beta*grad(p)) = ((rho_new-rho_old-Delta_rho)/dt+div(F*))/dt.
    Then u=u*-dt*grad(p)/rho_inertia and F=rho_transport*u.
    Sources are increments per volume, not rates. Pressure force and midpoint
    kinetic work are returned on the component dual volumes. No heat/SGS update
    is made here; pressure work is a mechanical ledger, not prescribed heating.

    PCG uses the positive operator -div(beta*grad) and its Jacobi diagonal.
    Closed-domain pressure has zero mean. Incompatible mean mass requirements
    are never repaired: the unmodified physical continuity residual must pass.
    Rejection restores predictor velocity and zeros all committed pressure,
    flux, impulse and work. Fields supplied by another stage are not owned here.
    """
    if not grid.is_uniform:
        raise ValueError("spray momentum pressure requires a uniform grid")
    if tolerance <= 0 or linear_tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid pressure iteration controls")
    periodic = (False, periodic_y, periodic_x)
    gradient = lambda p: momentum_pressure_gradient(
        p, grid, periodic_x=periodic_x, periodic_y=periodic_y, open_x_low=open_x_low
    )
    gauge = (lambda p: p - jnp.mean(p)) if periodic_x else (lambda p: p)

    def step(predictor, old, new, transport_density, inertia_density, increment, dt):
        if old.shape != new.shape or old.shape != increment.shape:
            raise ValueError("density and source shapes must match")
        if any(
            u.shape != t.shape or u.shape != r.shape
            for u, t, r in zip(predictor, transport_density, inertia_density)
        ):
            raise ValueError("face density and velocity shapes must match")
        valid = (dt > 0) & jnp.all(old > 0) & jnp.all(new > 0)
        valid &= jnp.all(jnp.isfinite(jnp.stack((old, new, increment))))
        for u, t, r in zip(predictor, transport_density, inertia_density):
            valid &= (
                jnp.all(jnp.isfinite(u))
                & jnp.all(jnp.isfinite(t))
                & jnp.all(jnp.isfinite(r))
            )
            valid &= jnp.all(t > 0) & jnp.all(r > 0)
        valid &= jnp.all(predictor.z[0] == 0) & jnp.all(predictor.z[-1] == 0)
        if not periodic_y:
            valid &= jnp.all(predictor.y[:, 0] == 0) & jnp.all(predictor.y[:, -1] == 0)
        beta = StaggeredVelocity(
            *(t / r for t, r in zip(transport_density, inertia_density))
        )
        flow = StaggeredVelocity(*(t * u for t, u in zip(transport_density, predictor)))
        physical = (new - old - increment) / dt + divergence(flow, grid)
        rhs = -physical / dt
        b = gauge(rhs)
        diagonal = _diagonal(beta, grid, periodic, open_x_low)
        operator = lambda p: (
            -divergence(
                StaggeredVelocity(*(a * g for a, g in zip(beta, gradient(p)))), grid
            )
        )
        norm = lambda p: jnp.sqrt(jnp.sum(p * p))
        z = gauge(b / diagonal)
        initial = (
            jnp.array(0),
            jnp.zeros_like(old),
            b,
            z,
            jnp.sum(b * z),
            norm(b),
            valid,
        )
        target = linear_tolerance * norm(b)

        def condition(state):
            n, _, _, _, _, error, finite = state
            return (n < max_iterations) & (error > target) & finite

        def iterate(state):
            n, p, residual, direction, rz, _, finite = state
            action = operator(direction)
            denom = jnp.sum(direction * action)
            alpha = rz / denom
            p = gauge(p + alpha * direction)
            residual = gauge(residual - alpha * action)
            z = gauge(residual / diagonal)
            rz_new = jnp.sum(residual * z)
            direction = gauge(z + (rz_new / rz) * direction)
            finite &= (denom > 0) & jnp.isfinite(rz_new) & jnp.all(jnp.isfinite(p))
            return n + 1, p, residual, direction, rz_new, norm(residual), finite

        n, pressure, _, _, _, _, finite = jax.lax.while_loop(
            condition, iterate, initial
        )
        grad = gradient(pressure)
        impulse = jax.tree.map(lambda g: -dt * g, grad)
        velocity = StaggeredVelocity(
            *(u + j / r for u, j, r in zip(predictor, impulse, inertia_density))
        )
        corrected = StaggeredVelocity(
            *(t * u for t, u in zip(transport_density, velocity))
        )
        residual = new - old - increment + dt * divergence(corrected, grid)
        continuity_error = jnp.max(jnp.abs(residual) / old)
        true_linear_error = norm(operator(pressure) - b)
        linear_error = true_linear_error / jnp.maximum(
            norm(b), jnp.finfo(old.dtype).tiny
        )
        # An exactly zero RHS needs no iterations; explicitly check the true
        # residual because recursive CG residuals alone can drift at roundoff.
        solved = true_linear_error <= jnp.maximum(
            target * 10, 100 * jnp.finfo(old.dtype).eps * norm(b)
        )
        accepted = valid & finite & solved & (continuity_error <= tolerance)
        work = StaggeredVelocity(
            *(j * (a + b) / 2 for j, a, b in zip(impulse, predictor, velocity))
        )
        choose = lambda a, b: jnp.where(accepted, a, b)
        commit = lambda tree: jax.tree.map(lambda q: choose(q, jnp.zeros_like(q)), tree)
        return MomentumProjection(
            jax.tree.map(choose, velocity, predictor),
            commit(corrected),
            commit(pressure),
            commit(impulse),
            commit(work),
            accepted,
            n,
            continuity_error,
            linear_error,
        )

    return step
