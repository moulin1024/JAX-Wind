"""Stationary-frame realizable k-epsilon constitutive and local source terms.

Shih et al. (1995); Fluent 12 Theory Guide 4.4.3, equations 4.4-15--19.
This is an opt-in carrier closure, not a fitted jet diffusivity. The surrounding
RANS integrator supplies transport and the existing standard wall functions.
Buoyant turbulence production and particle fluctuation production are omitted,
as in the existing standard-k-epsilon control; no exact Fluent match is claimed.
"""

import jax.numpy as jnp

from .rans_kepsilon import FLOOR, KEpsilonState
from .sgs import cell_gradients, edge_gradients


def invariants_from_gradient(gradient):
    """Return |S|, U* and As from a leading 3x3 Cartesian velocity gradient."""
    strain = 0.5 * (gradient + jnp.swapaxes(gradient, 0, 1))
    rotation = 0.5 * (gradient - jnp.swapaxes(gradient, 0, 1))
    s2 = jnp.sum(strain**2, axis=(0, 1))
    omega2 = jnp.sum(rotation**2, axis=(0, 1))
    cubic = sum(
        strain[i, j] * strain[j, k] * strain[k, i]
        for i in range(3)
        for j in range(3)
        for k in range(3)
    )
    denominator = jnp.maximum(s2**1.5, jnp.finfo(strain.dtype).tiny)
    w = jnp.where(s2 > 0, cubic / denominator, 0.0)
    # The analytic invariant bound can be exceeded by floating-point roundoff.
    phi = jnp.arccos(jnp.clip(jnp.sqrt(6.0) * w, -1.0, 1.0)) / 3.0
    return jnp.sqrt(2.0 * s2), jnp.sqrt(s2 + omega2), jnp.sqrt(6.0) * jnp.cos(phi)


def velocity_invariants(velocity, grid, boundaries, *, gradients=None):
    """Evaluate cell invariants using the caller's native edge gradients if supplied."""
    edge = edge_gradients(velocity, grid, boundaries) if gradients is None else gradients
    cell = cell_gradients(edge)
    return invariants_from_gradient(jnp.stack([jnp.stack(row) for row in cell]))


def viscosity_from_invariants(state, ustar, a_s):
    k, eps = state
    c_mu = 1.0 / (4.04 + a_s * ustar * k / jnp.maximum(eps, FLOOR))
    return c_mu * k**2 / jnp.maximum(eps, FLOOR)


def turbulent_viscosity(state, velocity, grid, boundaries, *, gradients=None):
    """Realizable viscosity consistent with the supplied boundary gradients."""
    _, ustar, a_s = velocity_invariants(
        velocity, grid, boundaries, gradients=gradients
    )
    return viscosity_from_invariants(state, ustar, a_s)


def local_sources(state, generation, strain_magnitude, molecular_viscosity, dt):
    """Positive first-order source update of the published realizable equations.

    dk/dt=P-eps; deps/dt=C1*S*eps-1.9*eps²/(k+sqrt(nu*eps)).
    Production is explicit, destruction uses a frozen-coefficient implicit sink.
    Realizable epsilon production is independent of P, unlike standard k-eps.
    """
    k, eps = state
    eta = strain_magnitude * k / jnp.maximum(eps, FLOOR)
    c1 = jnp.maximum(0.43, eta / (eta + 5.0))
    sink = 1.9 * eps / jnp.maximum(k + jnp.sqrt(molecular_viscosity * eps), FLOOR)
    return KEpsilonState(
        (k + dt * generation) / (1.0 + dt * eps / jnp.maximum(k, FLOOR)),
        (eps + dt * c1 * strain_magnitude * eps) / (1.0 + dt * sink),
    )
