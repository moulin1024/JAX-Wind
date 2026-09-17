"""Canonical spray-closure components; not coupled to the production spray solver.

The round jet is steady, constant density, unconfined and nonbuoyant, in still
ambient gas. Its entrainment coefficient is in the *equivalent top-hat*
convention (Ricou--Spalding alpha=0.08), not the Gaussian convention.
The stochastic component implements Pozorski--Apte (2009), equations 4.1--4.6,
for homogeneous SGS turbulence only. It is not an inhomogeneous LES closure.
See cases/SprayClosureValidation/README.md for evidence and applicability.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import erf


class RoundJetFlux(NamedTuple):
    """Integrated section fluxes, in kg/s, N, W, and kg/s per species."""

    mass: jax.Array
    momentum: jax.Array
    enthalpy: jax.Array
    species: jax.Array


def advance_round_jet(
    flux: RoundJetFlux,
    distance,
    density,
    ambient_enthalpy,
    ambient_species,
    *,
    alpha=0.08,
) -> RoundJetFlux:
    """Exact constant-coefficient integral step with stationary ambient intake.

    Preconditions: mass, momentum, density > 0; distance, alpha >= 0. Species
    are nonnegative mass fractions summing to one. All inputs must be finite.
    Enthalpy is a passive specific-enthalpy tracer, not an evaporation closure.
    Momentum is the first-order mean-flow approximation, omitting Reynolds
    stresses and pressure corrections. Ambient withdrawals must be debited if
    this component is ever coupled to a finite-volume gas inventory.
    """
    entrained = 2 * alpha * jnp.sqrt(jnp.pi * density * flux.momentum) * distance
    return RoundJetFlux(
        flux.mass + entrained,
        flux.momentum,
        flux.enthalpy + entrained * ambient_enthalpy,
        flux.species + entrained * jnp.asarray(ambient_species),
    )


def round_jet_gaussian(flux: RoundJetFlux, density):
    """Return centreline velocity and Gaussian e-folding radius in m/s and m.

    U(r)=Uc exp(-r^2/b^2); half-width=sqrt(log(2))*b. This reconstruction
    preserves both integral mass and mean-flow momentum, not just centre speed.
    """
    return (
        2 * flux.momentum / flux.mass,
        flux.mass / jnp.sqrt(2 * jnp.pi * density * flux.momentum),
    )


def round_jet_plane_fluxes(
    flux: RoundJetFlux, density, y_edges, z_edges, *, center=(0.0, 0.0)
) -> RoundJetFlux:
    """Analytic Gaussian integrals per rectangular face, shape (nz, ny).

    Edges must be strictly increasing. Mass/enthalpy/species use exp(-r^2/b^2),
    momentum uses exp(-2r^2/b^2). Species output shape is (nspecies, nz, ny).
    Fractions outside the supplied plane are *not* redistributed into it.
    These are separate conservative fluxes, not cell-centre velocity samples
    or a completed LES handoff operator. Scalar radial mixing is not modeled.
    """
    _, radius = round_jet_gaussian(flux, density)

    def fractions(scale):
        fy = jnp.diff(erf((jnp.asarray(y_edges) - center[0]) / scale)) / 2
        fz = jnp.diff(erf((jnp.asarray(z_edges) - center[1]) / scale)) / 2
        return fz[:, None] * fy[None, :]

    mass_fraction = fractions(radius)
    momentum_fraction = fractions(radius / jnp.sqrt(2.0))
    return RoundJetFlux(
        flux.mass * mass_fraction,
        flux.momentum * momentum_fraction,
        flux.enthalpy * mass_fraction,
        jnp.asarray(flux.species)[:, None, None] * mass_fraction,
    )


def homogeneous_sgs_velocity_step(
    residual,
    resolved_velocity,
    particle_velocity,
    sgs_energy,
    filter_width,
    dt,
    normal_sample,
    *,
    time_constant=1.0,
    crossing_ratio=1.0,
):
    """Frozen-coefficient exact OU update, including Csanady crossing times.

    Vector arguments have shape (3, n); scalar fields broadcast to (n,).
    normal_sample is independent N(0,1) per component and parcel per update;
    the caller owns the PRNG state. k_sgs [m^2/s^2] >= 0, Delta [m] > 0,
    dt [s] >= 0, C > 0 and beta >= 0. Variance per direction is 2*k_sgs/3.
    k_sgs is an explicit physical input, NOT inferred from AMD viscosity.

    This is a homogeneous one-particle velocity update only. It has no
    inhomogeneous drift, spatial noise correlation, drag integration, two-way
    SGS energy budget, evaporation or LES feedback. It cannot establish spray
    concentration/entrainment accuracy by itself. At k=0 the residual is zero
    for a positive step. dt=0 preserves the state exactly.
    """
    variance = 2 * jnp.asarray(sgs_energy) / 3
    slip = jnp.asarray(resolved_velocity) - jnp.asarray(particle_velocity)
    slip2 = jnp.sum(slip * slip, axis=0)
    # Avoid choosing a special coordinate direction when the relative speed is 0.
    direction = slip / jnp.sqrt(jnp.where(slip2 > 0, slip2, 1.0))
    parallel = direction * jnp.sum(direction * residual, axis=0)
    noise_parallel = direction * jnp.sum(direction * normal_sample, axis=0)
    rate_parallel = jnp.sqrt(variance + crossing_ratio**2 * slip2) / (
        time_constant * filter_width
    )
    rate_perpendicular = jnp.sqrt(variance + 4 * crossing_ratio**2 * slip2) / (
        time_constant * filter_width
    )

    def coefficients(rate):
        return jnp.exp(-dt * rate), jnp.sqrt(variance * -jnp.expm1(-2 * dt * rate))

    ap, bp = coefficients(rate_parallel)
    at, bt = coefficients(rate_perpendicular)
    updated = (
        at * residual
        + (ap - at) * parallel
        + bt * normal_sample
        + (bp - bt) * noise_parallel
    )
    updated = jnp.where(variance > 0, updated, 0.0)
    return jnp.where(jnp.asarray(dt) == 0, residual, updated)
