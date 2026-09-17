"""Compatible primary-cell energy terms for the uniform MAC carrier.

The thermal convention is E_internal = H_dilute - constant p0. Retaining
perturbation-pressure conversion and returning numerical mixing loss to H
closes the discretized H+K budget (up to exposed nonlinear iteration work).
This does not identify numerical dissipation with physical SGS turbulence.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .numerics.discretization import divergence
from .spray_momentum import _slice
from .spray_pressure import momentum_pressure_gradient
from .state import StaggeredVelocity


class CarrierEnergyTerms(NamedTuple):
    kinetic_flux: StaggeredVelocity
    pressure_flux: StaggeredVelocity
    numerical_heat: jax.Array
    pressure_conversion: jax.Array
    wall_export: jax.Array
    pressure_identity_residual: jax.Array


def face_density_to_cell(values, component, periodic):
    """Partition dual extensive energy equally onto its adjacent half cells."""
    if periodic:
        return 0.5 * (values + jnp.roll(values, -1, component))
    return 0.5 * (_slice(values, component, 0, -1) + _slice(values, component, 1, None))


def component_sum(values, periodic):
    return sum(
        face_density_to_cell(q, c, periodic[c]) for c, q in zip((2, 1, 0), values)
    )


def kinetic_flux_to_primary(fluxes, periodic):
    """Flux mapping commuting with dual divergence, including half end cells."""
    result = []
    for axis in (2, 1, 0):
        terms = []
        for component, flux in zip((2, 1, 0), fluxes):
            q = flux[(2, 1, 0).index(axis)]
            if axis == component and not periodic[component]:
                terms.append(
                    jnp.concatenate(
                        (
                            _slice(q, axis, 0, 1),
                            0.5 * (_slice(q, axis, 1, -2) + _slice(q, axis, 2, -1)),
                            _slice(q, axis, -1, None),
                        ),
                        axis=axis,
                    )
                )
            else:
                terms.append(face_density_to_cell(q, component, periodic[component]))
        result.append(sum(terms))
    return StaggeredVelocity(*result)


def _pressure_faces(pressure, periodic, open_x_low):
    parts = []
    for axis in (2, 1, 0):
        if periodic[axis]:
            q = 0.5 * (pressure + jnp.roll(pressure, 1, axis))
        else:
            low = _slice(pressure, axis, 0, 1)
            high = _slice(pressure, axis, -1, None)
            if axis == 2:
                low = jnp.zeros_like(low) if open_x_low else low
                high = jnp.zeros_like(high)
            middle = 0.5 * (
                _slice(pressure, axis, 0, -1) + _slice(pressure, axis, 1, None)
            )
            q = jnp.concatenate((low, middle, high), axis=axis)
        parts.append(q)
    return StaggeredVelocity(*parts)


def compatible_energy_terms(
    pressure,
    advective_momentum,
    inertia,
    wall_impulse,
    kinetic_loss,
    kinetic_fluxes,
    final_velocity,
    dt,
    grid,
    *,
    periodic_x,
    periodic_y,
    open_x_low=True,
):
    """Derive compatible energy transfers from the FINAL-flux momentum step.

    Wall removal is exported, not heated into the gas. Numerical mixing heat
    is the remainder of kinetic_loss after wall removal; no clipping is used.
    Pressure conversion is -dt*pi*div(u_mid). Its midpoint includes pressure
    projection's temporal kinetic change; it is not solely continuum physical
    pressure dilatation. Nonzero pressure_identity_residual beyond roundoff indicates failure
    of the discrete product identity; its sign has no physical interpretation.
    """
    periodic = (False, periodic_y, periodic_x)
    midpoint = StaggeredVelocity(
        *(
            0.5 * (p / r + u)
            for p, r, u in zip(advective_momentum, inertia, final_velocity)
        )
    )
    wall = StaggeredVelocity(
        *(
            ((p - j) ** 2 - p**2) / (2 * r)
            for p, j, r in zip(advective_momentum, wall_impulse, inertia)
        )
    )
    numerical = StaggeredVelocity(*(loss - w for loss, w in zip(kinetic_loss, wall)))
    gradient = momentum_pressure_gradient(
        pressure,
        grid,
        periodic_x=periodic_x,
        periodic_y=periodic_y,
        open_x_low=open_x_low,
    )
    work = component_sum(
        StaggeredVelocity(*(-dt * g * u for g, u in zip(gradient, midpoint))), periodic
    )
    pressure_flux = StaggeredVelocity(
        *(
            p * u
            for p, u in zip(_pressure_faces(pressure, periodic, open_x_low), midpoint)
        )
    )
    conversion = -dt * pressure * divergence(midpoint, grid)
    return CarrierEnergyTerms(
        kinetic_flux_to_primary(kinetic_fluxes, periodic),
        pressure_flux,
        component_sum(numerical, periodic),
        conversion,
        component_sum(wall, periodic),
        work + conversion + dt * divergence(pressure_flux, grid),
    )


def carrier_energy_terms(result, dt, poisson):
    """Recover the same terms from a committed CarrierStep without new solves.

    They were applied to enthalpy only if its builder enabled energy_coupling.
    Rejected carrier steps yield zero transfers/fluxes, while retained gas and
    velocity are not mistaken for committed changes.
    """
    momentum = StaggeredVelocity(
        *(
            r * u - j - e
            for r, u, j, e in zip(
                result.inertia_density,
                result.velocity,
                result.pressure_impulse,
                result.momentum_residual,
            )
        )
    )
    return compatible_energy_terms(
        result.pressure,
        momentum,
        result.inertia_density,
        result.wall_impulse,
        result.numerical_kinetic_loss,
        result.kinetic_fluxes,
        result.velocity,
        dt,
        poisson.grid,
        periodic_x=poisson.periodic_x,
        periodic_y=poisson.periodic_y,
        open_x_low=poisson.open_x_low,
    )
