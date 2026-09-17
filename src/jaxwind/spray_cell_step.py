"""Atomic single-cell droplet source and common-flux carrier transaction."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .numerics.discretization import divergence
from .spray_carrier import CarrierStep, build_carrier_step
from .spray_cell_exchange import build_cell_exchange
from .spray_core import WaterCoreBins
from .spray_low_mach import _specific_fluxes
from .spray_momentum import dual_average
from .state import StaggeredVelocity


class CellSprayStep(NamedTuple):
    carrier: CarrierStep
    liquid: WaterCoreBins
    unresolved_density: jax.Array
    unresolved_flux: StaggeredVelocity
    accepted: jax.Array
    source_accepted: jax.Array
    evaporated_mass: jax.Array
    drag_energy: jax.Array
    vapor_mixing_energy: jax.Array
    source_energy_error_joule: jax.Array
    source_relative_energy_error: jax.Array
    candidate_temperature: jax.Array


def build_cell_spray_step(
    poisson,
    cell,
    config,
    properties,
    ambient_specific,
    ambient_velocity,
    ambient_unresolved_specific,
    *,
    drag_heat_fraction,
    energy_tolerance=1e-10,
    **carrier_controls,
):
    """Advance one cell-owned bin group, committing all live inventories together.

    step(gas, velocity, unresolved_density, liquid, dt). The separately owned
    mechanical reservoir is transported with the final carrier mass flux. Its
    specific ambient value is J/kg, shape (nz,ny). No dissipation or physical SGS
    interpretation is assumed. Source exchange conserves energy; carrier pressure
    work and numerical kinetic loss remain diagnostic, not thermal closure.
    This is not an embedded-core entrainment/handoff or parcel-position driver.
    Overlapping cell groups must not independently source the same old gas.
    """
    grid = poisson.grid
    ambient_e = jnp.asarray(ambient_unresolved_specific)
    if ambient_e.shape != (grid.nz, grid.ny):
        raise ValueError("ambient unresolved energy must have shape (nz,ny)")
    exchange = build_cell_exchange(
        grid,
        cell,
        config,
        properties,
        periodic_x=poisson.periodic_x,
        periodic_y=poisson.periodic_y,
        drag_heat_fraction=drag_heat_fraction,
        energy_tolerance=energy_tolerance,
    )
    return _build_source_carrier_step(
        poisson,
        config,
        ambient_specific,
        ambient_velocity,
        ambient_e,
        exchange,
        **carrier_controls,
    )


def _build_source_carrier_step(
    poisson,
    config,
    ambient_specific,
    ambient_velocity,
    ambient_e,
    exchange,
    **carrier_controls,
):
    """Shared all-phase transaction for fixed or dynamically addressed sources."""
    grid = poisson.grid
    carrier_step = build_carrier_step(
        poisson, config, ambient_specific, ambient_velocity, **carrier_controls
    )
    periodic = (False, poisson.periodic_y, poisson.periodic_x)

    def step(gas, velocity, unresolved, liquid, dt, recipients=None):
        source = (
            exchange(gas, velocity, unresolved, liquid, dt)
            if recipients is None
            else exchange(gas, velocity, unresolved, liquid, dt, recipients)
        )
        carrier = carrier_step(
            gas, velocity, source.gas_increments, source.momentum_increment, dt
        )
        staged_rho = source.gas.dry_density + source.gas.vapor_density
        flux = _specific_fluxes(
            (source.unresolved_density / staged_rho)[None],
            carrier.mass_flux,
            ambient_e[None],
            poisson,
        )
        flux = jax.tree.map(lambda q: q[0], flux)
        energy = source.unresolved_density - dt * divergence(flux, grid)
        valid = (
            source.accepted
            & carrier.accepted
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(energy >= 0)
            & jnp.all(jnp.isfinite(ambient_e))
            & jnp.all(ambient_e >= 0)
        )
        choose = lambda a, b: jnp.where(valid, a, b)
        commit = lambda q: jax.tree.map(lambda a: choose(a, jnp.zeros_like(a)), q)
        old_rho = gas.dry_density + gas.vapor_density
        old_inertia = StaggeredVelocity(
            *(dual_average(old_rho, c, periodic[c]) for c in (2, 1, 0))
        )
        # Only the first three physical fields restore old values. Every other
        # committed ledger is zero on rejection; attempted diagnostics survive.
        carrier = CarrierStep(
            jax.tree.map(choose, carrier.gas, gas),
            jax.tree.map(choose, carrier.velocity, velocity),
            jax.tree.map(choose, carrier.inertia_density, old_inertia),
            *(commit(q) for q in carrier[3:15]),
            valid,
            *carrier[16:],
        )
        return CellSprayStep(
            carrier,
            jax.tree.map(choose, source.liquid, liquid),
            choose(energy, unresolved),
            commit(flux),
            valid,
            source.accepted,
            commit(source.evaporated_mass),
            commit(source.drag_energy),
            commit(source.vapor_mixing_energy),
            source.energy_error_joule,
            source.relative_energy_error,
            source.candidate_temperature,
        )

    return step
