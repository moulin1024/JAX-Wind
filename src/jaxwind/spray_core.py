"""Finite-gas water-bin exchange for the unresolved spray core.

Local, warm, adiabatic exchange only: no advection, gravity, pressure work,
entrainment or resolved LES deposition. Gas and liquid are disjoint inventories.
Uses the existing spherical drag and finite-temperature evaporation laws.
The caller must prescribe the fraction of lost mean kinetic energy converted
immediately to gas heat; the rest remains explicitly owned unresolved energy.
This fraction is a physical closure input, not a calibrated production default.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .physics.moisture import advance_water_droplet
from .spray_coupling import GasInventory, merge_gas_inventories
from .water_spray import water_droplet_drag_rate


class WaterCoreBins(NamedTuple):
    """One-dimensional bin arrays; mass is kg/drop, velocity has shape (3,N)."""

    mass: jax.Array
    multiplicity: jax.Array
    velocity: jax.Array
    temperature: jax.Array


class CoreExchange(NamedTuple):
    gas: GasInventory
    liquid: WaterCoreBins
    accepted: jax.Array
    evaporated_mass: jax.Array
    drag_energy: jax.Array
    vapor_mixing_energy: jax.Array
    candidate_gas_temperature: jax.Array


def core_gas_temperature(gas, config):
    """Two species [dry air, vapor], h_d=cp_d*(T-Tf), h_v=Lv.

    This is the existing dilute moisture enthalpy approximation, not a complete
    mixture EOS. Requires strictly positive dry-air mass. No vapor cp is added.
    """
    return config.freezing_temperature + (
        gas.enthalpy - config.water_vapor_latent_heat * gas.species[1]
    ) / (config.dry_air_heat_capacity * gas.species[0])


def relax_core_drag(gas, liquid, rates, dt):
    """Symmetric exact gas/bin pair relaxations for frozen nonnegative rates.

    Each pair solves dv/dt=k*(u-v), du/dt=(M/mg)*k*(v-u) exactly.
    Forward/reverse half sweeps are second order for fixed rates, without a
    stiff explicit gas reaction. Work is O(N), storage O(N). The loss of mean
    kinetic energy is returned separately and never counted twice. Empty bins
    cannot transfer momentum. No thermodynamic quantities change in this step.
    """
    masses = liquid.mass * liquid.multiplicity
    count = masses.shape[0]
    initial_u = gas.momentum / gas.mass

    def pair(index, state):
        u, velocities, loss = state
        mass = masses[index]
        fraction = mass / (gas.mass + mass)
        slip = velocities[:, index] - u
        z = rates[index] * (1 + mass / gas.mass) * (0.5 * dt)
        response = -jnp.expm1(-z)
        change = response * slip
        new_u = u + fraction * change
        new_v = velocities[:, index] - (1 - fraction) * change
        # Empty bins are inactive, including their stored velocity.
        new_v = jnp.where(mass > 0, new_v, velocities[:, index])
        reduced_mass = gas.mass * fraction
        dissipated = 0.5 * reduced_mass * jnp.sum(slip**2) * (-jnp.expm1(-2 * z))
        return new_u, velocities.at[:, index].set(new_v), loss + dissipated

    state = (initial_u, liquid.velocity, jnp.zeros_like(gas.mass))
    state = jax.lax.fori_loop(0, count, pair, state)
    state = jax.lax.fori_loop(0, count, lambda i, s: pair(count - 1 - i, s), state)
    u, velocity, loss = state
    return gas._replace(momentum=gas.mass * u), liquid._replace(velocity=velocity), loss


def advance_water_core(gas, liquid, dt, config, properties, *, drag_heat_fraction):
    """Transactional mechanical/thermal exchange with one owned gas inventory.

    Preconditions: finite warm inputs, gas.mass>0, species=[dry air,vapor]
    nonnegative summing to mass with dry air>0; nonnegative bin masses/counts;
    dt>=0, 0<=drag_heat_fraction<=1. No external force is included.

    Frozen nonlinear drag is followed by the shared frozen-reservoir thermal
    update, so this combined method is first order and needs subcycling. Vapor
    leaves each drop at its post-drag velocity, transfers its *full* momentum,
    and mixes into gas once. Heat and vapor formation enthalpy obey the shared
    droplet convention. Drag/mixing kinetic-energy loss is split explicitly
    between gas enthalpy and its unresolved reservoir using the supplied fraction.

    If a candidate depletes the finite carrier heat inventory below freezing or
    becomes nonfinite, return both ORIGINAL states and accepted=False. The caller
    must reduce dt/recompute, never accept a clipped temperature or partial state.
    Candidate diagnostics describe the rejected attempt; no ledger is committed.
    Supersaturation is not repaired here: fog/condensation and pressure/EOS
    integration are still required for the full core/LES model.
    """
    diameter = jnp.cbrt(6 * liquid.mass / (jnp.pi * config.water_density))
    slip = jnp.linalg.norm(liquid.velocity - (gas.momentum / gas.mass)[:, None], axis=0)
    rates = water_droplet_drag_rate(diameter, slip, config, properties)
    mechanical_gas, mechanical_liquid, drag_energy = relax_core_drag(
        gas, liquid, rates, dt
    )
    mechanical_gas = mechanical_gas._replace(
        enthalpy=mechanical_gas.enthalpy + drag_heat_fraction * drag_energy,
        unresolved_energy=mechanical_gas.unresolved_energy
        + (1 - drag_heat_fraction) * drag_energy,
    )
    update = advance_water_droplet(
        liquid.mass,
        liquid.temperature,
        core_gas_temperature(mechanical_gas, config),
        mechanical_gas.species[1] / mechanical_gas.species[0],
        jnp.linalg.norm(
            mechanical_liquid.velocity
            - (mechanical_gas.momentum / mechanical_gas.mass)[:, None],
            axis=0,
        ),
        dt * (liquid.multiplicity > 0),
        config,
        properties,
    )
    evaporated = update.evaporated_mass * liquid.multiplicity
    dm = jnp.sum(evaporated)
    vapor_momentum = jnp.sum(evaporated * mechanical_liquid.velocity, axis=1)
    vapor_velocity = vapor_momentum / jnp.where(dm > 0, dm, 1.0)
    # Variance between bin vapor velocities belongs to the vapor inventory.
    vapor_variance = 0.5 * jnp.sum(
        evaporated
        * jnp.sum((mechanical_liquid.velocity - vapor_velocity[:, None]) ** 2, axis=0)
    )
    vapor = GasInventory(
        dm,
        vapor_momentum,
        config.water_vapor_latent_heat * dm,
        jnp.stack((jnp.zeros_like(dm), dm)),
        vapor_variance,
    )
    candidate_gas, gas_mixing = merge_gas_inventories(mechanical_gas, vapor)
    mixing = vapor_variance + gas_mixing
    candidate_gas = candidate_gas._replace(
        enthalpy=candidate_gas.enthalpy
        - jnp.sum(update.gas_sensible_energy_loss * liquid.multiplicity)
        + drag_heat_fraction * mixing,
        unresolved_energy=mechanical_gas.unresolved_energy
        + (1 - drag_heat_fraction) * mixing,
    )
    candidate_liquid = mechanical_liquid._replace(
        mass=update.mass, temperature=update.temperature
    )
    temperature = core_gas_temperature(candidate_gas, config)
    finite = jnp.all(
        jnp.stack(
            [
                jnp.all(jnp.isfinite(field))
                for field in (*candidate_gas, *candidate_liquid)
            ]
        )
    )
    accepted = finite & (temperature >= config.freezing_temperature)
    choose = lambda new, old: jnp.where(accepted, new, old)
    return CoreExchange(
        jax.tree.map(choose, candidate_gas, gas),
        jax.tree.map(choose, candidate_liquid, liquid),
        accepted,
        dm,
        drag_energy,
        mixing,
        temperature,
    )
