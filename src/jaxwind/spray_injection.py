"""Prescribed parcel births with exact boundary budgets and atomic transport."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .spray_cell_exchange import all_finite
from .spray_cell_step import _build_source_carrier_step
from .spray_core import WaterCoreBins
from .spray_moving import MovingSprayStep, build_moving_source


class ParcelBirths(NamedTuple):
    liquid: WaterCoreBins
    position: jax.Array
    time_offset: jax.Array
    valid: jax.Array = True


class InjectionStage(NamedTuple):
    liquid: WaterCoreBins
    position: jax.Array
    residence_times: jax.Array
    injected_mass: jax.Array
    injected_momentum: jax.Array
    injected_enthalpy: jax.Array
    injected_kinetic_energy: jax.Array
    accepted: jax.Array
    requested_slots: jax.Array
    available_slots: jax.Array
    slots: jax.Array


class InjectedSprayStep(NamedTuple):
    moving: MovingSprayStep
    injected_mass: jax.Array
    injected_momentum: jax.Array
    injected_enthalpy: jax.Array
    injected_kinetic_energy: jax.Array
    injection_accepted: jax.Array
    requested_slots: jax.Array
    available_slots: jax.Array
    slots: jax.Array


def births_from_mass_flow(
    mass_flow,
    dt,
    diameter,
    mass_fraction,
    velocity,
    temperature,
    position,
    birth_fraction,
    config,
):
    """Prescribe a constant mass-flow quadrature over one step, without fitting.

    All distributions are caller supplied. Fractions are MASS fractions and
    must sum to one; they are never silently renormalized. Birth fractions in
    [0,1) specify temporal quadrature nodes. Zero flow makes all counts zero,
    consuming no slots. Invalid dynamic inputs carry valid=False into staging.
    This supplies no stochastic sampler or measured-inlet qualification.
    """
    diameter, weights = jnp.asarray(diameter), jnp.asarray(mass_fraction)
    n = diameter.size
    if n < 1 or diameter.shape != (n,) or weights.shape != (n,):
        raise ValueError("diameter and mass fraction must be nonempty vectors")
    velocity, temperature, position, fraction = map(
        jnp.asarray, (velocity, temperature, position, birth_fraction)
    )
    if (
        velocity.shape != (3, n)
        or temperature.shape != (n,)
        or position.shape != (3, n)
        or fraction.shape != (n,)
    ):
        raise ValueError("invalid prescribed inlet shapes")
    valid = all_finite(
        (
            diameter,
            weights,
            velocity,
            temperature,
            position,
            fraction,
            jnp.asarray(mass_flow),
            jnp.asarray(dt),
        )
    )
    valid &= (mass_flow >= 0) & (dt > 0) & jnp.all(diameter > 0) & jnp.all(weights >= 0)
    valid &= jnp.abs(jnp.sum(weights) - 1) <= 1e-12
    valid &= jnp.all((fraction >= 0) & (fraction < 1))
    mass = config.water_density * jnp.pi * diameter**3 / 6
    liquid = WaterCoreBins(mass, mass_flow * dt * weights / mass, velocity, temperature)
    return ParcelBirths(liquid, position, dt * fraction, valid)


def stage_parcel_births(liquid, position, births, dt, config, properties):
    """Reserve free slots and stage a prescribed batch; never drop partial flow.

    Birth offsets use [0,dt), so every newborn has positive residence time.
    Requested multiplicity is supplied by the source model; zero-count entries
    are no-ops. A slot is reusable when mass or multiplicity is zero. Capacity
    is reserved at the step start, without anticipating later exits. The caller
    must commit this stage with the gas/particle step or restore original state.
    All returned injection ledgers are per-step extensive quantities.
    """
    capacity = liquid.mass.size
    count = births.liquid.mass.size
    for q, n in ((liquid, capacity), (births.liquid, count)):
        if (
            q.mass.shape != (n,)
            or q.multiplicity.shape != (n,)
            or q.temperature.shape != (n,)
            or q.velocity.shape != (3, n)
        ):
            raise ValueError("invalid liquid shapes")
    if (
        capacity < 1
        or count < 1
        or position.shape != (3, capacity)
        or births.position.shape != (3, count)
        or births.time_offset.shape != (count,)
    ):
        raise ValueError("invalid parcel birth shapes")
    q = births.liquid
    active = q.multiplicity > 0
    free = (liquid.multiplicity == 0) | (liquid.mass == 0)
    requested, available = jnp.sum(active), jnp.sum(free)
    valid = (
        births.valid
        & all_finite((liquid, position, births))
        & jnp.isfinite(dt)
        & (dt > 0)
    )
    for bins in (liquid, q):
        valid &= jnp.all(bins.mass >= 0) & jnp.all(bins.multiplicity >= 0)
        valid &= jnp.all(bins.temperature >= config.freezing_temperature)
    valid &= jnp.all(~active | (q.mass > 0))
    valid &= jnp.all((births.time_offset >= 0) & (births.time_offset < dt))
    valid &= requested <= available
    free_slots = jnp.nonzero(free, size=capacity, fill_value=0)[0]
    ordinals = jnp.cumsum(active.astype(jnp.int32)) - 1
    destinations = free_slots[jnp.clip(ordinals, 0, capacity - 1)]
    ages = jnp.full((capacity,), dt, liquid.mass.dtype)

    def insert(i, state):
        def apply(state):
            bins, pos, times = state
            slot = destinations[i]
            bins = jax.tree.map(lambda a, b: a.at[..., slot].set(b[..., i]), bins, q)
            return (
                bins,
                pos.at[:, slot].set(births.position[:, i]),
                times.at[slot].set(dt - births.time_offset[i]),
            )

        return jax.lax.cond(valid & active[i], apply, lambda s: s, state)

    updated, pos, ages = jax.lax.fori_loop(0, count, insert, (liquid, position, ages))
    mass = q.mass * q.multiplicity
    choose = lambda a, b: jnp.where(valid, a, b)
    commit = lambda a: choose(a, jnp.zeros_like(a))
    return InjectionStage(
        jax.tree.map(choose, updated, liquid),
        choose(pos, position),
        choose(ages, jnp.full_like(ages, dt)),
        commit(jnp.sum(mass)),
        commit(jnp.sum(q.velocity * mass, axis=1)),
        commit(
            jnp.sum(
                mass
                * properties.liquid_heat_capacity
                * (q.temperature - config.freezing_temperature)
            )
        ),
        commit(0.5 * jnp.sum(mass * jnp.sum(q.velocity**2, axis=0))),
        valid,
        requested,
        available,
        jnp.where(valid & active, destinations, -1),
    )


def build_injected_spray_step(
    poisson,
    config,
    properties,
    ambient_specific,
    ambient_velocity,
    ambient_unresolved_specific,
    *,
    drag_heat_fraction,
    particle_acceleration=(0.0, 0.0, 0.0),
    max_segments=64,
    energy_tolerance=1e-10,
    **carrier_controls,
):
    """Commit prescribed births, movement, outflow and carrier as one step.

    step(gas, velocity, unresolved, liquid, dt, position, births). No random state
    is consumed here. A caller generating stochastic batches must retain its
    original RNG state until this transaction accepts. Inlet sampling, mass-flow
    quadrature, nozzle gas/entrainment, breakup and injection geometry are caller
    inputs, not fitted by this transport stage. Birth ages are exact, while gas
    source ordering and frozen particle paths retain first-order splitting.
    """
    ambient_e = jnp.asarray(ambient_unresolved_specific)
    if ambient_e.shape != (poisson.grid.nz, poisson.grid.ny):
        raise ValueError("invalid ambient unresolved shape")
    source_step = build_moving_source(
        poisson.grid,
        config,
        properties,
        periodic_x=poisson.periodic_x,
        periodic_y=poisson.periodic_y,
        drag_heat_fraction=drag_heat_fraction,
        particle_acceleration=particle_acceleration,
        max_segments=max_segments,
        energy_tolerance=energy_tolerance,
    )
    carrier_step = _build_source_carrier_step(
        poisson,
        config,
        ambient_specific,
        ambient_velocity,
        ambient_e,
        lambda gas, velocity, unresolved, liquid, dt, source: source,
        **carrier_controls,
    )

    def step(gas, velocity, unresolved, liquid, dt, position, births):
        injection = stage_parcel_births(
            liquid, position, births, dt, config, properties
        )
        moving = source_step(
            gas,
            velocity,
            unresolved,
            injection.liquid,
            dt,
            injection.position,
            injection.residence_times,
        )
        source = moving.source._replace(
            accepted=injection.accepted & moving.source.accepted
        )
        # Carrier rollback uses PRE-INJECTION liquid as its original inventory.
        phase = carrier_step(gas, velocity, unresolved, liquid, dt, source)
        commit = lambda q: jnp.where(phase.accepted, q, jnp.zeros_like(q))
        result = MovingSprayStep(
            phase,
            jnp.where(phase.accepted, moving.position, position),
            commit(moving.exited_mass),
            commit(moving.exited_momentum),
            commit(moving.exited_energy),
            moving.path_accepted,
            moving.path_segments,
            commit(moving.external_impulse),
            commit(moving.external_work),
        )
        return InjectedSprayStep(
            result,
            commit(injection.injected_mass),
            commit(injection.injected_momentum),
            commit(injection.injected_enthalpy),
            commit(injection.injected_kinetic_energy),
            injection.accepted,
            injection.requested_slots,
            injection.available_slots,
            jnp.where(phase.accepted, injection.slots, -1),
        )

    return step
