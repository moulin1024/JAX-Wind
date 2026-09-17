"""First-order moving-parcel residence exchange with conservative x outflow."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .spray_body_force import kick_particles
from .spray_cell_exchange import CellExchange, all_finite, build_cell_exchange
from .spray_cell_step import CellSprayStep, _build_source_carrier_step
from .spray_low_mach import _admissible
from .spray_paths import build_parcel_path


class MovingSource(NamedTuple):
    source: CellExchange
    position: jax.Array
    exited_mass: jax.Array
    exited_momentum: jax.Array
    exited_energy: jax.Array
    path_accepted: jax.Array
    path_segments: jax.Array
    external_impulse: jax.Array
    external_work: jax.Array


class MovingSprayStep(NamedTuple):
    phase: CellSprayStep
    position: jax.Array
    exited_mass: jax.Array
    exited_momentum: jax.Array
    exited_energy: jax.Array
    path_accepted: jax.Array
    path_segments: jax.Array
    external_impulse: jax.Array
    external_work: jax.Array


def build_moving_source(
    grid,
    config,
    properties,
    *,
    periodic_x,
    periodic_y,
    drag_heat_fraction,
    particle_acceleration=(0.0, 0.0, 0.0),
    max_segments=64,
    energy_tolerance=1e-10,
):
    """Exchange each parcel over its visited-cell residence intervals.

    step(gas, velocity, unresolved, liquid, dt, position), position/velocity xyz
    arrays (3,P), mass/count/T (P,). Paths freeze the OLD liquid velocity over
    dt. Sources evolve velocity and thermodynamics along those paths; this is
    first-order splitting, requiring timestep convergence, not exact drag paths.
    Parcels process serially against updated gas. Constant particle_acceleration
    supplies external body force, e.g. (0,0,-9.81) for liquid gravity. It has no
    implicit buoyancy correction or gas reaction. Symmetric velocity kicks use
    each residence interval's pre/post-exchange liquid mass. External impulse
    and midpoint-velocity work are explicit; frozen paths still give first-order
    position accuracy and do not exactly conserve gravitational potential energy.
    No stochastic motion, injection, impaction or core entrainment is supplied.
    Open-x exits export remaining liquid mass, momentum and sensible+kinetic
    energy once, then zero its live multiplicity. Exit ledgers are per step.
    Optional residence_times supplies one positive duration <=dt per parcel,
    allowing births within the step; omitted durations are all dt. Any invalid
    path/source rejects every parcel and every committed ledger.
    """
    if len(particle_acceleration) != 3 or not all(
        math.isfinite(a) for a in particle_acceleration
    ):
        raise ValueError("particle acceleration must be a finite xyz vector")
    trace = build_parcel_path(
        grid, periodic_x=periodic_x, periodic_y=periodic_y, max_segments=max_segments
    )
    exchange = build_cell_exchange(
        grid,
        None,
        config,
        properties,
        periodic_x=periodic_x,
        periodic_y=periodic_y,
        drag_heat_fraction=drag_heat_fraction,
        energy_tolerance=energy_tolerance,
    )

    def step(gas, velocity, unresolved, liquid, dt, position, residence_times=None):
        count = liquid.mass.size
        if (
            liquid.mass.shape != (count,)
            or liquid.multiplicity.shape != (count,)
            or liquid.temperature.shape != (count,)
            or liquid.velocity.shape != (3, count)
            or position.shape != (3, count)
            or count < 1
        ):
            raise ValueError("invalid parcel shapes")
        ages = (
            jnp.full((count,), dt, liquid.mass.dtype)
            if residence_times is None
            else jnp.asarray(residence_times)
        )
        if ages.shape != (count,):
            raise ValueError("residence_times must have one entry per parcel")
        paths = jax.vmap(trace, in_axes=(1, 1, 0))(position, liquid.velocity, ages)
        active = liquid.multiplicity > 0
        valid = jnp.all(~active | paths.accepted)
        valid &= jnp.all(jnp.isfinite(ages)) & jnp.all((ages > 0) & (ages <= dt))
        valid &= jnp.all(jnp.isfinite(liquid.multiplicity)) & jnp.all(
            liquid.multiplicity >= 0
        )
        valid &= jnp.isfinite(dt) & (dt > 0)
        valid &= all_finite(liquid) & all_finite(velocity) & _admissible(gas, config)
        valid &= jnp.all(liquid.mass >= 0) & jnp.all(
            liquid.temperature >= config.freezing_temperature
        )
        valid &= jnp.all(jnp.isfinite(unresolved)) & jnp.all(unresolved >= 0)
        zero = jnp.asarray(0.0, gas.dry_density.dtype)
        initial = CellExchange(
            gas,
            velocity,
            unresolved,
            liquid,
            jax.tree.map(jnp.zeros_like, gas),
            jnp.zeros((3, *gas.dry_density.shape), gas.dry_density.dtype),
            jnp.zeros_like(unresolved),
            valid,
            zero,
            zero,
            zero,
            zero,
            zero,
            jnp.full((count,), jnp.nan, gas.dry_density.dtype),
        )

        initial = (
            initial,
            jnp.zeros_like(liquid.velocity),
            jnp.zeros_like(liquid.mass),
        )

        def segment(k, state):
            i, j = k // max_segments, k % max_segments
            duration = paths.durations[i, j]

            def apply(state):
                state, external_impulse, external_work = state
                bins = jax.tree.map(
                    lambda q: jnp.take(q, jnp.array([i]), axis=-1), state.liquid
                )
                bins, before_impulse, before_work = kick_particles(
                    bins, particle_acceleration, duration / 2
                )
                r = exchange(
                    state.gas,
                    state.velocity,
                    state.unresolved_density,
                    bins,
                    duration,
                    paths.cells[i, j],
                )
                kicked, after_impulse, after_work = kick_particles(
                    r.liquid, particle_acceleration, duration / 2
                )
                updated = jax.tree.map(
                    lambda a, b: a.at[..., i].set(b[..., 0]), state.liquid, kicked
                )
                external_impulse = external_impulse.at[:, i].add(
                    (before_impulse + after_impulse)[:, 0]
                )
                external_work = external_work.at[i].add((before_work + after_work)[0])
                result = CellExchange(
                    r.gas,
                    r.velocity,
                    r.unresolved_density,
                    updated,
                    jax.tree.map(
                        lambda a, b: a + b, state.gas_increments, r.gas_increments
                    ),
                    state.momentum_increment + r.momentum_increment,
                    state.unresolved_increment + r.unresolved_increment,
                    state.accepted & r.accepted,
                    state.evaporated_mass + r.evaporated_mass,
                    state.drag_energy + r.drag_energy,
                    state.vapor_mixing_energy + r.vapor_mixing_energy,
                    state.energy_error_joule + r.energy_error_joule,
                    jnp.maximum(state.relative_energy_error, r.relative_energy_error),
                    state.candidate_temperature.at[i].set(r.candidate_temperature),
                )

                return result, external_impulse, external_work

            return jax.lax.cond(
                state[0].accepted & active[i] & (duration > 0),
                apply,
                lambda s: s,
                state,
            )

        result, external_impulse, external_work = jax.lax.fori_loop(
            0, count * max_segments, segment, initial
        )
        exit_mask = active & paths.exited
        masses = result.liquid.mass * result.liquid.multiplicity * exit_mask
        momentum = result.liquid.velocity * masses
        energy = masses * (
            properties.liquid_heat_capacity
            * (result.liquid.temperature - config.freezing_temperature)
            + 0.5 * jnp.sum(result.liquid.velocity**2, axis=0)
        )
        # Avoid 0*NaN in unused slots; live exported properties must be finite.
        masses = jnp.where(exit_mask, masses, 0.0)
        momentum = jnp.where(exit_mask[None], momentum, 0.0)
        energy = jnp.where(exit_mask, energy, 0.0)
        ok = (
            result.accepted
            & jnp.all(jnp.isfinite(masses))
            & jnp.all(jnp.isfinite(momentum))
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(jnp.isfinite(external_impulse))
            & jnp.all(jnp.isfinite(external_work))
        )
        updated = result.liquid._replace(
            multiplicity=jnp.where(exit_mask, 0.0, result.liquid.multiplicity)
        )
        choose = lambda a, b: jnp.where(ok, a, b)
        commit = lambda q: jax.tree.map(lambda a: choose(a, jnp.zeros_like(a)), q)
        result = CellExchange(
            jax.tree.map(choose, result.gas, gas),
            jax.tree.map(choose, result.velocity, velocity),
            choose(result.unresolved_density, unresolved),
            jax.tree.map(choose, updated, liquid),
            *(commit(q) for q in result[4:7]),
            ok,
            *(commit(q) for q in result[8:11]),
            *result[11:],
        )
        end = jnp.where(active[None], paths.position.T, position)
        return MovingSource(
            result,
            choose(end, position),
            commit(masses),
            commit(momentum),
            commit(energy),
            paths.accepted,
            paths.segments,
            commit(external_impulse),
            commit(external_work),
        )

    return step


def build_moving_spray_step(
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
    """Commit positions, outflow and all phases with one carrier transaction."""
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

    def step(gas, velocity, unresolved, liquid, dt, position, residence_times=None):
        moving = source_step(
            gas, velocity, unresolved, liquid, dt, position, residence_times
        )
        phase = carrier_step(gas, velocity, unresolved, liquid, dt, moving.source)
        commit = lambda q: jnp.where(phase.accepted, q, jnp.zeros_like(q))
        return MovingSprayStep(
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

    return step
