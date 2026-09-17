"""Transient spatial DPM parcels with residence-based conservative sources.

Uniform MAC grid, open x, periodic y or escape sides, ground trap/top escape.
Motion uses frozen-velocity residence paths with split forces: first order in
tracking timestep, requiring refinement. It does not claim Fluent's trajectory
interpolator. Every phase source is assigned to its traversed cell. Persistent
DRW eddies are renewed at interaction events; source work has an explicit ledger.
"""

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .fluent_dpm_source import build_dpm_cell_exchange, gas_temperature, material_config
from .physics.fluent_dpm import gas_properties, liquid_enthalpy, sample_drw_eddy
from .spray_core import WaterCoreBins
from .spray_paths import build_parcel_path
from .water_spray import water_droplet_drag_rate


class DPMParcels(NamedTuple):
    position: object
    velocity: object
    mass: object
    temperature: object
    multiplicity: object
    fluctuation: object
    eddy_remaining: object
    keys: object
    draws: object
    next_id: object


class DPMLedger(NamedTuple):
    # Each inventory vector is [mass, px, py, pz, liquid enthalpy, kinetic energy].
    injected: object
    escaped: object
    trapped: object
    gravity_impulse: object
    gravity_work: object
    stochastic_work: object
    wall_impulse: object
    wall_energy: object
    evaporated_mass: object
    maximum_source_energy_error: object


class DPMSpatialStep(NamedTuple):
    gas: object
    velocity: object
    parcels: DPMParcels
    ledger: DPMLedger
    accepted: object


def initial_parcels(options, dtype):
    n = options.capacity
    zero = jnp.zeros(n, dtype)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(options.seed), i))(
        jnp.arange(n)
    )
    return DPMParcels(
        jnp.zeros((3, n), dtype),
        jnp.zeros((3, n), dtype),
        zero,
        jnp.full(n, options.injection_temperature_k, dtype),
        zero,
        jnp.zeros((3, n), dtype),
        zero,
        keys,
        jnp.zeros(n, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )


def initial_ledger(dtype):
    z = jnp.asarray(0.0, dtype)
    return DPMLedger(
        jnp.zeros(6, dtype),
        jnp.zeros(6, dtype),
        jnp.zeros(6, dtype),
        jnp.zeros(3, dtype),
        z,
        z,
        jnp.zeros(3, dtype),
        z,
        z,
        z,
    )


def liquid_inventory(mass, number, velocity, temperature, material):
    physical = mass * number
    return jnp.concatenate(
        (
            jnp.atleast_1d(physical),
            physical * velocity,
            jnp.atleast_1d(physical * liquid_enthalpy(temperature, material)),
            jnp.atleast_1d(0.5 * physical * jnp.sum(velocity**2)),
        )
    )


def inject_parcels(parcels, ledger, source, center, time, dt, material):
    options = source.dpm
    count = len(options.diameters_m)
    free = (parcels.mass * parcels.multiplicity) == 0
    slots = jnp.argsort(~free, stable=True)[:count]
    ok = jnp.sum(free) >= count
    phase = jnp.clip((time + dt / 2) / max(source.ramp_time_s, 1e-30), 0, 1)
    ramp = 1.0 if source.ramp_time_s == 0 else 0.5 * (1 - jnp.cos(jnp.pi * phase))
    amount = (
        source.mass_flow_rate_kg_s * dt * ramp * jnp.asarray(options.mass_fractions)
    )
    masses = (
        jnp.pi / 6 * material.liquid_density * jnp.asarray(options.diameters_m) ** 3
    )
    number = amount / masses
    dtype = parcels.mass.dtype
    positions = jnp.broadcast_to(jnp.asarray(center, dtype)[:, None], (3, count))
    velocities = jnp.broadcast_to(
        jnp.asarray(options.injection_velocity_m_s, dtype)[:, None], (3, count)
    )
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(options.seed), i))(
        parcels.next_id + jnp.arange(count, dtype=jnp.int32)
    )
    new = parcels._replace(
        position=parcels.position.at[:, slots].set(positions),
        velocity=parcels.velocity.at[:, slots].set(velocities),
        mass=parcels.mass.at[slots].set(masses),
        temperature=parcels.temperature.at[slots].set(options.injection_temperature_k),
        multiplicity=parcels.multiplicity.at[slots].set(number),
        fluctuation=parcels.fluctuation.at[:, slots].set(0),
        eddy_remaining=parcels.eddy_remaining.at[slots].set(0),
        keys=parcels.keys.at[slots].set(keys),
        draws=parcels.draws.at[slots].set(0),
        next_id=parcels.next_id + count,
    )
    inventories = jax.vmap(
        lambda m, n, v: liquid_inventory(
            m, n, v, options.injection_temperature_k, material
        ),
        in_axes=(0, 0, 1),
    )(masses, number, velocities)
    updated = ledger._replace(injected=ledger.injected + jnp.sum(inventories, axis=0))
    # Zero-flow control must neither consume capacity nor RNG sequence IDs.
    enabled = jnp.sum(amount) > 0
    choose = lambda a, b: jnp.where(ok & enabled, a, b)
    return (
        jax.tree.map(choose, new, parcels),
        jax.tree.map(choose, updated, ledger),
        ok | ~enabled,
    )


def build_spatial_step(
    grid, source, material, pressure, *, periodic_y=True, gravity=(0.0, 0.0, -9.81)
):
    options = source.dpm
    config = material_config(material, pressure)
    props = SimpleNamespace(
        liquid_heat_capacity=material.liquid_cp,
        air_dynamic_viscosity=material.viscosity,
    )
    exchange = build_dpm_cell_exchange(
        grid,
        None,
        config,
        props,
        periodic_x=False,
        periodic_y=periodic_y,
        drag_heat_fraction=1.0,
        vaporization=options.vaporization,
    )
    trace = build_parcel_path(
        grid,
        periodic_x=False,
        periodic_y=periodic_y,
        max_segments=options.max_path_segments,
        wall_policy="trap",
    )

    def step(gas, velocity, parcels, ledger, dt, les):
        initial = (gas, velocity, parcels, ledger)
        zero = jnp.zeros_like(gas.dry_density)

        def advance_one(i, state):
            _gas, _velocity, parcels, _ledger, accepted = state

            def active(state):
                gas, velocity, parcels, ledger, accepted = state
                path = trace(parcels.position[:, i], parcels.velocity[:, i], dt)
                accepted &= path.accepted
                p = WaterCoreBins(
                    jnp.atleast_1d(parcels.mass[i]),
                    jnp.atleast_1d(parcels.multiplicity[i]),
                    parcels.velocity[:, i, None],
                    jnp.atleast_1d(parcels.temperature[i]),
                )
                eddy = (
                    parcels.fluctuation[:, i],
                    parcels.eddy_remaining[i],
                    parcels.keys[i],
                    parcels.draws[i],
                )

                def segment(s, state):
                    gas, velocity, p, eddy, ledger, ok = state
                    cell = path.cells[s]
                    address = tuple(cell)
                    duration = path.durations[s]
                    k, epsilon, length = (
                        les.kinetic_energy[address],
                        les.dissipation[address],
                        jnp.broadcast_to(les.length, gas.dry_density.shape)[address],
                    )

                    def event(state):
                        elapsed, gas, velocity, p, eddy, ledger, ok, events = state
                        noise, remaining, key, draws = eddy
                        z, y, x = cell
                        upper_y = (y + 1) % grid.ny if periodic_y else y + 1
                        mean = jnp.array(
                            [
                                (velocity.x[z, y, x] + velocity.x[z, y, x + 1]) / 2,
                                (velocity.y[z, y, x] + velocity.y[z, upper_y, x]) / 2,
                                (velocity.z[z, y, x] + velocity.z[z + 1, y, x]) / 2,
                            ]
                        )
                        density, _ = gas_properties(
                            gas_temperature(gas, config)[address],
                            gas.vapor_density[address]
                            / (gas.dry_density[address] + gas.vapor_density[address]),
                            pressure,
                            material,
                        )

                        def renew(eddy):
                            _noise, _remaining, key, draws = eddy
                            key, nk, uk = jax.random.split(key, 3)
                            normal = jax.random.normal(nk, (3,), dtype=p.mass.dtype)
                            uniform = jax.random.uniform(
                                uk,
                                (),
                                dtype=p.mass.dtype,
                                minval=jnp.finfo(p.mass.dtype).eps,
                            )
                            slip = jnp.linalg.norm(
                                mean + normal * jnp.sqrt(2 * k / 3) - p.velocity[:, 0]
                            )
                            diameter = jnp.cbrt(
                                6 * p.mass[0] / (jnp.pi * material.liquid_density)
                            )
                            drag = water_droplet_drag_rate(
                                diameter,
                                slip,
                                SimpleNamespace(
                                    dry_air_density=density,
                                    water_density=material.liquid_density,
                                ),
                                props,
                            )
                            noise, remaining = sample_drw_eddy(
                                normal,
                                uniform,
                                k,
                                epsilon,
                                1 / drag,
                                slip,
                                length,
                                random_lifetime=options.random_lifetime,
                            )
                            return noise, remaining, key, draws + 1

                        if options.dispersion == "drw":
                            eddy = jax.lax.cond(
                                k > 0,
                                lambda e: jax.lax.cond(
                                    (e[1] <= 0) | jnp.isinf(e[1]), renew, lambda e: e, e
                                ),
                                lambda e: (
                                    jnp.zeros(3, p.mass.dtype),
                                    jnp.asarray(jnp.inf, p.mass.dtype),
                                    e[2],
                                    e[3],
                                ),
                                eddy,
                            )
                        else:
                            eddy = (
                                jnp.zeros(3, p.mass.dtype),
                                jnp.asarray(jnp.inf, p.mass.dtype),
                                key,
                                draws,
                            )
                        noise, remaining, key, draws = eddy
                        h = jnp.minimum(duration - elapsed, remaining)
                        acceleration = jnp.asarray(gravity, p.mass.dtype) * (
                            1 - density / material.liquid_density
                        )
                        before_velocity = p.velocity
                        kicked = p._replace(
                            velocity=p.velocity + 0.5 * h * acceleration[:, None]
                        )
                        gwork = 0.5 * jnp.sum(
                            p.mass
                            * p.multiplicity
                            * jnp.sum(kicked.velocity**2 - before_velocity**2, axis=0)
                        )
                        impulse = (
                            0.5 * h * acceleration * jnp.sum(p.mass * p.multiplicity)
                        )
                        result = exchange(gas, velocity, zero, kicked, h, cell, noise)
                        after = result.liquid
                        final_velocity = (
                            after.velocity + 0.5 * h * acceleration[:, None]
                        )
                        gwork += 0.5 * jnp.sum(
                            after.mass
                            * after.multiplicity
                            * jnp.sum(final_velocity**2 - after.velocity**2, axis=0)
                        )
                        impulse += (
                            0.5
                            * h
                            * acceleration
                            * jnp.sum(after.mass * after.multiplicity)
                        )
                        p = after._replace(velocity=final_velocity)
                        ok &= result.accepted & (h > 0)
                        ledger = ledger._replace(
                            gravity_impulse=ledger.gravity_impulse + impulse,
                            gravity_work=ledger.gravity_work + gwork,
                            stochastic_work=ledger.stochastic_work
                            + result.stochastic_work,
                            wall_impulse=ledger.wall_impulse + result.wall_impulse,
                            wall_energy=ledger.wall_energy + result.wall_energy,
                            evaporated_mass=ledger.evaporated_mass
                            + result.evaporated_mass,
                            maximum_source_energy_error=jnp.maximum(
                                ledger.maximum_source_energy_error,
                                jnp.abs(result.energy_error_joule),
                            ),
                        )
                        eddy = (noise, jnp.maximum(remaining - h, 0), key, draws)
                        return (
                            elapsed + h,
                            result.gas,
                            result.velocity,
                            p,
                            eddy,
                            ledger,
                            ok,
                            events + 1,
                        )

                    def condition(s):
                        return (
                            s[6]
                            & (s[0] < duration)
                            & (s[7] < options.max_eddy_intervals)
                            & (s[3].mass[0] > 0)
                        )

                    elapsed, gas, velocity, p, eddy, ledger, ok, _ = jax.lax.while_loop(
                        condition,
                        event,
                        (
                            jnp.asarray(0.0, p.mass.dtype),
                            gas,
                            velocity,
                            p,
                            eddy,
                            ledger,
                            ok,
                            jnp.asarray(0, jnp.int32),
                        ),
                    )
                    ok &= (elapsed >= duration) | (p.mass[0] == 0)
                    return gas, velocity, p, eddy, ledger, ok

                gas, velocity, p, eddy, ledger, accepted = jax.lax.fori_loop(
                    0,
                    path.segments,
                    segment,
                    (gas, velocity, p, eddy, ledger, accepted),
                )
                inventory = liquid_inventory(
                    p.mass[0],
                    p.multiplicity[0],
                    p.velocity[:, 0],
                    p.temperature[0],
                    material,
                )
                trapped = path.trapped & (path.position[2] <= 0)
                escape = path.exited & ~trapped
                ledger = ledger._replace(
                    trapped=ledger.trapped + jnp.where(trapped, inventory, 0),
                    escaped=ledger.escaped + jnp.where(escape, inventory, 0),
                )
                alive = ~path.exited & (p.mass[0] > 0)
                parcels = parcels._replace(
                    position=parcels.position.at[:, i].set(path.position),
                    velocity=parcels.velocity.at[:, i].set(p.velocity[:, 0]),
                    mass=parcels.mass.at[i].set(jnp.where(alive, p.mass[0], 0)),
                    temperature=parcels.temperature.at[i].set(p.temperature[0]),
                    multiplicity=parcels.multiplicity.at[i].set(
                        jnp.where(alive, p.multiplicity[0], 0)
                    ),
                    fluctuation=parcels.fluctuation.at[:, i].set(eddy[0]),
                    eddy_remaining=parcels.eddy_remaining.at[i].set(eddy[1]),
                    keys=parcels.keys.at[i].set(eddy[2]),
                    draws=parcels.draws.at[i].set(eddy[3]),
                )
                return gas, velocity, parcels, ledger, accepted

            live = (parcels.mass[i] * parcels.multiplicity[i] > 0) & accepted
            return jax.lax.cond(live, active, lambda s: s, state)

        gas, velocity, parcels, ledger, ok = jax.lax.fori_loop(
            0,
            options.capacity,
            advance_one,
            (gas, velocity, parcels, ledger, jnp.isfinite(dt) & (dt > 0)),
        )
        choose = lambda new, old: jnp.where(ok, new, old)
        gas, velocity, parcels, ledger = jax.tree.map(
            choose, (gas, velocity, parcels, ledger), initial
        )
        return DPMSpatialStep(gas, velocity, parcels, ledger, ok)

    return step
