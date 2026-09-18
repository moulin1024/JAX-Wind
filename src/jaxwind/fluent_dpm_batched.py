"""GPU event-wave tracking with simultaneous conservative parcel sources.

One wave advances every live parcel to its next cell/eddy event. There is no
loop over parcel slots. Frozen trajectories, DRW persistence, disappearance,
wall/separator ledgers and whole-step rollback retain the serial contracts.
The simultaneous backward-Euler drag/source split requires timestep refinement
against the serial reference; bitwise serial equivalence is not claimed.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from .fluent_dpm_batch_source import build_batch_exchange, cell_average
from .fluent_dpm_source import _admissible, gas_temperature, material_config
from .fluent_dpm_spatial import DPMSpatialStep
from .physics.fluent_dpm import gas_properties, liquid_enthalpy, sample_drw_eddy
from .spray_paths import build_parcel_path
from .water_spray import water_droplet_drag_rate


def build_batched_spatial_step(
    grid, source, material, pressure, *, periodic_y, gravity, tunnel_walls
):
    options = source.dpm
    if tunnel_walls and periodic_y:
        raise ValueError("tunnel walls require nonperiodic y")
    separator = options.eliminator_x_m
    if separator is not None and not 0 < separator < grid.lx:
        raise ValueError("eliminator must lie inside the domain")
    config = material_config(material, pressure)
    trace = build_parcel_path(
        grid,
        periodic_x=False,
        periodic_y=periodic_y,
        max_segments=options.max_path_segments,
        wall_policy="trap",
    )
    exchange = build_batch_exchange(
        grid, source, material, pressure, periodic_y=periodic_y
    )
    props = SimpleNamespace(air_dynamic_viscosity=material.viscosity)
    periodic = (False, periodic_y, False)
    slot = jnp.arange(options.capacity)

    def step(gas, velocity, parcels, ledger, dt, les):
        initial = (gas, velocity, parcels, ledger)
        live = parcels.mass * parcels.multiplicity > 0
        if separator is None:
            hits = jnp.zeros_like(live)
            travel = jnp.full_like(parcels.mass, dt)
        else:
            vx = parcels.velocity[0]
            crossing = (separator - parcels.position[0]) / jnp.where(vx != 0, vx, 1.0)
            hits = live & (vx != 0) & (crossing >= 0) & (crossing <= dt)
            travel = jnp.where(hits, crossing, dt)
        paths = jax.vmap(trace)(
            parcels.position.T, parcels.velocity.T, jnp.where(travel > 0, travel, dt)
        )
        immediate = hits & (travel == 0)
        # Keep a simultaneous outward wall hit ahead of separator collection.
        side = ((parcels.position[1] <= 0) & (parcels.velocity[1] < 0)) | (
            (parcels.position[1] >= grid.ly) & (parcels.velocity[1] > 0)
        )
        wall = ((parcels.position[2] <= 0) & (parcels.velocity[2] < 0)) | (
            (parcels.position[2] >= grid.lz) & (parcels.velocity[2] > 0)
        )
        if not periodic_y:
            wall |= side
        paths = paths._replace(
            position=jnp.where(immediate[:, None], parcels.position.T, paths.position),
            segments=jnp.where(immediate, 0, paths.segments),
            exited=jnp.where(immediate, wall, paths.exited),
            trapped=jnp.where(immediate, wall, paths.trapped),
        )
        valid = (
            jnp.isfinite(dt)
            & (dt > 0)
            & _admissible(gas, config)
            & jnp.all(~live | paths.accepted)
        )
        valid &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in velocity]))
        valid &= jnp.all(jnp.isfinite(parcels.mass)) & jnp.all(parcels.mass >= 0)
        valid &= jnp.all(jnp.isfinite(parcels.multiplicity)) & jnp.all(
            parcels.multiplicity >= 0
        )
        segments = jnp.zeros(options.capacity, jnp.int32)
        elapsed = jnp.zeros_like(parcels.mass)
        events = jnp.zeros(options.capacity, jnp.int32)

        def condition(state):
            _g, _u, p, _l, segment, _elapsed, _events, ok, iteration = state
            unfinished = live & (segment < paths.segments) & (p.mass > 0)
            return (
                ok
                & jnp.any(unfinished)
                & (iteration < options.max_path_segments * options.max_eddy_intervals)
            )

        def wave(state):
            gas, velocity, p, ledger, segment, elapsed, events, ok, iteration = state
            active = live & (segment < paths.segments) & (p.mass > 0)
            index = jnp.minimum(segment, options.max_path_segments - 1)
            cells = paths.cells[slot, index]
            address = tuple(cells.T)
            duration = paths.durations[slot, index]
            remaining = duration - elapsed
            temp = gas_temperature(gas, config)[address]
            density, _ = gas_properties(
                temp,
                gas.vapor_density[address]
                / (gas.dry_density[address] + gas.vapor_density[address]),
                pressure,
                material,
            )
            means = jnp.stack(
                [
                    cell_average(u, c, periodic[c])[address]
                    for c, u in zip((2, 1, 0), velocity)
                ]
            )
            k = les.kinetic_energy[address]
            epsilon = les.dissipation[address]
            length = jnp.broadcast_to(les.length, gas.dry_density.shape)[address]
            if options.dispersion == "drw":
                renew = (
                    active
                    & (k > 0)
                    & ((p.eddy_remaining <= 0) | jnp.isinf(p.eddy_remaining))
                )

                def draw(key, mass, v, mean, rho, k, epsilon, length):
                    key, nk, uk = jax.random.split(key, 3)
                    normal = jax.random.normal(nk, (3,), dtype=mass.dtype)
                    uniform = jax.random.uniform(
                        uk, (), dtype=mass.dtype, minval=jnp.finfo(mass.dtype).eps
                    )
                    slip = jnp.linalg.norm(mean + normal * jnp.sqrt(2 * k / 3) - v)
                    d = jnp.cbrt(
                        6
                        * jnp.maximum(mass, 1e-30)
                        / (jnp.pi * material.liquid_density)
                    )
                    drag = water_droplet_drag_rate(
                        d,
                        slip,
                        SimpleNamespace(
                            dry_air_density=rho, water_density=material.liquid_density
                        ),
                        props,
                    )
                    noise, lifetime = sample_drw_eddy(
                        normal,
                        uniform,
                        k,
                        epsilon,
                        1 / drag,
                        slip,
                        length,
                        random_lifetime=options.random_lifetime,
                    )
                    return key, noise, lifetime

                keys, noise, lifetime = jax.vmap(draw)(
                    p.keys, p.mass, p.velocity.T, means.T, density, k, epsilon, length
                )
                fluct = jnp.where(renew[None, :], noise.T, p.fluctuation)
                life = jnp.where(renew, lifetime, p.eddy_remaining)
                calm = active & (k <= 0)
                fluct = jnp.where(calm[None, :], 0.0, fluct)
                life = jnp.where(calm, jnp.inf, life)
                keys = jnp.where(renew[:, None], keys, p.keys)
                draws = p.draws + renew.astype(jnp.int32)
            else:
                fluct = jnp.where(active[None, :], 0.0, p.fluctuation)
                life = jnp.where(active, jnp.inf, p.eddy_remaining)
                keys, draws = p.keys, p.draws
            h = jnp.where(active, jnp.minimum(remaining, life), 0.0)
            ok &= jnp.all(~active | ((h > 0) & (events < options.max_eddy_intervals)))
            acceleration = (
                jnp.asarray(gravity, p.mass.dtype)[:, None]
                * (1 - density / material.liquid_density)[None, :]
            )
            kick = 0.5 * h[None, :] * acceleration
            kicked = p._replace(velocity=p.velocity + kick)
            before_mass = p.mass * p.multiplicity
            gravity_work = 0.5 * jnp.sum(
                before_mass
                * jnp.sum(
                    (kicked.velocity - p.velocity) * (kicked.velocity + p.velocity),
                    axis=0,
                )
            )
            impulse = jnp.sum(before_mass[None, :] * kick, axis=1)
            result = exchange(gas, velocity, kicked, cells, h, fluct, active)
            final_velocity = result.particle_velocity + kick
            after_mass = result.mass * p.multiplicity
            gravity_work += 0.5 * jnp.sum(
                after_mass
                * jnp.sum(
                    (final_velocity - result.particle_velocity)
                    * (final_velocity + result.particle_velocity),
                    axis=0,
                )
            )
            impulse += jnp.sum(after_mass[None, :] * kick, axis=1)
            p = p._replace(
                mass=result.mass,
                temperature=result.temperature,
                velocity=final_velocity,
                fluctuation=fluct,
                eddy_remaining=jnp.where(
                    active, jnp.maximum(life - h, 0.0), p.eddy_remaining
                ),
                keys=keys,
                draws=draws,
            )
            ledger = ledger._replace(
                gravity_impulse=ledger.gravity_impulse + impulse,
                gravity_work=ledger.gravity_work + gravity_work,
                stochastic_work=ledger.stochastic_work + result.stochastic_work,
                wall_impulse=ledger.wall_impulse + result.wall_impulse,
                evaporated_mass=ledger.evaporated_mass + result.evaporated_mass,
                maximum_source_energy_error=jnp.maximum(
                    ledger.maximum_source_energy_error, abs(result.energy_error)
                ),
            )
            crossed = active & (h >= remaining)
            segment += crossed.astype(jnp.int32)
            elapsed = jnp.where(crossed, 0.0, elapsed + h)
            events = jnp.where(crossed, 0, events + active.astype(jnp.int32))
            return (
                result.gas,
                result.velocity,
                p,
                ledger,
                segment,
                elapsed,
                events,
                ok & result.accepted,
                iteration + 1,
            )

        gas, velocity, p, ledger, segments, _, _, ok, _ = jax.lax.while_loop(
            condition,
            wave,
            (
                gas,
                velocity,
                parcels,
                ledger,
                segments,
                elapsed,
                events,
                valid,
                jnp.asarray(0, jnp.int32),
            ),
        )
        ok &= jnp.all(~live | (segments >= paths.segments) | (p.mass == 0))
        trapped = live & paths.trapped & (tunnel_walls | (paths.position[:, 2] <= 0))
        collected = live & hits & ~paths.exited
        escaped = live & paths.exited & ~trapped
        physical = p.mass * p.multiplicity
        inventory = jnp.concatenate(
            (
                physical[None, :],
                physical[None, :] * p.velocity,
                (physical * liquid_enthalpy(p.temperature, material))[None, :],
                (0.5 * physical * jnp.sum(p.velocity**2, axis=0))[None, :],
            ),
            axis=0,
        )
        ledger = ledger._replace(
            trapped=ledger.trapped
            + jnp.sum(jnp.where(trapped[None, :], inventory, 0.0), axis=1),
            collected=ledger.collected
            + jnp.sum(jnp.where(collected[None, :], inventory, 0.0), axis=1),
            escaped=ledger.escaped
            + jnp.sum(jnp.where(escaped[None, :], inventory, 0.0), axis=1),
        )
        alive = live & ~paths.exited & ~collected & (p.mass > 0)
        p = p._replace(
            position=jnp.where(live[None, :], paths.position.T, p.position),
            mass=jnp.where(alive, p.mass, 0.0),
            multiplicity=jnp.where(alive, p.multiplicity, 0.0),
        )
        gas, velocity, p, ledger = jax.tree.map(
            lambda new, old: jnp.where(ok, new, old),
            (gas, velocity, p, ledger),
            initial,
        )
        return DPMSpatialStep(gas, velocity, p, ledger, ok)

    return step
