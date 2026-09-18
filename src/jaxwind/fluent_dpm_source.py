"""Spatial DPM source exchange with adjoint staggered momentum accounting."""

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .physics.fluent_dpm import advance_droplet, gas_properties
from .physics.moisture import WaterDropletUpdate
from .spray_cell_exchange import all_finite, relax_staggered_drag
from .spray_low_mach import MoistGasFields
from .spray_momentum import dual_average, dual_volumes
from .state import StaggeredVelocity
from .water_spray import water_droplet_drag_rate


class DPMSourceExchange(NamedTuple):
    gas: object
    velocity: object
    unresolved_density: object
    liquid: object
    gas_increments: object
    momentum_increment: object
    unresolved_increment: object
    accepted: object
    evaporated_mass: object
    drag_energy: object
    vapor_mixing_energy: object
    energy_error_joule: object
    relative_energy_error: object
    candidate_temperature: object
    stochastic_work: object
    wall_impulse: object
    wall_energy: object


def material_config(material, pressure):
    return SimpleNamespace(
        material=material,
        pressure=pressure,
        freezing_temperature=material.reference_temperature,
        dry_air_heat_capacity=material.dry_air_cp,
        vapor_cp=material.vapor_cp,
        water_vapor_latent_heat=material.latent_heat_reference,
        water_density=material.liquid_density,
    )


def gas_temperature(gas, config):
    return config.freezing_temperature + (
        gas.enthalpy_density - config.water_vapor_latent_heat * gas.vapor_density
    ) / (
        config.dry_air_heat_capacity * gas.dry_density
        + config.vapor_cp * gas.vapor_density
    )


def _admissible(gas, config):
    temp = gas_temperature(gas, config)
    return (
        jnp.all(jnp.isfinite(jnp.stack(gas)))
        & jnp.all(gas.dry_density > 0)
        & jnp.all(gas.vapor_density >= 0)
        & jnp.all(temp >= config.freezing_temperature)
        & jnp.all(temp < config.material.boiling_temperature)
    )


def build_dpm_cell_exchange(
    grid,
    cell,
    config,
    properties,
    *,
    periodic_x,
    periodic_y,
    drag_heat_fraction,
    energy_tolerance=1e-10,
    vaporization="diffusion-controlled",
    prevalidated_fields=False,
):
    """Actual MAC face-inventory exchange with Fluent thermal properties.

    Same adjoint half-face gather/deposit and finite-inertia drag transaction
    as the verified spray_cell_exchange. Near walls, normal face momentum and
    KE are exported to explicit wall ledgers. A prescribed stochastic velocity
    drives drag; its work is reported, never hidden as energy conservation.
    ``prevalidated_fields`` is internal to the serial spatial transaction: the
    caller checks all gas/velocity/unresolved fields once, and each exchange
    checks its touched cell/faces. Untouched entries cannot change validity.
    """
    periodic = (False, periodic_y, periodic_x)
    shape = (grid.nz, grid.ny, grid.nx)
    if not grid.is_uniform:
        raise ValueError("nonuniform grid")
    if cell is not None:
        if len(cell) != 3 or any(not 0 <= i < n for i, n in zip(cell, shape)):
            raise ValueError("invalid cell")
        fixed_cell = tuple(cell)
    else:
        fixed_cell = None
    if any(p and n < 2 for p, n in zip(periodic, shape)):
        raise ValueError("periodic gather requires two distinct faces")
    if not 0 <= drag_heat_fraction <= 1 or energy_tolerance <= 0:
        raise ValueError("invalid mechanical heat fraction or energy tolerance")
    volume = grid.dx * grid.dy * grid.dz

    def step(gas, velocity, unresolved, liquid, dt, recipient=None, fluctuation=None):
        if fixed_cell is not None and recipient is not None:
            raise ValueError("fixed-cell exchange does not accept another recipient")
        address = jnp.asarray(fixed_cell if recipient is None else recipient)
        if address.shape != (3,) or not jnp.issubdtype(address.dtype, jnp.integer):
            raise ValueError("recipient must contain three integer cell indices")
        cell_valid = jnp.all((address >= 0) & (address < jnp.array(shape)))
        # Safe attempted arithmetic for invalid addresses; never commit it.
        cell = tuple(jnp.clip(address, 0, jnp.array(shape) - 1))
        indices = []
        for c in (2, 1, 0):
            low, high = list(cell), list(cell)
            high[c] += 1
            if periodic[c]:
                high[c] %= shape[c]
            indices.append((tuple(low), tuple(high)))
        rho = gas.dry_density + gas.vapor_density
        face_mass = []
        face_velocity = []
        if prevalidated_fields:
            # Gather just the neighbouring primary inventories rather than
            # constructing three whole dual-density fields per parcel.
            for c, u, (lo, hi) in zip((2, 1, 0), velocity, indices):
                masses = []
                for face in (lo, hi):
                    left, right = list(face), list(face)
                    left[c] -= 1
                    if periodic[c]:
                        left[c] %= shape[c]
                        right[c] %= shape[c]
                    else:
                        left[c] = jnp.clip(left[c], 0, shape[c] - 1)
                        right[c] = jnp.clip(right[c], 0, shape[c] - 1)
                    left_rho = (
                        gas.dry_density[tuple(left)] + gas.vapor_density[tuple(left)]
                    )
                    right_rho = (
                        gas.dry_density[tuple(right)] + gas.vapor_density[tuple(right)]
                    )
                    boundary = (
                        jnp.asarray(False)
                        if periodic[c]
                        else ((face[c] == 0) | (face[c] == shape[c]))
                    )
                    density = jnp.where(
                        boundary, right_rho, 0.5 * (left_rho + right_rho)
                    )
                    width = (grid.dz, grid.dy, grid.dx)[c]
                    dual_volume = jnp.where(boundary, 0.5 * width, width) * (
                        volume / width
                    )
                    masses.append(density * dual_volume)
                face_mass.append(jnp.stack(masses))
                face_velocity.append(jnp.stack((u[lo], u[hi])))
        else:
            inertia = StaggeredVelocity(
                *(dual_average(rho, c, periodic[c]) for c in (2, 1, 0))
            )
            for c, r, u, (lo, hi) in zip((2, 1, 0), inertia, velocity, indices):
                volumes = jnp.broadcast_to(
                    dual_volumes(grid, c, periodic[c], rho.dtype), r.shape
                )
                face_mass.append(jnp.stack((r[lo] * volumes[lo], r[hi] * volumes[hi])))
                face_velocity.append(jnp.stack((u[lo], u[hi])))
        face_mass = jnp.stack(face_mass)
        face_velocity = jnp.stack(face_velocity)
        dry = gas.dry_density[cell] * volume
        vapor = gas.vapor_density[cell] * volume
        enthalpy = gas.enthalpy_density[cell] * volume
        fluctuation = jnp.zeros(3, rho.dtype) if fluctuation is None else fluctuation
        mean_velocity = jnp.mean(face_velocity, axis=1) + fluctuation
        local_temperature = gas_temperature(
            MoistGasFields(*(a[cell] for a in gas)), config
        )
        local_density, _ = gas_properties(
            local_temperature, vapor / (dry + vapor), config.pressure, config.material
        )
        drag_config = SimpleNamespace(
            dry_air_density=local_density, water_density=config.water_density
        )
        diameter = jnp.cbrt(6 * liquid.mass / (jnp.pi * config.water_density))
        slip = jnp.linalg.norm(liquid.velocity - mean_velocity[:, None], axis=0)
        rates = water_droplet_drag_rate(diameter, slip, drag_config, properties)
        rates = jnp.where(liquid.multiplicity > 0, rates, 0.0)
        faces, drop_velocity, drag_impulse, drag_loss = relax_staggered_drag(
            face_mass, face_velocity + fluctuation[:, None], liquid, rates, dt
        )
        faces = faces - fluctuation[:, None]
        stochastic_work = -jnp.dot(drag_impulse, fluctuation)
        heated = enthalpy + drag_heat_fraction * drag_loss
        temperature = config.freezing_temperature + (
            heated - config.water_vapor_latent_heat * vapor
        ) / (config.dry_air_heat_capacity * dry + config.vapor_cp * vapor)
        slip = jnp.linalg.norm(
            drop_velocity - (jnp.mean(faces, axis=1) + fluctuation)[:, None], axis=0
        )

        def thermal(m, t, speed, active):
            update = advance_droplet(
                jnp.where(active, m, 1e-12),
                t,
                temperature,
                vapor / (dry + vapor),
                config.pressure,
                speed,
                jnp.where(active, dt, 0),
                config.material,
                vaporization=vaporization,
            )
            return WaterDropletUpdate(
                jnp.where(active, jnp.where(update.accepted, update.mass, jnp.nan), m),
                jnp.cbrt(6 * update.mass / (jnp.pi * config.water_density)),
                jnp.where(active, update.temperature, t),
                jnp.where(active, update.vapor_mass, 0),
                jnp.where(
                    active,
                    config.water_vapor_latent_heat * update.vapor_mass
                    - update.gas_enthalpy_gain,
                    0,
                ),
            )

        update = jax.vmap(thermal)(
            liquid.mass,
            liquid.temperature,
            slip,
            (liquid.mass > 0) & (liquid.multiplicity > 0),
        )
        evaporated = update.evaporated_mass * liquid.multiplicity
        dm = jnp.sum(evaporated)
        vapor_impulse = jnp.sum(evaporated * drop_velocity, axis=1)
        vapor_velocity = vapor_impulse / jnp.where(dm > 0, dm, 1.0)
        vapor_variance = 0.5 * jnp.sum(
            evaporated * jnp.sum((drop_velocity - vapor_velocity[:, None]) ** 2, axis=0)
        )
        added = dm / 2
        reduced = face_mass * (added / (face_mass + added))
        mixing = vapor_variance + 0.5 * jnp.sum(
            reduced * (faces - vapor_velocity[:, None]) ** 2
        )
        source_faces = (face_mass * faces + 0.5 * vapor_impulse[:, None]) / (
            face_mass + added
        )
        wall_mask = []
        for axis, (lo, hi) in zip((2, 1, 0), indices):
            wall_mask.append(
                jnp.array([lo[axis] == 0, hi[axis] == shape[axis]])
                if axis != 2 and not periodic[axis]
                else jnp.zeros(2, bool)
            )
        wall_mask = jnp.stack(wall_mask)
        wall_impulse = jnp.sum(
            jnp.where(wall_mask, (face_mass + added) * source_faces, 0), axis=1
        )
        wall_energy = 0.5 * jnp.sum(
            jnp.where(wall_mask, (face_mass + added) * source_faces**2, 0)
        )
        source_faces = jnp.where(wall_mask, 0, source_faces)
        dp = drag_impulse + vapor_impulse
        dh = (
            drag_heat_fraction * (drag_loss + mixing)
            + config.water_vapor_latent_heat * dm
            - jnp.sum(update.gas_sensible_energy_loss * liquid.multiplicity)
        )
        du = (1 - drag_heat_fraction) * (drag_loss + mixing)
        zeros = jnp.zeros_like(rho)
        increments = MoistGasFields(
            zeros, zeros.at[cell].set(dm / volume), zeros.at[cell].set(dh / volume)
        )
        momentum = (
            jnp.zeros((3, *rho.shape), rho.dtype)
            .at[(slice(None), *cell)]
            .set(dp / volume)
        )
        unresolved_increment = zeros.at[cell].set(du / volume)
        if prevalidated_fields:
            staged = MoistGasFields(
                gas.dry_density,
                gas.vapor_density.at[cell].add(dm / volume),
                gas.enthalpy_density.at[cell].add(dh / volume),
            )
        else:
            staged = jax.tree.map(lambda a, b: a + b, gas, increments)
        staged_unresolved = unresolved + unresolved_increment
        staged_velocity = []
        for k, (u, (lo, hi)) in enumerate(zip(velocity, indices)):
            staged_velocity.append(
                u.at[lo].set(source_faces[k, 0]).at[hi].set(source_faces[k, 1])
            )
        staged_velocity = StaggeredVelocity(*staged_velocity)
        staged_liquid = liquid._replace(
            mass=update.mass, temperature=update.temperature, velocity=drop_velocity
        )
        liquid_h = lambda q: jnp.sum(
            q.mass
            * q.multiplicity
            * properties.liquid_heat_capacity
            * (q.temperature - config.freezing_temperature)
        )
        liquid_k = lambda q: (
            0.5 * jnp.sum(q.mass * q.multiplicity * jnp.sum(q.velocity**2, axis=0))
        )
        before_k = 0.5 * jnp.sum(face_mass * face_velocity**2) + liquid_k(liquid)
        after_k = 0.5 * jnp.sum((face_mass + added) * source_faces**2) + liquid_k(
            staged_liquid
        )
        error = (
            dh
            + du
            + liquid_h(staged_liquid)
            - liquid_h(liquid)
            + after_k
            - before_k
            + wall_energy
            - stochastic_work
        )
        scale = jnp.maximum(
            jnp.abs(enthalpy)
            + jnp.abs(liquid_h(liquid))
            + before_k
            + unresolved[cell] * volume,
            jnp.finfo(rho.dtype).tiny,
        )
        relative = jnp.abs(error) / scale
        if prevalidated_fields:
            checked_gas = MoistGasFields(*(a[cell] for a in gas))
            checked_staged = MoistGasFields(*(a[cell] for a in staged))
            checked_unresolved = unresolved[cell]
            checked_staged_unresolved = staged_unresolved[cell]
            checked_velocity = face_velocity
        else:
            checked_gas, checked_staged = gas, staged
            checked_unresolved, checked_staged_unresolved = (
                unresolved,
                staged_unresolved,
            )
            checked_velocity = velocity
        input_valid = (
            cell_valid
            & _admissible(checked_gas, config)
            & jnp.all(jnp.isfinite(checked_unresolved))
            & jnp.all(checked_unresolved >= 0)
            & (dt > 0)
        )
        input_valid &= (
            jnp.all(liquid.mass >= 0)
            & jnp.all(liquid.multiplicity >= 0)
            & jnp.all(liquid.temperature >= config.freezing_temperature)
        )
        input_valid &= (
            all_finite(liquid) & all_finite(checked_velocity) & jnp.all(face_mass > 0)
        )
        input_valid &= jnp.isfinite(dt)
        valid = (
            input_valid
            & _admissible(checked_staged, config)
            & jnp.isfinite(relative)
            & (relative <= energy_tolerance)
        )
        valid &= (
            jnp.all(jnp.isfinite(staged_liquid.velocity))
            & jnp.all(staged_liquid.mass >= 0)
            & jnp.all(jnp.isfinite(staged_liquid.temperature))
        )
        valid &= jnp.all(jnp.isfinite(checked_staged_unresolved)) & jnp.all(
            checked_staged_unresolved >= 0
        )
        choose = lambda a, b: jnp.where(valid, a, b)
        commit = lambda x: jax.tree.map(lambda q: choose(q, jnp.zeros_like(q)), x)
        candidate_temperature = config.freezing_temperature + (
            staged.enthalpy_density[cell]
            - config.water_vapor_latent_heat * staged.vapor_density[cell]
        ) / (
            config.dry_air_heat_capacity * staged.dry_density[cell]
            + config.vapor_cp * staged.vapor_density[cell]
        )
        if prevalidated_fields:
            # Only one gas cell and six MAC faces can change. Avoid selecting
            # every untouched grid entry for every parcel transaction.
            committed_gas = jax.tree.map(
                lambda old, new: old.at[cell].set(choose(new[cell], old[cell])),
                gas,
                staged,
            )
            committed_velocity = StaggeredVelocity(
                *(
                    old.at[lo]
                    .set(choose(new[lo], old[lo]))
                    .at[hi]
                    .set(choose(new[hi], old[hi]))
                    for old, new, (lo, hi) in zip(velocity, staged_velocity, indices)
                )
            )
            committed_unresolved = unresolved.at[cell].set(
                choose(staged_unresolved[cell], unresolved[cell])
            )
        else:
            committed_gas = jax.tree.map(choose, staged, gas)
            committed_velocity = jax.tree.map(choose, staged_velocity, velocity)
            committed_unresolved = choose(staged_unresolved, unresolved)
        return DPMSourceExchange(
            committed_gas,
            committed_velocity,
            committed_unresolved,
            jax.tree.map(choose, staged_liquid, liquid),
            commit(increments),
            commit(momentum),
            commit(unresolved_increment),
            valid,
            commit(dm),
            commit(drag_loss),
            commit(mixing),
            error,
            relative,
            candidate_temperature,
            commit(stochastic_work),
            commit(wall_impulse),
            commit(wall_energy),
        )

    return step
