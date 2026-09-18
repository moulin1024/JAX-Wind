"""Simultaneous conservative MAC/parcel exchange for GPU event batches.

Drag uses a coupled backward-Euler solve with finite gas and particle inertia.
This is a different first-order splitting from the serial analytic pair sweeps.
Thermal laws and adaptive Cash-Karp droplet integration are shared unchanged.
Momentum is committed from the actual parcel impulse, not an unconverged linear
solver iterate. Cell heat is the corresponding mechanical energy loss, including
stochastic work and vapor mixing. No gas reservoir is duplicated per parcel.
"""

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from .fluent_dpm_source import _admissible, gas_temperature, material_config
from .physics.fluent_dpm import advance_droplet, gas_properties, liquid_enthalpy
from .spray_momentum import dual_average, dual_volumes
from .state import StaggeredVelocity
from .water_spray import water_droplet_drag_rate


class BatchExchange(NamedTuple):
    gas: object
    velocity: object
    mass: object
    temperature: object
    particle_velocity: object
    evaporated_mass: object
    stochastic_work: object
    wall_impulse: object
    energy_error: object
    accepted: object


def cell_average(face, axis, periodic):
    if periodic:
        return 0.5 * (face + jnp.roll(face, -1, axis))
    lo, hi = [slice(None)] * 3, [slice(None)] * 3
    lo[axis], hi[axis] = slice(None, -1), slice(1, None)
    return 0.5 * (face[tuple(lo)] + face[tuple(hi)])


def face_deposit(cell, axis, periodic):
    """Adjoint of cell_average; cell inputs are extensive inventories."""
    if periodic:
        return 0.5 * (cell + jnp.roll(cell, 1, axis))
    edge = [slice(None)] * 3
    edge[axis] = slice(0, 1)
    zero = jnp.zeros_like(cell[tuple(edge)])
    return 0.5 * (
        jnp.concatenate((zero, cell), axis) + jnp.concatenate((cell, zero), axis)
    )


def build_batch_exchange(grid, source, material, pressure, *, periodic_y):
    config = material_config(material, pressure)
    shape = (grid.nz, grid.ny, grid.nx)
    count = grid.nx * grid.ny * grid.nz
    volume = grid.dx * grid.dy * grid.dz
    periodic = (False, periodic_y, False)
    props = SimpleNamespace(air_dynamic_viscosity=material.viscosity)

    def step(gas, velocity, parcels, cells, durations, noise, live):
        z, y, x = cells.T
        address = (z, y, x)
        ids = jnp.where(live, (z * grid.ny + y) * grid.nx + x, count)

        def deposit(values):
            return (
                jnp.zeros(count, values.dtype)
                .at[ids]
                .add(values, mode="drop")
                .reshape(shape)
            )

        rho = gas.dry_density + gas.vapor_density
        temp = gas_temperature(gas, config)
        density, _ = gas_properties(
            temp[address], gas.vapor_density[address] / rho[address], pressure, material
        )
        diameter = jnp.cbrt(
            6
            * jnp.where(live, parcels.mass, 1e-12)
            / (jnp.pi * material.liquid_density)
        )
        means = jnp.stack(
            [
                cell_average(u, c, periodic[c])[address]
                for c, u in zip((2, 1, 0), velocity)
            ]
        )
        slip = jnp.linalg.norm(parcels.velocity - means - noise, axis=0)
        rates = water_droplet_drag_rate(
            diameter,
            slip,
            SimpleNamespace(
                dry_air_density=density, water_density=material.liquid_density
            ),
            props,
        )
        alpha = jnp.where(live, durations * rates / (1 + durations * rates), 0.0)
        physical = jnp.where(live, parcels.mass * parcels.multiplicity, 0.0)
        coefficient = physical * alpha
        cell_coefficient = deposit(coefficient)
        faces_mass = []
        masks = []
        solved = []
        residual_ok = jnp.asarray(True)
        for component, (c, u0) in enumerate(zip((2, 1, 0), velocity)):
            face_mass = dual_average(rho, c, periodic[c]) * dual_volumes(
                grid, c, periodic[c], rho.dtype
            )
            mask = jnp.ones_like(u0, dtype=bool)
            if c != 2 and not periodic[c]:
                first, last = [slice(None)] * 3, [slice(None)] * 3
                first[c], last[c] = 0, -1
                mask = mask.at[tuple(first)].set(False).at[tuple(last)].set(False)
            rhs = face_mass * u0 + face_deposit(
                deposit(coefficient * (parcels.velocity[component] - noise[component])),
                c,
                periodic[c],
            )
            rhs = jnp.where(mask, rhs, 0.0)

            def matrix(u, face_mass=face_mass, mask=mask, c=c):
                free = jnp.where(mask, u, 0.0)
                coupling = face_deposit(
                    cell_coefficient * cell_average(free, c, periodic[c]),
                    c,
                    periodic[c],
                )
                return face_mass * u + jnp.where(mask, coupling, 0.0)

            diagonal = face_mass + jnp.where(
                mask, 0.5 * face_deposit(cell_coefficient, c, periodic[c]), 0.0
            )
            answer, _ = cg(
                matrix,
                rhs,
                x0=u0,
                tol=1e-11,
                atol=1e-15,
                maxiter=80,
                M=lambda r, diagonal=diagonal: r / diagonal,
            )
            residual = jnp.linalg.norm(matrix(answer) - rhs)
            residual_ok &= residual <= 1e-10 * jnp.linalg.norm(rhs) + 1e-14
            residual_ok &= jnp.all(jnp.where(mask, True, u0 == 0))
            faces_mass.append(face_mass)
            masks.append(mask)
            solved.append(answer)
        gathered = jnp.stack(
            [
                cell_average(u, c, periodic[c])[address]
                for c, u in zip((2, 1, 0), solved)
            ]
        )
        dragged = parcels.velocity + alpha[None, :] * (
            gathered + noise - parcels.velocity
        )
        impulse = physical[None, :] * (parcels.velocity - dragged)
        cell_impulse = jax.vmap(deposit)(impulse)
        drag_velocity = []
        wall_impulse = []
        for component, (c, u0, mass, mask) in enumerate(
            zip((2, 1, 0), velocity, faces_mass, masks)
        ):
            pushed = face_deposit(cell_impulse[component], c, periodic[c])
            drag_velocity.append(jnp.where(mask, u0 + pushed / mass, 0.0))
            wall_impulse.append(jnp.sum(jnp.where(mask, 0.0, pushed)))
        drag_velocity = StaggeredVelocity(*drag_velocity)
        mean_average = jnp.stack(
            [
                cell_average(0.5 * (u0 + u1), c, periodic[c])[address]
                for c, u0, u1 in zip((2, 1, 0), velocity, drag_velocity)
            ]
        )
        stochastic = -jnp.sum(impulse * noise)
        drag_heat = deposit(
            jnp.sum(
                impulse * (0.5 * (parcels.velocity + dragged) - mean_average - noise),
                axis=0,
            )
        )
        heated = gas._replace(
            enthalpy_density=gas.enthalpy_density + drag_heat / volume
        )
        heated_temp = gas_temperature(heated, config)
        after_drag_mean = jnp.stack(
            [
                cell_average(u, c, periodic[c])[address]
                for c, u in zip((2, 1, 0), drag_velocity)
            ]
        )
        thermal_slip = jnp.linalg.norm(dragged - after_drag_mean - noise, axis=0)

        def thermal(m, t, tg, vapor, slip, h, active):
            return advance_droplet(
                jnp.where(active, m, 1e-12),
                t,
                tg,
                vapor,
                pressure,
                slip,
                jnp.where(active, h, 0.0),
                material,
                vaporization=source.dpm.vaporization,
            )

        update = jax.vmap(thermal)(
            parcels.mass,
            parcels.temperature,
            heated_temp[address],
            gas.vapor_density[address] / rho[address],
            thermal_slip,
            durations,
            live,
        )
        mass = jnp.where(live, update.mass, parcels.mass)
        temperature = jnp.where(live, update.temperature, parcels.temperature)
        evaporated = jnp.where(live, update.vapor_mass * parcels.multiplicity, 0.0)
        dm = deposit(evaporated)
        vapor_momentum = jax.vmap(deposit)(evaporated[None, :] * dragged)
        thermal_heat = deposit(
            jnp.where(live, update.gas_enthalpy_gain * parcels.multiplicity, 0.0)
        )
        vapor_ke = deposit(0.5 * evaporated * jnp.sum(dragged**2, axis=0))
        mix_heat = vapor_ke
        final_velocity = []
        final_mass = []
        for component, (c, u0, mass0, mask) in enumerate(
            zip((2, 1, 0), drag_velocity, faces_mass, masks)
        ):
            added = face_deposit(dm, c, periodic[c])
            pushed = face_deposit(vapor_momentum[component], c, periodic[c])
            mass1 = mass0 + added
            u1 = jnp.where(mask, (mass0 * u0 + pushed) / mass1, 0.0)
            mix_heat -= vapor_momentum[component] * cell_average(
                0.5 * (u0 + u1), c, periodic[c]
            )
            mix_heat += 0.5 * dm * cell_average(u0 * u1, c, periodic[c])
            wall_impulse[component] += jnp.sum(jnp.where(mask, 0.0, pushed))
            final_velocity.append(u1)
            final_mass.append(mass1)
        final_velocity = StaggeredVelocity(*final_velocity)
        enthalpy_gain = drag_heat + thermal_heat + mix_heat
        new_gas = gas._replace(
            vapor_density=gas.vapor_density + dm / volume,
            enthalpy_density=gas.enthalpy_density + enthalpy_gain / volume,
        )
        delta_liquid_h = jnp.sum(
            jnp.where(
                live,
                parcels.multiplicity
                * (
                    mass * liquid_enthalpy(temperature, material)
                    - parcels.mass * liquid_enthalpy(parcels.temperature, material)
                ),
                0.0,
            )
        )
        delta_liquid_ke = 0.5 * jnp.sum(
            jnp.where(
                live,
                parcels.multiplicity
                * (
                    mass * jnp.sum(dragged**2, axis=0)
                    - parcels.mass * jnp.sum(parcels.velocity**2, axis=0)
                ),
                0.0,
            )
        )
        delta_gas_ke = sum(
            0.5 * jnp.sum(m1 * (u1 - u0) * (u1 + u0) + (m1 - m0) * u0**2)
            for m0, m1, u0, u1 in zip(faces_mass, final_mass, velocity, final_velocity)
        )
        error = (
            jnp.sum(enthalpy_gain)
            + delta_liquid_h
            + delta_liquid_ke
            + delta_gas_ke
            - stochastic
        )
        scale = jnp.sum(jnp.abs(gas.enthalpy_density)) * volume + jnp.sum(
            physical * abs(liquid_enthalpy(parcels.temperature, material))
        )
        ok = (
            residual_ok
            & _admissible(new_gas, config)
            & jnp.all(~live | update.accepted)
        )
        ok &= jnp.isfinite(error) & (abs(error) <= 1e-10 * jnp.maximum(scale, 1.0))
        ok &= jnp.all(jnp.isfinite(dragged)) & jnp.all(
            jnp.isfinite(jnp.stack(wall_impulse))
        )
        return BatchExchange(
            new_gas,
            final_velocity,
            mass,
            temperature,
            dragged,
            jnp.sum(evaporated),
            stochastic,
            jnp.stack(wall_impulse),
            error,
            ok,
        )

    return step
