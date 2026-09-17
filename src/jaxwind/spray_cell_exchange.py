"""Conservative water-bin exchange with the actual staggered gas inventory.

A fixed group of bins belongs to one primary cell. The gas is a view of the live
carrier, never a second persistent core inventory. Drag uses the effective
inertia of the adjoint gather/deposit pair, not an isolated lumped-cell mass.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .physics.moisture import advance_water_droplet
from .spray_core import WaterCoreBins
from .spray_low_mach import MoistGasFields, _admissible
from .spray_momentum import dual_average, dual_volumes
from .state import StaggeredVelocity
from .water_spray import water_droplet_drag_rate


def all_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(tree)]))


class CellExchange(NamedTuple):
    gas: MoistGasFields
    velocity: StaggeredVelocity
    unresolved_density: jax.Array
    liquid: WaterCoreBins
    gas_increments: MoistGasFields
    momentum_increment: jax.Array
    unresolved_increment: jax.Array
    accepted: jax.Array
    evaporated_mass: jax.Array
    drag_energy: jax.Array
    vapor_mixing_energy: jax.Array
    energy_error_joule: jax.Array
    relative_energy_error: jax.Array
    candidate_temperature: jax.Array


def relax_staggered_drag(face_mass, face_velocity, liquid, rates, dt):
    """Exact component pair relaxation with frozen rates, symmetric bin sweeps.

    face_mass/velocity have shape (3,2), one distinct pair per component.
    Cell velocity is the arithmetic mean of its two faces; an impulse splits
    equally between them. Effective inverse inertia is sum((1/2)^2/M_face).
    Returns local faces, liquid velocities, full gas impulse and exact lost KE.
    """
    inverse = jnp.sum(0.25 / face_mass, axis=1)
    effective = 1 / inverse
    masses = liquid.mass * liquid.multiplicity
    count = masses.size

    def pair(i, state):
        faces, velocities, impulse, loss = state
        mass = masses[i]
        slip = velocities[:, i] - jnp.mean(faces, axis=1)
        z = rates[i] * (1 + mass / effective) * (dt / 2)
        change = (effective / (effective + mass)) * slip * (-jnp.expm1(-z))
        change = jnp.where(mass > 0, change, 0.0)
        force_impulse = mass * change
        faces = faces + 0.5 * force_impulse[:, None] / face_mass
        velocities = velocities.at[:, i].add(-change)
        reduced = mass * (effective / (effective + mass))
        dissipated = 0.5 * jnp.sum(reduced * slip**2 * (-jnp.expm1(-2 * z)))
        return faces, velocities, impulse + force_impulse, loss + dissipated

    initial = (
        face_velocity,
        liquid.velocity,
        jnp.zeros(3, face_velocity.dtype),
        jnp.asarray(0.0, face_velocity.dtype),
    )
    state = jax.lax.fori_loop(0, count, pair, initial)
    return jax.lax.fori_loop(0, count, lambda i, s: pair(count - 1 - i, s), state)


def build_cell_exchange(
    grid,
    cell,
    config,
    properties,
    *,
    periodic_x,
    periodic_y,
    drag_heat_fraction,
    energy_tolerance=1e-10,
):
    """Build source-only exchange; gas/velocity increments commit together.

    step(gas, velocity, unresolved_density, liquid_bins, dt). Unresolved density
    is a separately owned undissipated mechanical-energy inventory, J/m3; this
    function does not infer its relation to physical SGS k. The heat fraction is
    REQUIRED, with no production default. Remaining mechanical loss is credited
    once to the owned unresolved inventory.

    Uniform grids; periodic directions need >=2 cells so the two gathered faces
    are distinct. The recipient may touch a pressure-open x boundary, but must
    be interior to impermeable y/z walls. Wall-adjacent force/impaction partition
    needs a wall reaction/thermal model and is rejected at construction.
    Set cell=None to supply an integer (z,y,x) recipient as the final step
    argument. Invalid dynamic recipients reject the transaction without wrapping
    or clipping committed ownership. No advection, EOS projection, entrainment
    or ownership transfer occurs here.
    """
    periodic = (False, periodic_y, periodic_x)
    shape = (grid.nz, grid.ny, grid.nx)
    if not grid.is_uniform:
        raise ValueError("nonuniform grid")
    if cell is not None:
        if len(cell) != 3 or any(not 0 <= i < n for i, n in zip(cell, shape)):
            raise ValueError("invalid cell")
        if cell[0] in (0, grid.nz - 1) or (
            not periodic_y and cell[1] in (0, grid.ny - 1)
        ):
            raise ValueError("cell exchange at an impermeable wall needs a wall model")
        fixed_cell = tuple(cell)
    else:
        fixed_cell = None
    if any(p and n < 2 for p, n in zip(periodic, shape)):
        raise ValueError("periodic gather requires two distinct faces")
    if not 0 <= drag_heat_fraction <= 1 or energy_tolerance <= 0:
        raise ValueError("invalid mechanical heat fraction or energy tolerance")
    volume = grid.dx * grid.dy * grid.dz

    def step(gas, velocity, unresolved, liquid, dt, recipient=None):
        if fixed_cell is not None and recipient is not None:
            raise ValueError("fixed-cell exchange does not accept another recipient")
        address = jnp.asarray(fixed_cell if recipient is None else recipient)
        if address.shape != (3,) or not jnp.issubdtype(address.dtype, jnp.integer):
            raise ValueError("recipient must contain three integer cell indices")
        cell_valid = jnp.all((address >= 0) & (address < jnp.array(shape)))
        cell_valid &= (address[0] > 0) & (address[0] < grid.nz - 1)
        if not periodic_y:
            cell_valid &= (address[1] > 0) & (address[1] < grid.ny - 1)
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
        inertia = StaggeredVelocity(
            *(dual_average(rho, c, periodic[c]) for c in (2, 1, 0))
        )
        face_mass = []
        face_velocity = []
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
        mean_velocity = jnp.mean(face_velocity, axis=1)
        diameter = jnp.cbrt(6 * liquid.mass / (jnp.pi * config.water_density))
        slip = jnp.linalg.norm(liquid.velocity - mean_velocity[:, None], axis=0)
        rates = water_droplet_drag_rate(diameter, slip, config, properties)
        rates = jnp.where(liquid.multiplicity > 0, rates, 0.0)
        faces, drop_velocity, drag_impulse, drag_loss = relax_staggered_drag(
            face_mass, face_velocity, liquid, rates, dt
        )
        heated = enthalpy + drag_heat_fraction * drag_loss
        temperature = config.freezing_temperature + (
            heated - config.water_vapor_latent_heat * vapor
        ) / (config.dry_air_heat_capacity * dry)
        update = advance_water_droplet(
            liquid.mass,
            liquid.temperature,
            temperature,
            vapor / dry,
            jnp.linalg.norm(drop_velocity - jnp.mean(faces, axis=1)[:, None], axis=0),
            dt * (liquid.multiplicity > 0),
            config,
            properties,
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
            dh + du + liquid_h(staged_liquid) - liquid_h(liquid) + after_k - before_k
        )
        scale = jnp.maximum(
            jnp.abs(enthalpy)
            + jnp.abs(liquid_h(liquid))
            + before_k
            + unresolved[cell] * volume,
            jnp.finfo(rho.dtype).tiny,
        )
        relative = jnp.abs(error) / scale
        input_valid = (
            cell_valid
            & _admissible(gas, config)
            & jnp.all(jnp.isfinite(unresolved))
            & jnp.all(unresolved >= 0)
            & (dt > 0)
        )
        input_valid &= (
            jnp.all(liquid.mass >= 0)
            & jnp.all(liquid.multiplicity >= 0)
            & jnp.all(liquid.temperature >= config.freezing_temperature)
        )
        input_valid &= (
            all_finite(liquid) & all_finite(velocity) & jnp.all(face_mass > 0)
        )
        input_valid &= jnp.isfinite(dt)
        valid = (
            input_valid
            & _admissible(staged, config)
            & jnp.isfinite(relative)
            & (relative <= energy_tolerance)
        )
        valid &= (
            jnp.all(jnp.isfinite(staged_liquid.velocity))
            & jnp.all(staged_liquid.mass >= 0)
            & jnp.all(jnp.isfinite(staged_liquid.temperature))
        )
        valid &= jnp.all(jnp.isfinite(staged_unresolved)) & jnp.all(
            staged_unresolved >= 0
        )
        choose = lambda a, b: jnp.where(valid, a, b)
        commit = lambda x: jax.tree.map(lambda q: choose(q, jnp.zeros_like(q)), x)
        candidate_temperature = config.freezing_temperature + (
            staged.enthalpy_density[cell]
            - config.water_vapor_latent_heat * staged.vapor_density[cell]
        ) / (config.dry_air_heat_capacity * staged.dry_density[cell])
        return CellExchange(
            jax.tree.map(choose, staged, gas),
            jax.tree.map(choose, staged_velocity, velocity),
            choose(staged_unresolved, unresolved),
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
        )

    return step
