"""Finite-cell phase exchange for the Fluent-reference warm-water equations.

This local source operator uses the cell's gas directly, with vapor sensible
enthalpy. It has no plume environment, spatial transport, projection, gravity,
DRW, or Fluent source-iteration emulation. Species, momentum and H+KE+owned
unresolved energy are conserved. That numerical contract is tested separately
from reproducing the selected Fluent heat/mass transfer laws.
"""

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .physics.fluent_dpm import advance_droplet, gas_properties
from .spray_core import relax_core_drag
from .spray_coupling import GasInventory, merge_gas_inventories
from .water_spray import water_droplet_drag_rate


class DPMCellExchange(NamedTuple):
    gas: GasInventory
    liquid: object
    accepted: jax.Array
    evaporated_mass: jax.Array
    terminal_mass: jax.Array
    thermal_attempts: jax.Array


def cell_temperature(gas, material):
    """Invert binary species enthalpy; dry-air and vapor masses are extensive."""
    capacity = gas.species[0]*material.dry_air_cp+gas.species[1]*material.vapor_cp
    return material.reference_temperature + (
        gas.enthalpy-gas.species[1]*material.latent_heat_reference)/capacity


def advance_cell_exchange(gas, liquid, pressure, dt, material, *,
                          drag_heat_fraction, substeps=8,
                          vaporization="diffusion-controlled"):
    """Conservative source update for all size bins sharing one finite cell.

    GasInventory stores [dry air,vapor] masses with material-consistent H.
    WaterCoreBins stores physical mass/drop, multiplicity, velocity and T.
    Frozen drag is integrated with exact finite-inertia gas/bin pair sweeps;
    thermal transfer sees the updated gas. Subcycling controls splitting error.
    Source pressure is held fixed; an external carrier must supply expansion
    and transport. This is not a sealed constant-volume thermodynamic process.

    Kinetic-energy dissipation ownership is an explicit caller choice, not an
    inferred Fluent default: fraction to gas H, remainder to unresolved_energy.
    Any failed substep rolls back the entire input state and exchange ledgers.
    """
    if type(substeps) is not int or substeps < 1 or not 0 <= drag_heat_fraction <= 1:
        raise ValueError("positive substeps and a dissipation heat fraction in [0,1] required")

    def admissible(g, p):
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves((g, p))]))
        t = cell_temperature(g, material)
        return (finite & (g.mass > 0) & (g.species[0] > 0) & jnp.all(g.species >= 0)
                & (g.unresolved_energy >= 0)
                & (abs(g.mass-jnp.sum(g.species)) <= 1e-12*g.mass)
                & jnp.all(p.mass >= 0) & jnp.all(p.multiplicity >= 0)
                & (t >= material.reference_temperature) & (t < material.boiling_temperature)
                & jnp.all(p.temperature >= material.reference_temperature)
                & jnp.all(p.temperature < material.boiling_temperature))

    def subcycle(_, state):
        g, p, accepted, total_evap, terminal, attempts = state
        temp = cell_temperature(g, material)
        y = g.species[1]/g.mass
        rho, _ = gas_properties(temp, y, pressure, material)
        diameter = jnp.cbrt(6*p.mass/(jnp.pi*material.liquid_density))
        slip = jnp.linalg.norm(p.velocity-(g.momentum/g.mass)[:, None], axis=0)
        config = SimpleNamespace(dry_air_density=rho, water_density=material.liquid_density)
        props = SimpleNamespace(air_dynamic_viscosity=material.viscosity)
        rates = water_droplet_drag_rate(diameter, slip, config, props)
        mech_g, mech_p, loss = relax_core_drag(g, p, rates, dt/substeps)
        mech_g = mech_g._replace(enthalpy=mech_g.enthalpy+drag_heat_fraction*loss,
            unresolved_energy=mech_g.unresolved_energy+(1-drag_heat_fraction)*loss)
        temp = cell_temperature(mech_g, material)
        slip = jnp.linalg.norm(mech_p.velocity-(mech_g.momentum/mech_g.mass)[:, None], axis=0)
        live = (p.mass > 0) & (p.multiplicity > 0)
        def thermal(m, t, speed, active):
            return advance_droplet(jnp.where(active, m, 1e-12), t, temp, y, pressure,
                speed, jnp.where(active, dt/substeps, 0), material, vaporization=vaporization)
        update = jax.vmap(thermal)(p.mass, p.temperature, slip, live)
        dm_bins = jnp.where(live, update.vapor_mass*p.multiplicity, 0)
        dm = jnp.sum(dm_bins)
        dp = jnp.sum(dm_bins*mech_p.velocity, axis=1)
        uv = dp/jnp.where(dm > 0, dm, 1)
        variance = 0.5*jnp.sum(dm_bins*jnp.sum((mech_p.velocity-uv[:, None])**2, axis=0))
        vapor = GasInventory(dm, dp, jnp.asarray(0.0, dm.dtype),
                             jnp.array([0.0, dm]), variance)
        new_g, mixing = merge_gas_inventories(mech_g, vapor)
        mixing = mixing+variance
        heat = jnp.sum(jnp.where(live, update.gas_enthalpy_gain*p.multiplicity, 0))
        new_g = new_g._replace(enthalpy=mech_g.enthalpy+heat+drag_heat_fraction*mixing,
            unresolved_energy=mech_g.unresolved_energy+(1-drag_heat_fraction)*mixing)
        new_p = mech_p._replace(mass=jnp.where(live, update.mass, p.mass),
                                temperature=jnp.where(live, update.temperature, p.temperature))
        valid = accepted & jnp.all(~live | update.accepted) & admissible(new_g, new_p)
        choose = lambda new, old: jnp.where(valid, new, old)
        return (jax.tree.map(choose, new_g, g), jax.tree.map(choose, new_p, p), valid,
                total_evap+jnp.where(valid, dm, 0),
                terminal+jnp.where(valid, jnp.sum(jnp.where(live, update.terminal_mass*p.multiplicity, 0)), 0),
                attempts+jnp.sum(update.attempts))

    initial_ok = admissible(gas, liquid) & jnp.isfinite(dt) & (dt >= 0) & jnp.isfinite(pressure) & (pressure > 0)
    zero = jnp.zeros_like(gas.mass)
    final = jax.lax.fori_loop(0, substeps, subcycle,
        (gas, liquid, initial_ok, zero, zero, jnp.asarray(0, jnp.int64)))
    g, p, ok, dm, terminal, attempts = final
    choose = lambda new, old: jnp.where(ok, new, old)
    return DPMCellExchange(jax.tree.map(choose, g, gas), jax.tree.map(choose, p, liquid),
                          ok, jnp.where(ok, dm, 0), jnp.where(ok, terminal, 0), attempts)
