"""Sequential live-inventory exchange for dynamically addressed droplet groups."""

import jax
import jax.numpy as jnp

from .spray_cell_exchange import CellExchange, build_cell_exchange
from .spray_cell_step import _build_source_carrier_step


def build_group_exchange(
    grid,
    config,
    properties,
    *,
    periodic_x,
    periodic_y,
    drag_heat_fraction,
    energy_tolerance=1e-10,
):
    """Build an atomic source stage for groups that may share faces or cells.

    step(gas, velocity, unresolved, groups, dt, recipients). Group mass/count/T
    have shape (G,N), velocity (G,3,N), recipients (G,3) integer z,y,x. Each
    group's bins are owned once by their array index. The supplied order is a
    first-order source splitting: each group sees the updated live face masses,
    velocities and thermodynamics. Group-order dependence must converge with dt.
    Recipient reassignment is allowed between calls, but this routine does not
    integrate positions, crossing times, impaction, injection or outflow.
    One invalid group rejects ALL groups, restoring their old inventories.
    """
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

    def step(gas, velocity, unresolved, groups, dt, recipients):
        count = groups.mass.shape[0]
        if (
            groups.mass.ndim != 2
            or groups.multiplicity.shape != groups.mass.shape
            or groups.temperature.shape != groups.mass.shape
            or groups.velocity.shape != (count, 3, groups.mass.shape[1])
            or recipients.shape != (count, 3)
            or count < 1
        ):
            raise ValueError("invalid group array shapes")
        zero = jnp.asarray(0.0, gas.dry_density.dtype)
        initial = CellExchange(
            gas,
            velocity,
            unresolved,
            groups,
            jax.tree.map(jnp.zeros_like, gas),
            jnp.zeros((3, *gas.dry_density.shape), gas.dry_density.dtype),
            jnp.zeros_like(unresolved),
            jnp.asarray(True),
            zero,
            zero,
            zero,
            zero,
            zero,
            jnp.full((count,), jnp.nan, gas.dry_density.dtype),
        )

        def update(i, state):
            bins = jax.tree.map(lambda q: q[i], state.liquid)
            result = exchange(
                state.gas,
                state.velocity,
                state.unresolved_density,
                bins,
                dt,
                recipients[i],
            )
            liquid = jax.tree.map(
                lambda a, b: a.at[i].set(b), state.liquid, result.liquid
            )
            add = lambda a, b: a + b
            return CellExchange(
                result.gas,
                result.velocity,
                result.unresolved_density,
                liquid,
                jax.tree.map(add, state.gas_increments, result.gas_increments),
                state.momentum_increment + result.momentum_increment,
                state.unresolved_increment + result.unresolved_increment,
                state.accepted & result.accepted,
                state.evaporated_mass + result.evaporated_mass,
                state.drag_energy + result.drag_energy,
                state.vapor_mixing_energy + result.vapor_mixing_energy,
                state.energy_error_joule + result.energy_error_joule,
                jnp.maximum(state.relative_energy_error, result.relative_energy_error),
                state.candidate_temperature.at[i].set(result.candidate_temperature),
            )

        result = jax.lax.fori_loop(0, count, update, initial)
        choose = lambda a, b: jnp.where(result.accepted, a, b)
        commit = lambda q: jax.tree.map(lambda a: choose(a, jnp.zeros_like(a)), q)
        return CellExchange(
            jax.tree.map(choose, result.gas, gas),
            jax.tree.map(choose, result.velocity, velocity),
            choose(result.unresolved_density, unresolved),
            jax.tree.map(choose, result.liquid, groups),
            *(commit(q) for q in result[4:7]),
            result.accepted,
            *(commit(q) for q in result[8:11]),
            *result[11:],
        )

    return step


def build_group_spray_step(
    poisson,
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
    """Exchange all groups once, then commit with ONE common-flux carrier step."""
    ambient_e = jnp.asarray(ambient_unresolved_specific)
    if ambient_e.shape != (poisson.grid.nz, poisson.grid.ny):
        raise ValueError("ambient unresolved energy must have shape (nz,ny)")
    exchange = build_group_exchange(
        poisson.grid,
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
