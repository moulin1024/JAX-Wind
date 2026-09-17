"""Conservative gas-inventory operations for a future embedded spray closure.

Ambient and unresolved-jet inventories are DISJOINT parts of the gas in each
cell. These operations do not create additional carrier mass or prescribe an
entrainment rate. They are not yet connected to the fixed-density LES step.
All fields are extensive (kg, kg m/s, J, kg per species, J). Mean kinetic energy
is |momentum|^2/(2 mass); unresolved_energy stores kinetic energy absent from
that mean, not thermal enthalpy. Mechanical mixing transfers the exact mean
kinetic-energy deficit to unresolved_energy. A transport/dissipation model for
that reservoir is still required before production use.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class GasInventory(NamedTuple):
    """Cell arrays; momentum/species have a leading component/species axis.

    Preconditions: finite inputs, mass/species/unresolved_energy >= 0; species
    sum to mass. An empty inventory has all fields zero. Specific enthalpy can
    use any common reference. No volume, density or turbulent length scale is
    inferred from this bookkeeping state.
    """

    mass: jax.Array
    momentum: jax.Array
    enthalpy: jax.Array
    species: jax.Array
    unresolved_energy: jax.Array


class GasTransfer(NamedTuple):
    donor: GasInventory
    receiver: GasInventory
    transferred_mass: jax.Array
    unmet_mass: jax.Array
    mixing_energy: jax.Array


def mean_kinetic_energy(gas: GasInventory):
    """Mean kinetic energy in J; empty inventories contribute exactly zero."""
    mass = jnp.asarray(gas.mass)
    denominator = jnp.where(mass > 0, mass, 1.0)
    return jnp.where(
        mass > 0, jnp.sum(jnp.asarray(gas.momentum) ** 2, axis=0) / (2 * denominator), 0
    )


def scale_inventory(gas: GasInventory, fraction):
    return GasInventory(*(jnp.asarray(q) * fraction for q in gas))


def partition_gas_inventory(gas: GasInventory, jet_fraction):
    """Split existing gas without adding inventory or changing its velocity.

    Preconditions: 0 <= jet_fraction <= 1, cellwise. This fraction is an
    externally supplied inventory fraction, NOT a jet volume/overlap model.
    Returns (ambient, jet). The split must replace the original ownership;
    callers must not keep evolving an additional copy of ``gas``.
    """
    return scale_inventory(gas, 1 - jet_fraction), scale_inventory(gas, jet_fraction)


def merge_gas_inventories(first: GasInventory, second: GasInventory):
    """Merge disjoint inventories and retain relative-motion energy exactly.

    Returns (merged inventory, mixing energy). The stable reduced-mass formula
    is nonnegative and Galilean invariant. Do not add the resulting inventory
    to the old LES gas: it already contains both inputs. Partition variance
    and separately owned SGS energy are counted once each.
    """
    a, b = jnp.asarray(first.mass), jnp.asarray(second.mass)
    ua = first.momentum / jnp.where(a > 0, a, 1.0)
    ub = second.momentum / jnp.where(b > 0, b, 1.0)
    total = a + b
    # a*(b/total) avoids overflow from forming a*b before division.
    reduced_mass = a * (b / jnp.where(total > 0, total, 1.0))
    mixing = 0.5 * reduced_mass * jnp.sum((ua - ub) ** 2, axis=0)
    merged = GasInventory(
        total,
        first.momentum + second.momentum,
        first.enthalpy + second.enthalpy,
        first.species + second.species,
        first.unresolved_energy + second.unresolved_energy + mixing,
    )
    return merged, mixing


def transfer_gas_mass(donor: GasInventory, receiver: GasInventory, requested_mass):
    """Transfer homogeneous donor gas, with explicit inventory exhaustion.

    Preconditions: requested_mass >= 0. At most donor.mass is transferred;
    unmet_mass reports the rest so a caller can reject/subcycle the closure
    step instead of silently accepting a reduced entrainment rate. Donor
    velocity, composition and specific enthalpy are sampled before transfer.
    Existing SGS energy follows donor mass. Mixing energy is deposited only
    into the receiving inventory. No clipping of species or energy is used.

    Entrainment uses (ambient, jet). Handoff uses (jet, ambient) and requests
    the whole remaining jet.mass; repeating a complete handoff is a no-op.
    These are transfers of gas ownership, not independent LES source terms.
    """
    mass = jnp.asarray(donor.mass)
    requested = jnp.asarray(requested_mass)
    moved = jnp.minimum(requested, mass)
    fraction = moved / jnp.where(mass > 0, mass, 1.0)
    remaining = scale_inventory(donor, 1 - fraction)
    received, mixing = merge_gas_inventories(receiver, scale_inventory(donor, fraction))
    return GasTransfer(remaining, received, moved, requested - moved, mixing)


def gather_gas_inventory(cells: GasInventory):
    """Collect disjoint cell inventories into one mixed stream/control volume.

    Spatial axes are all mass axes. The velocity-variance form avoids
    subtracting large kinetic energies in a moving reference frame. Returns
    (inventory, mixing energy). This operation destroys spatial information;
    use it on withdrawn gas, never on an additional copy of the live carrier.
    """
    mass = jnp.asarray(cells.mass)
    total = jnp.sum(mass)
    momentum = jnp.sum(cells.momentum, axis=tuple(range(1, mass.ndim + 1)))
    velocity = cells.momentum / jnp.where(mass > 0, mass, 1.0)
    mean = momentum / jnp.where(total > 0, total, 1.0)
    offset = velocity - mean.reshape((3,) + (1,) * mass.ndim)
    mixing = 0.5 * jnp.sum(mass * jnp.sum(offset**2, axis=0))
    return GasInventory(
        total,
        momentum,
        jnp.sum(cells.enthalpy),
        jnp.sum(cells.species, axis=tuple(range(1, mass.ndim + 1))),
        jnp.sum(cells.unresolved_energy) + mixing,
    ), mixing


def withdraw_gas_to_core(ambient: GasInventory, core: GasInventory, requested_mass):
    """Debit a spatial ambient shell and credit a single unresolved inventory.

    Requests are kg per ambient cell, including any timestep/quadrature factor.
    Caller-defined sampling weights must represent a physical shell; this
    routine does not select or renormalize them. Exhaustion is reported in each
    original cell. Returned mixing_energy includes mixing within the withdrawn
    shell and mixing into the existing core. The receiver is scalar spatially.
    """
    mass = jnp.asarray(ambient.mass)
    requested = jnp.asarray(requested_mass)
    moved = jnp.minimum(requested, mass)
    fraction = moved / jnp.where(mass > 0, mass, 1.0)
    remaining = scale_inventory(ambient, 1 - fraction)
    intake, shell_mixing = gather_gas_inventory(scale_inventory(ambient, fraction))
    updated, core_mixing = merge_gas_inventories(core, intake)
    return GasTransfer(
        remaining, updated, moved, requested - moved, shell_mixing + core_mixing
    )
