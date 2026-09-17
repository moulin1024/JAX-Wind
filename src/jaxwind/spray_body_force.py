"""Explicit external particle acceleration, separate from gas reaction forces."""

import jax.numpy as jnp


def kick_particles(liquid, acceleration, dt):
    """Exact constant-acceleration velocity kick at fixed current liquid mass.

    Returns updated liquid, external impulse (3,N), and external work (N).
    Work is impulse dot midpoint velocity and equals the kinetic-energy change.
    This primitive neither transfers an opposite impulse to gas nor advances
    positions. Preconditions: finite admissible bins, finite xyz acceleration,
    nonnegative duration. Empty/inactive bins receive no impulse or velocity kick.
    Evaporation is a separate mass-transfer stage, so callers must use the
    current mass at every kick and stop kicks at a parcel's exit time.
    """
    mass = liquid.mass * liquid.multiplicity
    change = jnp.asarray(acceleration, liquid.velocity.dtype)[:, None] * dt
    change = jnp.where(mass[None] > 0, change, 0.0)
    impulse = mass * change
    work = jnp.sum(impulse * (liquid.velocity + 0.5 * change), axis=0)
    return liquid._replace(velocity=liquid.velocity + change), impulse, work
