"""Conservative, bounded scalar transport with SSP-RK3 on uniform FV grids.

MC reconstruction is second order in smooth regions and reduces to donor-cell
at extrema and open endpoints. A sufficient forward-Euler positivity bound is
2*dt*outgoing_rate + dt*diffusion_rate <= 1: every reconstructed outflow state
is at most twice its nonnegative cell average. Incoming reconstructed states
and diffusion weights are nonnegative. SSP-RK3 inherits this bound. No field
clipping is applied. Constant upper bounds additionally require solenoidal
face velocities and compatible boundary reservoirs.
"""

import jax
import jax.numpy as jnp

from .numerics.momentum import _mc_slope
from .scalar import PassiveScalar, open_scalar_tendency, scalar_tendency
from .state import spanwise_is_periodic, streamwise_is_periodic


def _correction(field, velocity, grid):
    """MC minus donor-cell advective tendency, with shared interior fluxes."""
    result = jnp.zeros_like(field)
    for axis, speed, periodic, width in (
        (0, velocity.z, False, grid.dz),
        (1, velocity.y, spanwise_is_periodic(velocity, grid), grid.dy),
        (2, velocity.x, streamwise_is_periodic(velocity, grid), grid.dx),
    ):
        slope = _mc_slope(field, axis, periodic)
        if periodic:
            increment = jnp.where(speed >= 0, jnp.roll(slope, 1, axis), -slope)
            flux = 0.5 * speed * increment
            difference = jnp.roll(flux, -1, axis) - flux
        else:
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            interior = [slice(None)] * 3
            interior[axis] = slice(1, -1)
            speed_inside = speed[tuple(interior)]
            increment = jnp.where(
                speed_inside >= 0, slope[tuple(lower)], -slope[tuple(upper)]
            )
            edge = [slice(None)] * 3
            edge[axis] = slice(0, 1)
            zero = jnp.zeros_like(field[tuple(edge)])
            flux = jnp.concatenate((zero, 0.5 * speed_inside * increment, zero), axis)
            difference = jnp.diff(flux, axis=axis)
        result = result - difference / width
    return result


def transport_scalars(
    fields, velocity, grid, dt, ambient, diffusivity, *, scheme="muscl-mc"
):
    """Advance stacked (field,z,y,x) scalars on frozen face fluxes/diffusivity.

    Open x uses prescribed incoming reservoir flux and interior outflow; y is
    periodic or impermeable and z is impermeable. No wall scalar flux/source
    is supported here. ``ambient`` has shape (field,z,y). ``upwind`` selects a
    donor-cell control with exactly the same SSP-RK3/subcycling/coupling path.
    Full coupled temporal order is not implied by the SSP-RK3 scalar substep.
    """
    if not grid.is_uniform:
        raise ValueError("bounded scalar transport requires a uniform grid")
    if scheme not in {"muscl-mc", "upwind"}:
        raise ValueError("bounded scalar transport must be muscl-mc or upwind")
    outgoing = jnp.zeros_like(fields[0])
    periodic_x = streamwise_is_periodic(velocity, grid)
    for axis, speed, periodic, width in (
        (0, velocity.z, False, grid.dz),
        (1, velocity.y, spanwise_is_periodic(velocity, grid), grid.dy),
        (2, velocity.x, periodic_x, grid.dx),
    ):
        if periodic:
            lower, upper = speed, jnp.roll(speed, -1, axis)
        else:
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis], high[axis] = slice(None, -1), slice(1, None)
            lower, upper = speed[tuple(low)], speed[tuple(high)]
        outgoing = outgoing + (jnp.maximum(upper, 0) - jnp.minimum(lower, 0)) / width
    diffusion_rate = (
        2
        * jnp.max(jnp.asarray(diffusivity))
        * (grid.dx**-2 + grid.dy**-2 + grid.dz**-2)
    )
    # Use the stricter MC bound for both schemes to isolate reconstruction.
    rate = 2 * jnp.max(outgoing) + diffusion_rate
    count = jnp.maximum(1, jnp.ceil(dt * rate / 0.8).astype(jnp.int32))
    h = dt / count
    model = PassiveScalar(advection_scheme="upwind")

    def rhs_one(field, reservoir):
        if periodic_x:
            rhs = scalar_tendency(
                field, velocity, grid, model, eddy_viscosity=diffusivity
            )
        else:
            rhs = open_scalar_tendency(
                field, velocity, grid, model, reservoir, eddy_viscosity=diffusivity
            )
        if scheme == "muscl-mc":
            rhs = rhs + _correction(field, velocity, grid)
        return rhs

    rhs = jax.vmap(rhs_one)

    def substep(_, q):
        q1 = q + h * rhs(q, ambient)
        q2 = 0.75 * q + 0.25 * (q1 + h * rhs(q1, ambient))
        return q / 3 + (2 / 3) * (q2 + h * rhs(q2, ambient))

    return jax.lax.fori_loop(0, count, substep, fields)
