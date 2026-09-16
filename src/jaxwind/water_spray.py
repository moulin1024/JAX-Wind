"""Water injection geometry; all phase change belongs to physics.moisture."""

import math

import jax.numpy as jnp
import numpy as np


def build_water_injection(
    grid, center, widths, mass_flow, diameter, ramp_time, config, *, dtype="float32"
):
    """Return liquid mixing-ratio and number tendencies in a downstream kernel.

    The specified diameter describes entrained mist at the injection plane,
    after atomization and velocity/thermal relaxation. No latent sink or vapor
    source is applied here. The discrete integral equals the liquid mass flow.
    """
    if (
        len(center) != 3
        or len(widths) != 3
        or not all(
            math.isfinite(v) for v in (*center, *widths, mass_flow, diameter, ramp_time)
        )
        or min(*widths, diameter) <= 0
        or min(mass_flow, ramp_time) < 0
    ):
        raise ValueError("invalid water injection geometry or flow")
    if not all(
        0 <= v < length for v, length in zip(center, (grid.lx, grid.ly, grid.lz))
    ):
        raise ValueError("water injection lies outside the domain")
    x, y, z = (np.asarray(getattr(grid, axis + "_centers")) for axis in "xyz")
    exponent = (
        ((x[None, None, :] - center[0]) / widths[0]) ** 2
        + ((y[None, :, None] - center[1]) / widths[1]) ** 2
        + ((z[:, None, None] - center[2]) / widths[2]) ** 2
    )
    support = np.broadcast_to(x[None, None, :] >= center[0], exponent.shape).copy()
    support[..., 0] = False
    support[..., -1] = False
    if not support.any():
        raise ValueError("water injection has no interior downstream cells")
    # Subtract the smallest exponent on support to avoid unresolved-kernel underflow.
    weights = np.where(
        support, np.exp(-0.5 * np.maximum(exponent - exponent[support].min(), 0)), 0
    )
    kernel = jnp.asarray(
        weights / np.sum(weights * np.asarray(grid.cell_volumes)), dtype
    )
    mass_per_drop = config.water_density * math.pi * diameter**3 / 6

    def source(time):
        phase = jnp.clip(time / max(ramp_time, 1.0e-30), 0.0, 1.0)
        ramp = 1.0 if ramp_time == 0 else 0.5 * (1 - jnp.cos(jnp.pi * phase))
        liquid = ramp * mass_flow / config.dry_air_density * kernel
        return liquid, liquid / mass_per_drop

    return source


def water_droplet_drag_rate(diameter, slip_speed, config, properties):
    """Morsi--Alexander spherical drag, as in Montazeri (2015), Table 1.

    Returns the inverse velocity relaxation time. The table applies below
    Re=50000; callers must check that range. Its final segment is extrapolated
    for finite numerical values above that limit, without asserting validity.
    The Stokes branch is evaluated directly, including exactly zero slip.
    """
    diameter = jnp.maximum(jnp.asarray(diameter), 1.0e-12)
    re = (
        config.dry_air_density
        * jnp.maximum(slip_speed, 0)
        * diameter
        / properties.air_dynamic_viscosity
    )
    index = jnp.searchsorted(
        jnp.asarray([0.1, 1, 10, 100, 1000, 5000, 10000]), re, side="right"
    )
    k1 = jnp.asarray([24, 22.73, 29.17, 46.50, 98.33, 148.62, -490.546, -1662.50])[
        index
    ]
    k2 = jnp.asarray([0, 0.09, -3.89, -116.67, -2778, -47500, 578700, 5416700])[index]
    k3 = jnp.asarray([0, 3.69, 1.22, 0.62, 0.36, 0.36, 0.46, 0.52])[index]
    cd_re = jnp.where(re < 0.1, 24.0, k1 + k2 / jnp.maximum(re, 0.1) + k3 * re)
    return (
        0.75
        * properties.air_dynamic_viscosity
        * cd_re
        / (config.water_density * diameter**2)
    )


def advance_water_droplet_motion(
    velocity, gas_velocity, diameter, dt, config, properties, gravity=(0.0, 0.0, -9.81)
):
    """Frozen-drag analytic motion and reaction impulse per unit liquid mass.

    Vectors use a leading xyz axis; gravity includes displaced-gas buoyancy.
    Position integration is exact for the frozen drag coefficient. Mechanical
    reaction excludes the *full* gravitational impulse, not just its damped
    velocity contribution: a terminally settling drop still pushes on the gas.
    Mass loss must be handled separately by a conservative parcel coupler.
    """
    velocity = jnp.asarray(velocity)
    gas_velocity = jnp.asarray(gas_velocity)
    gravity = jnp.asarray(gravity, velocity.dtype).reshape(
        (3,) + (1,) * (velocity.ndim - 1)
    )
    gravity = gravity * (1 - config.dry_air_density / config.water_density)
    slip = gas_velocity - velocity
    rate = water_droplet_drag_rate(
        diameter, jnp.linalg.norm(slip, axis=0), config, properties
    )
    response = -jnp.expm1(-rate * dt) / rate
    new_velocity = velocity + rate * response * slip + response * gravity
    displacement = (
        gas_velocity * dt - slip * response + gravity / rate * (dt - response)
    )
    gas_impulse = velocity - new_velocity + gravity * dt
    return new_velocity, displacement, gas_impulse
