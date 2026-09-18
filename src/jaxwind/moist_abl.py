"""Moist atmospheric state and split coupling to the open FV carrier solver."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .abl import AtmosphericSolution
from .physics.moisture import (
    MoistureState,
    advance_moisture,
    saturation_mixing_ratio,
    virtual_temperature_offset,
)
from .scalar import PassiveScalar, scalar_tendency
from .sgs import eddy_viscosity
from .state import StaggeredVelocity


class MoistAtmosphericSolution(NamedTuple):
    # Preserve the atmospheric field interface for diagnostics and full-state IO.
    velocity: StaggeredVelocity
    pressure: jax.Array
    momentum_tendency: StaggeredVelocity
    scalar: jax.Array
    scalar_tendency: jax.Array
    time: jax.Array
    step: jax.Array
    moisture: MoistureState


def initialize_moisture(flow, temperature_offset, relative_humidity, config):
    temperature = flow.scalar + temperature_offset
    ambient_vapor = relative_humidity * saturation_mixing_ratio(
        temperature[..., 0], config.pressure, config
    )
    vapor = relative_humidity * saturation_mixing_ratio(
        temperature, config.pressure, config
    )
    zero = jnp.zeros_like(flow.scalar)
    water = MoistureState(vapor, zero, zero, zero, zero)
    return MoistAtmosphericSolution(*flow, water), ambient_vapor


def transport_moisture(water, velocity, grid, dt, ambient_vapor, diffusivity):
    """Conservative upwind transport with prescribed inlet *flux*, not cells.

    Explicit subcycling includes advective and diffusive positivity limits.
    All five fields share transport, preserving water mass and size moments.
    Outflow carries the interior concentration; backflow is ambient vapor and
    zero condensate. Vertical walls have zero water flux in this mist model.
    """
    model = PassiveScalar(advection_scheme="upwind")
    dx, dy, dz = (float(min(getattr(grid, a + "_widths"))) for a in "xyz")
    widths = [
        jnp.asarray(getattr(grid, a + "_widths"), water.vapor.dtype) for a in "xyz"
    ]
    x_out = jnp.maximum(velocity.x[..., 1:], 0.0) - jnp.minimum(
        velocity.x[..., :-1], 0.0
    )
    if velocity.y.shape[1] == grid.ny:
        y_out = jnp.maximum(jnp.roll(velocity.y, -1, axis=1), 0.0) - jnp.minimum(
            velocity.y, 0.0
        )
    else:
        y_out = jnp.maximum(velocity.y[:, 1:], 0.0) - jnp.minimum(
            velocity.y[:, :-1], 0.0
        )
    z_out = jnp.maximum(velocity.z[1:], 0.0) - jnp.minimum(velocity.z[:-1], 0.0)
    outgoing = (
        x_out / widths[0][None, None, :]
        + y_out / widths[1][None, :, None]
        + z_out / widths[2][:, None, None]
    )
    limit = jnp.max(outgoing) + 2 * jnp.max(jnp.asarray(diffusivity)) * (
        dx**-2 + dy**-2 + dz**-2
    )
    count = jnp.maximum(1, jnp.ceil(dt * limit / 0.8).astype(jnp.int32))
    h = dt / count

    def one(_, current):
        updated = []
        for index, field in enumerate(current):
            ambient = ambient_vapor if index == 0 else 0.0
            rhs = scalar_tendency(
                field, velocity, grid, model, eddy_viscosity=diffusivity
            )
            # scalar_tendency uses the boundary cell on both sides of an open
            # boundary. Replace just incoming advective fluxes with the reservoir.
            left = jnp.maximum(velocity.x[..., 0], 0.0) * (ambient - field[..., 0])
            right = jnp.minimum(velocity.x[..., -1], 0.0) * (ambient - field[..., -1])
            rhs = rhs.at[..., 0].add(left / widths[0][0])
            rhs = rhs.at[..., -1].add(-right / widths[0][-1])
            updated.append(field + h * rhs)
        return MoistureState(*updated)

    return jax.lax.fori_loop(0, count, one, water)


def build_moist_atmospheric_step(
    flow_step,
    grid,
    boundaries,
    momentum,
    scalar,
    config,
    temperature_offset,
    reference_temperature,
    ambient_vapor,
    injection=None,
    *,
    scalar_transport_scheme=None,
    vapor_turbulent_schmidt=None,
    eddy_viscosity_override=None,
):
    """Symmetric microphysics split with midpoint moist buoyancy.

    Moisture transport is first-order upwind/forward Euler (subcycled); the
    coupled scheme does not inherit the carrier solver's RK3 temporal order.
    Optional scalar_transport_scheme advances heat and all moisture fields
    together with conservative SSP-RK3 on the incoming projected velocity.
    That option requires disabling scalar transport in flow_step.
    An explicit vapor_turbulent_schmidt separates moisture diffusion from
    scalar thermal diffusivity/Prandtl; None preserves shared coefficients.
    eddy_viscosity_override(velocity, inflow) supplies a boundary-consistent
    coefficient when momentum uses physical-face inlet gradients.
    """

    def exchange(flow, water, h, time):
        if injection is not None:
            liquid, number = injection(time)
            water = water._replace(
                spray_liquid=water.spray_liquid + h * liquid,
                spray_number=water.spray_number + h * number,
            )
        previous = water
        _, water = advance_moisture(flow.scalar + temperature_offset, water, h, config)
        # Apply enthalpy increments to the anomaly directly. Forming T - 300 K
        # would erase weak cooling on every float32 substep through cancellation.
        delta_temperature = (
            -config.water_vapor_latent_heat * (water.vapor - previous.vapor)
            + (config.ice_sublimation_latent_heat - config.water_vapor_latent_heat)
            * (water.cloud_ice - previous.cloud_ice)
        ) / config.dry_air_heat_capacity
        return flow._replace(scalar=flow.scalar + delta_temperature), water

    def step(state, dt, inflow):
        flow = AtmosphericSolution(*state[:7])
        flow, water = exchange(flow, state.moisture, dt / 2, state.time + dt / 4)
        # The precursor has no humidity channel: prescribe the initial ambient
        # vapor profile at inflow (initial RH is the configuration input).
        ambient = ambient_vapor
        offset = virtual_temperature_offset(
            water, reference_temperature, ambient[..., None], config
        )
        viscosity = (
            0.0
            if momentum.subfilter is None
            else eddy_viscosity(flow.velocity, grid, boundaries, momentum.subfilter)
        )
        if eddy_viscosity_override is not None:
            viscosity = eddy_viscosity_override(flow.velocity, inflow)
        schmidt = (
            scalar.turbulent_prandtl
            if vapor_turbulent_schmidt is None else vapor_turbulent_schmidt
        )
        diffusivity = config.vapor_diffusivity + viscosity / schmidt
        if scalar_transport_scheme is None:
            water = transport_moisture(water, flow.velocity, grid, dt, ambient, diffusivity)
        else:
            from .scalar_transport import transport_scalars

            # Use absolute temperature so its positivity has the same meaning
            # as that of water. Convert back only after conservative transport.
            fields = jnp.stack((flow.scalar + temperature_offset, *water))
            zero = jnp.zeros_like(ambient)
            reservoirs = jnp.stack((inflow.scalar + temperature_offset, ambient, zero, zero, zero, zero))
            coefficients = diffusivity
            if vapor_turbulent_schmidt is not None:
                coefficients = jnp.broadcast_to(diffusivity, fields.shape)
                coefficients = coefficients.at[0].set(
                    scalar.diffusivity + viscosity / scalar.turbulent_prandtl
                )
            transported = transport_scalars(
                fields, flow.velocity, grid, dt, reservoirs, coefficients,
                scheme=scalar_transport_scheme,
            )
            # Fail visibly before microphysics can clip a transport violation.
            # NaNs propagate through the compiled block to host validation.
            valid = jnp.all(jnp.isfinite(transported)) & jnp.all(transported >= 0)
            transported = jnp.where(valid, transported, jnp.nan)
            flow = flow._replace(
                scalar=transported[0] - temperature_offset,
                scalar_tendency=jnp.zeros_like(flow.scalar_tendency),
            )
            water = MoistureState(*transported[1:])
        flow = flow_step(flow, dt, inflow, offset)
        flow, water = exchange(flow, water, dt / 2, state.time + 3 * dt / 4)
        return MoistAtmosphericSolution(*flow, water)

    return step
