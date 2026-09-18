"""Explicitly selected transported-RANS control for the inertial benchmark."""

import math
from dataclasses import replace

import jax
import jax.numpy as jnp

from jaxwind import LinearBoussinesqBuoyancy, build_open_atmospheric_step
from jaxwind.moist_abl import build_moist_atmospheric_step
from jaxwind.rans_kepsilon import (
    CMU,
    KEpsilonState,
    RANSInertialSolution,
    advance_turbulence,
    constrain_wall_epsilon,
    turbulent_viscosity,
    wall_terms,
)
from jaxwind.sgs import TransportedEddyViscosity
from jaxwind.water_parcels import (
    InertialMoistAtmosphericSolution,
    build_inertial_water_step,
)


def build_rans_control(
    initial,
    grid,
    boundaries,
    pressure,
    momentum,
    scalar,
    config,
    moist,
    ambient,
    source,
    project_velocity,
    turbulence_input,
    mean_speed,
    *,
    scalar_transport_scheme=None,
    vapor_turbulent_schmidt=None,
    turbulence_model="standard-k-epsilon",
    turbulence_transport_scheme="upwind",
):
    """Freeze transported viscosity over each coupled first-order split step.

    No inlet fluctuation is added to these total RANS k/epsilon reservoirs.
    Particle heat/drag laws, mass flow, geometry and wet-wall rule are shared
    with the inertial LES baseline. Standard and realizable k-epsilon are
    explicit alternatives. Shared wall/source approximations still differ from
    a fully specified Fluent setup.
    """
    if turbulence_model not in ("standard-k-epsilon", "realizable-k-epsilon"):
        raise ValueError("unsupported transported turbulence model")
    if turbulence_transport_scheme not in ("upwind", "muscl-mc"):
        raise ValueError("turbulence transport must be upwind or muscl-mc")
    intensity = turbulence_input["intensity"]
    length = turbulence_input["length_scale_m"]
    if not (
        math.isfinite(intensity)
        and 0 < intensity < 1
        and math.isfinite(length)
        and length > 0
        and math.isfinite(mean_speed)
        and mean_speed > 0
    ):
        raise ValueError(
            "RANS inlet requires positive finite speed/length and intensity in (0,1)"
        )
    k0 = (mean_speed * intensity) ** 2
    eps0 = CMU**0.75 * k0**1.5 / turbulence_input["length_scale_m"]
    turbulent = KEpsilonState(
        jnp.full_like(initial.scalar, k0), jnp.full_like(initial.scalar, eps0)
    )
    turbulent = constrain_wall_epsilon(
        turbulent, initial.velocity, grid, momentum.viscosity
    )
    initial = RANSInertialSolution(*initial, turbulent)

    diffusion_metric = sum(
        float(min(getattr(grid, a + "_widths"))) ** -2 for a in "xyz"
    )

    physical_inlet = pressure.physical_transverse_inlet
    if physical_inlet:
        from jaxwind.inlet_momentum import physical_inlet_gradients

    def step(state, dt, inflow):
        turbulent = constrain_wall_epsilon(
            state.turbulence, state.velocity, grid, momentum.viscosity
        )
        if turbulence_model == "realizable-k-epsilon":
            from jaxwind.rans_realizable import (
                turbulent_viscosity as realizable_viscosity,
            )

            inlet_gradients = (
                physical_inlet_gradients(state.velocity, inflow, grid, boundaries)
                if physical_inlet else None
            )
            nut = realizable_viscosity(
                turbulent, state.velocity, grid, boundaries, gradients=inlet_gradients
            )
        else:
            nut = turbulent_viscosity(turbulent)
        # This guard is inside the compiled step, so every attempted step is
        # checked. NaN rejects the completed block on the host, never silently
        # capping physical viscosity or accepting an unstable trajectory.
        diffusion_number = 2 * dt * jnp.max(nut + momentum.viscosity) * diffusion_metric
        valid = (
            jnp.all(jnp.isfinite(nut)) & jnp.all(nut >= 0) & (diffusion_number <= 0.5)
        )
        live_model = replace(
            momentum,
            subfilter=TransportedEddyViscosity(nut),
            forcing=lambda velocity, time: wall_terms(
                velocity, turbulent, grid, momentum.viscosity
            )[0],
        )
        # Builders run during Python tracing; numerical fields remain dynamic.
        carrier = build_open_atmospheric_step(
            grid,
            boundaries,
            pressure,
            live_model,
            scalar,
            LinearBoussinesqBuoyancy(9.81 / moist.reference_temperature_k),
            scheme="fast-rk3",
            scalar_boundary="flux",
            transport_scalar=scalar_transport_scheme is None,
        )
        moist_step = build_moist_atmospheric_step(
            carrier,
            grid,
            boundaries,
            live_model,
            scalar,
            config,
            moist.temperature_offset_k,
            moist.reference_temperature_k,
            ambient,
            scalar_transport_scheme=scalar_transport_scheme,
            vapor_turbulent_schmidt=vapor_turbulent_schmidt,
        )
        coupled = build_inertial_water_step(
            moist_step,
            grid,
            source,
            config,
            moist.temperature_offset_k,
            project_velocity=project_velocity,
        )
        flow = coupled(InertialMoistAtmosphericSolution(*state[:9]), dt, inflow)
        turbulent = advance_turbulence(
            turbulent,
            flow.velocity,
            grid,
            boundaries,
            momentum.viscosity,
            (k0, eps0),
            dt,
            model=turbulence_model,
            transport_scheme=turbulence_transport_scheme,
            gradients=(
                physical_inlet_gradients(flow.velocity, inflow, grid, boundaries)
                if physical_inlet else None
            ),
        )
        valid &= jnp.all(jnp.isfinite(jnp.stack(turbulent))) & jnp.all(
            jnp.stack(turbulent) > 0
        )
        turbulent = jax.tree.map(
            lambda field: jnp.where(valid, field, jnp.nan), turbulent
        )
        return RANSInertialSolution(*flow, turbulent)

    return initial, step
