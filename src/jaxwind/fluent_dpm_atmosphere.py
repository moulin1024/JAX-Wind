"""Fluent-reference parcels coupled to the recorded-inflow turbine LES.

Boussinesq carrier: dry-air reference density and divergence-free volume flow
are retained. Vapor and full species enthalpy use common conservative fluxes;
this is not a variable-density Fluent pressure/EOS solver. Cloud equilibrium
is disabled in this pure DPM baseline. Parcel drag supplies liquid loading.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .abl import AtmosphericSolution
from .fluent_dpm_les import dpm_les_fields
from .fluent_dpm_spatial import (
    build_spatial_step,
    initial_ledger,
    initial_parcels,
    inject_parcels,
)
from .physics.fluent_dpm import DPMWaterMaterial
from .physics.moisture import MoistureState, saturation_vapor_pressure_water
from .scalar_transport import transport_scalars
from .spray_low_mach import MoistGasFields
from .state import OPEN, PERIODIC


class DPMAtmosphericSolution(NamedTuple):
    velocity: object
    pressure: object
    momentum_tendency: object
    scalar: object
    scalar_tendency: object
    time: object
    step: object
    moisture: object
    enthalpy: object  # J/kg reference dry air; includes vapor sensible/formation H
    parcels: object
    dpm_ledger: object
    accepted: object
    transport_water: object
    transport_enthalpy: object


def specific_enthalpy(temperature, vapor_ratio, material):
    return (material.dry_air_cp + vapor_ratio * material.vapor_cp) * (
        temperature - material.reference_temperature
    ) + vapor_ratio * material.latent_heat_reference


def temperature_from_enthalpy(enthalpy, vapor_ratio, material):
    return material.reference_temperature + (
        enthalpy - material.latent_heat_reference * vapor_ratio
    ) / (material.dry_air_cp + material.vapor_cp * vapor_ratio)


def initialize_dpm(flow, moisture, source, material=None):
    material = DPMWaterMaterial() if material is None else material
    temp = flow.scalar + moisture.temperature_offset_k
    partial = moisture.ambient_relative_humidity * saturation_vapor_pressure_water(temp)
    q = (
        (material.dry_air_gas_constant / material.vapor_gas_constant)
        * partial
        / (moisture.thermodynamics.pressure - partial)
    )
    zero = jnp.zeros_like(q)
    water = MoistureState(q, zero, zero, zero, zero)
    return DPMAtmosphericSolution(
        *flow,
        water,
        specific_enthalpy(temp, q, material),
        initial_parcels(source.dpm, flow.scalar.dtype),
        initial_ledger(flow.scalar.dtype),
        jnp.asarray(True),
        jnp.asarray(0.0, flow.scalar.dtype),
        jnp.asarray(0.0, flow.scalar.dtype),
    )


def build_dpm_atmospheric_step(
    flow_step,
    grid,
    boundaries,
    momentum,
    scalar,
    moisture,
    source,
    center,
    ambient_vapor,
    project_velocity,
    *,
    material=None,
    gravity=(0.0, 0.0, -9.81),
):
    material = DPMWaterMaterial() if material is None else material
    if not grid.is_uniform:
        raise ValueError("Fluent DPM spatial coupling requires a uniform grid")
    if boundaries.streamwise != OPEN or boundaries.spanwise != PERIODIC:
        raise ValueError("DPM atmosphere requires open x and periodic y boundaries")
    if scalar is None or scalar.lower_flux != 0 or scalar.upper_flux != 0:
        raise ValueError(
            "DPM enthalpy transport currently requires zero imposed boundary heat flux"
        )
    if not all(
        0 < x < length for x, length in zip(center, (grid.lx, grid.ly, grid.lz))
    ):
        raise ValueError("DPM injection must lie inside the domain")
    pressure = moisture.thermodynamics.pressure
    rho_d = moisture.thermodynamics.dry_air_density
    spatial = build_spatial_step(
        grid, source, material, pressure, periodic_y=True, gravity=gravity
    )
    shape = (grid.nz, grid.ny, grid.nx)
    dry = jnp.full(shape, rho_d)
    volume = grid.dx * grid.dy * grid.dz

    def step(state, dt, inflow):
        def advance(state):
            parcels, ledger, ok = inject_parcels(
                state.parcels,
                state.dpm_ledger,
                source,
                center,
                state.time,
                dt,
                material,
            )
            gas = MoistGasFields(
                dry, rho_d * state.moisture.vapor, rho_d * state.enthalpy
            )

            def substep(_, values):
                gas, velocity, parcels, ledger, ok = values
                les = dpm_les_fields(
                    velocity,
                    grid,
                    boundaries,
                    momentum.subfilter,
                    amd_length=source.dpm.amd_length_scale_m,
                )
                result = spatial(
                    gas,
                    velocity,
                    parcels,
                    ledger,
                    dt / source.dpm.tracking_substeps,
                    les,
                )
                return (
                    result.gas,
                    result.velocity,
                    result.parcels,
                    result.ledger,
                    ok & result.accepted,
                )

            gas, velocity, parcels, ledger, ok = jax.lax.fori_loop(
                0,
                source.dpm.tracking_substeps,
                substep,
                (gas, state.velocity, parcels, ledger, ok),
            )
            velocity = project_velocity(velocity, dt, inflow)
            q, H = gas.vapor_density / rho_d, gas.enthalpy_density / rho_d
            les = dpm_les_fields(
                velocity,
                grid,
                boundaries,
                momentum.subfilter,
                amd_length=source.dpm.amd_length_scale_m,
            )
            diffusivity = (
                material.binary_diffusivity + les.viscosity / scalar.turbulent_prandtl
            )
            fields = jnp.stack((H, q))
            reservoir = jnp.stack(
                (
                    specific_enthalpy(
                        inflow.scalar + moisture.temperature_offset_k,
                        ambient_vapor,
                        material,
                    ),
                    ambient_vapor,
                )
            )
            transported = transport_scalars(
                fields, velocity, grid, dt, reservoir, diffusivity, scheme="muscl-mc"
            )
            H, q = transported
            temperature = temperature_from_enthalpy(H, q, material)
            ok &= jnp.all(jnp.isfinite(transported)) & jnp.all(q >= 0)
            ok &= jnp.all(temperature >= material.reference_temperature) & jnp.all(
                temperature < material.boiling_temperature
            )
            flow = AtmosphericSolution(*state[:7])._replace(
                velocity=velocity,
                scalar=temperature - moisture.temperature_offset_k,
                scalar_tendency=jnp.zeros_like(H),
            )
            offset = (
                moisture.reference_temperature_k
                * (material.vapor_gas_constant / material.dry_air_gas_constant - 1)
                * (q - ambient_vapor[..., None])
            )
            flow = flow_step(flow, dt, inflow, offset)
            ok &= jnp.all(
                jnp.stack(
                    [jnp.all(jnp.isfinite(a)) for a in (*flow.velocity, flow.scalar)]
                )
            )
            result = DPMAtmosphericSolution(
                *flow,
                state.moisture._replace(vapor=q),
                H,
                parcels,
                ledger,
                ok,
                state.transport_water
                + rho_d * volume * jnp.sum(transported[1] - fields[1]),
                state.transport_enthalpy
                + rho_d * volume * jnp.sum(transported[0] - fields[0]),
            )
            # Reject atomically. Keep accepted=False so block execution stops.
            result = jax.tree.map(
                lambda new, old: jnp.where(ok, new, old), result, state
            )
            return result._replace(accepted=ok)

        return jax.lax.cond(state.accepted, advance, lambda s: s, state)

    return step


def dpm_diagnostics(state):
    p, l = state.parcels, state.dpm_ledger
    return {
        "dpm_active_parcels": jnp.sum(p.mass * p.multiplicity > 0),
        "dpm_injected_mass_kg": l.injected[0],
        "dpm_liquid_mass_kg": jnp.sum(p.mass * p.multiplicity),
        "dpm_escaped_mass_kg": l.escaped[0],
        "dpm_trapped_mass_kg": l.trapped[0],
        "dpm_evaporated_mass_kg": l.evaporated_mass,
        "dpm_water_budget_error_kg": jnp.sum(p.mass * p.multiplicity)
        + l.escaped[0]
        + l.trapped[0]
        + l.evaporated_mass
        - l.injected[0],
        "dpm_stochastic_work_J": l.stochastic_work,
        "dpm_gravity_work_J": l.gravity_work,
        "dpm_wall_energy_J": l.wall_energy,
        "dpm_max_source_energy_error_J": l.maximum_source_energy_error,
        "dpm_drw_draws_live_slots": jnp.sum(p.draws),
        "dpm_net_vapor_transport_kg": state.transport_water,
        "dpm_net_enthalpy_transport_J": state.transport_enthalpy,
    }
