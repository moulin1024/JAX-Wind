"""Conservative reduced microphysics for liquid-nitrogen cold plumes.

All quantities use SI units.  The module is deliberately independent of a
particular time integrator so the parcel, moist-phase, and low-Mach outlet
closures can be tested before they are coupled to the distributed LES state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


# Backward-compatible exports from the shared water thermodynamics module.
from .moisture import (
    SATURATION_TEMPERATURE_CEILING,
    SATURATION_TEMPERATURE_FLOOR,
    FogMicrophysicsUpdate as FogMicrophysicsUpdate,
    advance_fog_microphysics as advance_fog_microphysics,
    saturation_adjustment as saturation_adjustment,
    saturation_mixing_ratio as saturation_mixing_ratio,
    saturation_vapor_pressure_ice as saturation_vapor_pressure_ice,
    saturation_vapor_pressure_water as saturation_vapor_pressure_water,
)


CRYOGENIC_TEMPERATURE_CEILING = SATURATION_TEMPERATURE_CEILING
CRYOGENIC_TEMPERATURE_FLOOR = SATURATION_TEMPERATURE_FLOOR


@dataclass(frozen=True)
class CryogenicMicrophysicsConfig:
    pressure: float = 100_000.0
    dry_air_density: float = 1.225
    dry_air_heat_capacity: float = 1005.0
    dry_air_gas_constant: float = 287.05
    water_vapor_gas_constant: float = 461.5
    nitrogen_gas_constant: float = 296.80
    nitrogen_gas_heat_capacity: float = 1040.0
    water_vapor_latent_heat: float = 2.50e6
    ice_sublimation_latent_heat: float = 2.834e6
    water_fusion_latent_heat: float = 3.34e5
    freezing_temperature: float = 273.15

    nitrogen_boiling_temperature: float = 77.34
    liquid_nitrogen_density: float = 806.11
    liquid_nitrogen_heat_capacity: float = 2040.0
    liquid_nitrogen_latent_heat: float = 199_180.0
    air_dynamic_viscosity: float = 1.81e-5
    air_thermal_conductivity: float = 0.0257
    air_prandtl: float = 0.71

    outlet_start_x: float = 7.5
    outlet_end_x: float = 8.0
    outlet_scalar_timescale: float = 1.0
    saturation_iterations: int = 6
    saturation_relaxation_timescale: float = 0.01
    freezing_timescale: float = 0.05
    melting_timescale: float = 0.05
    liquid_fog_diameter: float = 10.0e-6
    ice_fog_diameter: float = 20.0e-6
    water_density: float = 997.0
    ice_density: float = 917.0

    def __post_init__(self) -> None:
        positive = (
            self.pressure,
            self.dry_air_density,
            self.dry_air_heat_capacity,
            self.dry_air_gas_constant,
            self.water_vapor_gas_constant,
            self.nitrogen_gas_constant,
            self.nitrogen_gas_heat_capacity,
            self.water_vapor_latent_heat,
            self.ice_sublimation_latent_heat,
            self.water_fusion_latent_heat,
            self.freezing_temperature,
            self.nitrogen_boiling_temperature,
            self.liquid_nitrogen_density,
            self.liquid_nitrogen_heat_capacity,
            self.liquid_nitrogen_latent_heat,
            self.air_dynamic_viscosity,
            self.air_thermal_conductivity,
            self.air_prandtl,
            self.outlet_scalar_timescale,
            self.saturation_relaxation_timescale,
            self.freezing_timescale,
            self.melting_timescale,
            self.liquid_fog_diameter,
            self.ice_fog_diameter,
            self.water_density,
            self.ice_density,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("cryogenic material and timescale values must be positive")
        if self.outlet_end_x <= self.outlet_start_x:
            raise ValueError("outlet_end_x must exceed outlet_start_x")
        if self.saturation_iterations <= 0:
            raise ValueError("saturation_iterations must be positive")


class NitrogenDropletUpdate(NamedTuple):
    mass: jax.Array
    diameter: jax.Array
    temperature: jax.Array
    evaporated_mass: jax.Array
    gas_energy_loss: jax.Array


class MassOnlyOutletUpdate(NamedTuple):
    """Terms produced by the weak gas-mass outlet closure."""

    target_divergence: jax.Array
    nitrogen_tendency: jax.Array
    volume_sink: jax.Array


def stokes_terminal_velocity(
    diameter: float,
    particle_density: float,
    config: CryogenicMicrophysicsConfig,
) -> float:
    """Return the small-particle Stokes settling speed [m/s]."""

    return (
        max(particle_density - config.dry_air_density, 0.0)
        * 9.81
        * diameter**2
        / (18.0 * config.air_dynamic_viscosity)
    )


def advance_nitrogen_droplet(
    mass: jax.Array,
    diameter: jax.Array,
    temperature: jax.Array,
    gas_temperature: jax.Array,
    relative_speed: jax.Array,
    dt: float,
    config: CryogenicMicrophysicsConfig,
) -> NitrogenDropletUpdate:
    """Heat and evaporate spherical LN2 droplets with a Ranz--Marshall Nu."""

    dtype = mass.dtype
    tiny = jnp.asarray(1.0e-30, dtype=dtype)
    diameter = jnp.maximum(diameter, tiny)
    reynolds = (
        config.dry_air_density
        * jnp.abs(relative_speed)
        * diameter
        / config.air_dynamic_viscosity
    )
    nusselt = 2.0 + 0.6 * jnp.sqrt(reynolds) * config.air_prandtl ** (1.0 / 3.0)
    conductance = (
        nusselt
        * config.air_thermal_conductivity
        / diameter
        * jnp.pi
        * diameter**2
    )
    available_heat = jnp.maximum(
        conductance
        * jnp.asarray(dt, dtype=dtype)
        * (gas_temperature - temperature),
        0.0,
    )
    boiling = jnp.asarray(
        config.nitrogen_boiling_temperature, dtype=temperature.dtype
    )
    sensible_needed = mass * config.liquid_nitrogen_heat_capacity * jnp.maximum(
        boiling - temperature, 0.0
    )
    sensible_used = jnp.minimum(available_heat, sensible_needed)
    warmed_temperature = temperature + sensible_used / jnp.maximum(
        mass * config.liquid_nitrogen_heat_capacity,
        tiny,
    )
    latent_energy = jnp.maximum(available_heat - sensible_used, 0.0)
    evaporated = jnp.minimum(
        mass,
        latent_energy / config.liquid_nitrogen_latent_heat,
    )
    new_mass = jnp.maximum(mass - evaporated, 0.0)
    new_temperature = jnp.where(
        evaporated > 0.0,
        boiling,
        jnp.minimum(warmed_temperature, boiling),
    )
    evaluation_mass = jnp.maximum(new_mass, tiny)
    new_diameter = (
        6.0
        * evaluation_mass
        / (jnp.pi * config.liquid_nitrogen_density)
    ) ** (1.0 / 3.0)
    new_diameter = jnp.where(
        new_mass > 0.0, new_diameter, 0.0
    )
    return NitrogenDropletUpdate(
        mass=new_mass,
        diameter=new_diameter,
        temperature=new_temperature,
        evaporated_mass=evaporated,
        gas_energy_loss=sensible_used
        + evaporated * config.liquid_nitrogen_latent_heat,
    )


def smooth_outlet_window(
    x: jax.Array,
    start: float,
    end: float,
) -> jax.Array:
    """C-infinity rise from zero to one across the outlet interval."""

    coordinate = (x - start) / (end - start)
    epsilon = jnp.asarray(jnp.finfo(x.dtype).eps, dtype=x.dtype)
    safe = jnp.clip(coordinate, epsilon, 1.0 - epsilon)
    interior = jax.nn.sigmoid(1.0 / (1.0 - safe) - 1.0 / safe)
    return jnp.where(
        coordinate <= 0.0,
        jnp.zeros_like(interior),
        jnp.where(coordinate >= 1.0, jnp.ones_like(interior), interior),
    )


def balanced_volume_divergence(
    evaporation_mass_rate: jax.Array,
    gas_temperature: jax.Array,
    outlet_weight: jax.Array,
    cell_volume: float,
    config: CryogenicMicrophysicsConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return zero-integral volume-expansion constraint and outlet sink.

    ``evaporation_mass_rate`` has units kg m-3 s-1.  The local ideal-gas
    nitrogen density converts it to volumetric strain [s-1].  A normalized
    weak outlet sink removes exactly the same integrated gas volume without
    forcing velocity or imposing an inlet profile.
    """

    pressure = jnp.asarray(config.pressure, dtype=gas_temperature.dtype)
    nitrogen_density = pressure / (
        config.nitrogen_gas_constant * gas_temperature
    )
    expansion = evaporation_mass_rate / jnp.maximum(
        nitrogen_density,
        jnp.asarray(1.0e-12, dtype=gas_temperature.dtype),
    )
    total_volume_rate = jnp.sum(expansion) * cell_volume
    normalization = jnp.sum(outlet_weight) * cell_volume
    sink = (
        total_volume_rate
        * outlet_weight
        / jnp.maximum(
            normalization,
            jnp.asarray(1.0e-30, dtype=outlet_weight.dtype),
        )
    )
    return expansion - sink, sink


def outlet_scalar_tendency(
    field: jax.Array,
    ambient_value: float,
    outlet_weight: jax.Array,
    config: CryogenicMicrophysicsConfig,
) -> jax.Array:
    """Absorb scalar anomalies in the outlet without momentum forcing."""

    return (
        outlet_weight
        * (jnp.asarray(ambient_value, dtype=field.dtype) - field)
        / config.outlet_scalar_timescale
    )


def mass_only_outlet_update(
    evaporation_mass_rate: jax.Array,
    gas_temperature: jax.Array,
    nitrogen_mass_fraction: jax.Array,
    x_coordinates: jax.Array,
    cell_volume: float,
    config: CryogenicMicrophysicsConfig,
) -> MassOnlyOutletUpdate:
    """Close added gas volume without imposing an outlet velocity profile.

    The pressure projection receives a zero-integral divergence constraint:
    positive volume production where LN2 evaporates and an equal, smoothly
    distributed sink in the outlet strip.  Only the nitrogen mass-fraction
    anomaly is weakly absorbed to prevent periodic re-entry.  Temperature,
    water phases, and all three momentum components are deliberately absent
    from this API.
    """

    x = jnp.asarray(x_coordinates, dtype=nitrogen_mass_fraction.dtype)
    while x.ndim < nitrogen_mass_fraction.ndim:
        x = x[..., None]
    outlet_weight = jnp.broadcast_to(
        smooth_outlet_window(
            x,
            config.outlet_start_x,
            config.outlet_end_x,
        ),
        nitrogen_mass_fraction.shape,
    )
    target_divergence, volume_sink = balanced_volume_divergence(
        evaporation_mass_rate,
        gas_temperature,
        outlet_weight,
        cell_volume,
        config,
    )
    return MassOnlyOutletUpdate(
        target_divergence=target_divergence,
        nitrogen_tendency=outlet_scalar_tendency(
            nitrogen_mass_fraction,
            0.0,
            outlet_weight,
            config,
        ),
        volume_sink=volume_sink,
    )
