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


# Bracket for the fixed-enthalpy saturation solve. The floor is below the
# 77.34 K nitrogen boiling point so that air chilled by an LN2 source is
# bracketed rather than clipped; note that the Murphy-Koop ice saturation fit
# is only calibrated down to about 110 K, so temperatures below that are an
# extrapolation (saturation vapour pressure there is negligible in any case).
CRYOGENIC_TEMPERATURE_FLOOR = 50.0
CRYOGENIC_TEMPERATURE_CEILING = 450.0


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


class FogMicrophysicsUpdate(NamedTuple):
    temperature: jax.Array
    qv: jax.Array
    ql: jax.Array
    qi: jax.Array
    condensed_or_deposited: jax.Array
    evaporated_or_sublimated: jax.Array
    frozen: jax.Array
    melted: jax.Array


def saturation_vapor_pressure_water(temperature: jax.Array) -> jax.Array:
    """Murphy--Koop (2005) saturation pressure over liquid water [Pa]."""

    temperature = jnp.asarray(temperature)
    log_t = jnp.log(temperature)
    transition = jnp.tanh(0.0415 * (temperature - 218.8))
    log_pressure = (
        54.842763
        - 6763.22 / temperature
        - 4.210 * log_t
        + 0.000367 * temperature
        + transition
        * (
            53.878
            - 1331.22 / temperature
            - 9.44523 * log_t
            + 0.014025 * temperature
        )
    )
    return jnp.exp(log_pressure)


def saturation_vapor_pressure_ice(temperature: jax.Array) -> jax.Array:
    """Murphy--Koop (2005) saturation pressure over hexagonal ice [Pa]."""

    temperature = jnp.asarray(temperature)
    return jnp.exp(
        9.550426
        - 5723.265 / temperature
        + 3.53068 * jnp.log(temperature)
        - 0.00728332 * temperature
    )


def saturation_mixing_ratio(
    temperature: jax.Array,
    pressure: jax.Array | float,
    config: CryogenicMicrophysicsConfig,
) -> jax.Array:
    """Return water-vapour saturation mixing ratio [kg/kg dry air]."""

    pressure = jnp.asarray(pressure, dtype=temperature.dtype)
    over_ice = temperature < config.freezing_temperature
    vapor_pressure = jnp.where(
        over_ice,
        saturation_vapor_pressure_ice(temperature),
        saturation_vapor_pressure_water(temperature),
    )
    vapor_pressure = jnp.minimum(vapor_pressure, 0.99 * pressure)
    epsilon = config.dry_air_gas_constant / config.water_vapor_gas_constant
    return epsilon * vapor_pressure / jnp.maximum(
        pressure - vapor_pressure,
        jnp.asarray(1.0, dtype=temperature.dtype),
    )


def saturation_adjustment(
    temperature: jax.Array,
    qv: jax.Array,
    ql: jax.Array,
    qi: jax.Array,
    config: CryogenicMicrophysicsConfig,
    pressure: jax.Array | float | None = None,
    heat_capacity: jax.Array | float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Equilibrate vapour, liquid fog, and ice fog at fixed moist enthalpy.

    Positive supersaturation condenses or deposits into the stable condensed
    phase.  Subsaturation evaporates liquid and sublimates ice.  A fixed
    iteration count accounts for the saturation-pressure change caused by
    latent heating without introducing a data-dependent JAX loop.
    """

    pressure_value = config.pressure if pressure is None else pressure
    cp = jnp.asarray(
        (
            config.dry_air_heat_capacity
            if heat_capacity is None
            else heat_capacity
        ),
        dtype=temperature.dtype,
    )
    lv = jnp.asarray(config.water_vapor_latent_heat, dtype=temperature.dtype)
    ls = jnp.asarray(config.ice_sublimation_latent_heat, dtype=temperature.dtype)
    freezing = jnp.asarray(config.freezing_temperature, dtype=temperature.dtype)

    qv = jnp.maximum(qv, 0.0)
    ql = jnp.maximum(ql, 0.0)
    qi = jnp.maximum(qi, 0.0)
    total_water = qv + ql + qi
    enthalpy = cp * temperature + lv * qv - (
        ls - lv
    ) * qi

    def phases_at(temp):
        vapor = jnp.minimum(
            total_water,
            saturation_mixing_ratio(temp, pressure_value, config),
        )
        condensate = jnp.maximum(total_water - vapor, 0.0)
        cold = temp < freezing
        liquid = jnp.where(cold, 0.0, condensate)
        ice = jnp.where(cold, condensate, 0.0)
        modeled_enthalpy = cp * temp + lv * vapor - (
            ls - lv
        ) * ice
        return vapor, liquid, ice, modeled_enthalpy

    # Solve the monotone fixed-enthalpy saturation problem. The former
    # condense/evaporate iteration could form condensate, warm past the new
    # saturation point, evaporate it all, and return unchanged after every
    # even iteration. Bisection cannot enter that two-cycle.
    #
    # The bracket is absolute rather than a window around the incoming
    # temperature: a strong cryogenic source can pull a cell far colder than
    # any fixed offset would allow, and a bracket that does not contain the
    # root silently returns its own endpoint as the answer. The lower bound
    # sits below the nitrogen boiling point so LN2-cooled air is bracketed
    # rather than clipped. Bisection cost is fixed by the iteration count, so
    # a wide bracket costs nothing but a few more halvings.
    lower = jnp.full_like(temperature, CRYOGENIC_TEMPERATURE_FLOOR)
    upper = jnp.full_like(temperature, CRYOGENIC_TEMPERATURE_CEILING)

    def bisect(_, bounds):
        low, high = bounds
        mid = 0.5 * (low + high)
        residual = phases_at(mid)[3] - enthalpy
        return (
            jnp.where(residual <= 0.0, mid, low),
            jnp.where(residual > 0.0, mid, high),
        )

    lower, upper = jax.lax.fori_loop(
        0,
        4 * config.saturation_iterations,
        bisect,
        (lower, upper),
    )
    adjusted_temperature = 0.5 * (lower + upper)
    vapor, liquid, ice, _ = phases_at(adjusted_temperature)

    # At the freezing point, a liquid/ice mixture spans the fusion-enthalpy
    # discontinuity and is the conservative equilibrium state.
    freezing_vapor = jnp.minimum(
        total_water,
        saturation_mixing_ratio(freezing, pressure_value, config),
    )
    freezing_condensate = jnp.maximum(total_water - freezing_vapor, 0.0)
    liquid_enthalpy = cp * freezing + lv * freezing_vapor
    ice_enthalpy = liquid_enthalpy - (ls - lv) * freezing_condensate
    mixed = (
        (freezing_condensate > 0.0)
        & (enthalpy >= ice_enthalpy)
        & (enthalpy <= liquid_enthalpy)
    )
    mixed_ice = jnp.clip(
        (liquid_enthalpy - enthalpy)
        / jnp.maximum(
            ls - lv,
            jnp.asarray(jnp.finfo(temperature.dtype).tiny),
        ),
        0.0,
        freezing_condensate,
    )
    adjusted_temperature = jnp.where(mixed, freezing, adjusted_temperature)
    vapor = jnp.where(mixed, freezing_vapor, vapor)
    ice = jnp.where(mixed, mixed_ice, ice)
    liquid = jnp.where(mixed, freezing_condensate - mixed_ice, liquid)
    return adjusted_temperature, vapor, liquid, ice


def advance_fog_microphysics(
    temperature: jax.Array,
    qv: jax.Array,
    ql: jax.Array,
    qi: jax.Array,
    dt: float,
    config: CryogenicMicrophysicsConfig,
    pressure: jax.Array | float | None = None,
    heat_capacity: jax.Array | float | None = None,
) -> FogMicrophysicsUpdate:
    """Vapour/fog/ice exchange via the enthalpy-conserving equilibrium.

    Water-phase change is treated as fast relative to the LES timestep (valid
    for typical LES dt on the order of milliseconds), so the state jumps
    directly to the `saturation_adjustment` equilibrium every call instead of
    relaxing toward it. Relaxing `temperature` and `qv` independently by the
    same exponential factor does not preserve `qv <= qsat(T)` at intermediate
    steps, because `qsat(T)` is a steeply convex function of temperature: a
    state partway between a supersaturated point and its correctly-saturated
    equilibrium can still read as strongly supersaturated. `dt` is accepted
    for interface compatibility but is otherwise unused.

    This intentionally drops any nucleation-delay physics (how long it takes
    supersaturated vapour to actually nucleate into droplets/crystals) in
    favour of getting the bulk sensible/latent heat budget - and hence the
    buoyancy it drives - right without depending on an under-resolved
    relaxation timescale.
    """

    del dt
    initial_qv = jnp.maximum(qv, 0.0)
    initial_ql = jnp.maximum(ql, 0.0)
    initial_qi = jnp.maximum(qi, 0.0)
    temp, vapor, liquid, ice = saturation_adjustment(
        temperature,
        initial_qv,
        initial_ql,
        initial_qi,
        config,
        pressure,
        heat_capacity,
    )

    vapor_change = vapor - initial_qv
    frozen = jnp.maximum(ice - initial_qi, 0.0)
    melted = jnp.maximum(initial_qi - ice, 0.0)
    return FogMicrophysicsUpdate(
        temperature=temp,
        qv=vapor,
        ql=jnp.maximum(liquid, 0.0),
        qi=jnp.maximum(ice, 0.0),
        condensed_or_deposited=jnp.maximum(-vapor_change, 0.0),
        evaporated_or_sublimated=jnp.maximum(vapor_change, 0.0),
        frozen=frozen,
        melted=melted,
    )


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
    new_diameter = (
        6.0
        * new_mass
        / (jnp.pi * config.liquid_nitrogen_density)
    ) ** (1.0 / 3.0)
    new_diameter = jnp.where(new_mass > 0.0, new_diameter, 0.0)
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
