"""Shared humidity, cloud liquid/ice, and finite-rate spray thermodynamics.

Mixing ratios are kg/kg dry air. The dilute enthalpy convention is
cp_d * T + L_v * qv - (L_s - L_v) * qi; condensate sensible heat is neglected.
Nitrogen and atmospheric moisture use the same saturation adjustment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import NamedTuple

import jax
import jax.numpy as jnp

# Preserve the existing cryogenic bracket (ice fit extrapolated below 110 K).
SATURATION_TEMPERATURE_FLOOR = 50.0
SATURATION_TEMPERATURE_CEILING = 450.0


@dataclass(frozen=True)
class MoistureConfig:
    pressure: float = 100_000.0
    dry_air_density: float = 1.225
    dry_air_heat_capacity: float = 1005.0
    dry_air_gas_constant: float = 287.05
    water_vapor_gas_constant: float = 461.5
    water_vapor_latent_heat: float = 2.50e6
    ice_sublimation_latent_heat: float = 2.834e6
    water_fusion_latent_heat: float = 3.34e5
    freezing_temperature: float = 273.15
    saturation_iterations: int = 6
    water_density: float = 997.0
    vapor_diffusivity: float = 2.5e-5

    def __post_init__(self):
        if any(
            not math.isfinite(getattr(self, f.name)) or getattr(self, f.name) <= 0
            for f in fields(self)
        ):
            raise ValueError("moisture properties must be finite and positive")
        if type(self.saturation_iterations) is not int:
            raise ValueError("saturation_iterations must be a positive integer")
        if self.ice_sublimation_latent_heat <= self.water_vapor_latent_heat:
            raise ValueError("sublimation latent heat must exceed vaporization heat")


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
        * (53.878 - 1331.22 / temperature - 9.44523 * log_t + 0.014025 * temperature)
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
    config: MoistureConfig,
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
    return (
        epsilon
        * vapor_pressure
        / jnp.maximum(
            pressure - vapor_pressure,
            jnp.asarray(1.0, dtype=temperature.dtype),
        )
    )


def saturation_adjustment(
    temperature: jax.Array,
    qv: jax.Array,
    ql: jax.Array,
    qi: jax.Array,
    config: MoistureConfig,
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
        (config.dry_air_heat_capacity if heat_capacity is None else heat_capacity),
        dtype=temperature.dtype,
    )
    lv = jnp.asarray(config.water_vapor_latent_heat, dtype=temperature.dtype)
    ls = jnp.asarray(config.ice_sublimation_latent_heat, dtype=temperature.dtype)
    freezing = jnp.asarray(config.freezing_temperature, dtype=temperature.dtype)

    qv = jnp.maximum(qv, 0.0)
    ql = jnp.maximum(ql, 0.0)
    qi = jnp.maximum(qi, 0.0)
    total_water = qv + ql + qi
    enthalpy = cp * temperature + lv * qv - (ls - lv) * qi

    def phases_at(temp):
        vapor = jnp.minimum(
            total_water,
            saturation_mixing_ratio(temp, pressure_value, config),
        )
        condensate = jnp.maximum(total_water - vapor, 0.0)
        cold = temp < freezing
        liquid = jnp.where(cold, 0.0, condensate)
        ice = jnp.where(cold, condensate, 0.0)
        modeled_enthalpy = cp * temp + lv * vapor - (ls - lv) * ice
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
    lower = jnp.full_like(temperature, SATURATION_TEMPERATURE_FLOOR)
    upper = jnp.full_like(temperature, SATURATION_TEMPERATURE_CEILING)

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
    # Leave an already unsaturated, cloud-free gas exactly unchanged. This
    # avoids accumulation of bisection roundoff in long dry/zero-spray runs.
    unchanged = (
        (ql == 0)
        & (qi == 0)
        & (qv <= saturation_mixing_ratio(temperature, pressure_value, config))
    )
    return (
        jnp.where(unchanged, temperature, adjusted_temperature),
        jnp.where(unchanged, qv, vapor),
        jnp.where(unchanged, ql, liquid),
        jnp.where(unchanged, qi, ice),
    )


def advance_fog_microphysics(
    temperature: jax.Array,
    qv: jax.Array,
    ql: jax.Array,
    qi: jax.Array,
    dt: float,
    config: MoistureConfig,
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


class MoistureState(NamedTuple):
    """Cloud reservoirs and a distinct entrained spray population.

    Number is droplets/kg dry air. Spray diameter is recovered from mass and
    number, so evaporation reduces size rather than removing a fixed fraction
    of an unchanging droplet population.
    """

    vapor: jax.Array
    cloud_liquid: jax.Array
    cloud_ice: jax.Array
    spray_liquid: jax.Array
    spray_number: jax.Array


def advance_moisture(temperature, water: MoistureState, dt, config: MoistureConfig):
    """Couple finite-rate warm spray evaporation to shared cloud adjustment.

    Dilute, thermally equilibrated, entrained droplets use the diffusion limit
    Sh=2 (zero slip), with Stefan's logarithmic mass-transfer driving force.
    A backward-Euler cell solve couples all spray water to the same gas heat
    and humidity inventory. It cannot exhaust liquid, cross saturation, or
    cool the warm spray below freezing. Inertial/thermal droplet lag, spray
    freezing and sedimentation are outside this reduced closure.
    """
    temperature, vapor, liquid, ice = saturation_adjustment(
        temperature, water.vapor, water.cloud_liquid, water.cloud_ice, config
    )
    spray = water.spray_liquid
    number = water.spray_number
    latent_over_cp = config.water_vapor_latent_heat / config.dry_air_heat_capacity
    upper = jnp.minimum(
        spray,
        jnp.maximum((temperature - config.freezing_temperature) / latent_over_cp, 0.0),
    )

    def rate(evaporated):
        temp = temperature - latent_over_cp * evaporated
        qv = vapor + evaporated
        qs = saturation_mixing_ratio(temp, config.pressure, config)
        # Y = q/(1+q); log((1-Yv)/(1-Ys)) = log((1+qs)/(1+qv)).
        driving = jnp.maximum(jnp.log1p(qs) - jnp.log1p(qv), 0.0)
        diameter = jnp.cbrt(
            6.0
            * jnp.maximum(spray - evaporated, 0.0)
            / (jnp.pi * config.water_density * jnp.maximum(number, 1.0e-30))
        )
        return (
            number
            * 2.0
            * jnp.pi
            * diameter
            * config.dry_air_density
            * config.vapor_diffusivity
            * driving
        )

    def bisect(_, bounds):
        low, high = bounds
        middle = 0.5 * (low + high)
        residual = middle - dt * rate(middle)
        return (
            jnp.where(residual <= 0.0, middle, low),
            jnp.where(residual > 0.0, middle, high),
        )

    low, _ = jax.lax.fori_loop(0, 32, bisect, (jnp.zeros_like(spray), upper))
    # Low is conservative at the saturation/energy boundary and exactly zero
    # when there is no driving force or dt=0.
    evaporated = low
    spray = spray - evaporated
    temperature = temperature - latent_over_cp * evaporated
    vapor = vapor + evaporated
    temperature, vapor, liquid, ice = saturation_adjustment(
        temperature, vapor, liquid, ice, config
    )
    return temperature, MoistureState(
        vapor, liquid, ice, spray, jnp.where(spray > 0.0, number, 0.0)
    )


def virtual_temperature_offset(
    water: MoistureState, reference_temperature, ambient_vapor, config: MoistureConfig
):
    """Linear moist Boussinesq correction in K, including condensate loading."""
    epsilon = config.water_vapor_gas_constant / config.dry_air_gas_constant - 1.0
    return reference_temperature * (
        epsilon * (water.vapor - ambient_vapor)
        - water.cloud_liquid
        - water.cloud_ice
        - water.spray_liquid
    )


@dataclass(frozen=True)
class WaterDropletProperties:
    """Warm-water transport properties for unresolved inertial droplets.

    Constant properties near 300 K. Shared MoistureConfig supplies saturation,
    vapor diffusion, density and latent enthalpy. Freezing is not modeled here.
    """

    liquid_heat_capacity: float = 4182.0
    air_dynamic_viscosity: float = 1.9e-5
    air_thermal_conductivity: float = 0.0265

    def __post_init__(self):
        if any(
            not math.isfinite(getattr(self, f.name)) or getattr(self, f.name) <= 0
            for f in fields(self)
        ):
            raise ValueError("droplet transport properties must be finite and positive")


class WaterDropletUpdate(NamedTuple):
    mass: jax.Array
    diameter: jax.Array
    temperature: jax.Array
    evaporated_mass: jax.Array
    gas_sensible_energy_loss: jax.Array


def water_droplet_transfer_coefficients(diameter, slip_speed, config, properties):
    """Return Re, Nu and Sh with the Ranz--Marshall sphere correlations.

    Nu=2+0.6 sqrt(Re) Pr^(1/3), Sh=2+0.6 sqrt(Re) Sc^(1/3).
    This is a warm, low-transfer-rate closure, not an atomization model.
    """
    re = (
        config.dry_air_density
        * jnp.maximum(slip_speed, 0.0)
        * jnp.maximum(diameter, 0.0)
        / properties.air_dynamic_viscosity
    )
    pr = (
        config.dry_air_heat_capacity
        * properties.air_dynamic_viscosity
        / properties.air_thermal_conductivity
    )
    sc = properties.air_dynamic_viscosity / (
        config.dry_air_density * config.vapor_diffusivity
    )
    return (
        re,
        2 + 0.6 * jnp.sqrt(re) * pr ** (1 / 3),
        2 + 0.6 * jnp.sqrt(re) * sc ** (1 / 3),
    )


def advance_water_droplet(
    mass,
    temperature,
    gas_temperature,
    gas_vapor,
    slip_speed,
    dt,
    config,
    properties=WaterDropletProperties(),
):
    """Conservative finite-temperature evaporation into a frozen gas reservoir.

    All temperatures are K, mass is kg per droplet, gas_vapor is kg/kg dry air.
    The surface saturation pressure is evaluated at the droplet temperature.
    An implicit temperature solve couples evaporation and sensible heat, with
    transfer coefficients evaluated at the initial diameter and slip speed.
    This first-order update needs time-step convergence and carrier feedback
    when many droplets share a gas cell. It does not by itself limit depletion
    of the carrier heat inventory or prevent carrier supersaturation.

    The enthalpy convention matches shared cloud physics: h_v = L_v, with no
    vapor sensible heat, and h_l = cp_l*(T_l-T_freeze). Therefore the effective
    latent heat at T_l is L_v - cp_l*(T_l-T_freeze). Returned gas sensible loss
    plus deposited L_v*evaporated_mass closes the droplet/gas enthalpy budget.
    Negative sensible loss heats the gas. Condensation onto spray, boiling,
    freezing, breakup, collision and film corrections are outside this closure.
    Inputs must be warm (T_l and T_g >= freezing) and dt >= 0.
    """
    mass = jnp.asarray(mass)
    temperature = jnp.asarray(temperature, mass.dtype)
    diameter = jnp.cbrt(6 * mass / (jnp.pi * config.water_density))
    _, nu, sh = water_droplet_transfer_coefficients(
        diameter, slip_speed, config, properties
    )
    conductance = jnp.pi * diameter * properties.air_thermal_conductivity * nu
    transfer = (
        jnp.pi * diameter * config.dry_air_density * config.vapor_diffusivity * sh
    )
    cp = properties.liquid_heat_capacity
    lv = config.water_vapor_latent_heat
    reference = config.freezing_temperature
    initial_enthalpy = mass * cp * (temperature - reference)
    # Warm closure: do not fund evaporation by fictitious cooling below freezing.
    available = initial_enthalpy + dt * conductance * (gas_temperature - reference)
    maximum_evaporation = jnp.minimum(mass, jnp.maximum(available / lv, 0.0))

    def exchange(surface_temperature):
        qs = saturation_mixing_ratio(surface_temperature, config.pressure, config)
        driving = jnp.maximum(jnp.log1p(qs) - jnp.log1p(gas_vapor), 0.0)
        evaporated = jnp.minimum(dt * transfer * driving, maximum_evaporation)
        final_enthalpy = (mass - evaporated) * cp * (surface_temperature - reference)
        heat = dt * conductance * (gas_temperature - surface_temperature)
        return final_enthalpy - initial_enthalpy + lv * evaporated - heat, evaporated

    def bisect(_, bracket):
        low, high = bracket
        middle = 0.5 * (low + high)
        residual, _ = exchange(middle)
        return jnp.where(residual <= 0, middle, low), jnp.where(
            residual > 0, middle, high
        )

    low, high = jax.lax.fori_loop(
        0,
        40,
        bisect,
        (
            jnp.full_like(temperature, reference),
            jnp.maximum(temperature, gas_temperature),
        ),
    )
    updated_temperature = jnp.where(
        (mass > 0) & (dt > 0), 0.5 * (low + high), temperature
    )
    _, evaporated = exchange(updated_temperature)
    updated_mass = mass - evaporated
    final_enthalpy = updated_mass * cp * (updated_temperature - reference)
    # Compute from state changes so the exchange is conservative even in float32.
    heat = final_enthalpy - initial_enthalpy + lv * evaporated
    return WaterDropletUpdate(
        updated_mass,
        jnp.cbrt(6 * updated_mass / (jnp.pi * config.water_density)),
        updated_temperature,
        evaporated,
        heat,
    )
