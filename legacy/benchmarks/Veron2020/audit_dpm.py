#!/usr/bin/env python3
"""Audit the parcel integrator against an independent Veron (2020) ODE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "jax"))

from run_single_drop import simulate
from wireles_jax import SprayDPMConfig


AIR_TEMPERATURE = 291.15
DROP_TEMPERATURE = 293.15
AIR_SPEED = 15.0
RELATIVE_HUMIDITY = 0.75
PRESSURE = 1.0e5
INITIAL_RADIUS = 100.0e-6


def saturation_vapor_pressure(temperature: float) -> float:
    temperature_c = temperature - 273.15
    return 611.2 * np.exp(17.67 * temperature_c / (temperature_c + 243.5))


def osmotic_coefficient(molality: float, *, dynamic: bool) -> float:
    if not dynamic:
        return 0.93
    m = np.clip(molality, 0.0, 6.0)
    return float(
        0.9270
        - 2.164e-2 * m
        + 3.486e-2 * m**2
        - 5.956e-3 * m**3
        + 3.911e-4 * m**4
    )


def initial_material(config: SprayDPMConfig) -> tuple[float, float, float]:
    volume = 4.0 * np.pi * INITIAL_RADIUS**3 / 3.0
    mass = config.liquid_density * volume
    salt = config.salinity_mass_fraction * mass
    residual_volume = max(
        volume - (mass - salt) / config.water_density, 0.0
    )
    return mass, salt, residual_volume


def radius_from_mass(
    mass: float,
    salt: float,
    residual_volume: float,
    config: SprayDPMConfig,
) -> float:
    volume = residual_volume + max(mass - salt, 0.0) / config.water_density
    return (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)


def surface_relative_humidity(
    radius: float,
    mass: float,
    salt: float,
    temperature: float,
    config: SprayDPMConfig,
    *,
    dynamic_osmotic: bool,
) -> float:
    water = max(mass - salt, 1.0e-30)
    molality = salt / (config.salt_molar_mass * water)
    phi = osmotic_coefficient(molality, dynamic=dynamic_osmotic)
    solute = (
        config.salt_vant_hoff_factor
        * phi
        * molality
        * config.water_molar_mass
    )
    kelvin = (
        2.0
        * config.water_molar_mass
        * config.surface_tension
        / (
            config.universal_gas_constant
            * config.water_density
            * radius
            * temperature
        )
    )
    thermal = (
        config.latent_heat
        * config.water_molar_mass
        / config.universal_gas_constant
        * (1.0 / AIR_TEMPERATURE - 1.0 / temperature)
    )
    return AIR_TEMPERATURE / temperature * np.exp(thermal + kelvin - solute)


def reference_rhs(
    _time: float,
    state: np.ndarray,
    config: SprayDPMConfig,
    salt: float,
    residual_volume: float,
) -> np.ndarray:
    velocity, vertical_velocity, mass, temperature = state
    radius = radius_from_mass(mass, salt, residual_volume, config)
    diameter = 2.0 * radius
    volume = 4.0 * np.pi * radius**3 / 3.0
    density = mass / volume
    slip = AIR_SPEED - velocity
    slip_magnitude = np.hypot(slip, vertical_velocity)
    reynolds = (
        config.air_density
        * slip_magnitude
        * diameter
        / config.air_dynamic_viscosity
    )
    turbulent_drag = 0.42 / (
        1.0 + 42500.0 / max(reynolds, 1.0e-12) ** 1.16
    )
    drag_factor = (
        1.0
        + 0.15 * max(reynolds, 1.0e-12) ** 0.687
        + turbulent_drag * reynolds / 24.0
    )
    drag_rate = (
        18.0
        * config.air_dynamic_viscosity
        * drag_factor
        / (density * diameter**2)
    )
    ventilation = 1.0 + np.sqrt(reynolds) / 4.0
    vapor_denominator = (
        radius / (radius + config.vapor_jump_length)
        + config.vapor_diffusivity
        / (radius * config.vapor_accommodation_coefficient)
        * np.sqrt(
            2.0
            * np.pi
            * config.water_molar_mass
            / (config.universal_gas_constant * AIR_TEMPERATURE)
        )
    )
    vapor_diffusivity = ventilation * config.vapor_diffusivity / vapor_denominator
    mass_coefficient = (
        4.0
        * np.pi
        * radius
        * vapor_diffusivity
        * config.water_molar_mass
        * saturation_vapor_pressure(AIR_TEMPERATURE)
        / (config.universal_gas_constant * AIR_TEMPERATURE)
    )
    evaporation = mass_coefficient * (
        surface_relative_humidity(
            radius,
            mass,
            salt,
            temperature,
            config,
            dynamic_osmotic=True,
        )
        - RELATIVE_HUMIDITY
    )
    thermal_denominator = (
        radius / (radius + config.thermal_jump_length)
        + config.air_thermal_conductivity
        / (
            radius
            * config.thermal_accommodation_coefficient
            * config.air_density
            * config.air_heat_capacity
        )
        * np.sqrt(
            2.0
            * np.pi
            * config.dry_air_molar_mass
            / (config.universal_gas_constant * AIR_TEMPERATURE)
        )
    )
    thermal_conductivity = (
        ventilation * config.air_thermal_conductivity / thermal_denominator
    )
    heat_conductance = 4.0 * np.pi * radius * thermal_conductivity
    temperature_rate = (
        heat_conductance * (AIR_TEMPERATURE - temperature)
        - config.latent_heat * evaporation
    ) / (mass * config.liquid_heat_capacity)
    buoyancy_corrected_gravity = 9.81 * (
        1.0 - config.air_density / density
    )
    return np.asarray(
        (
            drag_rate * slip,
            -drag_rate * vertical_velocity - buoyancy_corrected_gravity,
            -evaporation,
            temperature_rate,
        )
    )


def equilibrium_radius(
    config: SprayDPMConfig,
    salt: float,
    residual_volume: float,
    *,
    dynamic_osmotic: bool,
) -> float:
    dry_radius = (3.0 * residual_volume / (4.0 * np.pi)) ** (1.0 / 3.0)

    def residual(radius: float) -> float:
        volume = 4.0 * np.pi * radius**3 / 3.0
        mass = salt + config.water_density * (volume - residual_volume)
        return surface_relative_humidity(
            radius,
            mass,
            salt,
            AIR_TEMPERATURE,
            config,
            dynamic_osmotic=dynamic_osmotic,
        ) - RELATIVE_HUMIDITY

    return brentq(residual, max(1.0001 * dry_radius, 1.0e-9), INITIAL_RADIUS)


def interpolate(data: dict[str, np.ndarray], name: str, times: np.ndarray) -> np.ndarray:
    return np.interp(times, data["time"], data[name])


def digitize_reference_radius(times: np.ndarray) -> np.ndarray:
    """Read the red curve at fixed log-time columns in the supplied raster."""
    image = np.asarray(Image.open(ROOT / "doc" / "reference" / "fig1.png").convert("RGB"))
    left, right, top, bottom = 235.0, 957.0, 58.0, 564.0
    red = (
        (image[:, :, 0] > 140)
        & (image[:, :, 0] > 1.5 * image[:, :, 1])
        & (image[:, :, 0] > 1.5 * image[:, :, 2])
    )
    values = []
    for time in times:
        column = int(
            round(left + (np.log10(time) + 5.0) / 10.0 * (right - left))
        )
        rows = np.flatnonzero(red[:, column])
        radius = 110.0 - (rows - top) / (bottom - top) * 70.0
        radius = radius[(radius >= 40.0) & (radius <= 110.0)]
        if radius.size == 0:
            raise RuntimeError(f"could not digitize reference radius at t={time}")
        values.append(float(np.median(radius)))
    return np.asarray(values)


def digitize_reference_temperature(times: np.ndarray) -> np.ndarray:
    image = np.asarray(Image.open(ROOT / "doc" / "reference" / "fig1.png").convert("RGB"))
    left, right, top, bottom = 235.0, 957.0, 58.0, 564.0
    blue = (
        (image[:, :, 2] > 100)
        & (image[:, :, 2] > 1.4 * image[:, :, 0])
        & (image[:, :, 2] > 1.4 * image[:, :, 1])
    )
    values = []
    for time in times:
        column = int(round(left + (np.log10(time) + 5.0) / 10.0 * (right - left)))
        rows = np.flatnonzero(blue[:, column])
        temperature = 21.0 - (rows - top) / (bottom - top) * 7.0
        temperature = temperature[(temperature >= 14.0) & (temperature <= 21.0)]
        if temperature.size == 0:
            raise RuntimeError(f"could not digitize temperature at t={time}")
        values.append(float(np.median(temperature)))
    return np.asarray(values)


def digitize_reference_velocity(times: np.ndarray) -> np.ndarray:
    image = np.asarray(Image.open(ROOT / "doc" / "reference" / "fig1.png").convert("RGB"))
    left, right, top, bottom = 235.0, 957.0, 58.0, 564.0
    black = np.max(image, axis=2) < 80
    values = []
    for time in times:
        column = int(round(left + (np.log10(time) + 5.0) / 10.0 * (right - left)))
        rows = np.flatnonzero(black[:, column])
        velocity = 16.0 - (rows - top) / (bottom - top) * 18.0
        velocity = velocity[(velocity >= -0.5) & (velocity <= 15.5)]
        if velocity.size == 0:
            raise RuntimeError(f"could not digitize velocity at t={time}")
        values.append(float(np.median(velocity)))
    return np.asarray(values)


def audit(output: Path) -> dict[str, object]:
    config = SprayDPMConfig(
        max_parcels=1,
        initial_parcels=1,
        injection_z=0.0,
        initial_diameter=2.0 * INITIAL_RADIUS,
        initial_temperature=DROP_TEMPERATURE,
        liquid_density=1025.0,
        salinity_mass_fraction=0.035,
        thermodynamic_transfer_model="veron2020",
        osmotic_coefficient_model="andreas1989",
        liquid_emissivity=0.0,
    )
    initial_mass, salt, residual_volume = initial_material(config)
    sample_times = np.asarray(
        (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0, 5000.0)
    )
    solution = solve_ivp(
        reference_rhs,
        (0.0, float(sample_times[-1])),
        (0.0, 0.0, initial_mass, DROP_TEMPERATURE),
        args=(config, salt, residual_volume),
        method="DOP853",
        rtol=2.0e-11,
        atol=(1.0e-12, 1.0e-12, 1.0e-24, 1.0e-11),
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    reference = solution.sol(sample_times)
    reference_radius = np.asarray(
        [
            radius_from_mass(value, salt, residual_volume, config)
            for value in reference[2]
        ]
    )
    early = simulate(
        1.0,
        0.0001,
        salinity_mass_fraction=0.035,
        liquid_density=1025.0,
        gravity=9.81,
        drag_correction_model="instantaneous_clift_gauvin",
        ventilation_correction_enabled=True,
    )
    late = simulate(
        5000.0,
        0.01,
        salinity_mass_fraction=0.035,
        liquid_density=1025.0,
        gravity=9.81,
        drag_correction_model="instantaneous_clift_gauvin",
        ventilation_correction_enabled=True,
    )
    production = {
        "u": np.where(
            sample_times <= 1.0,
            interpolate(early, "u", sample_times),
            interpolate(late, "u", sample_times),
        ),
        "temperature": np.where(
            sample_times <= 1.0,
            interpolate(early, "temperature", sample_times),
            interpolate(late, "temperature", sample_times),
        ),
        "w": np.where(
            sample_times <= 1.0,
            interpolate(early, "w", sample_times),
            interpolate(late, "w", sample_times),
        ),
        "radius": np.where(
            sample_times <= 1.0,
            interpolate(early, "radius", sample_times),
            interpolate(late, "radius", sample_times),
        ),
    }
    dynamic_equilibrium = equilibrium_radius(
        config, salt, residual_volume, dynamic_osmotic=True
    )
    constant_equilibrium = equilibrium_radius(
        config, salt, residual_volume, dynamic_osmotic=False
    )
    figure_times = np.asarray((0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 200.0, 500.0))
    reference_figure_radius = digitize_reference_radius(figure_times)
    temperature_times = np.asarray(
        (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 3.0, 10.0, 30.0, 100.0, 200.0, 500.0)
    )
    reference_figure_temperature = digitize_reference_temperature(
        temperature_times
    )
    velocity_times = np.asarray(
        (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    )
    reference_figure_velocity = digitize_reference_velocity(velocity_times)
    figure_early = simulate(
        1.0,
        0.0001,
        salinity_mass_fraction=0.035,
        liquid_density=1025.0,
        gravity=9.81,
        drag_correction_model="terminal_settling",
        ventilation_correction_enabled=False,
    )
    figure_case = simulate(
        500.0,
        0.01,
        salinity_mass_fraction=0.035,
        liquid_density=1025.0,
        gravity=9.81,
        drag_correction_model="terminal_settling",
        ventilation_correction_enabled=False,
    )
    figure_radius = 1.0e6 * interpolate(
        figure_case, "radius", figure_times
    )
    figure_temperature = np.where(
        temperature_times <= 1.0,
        interpolate(figure_early, "temperature", temperature_times),
        interpolate(figure_case, "temperature", temperature_times),
    ) - 273.15
    figure_velocity = interpolate(figure_early, "u", velocity_times)
    report: dict[str, object] = {
        "models": {
            "transfer": "Veron2020 equations 2.7, 2.12, 2.13",
            "osmotic": "Andreas1989 equation 27, clipped at 6 mol/kg",
        },
        "equilibrium_radius_um": {
            "constant_phi_0p93": 1.0e6 * constant_equilibrium,
            "dynamic_phi": 1.0e6 * dynamic_equilibrium,
            "production_at_5000s": 1.0e6 * production["radius"][-1],
            "paper_reported_approx": 46.0,
        },
        "production_vs_independent_ode": {
            "max_velocity_error_mps": float(
                np.max(np.abs(production["u"] - reference[0]))
            ),
            "max_vertical_velocity_error_mps": float(
                np.max(np.abs(production["w"] - reference[1]))
            ),
            "max_temperature_error_K": float(
                np.max(np.abs(production["temperature"] - reference[3]))
            ),
            "max_radius_error_um": float(
                1.0e6 * np.max(np.abs(production["radius"] - reference_radius))
            ),
        },
        "figure1_radius_raster_comparison": {
            "interpretation": (
                "figure-compatible terminal-settling response time and no "
                "Reynolds ventilation; inferred from the digitized curves"
            ),
            "times_s": figure_times.tolist(),
            "paper_radius_um": reference_figure_radius.tolist(),
            "production_radius_um": figure_radius.tolist(),
            "rmse_um": float(
                np.sqrt(np.mean((figure_radius - reference_figure_radius) ** 2))
            ),
            "max_abs_error_um": float(
                np.max(np.abs(figure_radius - reference_figure_radius))
            ),
        },
        "figure1_temperature_raster_comparison": {
            "times_s": temperature_times.tolist(),
            "rmse_K": float(
                np.sqrt(
                    np.mean(
                        (figure_temperature - reference_figure_temperature) ** 2
                    )
                )
            ),
            "max_abs_error_K": float(
                np.max(
                    np.abs(figure_temperature - reference_figure_temperature)
                )
            ),
        },
        "figure1_velocity_raster_comparison": {
            "times_s": velocity_times.tolist(),
            "paper_velocity_mps": reference_figure_velocity.tolist(),
            "production_velocity_mps": figure_velocity.tolist(),
            "rmse_mps": float(
                np.sqrt(
                    np.mean((figure_velocity - reference_figure_velocity) ** 2)
                )
            ),
            "max_abs_error_mps": float(
                np.max(np.abs(figure_velocity - reference_figure_velocity))
            ),
        },
        "sample_times_s": sample_times.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "dpm_audit.json",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2))


if __name__ == "__main__":
    main()
