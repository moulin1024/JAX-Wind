#!/usr/bin/env python3
"""No-feedback size-resolved flight diagnostic; not a tunnel prediction.

Use literature initial conditions in a uniform, frozen carrier to test the
entrained/isothermal assumptions without fitting outlet-temperature data.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.config.document import load_case
from jaxwind.config.moisture import load_moisture
from jaxwind.physics.moisture import (
    WaterDropletProperties,
    advance_water_droplet,
    water_droplet_transfer_coefficients,
)
from jaxwind.simulation.water_spray_benchmark import inlet_mixing_ratio
from jaxwind.water_spray import advance_water_droplet_motion


def diagnose(case, dt, size_count=20, azimuth_count=32):
    jax.config.update("jax_enable_x64", True)
    document = load_case(case).document
    reference = document["case"]["reference"]
    moisture, _ = load_moisture(document["physics"])
    config, props = moisture.thermodynamics, WaterDropletProperties()
    nodes, weights = np.polynomial.legendre.leggauss(size_count)
    lo, hi = reference["minimum_diameter_m"], reference["maximum_diameter_m"]
    sizes = lo + 0.5 * (nodes + 1) * (hi - lo)
    scale, n = reference["rosin_rammler_scale_m"], reference["rosin_rammler_spread"]
    mass_weights = (
        weights
        * n
        / scale
        * (sizes / scale) ** (n - 1)
        * np.exp(-((sizes / scale) ** n))
    )
    mass_weights /= mass_weights.sum()
    d = jnp.asarray(np.repeat(sizes, azimuth_count))
    mass = config.water_density * jnp.pi * d**3 / 6
    weights_per_mass = (
        jnp.asarray(np.repeat(mass_weights / azimuth_count, azimuth_count)) / mass
    )
    angle = jnp.asarray(
        np.tile(
            2 * np.pi * (np.arange(azimuth_count) + 0.5) / azimuth_count, size_count
        )
    )
    theta = np.deg2rad(reference["cone_half_angle_degrees"])
    speed = 0.9 * np.sqrt(
        2 * reference["water_gauge_pressure_pa"] / config.water_density
    )
    radius = reference["nozzle_diameter_m"] / 2
    length, width, height = document["mesh"]["lengths_m"]
    initial_position = jnp.stack(
        (
            jnp.zeros_like(angle),
            width / 2 + radius * jnp.cos(angle),
            height / 2 + radius * jnp.sin(angle),
        )
    )
    initial_velocity = speed * jnp.stack(
        (
            jnp.full_like(angle, np.cos(theta)),
            np.sin(theta) * jnp.cos(angle),
            np.sin(theta) * jnp.sin(angle),
        )
    )
    u = document["physics"]["flow"]["streamwise_velocity_m_s"]
    gas_velocity = jnp.stack(
        (jnp.full_like(angle, u), jnp.zeros_like(angle), jnp.zeros_like(angle))
    )
    gas_temperature = reference["dry_bulb_c"] + 273.15
    gas_vapor = inlet_mixing_ratio(reference, config)
    initial_temperature = jnp.full_like(angle, reference["water_inlet_c"] + 273.15)
    zero = jnp.zeros_like(angle)
    # State: position, velocity, mass, T, flight time, gas sensible loss, wall hit.
    initial = (
        initial_position,
        initial_velocity,
        mass,
        initial_temperature,
        zero,
        zero,
        jnp.zeros_like(angle, dtype=bool),
    )

    def step(_, state):
        position, velocity, m, t, time, heat, hit = state
        active = (position[0] < length - 1e-10) & (m > 0)
        h = jnp.where(
            active,
            jnp.minimum(dt, (length - position[0]) / jnp.maximum(velocity[0], 0.1)),
            0,
        )
        diameter = jnp.cbrt(6 * m / (jnp.pi * config.water_density))
        new_velocity, displacement, _ = advance_water_droplet_motion(
            velocity, gas_velocity, diameter, h, config, props
        )
        update = advance_water_droplet(
            m,
            t,
            gas_temperature,
            gas_vapor,
            jnp.linalg.norm(velocity - gas_velocity, axis=0),
            h,
            config,
            props,
        )
        candidate = position + displacement
        hit_y = (candidate[1] < 0) | (candidate[1] > width)
        hit_z = (candidate[2] < 0) | (candidate[2] > height)
        candidate = (
            candidate.at[1]
            .set(jnp.clip(candidate[1], 0, width))
            .at[2]
            .set(jnp.clip(candidate[2], 0, height))
        )
        new_velocity = (
            new_velocity.at[1]
            .set(jnp.where(hit_y, 0, new_velocity[1]))
            .at[2]
            .set(jnp.where(hit_z, 0, new_velocity[2]))
        )
        return (
            candidate,
            new_velocity,
            update.mass,
            update.temperature,
            time + h,
            heat + update.gas_sensible_energy_loss,
            hit | hit_y | hit_z,
        )

    position, velocity, remaining, temperature, flight, heat, hit = jax.jit(
        lambda: jax.lax.fori_loop(0, int(np.ceil(2 / dt)), step, initial)
    )()
    weighted = lambda a: float(jnp.sum(a * weights_per_mass))
    initial_re, initial_nu, initial_sh = water_droplet_transfer_coefficients(
        d, jnp.linalg.norm(initial_velocity - gas_velocity, axis=0), config, props
    )
    final_water = weighted(remaining)
    return {
        "scope": "Frozen uniform air, no carrier feedback: mechanism diagnostic, not a prediction of tunnel outlet cooling.",
        "dt_s": dt,
        "size_quadrature_points": size_count,
        "azimuth_points": azimuth_count,
        "water_speed_m_s": float(speed),
        "air_speed_m_s": u,
        "inlet_vapor_kg_kg_dry_air": gas_vapor,
        "nominal_entrained_flight_time_s": length / u,
        "mass_weighted_flight_time_s": weighted(mass * flight),
        "flight_time_range_s": [float(jnp.min(flight)), float(jnp.max(flight))],
        "initial_reynolds_range": [
            float(jnp.min(initial_re)),
            float(jnp.max(initial_re)),
        ],
        "initial_sherwood_range": [
            float(jnp.min(initial_sh)),
            float(jnp.max(initial_sh)),
        ],
        "evaporated_fraction": 1 - final_water,
        "outlet_liquid_mass_weighted_temperature_c": weighted(
            remaining * (temperature - 273.15)
        )
        / final_water,
        "outlet_liquid_temperature_range_c": [
            float(jnp.min(temperature) - 273.15),
            float(jnp.max(temperature) - 273.15),
        ],
        "gas_sensible_loss_j_per_kg_injected": weighted(heat),
        "latent_enthalpy_transferred_j_per_kg_injected": (1 - final_water)
        * config.water_vapor_latent_heat,
        "initial_mass_fraction_hitting_wall": weighted(mass * hit),
        "initial_mass_fraction_not_exited": weighted(
            mass * (position[0] < length - 1e-10)
        ),
        "thermal_inertia_enthalpy_residual_j_per_kg_injected": weighted(
            remaining
            * props.liquid_heat_capacity
            * (temperature - config.freezing_temperature)
            - mass
            * props.liquid_heat_capacity
            * (initial_temperature - config.freezing_temperature)
            - heat
        )
        + (1 - final_water) * config.water_vapor_latent_heat,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--case", type=Path, default=Path("cases/WaterSprayMontazeri2015/coarse.toml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cases/WaterSprayMontazeri2015/investigation/droplet_flight.json"),
    )
    args = parser.parse_args()
    result = {"runs": [diagnose(args.case, dt) for dt in [0.001, 0.0005, 0.00025]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
