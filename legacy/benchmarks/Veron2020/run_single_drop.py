#!/usr/bin/env python3
"""Compare fresh-water and seawater drops for Veron (2020), figure 1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "jax"))


def simulate(
    duration: float,
    dt: float,
    *,
    salinity_mass_fraction: float = 0.0,
    liquid_density: float | None = None,
    thermodynamic_transfer_model: str = "veron2020",
    osmotic_coefficient_model: str = "andreas1989",
    drag_correction_model: str = "terminal_settling",
    ventilation_correction_enabled: bool = False,
    gravity: float = 9.81,
) -> dict[str, np.ndarray]:
    import jax
    from jax import lax
    import jax.numpy as jnp

    from wireles_jax import Params, SprayDPMConfig, initialize_spray
    from wireles_jax.spray_dpm import (
        _advance_parcel_implicit,
        _parcel_rates_from_samples,
        _proposed_phase_change,
        saturation_vapor_pressure,
    )

    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive")
    nsteps = int(np.ceil(duration / dt))
    ambient_temperature = 273.15 + 18.0
    ambient_speed = 15.0
    relative_humidity = 0.75
    pressure = 100000.0
    epsilon = 287.05 / 461.5
    vapor_pressure = relative_humidity * float(
        saturation_vapor_pressure(jnp.asarray(ambient_temperature))
    )
    ambient_qv = epsilon * vapor_pressure / (pressure - vapor_pressure)

    params = Params(
        nx=2,
        ny=2,
        nz=2,
        lx=1.0,
        ly=1.0,
        lz=1.0,
        z_i=100.0,
        dt=dt / 100.0,
        thermo_enabled=True,
        moisture_enabled=True,
        theta0=ambient_temperature,
        qv0=ambient_qv,
        momentum_wall_model="free_slip",
        initial_velocity_noise=0.0,
        g=gravity,
        dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32,
        use_jit=True,
    )
    if liquid_density is None:
        liquid_density = 1025.0 if salinity_mass_fraction > 0.0 else 997.0
    config = SprayDPMConfig(
        max_parcels=1,
        initial_parcels=1,
        injection_z=0.0,
        initial_diameter=200.0e-6,
        initial_temperature=273.15 + 20.0,
        liquid_density=liquid_density,
        salinity_mass_fraction=salinity_mass_fraction,
        thermodynamic_transfer_model=thermodynamic_transfer_model,
        osmotic_coefficient_model=osmotic_coefficient_model,
        drag_correction_model=drag_correction_model,
        ventilation_correction_enabled=ventilation_correction_enabled,
        liquid_emissivity=0.0,
        shortwave_absorption_efficiency=0.0,
        substeps=1,
    )
    spray = initialize_spray(config, dtype=params.dtype)
    one = jnp.ones((1,), dtype=params.dtype)
    zero = jnp.zeros_like(one)
    gas_u = ambient_speed * one
    gas_temperature = ambient_temperature * one
    qv = ambient_qv * one
    theta = gas_temperature

    def step(state, _):
        rates = _parcel_rates_from_samples(
            state,
            gas_u,
            zero,
            zero,
            theta,
            qv,
            params,
            config,
        )
        phase_change = _proposed_phase_change(
            state, rates, dt, config
        )
        advanced = _advance_parcel_implicit(
            state,
            gas_u,
            zero,
            zero,
            rates,
            phase_change,
            dt,
            params,
            config,
        )
        active = state.active & (advanced.diameter >= config.min_diameter)
        updated = state._replace(
            u=jnp.where(state.active, advanced.u, state.u),
            v=jnp.where(state.active, advanced.v, state.v),
            w=jnp.where(state.active, advanced.w, state.w),
            mass=jnp.where(state.active, advanced.mass, state.mass),
            diameter=jnp.where(
                state.active, advanced.diameter, state.diameter
            ),
            temperature=jnp.where(
                state.active, advanced.temperature, state.temperature
            ),
            active=active,
        )
        sample = (
            updated.u[0],
            updated.w[0],
            updated.temperature[0],
            updated.diameter[0] * 0.5,
            updated.mass[0],
            phase_change[0] / dt,
            active[0],
        )
        return updated, sample

    _, history = jax.jit(
        lambda initial: lax.scan(step, initial, None, length=nsteps)
    )(spray)
    history = jax.device_get(history)
    initial = {
        "u": np.asarray([float(spray.u[0])]),
        "w": np.asarray([float(spray.w[0])]),
        "temperature": np.asarray([float(spray.temperature[0])]),
        "radius": np.asarray([0.5 * float(spray.diameter[0])]),
        "mass": np.asarray([float(spray.mass[0])]),
        "phase_change_rate": np.asarray([0.0]),
        "active": np.asarray([True]),
    }
    names = tuple(initial)
    result = {"time": np.arange(nsteps + 1, dtype=float) * dt}
    for name, values in zip(names, history, strict=True):
        result[name] = np.concatenate((initial[name], np.asarray(values)))
    inactive = np.flatnonzero(~result["active"].astype(bool))
    if inactive.size:
        end = max(int(inactive[0]), 1)
        result = {name: values[:end] for name, values in result.items()}
    return result


def write_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    fields = tuple(data)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(zip(*(data[field] for field in fields), strict=True))


def plot(
    path: Path,
    fresh: dict[str, np.ndarray],
    seawater: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    axes[0].plot(
        fresh["time"], fresh["u"], color="black", linestyle="--", label="fresh water"
    )
    axes[0].plot(
        seawater["time"], seawater["u"], color="black", label="seawater"
    )
    axes[0].axhline(15.0, color="0.6", linestyle="--", label="air velocity")
    axes[0].set_ylabel("velocity [m/s]")
    axes[0].legend()
    axes[1].plot(
        fresh["time"],
        fresh["temperature"] - 273.15,
        color="tab:blue",
        linestyle="--",
    )
    axes[1].plot(
        seawater["time"], seawater["temperature"] - 273.15, color="tab:blue"
    )
    axes[1].axhline(18.0, color="0.6", linestyle="--")
    axes[1].set_ylabel("drop temperature [degC]")
    axes[2].plot(
        fresh["time"],
        1.0e6 * fresh["radius"],
        color="tab:red",
        linestyle="--",
    )
    axes[2].plot(
        seawater["time"], 1.0e6 * seawater["radius"], color="tab:red"
    )
    axes[2].set_ylabel("drop radius [micrometre]")
    axes[2].set_xlabel("time [s]")
    figure.suptitle("Veron (2020) figure 1 setup: fresh water and seawater")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fresh = simulate(args.duration, args.dt)
    seawater = simulate(
        args.duration,
        args.dt,
        salinity_mass_fraction=0.035,
        liquid_density=1025.0,
    )
    for name, data in (("freshwater", fresh), ("seawater", seawater)):
        if not all(np.all(np.isfinite(values)) for values in data.values()):
            raise RuntimeError(f"non-finite value in {name} single-drop benchmark")
        write_csv(args.output_dir / f"veron2020_fig1_{name}.csv", data)
    plot(
        args.output_dir / "veron2020_fig1_freshwater_seawater.png",
        fresh,
        seawater,
    )


if __name__ == "__main__":
    main()
