"""Prescribed-path shrinking-sphere diagnostic; never modifies coupled physics.

All shell enthalpies use h_l=cp_l*(T-273.15), and escaping vapour has
h_v=Lv, matching the production dilute convention. Gas/slip histories are
prescribed. Step-start transfer coefficients match the native droplet law.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tomllib
from scipy.optimize import brentq

from jaxwind.config.moisture import load_moisture
from jaxwind.physics.moisture import (
    WaterDropletProperties,
    advance_water_droplet,
    saturation_mixing_ratio,
    water_droplet_transfer_coefficients,
)


def sphere_step_factory(
    config, shells, conductivity, *, evaporate=True, prescribed_exchange=None
):
    """Return a batched conservative backward-Euler moving-shell step.

    Geometric mass flux is exactly the swept shell volume. At the outer face
    the total outward flux is E*Lv-Q; the nonlinear surface residual equates
    it to outward conduction plus advected surface-liquid sensible enthalpy.
    The inner shell solves are tridiagonal. Bisection bounds are explicitly
    checked by callers; no evaporation/mass/temperature clipping is used.
    """
    p = WaterDropletProperties()
    cp, lv, ref = (
        p.liquid_heat_capacity,
        config.water_vapor_latent_heat,
        config.freezing_temperature,
    )
    # Equal radial spacing resolves the surface thermal layer more efficiently
    # than equal-volume shells; refinement is explicitly checked below.
    faces = jnp.linspace(0.0, 1.0, shells + 1)
    vf = faces[1:] ** 3 - faces[:-1] ** 3
    centers = 0.75 * (faces[1:] ** 4 - faces[:-1] ** 4) / vf

    def step(mass, theta, surface, gas, dt):
        tg, qg, slip = gas[:, 0], gas[:, 1], gas[:, 2]
        radius_old = jnp.cbrt(3 * mass / (4 * jnp.pi * config.water_density))
        diameter = 2 * radius_old
        _, nu, sh = water_droplet_transfer_coefficients(diameter, slip, config, p)
        conductance = jnp.pi * diameter * p.air_thermal_conductivity * nu
        transfer = (
            jnp.pi * diameter * config.dry_air_density * config.vapor_diffusivity * sh
        )
        old_h = mass[:, None] * vf * cp * theta

        def solve(ts):
            qs = saturation_mixing_ratio(ts + ref, config.pressure, config)
            rate = transfer * jnp.maximum(jnp.log1p(qs) - jnp.log1p(qg), 0.0)
            if not evaporate:
                rate = jnp.zeros_like(rate)
            heat = conductance * (tg - ref - ts)
            if prescribed_exchange is not None:
                rate = jnp.full_like(rate, prescribed_exchange[0])
                heat = jnp.full_like(heat, prescribed_exchange[1])
            new_mass = mass - dt * rate
            radius = jnp.cbrt(3 * new_mass / (4 * jnp.pi * config.water_density))
            # Exactly equals -rho*delta(volume_enclosed)/dt. Use mass loss
            # rather than subtracting cubed nearly equal radii.
            swept_rate = (mass - new_mass) / dt
            g = swept_rate[:, None] * faces[None, :] ** 3
            k = (
                4
                * jnp.pi
                * conductivity
                * radius[:, None]
                * faces[None, 1:-1] ** 2
                / jnp.diff(centers)[None, :]
            )
            lower = jnp.concatenate(
                (jnp.zeros_like(k[:, :1]), -dt * (k + cp * g[:, 1:-1])), axis=1
            )
            upper = jnp.concatenate((-dt * k, jnp.zeros_like(k[:, :1])), axis=1)
            kin = jnp.concatenate((jnp.zeros_like(k[:, :1]), k), axis=1)
            kout = jnp.concatenate((k, jnp.zeros_like(k[:, :1])), axis=1)
            # Last boundary is a prescribed total flux, so no last diagonal
            # advection term: its surface liquid enthalpy enters residual only.
            gout = jnp.concatenate((g[:, 1:-1], jnp.zeros_like(g[:, :1])), axis=1)
            diagonal = new_mass[:, None] * vf * cp + dt * (kin + kout + cp * gout)
            flux = rate * lv - heat
            rhs = old_h.at[:, -1].add(-dt * flux)
            new_theta = jax.lax.linalg.tridiagonal_solve(
                lower, diagonal, upper, rhs[..., None]
            )[..., 0]
            surface_k = 4 * jnp.pi * conductivity * radius / (1 - centers[-1])
            residual = surface_k * (new_theta[:, -1] - ts) + swept_rate * cp * ts - flux
            return residual, new_mass, new_theta, heat, rate

        low = jnp.zeros_like(surface)
        high = jnp.maximum(jnp.max(theta, axis=1), tg - ref) + 1e-9
        low_r, high_r = solve(low)[0], solve(high)[0]

        def bisect(_, bounds):
            lo, hi = bounds
            mid = (lo + hi) * 0.5
            residual = solve(mid)[0]
            # A hotter guessed surface reduces the residual.
            return (jnp.where(residual >= 0, mid, lo), jnp.where(residual < 0, mid, hi))

        low, high = jax.lax.fori_loop(0, 40, bisect, (low, high))
        ts = 0.5 * (low + high)
        residual, new_mass, new_theta, heat, rate = solve(ts)
        new_h = new_mass[:, None] * vf * cp * new_theta
        energy_error = jnp.sum(new_h - old_h, axis=1) + dt * (rate * lv - heat)
        return (
            new_mass,
            new_theta,
            ts,
            jnp.stack(
                (
                    heat * dt,
                    rate * dt,
                    energy_error,
                    residual,
                    low_r,
                    high_r,
                    residual / jnp.maximum(jnp.abs(heat) + rate * lv, 1e-10),
                ),
                axis=-1,
            ),
        )

    return step, np.asarray(vf), np.asarray(centers)


def integrate(
    config,
    initial_mass,
    initial_temperature,
    gases,
    dt,
    *,
    shells=None,
    conductivity=0.6,
    evaporate=True,
    prescribed_exchange=None,
):
    """Return every accepted state and its independent conservation ledger."""
    ref = config.freezing_temperature
    if shells is None:

        def body(state, gas):
            m, t = state
            active = (
                gas[:, 3] > 0 if gas.shape[1] == 4 else jnp.ones_like(m, dtype=bool)
            )
            result = advance_water_droplet(
                m, t, gas[:, 0], gas[:, 1], gas[:, 2], dt * active, config
            )
            new = result.mass, result.temperature
            out = jnp.stack(
                (
                    result.mass,
                    result.temperature - ref,
                    result.temperature - ref,
                    result.gas_sensible_energy_loss,
                    result.evaporated_mass,
                    jnp.zeros_like(m),
                    jnp.zeros_like(m),
                    jnp.ones_like(m),
                    -jnp.ones_like(m),
                    jnp.zeros_like(m),
                ),
                axis=-1,
            )
            return new, out

        initial = (jnp.asarray(initial_mass), jnp.asarray(initial_temperature))
        values = jax.jit(lambda: jax.lax.scan(body, initial, jnp.asarray(gases))[1])()
        values = np.asarray(values)
        profiles = None
    else:
        step, vf, _ = sphere_step_factory(
            config,
            shells,
            conductivity,
            evaporate=evaporate,
            prescribed_exchange=prescribed_exchange,
        )

        def body(state, gas):
            m, theta, ts = state
            mn, tn, sn, diagnostics = step(m, theta, ts, gas, dt)
            if gas.shape[1] == 4:
                active = gas[:, 3] > 0
                mn = jnp.where(active, mn, m)
                tn = jnp.where(active[:, None], tn, theta)
                sn = jnp.where(active, sn, ts)
                diagnostics = jnp.where(active[:, None], diagnostics, 0)
            mean = jnp.sum(tn * jnp.asarray(vf), axis=1)
            out = jnp.concatenate(
                (mn[:, None], mean[:, None], sn[:, None], diagnostics), axis=-1
            )
            return (mn, tn, sn), (out, tn)

        theta0 = jnp.asarray(initial_temperature) - ref
        initial = (
            jnp.asarray(initial_mass),
            jnp.broadcast_to(theta0[:, None], (len(initial_mass), shells)),
            theta0,
        )
        values, profiles = jax.jit(
            lambda: jax.lax.scan(body, initial, jnp.asarray(gases))[1]
        )()
        values, profiles = np.asarray(values), np.asarray(profiles)
    if not np.isfinite(values).all():
        raise AssertionError("Nonfinite state; diagnostic stops without clipping")
    if np.any(values[:, :, 0] <= 0):
        raise AssertionError("Droplet depleted; use a separately verified endpoint")
    if np.min(values[:, :, 7]) < -1e-10 or np.max(values[:, :, 8]) > 1e-10:
        raise AssertionError("Surface root escaped its physical bracket")
    cp, lv = (
        WaterDropletProperties().liquid_heat_capacity,
        config.water_vapor_latent_heat,
    )
    h0 = initial_mass * cp * (initial_temperature - ref)
    h = values[:, :, 0] * cp * values[:, :, 1]
    heat = np.cumsum(values[:, :, 3], axis=0)
    escaped = np.cumsum(values[:, :, 4], axis=0)
    energy_residual = h - h0 + lv * escaped - heat
    mass_residual = values[:, :, 0] - initial_mass + escaped
    ledger = {
        "max_relative_energy_residual": float(np.max(np.abs(energy_residual / h0))),
        "max_relative_mass_residual": float(
            np.max(np.abs(mass_residual / initial_mass))
        ),
        "max_relative_local_energy_residual": float(
            np.max(np.abs(values[:, :, 5] / h0))
        ),
        "max_surface_flux_residual_W": float(np.max(np.abs(values[:, :, 6]))),
        "max_relative_surface_flux_residual": float(np.max(np.abs(values[:, :, 9]))),
        "min_mass_kg": float(values[:, :, 0].min()),
        "minimum_surface_C": float(values[:, :, 2].min()),
    }
    if ledger["max_relative_surface_flux_residual"] > 1e-6:
        raise AssertionError(f"Surface flux boundary failed: {ledger}")
    if profiles is not None and (
        profiles.min() < -1e-7
        or profiles.max()
        > max(initial_temperature.max(), gases[:, :, 0].max()) - ref + 1e-7
    ):
        raise AssertionError("Shell temperatures escaped warm physical bracket")
    if ledger["max_relative_energy_residual"] > 1e-8:
        raise AssertionError(f"Energy ledger failed: {ledger}")
    if ledger["max_relative_mass_residual"] > 1e-10:
        raise AssertionError(f"Mass ledger failed: {ledger}")
    return values, profiles, ledger


def analytic_sphere(config, output):
    """Non-evaporating 500um sphere, analytic surface and volume-average T."""
    p = WaterDropletProperties()
    diameter, conductivity, tg, t0, slip = 500e-6, 0.6, 313.15, 293.15, 15.0
    mass = config.water_density * math.pi * diameter**3 / 6
    radius = diameter / 2
    nu = float(
        water_droplet_transfer_coefficients(
            jnp.asarray(diameter), jnp.asarray(slip), config, p
        )[1]
    )
    bi = nu * p.air_thermal_conductivity / (2 * conductivity)
    roots = [
        brentq(
            lambda z: 1 - z / np.tan(z) - bi, n * np.pi + 1e-7, (n + 1) * np.pi - 1e-7
        )
        for n in range(50)
    ]
    times = np.array([0.002, 0.01, 0.05, 0.1])
    alpha = conductivity / (config.water_density * p.liquid_heat_capacity)
    exact_surface = np.zeros_like(times)
    exact_mean = np.zeros_like(times)
    for lam in roots:
        a = 4 * (np.sin(lam) - lam * np.cos(lam)) / (2 * lam - np.sin(2 * lam))
        decay = np.exp(-(lam**2) * alpha * times / radius**2)
        exact_surface += a * np.sin(lam) / lam * decay
        exact_mean += a * 3 * (np.sin(lam) - lam * np.cos(lam)) / lam**3 * decay
    exact_surface = tg + (t0 - tg) * exact_surface - 273.15
    exact_mean = tg + (t0 - tg) * exact_mean - 273.15
    rows = []
    for shells, dt in [(16, 0.000125), (32, 0.0000625)]:
        gases = np.tile([[[tg, 0.02, slip]]], (round(0.1 / dt), 1, 1))
        values, _, ledger = integrate(
            config,
            np.array([mass]),
            np.array([t0]),
            gases,
            dt,
            shells=shells,
            conductivity=conductivity,
            evaporate=False,
        )
        index = np.rint(times / dt).astype(int) - 1
        mean, surface = values[index, 0, 1], values[index, 0, 2]
        rows.append(
            {
                "shells": shells,
                "dt_s": dt,
                "max_mean_error_K": float(np.max(np.abs(mean - exact_mean))),
                "max_surface_error_K": float(np.max(np.abs(surface - exact_surface))),
                "ledger": ledger,
            }
        )
    assert rows[-1]["max_mean_error_K"] < 0.01
    assert rows[-1]["max_surface_error_K"] < 0.015
    output["non_evaporating_sphere"] = {
        "Bi_R": bi,
        "times_s": times.tolist(),
        "exact_mean_C": exact_mean.tolist(),
        "exact_surface_C": exact_surface.tolist(),
        "resolutions": rows,
    }
    print("Analytic sphere", json.dumps(rows), flush=True)


def geometric_conservation(config, output):
    cp, lv = (
        WaterDropletProperties().liquid_heat_capacity,
        config.water_vapor_latent_heat,
    )
    mass = config.water_density * math.pi * (500e-6) ** 3 / 6
    theta, dt, duration = 30.0, 0.000125, 0.05
    # Remove 10% mass with a surface flux balanced at a constant liquid T.
    rate = 0.1 * mass / duration
    heat = rate * (lv - cp * theta)
    gases = np.tile([[[313.15, 0.01, 15.0]]], (round(duration / dt), 1, 1))
    values, profiles, ledger = integrate(
        config,
        np.array([mass]),
        np.array([theta + 273.15]),
        gases,
        dt,
        shells=16,
        conductivity=0.6,
        prescribed_exchange=(rate, heat),
    )
    defect = float(np.max(np.abs(profiles - theta)))
    assert defect < 1e-6, defect
    output["shrinking_uniform_sphere"] = {
        "max_temperature_defect_K": defect,
        "final_mass_fraction": float(values[-1, 0, 0] / mass),
        "ledger": ledger,
    }
    print("Shrinking uniform sphere", output["shrinking_uniform_sphere"], flush=True)


def run_histories(config, args, output):
    with np.load(args.history) as archive:
        history = archive["history"]
        diameters = archive["initial_diameter_m"]
    sizes = np.unique(diameters) if args.all_sizes else np.array([diameters.min(), diameters.max()])
    groups = [np.flatnonzero(diameters == size) for size in sizes]
    assert all(len(group) % args.rays == 0 for group in groups)
    indices = np.concatenate([group[::len(group) // args.rays] for group in groups])
    output["selected_diameters_um"] = (sizes * 1e6).tolist()
    output["rays_per_size"] = args.rays
    history, diameters = history[:, :, indices], diameters[indices]
    initial_mass = history[0, 5]
    initial_temperature = history[0, 6]
    native_times = history[:, 0, 0]
    stations = np.array([0.25, 0.5, 0.75, 0.99]) * args.length
    crossing = np.array(
        [
            [
                np.interp(x, history[:, 4, ray], native_times)
                for ray in range(len(indices))
            ]
            for x in stations
        ]
    )
    duration = crossing[-1].max()
    models = [
        ("lumped", None, 0.6, 0.0000625),
        ("k0.6_n8", 8, 0.6, 0.000125),
        ("k0.6_n16", 16, 0.6, 0.000125),
        ("k0.6_n32", 32, 0.6, 0.000125),
        ("k0.6_n32_halfdt", 32, 0.6, 0.0000625),
        ("k60_n16", 16, 60.0, 0.0000625),
    ]
    results = {}
    for name, shells, conductivity, dt in models:
        start = time.monotonic()
        steps = math.ceil(duration / dt)
        t_old = np.arange(steps) * dt
        times = np.arange(steps + 1) * dt
        gases = np.stack(
            [
                np.stack(
                    [
                        np.interp(t_old, native_times, history[:, field, ray])
                        for field in [1, 2, 3]
                    ],
                    axis=-1,
                )
                for ray in range(len(indices))
            ],
            axis=1,
        )
        gases = np.concatenate(
            (gases, (t_old[:, None] < crossing[-1][None, :])[..., None]), axis=-1
        )
        values, profiles, ledger = integrate(
            config,
            initial_mass,
            initial_temperature,
            gases,
            dt,
            shells=shells,
            conductivity=conductivity,
        )
        initial = np.zeros((1, len(indices), values.shape[-1]))
        initial[0, :, 0] = initial_mass
        initial[0, :, 1:3] = (initial_temperature - 273.15)[:, None]
        values = np.concatenate((initial, values), axis=0)
        # Fields for comparing cumulative gas exchanges and liquid state.
        values[:, :, 3:5] = np.cumsum(values[:, :, 3:5], axis=0)
        rows = []
        for station, x in enumerate(stations):
            for ray, ray_id in enumerate(indices):
                age = crossing[station, ray]
                state = np.array(
                    [np.interp(age, times, values[:, ray, field]) for field in range(5)]
                )
                rows.append(
                    {
                        "x_over_L": float(x / args.length),
                        "ray_id": int(ray_id),
                        "diameter_um": float(diameters[ray] * 1e6),
                        "age_s": float(age),
                        "mass_kg": float(state[0]),
                        "mean_temperature_C": float(state[1]),
                        "surface_temperature_C": float(state[2]),
                        "cumulative_heat_J": float(state[3]),
                        "cumulative_evaporated_kg": float(state[4]),
                        "native_temperature_C": float(
                            np.interp(age, native_times, history[:, 6, ray]) - 273.15
                        ),
                        "native_mass_kg": float(
                            np.interp(age, native_times, history[:, 5, ray])
                        ),
                    }
                )
        results[name] = {
            "shells": shells,
            "conductivity_W_m_K": conductivity,
            "dt_s": dt,
            "wall_seconds": time.monotonic() - start,
            "ledger": ledger,
            "samples": rows,
        }
        np.savez_compressed(
            args.output / f"{name}.npz",
            times=times,
            values=values,
            profiles=profiles,
            ray_indices=indices,
            crossing_times=crossing,
        )
        output["history_runs"] = results
        (args.output / "results.json").write_text(json.dumps(output, indent=2) + "\n")
        print(
            "Finished",
            name,
            "seconds",
            results[name]["wall_seconds"],
            "ledger",
            ledger,
            flush=True,
        )
    baseline = results["lumped"]["samples"]
    comparisons = {}
    for name, result in results.items():
        rows = result["samples"]
        comparisons[name] = {
            "max_native_temperature_error_K": max(
                abs(r["mean_temperature_C"] - r["native_temperature_C"]) for r in rows
            ),
            "max_native_relative_mass_error": max(
                abs(r["mass_kg"] / r["native_mass_kg"] - 1) for r in rows
            ),
            "max_mean_shift_from_lumped_K": max(
                abs(r["mean_temperature_C"] - b["mean_temperature_C"])
                for r, b in zip(rows, baseline)
            ),
        }
    output["comparisons"] = comparisons
    sensitivities = {}
    for coarse, fine in [
        ("k0.6_n8", "k0.6_n16"),
        ("k0.6_n16", "k0.6_n32"),
        ("k0.6_n32", "k0.6_n32_halfdt"),
    ]:
        a, b = results[coarse]["samples"], results[fine]["samples"]
        sensitivities[f"{coarse}_to_{fine}"] = {
            "max_mean_temperature_change_K": max(
                abs(x["mean_temperature_C"] - y["mean_temperature_C"])
                for x, y in zip(a, b)
            ),
            "max_surface_temperature_change_K": max(
                abs(x["surface_temperature_C"] - y["surface_temperature_C"])
                for x, y in zip(a, b)
            ),
            "max_relative_evaporation_change": max(
                abs(x["cumulative_evaporated_kg"] / y["cumulative_evaporated_kg"] - 1)
                for x, y in zip(a, b)
            ),
        }
    output["sensitivities"] = sensitivities
    for key in ["k0.6_n16_to_k0.6_n32", "k0.6_n32_to_k0.6_n32_halfdt"]:
        row = sensitivities[key]
        assert row["max_mean_temperature_change_K"] < 0.05, (key, row)
        assert row["max_surface_temperature_change_K"] < 0.05, (key, row)
        assert row["max_relative_evaporation_change"] < 0.01, (key, row)
    assert comparisons["lumped"]["max_native_temperature_error_K"] < 0.05, comparisons[
        "lumped"
    ]
    assert comparisons["lumped"]["max_native_relative_mass_error"] < 1e-4, comparisons[
        "lumped"
    ]
    assert comparisons["k60_n16"]["max_mean_shift_from_lumped_K"] < 0.1, comparisons[
        "k60_n16"
    ]


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rays", type=int, choices=[1, 2, 4, 8, 16, 32], default=4)
    ap.add_argument("--all-sizes", action="store_true", help="Replay every diameter in the archived history")
    ap.add_argument("--length", type=float, default=1.9)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    jax.config.update("jax_enable_x64", True)
    assert args.allow_cpu or jax.default_backend() == "gpu", jax.devices()
    args.output.mkdir(parents=True, exist_ok=True)
    doc = tomllib.loads(args.case.read_text())
    config = load_moisture(doc["physics"])[0].thermodynamics
    output = {
        "backend": jax.default_backend(),
        "devices": [str(x) for x in jax.devices()],
        "scope": __doc__,
        "history": str(args.history),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "history_sha256": hashlib.sha256(args.history.read_bytes()).hexdigest(),
        "case_sha256": hashlib.sha256(args.case.read_bytes()).hexdigest(),
    }
    print("Backend", output["backend"], output["devices"], flush=True)
    analytic_sphere(config, output)
    geometric_conservation(config, output)
    if not args.verify_only:
        run_histories(config, args, output)
    (args.output / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print("All requested diagnostic checks passed", flush=True)


if __name__ == "__main__":
    main()
