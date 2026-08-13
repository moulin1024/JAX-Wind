#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
JAX_ROOT = ROOT / "legacy" / "jax"
sys.path.insert(0, str(JAX_ROOT))

from run_single import RUN_DEFAULTS, load_config_file, params_from_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the actuator-disk liquid-nitrogen hub-cooling LES."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "quick.toml",
    )
    parser.add_argument(
        "--case", choices=("baseline", "cold", "both"), default="both"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "LiquidNitrogenHubJet",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Override the configuration with an 8-step plumbing check.",
    )
    return parser.parse_args()


def write_diagnostics(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = rows[0]._fields
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for row in rows:
            writer.writerow([float(getattr(row, name)) for name in fields])


def save_state(path: Path, state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        u=np.asarray(state.u),
        v=np.asarray(state.v),
        w=np.asarray(state.w),
        p=np.asarray(state.p),
        theta=np.asarray(state.theta),
        qv=np.asarray(state.qv),
        step=np.asarray(state.step),
    )


def run_case(name: str, params, output_dir: Path):
    from wireles_jax import run

    print(f"[{name}] starting {params.nx}x{params.ny}x{params.nz}, steps={params.nsteps}")
    state, rows = run(
        params,
        log_every=params.c_count,
        status_callback=lambda message: print(f"[{name}] {message}", flush=True),
        log_callback=lambda diag: print(
            f"[{name}] step={int(diag.step):6d} "
            f"CFL={max(float(diag.cfl_x), float(diag.cfl_y), float(diag.cfl_z)):.3f} "
            f"theta_min={float(diag.theta_v_min):.3f}",
            flush=True,
        ),
    )
    save_state(output_dir / name / "final_state.npz", state)
    write_diagnostics(output_dir / name / "diagnostics.csv", rows)
    return state


def plot_results(output_dir: Path, states: dict[str, object], params) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diameter = params.actuator_disk_diameter
    x = (np.arange(params.nx) + 0.5) * params.dx * params.z_i / diameter
    z = (np.arange(params.nz) + 0.5) * params.dz * params.z_i / diameter
    y_index = int(round(params.actuator_disk_y / (params.dy * params.z_i) - 0.5))
    y_index = min(max(y_index, 0), params.ny - 1)
    rotor_x = params.actuator_disk_x / diameter
    rotor_bottom = (params.actuator_disk_z - 0.5 * diameter) / diameter
    rotor_top = (params.actuator_disk_z + 0.5 * diameter) / diameter

    columns = len(states)
    fig, axes = plt.subplots(3, columns, figsize=(6.2 * columns, 10.0), squeeze=False)
    for column, (name, state) in enumerate(states.items()):
        fields = (
            (np.asarray(state.u)[:, y_index, :].T / params.uniform_u, r"$u/U_\infty$", "viridis"),
            (params.theta0 - np.asarray(state.theta)[:, y_index, :].T, r"$\theta_0-\theta$ [K]", "inferno"),
            (np.asarray(state.w)[:, y_index, :].T / params.uniform_u, r"$w/U_\infty$", "RdBu_r"),
        )
        for row, (field, label, cmap) in enumerate(fields):
            axis = axes[row, column]
            if row == 2:
                limit = max(float(np.max(np.abs(field))), 1.0e-6)
                image = axis.pcolormesh(x, z, field, shading="auto", cmap=cmap, vmin=-limit, vmax=limit)
            else:
                image = axis.pcolormesh(x, z, field, shading="auto", cmap=cmap)
            axis.plot([rotor_x, rotor_x], [rotor_bottom, rotor_top], color="white", lw=2.0)
            axis.set_ylabel(r"$z/D$")
            axis.set_title(f"{name}: {label}")
            fig.colorbar(image, ax=axis, pad=0.02)
        axes[-1, column].set_xlabel(r"$x/D$")
    fig.tight_layout()
    fig.savefig(output_dir / "final_centerplane.png", dpi=180)
    plt.close(fig)

    if "baseline" in states and "cold" in states:
        baseline = states["baseline"]
        cold = states["cold"]
        delta_u = (
            np.asarray(cold.u)[:, y_index, :]
            - np.asarray(baseline.u)[:, y_index, :]
        ).T / params.uniform_u
        delta_theta = (
            np.asarray(cold.theta)[:, y_index, :]
            - np.asarray(baseline.theta)[:, y_index, :]
        ).T
        fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True)
        for axis, field, label in (
            (axes[0], delta_u, r"$(u_{cold}-u_{base})/U_\infty$"),
            (axes[1], delta_theta, r"$\theta_{cold}-\theta_{base}$ [K]"),
        ):
            limit = max(float(np.max(np.abs(field))), 1.0e-6)
            image = axis.pcolormesh(
                x, z, field, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
            )
            axis.plot([rotor_x, rotor_x], [rotor_bottom, rotor_top], color="black", lw=2.0)
            axis.set_ylabel(r"$z/D$")
            axis.set_title(label)
            fig.colorbar(image, ax=axis, pad=0.02)
        axes[-1].set_xlabel(r"$x/D$")
        fig.tight_layout()
        fig.savefig(output_dir / "cold_minus_baseline.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    settings = dict(RUN_DEFAULTS)
    settings.update(load_config_file(args.config))
    if args.smoke:
        settings.update(nx=24, ny=12, nz=12, steps=8, log_every=4, use_jit=False)

    if settings["precision"] == "float64" or settings["sgs_precision"] == "float64":
        from jax import config as jax_config

        jax_config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    params = params_from_settings(settings, jnp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = ("baseline", "cold") if args.case == "both" else (args.case,)
    states = {}
    for name in selected:
        case_params = replace(params, cold_source_enabled=(name == "cold"))
        states[name] = run_case(name, case_params, args.output_dir)
    plot_results(args.output_dir, states, params)
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
