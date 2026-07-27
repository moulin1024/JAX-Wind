#!/usr/bin/env python3
"""Run the thrust-only yawed-WiRE-01 benchmark from Lin & Porté-Agel (2019)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy" / "jax"))

from benchmark.LinPorteAgel2019.case import (  # noqa: E402
    PAPER_CASE,
    THRUST_COEFFICIENT_BY_YAW,
    paper_settings,
)
from run_single import RUN_DEFAULTS, params_from_settings  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaw",
        nargs="+",
        type=float,
        default=list(PAPER_CASE.yaw_degrees),
        help="paper yaw angle(s) in degrees (default: 10 20 30)",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--sample-start", type=int, default=4_000)
    parser.add_argument("--sample-every", type=int)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "LinPorteAgel2019",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run an 8-step reduced-grid end-to-end smoke check",
    )
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.sample_start < 0:
        parser.error("--sample-start must be non-negative")
    if args.sample_every is not None and args.sample_every <= 0:
        parser.error("--sample-every must be positive")
    unsupported = sorted(set(args.yaw) - set(PAPER_CASE.yaw_degrees))
    if unsupported:
        parser.error(f"unsupported paper yaw angle(s): {unsupported}")
    if args.quick:
        args.sample_start = 0
    elif args.steps is not None and args.sample_start > args.steps:
        parser.error("--sample-start must not exceed --steps")
    return args


class FlowStatistics:
    def __init__(self, sample_start: int) -> None:
        self.sample_start = sample_start
        self.count = 0
        self.sum_u: np.ndarray | None = None
        self.sum_u2: np.ndarray | None = None
        self.disk_velocity_sum = 0.0
        self.disk_velocity2_sum = 0.0
        self._disk: np.ndarray | None = None
        self._normal = (1.0, 0.0)

    def configure_disk(self, params) -> None:
        from wireles_jax.wind_tunnel import actuator_disk_kernel

        self._disk = np.asarray(actuator_disk_kernel(params), dtype=np.float64)
        yaw = np.deg2rad(params.actuator_disk_yaw_degrees)
        self._normal = (float(np.cos(yaw)), float(np.sin(yaw)))

    def sample(self, state, diagnostic) -> None:
        if int(diagnostic.step) < self.sample_start:
            return
        u = np.asarray(state.u, dtype=np.float64)
        v = np.asarray(state.v, dtype=np.float64)
        if self.sum_u is None:
            self.sum_u = np.zeros_like(u)
            self.sum_u2 = np.zeros_like(u)
        self.sum_u += u
        self.sum_u2 += u * u
        disk = self._disk
        normal_velocity = u * self._normal[0] + v * self._normal[1]
        disk_velocity = float(np.sum(normal_velocity * disk) / np.sum(disk))
        self.disk_velocity_sum += disk_velocity
        self.disk_velocity2_sum += disk_velocity * disk_velocity
        self.count += 1

    def finish(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        if self.count == 0 or self.sum_u is None or self.sum_u2 is None:
            raise RuntimeError("no statistics samples were collected")
        mean_u = self.sum_u / self.count
        variance_u = np.maximum(self.sum_u2 / self.count - mean_u * mean_u, 0.0)
        return (
            mean_u,
            np.sqrt(variance_u),
            self.disk_velocity_sum / self.count,
            self.disk_velocity2_sum / self.count,
        )


def _linear_sample(array: np.ndarray, coordinate: float, spacing: float, axis: int):
    fractional = coordinate / spacing - 0.5
    lower = int(np.floor(fractional))
    lower = max(0, min(lower, array.shape[axis] - 2))
    weight = fractional - lower
    first = np.take(array, lower, axis=axis)
    second = np.take(array, lower + 1, axis=axis)
    return (1.0 - weight) * first + weight * second


def wake_profiles(mean_u: np.ndarray, rms_u: np.ndarray, params):
    hub_mean = _linear_sample(
        mean_u, PAPER_CASE.hub_height, params.dz * params.z_i, axis=2
    )
    hub_rms = _linear_sample(
        rms_u, PAPER_CASE.hub_height, params.dz * params.z_i, axis=2
    )
    profiles = []
    for x_over_d in PAPER_CASE.profile_x_over_d:
        x = PAPER_CASE.turbine_x + x_over_d * PAPER_CASE.rotor_diameter
        profiles.append(
            (
                x_over_d,
                _linear_sample(hub_mean, x, params.dx * params.z_i, axis=0),
                _linear_sample(hub_rms, x, params.dx * params.z_i, axis=0),
            )
        )
    return profiles


def write_profiles(path: Path, profiles, params) -> list[dict[str, float]]:
    y = (np.arange(params.ny) + 0.5) * params.dy * params.z_i
    y_over_d = (y - PAPER_CASE.turbine_y) / PAPER_CASE.rotor_diameter
    fields = ["y_over_d"]
    for x_over_d, _, _ in profiles:
        suffix = f"{x_over_d:g}d"
        fields.extend((f"deficit_{suffix}", f"ti_{suffix}"))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, transverse in enumerate(y_over_d):
            row = {"y_over_d": float(transverse)}
            for x_over_d, mean_profile, rms_profile in profiles:
                suffix = f"{x_over_d:g}d"
                row[f"deficit_{suffix}"] = float(
                    (PAPER_CASE.hub_velocity - mean_profile[index])
                    / PAPER_CASE.hub_velocity
                )
                row[f"ti_{suffix}"] = float(
                    rms_profile[index] / PAPER_CASE.hub_velocity
                )
            writer.writerow(row)

    summary = []
    rotor_window = np.abs(y_over_d) <= 1.5
    for x_over_d, mean_profile, _ in profiles:
        deficit = (
            PAPER_CASE.hub_velocity - mean_profile
        ) / PAPER_CASE.hub_velocity
        masked_indices = np.flatnonzero(rotor_window)
        wake_index = masked_indices[np.argmax(deficit[rotor_window])]
        summary.append(
            {
                "x_over_d": x_over_d,
                "maximum_deficit": float(deficit[wake_index]),
                "wake_center_y_over_d": float(y_over_d[wake_index]),
            }
        )
    return summary


def plot_profiles(path: Path, profiles, params, yaw_degrees: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = (np.arange(params.ny) + 0.5) * params.dy * params.z_i
    y_over_d = (y - PAPER_CASE.turbine_y) / PAPER_CASE.rotor_diameter
    fig, axes = plt.subplots(1, len(profiles), figsize=(12.0, 4.0), sharey=True)
    for axis, (x_over_d, mean_profile, _) in zip(axes, profiles, strict=True):
        deficit = (
            PAPER_CASE.hub_velocity - mean_profile
        ) / PAPER_CASE.hub_velocity
        axis.plot(deficit, y_over_d, color="black")
        axis.axhline(0.5, color="black", ls="--", lw=0.7)
        axis.axhline(-0.5, color="black", ls="--", lw=0.7)
        axis.set_title(rf"$x/D={x_over_d:g}$")
        axis.set_xlabel(r"$\Delta \bar{u}/u_h$")
        axis.set_xlim(left=0.0)
        axis.set_ylim(-1.0, 1.5)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"$y/D$")
    fig.suptitle(f"Thrust-only ADM, yaw={yaw_degrees:g}°")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_yaw(args: argparse.Namespace, yaw_degrees: float) -> None:
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from wireles_jax import run

    settings = dict(RUN_DEFAULTS)
    settings.update(
        paper_settings(
            yaw_degrees,
            quick=args.quick,
            steps=args.steps,
            sample_every=args.sample_every,
        )
    )
    params = params_from_settings(settings, jnp)
    case_dir = args.output_dir / f"yaw_{yaw_degrees:g}deg"
    case_dir.mkdir(parents=True, exist_ok=True)
    statistics = FlowStatistics(args.sample_start)
    statistics.configure_disk(params)

    print(
        f"[yaw={yaw_degrees:g}°] {params.nx}x{params.ny}x{params.nz}, "
        f"steps={params.nsteps}, CT={THRUST_COEFFICIENT_BY_YAW[yaw_degrees]:.3f}, "
        f"CT'={params.actuator_disk_ct_prime:.3f}",
        flush=True,
    )
    final_state, diagnostics = run(
        params,
        seed=args.seed,
        log_every=params.c_count,
        log_state_callback=statistics.sample,
        status_callback=lambda message: print(message, flush=True),
        log_callback=lambda row: print(
            f"[yaw={yaw_degrees:g}°] step={int(row.step):6d} "
            f"CFL={max(float(row.cfl_x), float(row.cfl_y), float(row.cfl_z)):.3f}",
            flush=True,
        ),
    )
    mean_u, rms_u, mean_disk_velocity, mean_disk_velocity2 = statistics.finish()
    profiles = wake_profiles(mean_u, rms_u, params)
    wake_summary = write_profiles(case_dir / "profiles.csv", profiles, params)
    plot_profiles(case_dir / "profiles.png", profiles, params, yaw_degrees)

    cell_volume = params.dx * params.dy * params.dz * params.z_i**3
    from wireles_jax.wind_tunnel import actuator_disk_kernel

    loaded_area = float(np.sum(np.asarray(actuator_disk_kernel(params))) * cell_volume)
    density = 1.225
    mean_thrust = (
        0.5
        * density
        * params.actuator_disk_ct_prime
        * mean_disk_velocity2
        * loaded_area
    )
    summary = {
        "paper": "Lin and Porte-Agel (2019), doi:10.3390/en12234574",
        "model": "uniform pure-thrust yawed actuator disk",
        "yaw_degrees": yaw_degrees,
        "thrust_coefficient": THRUST_COEFFICIENT_BY_YAW[yaw_degrees],
        "local_thrust_coefficient": params.actuator_disk_ct_prime,
        "samples": statistics.count,
        "mean_disk_normal_velocity_m_s": mean_disk_velocity,
        "mean_thrust_n": mean_thrust,
        "adm_power_estimate_w": mean_thrust * mean_disk_velocity,
        "wake": wake_summary,
        "final_step": int(final_state.step),
        "maximum_cfl": max(
            max(float(row.cfl_x), float(row.cfl_y), float(row.cfl_z))
            for row in diagnostics
        ),
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(case_dir / "mean_fields.npz", mean_u=mean_u, rms_u=rms_u)
    print(f"[yaw={yaw_degrees:g}°] results: {case_dir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for yaw_degrees in args.yaw:
        run_yaw(args, yaw_degrees)


if __name__ == "__main__":
    main()
