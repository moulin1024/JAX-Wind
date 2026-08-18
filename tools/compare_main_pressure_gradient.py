#!/usr/bin/env python3
"""Compare pressure-gradient-off/on JAX wake runs with the legacy baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pressure_off_csv", type=Path)
    parser.add_argument("pressure_on_csv", type=Path)
    parser.add_argument("fortran_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    off = np.genfromtxt(args.pressure_off_csv, delimiter=",", names=True)
    on = np.genfromtxt(args.pressure_on_csv, delimiter=",", names=True)
    fort = np.genfromtxt(args.fortran_csv, delimiter=",", names=True)
    ox, od, ou = off["x_over_D"], off["les_deficit"], off["les_mean_u_m_s"]
    nx, nd, nu = on["x_over_D"], on["les_deficit"], on["les_mean_u_m_s"]
    fx, fd, fu = (
        fort["x_over_D"],
        fort["legacy_deficit"],
        fort["legacy_mean_u_m_s"],
    )
    compare = (ox >= 3.0) & (ox <= 10.0)
    on_deficit = np.interp(ox[compare], nx, nd)
    on_velocity = np.interp(ox[compare], nx, nu)
    fort_deficit = np.interp(ox[compare], fx, fd)
    fort_velocity = np.interp(ox[compare], fx, fu)
    gaussian = off["gaussian_ti_based_deficit"]
    upstream = (ox >= -4.0) & (ox <= -1.0)

    def upstream_metrics(x_m: np.ndarray, velocity: np.ndarray) -> dict[str, float]:
        slope_m_s_per_m = float(np.polyfit(x_m[upstream], velocity[upstream], 1)[0])
        return {
            "mean_velocity_m_s": float(np.mean(velocity[upstream])),
            "streamwise_velocity_slope_m_s_per_km": 1000.0 * slope_m_s_per_m,
        }

    metrics = {
        "comparison_interval_D": [3.0, 10.0],
        "pressure_on_vs_off": {
            "deficit_rmse": _rmse(on_deficit, od[compare]),
            "mean_velocity_rmse_m_s": _rmse(on_velocity, ou[compare]),
            "mean_velocity_bias_m_s": float(np.mean(on_velocity - ou[compare])),
        },
        "pressure_off_vs_fortran": {
            "deficit_rmse": _rmse(od[compare], fort_deficit),
            "mean_velocity_rmse_m_s": _rmse(ou[compare], fort_velocity),
        },
        "pressure_on_vs_fortran": {
            "deficit_rmse": _rmse(on_deficit, fort_deficit),
            "mean_velocity_rmse_m_s": _rmse(on_velocity, fort_velocity),
        },
        "upstream_interval_D": [-4.0, -1.0],
        "upstream_pressure_off": upstream_metrics(off["x_m"], ou),
        "upstream_pressure_on": upstream_metrics(on["x_m"], np.interp(ox, nx, nu)),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=True, constrained_layout=True)
    axes[0].plot(ox, ou, linewidth=2.2, label="JAX, pressure gradient off")
    axes[0].plot(nx, nu, linewidth=2.2, label="JAX, pressure gradient on")
    axes[0].plot(fx, fu, linewidth=2.0, label="Legacy Fortran (pressure gradient off)")
    axes[0].set(ylabel=r"hub-height centerline $\overline{u}$ (m s$^{-1}$)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(ox, od, linewidth=2.2, label="JAX, pressure gradient off")
    axes[1].plot(nx, nd, linewidth=2.2, label="JAX, pressure gradient on")
    axes[1].plot(fx, fd, linewidth=2.0, label="Legacy Fortran (pressure gradient off)")
    axes[1].plot(ox, gaussian, "--", linewidth=1.8, label="Gaussian model")
    axes[1].axvspan(3.0, 10.0, color="0.5", alpha=0.08)
    axes[1].set(
        xlim=(-5.5, 11.6),
        ylim=(-0.12, 0.9),
        xlabel=r"downstream distance $(x-x_T)/D$",
        ylabel=r"centerline deficit $1-\overline{u}/U_\infty$",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Main-domain pressure-gradient A/B test")
    fig.savefig(args.output / "pressure_gradient_ab.png", dpi=180)
    (args.output / "pressure_gradient_ab.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
