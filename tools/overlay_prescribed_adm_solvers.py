#!/usr/bin/env python3
"""Overlay matched JAX and legacy-Fortran prescribed-ADM wake centerlines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jax_csv", type=Path)
    parser.add_argument("fortran_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jax = np.genfromtxt(args.jax_csv, delimiter=",", names=True)
    fort = np.genfromtxt(args.fortran_csv, delimiter=",", names=True)
    jx, jd = jax["x_over_D"], jax["les_deficit"]
    fx, fd = fort["x_over_D"], fort["legacy_deficit"]
    gaussian = jax["gaussian_ti_based_deficit"]
    compare = (jx >= 3.0) & (jx <= 10.0)
    interpolated_fortran = np.interp(jx[compare], fx, fd)
    solver_rmse = float(np.sqrt(np.mean((jd[compare] - interpolated_fortran) ** 2)))
    jax_gaussian_rmse = float(np.sqrt(np.mean((jd[compare] - gaussian[compare]) ** 2)))
    fortran_gaussian = np.interp(fx, jx, gaussian)
    fort_compare = (fx >= 3.0) & (fx <= 10.0)
    fortran_gaussian_rmse = float(
        np.sqrt(np.mean((fd[fort_compare] - fortran_gaussian[fort_compare]) ** 2))
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    ax.plot(jx, jd, linewidth=2.3, label="JAX prescribed ADM")
    ax.plot(fx, fd, linewidth=2.3, label="Legacy Fortran prescribed ADM")
    ax.plot(jx, gaussian, "--", linewidth=2.0, label=r"Gaussian, $C_T=0.84$, $k=0.0273$")
    ax.axvspan(3.0, 10.0, color="0.5", alpha=0.08, label="comparison interval")
    ax.set(
        xlim=(0.0, 11.6),
        ylim=(0.0, 0.75),
        xlabel=r"downstream distance $(x-x_T)/D$",
        ylabel=r"centerline deficit $1-\overline{u}/U_\infty$",
        title="Matched prescribed ADM: JAX vs legacy Fortran",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    ax.text(
        0.98,
        0.03,
        f"JAX–Fortran RMSE (3–10D): {solver_rmse:.4f}\n"
        f"JAX–Gaussian RMSE: {jax_gaussian_rmse:.4f}\n"
        f"Fortran–Gaussian RMSE: {fortran_gaussian_rmse:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
    )
    fig.savefig(args.output, dpi=180)
    print(f"jax_fortran_rmse={solver_rmse:.8f}")
    print(f"jax_gaussian_rmse={jax_gaussian_rmse:.8f}")
    print(f"fortran_gaussian_rmse={fortran_gaussian_rmse:.8f}")


if __name__ == "__main__":
    main()
