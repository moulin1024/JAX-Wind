#!/usr/bin/env python3
"""Compare a legacy WiRE-LES ADM wake with the Bastankhah Gaussian model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def gaussian_deficit(x_over_d: np.ndarray, ct: float, k: float) -> np.ndarray:
    beta = 0.5 * (1.0 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct)
    sigma_over_d = k * x_over_d + 0.2 * np.sqrt(beta)
    radicand = 1.0 - ct / (8.0 * sigma_over_d**2)
    return np.where(radicand >= 0.0, 1.0 - np.sqrt(np.maximum(radicand, 0.0)), np.nan)


def interpolate_plane(field: np.ndarray, coordinates: np.ndarray, target: float, axis: int) -> np.ndarray:
    upper = int(np.searchsorted(coordinates, target))
    lower = upper - 1
    weight = (target - coordinates[lower]) / (coordinates[upper] - coordinates[lower])
    return (1.0 - weight) * np.take(field, lower, axis=axis) + weight * np.take(field, upper, axis=axis)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--lx-m", type=float, default=4096.0)
    parser.add_argument("--ly-m", type=float, default=1024.0)
    parser.add_argument("--lz-m", type=float, default=1024.0)
    parser.add_argument("--turbine-x-m", type=float, default=1000.0)
    parser.add_argument("--turbine-y-m", type=float, default=512.0)
    parser.add_argument("--hub-height-m", type=float, default=119.0)
    parser.add_argument("--diameter-m", type=float, default=178.3)
    parser.add_argument("--ct", type=float, default=0.840)
    parser.add_argument("--prescribed-uinf-m-s", type=float, default=11.08514881)
    parser.add_argument("--fit-min-d", type=float, default=3.0)
    parser.add_argument("--fit-max-d", type=float, default=15.0)
    args = parser.parse_args()

    output_dir = args.case / "src" / "output"
    shape = (args.nx, args.ny, args.nz)
    mean_u = np.memmap(output_dir / "ta_u.bin", dtype=np.float32, mode="r", shape=shape, order="F")
    mean_u2 = np.memmap(output_dir / "ta_u2.bin", dtype=np.float32, mode="r", shape=shape, order="F")
    if not np.isfinite(mean_u).all() or not np.isfinite(mean_u2).all():
        raise ValueError("legacy time-average fields contain NaN or Inf")

    dx = args.lx_m / args.nx
    dy = args.ly_m / args.ny
    dz = args.lz_m / (args.nz - 1)
    # Legacy u nodes use i*dx and j*dy with one-based Fortran indices; vertical
    # uv nodes begin at dz/2 in the recomposed field.
    x = (np.arange(args.nx) + 1.0) * dx
    y = (np.arange(args.ny) + 1.0) * dy
    z = (np.arange(args.nz) + 0.5) * dz

    hub_u_yx = interpolate_plane(mean_u, z, args.hub_height_m, axis=2)
    centerline_u = interpolate_plane(hub_u_yx, y, args.turbine_y_m, axis=1)
    x_over_d = (x - args.turbine_x_m) / args.diameter_m
    les_deficit = 1.0 - centerline_u / args.prescribed_uinf_m_s

    rotor_radius = 0.5 * args.diameter_m
    yy, zz = np.meshgrid(y, z, indexing="ij")
    rotor = (yy - args.turbine_y_m) ** 2 + (zz - args.hub_height_m) ** 2 <= rotor_radius**2
    upstream_target = args.turbine_x_m - 1.5 * args.diameter_m
    upstream_i = int(np.argmin(np.abs(x - upstream_target)))
    variance = np.maximum(np.asarray(mean_u2[upstream_i]) - np.asarray(mean_u[upstream_i]) ** 2, 0.0)
    disk_mean = float(np.asarray(mean_u[upstream_i])[rotor].mean())
    incoming_ti = float(np.sqrt(variance[rotor].mean()) / disk_mean)
    ti_k = 0.38 * incoming_ti + 0.004

    fit = (x_over_d >= args.fit_min_d) & (x_over_d <= args.fit_max_d)
    candidates = np.linspace(0.01, 0.20, 1901)
    errors = np.array([
        np.nanmean((gaussian_deficit(x_over_d[fit], args.ct, k) - les_deficit[fit]) ** 2)
        for k in candidates
    ])
    fitted_k = float(candidates[np.nanargmin(errors)])
    ti_model = gaussian_deficit(x_over_d, args.ct, ti_k)
    fitted_model = gaussian_deficit(x_over_d, args.ct, fitted_k)
    ti_rmse = float(np.sqrt(np.nanmean((ti_model[fit] - les_deficit[fit]) ** 2)))
    fitted_rmse = float(np.sqrt(np.nanmean((fitted_model[fit] - les_deficit[fit]) ** 2)))

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "legacy_adm_gaussian_comparison.csv"
    np.savetxt(
        csv_path,
        np.column_stack((x, x_over_d, centerline_u, les_deficit, ti_model, fitted_model)),
        delimiter=",",
        header="x_m,x_over_D,legacy_mean_u_m_s,legacy_deficit,gaussian_ti_deficit,gaussian_fitted_deficit",
        comments="",
    )

    figure, axis = plt.subplots(figsize=(8.6, 4.9), constrained_layout=True)
    downstream = x_over_d >= 0.0
    axis.plot(x_over_d[downstream], les_deficit[downstream], linewidth=2.2, label="Legacy LES, time mean")
    axis.plot(x_over_d[downstream], ti_model[downstream], "--", linewidth=1.8,
              label=fr"Gaussian from $I_u$, $k={ti_k:.3f}$")
    axis.plot(x_over_d[downstream], fitted_model[downstream], ":", linewidth=2.2,
              label=fr"Gaussian fit, $k={fitted_k:.3f}$")
    axis.axvspan(args.fit_min_d, args.fit_max_d, color="0.5", alpha=0.08, label="fit interval")
    axis.axhline(0.0, color="0.5", linewidth=0.8)
    axis.set(
        xlabel=r"Downstream distance $(x-x_T)/D$",
        ylabel=r"Centerline deficit $1-\overline{u}/U_\infty$",
        title="DTU 10 MW prescribed ADM: legacy LES vs Gaussian wake",
        xlim=(0.0, float(x_over_d.max())),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(args.output / "legacy_adm_gaussian_comparison.png", dpi=200)
    plt.close(figure)

    summary = {
        "quantity": "hub-height lateral-centerline time-mean streamwise deficit",
        "prescribed_uinf_m_s": args.prescribed_uinf_m_s,
        "ct": args.ct,
        "incoming_ti": incoming_ti,
        "incoming_ti_plane_x_m": float(x[upstream_i]),
        "incoming_disk_mean_u_m_s": disk_mean,
        "fit_interval_D": [args.fit_min_d, args.fit_max_d],
        "ti_based_gaussian": {"k": ti_k, "rmse_deficit": ti_rmse},
        "fitted_gaussian": {"k": fitted_k, "rmse_deficit": fitted_rmse},
    }
    (args.output / "legacy_adm_gaussian_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
