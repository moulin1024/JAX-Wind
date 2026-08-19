#!/usr/bin/env python3
"""Compare the sampled LES hub-height wake centerline with a Gaussian model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def gaussian_centerline_deficit(x_over_d: np.ndarray, ct: float, k: float) -> np.ndarray:
    """Bastankhah Gaussian-wake centerline deficit with the Niayifar offset."""

    beta = 0.5 * (1.0 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct)
    sigma_over_d = k * x_over_d + 0.2 * np.sqrt(beta)
    radicand = 1.0 - ct / (8.0 * sigma_over_d**2)
    return np.where(radicand >= 0.0, 1.0 - np.sqrt(np.maximum(radicand, 0.0)), np.nan)


def precursor_rotor_turbulence_intensity(
    path: Path,
    *,
    section: str,
    dt_seconds: float,
    spinup_seconds: float,
    ly_m: float,
    lz_m: float,
    turbine_y_m: float,
    hub_height_m: float,
    rotor_diameter_m: float,
) -> tuple[float, dict[str, object]]:
    """Measure streamwise rotor-area TI from a precursor HDF5 recording."""

    import h5py

    if dt_seconds <= 0.0:
        raise ValueError("precursor time step must be positive")
    with h5py.File(path, "r") as archive:
        names = tuple(
            value.decode() if isinstance(value, bytes) else str(value)
            for value in archive["sections/name"][:]
        )
        if section not in names:
            raise ValueError(
                f"precursor section {section!r} is not one of {names}"
            )
        section_index = names.index(section)
        step = np.asarray(archive["step"], dtype=np.int64)
        elapsed = (step - step[0]) * dt_seconds
        selected = np.flatnonzero(elapsed >= spinup_seconds)
        if selected.size == 0:
            raise ValueError("precursor spinup excludes every recorded sample")
        sample_start = int(selected[0])
        execution_ly = float(archive.attrs["ly"])
        execution_lz = float(archive.attrs["lz"])
        y = np.asarray(archive["coordinates/y"]) * ly_m / execution_ly
        z = np.asarray(archive["coordinates/z_cell"]) * lz_m / execution_lz
        y_distance = np.abs(y - turbine_y_m)
        y_distance = np.minimum(y_distance, ly_m - y_distance)
        rotor_radius = 0.5 * rotor_diameter_m
        rotor = (
            (z[:, None] - hub_height_m) ** 2 + y_distance[None, :] ** 2
            <= rotor_radius**2
        )
        if not np.any(rotor):
            raise ValueError("precursor grid contains no rotor-area cells")
        velocity = archive["velocity"]
        plane_count = velocity.shape[-1]
        total = np.zeros((z.size, y.size, plane_count), dtype=np.float64)
        total_square = np.zeros_like(total)
        sample_count = 0
        batch_size = 1 if velocity.chunks is None else velocity.chunks[0]
        for lower in range(sample_start, len(step), batch_size):
            upper = min(lower + batch_size, len(step))
            values = np.asarray(
                velocity[lower:upper, section_index, 0, :, :, :],
                dtype=np.float64,
            )
            sample_count += values.shape[0]
            total += values.sum(axis=0)
            total_square += np.square(values).sum(axis=0)
        mean = total / sample_count
        variance = np.maximum(total_square / sample_count - mean * mean, 0.0)
        rotor_mean = float(mean[rotor, :].mean())
        turbulence_intensity = float(
            np.sqrt(variance[rotor, :].mean()) / rotor_mean
        )
        details: dict[str, object] = {
            "recording": str(path),
            "section": section,
            "samples": sample_count,
            "time_window_seconds": [
                float(elapsed[sample_start]),
                float(elapsed[-1]),
            ],
            "section_planes": plane_count,
            "rotor_cells_per_plane": int(rotor.sum()),
            "streamwise_turbulence_intensity": turbulence_intensity,
        }
        return turbulence_intensity, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turbine-x-m", type=float, default=1000.0)
    parser.add_argument("--hub-height-m", type=float, default=119.0)
    parser.add_argument("--rotor-diameter-m", type=float, default=178.3)
    parser.add_argument("--ct-prime", type=float, default=4.0 / 3.0)
    parser.add_argument(
        "--ct",
        type=float,
        help="freestream thrust coefficient; overrides conversion from --ct-prime",
    )
    parser.add_argument(
        "--reference-velocity-m-s",
        type=float,
        help="fixed velocity used to normalize the deficit instead of the sampled upstream mean",
    )
    parser.add_argument("--lx-m", type=float, default=4096.0)
    parser.add_argument("--lz-m", type=float, default=1024.0)
    parser.add_argument("--spinup-seconds", type=float, default=600.0)
    parser.add_argument("--fit-min-d", type=float, default=4.0)
    parser.add_argument("--fit-max-d", type=float, default=10.0)
    parser.add_argument("--reference-k", type=float, default=0.05)
    turbulence = parser.add_mutually_exclusive_group()
    turbulence.add_argument(
        "--incoming-ti",
        type=float,
        help="incoming streamwise turbulence intensity as a fraction; when set, use k=0.38 Iu+0.004",
    )
    turbulence.add_argument(
        "--precursor-recording",
        type=Path,
        help="measure rotor-area incoming TI from this precursor HDF5 file",
    )
    parser.add_argument("--precursor-section", default="inflow")
    parser.add_argument("--precursor-dt-seconds", type=float, default=0.1)
    parser.add_argument("--turbine-y-m", type=float, default=512.0)
    parser.add_argument("--ly-m", type=float, default=1024.0)
    args = parser.parse_args()
    incoming_ti = args.incoming_ti
    precursor_statistics = None
    if args.precursor_recording is not None:
        incoming_ti, precursor_statistics = precursor_rotor_turbulence_intensity(
            args.precursor_recording,
            section=args.precursor_section,
            dt_seconds=args.precursor_dt_seconds,
            spinup_seconds=args.spinup_seconds,
            ly_m=args.ly_m,
            lz_m=args.lz_m,
            turbine_y_m=args.turbine_y_m,
            hub_height_m=args.hub_height_m,
            rotor_diameter_m=args.rotor_diameter_m,
        )
    reference_k = (
        args.reference_k
        if incoming_ti is None
        else 0.38 * incoming_ti + 0.004
    )

    with np.load(args.frames) as archive:
        velocity = archive["u_xz_m_s"]
        elapsed = archive["elapsed_seconds"]
    nz, nx = velocity.shape[1:]
    x = (np.arange(nx) + 0.5) * args.lx_m / nx
    z = (np.arange(nz) + 0.5) * args.lz_m / nz
    upper = int(np.searchsorted(z, args.hub_height_m))
    lower = upper - 1
    weight = (args.hub_height_m - z[lower]) / (z[upper] - z[lower])
    hub_velocity = (1.0 - weight) * velocity[:, lower, :] + weight * velocity[:, upper, :]
    sample_mask = elapsed >= args.spinup_seconds
    mean_velocity = hub_velocity[sample_mask].mean(axis=0)

    upstream = x <= args.turbine_x_m - args.rotor_diameter_m
    sampled_reference_velocity = float(mean_velocity[upstream].mean())
    reference_velocity = (
        sampled_reference_velocity
        if args.reference_velocity_m_s is None
        else args.reference_velocity_m_s
    )
    x_over_d = (x - args.turbine_x_m) / args.rotor_diameter_m
    les_deficit = 1.0 - mean_velocity / reference_velocity
    fit = (x_over_d >= args.fit_min_d) & (x_over_d <= args.fit_max_d)

    ct = (
        args.ct_prime / (1.0 + args.ct_prime / 4.0) ** 2
        if args.ct is None
        else args.ct
    )
    candidates = np.linspace(0.01, 0.20, 1901)
    errors = []
    for k in candidates:
        prediction = gaussian_centerline_deficit(x_over_d[fit], ct, k)
        errors.append(
            np.inf
            if np.any(~np.isfinite(prediction))
            else np.mean((prediction - les_deficit[fit]) ** 2)
        )
    errors = np.asarray(errors)
    fitted_k = float(candidates[np.argmin(errors)])
    reference_model = gaussian_centerline_deficit(x_over_d, ct, reference_k)
    fitted_model = gaussian_centerline_deficit(x_over_d, ct, fitted_k)
    reference_rmse = float(np.sqrt(np.mean((reference_model[fit] - les_deficit[fit]) ** 2)))
    fitted_rmse = float(np.sqrt(np.mean((fitted_model[fit] - les_deficit[fit]) ** 2)))

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "hub_height_wake_comparison.csv"
    np.savetxt(
        csv_path,
        np.column_stack((x, x_over_d, mean_velocity, les_deficit, reference_model, fitted_model)),
        delimiter=",",
        header="x_m,x_over_D,les_mean_u_m_s,les_deficit,gaussian_ti_based_deficit,gaussian_fitted_deficit",
        comments="",
    )

    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    downstream = x_over_d >= 0.0
    domain_end_d = (
        args.lx_m - args.turbine_x_m
    ) / args.rotor_diameter_m
    axis.plot(x_over_d[downstream], les_deficit[downstream], label="LES time mean", linewidth=2.2)
    reference_label = (
        f"TI-based Gaussian, k={reference_k:.3f}"
        if incoming_ti is not None
        else f"Gaussian, k={reference_k:.3f}"
    )
    axis.plot(x_over_d[downstream], reference_model[downstream], "--", label=reference_label)
    axis.plot(x_over_d[downstream], fitted_model[downstream], ":", label=f"Gaussian fit, k={fitted_k:.3f}", linewidth=2.2)
    axis.axvspan(args.fit_min_d, args.fit_max_d, alpha=0.08, label="fit interval")
    axis.axhline(0.0, color="0.5", linewidth=0.8)
    axis.set(
        xlabel=r"downstream distance $(x-x_T)/D$",
        ylabel=r"hub-height centerline deficit $1-\overline{u}/U_{ref}$",
        title="DTU 10-MW wake: LES vs Gaussian centerline model",
        xlim=(0.0, domain_end_d),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(args.output / "hub_height_wake_comparison.png", dpi=180)
    plt.close(figure)

    summary = {
        "quantity": "center-y, hub-height, time-mean streamwise wake centerline",
        "frames_used": int(sample_mask.sum()),
        "time_window_seconds": [float(elapsed[sample_mask][0]), float(elapsed[sample_mask][-1])],
        "reference_velocity_m_s": reference_velocity,
        "sampled_upstream_reference_velocity_m_s": sampled_reference_velocity,
        "ct_prime": args.ct_prime,
        "ct": ct,
        "fit_interval_D": [args.fit_min_d, args.fit_max_d],
        "domain_end_D": domain_end_d,
        "incoming_turbulence_intensity": incoming_ti,
        "precursor_inflow_statistics": precursor_statistics,
        "reference_gaussian": {"k": reference_k, "rmse_deficit": reference_rmse},
        "fitted_gaussian": {"k": fitted_k, "rmse_deficit": fitted_rmse},
    }
    (args.output / "hub_height_wake_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
