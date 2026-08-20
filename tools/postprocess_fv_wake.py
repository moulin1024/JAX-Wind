#!/usr/bin/env python3
"""Post-process an FV turbine run against the Bastankhah Gaussian wake model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def gaussian_deficit(
    x_over_d: np.ndarray,
    ct: float,
    expansion_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return centerline deficit and Gaussian width, both nondimensional."""
    root = np.sqrt(1.0 - ct)
    beta = 0.5 * (1.0 + root) / root
    sigma_over_d = expansion_rate * x_over_d + 0.2 * np.sqrt(beta)
    radicand = 1.0 - ct / (8.0 * sigma_over_d**2)
    deficit = 1.0 - np.sqrt(np.maximum(radicand, 0.0))
    valid = (x_over_d >= 0.0) & (radicand >= 0.0)
    return np.where(valid, deficit, np.nan), sigma_over_d


def rotor_core_deficit(
    centerline_deficit: np.ndarray,
    sigma_over_d: np.ndarray,
) -> np.ndarray:
    """Area-average a Gaussian deficit over a radius-D/2 disk."""
    radius_over_d = 0.5
    factor = 2.0 * sigma_over_d**2 / radius_over_d**2
    factor *= 1.0 - np.exp(
        -(radius_over_d**2) / (2.0 * sigma_over_d**2)
    )
    return centerline_deficit * factor


def smooth(values: np.ndarray, points: int) -> np.ndarray:
    """Centred moving average with edge extension."""
    points = max(1, int(points))
    if points % 2 == 0:
        points += 1
    half = points // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.ones(points) / points, mode="valid")


def interpolate_yz(
    values: np.ndarray,
    *,
    y_m: float,
    z_m: float,
    dy_m: float,
    dz_m: float,
) -> np.ndarray:
    """Bilinearly interpolate a ``(z, y, ...)`` cell-centred field."""
    nz, ny = values.shape[:2]
    y_index = y_m / dy_m - 0.5
    y_lower = int(np.floor(y_index)) % ny
    y_weight = y_index - np.floor(y_index)
    y_upper = (y_lower + 1) % ny
    z_index = np.clip(z_m / dz_m - 0.5, 0.0, nz - 1.0)
    z_lower = int(np.floor(z_index))
    z_upper = min(z_lower + 1, nz - 1)
    z_weight = z_index - z_lower
    lower = (1.0 - y_weight) * values[z_lower, y_lower] + y_weight * values[
        z_lower, y_upper
    ]
    upper = (1.0 - y_weight) * values[z_upper, y_lower] + y_weight * values[
        z_upper, y_upper
    ]
    return (1.0 - z_weight) * lower + z_weight * upper


def precursor_statistics(
    path: Path,
    rotor_mask: np.ndarray,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Stream the precursor plane to obtain mean, variance, and rotor TI."""
    velocity = np.load(path, mmap_mode="r")
    total = np.zeros(velocity.shape[1:], dtype=np.float64)
    total_square = np.zeros_like(total)
    count = 0
    for start in range(0, velocity.shape[0], batch_size):
        values = np.asarray(velocity[start : start + batch_size], dtype=np.float64)
        total += values.sum(axis=0)
        total_square += np.square(values).sum(axis=0)
        count += values.shape[0]
    mean = total / count
    variance = np.maximum(total_square / count - mean * mean, 0.0)
    rotor_mean = float(mean[rotor_mask].mean())
    turbulence_intensity = float(
        np.sqrt(variance[rotor_mask].mean()) / rotor_mean
    )
    return mean, variance, turbulence_intensity


def realized_thrust_coefficient(workflow, disk, velocity, reference_velocity: float):
    """Evaluate the shared AD-BEM kernel and infer its freestream Ct."""
    import jax
    import jax.numpy as jnp

    from jaxwind._jax.wind import build_blade_element_disk_kernel
    from jaxwind.fv import StaggeredVelocity, cell_velocity

    grid = workflow.case.physical.physical_grid
    staggered = StaggeredVelocity(
        jnp.asarray(velocity["velocity_x"]),
        jnp.asarray(velocity["velocity_y"]),
        jnp.asarray(velocity["velocity_z"]),
    )
    u, v, _ = cell_velocity(staggered)
    kernel = build_blade_element_disk_kernel(
        grid=grid,
        axis_name="fv_wake_post",
        partition_count=1,
    )

    def local(u_local, v_local, w_upper_local):
        dtype = u_local.dtype
        return kernel(
            u_local,
            v_local,
            w_upper_local,
            disk.x,
            disk.y,
            disk.z,
            disk.blade_count,
            disk.hub_radius,
            disk.tip_radius,
            disk.angular_velocity,
            jnp.asarray(disk.element_smoothing_widths, dtype=dtype),
            jnp.asarray(disk.element_radii, dtype=dtype),
            jnp.asarray(disk.element_widths, dtype=dtype),
            jnp.asarray(disk.element_chords, dtype=dtype),
            jnp.asarray(disk.element_twist_degrees, dtype=dtype),
            jnp.asarray(disk.element_airfoil_ids, dtype=jnp.int32),
            jnp.asarray(disk.polar_alpha_degrees, dtype=dtype),
            jnp.asarray(disk.polar_lift_coefficients, dtype=dtype),
            jnp.asarray(disk.polar_drag_coefficients, dtype=dtype),
            disk.pitch_degrees,
            disk.tip_loss,
            disk.root_loss,
        )

    mapped = jax.pmap(local, axis_name="fv_wake_post")
    values = mapped(u[None], v[None], staggered.z[1:][None])
    forces = np.asarray(values[3][0], dtype=np.float64)
    sampled = np.asarray(values[4][0], dtype=np.float64)
    thrust = -float(forces[:, 0].sum())
    area = np.pi * disk.tip_radius**2
    ct = thrust / (0.5 * reference_velocity**2 * area)
    return ct, thrust, sampled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fit-min-d", type=float, default=4.0)
    parser.add_argument("--fit-max-d", type=float, default=12.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    from applications.fv_abl.workflow import (
        _build_turbine_definition,
        load_workflow,
    )
    from jaxwind.domain import ScaleSystem

    workflow = load_workflow(arguments.config)
    options = workflow.turbine
    if options is None:
        raise ValueError("wake post-processing requires a configured turbine")
    case = workflow.case.physical
    grid = case.physical_grid
    output = arguments.output or workflow.options.output_directory / "wake_recovery"
    output.mkdir(parents=True, exist_ok=True)
    result_path = workflow.options.output_directory / "main_final.npz"
    inflow_path = workflow.options.output_directory / "precursor_inflow" / "x_velocity.npy"

    turbine = _build_turbine_definition(workflow)
    disk = turbine.to_actuator_disk(scales=ScaleSystem(1.0, 1.0))
    diameter = 2.0 * disk.tip_radius

    x = (np.arange(grid.nx) + 0.5) * grid.dx
    y = (np.arange(grid.ny) + 0.5) * grid.dy
    z = (np.arange(grid.nz) + 0.5) * grid.dz
    y_distance = np.abs(y - disk.y)
    y_distance = np.minimum(y_distance, grid.ly - y_distance)
    rotor_mask = (
        (z[:, None] - disk.z) ** 2 + y_distance[None, :] ** 2
        <= disk.tip_radius**2
    )
    inflow_mean, inflow_variance, turbulence_intensity = precursor_statistics(
        inflow_path,
        rotor_mask,
    )
    rotor_reference_velocity = float(inflow_mean[rotor_mask].mean())
    hub_reference_velocity = float(
        interpolate_yz(
            inflow_mean,
            y_m=disk.y,
            z_m=disk.z,
            dy_m=grid.dy,
            dz_m=grid.dz,
        )
    )

    with np.load(result_path) as archive:
        stored = {name: np.asarray(archive[name]) for name in archive.files}
    u_cell = 0.5 * (stored["velocity_x"][..., :-1] + stored["velocity_x"][..., 1:])
    instantaneous_centerline_velocity = interpolate_yz(
        u_cell,
        y_m=disk.y,
        z_m=disk.z,
        dy_m=grid.dy,
        dz_m=grid.dz,
    )
    frame_path = workflow.options.output_directory / "main_flow_frames.npz"
    frames_used = 0
    frame_time_range = None
    if frame_path.exists():
        with np.load(frame_path) as frame_archive:
            centre_frames = np.asarray(
                frame_archive["u_center_zx"], dtype=np.float64
            )
            frame_times = np.asarray(
                frame_archive["time_seconds"], dtype=np.float64
            )
        start_frame = centre_frames.shape[0] // 2
        selected_frames = centre_frames[start_frame:]
        z_index = np.clip(disk.z / grid.dz - 0.5, 0.0, grid.nz - 1.0)
        z_lower = int(np.floor(z_index))
        z_upper = min(z_lower + 1, grid.nz - 1)
        z_weight = z_index - z_lower
        centerline_velocity = np.mean(
            (1.0 - z_weight) * selected_frames[:, z_lower]
            + z_weight * selected_frames[:, z_upper],
            axis=0,
        )
        frames_used = int(selected_frames.shape[0])
        frame_time_range = [
            float(frame_times[start_frame]),
            float(frame_times[-1]),
        ]
    else:
        centerline_velocity = instantaneous_centerline_velocity
    rotor_core_velocity = u_cell[rotor_mask].mean(axis=0)
    smoothing_points = max(3, int(round(diameter / grid.dx)))
    if smoothing_points % 2 == 0:
        smoothing_points += 1
    centerline_smoothed = smooth(centerline_velocity, smoothing_points)
    rotor_core_smoothed = smooth(rotor_core_velocity, smoothing_points)
    centerline_ratio = centerline_smoothed / hub_reference_velocity
    rotor_core_ratio = rotor_core_smoothed / rotor_reference_velocity
    x_over_d = (x - disk.x) / diameter

    ct_raw, thrust, sampled = realized_thrust_coefficient(
        workflow,
        disk,
        stored,
        rotor_reference_velocity,
    )
    ct = float(np.clip(ct_raw, 1.0e-6, 0.999))
    ti_expansion = 0.38 * turbulence_intensity + 0.004
    fit = (x_over_d >= arguments.fit_min_d) & (x_over_d <= arguments.fit_max_d)
    candidates = np.linspace(0.01, 0.20, 1901)
    errors = np.full_like(candidates, np.inf)
    observed_deficit = 1.0 - centerline_ratio
    for index, expansion in enumerate(candidates):
        prediction, _ = gaussian_deficit(x_over_d[fit], ct, expansion)
        if np.all(np.isfinite(prediction)):
            errors[index] = np.mean((prediction - observed_deficit[fit]) ** 2)
    fitted_expansion = float(candidates[np.argmin(errors)])
    ti_deficit, ti_sigma = gaussian_deficit(x_over_d, ct, ti_expansion)
    fit_deficit, fit_sigma = gaussian_deficit(x_over_d, ct, fitted_expansion)
    ti_core_deficit = rotor_core_deficit(ti_deficit, ti_sigma)
    fit_core_deficit = rotor_core_deficit(fit_deficit, fit_sigma)

    downstream = x_over_d >= 0.0
    csv_path = output / "wake_recovery.csv"
    np.savetxt(
        csv_path,
        np.column_stack(
            (
                x,
                x_over_d,
                centerline_velocity,
                centerline_smoothed,
                centerline_ratio,
                rotor_core_velocity,
                rotor_core_smoothed,
                rotor_core_ratio,
                1.0 - ti_deficit,
                1.0 - fit_deficit,
                1.0 - ti_core_deficit,
                1.0 - fit_core_deficit,
            )
        ),
        delimiter=",",
        header=(
            "x_m,x_over_D,les_centerline_u_m_s,les_centerline_smoothed_u_m_s,"
            "les_centerline_u_over_uref,les_rotor_core_u_m_s,"
            "les_rotor_core_smoothed_u_m_s,les_rotor_core_u_over_uref,"
            "gaussian_ti_centerline_u_over_uref,gaussian_fit_centerline_u_over_uref,"
            "gaussian_ti_rotor_core_u_over_uref,gaussian_fit_rotor_core_u_over_uref"
        ),
        comments="",
    )

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True, constrained_layout=True)
    panels = (
        (
            axes[0],
            centerline_velocity / hub_reference_velocity,
            centerline_ratio,
            1.0 - ti_deficit,
            1.0 - fit_deficit,
            r"hub-centerline velocity $u/U_{ref}$",
        ),
        (
            axes[1],
            rotor_core_velocity / rotor_reference_velocity,
            rotor_core_ratio,
            1.0 - ti_core_deficit,
            1.0 - fit_core_deficit,
            r"rotor-core mean velocity $\langle u\rangle_A/U_{ref,A}$",
        ),
    )
    for axis, raw, filtered, ti_model, fit_model, ylabel in panels:
        axis.plot(x_over_d[downstream], raw[downstream], color="0.70", linewidth=1.0, label="FV raw")
        axis.plot(x_over_d[downstream], filtered[downstream], color="#0057b8", linewidth=2.2, label="FV, D-wide smoothed")
        axis.plot(x_over_d[downstream], ti_model[downstream], "--", color="#d55e00", linewidth=2.0, label=f"Gaussian, TI-based k={ti_expansion:.3f}")
        axis.plot(x_over_d[downstream], fit_model[downstream], ":", color="#009e73", linewidth=2.2, label=f"Gaussian fit k={fitted_expansion:.3f}")
        axis.axhline(1.0, color="0.25", linewidth=0.8)
        axis.axvspan(arguments.fit_min_d, arguments.fit_max_d, color="0.5", alpha=0.08)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].set_title(
        f"{options.model} wake recovery, FV-GMG "
        f"(Ct={ct_raw:.3f}, Iu={100*turbulence_intensity:.1f}%)"
    )
    axes[0].legend(ncol=2, fontsize=8.5)
    axes[1].set_xlabel(r"downstream distance $(x-x_T)/D$")
    axes[1].set_xlim(0.0, float(x_over_d.max()))
    figure_path = output / "wake_recovery.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    ti_rmse = float(
        np.sqrt(np.mean(((1.0 - ti_deficit[fit]) - centerline_ratio[fit]) ** 2))
    )
    fit_rmse = float(
        np.sqrt(np.mean(((1.0 - fit_deficit[fit]) - centerline_ratio[fit]) ** 2))
    )
    summary = {
        "source": str(result_path),
        "quantity": (
            "time-averaged hub-centerline wake over latter half of saved frames; "
            "final instantaneous rotor-core wake; D-wide streamwise smoothing"
        ),
        "frames_used": frames_used,
        "frame_time_range_seconds": frame_time_range,
        "turbine": {
            "x_m": disk.x,
            "y_m": disk.y,
            "hub_height_m": disk.z,
            "rotor_diameter_m": diameter,
            "realized_thrust_per_density_m4_s2": thrust,
            "realized_ct": ct_raw,
            "model_ct": ct,
            "mean_sampled_disk_axial_velocity_m_s": float(sampled[:, 0].mean()),
        },
        "precursor": {
            "samples": int(np.load(inflow_path, mmap_mode="r").shape[0]),
            "hub_reference_velocity_m_s": hub_reference_velocity,
            "rotor_reference_velocity_m_s": rotor_reference_velocity,
            "rotor_streamwise_turbulence_intensity": turbulence_intensity,
            "variance_min_max_m2_s2": [
                float(inflow_variance.min()),
                float(inflow_variance.max()),
            ],
        },
        "gaussian_model": {
            "name": "Bastankhah Gaussian wake with Niayifar initial width",
            "fit_interval_D": [arguments.fit_min_d, arguments.fit_max_d],
            "ti_based_expansion_rate": ti_expansion,
            "far_wake_valid_from_D": float(
                (np.sqrt(ct / 8.0) - 0.2 * np.sqrt(
                    0.5 * (1.0 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct)
                )) / ti_expansion
            ),
            "ti_based_centerline_velocity_ratio_rmse": ti_rmse,
            "fitted_expansion_rate": fitted_expansion,
            "fitted_centerline_velocity_ratio_rmse": fit_rmse,
        },
        "smoothing_points": smoothing_points,
        "outputs": {"figure": str(figure_path), "csv": str(csv_path)},
    }
    summary_path = output / "wake_recovery_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
