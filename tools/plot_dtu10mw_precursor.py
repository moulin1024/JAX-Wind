"""Compare a completed historical DTU10MW precursor with the neutral log law.

Run on a compute node with NumPy and Matplotlib available. Profiles are sampled
from the complete horizontal cell layers, not inferred from the inlet plane.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cell_average_log(z_faces, roughness):
    """Integral of log(z/z0) over each cell, matching CELL_AVERAGE wall sampling."""
    primitive = np.zeros_like(z_faces, dtype=np.float64)
    positive = z_faces > 0.
    primitive[positive] = z_faces[positive] * (np.log(z_faces[positive] / roughness) - 1.)
    return np.diff(primitive) / np.diff(z_faces)


def read_csv(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def transport_label(case):
    numerics, time = case["numerics"], case["time"]
    scheme = numerics["momentum_advection_scheme"].upper()
    integrator = numerics["time_integration"].upper()
    timing = f"CFL {time['cfl']:g}" if "cfl" in time else f"dt {time['dt_seconds']:g} s"
    return f"{scheme} / {integrator} / {timing}"


def analyze(run, output):
    from jaxwind.config.document import load_case
    from jaxwind.config.abl import load_fv_abl
    from jaxwind.config.abl_resolved import resolved
    from jaxwind.io.checkpoint import checkpoint_metadata

    run, output = run.resolve(), output.resolve()
    summaries = {}
    for stage in ("warmup", "precursor"):
        summaries[stage] = json.loads((run / stage / "summary.json").read_text())
        if summaries[stage]["status"] != "complete":
            raise ValueError(f"{stage} is incomplete")
    case = load_case(run / "precursor/resolved_case.toml").document
    configured = load_fv_abl(run / "precursor/resolved_case.toml")
    numerical = resolved(configured)
    numerical.update(wall_gradient_correction=configured.options.wall_gradient_correction, wall_averaging=configured.options.wall_averaging, wall_sampling="cell-average")
    metadata = checkpoint_metadata(run / "precursor/checkpoint.npz")
    profiles = read_csv(run / "precursor/profiles.csv")
    history = read_csv(run / "precursor/history.csv")
    warm_history = read_csv(run / "warmup/history.csv")
    warm_profile = read_csv(run / "warmup/profiles.csv")
    z, u = profiles["z_m"], profiles["mean_u_m_s"]
    if not np.all(np.isfinite(u)):
        raise ValueError("precursor velocity profile is non-finite")
    flow = case["physics"]["flow"]
    roughness, kappa = flow["roughness_length_m"], flow["von_karman"]
    height = case["mesh"]["lengths_m"][2]
    acceleration = flow["pressure_acceleration_m_s2"][0]
    expected_ustar = float(np.sqrt(acceleration * height))
    z_faces = np.asarray(metadata["mesh"]["z_faces"])
    log_average = cell_average_log(z_faces, roughness)
    log_u = expected_ustar / kappa * log_average
    wall_ustar = float(np.sqrt(-np.mean(history["surface_uw_m2_s2"])))
    measured_wall_log_u = wall_ustar / kappa * log_average
    band = (z >= 20.) & (z <= .1 * height)
    if np.count_nonzero(band) < 3:
        raise ValueError("not enough levels in the declared 20 m to 0.1 H comparison band")
    fit = np.polyfit(log_average[band], u[band], 1)
    fitted = np.polyval(fit, log_average[band])
    residual = u - log_u
    hub_height = case["physics"]["turbine"]["hub_height_m"]
    expected_samples = case["time"]["steps"] // case["diagnostics"]["sample_every_steps"]
    actual_samples = summaries["precursor"]["runtime"]["profile_samples"]
    if actual_samples != expected_samples:
        raise ValueError(f"expected {expected_samples} precursor profiles, found {actual_samples}")
    inflow = json.loads((run / "precursor/inflow/metadata.json").read_text())
    adaptive = "cfl" in case["time"]
    if not adaptive and inflow["samples"] != case["time"]["steps"]:
        raise ValueError("inflow recording does not cover the complete precursor")
    from jaxwind.io.recording import InflowReader
    InflowReader(run / "precursor/inflow")
    recorded_dt = []
    first_time, recorded_end = None, None
    for chunk in inflow["chunks"]:
        with np.load(run / "precursor/inflow" / chunk["file"], allow_pickle=False) as data:
            recorded_dt.extend(data["dt_seconds"].astype(np.float64))
            if first_time is None:
                first_time = float(data["time_seconds"][0])
            recorded_end = float(data["time_seconds"][-1] + data["dt_seconds"][-1])
    clock_tolerance = 8 * np.finfo(np.float32).eps * metadata["target_time"]
    if abs(first_time - metadata["initial_time"]) > clock_tolerance or abs(recorded_end - metadata["target_time"]) > clock_tolerance:
        raise ValueError("recorded inflow does not cover the requested physical-time window")
    plane_u = np.asarray(inflow["mean_x_velocity_profile_m_s"])
    duration = case["time"]["steps"] * case["time"]["dt_seconds"]
    bulk_endpoints = []
    for stage in ("warmup", "precursor"):
        with np.load(run / stage / "checkpoint.npz", allow_pickle=False) as fields:
            mean_u = np.mean(fields["state/velocity/x"], axis=(1, 2), dtype=np.float64)
        bulk_endpoints.append(float(np.dot(mean_u, np.diff(z_faces)) / height))
    bulk_change = bulk_endpoints[1] - bulk_endpoints[0]
    momentum_acceleration = acceleration + float(np.mean(history["surface_uw_m2_s2"])) / height
    errors = {
        "rmse_m_s": float(np.sqrt(np.mean(residual[band] ** 2))),
        "mean_bias_m_s": float(np.mean(residual[band])),
        "maximum_absolute_relative_error_percent": float(100. * np.max(np.abs(residual[band] / log_u[band]))),
        "relative_rmse_percent": float(100. * np.sqrt(np.mean(residual[band] ** 2)) / np.mean(log_u[band])),
    }
    continuation_path = run.parent / "continuation.json"
    continuation = json.loads(continuation_path.read_text()) if continuation_path.exists() else None
    warm_metadata = checkpoint_metadata(run / "warmup/checkpoint.npz")
    continued_steps = summaries["warmup"]["step"] - warm_metadata["initial_step"]
    mesh_label = " × ".join(str(n) for n in case["mesh"]["cells"])
    scheme_label = f"{configured.options.momentum_advection_scheme.upper()} / {configured.options.time_integration.upper()}"
    correction_label = "Porté-Agel on" if configured.options.wall_gradient_correction else "Porté-Agel off"
    band_label = f"{z[band][0]:g}–{z[band][-1]:g} m"
    if continuation:
        continuation_note = (
            f"Checkpoint continuation from {continuation['source_time_seconds']/3600.:.4f} h; "
            f"the grid was changed from {continuation.get('source_mesh_cells_xyz', case['mesh']['cells'])} "
            f"to {case['mesh']['cells']} and the Porté-Agel correction was enabled then. "
            f"The corrected flow develops for {continuation['corrected_warmup_duration_seconds']/3600.:.4f} h before precursor sampling. "
        )
    else:
        continuation_note = ""
    report = {
        "continuation": continuation,
        "warmup_steps_in_this_continuation": continued_steps,
        "precursor_actual_steps": summaries["precursor"]["step"] - metadata["initial_step"],
        "run_directory": str(run),
        "mesh_cells_xyz": case["mesh"]["cells"],
        "numerics": {name: numerical[name] for name in ("momentum_advection_scheme", "scalar_advection_scheme", "time_integration", "pressure_backend", "fft_method", "dt_seconds", "dt_interpretation", "cfl_ceiling", "wall_gradient_correction", "wall_averaging", "wall_sampling")},
        "domain_m_xyz": case["mesh"]["lengths_m"],
        "warmup_seconds": summaries["warmup"]["time_seconds"],
        "precursor_duration_seconds": duration,
        "precursor_start_seconds": metadata["initial_time"],
        "precursor_end_seconds": summaries["precursor"]["time_seconds"],
        "averaging": {
            "definition": "Equal-volume mean over all x,y cells at each height, then equal-time mean of regularly spaced precursor samples.",
            "sample_count": actual_samples,
            "sample_interval_seconds": case["diagnostics"]["sample_every_steps"] * case["time"]["dt_seconds"],
            "first_sample_seconds": metadata["initial_time"] + case["diagnostics"]["sample_every_steps"] * case["time"]["dt_seconds"],
            "last_sample_seconds": summaries["precursor"]["time_seconds"],
            "inflow_plane_is_separate": True,
        },
        "reference": {
            "law": "U(z) = u_star/kappa * ln(z/z0); comparison uses its vertical cell integral",
            "roughness_length_m": roughness,
            "von_karman": kappa,
            "pressure_acceleration_m_s2": acceleration,
            "pressure_balance_ustar_m_s": expected_ustar,
            "mean_wall_stress_ustar_m_s": wall_ustar,
            "mean_instantaneous_ustar_m_s": summaries["precursor"]["runtime"]["ustar_m_s"],
            "wall_stress_to_pressure_balance_ratio": wall_ustar ** 2 / expected_ustar ** 2,
        },
        "surface_layer_comparison": {
            "declared_height_band_m": [20., .1 * height],
            "actual_height_band_m": [float(z[band][0]), float(z[band][-1])],
            "levels": int(np.count_nonzero(band)),
            **errors,
            "free_log_fit_ustar_m_s": float(kappa * fit[0]),
            "free_log_fit_intercept_m_s": float(fit[1]),
            "free_log_fit_r_squared": float(1. - np.sum((u[band] - fitted) ** 2) / np.sum((u[band] - np.mean(u[band])) ** 2)),
            "free_fit_note": "A free log fit is descriptive; agreement with the prescribed roughness and forcing is tested separately.",
        },
        "hub_height": {
            "height_m": hub_height,
            "mean_u_m_s": float(np.interp(hub_height, z, u)),
            "cell_average_log_law_m_s": float(np.interp(hub_height, z, log_u)),
            "point_log_law_m_s": float(expected_ustar / kappa * np.log(hub_height / roughness)),
        },
        "checks": {
            "warmup_final_cfl": summaries["warmup"]["final_cfl"],
            "precursor_final_cfl": summaries["precursor"]["final_cfl"],
            "maximum_precursor_divergence_s": float(np.max(history["maximum_divergence_s"])),
            "recorded_inflow_samples": inflow["samples"],
            "recorded_dt_min_seconds": float(np.min(recorded_dt)),
            "recorded_dt_max_seconds": float(np.max(recorded_dt)),
            "recorded_dt_median_seconds": float(np.median(recorded_dt)),
            "maximum_actual_step_cfl": float(np.max(history["block_maximum_cfl"])) if "block_maximum_cfl" in history.dtype.names else None,
            "bulk_u_at_precursor_start_m_s": bulk_endpoints[0],
            "bulk_u_at_precursor_end_m_s": bulk_endpoints[1],
            "bulk_u_change_during_precursor_m_s": bulk_change,
            "observed_bulk_acceleration_m_s2": bulk_change / duration,
            "pressure_minus_wall_drag_acceleration_m_s2": momentum_acceleration,
            "stress_profile_note": "The figure shows resolved covariance only. The existing SGS profile diagnostic does not apply the enabled wall-gradient correction, so its SGS/total-stress columns are excluded from this comparison. MUSCL numerical momentum transport is also excluded.",
            "inlet_plane_minus_volume_rmse_m_s": float(np.sqrt(np.mean((plane_u - u) ** 2))),
            "precursor_minus_last_warmup_hour_rmse_m_s": float(np.sqrt(np.mean((u - warm_profile["mean_u_m_s"]) ** 2))),
            "stationarity_note": "One hour is a finite averaging window; matching the log-law shape alone does not establish statistical convergence.",
        },
    }
    middle = .5 * (history["time_hours"][0] + history["time_hours"][-1])
    halves = (history[history["time_hours"] <= middle], history[history["time_hours"] > middle])
    report["checks"]["precursor_half_window_means"] = [
        {"ustar_m_s": float(np.mean(part["ustar_m_s"])),
         "integrated_resolved_tke_m3_s2": float(np.mean(part["integrated_resolved_tke_m3_s2"]))}
        for part in halves
    ]
    current_label = transport_label(case)
    previous_profile = None
    if continuation and continuation.get("comparison_run"):
        previous_run = Path(continuation["comparison_run"])
        previous_case = load_case(previous_run / "precursor/resolved_case.toml").document
        previous_label = transport_label(previous_case)
        previous_profile = read_csv(previous_run / "precursor/profiles.csv")
        np.testing.assert_allclose(previous_profile["z_m"], z)
        previous_u = previous_profile["mean_u_m_s"]
        report["comparison_with_previous"] = {
            "previous_run": str(previous_run),
            "previous_label": previous_label,
            "current_label": current_label,
            "previous_surface_rmse_m_s": float(np.sqrt(np.mean((previous_u[band] - log_u[band]) ** 2))),
            "current_surface_rmse_m_s": errors["rmse_m_s"],
            "previous_hub_u_m_s": float(np.interp(hub_height, z, previous_u)),
            "current_hub_u_m_s": report["hub_height"]["mean_u_m_s"],
            "note": continuation.get("comparison_note", "Same initial velocity checkpoint and physical averaging window. Advection, timestep control, and time integration changed together; this does not isolate advection alone."),
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    columns = {
        "z_m": z, "mean_u_m_s": u,
        "mean_v_m_s": profiles["mean_v_m_s"], "mean_w_m_s": profiles["mean_w_m_s"],
        "log_law_cell_average_m_s": log_u,
        "log_law_point_m_s": expected_ustar / kappa * np.log(z / roughness),
        "log_law_measured_wall_stress_m_s": measured_wall_log_u,
        "error_m_s": residual, "relative_error_percent": 100. * residual / log_u,
        "warmup_last_hour_mean_u_m_s": warm_profile["mean_u_m_s"],
        "recorded_inlet_mean_u_m_s": plane_u,
    }
    np.savetxt(output / "log_law_profile.csv", np.column_stack(list(columns.values())),
               delimiter=",", header=",".join(columns), comments="")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
    for ax in axes:
        ax.plot(u, z, color="#126782", lw=2.2, label="Precursor: volume + time mean")
        ax.plot(log_u, z, color="#d65f00", ls="--", lw=2., label=r"Log law: $u_*=0.4$ m/s, cell average")
        ax.plot(measured_wall_log_u, z, color="#9254a0", ls="-.", lw=1.4, label="Log law: measured wall stress")
        ax.plot(warm_profile["mean_u_m_s"], z, color="#56854a", ls=":", lw=1.7, label="Final warmup hour: mean")
        ax.set(xlabel=r"Mean streamwise velocity $U$ [m/s]", ylabel="Height z [m]")
        ax.grid(True, alpha=.2)
    axes[0].set(title="Full domain", ylim=(0., height))
    axes[0].legend(loc="upper left", fontsize=8.5)
    axes[1].set(title="Near surface (logarithmic height)", yscale="log", ylim=(z[0], .25 * height))
    axes[1].axhspan(20., .1 * height, alpha=.07, color="#126782")
    axes[1].text(.04, .94, f"{band_label} reference RMSE: {errors['rmse_m_s']:.3f} m/s\nHub-height mean: {report['hub_height']['mean_u_m_s']:.3f} m/s", transform=axes[1].transAxes, va="top", fontsize=9)
    fig.suptitle(f"DTU10MW precursor · {mesh_label} · {scheme_label} · {correction_label}\n10 h total warmup + 1 h averaging · {actual_samples} full-volume profile samples", fontsize=13)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"log_law_profile.{extension}", dpi=180)
    plt.close(fig)

    if previous_profile is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
        for ax in axes:
            ax.plot(u, z, color="#126782", lw=2., label=current_label)
            ax.plot(previous_profile["mean_u_m_s"], z, color="#6c757d", ls="-.", lw=1.8, label=previous_label)
            ax.plot(log_u, z, color="#d65f00", ls="--", lw=1.8, label="Prescribed log law (cell average)")
            ax.set(xlabel="Mean streamwise velocity [m/s]", ylabel="Height [m]")
            ax.grid(True, alpha=.2)
        axes[0].set(ylim=(0., height), title="Full domain")
        axes[0].legend(loc="upper left", fontsize=8.5)
        axes[1].set(yscale="log", ylim=(z[0], .25 * height), title="Near surface")
        fig.suptitle(f"DTU10MW · {mesh_label} · Porté-Agel enabled in both runs\nSame initial checkpoint; volume and time averages over hours 10–11", fontsize=13)
        fig.savefig(output / "advection_comparison.png", dpi=180)
        fig.savefig(output / "advection_comparison.pdf")
        plt.close(fig)
        np.savetxt(output / "advection_comparison.csv", np.column_stack((z, u, previous_profile["mean_u_m_s"], log_u)), delimiter=",", header="z_m,current_u_m_s,previous_u_m_s,log_law_cell_average_m_s", comments="")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for hist in (warm_history, history):
        axes[0].plot(hist["time_hours"], hist["ustar_m_s"], color="#126782", lw=1.)
        axes[1].plot(hist["time_hours"], hist["integrated_resolved_tke_m3_s2"], color="#126782", lw=1.)
    axes[0].axhline(expected_ustar, color="#d65f00", ls="--", label="Pressure balance")
    axes[0].set(xlabel="Simulation time [h]", ylabel=r"$u_*$ [m/s]", title="Surface friction velocity")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Simulation time [h]", ylabel=r"Integrated resolved TKE [m$^3$/s$^2$]", title="Turbulence development")
    axes[2].plot(-profiles["resolved_uw_m2_s2"], z, color="#126782", label="Resolved covariance only")
    axes[2].plot(acceleration * (height - z), z, color="#d65f00", ls="--", label="Required total flux at equilibrium")
    axes[2].set(xlabel=r"Momentum flux [m$^2$/s$^2$]", ylabel="Height [m]", title="Resolved part of momentum flux")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=.2)
    for ax in axes[:2]:
        ax.axvspan(metadata["initial_time"] / 3600., summaries["precursor"]["time_seconds"] / 3600., color="#126782", alpha=.07)
    fig.savefig(output / "development_and_stress.png", dpi=180)
    plt.close(fig)
    timing_note = ""
    if adaptive:
        timing_note = (
            f"Adaptive CFL ceiling: {configured.options.cfl_ceiling:g}; "
            f"maximum measured pre-step CFL: {report['checks']['maximum_actual_step_cfl']:.8f}; "
            f"median recorded timestep: {report['checks']['recorded_dt_median_seconds']:.4f} s. "
        )
    comparison_note = ""
    if previous_profile is not None:
        previous = report["comparison_with_previous"]
        comparison_note = (
            f"\nComparison with {previous_label}: surface-layer RMSE changes "
            f"from {previous['previous_surface_rmse_m_s']:.4f} to {errors['rmse_m_s']:.4f} m/s. "
            "See [comparison plot](advection_comparison.png) and [comparison CSV](advection_comparison.csv). "
            + previous["note"] + "\n"
        )
    (output / "README.md").write_text(
        "# Historical DTU10MW precursor profile\n\n"
        f"Mesh: {mesh_label}. Completed {continued_steps:,} warmup continuation steps and {report['precursor_actual_steps']:,} precursor steps; dt setting = {case['time']['dt_seconds']:g} s ({numerical['dt_interpretation']}), with {scheme_label}. {correction_label}. "
        + timing_note + continuation_note +
        f"The precursor profile averages {actual_samples} full horizontal cell layers at 10 s intervals over hours 10–11. "
        "On this uniform mesh, the horizontal cell mean is the volume-weighted mean within each height layer. "
        "Height is retained; averaging over height would remove the profile.\n\n"
        "The reference uses U = (u*/κ) ln(z/z0), κ = 0.4, z0 = 0.001 m, and "
        "u* = sqrt(a_x H) = 0.4 m/s. Both point and vertical-cell-integrated references are exported. "
        "The latter matches the finite-volume wall convention, including the first cell.\n\n"
        f"For the prespecified 20–{.1 * height:g} m surface band (cell centers {band_label}), "
        f"RMSE = {errors['rmse_m_s']:.4f} m/s ({errors['relative_rmse_percent']:.2f}%), "
        f"mean bias = {errors['mean_bias_m_s']:.4f} m/s. "
        f"At hub height {hub_height:.3f} m, U = {report['hub_height']['mean_u_m_s']:.4f} m/s "
        f"against {report['hub_height']['point_log_law_m_s']:.4f} m/s for the point log law.\n\n"
        f"The friction velocity from mean streamwise wall stress is {wall_ustar:.4f} m/s "
        f"({100. * wall_ustar**2 / expected_ustar**2:.1f}% of equilibrium streamwise wall stress). "
        f"The domain-mean velocity changes by {bulk_change:+.4f} m/s during the precursor hour. "
        "`development_and_stress.png` shows turbulence development and resolved momentum covariance. "
        "SGS/total stress from the existing profile diagnostic is omitted because that diagnostic does not apply the wall-gradient correction; this does not affect the velocity or wall-stress averages. "
        "The separate inlet-plane average is exported for comparison; it is not the volume average. "
        "A one-hour averaging window and agreement in shape alone do not establish convergence.\n\n"
        "Files: `log_law_profile.png`, `log_law_profile.pdf`, `log_law_profile.csv`, "
        "`comparison.json`, and `development_and_stress.png`.\n\n"
        "For context on finite-volume/filtering effects in a log-law comparison, see "
        "[The Influence of WENO Schemes on Large-Eddy Simulations of a Neutral Atmospheric Boundary Layer](https://journals.ametsoc.org/view/journals/atsc/78/11/JAS-D-21-0033.1.xml).\n"
    )
    if comparison_note:
        with (output / "README.md").open("a") as stream:
            stream.write(comparison_note)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="workflow output containing warmup and precursor")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.run, args.output or args.run.parent / "analysis")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
