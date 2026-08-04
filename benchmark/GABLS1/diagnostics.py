"""Diagnostics and official-ensemble comparison for GABLS1."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from benchmark.GABLS1.reference import (
    ReferenceSet,
    ensemble_on_grid,
    load_period_sets,
    load_time_series,
)


@dataclass(frozen=True)
class GABLS1Case:
    domain: float = 400.0
    geostrophic_u: float = 8.0
    geostrophic_v: float = 0.0
    coriolis: float = 1.39e-4
    theta_initial: float = 265.0
    theta_reference: float = 263.5
    inversion_base: float = 100.0
    inversion_gradient: float = 0.01
    surface_cooling_rate: float = -0.25 / 3600.0
    roughness_length: float = 0.1
    gravity: float = 9.81

    def surface_temperature(self, time: float) -> float:
        return self.theta_initial + self.surface_cooling_rate * time


def _face_to_cell(field: np.ndarray) -> np.ndarray:
    return 0.5 * (field[:-1] + field[1:])


def _cell_to_zero_boundary_faces(
    field: np.ndarray,
    z_centers: np.ndarray,
    z_faces: np.ndarray,
) -> np.ndarray:
    """Interpolate a cell profile to faces with homogeneous wall fluctuations."""
    faces = np.zeros(
        (field.shape[0] + 1, *field.shape[1:]),
        dtype=field.dtype,
    )
    distance = z_centers[1:] - z_centers[:-1]
    lower_weight = (z_centers[1:] - z_faces[1:-1]) / distance
    upper_weight = (z_faces[1:-1] - z_centers[:-1]) / distance
    shape = (distance.size,) + (1,) * (field.ndim - 1)
    faces[1:-1] = (
        lower_weight.reshape(shape) * field[:-1]
        + upper_weight.reshape(shape) * field[1:]
    )
    return faces


def _boundary_layer_height(
    z_faces: np.ndarray,
    uw_total: np.ndarray,
    vw_total: np.ndarray,
) -> float:
    magnitude = np.sqrt(uw_total * uw_total + vw_total * vw_total)
    threshold = 0.05 * magnitude[0]
    below = np.flatnonzero(magnitude <= threshold)
    if below.size == 0:
        return float(z_faces[-1] / 0.95)
    upper = int(below[0])
    if upper == 0:
        crossing = z_faces[0]
    else:
        lower = upper - 1
        denominator = magnitude[upper] - magnitude[lower]
        fraction = (
            0.0
            if abs(denominator) < 1.0e-15
            else (threshold - magnitude[lower]) / denominator
        )
        crossing = z_faces[lower] + fraction * (z_faces[upper] - z_faces[lower])
    return float(crossing / 0.95)


def snapshot_statistics(coupled, state) -> dict[str, np.ndarray | float]:
    """Collect resolved, SGS, and numerical horizontally averaged diagnostics."""
    from jaxwind.momentum.neutral_abl import _cell_velocity

    cells = np.asarray(_cell_velocity(state.velocity))
    theta = np.asarray(state.potential_temperature)
    pressure = np.asarray(state.pressure)
    fields = coupled.diagnostic_fields(state)
    diagnostic = {name: np.asarray(value) for name, value in fields._asdict().items()}

    z = 0.5 * (
        np.asarray(coupled.grid.z_faces[:-1]) + np.asarray(coupled.grid.z_faces[1:])
    )
    z_faces = np.asarray(coupled.grid.z_faces)
    mean_velocity = np.mean(cells, axis=(1, 2))
    fluctuation = cells - mean_velocity[:, None, None, :]
    theta_mean = np.mean(theta, axis=(1, 2))
    theta_fluctuation = theta - theta_mean[:, None, None]
    pressure_fluctuation = pressure - np.mean(
        pressure,
        axis=(1, 2),
        keepdims=True,
    )
    u_fluctuation = fluctuation[..., 0]
    v_fluctuation = fluctuation[..., 1]
    w_fluctuation = fluctuation[..., 2]

    velocity_gradient = coupled.momentum.velocity_gradient(
        _cell_velocity(state.velocity)
    )
    wall_stress = coupled._momentum_wall_stress(coupled.surface_layer_fluxes(state))
    stress_faces = np.asarray(
        coupled.momentum.vertical_sgs_stress_flux(
            _cell_velocity(state.velocity),
            gradient=velocity_gradient,
            wall_stress=wall_stress,
        )
    )
    uw_sgs_faces = -stress_faces[..., 0].mean(axis=(1, 2))
    vw_sgs_faces = -stress_faces[..., 1].mean(axis=(1, 2))
    uw_sgs_at_cells = _face_to_cell(uw_sgs_faces)
    vw_sgs_at_cells = _face_to_cell(vw_sgs_faces)

    wtheta_resolved_at_cells = np.mean(
        w_fluctuation * theta_fluctuation,
        axis=(1, 2),
    )
    utheta_resolved_at_cells = np.mean(
        u_fluctuation * theta_fluctuation,
        axis=(1, 2),
    )
    vtheta_resolved_at_cells = np.mean(
        v_fluctuation * theta_fluctuation,
        axis=(1, 2),
    )
    wtheta_sgs_faces = diagnostic["scalar_flux_z"].mean(axis=(1, 2))
    wtheta_sgs_at_cells = _face_to_cell(wtheta_sgs_faces)
    momentum_numerical_faces = diagnostic["momentum_numerical_flux_z"].mean(
        axis=(1, 2)
    )
    uw_numerical_faces = momentum_numerical_faces[..., 0]
    vw_numerical_faces = momentum_numerical_faces[..., 1]
    wtheta_numerical_faces = diagnostic["scalar_numerical_flux_z"].mean(
        axis=(1, 2)
    )
    utheta_sgs_at_cells = diagnostic["scalar_flux_x"].mean(axis=(1, 2))
    vtheta_sgs_at_cells = diagnostic["scalar_flux_y"].mean(axis=(1, 2))

    u_face = _cell_to_zero_boundary_faces(u_fluctuation, z, z_faces)
    v_face = _cell_to_zero_boundary_faces(v_fluctuation, z, z_faces)
    w_faces = np.asarray(state.velocity.z)
    w_face_mean = np.mean(w_faces, axis=(1, 2), keepdims=True)
    w_face_fluctuation = w_faces - w_face_mean
    uw_resolved_faces = np.mean(
        u_face * w_face_fluctuation,
        axis=(1, 2),
    )
    vw_resolved_faces = np.mean(
        v_face * w_face_fluctuation,
        axis=(1, 2),
    )
    theta_face = _cell_to_zero_boundary_faces(
        theta_fluctuation,
        z,
        z_faces,
    )
    wtheta_resolved_faces = np.mean(
        theta_face * w_face_fluctuation,
        axis=(1, 2),
    )
    utheta_resolved_faces = _cell_to_zero_boundary_faces(
        utheta_resolved_at_cells,
        z,
        z_faces,
    )
    vtheta_resolved_faces = _cell_to_zero_boundary_faces(
        vtheta_resolved_at_cells,
        z,
        z_faces,
    )
    utheta_sgs_faces = _cell_to_zero_boundary_faces(
        utheta_sgs_at_cells,
        z,
        z_faces,
    )
    vtheta_sgs_faces = _cell_to_zero_boundary_faces(
        vtheta_sgs_at_cells,
        z,
        z_faces,
    )
    boundary_layer_height = _boundary_layer_height(
        z_faces,
        uw_resolved_faces + uw_sgs_faces + uw_numerical_faces,
        vw_resolved_faces + vw_sgs_faces + vw_numerical_faces,
    )

    u_variance = np.mean(u_fluctuation * u_fluctuation, axis=(1, 2))
    v_variance = np.mean(v_fluctuation * v_fluctuation, axis=(1, 2))
    w_variance = np.mean(w_fluctuation * w_fluctuation, axis=(1, 2))
    theta_variance = np.mean(
        theta_fluctuation * theta_fluctuation,
        axis=(1, 2),
    )
    third_moment = np.mean(w_fluctuation**3, axis=(1, 2))
    skewness = np.where(
        w_variance > 1.0e-15,
        third_moment / np.maximum(w_variance, 1.0e-15) ** 1.5,
        0.0,
    )
    resolved_tke = 0.5 * (u_variance + v_variance + w_variance)
    sgs_tke = diagnostic["sgs_tke"].mean(axis=(1, 2))

    dudz = np.gradient(mean_velocity[:, 0], z, edge_order=1)
    dvdz = np.gradient(mean_velocity[:, 1], z, edge_order=1)
    uw_resolved_at_cells = _face_to_cell(uw_resolved_faces)
    vw_resolved_at_cells = _face_to_cell(vw_resolved_faces)
    shear_resolved = -(uw_resolved_at_cells * dudz + vw_resolved_at_cells * dvdz)
    shear_sgs = -(uw_sgs_at_cells * dudz + vw_sgs_at_cells * dvdz)
    shear_numerical = -(
        _face_to_cell(uw_numerical_faces) * dudz
        + _face_to_cell(vw_numerical_faces) * dvdz
    )
    buoyancy_production = coupled.config.buoyancy_coefficient * (
        wtheta_resolved_at_cells
        + wtheta_sgs_at_cells
        + _face_to_cell(wtheta_numerical_faces)
    )
    dissipation_amd = diagnostic["amd_energy_dissipation"].mean(axis=(1, 2))
    dissipation_numerical = diagnostic["mp5_energy_dissipation"].mean(axis=(1, 2))
    resolved_energy = 0.5 * np.sum(fluctuation * fluctuation, axis=-1)
    energy_pressure = resolved_energy + pressure_fluctuation
    energy_pressure_faces = _cell_to_zero_boundary_faces(
        energy_pressure,
        z,
        z_faces,
    )
    resolved_transport_flux = np.mean(
        w_face_fluctuation * energy_pressure_faces,
        axis=(1, 2),
    )
    transport_resolved = -np.diff(resolved_transport_flux) / np.diff(z_faces)

    speed = np.sqrt(mean_velocity[:, 0] ** 2 + mean_velocity[:, 1] ** 2)
    jet_index = int(np.argmax(speed))
    surface_heat_flux = float(np.mean(diagnostic["surface_heat_flux"]))
    mean_surface_stress = np.mean(
        diagnostic["surface_momentum_stress"],
        axis=(0, 1),
    )
    friction_velocity = float(np.sqrt(np.linalg.norm(mean_surface_stress)))
    if abs(surface_heat_flux) > 1.0e-15:
        obukhov_length = float(
            -(friction_velocity**3)
            * coupled.config.reference_potential_temperature
            / (
                coupled.momentum.config.von_karman
                * coupled.config.gravity
                * surface_heat_flux
            )
        )
    else:
        obukhov_length = np.inf

    return {
        "z": z,
        "z_flux": z_faces,
        "u_mean": mean_velocity[:, 0],
        "v_mean": mean_velocity[:, 1],
        "theta_mean": theta_mean,
        "u_var_resolved": u_variance,
        "v_var_resolved": v_variance,
        "w_var_resolved": w_variance,
        "w_skewness": skewness,
        "sgs_tke": sgs_tke,
        "theta_var_resolved": theta_variance,
        "theta_var_sgs": diagnostic["scalar_variance"].mean(axis=(1, 2)),
        "uw_resolved": uw_resolved_faces,
        "uw_sgs": uw_sgs_faces,
        "uw_numerical": uw_numerical_faces,
        "uw_turbulent_total": uw_resolved_faces + uw_sgs_faces,
        "uw_total": uw_resolved_faces + uw_sgs_faces + uw_numerical_faces,
        "vw_resolved": vw_resolved_faces,
        "vw_sgs": vw_sgs_faces,
        "vw_numerical": vw_numerical_faces,
        "vw_turbulent_total": vw_resolved_faces + vw_sgs_faces,
        "vw_total": vw_resolved_faces + vw_sgs_faces + vw_numerical_faces,
        "wtheta_resolved": wtheta_resolved_faces,
        "wtheta_sgs": wtheta_sgs_faces,
        "wtheta_numerical": wtheta_numerical_faces,
        "wtheta_turbulent_total": wtheta_resolved_faces + wtheta_sgs_faces,
        "wtheta_total": (
            wtheta_resolved_faces + wtheta_sgs_faces + wtheta_numerical_faces
        ),
        "utheta_resolved": utheta_resolved_faces,
        "utheta_sgs": utheta_sgs_faces,
        "utheta_total": utheta_resolved_faces + utheta_sgs_faces,
        "vtheta_resolved": vtheta_resolved_faces,
        "vtheta_sgs": vtheta_sgs_faces,
        "vtheta_total": vtheta_resolved_faces + vtheta_sgs_faces,
        "tke_resolved": resolved_tke,
        "tke_total": resolved_tke + sgs_tke,
        "shear_production_resolved": shear_resolved,
        "shear_production_sgs": shear_sgs,
        "shear_production_numerical": shear_numerical,
        "buoyancy_production": buoyancy_production,
        "transport_total": transport_resolved,
        "dissipation_amd": dissipation_amd,
        "dissipation_numerical": dissipation_numerical,
        "dissipation_mp5": dissipation_numerical,
        "dissipation_total": dissipation_amd + dissipation_numerical,
        "boundary_layer_height": boundary_layer_height,
        "surface_heat_flux": surface_heat_flux,
        "friction_velocity": friction_velocity,
        "obukhov_length": obukhov_length,
        "maximum_abs_w": float(np.max(np.abs(w_faces))),
        "jet_speed": float(speed[jet_index]),
        "jet_height": float(z[jet_index]),
    }


def average_samples(
    samples: list[dict[str, np.ndarray | float]],
) -> dict[str, np.ndarray | float]:
    if not samples:
        raise ValueError("at least one GABLS sample is required")
    result = {}
    for key in samples[0]:
        result[key] = np.mean(
            np.stack([np.asarray(sample[key]) for sample in samples]),
            axis=0,
        )
    return result


def _write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = tuple(columns)
    count = len(np.asarray(columns[names[0]]))
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for index in range(count):
            writer.writerow([np.asarray(columns[name])[index] for name in names])


def _reference_ensembles(
    reference_dir: Path,
    z: np.ndarray,
    z_flux: np.ndarray,
):
    result = {
        name: ensemble_on_grid(
            load_period_sets(reference_dir, name, period=9),
            "z",
            z_flux if name == "C" else z,
        )
        for name in "ABCD"
    }
    flux_sets = load_period_sets(reference_dir, "C", period=9)
    total_sets = []
    for dataset in flux_sets:
        values = dict(dataset.values)
        for prefix in ("uw", "vw", "wtheta", "utheta", "vtheta"):
            values[f"{prefix}_total"] = (
                values[f"{prefix}_resolved"] + values[f"{prefix}_sgs"]
            )
        total_sets.append(
            ReferenceSet(dataset.participant, dataset.description, values)
        )
    result["C"] = ensemble_on_grid(total_sets, "z", z_flux)
    return result


def _reference_resolution_m(reference_dir: Path | None) -> float | None:
    if reference_dir is None:
        return None
    metadata_path = reference_dir / "SOURCE.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "resolution_m" in metadata:
            return float(metadata["resolution_m"])
    name = reference_dir.name
    if name.startswith("official_") and name.endswith("m"):
        encoded = name.removeprefix("official_").removesuffix("m")
        try:
            return float(encoded.replace("p", "."))
        except ValueError:
            pass
    return None


def _plot_comparison(
    output: Path,
    mean: dict[str, np.ndarray | float],
    reference: dict,
    reference_dir: Path | None,
    model_resolution_m: float,
    official_resolution_m: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.asarray(mean["z"])
    z_flux = np.asarray(mean["z_flux"])
    figure, axes = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)

    def panel(axis, set_name, variable, model, xlabel):
        ensemble = reference.get(set_name, {}).get(variable)
        if ensemble is not None:
            axis.fill_betweenx(
                z,
                ensemble["minimum"],
                ensemble["maximum"],
                color="0.85",
                label=(
                    f"official {official_resolution_m:g} m range"
                    if official_resolution_m is not None
                    else "official LES range"
                ),
            )
            axis.plot(ensemble["mean"], z, "k--", label="official mean")
        axis.plot(
            np.asarray(model),
            z,
            color="crimson",
            lw=2,
            label=f"AMD {model_resolution_m:g} m",
        )
        axis.set(xlabel=xlabel, ylabel="z (m)", ylim=(0.0, 400.0))
        axis.grid(alpha=0.25)

    panel(axes[0, 0], "A", "u_mean", mean["u_mean"], "U (m/s)")
    panel(axes[0, 1], "A", "v_mean", mean["v_mean"], "V (m/s)")
    panel(axes[0, 2], "A", "theta_mean", mean["theta_mean"], "theta (K)")
    panel(axes[0, 3], "B", "sgs_tke", mean["sgs_tke"], "SGS TKE")
    panel(axes[1, 0], "B", "u_var_resolved", mean["u_var_resolved"], "u variance")
    panel(axes[1, 1], "B", "v_var_resolved", mean["v_var_resolved"], "v variance")
    panel(axes[1, 2], "B", "w_var_resolved", mean["w_var_resolved"], "w variance")
    panel(
        axes[1, 3],
        "B",
        "theta_var_resolved",
        mean["theta_var_resolved"],
        "theta variance",
    )

    flux_specs = (
        ("uw", "x-momentum flux"),
        ("vw", "y-momentum flux"),
        ("wtheta", "vertical heat flux"),
    )
    for axis, (prefix, label) in zip(axes[2, :3], flux_specs, strict=True):
        resolved_name = f"{prefix}_resolved"
        sgs_name = f"{prefix}_sgs"
        ensemble = reference.get("C", {}).get(f"{prefix}_total")
        if ensemble is not None:
            axis.fill_betweenx(
                z_flux,
                ensemble["minimum"],
                ensemble["maximum"],
                color="0.85",
            )
            axis.plot(
                ensemble["mean"],
                z_flux,
                "k--",
                label="official total mean",
            )
        axis.plot(
            mean[f"{prefix}_total"],
            z_flux,
            "r",
            lw=2,
            label=f"AMD {model_resolution_m:g} m total",
        )
        axis.plot(mean[resolved_name], z_flux, "r:", label="resolved")
        axis.plot(
            mean[sgs_name],
            z_flux,
            color="royalblue",
            ls="--",
            label="SGS",
        )
        axis.plot(
            mean[f"{prefix}_numerical"],
            z_flux,
            color="darkorange",
            ls="-.",
            label="numerical",
        )
        axis.set(xlabel=label, ylabel="z (m)", ylim=(0.0, 400.0))
        axis.grid(alpha=0.25)

    time_axis = axes[2, 3]
    if reference_dir is not None:
        for dataset in load_time_series(reference_dir):
            time_axis.plot(
                dataset.values["time_s"] / 3600.0,
                dataset.values["boundary_layer_height"],
                color="0.7",
                lw=0.8,
            )
    time_axis.axhline(
        float(mean["boundary_layer_height"]),
        color="crimson",
        lw=2,
        label="AMD 8-9 h mean",
    )
    time_axis.set(xlabel="time (h)", ylabel="boundary-layer height (m)")
    time_axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[2, 0].legend(fontsize=8)
    time_axis.legend(fontsize=8)
    official_label = (
        f"official {official_resolution_m:g} m LES"
        if official_resolution_m is not None
        else "official LES"
    )
    figure.suptitle(
        f"GABLS1: non-spectral AMD {model_resolution_m:g} m vs {official_label}"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_outputs(
    output_dir: Path,
    *,
    samples: list[dict[str, np.ndarray | float]],
    time_rows: list[dict[str, float]],
    reference_dir: Path | None,
    metadata: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    """Write profiles, reference envelope, plots, and scalar summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mean = average_samples(samples)
    cell_count = len(np.asarray(mean["z"]))
    flux_count = len(np.asarray(mean["z_flux"]))
    profile_columns = {
        key: np.asarray(value)
        for key, value in mean.items()
        if np.asarray(value).ndim == 1
        and len(np.asarray(value)) == cell_count
        and key != "z_flux"
    }
    _write_csv(output_dir / "profiles.csv", profile_columns)
    flux_columns = {
        "z_flux": np.asarray(mean["z_flux"]),
        **{
            key: np.asarray(value)
            for key, value in mean.items()
            if np.asarray(value).ndim == 1
            and len(np.asarray(value)) == flux_count
            and key != "z_flux"
        },
    }
    _write_csv(output_dir / "flux_profiles.csv", flux_columns)
    if time_rows:
        _write_csv(
            output_dir / "time_series.csv",
            {key: np.asarray([row[key] for row in time_rows]) for key in time_rows[0]},
        )

    reference = {}
    official_resolution_m = None
    if reference_dir is not None and reference_dir.exists():
        official_resolution_m = _reference_resolution_m(reference_dir)
        z = np.asarray(mean["z"])
        z_flux = np.asarray(mean["z_flux"])
        reference = _reference_ensembles(reference_dir, z, z_flux)
        profile_reference: dict[str, np.ndarray] = {"z": z}
        flux_reference: dict[str, np.ndarray] = {"z_flux": z_flux}
        for set_name, variables in reference.items():
            for variable, statistics in variables.items():
                for statistic, values in statistics.items():
                    destination = (
                        flux_reference if set_name == "C" else profile_reference
                    )
                    destination[f"{set_name}_{variable}_{statistic}"] = values
        _write_csv(
            output_dir / "reference_profile_ensemble.csv",
            profile_reference,
        )
        _write_csv(
            output_dir / "reference_flux_ensemble.csv",
            flux_reference,
        )

    scalar_mean = {
        key: float(np.asarray(value))
        for key, value in mean.items()
        if np.asarray(value).ndim == 0
    }
    summary: dict[str, float | int | str] = {
        "schema": "jaxwind.gabls1.nonspectral-amd.v1",
        "sample_count": len(samples),
        "official_participants": (
            len(load_period_sets(reference_dir, "A", 9))
            if reference_dir is not None and reference_dir.exists()
            else 0
        ),
        "official_reference_resolution_m": (
            official_resolution_m if official_resolution_m is not None else "unknown"
        ),
        **metadata,
        **scalar_mean,
    }
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("quantity", "value"))
        writer.writerows(summary.items())
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "benchmark_stats.npz",
        **{f"mean_{key}": np.asarray(value) for key, value in mean.items()},
    )
    model_resolution_m = float(metadata.get("grid_spacing_m", 12.5))
    resolution_tag = f"{model_resolution_m:g}".replace(".", "p")
    if official_resolution_m is None or np.isclose(
        model_resolution_m,
        official_resolution_m,
    ):
        comparison_name = f"GABLS1_AMD_{resolution_tag}m_comparison.png"
    else:
        official_tag = f"{official_resolution_m:g}".replace(".", "p")
        comparison_name = (
            f"GABLS1_AMD_{resolution_tag}m_vs_official_{official_tag}m_comparison.png"
        )
    _plot_comparison(
        output_dir / comparison_name,
        mean,
        reference,
        reference_dir,
        model_resolution_m,
        official_resolution_m,
    )
    return summary


__all__ = [
    "GABLS1Case",
    "average_samples",
    "save_outputs",
    "snapshot_statistics",
]
