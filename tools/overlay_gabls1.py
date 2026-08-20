"""Overlay a uniform ABL result on the official GABLS1 8--9 h ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib.ticker import MaxNLocator  # noqa: E402

try:
    from tools.gabls1_reference import (
        ensemble_on_grid,
        ensemble_statistics,
        interpolate_values,
        load_period_sets,
        load_time_series_sets,
    )
except ModuleNotFoundError:  # direct ``python tools/overlay_gabls1.py``
    from gabls1_reference import (
        ensemble_on_grid,
        ensemble_statistics,
        interpolate_values,
        load_period_sets,
        load_time_series_sets,
    )


import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "outputs" / "gabls1_lasd_32x32x32"
REFERENCE_ROOT = ROOT / "cases" / "GABLS1" / "reference"
REFERENCE_DIRECTORIES = {
    6.25: REFERENCE_ROOT / "official_6p25m",
    12.5: REFERENCE_ROOT / "official_12p5m",
}
PANELS = (
    ("u_mean", "mean_u_m_s", r"$\langle u\rangle$ (m s$^{-1}$)"),
    ("v_mean", "mean_v_m_s", r"$\langle v\rangle$ (m s$^{-1}$)"),
    ("theta_mean", "mean_scalar", r"$\langle\theta\rangle$ (K)"),
    ("uw_total", "total_uw_m2_s2", r"total $\langle u'w'\rangle$ (m$^2$ s$^{-2}$)"),
    ("vw_total", "total_vw_m2_s2", r"total $\langle v'w'\rangle$ (m$^2$ s$^{-2}$)"),
    (
        "wtheta_total",
        "total_scalar_flux",
        r"total $\langle w'\theta'\rangle$ (K m s$^{-1}$)",
    ),
)
PROFILE_GROUPS = (
    (
        "A",
        "Set A — mean profiles",
        (
            ("u_mean", r"$\langle u\rangle$ (m s$^{-1}$)"),
            ("v_mean", r"$\langle v\rangle$ (m s$^{-1}$)"),
            ("theta_mean", r"$\langle\theta\rangle$ (K)"),
        ),
    ),
    (
        "B",
        "Set B — variances and vertical-velocity skewness",
        (
            ("u_var_resolved", r"$\langle u'^2\rangle$ (m$^2$ s$^{-2}$)"),
            ("v_var_resolved", r"$\langle v'^2\rangle$ (m$^2$ s$^{-2}$)"),
            ("w_var_resolved", r"$\langle w'^2\rangle$ (m$^2$ s$^{-2}$)"),
            ("w_skewness", r"$\langle w'^3\rangle/\langle w'^2\rangle^{3/2}$"),
            ("sgs_tke", r"SGS TKE (m$^2$ s$^{-2}$)"),
            ("theta_var_resolved", r"$\langle\theta'^2\rangle$ (K$^2$)"),
        ),
    ),
    (
        "C",
        "Set C — resolved and subgrid fluxes",
        (
            ("uw_resolved", r"resolved $\langle u'w'\rangle$ (m$^2$ s$^{-2}$)"),
            ("uw_sgs", r"SGS $\langle u'w'\rangle$ (m$^2$ s$^{-2}$)"),
            ("vw_resolved", r"resolved $\langle v'w'\rangle$ (m$^2$ s$^{-2}$)"),
            ("vw_sgs", r"SGS $\langle v'w'\rangle$ (m$^2$ s$^{-2}$)"),
            ("wtheta_resolved", r"resolved $\langle w'\theta'\rangle$ (K m s$^{-1}$)"),
            ("wtheta_sgs", r"SGS $\langle w'\theta'\rangle$ (K m s$^{-1}$)"),
            ("utheta_resolved", r"resolved $\langle u'\theta'\rangle$ (K m s$^{-1}$)"),
            ("utheta_sgs", r"SGS $\langle u'\theta'\rangle$ (K m s$^{-1}$)"),
            ("vtheta_resolved", r"resolved $\langle v'\theta'\rangle$ (K m s$^{-1}$)"),
            ("vtheta_sgs", r"SGS $\langle v'\theta'\rangle$ (K m s$^{-1}$)"),
        ),
    ),
    (
        "D",
        "Set D — resolved-TKE budget",
        (
            (
                "shear_production_resolved",
                r"resolved shear production (m$^2$ s$^{-3}$)",
            ),
            ("shear_production_sgs", r"SGS shear production (m$^2$ s$^{-3}$)"),
            ("buoyancy_production", r"resolved buoyancy production (m$^2$ s$^{-3}$)"),
            ("transport_total", r"total transport (m$^2$ s$^{-3}$)"),
            ("dissipation", r"dissipation (m$^2$ s$^{-3}$)"),
            ("storage", r"storage (m$^2$ s$^{-3}$)"),
        ),
    ),
)
TIME_PANELS = (
    ("boundary_layer_height", r"boundary-layer height (m)"),
    ("surface_heat_flux", r"surface $\langle w'\theta'\rangle$ (K m s$^{-1}$)"),
    ("friction_velocity", r"friction velocity (m s$^{-1}$)"),
    ("obukhov_length", r"Monin–Obukhov length (m)"),
    ("maximum_abs_w", r"domain maximum $|w|$ (m s$^{-1}$)"),
)


def _read_csv(path: Path) -> np.ndarray:
    values = np.genfromtxt(path, delimiter=",", names=True)
    if values.size == 0:
        raise ValueError(f"empty result file: {path}")
    return np.atleast_1d(values)


def _select_reference_dir(result_dir: Path) -> Path:
    model = _read_csv(result_dir / "profiles.csv")
    z = np.asarray(model["z_m"], dtype=float)
    if z.size < 2:
        raise ValueError("at least two profile heights are required")
    spacing = float(np.median(np.diff(z)))
    resolution = min(REFERENCE_DIRECTORIES, key=lambda value: abs(value - spacing))
    return REFERENCE_DIRECTORIES[resolution]


def _reference_metadata(reference_dir: Path) -> dict:
    path = reference_dir / "SOURCE.json"
    if not path.is_file():
        raise FileNotFoundError(f"reference provenance is missing: {path}")
    metadata = json.loads(path.read_text())
    if "resolution_m" not in metadata:
        name = reference_dir.name
        metadata["resolution_m"] = 12.5 if "12p5" in name else 6.25
    return metadata


def _flux_ensemble(reference_dir: Path, z: np.ndarray) -> dict:
    datasets = load_period_sets(reference_dir, "C", period=9)
    components = {
        "uw_total": ("uw_resolved", "uw_sgs"),
        "vw_total": ("vw_resolved", "vw_sgs"),
        "wtheta_total": ("wtheta_resolved", "wtheta_sgs"),
    }
    result = {}
    for name, (resolved, subgrid) in components.items():
        # Sum participant components before statistics, retaining covariance.
        participant_values = []
        for dataset in datasets:
            participant_values.append(
                interpolate_values(dataset, "z", resolved, z)
                + interpolate_values(dataset, "z", subgrid, z)
            )
        stack = np.stack(participant_values)
        result[name] = ensemble_statistics(stack)
    return result


def _complete_profile_references(
    reference_dir: Path,
    z: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[str]]]:
    references: dict[str, dict[str, np.ndarray]] = {}
    participants: dict[str, list[str]] = {}
    for set_name, _title, _panels in PROFILE_GROUPS:
        datasets = load_period_sets(reference_dir, set_name, period=9)
        references.update(ensemble_on_grid(datasets, z))
        participants[set_name] = [dataset.participant for dataset in datasets]
    return references, participants


def _model_profile_values(result_dir: Path, model: np.ndarray) -> dict[str, np.ndarray]:
    names = set(model.dtype.names or ())
    values: dict[str, np.ndarray] = {}

    direct = {
        "u_mean": "mean_u_m_s",
        "v_mean": "mean_v_m_s",
        "theta_mean": "mean_scalar",
        "u_var_resolved": "resolved_u_variance_m2_s2",
        "v_var_resolved": "resolved_v_variance_m2_s2",
        "w_var_resolved": "resolved_w_variance_m2_s2",
        "sgs_tke": "sgs_tke_m2_s2",
        "theta_var_resolved": "resolved_scalar_variance",
        "uw_resolved": "resolved_uw_m2_s2",
        "uw_sgs": "sgs_uw_m2_s2",
        "vw_resolved": "resolved_vw_m2_s2",
        "vw_sgs": "sgs_vw_m2_s2",
        "wtheta_resolved": "resolved_scalar_flux",
        "wtheta_sgs": "sgs_scalar_flux",
        "utheta_resolved": "resolved_uc",
        "utheta_sgs": "sgs_uc",
        "vtheta_resolved": "resolved_vc",
        "vtheta_sgs": "sgs_vc",
    }
    for reference_name, model_name in direct.items():
        if model_name in names:
            values[reference_name] = np.asarray(model[model_name], dtype=float)

    skewness_columns = {
        "resolved_w_variance_m2_s2",
        "w_third_moment_m3_s3",
    }
    if skewness_columns.issubset(names):
        variance = np.asarray(model["resolved_w_variance_m2_s2"], dtype=float)
        third_moment = np.asarray(model["w_third_moment_m3_s3"], dtype=float)
        denominator = np.where(variance > 0.0, variance**1.5, np.nan)
        values["w_skewness"] = third_moment / denominator

    z = np.asarray(model["z_m"], dtype=float)
    mean_and_stress = {
        "mean_u_m_s",
        "mean_v_m_s",
        "resolved_uw_m2_s2",
        "resolved_vw_m2_s2",
        "sgs_uw_m2_s2",
        "sgs_vw_m2_s2",
    }
    if mean_and_stress.issubset(names):
        du_dz = np.gradient(np.asarray(model["mean_u_m_s"], dtype=float), z)
        dv_dz = np.gradient(np.asarray(model["mean_v_m_s"], dtype=float), z)
        values["shear_production_resolved"] = -(
            np.asarray(model["resolved_uw_m2_s2"], dtype=float) * du_dz
            + np.asarray(model["resolved_vw_m2_s2"], dtype=float) * dv_dz
        )
        values["shear_production_sgs"] = -(
            np.asarray(model["sgs_uw_m2_s2"], dtype=float) * du_dz
            + np.asarray(model["sgs_vw_m2_s2"], dtype=float) * dv_dz
        )

    summary_path = result_dir / "summary.json"
    if "resolved_scalar_flux" in names and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        acceleration = summary.get("physics", {}).get(
            "buoyancy_acceleration_per_scalar"
        )
        if acceleration is not None:
            values["buoyancy_production"] = float(acceleration) * np.asarray(
                model["resolved_scalar_flux"],
                dtype=float,
            )

    if "resolved_tke_sgs_transfer_m2_s3" in names:
        # Set D defines DISS as the positive sink in
        # TEND = RSPROD + TTRAN + BPROD - DISS + SSPROD.
        values["dissipation"] = -np.asarray(
            model["resolved_tke_sgs_transfer_m2_s3"],
            dtype=float,
        )
    return values


def _time_references(
    reference_dir: Path,
    target_time_s: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    datasets = load_time_series_sets(reference_dir)
    if not datasets:
        raise ValueError("no GABLS1 Set-E participant records were found")
    references = {}
    for variable, _label in TIME_PANELS:
        stack = np.stack(
            [
                interpolate_values(
                    dataset,
                    "time_s",
                    variable,
                    target_time_s,
                )
                for dataset in datasets
            ]
        )
        references[variable] = ensemble_statistics(stack)
    return references, [dataset.participant for dataset in datasets]


def _model_time_values(result_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    history_path = result_dir / "history.csv"
    if not history_path.is_file():
        return {}
    history = _read_csv(history_path)
    names = set(history.dtype.names or ())
    if "time_hours" not in names:
        return {}
    time_s = np.asarray(history["time_hours"], dtype=float) * 3600.0
    direct = {
        "surface_heat_flux": "surface_scalar_flux",
        "friction_velocity": "ustar_m_s",
        "obukhov_length": "obukhov_length_m",
        "maximum_abs_w": "maximum_abs_w_m_s",
    }
    return {
        reference_name: (time_s, np.asarray(history[model_name], dtype=float))
        for reference_name, model_name in direct.items()
        if model_name in names
    }


def _complete_time_grid(
    reference_dir: Path,
    model_time: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    histories = load_time_series_sets(reference_dir)
    maximum = min(
        9.0 * 3600.0,
        max(float(np.nanmax(data.values["time_s"])) for data in histories),
    )
    native_times = [time for time, _values in model_time.values()]
    if native_times:
        combined = np.unique(np.concatenate((np.asarray([0.0]), *native_times)))
        return combined[(combined >= 0.0) & (combined <= maximum)]
    return np.linspace(0.0, maximum, 109)


def _write_comparison(
    path: Path,
    z: np.ndarray,
    model: np.ndarray,
    references: dict,
) -> None:
    fields = ["z_m"]
    for reference_name, model_name, _label in PANELS:
        fields.extend(
            (
                f"model_{reference_name}",
                f"reference_{reference_name}_mean",
                f"reference_{reference_name}_minimum",
                f"reference_{reference_name}_maximum",
                f"reference_{reference_name}_standard_deviation",
                f"reference_{reference_name}_count",
            )
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, height in enumerate(z):
            row = {"z_m": float(height)}
            for reference_name, model_name, _label in PANELS:
                stats = references[reference_name]
                row[f"model_{reference_name}"] = float(model[model_name][index])
                for statistic in (
                    "mean",
                    "minimum",
                    "maximum",
                    "standard_deviation",
                    "count",
                ):
                    row[f"reference_{reference_name}_{statistic}"] = float(
                        stats[statistic][index]
                    )
            writer.writerow(row)


def _write_complete_profile_comparison(
    path: Path,
    z: np.ndarray,
    model_values: dict[str, np.ndarray],
    references: dict[str, dict[str, np.ndarray]],
) -> None:
    variables = [
        name
        for _set_name, _title, panels in PROFILE_GROUPS
        for name, _label in panels
    ]
    statistics = ("mean", "minimum", "maximum", "standard_deviation", "count")
    fields = ["z_m"]
    for name in variables:
        fields.append(f"model_{name}")
        fields.extend(f"reference_{name}_{statistic}" for statistic in statistics)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, height in enumerate(z):
            row = {"z_m": float(height)}
            for name in variables:
                model = model_values.get(name)
                row[f"model_{name}"] = (
                    float(model[index]) if model is not None else math.nan
                )
                for statistic in statistics:
                    row[f"reference_{name}_{statistic}"] = float(
                        references[name][statistic][index]
                    )
            writer.writerow(row)


def _write_complete_time_comparison(
    path: Path,
    time_s: np.ndarray,
    model_values: dict[str, tuple[np.ndarray, np.ndarray]],
    references: dict[str, dict[str, np.ndarray]],
) -> None:
    statistics = ("mean", "minimum", "maximum", "standard_deviation", "count")
    fields = ["time_s", "time_hours"]
    for name, _label in TIME_PANELS:
        fields.append(f"model_{name}")
        fields.extend(f"reference_{name}_{statistic}" for statistic in statistics)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        interpolated_model = {}
        for name, (native_time, native_values) in model_values.items():
            values = np.full_like(time_s, np.nan, dtype=float)
            inside = (time_s >= native_time[0]) & (time_s <= native_time[-1])
            values[inside] = np.interp(
                time_s[inside],
                native_time,
                native_values,
            )
            interpolated_model[name] = values
        for index, time in enumerate(time_s):
            row = {"time_s": float(time), "time_hours": float(time / 3600.0)}
            for name, _label in TIME_PANELS:
                model = interpolated_model.get(name)
                row[f"model_{name}"] = (
                    float(model[index]) if model is not None else math.nan
                )
                for statistic in statistics:
                    row[f"reference_{name}_{statistic}"] = float(
                        references[name][statistic][index]
                    )
            writer.writerow(row)


def _style_profile_axis(
    axis,
    z: np.ndarray,
    stats: dict[str, np.ndarray],
    model: np.ndarray | None,
    label: str,
    max_height: float,
    model_label: str,
) -> None:
    axis.fill_betweenx(
        z,
        stats["minimum"],
        stats["maximum"],
        color="#9aa0a6",
        alpha=0.32,
        label="official participant range",
    )
    axis.plot(stats["mean"], z, "k--", lw=1.6, label="official ensemble mean")
    if model is not None:
        axis.plot(model, z, color="#d62728", lw=2.1, label=model_label)
    else:
        axis.text(
            0.98,
            0.96,
            "reference only — not recorded",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#5f6772",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )
    axis.set(xlabel=label, ylabel="z (m)", ylim=(0.0, max_height))
    axis.xaxis.set_major_locator(MaxNLocator(5))
    axis.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    axis.grid(alpha=0.24)


def _render_profile_group(
    path: Path,
    title: str,
    panels: tuple[tuple[str, str], ...],
    z: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
    model_values: dict[str, np.ndarray],
    *,
    max_height: float,
    columns: int = 3,
    model_label: str = "JAX-Wind",
) -> None:
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.7 * columns, 4.2 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (name, label) in zip(axes.flat, panels, strict=False):
        _style_profile_axis(
            axis,
            z,
            references[name],
            model_values.get(name),
            label,
            max_height,
            model_label,
        )
    for axis in axes.flat[len(panels) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=8, loc="best")
    figure.suptitle(title, fontsize=16)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _render_time_group(
    path: Path,
    title: str,
    time_s: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
    model_values: dict[str, tuple[np.ndarray, np.ndarray]],
    model_label: str = "JAX-Wind",
) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.1, 8.4),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (name, label) in zip(axes.flat, TIME_PANELS, strict=False):
        stats = references[name]
        hours = time_s / 3600.0
        axis.fill_between(
            hours,
            stats["minimum"],
            stats["maximum"],
            color="#9aa0a6",
            alpha=0.32,
            label="official participant range",
        )
        axis.plot(
            hours,
            stats["mean"],
            "k--",
            lw=1.6,
            label="official ensemble mean",
        )
        model = model_values.get(name)
        if model is not None:
            axis.plot(
                model[0] / 3600.0,
                model[1],
                color="#d62728",
                lw=2.1,
                label=model_label,
            )
        else:
            axis.text(
                0.98,
                0.96,
                "reference only — not recorded",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#5f6772",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.72,
                    "edgecolor": "none",
                },
            )
        axis.set(xlabel="time (h)", ylabel=label, xlim=(0.0, 9.0))
        axis.xaxis.set_major_locator(MaxNLocator(6))
        if name == "obukhov_length":
            axis.set_yscale("symlog", linthresh=100.0)
        else:
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axis.grid(alpha=0.24)
    axes.flat[-1].set_visible(False)
    axes.flat[0].legend(fontsize=8, loc="best")
    figure.suptitle(title, fontsize=16)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _vertical_montage(paths: list[Path], output: Path) -> None:
    images = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    gap = 28
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    montage = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        montage.paste(image, ((width - image.width) // 2, y))
        y += image.height + gap
    montage.save(output, optimize=True)


def _checkout(
    z: np.ndarray,
    model: np.ndarray,
    references: dict,
    participants: list[str],
) -> dict:
    below_200 = z <= 200.0
    theta = np.asarray(model["mean_scalar"], dtype=float)
    theta_reference = references["theta_mean"]["mean"]
    speed = np.hypot(model["mean_u_m_s"], model["mean_v_m_s"])
    reference_speed = np.hypot(
        references["u_mean"]["mean"],
        references["v_mean"]["mean"],
    )
    jet = z <= 300.0
    model_jet = int(np.nanargmax(np.where(jet, speed, np.nan)))
    reference_jet = int(np.nanargmax(np.where(jet, reference_speed, np.nan)))
    within = (
        (theta >= references["theta_mean"]["minimum"])
        & (theta <= references["theta_mean"]["maximum"])
        & below_200
    )
    compared = below_200 & np.isfinite(theta_reference)
    lower = int(np.flatnonzero(np.isfinite(references["wtheta_total"]["mean"]))[0])
    model_uw = float(model["total_uw_m2_s2"][lower])
    model_vw = float(model["total_vw_m2_s2"][lower])
    return {
        "comparison_period": "hours 8-9 (official A9 and C9 records)",
        "participants": participants,
        "participant_count": len(participants),
        "lowest_compared_height_m": float(z[lower]),
        "model_lower_scalar_flux_k_m_s": float(
            model["total_scalar_flux"][lower]
        ),
        "reference_lower_scalar_flux_mean_k_m_s": float(
            references["wtheta_total"]["mean"][lower]
        ),
        "reference_lower_scalar_flux_range_k_m_s": [
            float(references["wtheta_total"]["minimum"][lower]),
            float(references["wtheta_total"]["maximum"][lower]),
        ],
        "model_lower_friction_velocity_from_stress_m_s": float(
            np.hypot(model_uw, model_vw) ** 0.5
        ),
        "theta_rmse_below_200m_k": float(
            np.sqrt(np.nanmean((theta[below_200] - theta_reference[below_200]) ** 2))
        ),
        "theta_fraction_within_participant_range_below_200m": float(
            np.count_nonzero(within) / np.count_nonzero(compared)
        ),
        "model_low_level_jet": {
            "speed_m_s": float(speed[model_jet]),
            "height_m": float(z[model_jet]),
        },
        "reference_ensemble_mean_low_level_jet": {
            "speed_m_s": float(reference_speed[reference_jet]),
            "height_m": float(z[reference_jet]),
        },
    }


def _result_model_label(result_dir: Path) -> str:
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return "JAX-Wind"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    solver = summary.get("solver", {})
    if solver.get("discretization") != "finite-volume":
        return "JAX-Wind"
    backend = str(solver.get("pressure_backend", "")).upper()
    return f"JAX-Wind FV-{backend}" if backend else "JAX-Wind FV"


def overlay_results(
    result_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    *,
    max_height: float = 250.0,
) -> dict[str, Path]:
    model = _read_csv(result_dir / "profiles.csv")
    model_label = _result_model_label(result_dir)
    z = np.asarray(model["z_m"], dtype=float)
    metadata = _reference_metadata(reference_dir)
    reference_resolution = float(metadata["resolution_m"])
    profile_sets = load_period_sets(reference_dir, "A", period=9)
    profiles = ensemble_on_grid(profile_sets, z)
    references = {**profiles, **_flux_ensemble(reference_dir, z)}
    complete_references, participants_by_set = _complete_profile_references(
        reference_dir,
        z,
    )
    model_profile_values = _model_profile_values(result_dir, model)
    model_time_values = _model_time_values(result_dir)
    complete_time_s = _complete_time_grid(reference_dir, model_time_values)
    time_references, time_participants = _time_references(
        reference_dir,
        complete_time_s,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "gabls1_official_overlay.png"
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 8.4), constrained_layout=True)
    for axis, (reference_name, model_name, label) in zip(
        axes.flat,
        PANELS,
        strict=True,
    ):
        stats = references[reference_name]
        axis.fill_betweenx(
            z,
            stats["minimum"],
            stats["maximum"],
            color="#9aa0a6",
            alpha=0.32,
            label="official participant range",
        )
        axis.plot(stats["mean"], z, "k--", lw=1.6, label="official ensemble mean")
        axis.plot(model[model_name], z, color="#d62728", lw=2.2, label=model_label)
        axis.set(xlabel=label, ylabel="z (m)", ylim=(0.0, max_height))
        axis.xaxis.set_major_locator(MaxNLocator(5))
        finite = np.asarray(model[model_name], dtype=float)
        if np.nanmax(np.abs(finite)) < 0.1:
            axis.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        else:
            axis.ticklabel_format(axis="x", style="plain")
        axis.grid(alpha=0.24)
    axes.flat[0].legend(fontsize=8, loc="best")
    figure.suptitle(
        f"GABLS1 hours 8–9: {model_label} vs official "
        f"{reference_resolution:g} m LES ensemble"
    )
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    group_paths: dict[str, Path] = {}
    group_filenames = {
        "A": "gabls1_set_a_means_overlay.png",
        "B": "gabls1_set_b_variances_overlay.png",
        "C": "gabls1_set_c_fluxes_overlay.png",
        "D": "gabls1_set_d_tke_budget_overlay.png",
    }
    for set_name, set_title, panels in PROFILE_GROUPS:
        path = output_dir / group_filenames[set_name]
        columns = 5 if set_name == "C" else 3
        _render_profile_group(
            path,
            f"GABLS1 hours 8–9, official {reference_resolution:g} m: {set_title}",
            panels,
            z,
            complete_references,
            model_profile_values,
            max_height=max_height,
            columns=columns,
            model_label=model_label,
        )
        group_paths[set_name] = path
    time_figure_path = output_dir / "gabls1_set_e_time_series_overlay.png"
    _render_time_group(
        time_figure_path,
        f"GABLS1, official {reference_resolution:g} m: Set E — time histories",
        complete_time_s,
        time_references,
        model_time_values,
        model_label=model_label,
    )
    complete_figure_path = output_dir / "gabls1_complete_overlay.png"
    _vertical_montage(
        [*(group_paths[name] for name in "ABCD"), time_figure_path],
        complete_figure_path,
    )

    comparison_path = output_dir / "official_ensemble_comparison.csv"
    _write_comparison(comparison_path, z, model, references)
    complete_profile_path = output_dir / "official_complete_profile_comparison.csv"
    _write_complete_profile_comparison(
        complete_profile_path,
        z,
        model_profile_values,
        complete_references,
    )
    complete_time_path = output_dir / "official_complete_time_comparison.csv"
    _write_complete_time_comparison(
        complete_time_path,
        complete_time_s,
        model_time_values,
        time_references,
    )
    profile_panel_names = [
        name
        for _set_name, _title, panels in PROFILE_GROUPS
        for name, _label in panels
    ]
    time_panel_names = [name for name, _label in TIME_PANELS]
    manifest_path = output_dir / "complete_overlay_manifest.json"
    manifest = {
        "comparison_period": "A9-D9: hours 8-9; E: full 0-9 h history",
        "reference_resolution_m": reference_resolution,
        "reference_source": metadata["source"],
        "panels_total": len(profile_panel_names) + len(time_panel_names),
        "participants_by_set": {
            **participants_by_set,
            "E": time_participants,
        },
        "model_overlaid": sorted(
            [name for name in profile_panel_names if name in model_profile_values]
            + [name for name in time_panel_names if name in model_time_values]
        ),
        "reference_only": sorted(
            [name for name in profile_panel_names if name not in model_profile_values]
            + [name for name in time_panel_names if name not in model_time_values]
        ),
        "derived_model_quantities": {
            "w_skewness": "w_third_moment / resolved_w_variance**1.5",
            "shear_production_resolved": "-(resolved_uw*du/dz + resolved_vw*dv/dz)",
            "shear_production_sgs": "-(sgs_uw*du/dz + sgs_vw*dv/dz)",
            "buoyancy_production": (
                "buoyancy_acceleration_per_scalar * resolved_scalar_flux"
            ),
            "dissipation": "negative resolved-TKE-to-SGS transfer",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    checkout_path = output_dir / "overlay_checkout.json"
    checkout = _checkout(
        z,
        model,
        references,
        [dataset.participant for dataset in profile_sets],
    )
    checkout["reference_resolution_m"] = reference_resolution
    checkout["reference_source"] = metadata["source"]
    checkout_path.write_text(json.dumps(checkout, indent=2) + "\n")
    return {
        "figure": figure_path,
        "complete_figure": complete_figure_path,
        "set_a_figure": group_paths["A"],
        "set_b_figure": group_paths["B"],
        "set_c_figure": group_paths["C"],
        "set_d_figure": group_paths["D"],
        "set_e_figure": time_figure_path,
        "comparison": comparison_path,
        "complete_profile_comparison": complete_profile_path,
        "complete_time_comparison": complete_time_path,
        "complete_manifest": manifest_path,
        "checkout": checkout_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, nargs="?", default=DEFAULT_RESULTS)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-height", type=float, default=250.0)
    args = parser.parse_args(argv)
    output = args.output_dir or args.result_dir / "overlays"
    reference = args.reference_dir or _select_reference_dir(args.result_dir)
    written = overlay_results(
        args.result_dir,
        reference,
        output,
        max_height=args.max_height,
    )
    print(json.dumps({name: str(path) for name, path in written.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
