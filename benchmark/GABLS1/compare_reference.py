"""Compare a GABLS1 run directly with the official raw LES submissions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from benchmark.GABLS1.reference import (
    ensemble_on_grid,
    ensemble_statistics,
    interpolate_values,
    load_period_sets,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
PROFILE_VARIABLES = {
    "u_mean": "mean_u_m_s",
    "v_mean": "mean_v_m_s",
    "theta_mean": "mean_potential_temperature_k",
}
FLUX_VARIABLES = {
    "uw_total": "uw_total_m2_s2",
    "vw_total": "vw_total_m2_s2",
    "wtheta_total": "wtheta_total_k_m_s",
}
FLUX_COMPONENTS = {
    "uw_total": ("uw_resolved", "uw_sgs"),
    "vw_total": ("vw_resolved", "vw_sgs"),
    "wtheta_total": ("wtheta_resolved", "wtheta_sgs"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=HERE / "reference" / "official_12p5m",
        help="directory containing raw participant .dat files",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-height", type=float, default=250.0)
    return parser.parse_args()


def _read_csv(path: Path) -> np.ndarray:
    values = np.genfromtxt(path, delimiter=",", names=True)
    if values.size == 0:
        raise ValueError(f"empty result file: {path}")
    return np.atleast_1d(values)


def _total_flux_ensemble(reference_dir: Path, target: np.ndarray) -> dict:
    datasets = load_period_sets(reference_dir, "C", period=9)
    result = {}
    for total_name, components in FLUX_COMPONENTS.items():
        participant_values = []
        for dataset in datasets:
            resolved = interpolate_values(dataset, "z", components[0], target)
            subgrid = interpolate_values(dataset, "z", components[1], target)
            participant_values.append(resolved + subgrid)
        result[total_name] = ensemble_statistics(np.stack(participant_values))
    return result


def _write_ensemble_csv(
    path: Path,
    coordinate_name: str,
    coordinate: np.ndarray,
    model: np.ndarray,
    variables: dict[str, str],
    ensemble: dict,
) -> None:
    flat_fields = [coordinate_name]
    for reference_name in variables:
        flat_fields.append(f"model_{reference_name}")
        flat_fields.extend(
            f"reference_{reference_name}_{stat}"
            for stat in ("mean", "minimum", "maximum", "standard_deviation", "count")
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_fields)
        writer.writeheader()
        for index, z_value in enumerate(coordinate):
            row = {coordinate_name: float(z_value)}
            for reference_name, model_name in variables.items():
                row[f"model_{reference_name}"] = float(model[model_name][index])
                for stat in ("mean", "minimum", "maximum", "standard_deviation"):
                    row[f"reference_{reference_name}_{stat}"] = float(
                        ensemble[reference_name][stat][index]
                    )
                row[f"reference_{reference_name}_count"] = int(
                    ensemble[reference_name]["count"][index]
                )
            writer.writerow(row)


def _in_range_fraction(values: np.ndarray, stats: dict, mask: np.ndarray) -> float:
    inside = (values >= stats["minimum"]) & (values <= stats["maximum"]) & mask
    denominator = np.count_nonzero(mask & np.isfinite(stats["mean"]))
    return float(np.count_nonzero(inside) / denominator) if denominator else np.nan


def _metrics(
    profile: np.ndarray,
    flux: np.ndarray,
    profile_ensemble: dict,
    flux_ensemble: dict,
    participants: list[str],
    source: dict,
) -> dict:
    z = np.asarray(profile["z_m"], dtype=float)
    z_face = np.asarray(flux["z_face_m"], dtype=float)
    below_200 = z <= 200.0
    theta = np.asarray(profile["mean_potential_temperature_k"], dtype=float)
    theta_reference = profile_ensemble["theta_mean"]["mean"]
    u = np.asarray(profile["mean_u_m_s"], dtype=float)
    v = np.asarray(profile["mean_v_m_s"], dtype=float)
    speed = np.hypot(u, v)
    ref_speed = np.hypot(
        profile_ensemble["u_mean"]["mean"],
        profile_ensemble["v_mean"]["mean"],
    )
    jet_mask = z <= 300.0
    model_jet_index = np.nanargmax(np.where(jet_mask, speed, np.nan))
    reference_jet_index = np.nanargmax(np.where(jet_mask, ref_speed, np.nan))
    uw_model = np.asarray(flux["uw_total_m2_s2"], dtype=float)
    vw_model = np.asarray(flux["vw_total_m2_s2"], dtype=float)
    uw_ref = flux_ensemble["uw_total"]["mean"]
    vw_ref = flux_ensemble["vw_total"]["mean"]
    return {
        "comparison_period": "hours 8-9 (official A9/B9/C9 records)",
        "raw_reference_directory": "benchmark/GABLS1/reference/official_12p5m",
        "raw_reference_archive": source,
        "participants": participants,
        "participant_count": len(participants),
        "model_surface_heat_flux_k_m_s": float(flux["wtheta_total_k_m_s"][0]),
        "reference_surface_heat_flux_mean_k_m_s": float(
            flux_ensemble["wtheta_total"]["mean"][0]
        ),
        "reference_surface_heat_flux_range_k_m_s": [
            float(flux_ensemble["wtheta_total"]["minimum"][0]),
            float(flux_ensemble["wtheta_total"]["maximum"][0]),
        ],
        "model_surface_friction_velocity_m_s": float(
            np.hypot(uw_model[0], vw_model[0]) ** 0.5
        ),
        "reference_surface_friction_velocity_from_mean_stress_m_s": float(
            np.hypot(uw_ref[0], vw_ref[0]) ** 0.5
        ),
        "theta_rmse_below_200m_k": float(
            np.sqrt(np.nanmean((theta[below_200] - theta_reference[below_200]) ** 2))
        ),
        "theta_fraction_within_participant_range_below_200m": _in_range_fraction(
            theta, profile_ensemble["theta_mean"], below_200
        ),
        "model_low_level_jet": {
            "speed_m_s": float(speed[model_jet_index]),
            "height_m": float(z[model_jet_index]),
        },
        "reference_ensemble_mean_low_level_jet": {
            "speed_m_s": float(ref_speed[reference_jet_index]),
            "height_m": float(z[reference_jet_index]),
        },
        "maximum_compared_height_m": float(min(250.0, z_face[-1])),
    }


def _plot(
    output: Path,
    profile: np.ndarray,
    flux: np.ndarray,
    profile_ensemble: dict,
    flux_ensemble: dict,
    model_label: str,
    participant_count: int,
    max_height: float,
) -> None:
    panels = (
        (profile, profile_ensemble, "u_mean", "mean_u_m_s", r"$\langle u\rangle$ (m s$^{-1}$)"),
        (profile, profile_ensemble, "v_mean", "mean_v_m_s", r"$\langle v\rangle$ (m s$^{-1}$)"),
        (profile, profile_ensemble, "theta_mean", "mean_potential_temperature_k", r"$\langle\theta\rangle$ (K)"),
        (flux, flux_ensemble, "uw_total", "uw_total_m2_s2", r"total $\langle u'w'\rangle$ (m$^2$ s$^{-2}$)"),
        (flux, flux_ensemble, "vw_total", "vw_total_m2_s2", r"total $\langle v'w'\rangle$ (m$^2$ s$^{-2}$)"),
        (flux, flux_ensemble, "wtheta_total", "wtheta_total_k_m_s", r"total $\langle w'\theta'\rangle$ (K m s$^{-1}$)"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 8.4), constrained_layout=True)
    for axis, (model, reference, ref_name, model_name, xlabel) in zip(
        axes.flat, panels, strict=True
    ):
        z_name = "z_m" if model is profile else "z_face_m"
        z = np.asarray(model[z_name], dtype=float)
        stats = reference[ref_name]
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
        axis.set(xlabel=xlabel, ylabel="z (m)", ylim=(0.0, max_height))
        axis.grid(alpha=0.24)
    axes.flat[0].legend(fontsize=8, loc="best")
    figure.suptitle(
        f"GABLS1 hours 8–9: {model_label} vs raw official 12.5 m LES "
        f"ensemble (n={participant_count})"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.reference_dir.is_dir():
        raise FileNotFoundError(f"raw reference directory not found: {args.reference_dir}")
    profile = _read_csv(args.result_dir / "profiles.csv")
    flux = _read_csv(args.result_dir / "flux_profiles.csv")
    z = np.asarray(profile["z_m"], dtype=float)
    z_face = np.asarray(flux["z_face_m"], dtype=float)
    raw_a = load_period_sets(args.reference_dir, "A", period=9)
    participants = [dataset.participant for dataset in raw_a]
    profile_ensemble = ensemble_on_grid(raw_a, "z", z)
    flux_ensemble = _total_flux_ensemble(args.reference_dir, z_face)
    source_path = args.reference_dir / "SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8"))
    model_name = summary["case"]["sgs"]["model"].upper()
    shape = summary["case"]["domain"]["nx"]
    model_label = f"JAX-Wind {model_name} {shape}³"
    output = args.output or args.result_dir / "gabls1_raw_reference_comparison.png"
    _plot(
        output,
        profile,
        flux,
        profile_ensemble,
        flux_ensemble,
        model_label,
        len(participants),
        args.max_height,
    )
    _write_ensemble_csv(
        args.result_dir / "raw_reference_profile_comparison.csv",
        "z_m",
        z,
        profile,
        PROFILE_VARIABLES,
        profile_ensemble,
    )
    _write_ensemble_csv(
        args.result_dir / "raw_reference_flux_comparison.csv",
        "z_face_m",
        z_face,
        flux,
        FLUX_VARIABLES,
        flux_ensemble,
    )
    metrics = _metrics(
        profile,
        flux,
        profile_ensemble,
        flux_ensemble,
        participants,
        source,
    )
    (args.result_dir / "raw_reference_comparison.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
