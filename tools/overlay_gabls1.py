"""Overlay a uniform ABL result on the official GABLS1 8--9 h ensemble."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.ticker import MaxNLocator  # noqa: E402

try:
    from tools.gabls1_reference import (
        ensemble_on_grid,
        ensemble_statistics,
        interpolate_values,
        load_period_sets,
    )
except ModuleNotFoundError:  # direct ``python tools/overlay_gabls1.py``
    from gabls1_reference import (
        ensemble_on_grid,
        ensemble_statistics,
        interpolate_values,
        load_period_sets,
    )


import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "outputs" / "gabls1_lasd_32x32x32"
DEFAULT_REFERENCE = ROOT / "cases" / "GABLS1" / "reference" / "official_12p5m"
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


def _read_csv(path: Path) -> np.ndarray:
    values = np.genfromtxt(path, delimiter=",", names=True)
    if values.size == 0:
        raise ValueError(f"empty result file: {path}")
    return np.atleast_1d(values)


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


def overlay_results(
    result_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    *,
    max_height: float = 250.0,
) -> dict[str, Path]:
    model = _read_csv(result_dir / "profiles.csv")
    z = np.asarray(model["z_m"], dtype=float)
    profile_sets = load_period_sets(reference_dir, "A", period=9)
    profiles = ensemble_on_grid(profile_sets, z)
    references = {**profiles, **_flux_ensemble(reference_dir, z)}
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
        axis.plot(model[model_name], z, color="#d62728", lw=2.2, label="JAX-Wind")
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
        "GABLS1 hours 8–9: uniform JAX-Wind ABL solver vs official 12.5 m LES ensemble"
    )
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    comparison_path = output_dir / "official_ensemble_comparison.csv"
    _write_comparison(comparison_path, z, model, references)
    checkout_path = output_dir / "overlay_checkout.json"
    checkout = _checkout(
        z,
        model,
        references,
        [dataset.participant for dataset in profile_sets],
    )
    checkout_path.write_text(json.dumps(checkout, indent=2) + "\n")
    return {
        "figure": figure_path,
        "comparison": comparison_path,
        "checkout": checkout_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, nargs="?", default=DEFAULT_RESULTS)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-height", type=float, default=250.0)
    args = parser.parse_args(argv)
    output = args.output_dir or args.result_dir / "overlays"
    written = overlay_results(
        args.result_dir,
        args.reference_dir,
        output,
        max_height=args.max_height,
    )
    print(json.dumps({name: str(path) for name, path in written.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
