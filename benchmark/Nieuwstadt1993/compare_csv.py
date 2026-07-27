#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_DIR = BENCHMARK_DIR / "reference" / "data"


@dataclass(frozen=True)
class Comparison:
    figure: str
    quantity: str
    profile_column: str
    resolved_column: str | None
    sgs_column: str | None
    xlabel: str
    csv_prefix: str


COMPARISONS = (
    Comparison(
        figure="Fig. 3",
        quantity="vertical_velocity_variance",
        profile_column="w_var_over_wstar_sq",
        resolved_column="w_var_resolved_over_wstar_sq",
        sgs_column="w_var_sgs_over_wstar_sq",
        xlabel=r"$\langle w'^2\rangle/w_*^2$",
        csv_prefix="fig3",
    ),
    Comparison(
        figure="Fig. 4",
        quantity="horizontal_velocity_variance",
        profile_column="horizontal_var_over_wstar_sq",
        resolved_column="horizontal_var_resolved_over_wstar_sq",
        sgs_column="horizontal_var_sgs_over_wstar_sq",
        xlabel=r"$\langle u_h'^2\rangle/w_*^2$",
        csv_prefix="fig4",
    ),
    Comparison(
        figure="Fig. 7",
        quantity="vertical_velocity_third_moment",
        profile_column="w3_over_wstar3",
        resolved_column=None,
        sgs_column=None,
        xlabel=r"$\langle w'^3\rangle/w_*^3$",
        # The extracted files predate the corrected paper-figure numbering.
        csv_prefix="fig5",
    ),
)


DATASETS = (
    ("num", "1993 numerical curve", "s", "#1f77b4"),
    ("exp", "laboratory", "^", "#d62728"),
    ("field", "atmosphere", "o", "#2ca02c"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare JAX Nieuwstadt1993 profiles with digitized paper/lab/field CSV data."
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=ROOT / "benchmark_results" / "Nieuwstadt1993" / "profiles.csv",
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "Nieuwstadt1993" / "comparison",
    )
    parser.add_argument(
        "--run-label",
        default="JAX run",
        help="Simulation label used in the comparison-figure title.",
    )
    return parser.parse_args()


def read_profiles(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No profile rows found in {path}")
    required = {"z_over_zi", *(comparison.profile_column for comparison in COMPARISONS)}
    required.update(
        column
        for comparison in COMPARISONS
        for column in (comparison.resolved_column, comparison.sgs_column)
        if column is not None
    )
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing profile columns in {path}: {sorted(missing)}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in required
    }


def read_digitized_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected two columns (value, z/zi) in {path}, got {data.shape}")
    finite = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    return data[finite, 0], data[finite, 1]


def interpolate_profile(
    model_value: np.ndarray,
    model_z: np.ndarray,
    reference_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(model_z)
    z_sorted = model_z[order]
    value_sorted = model_value[order]
    inside = (reference_z >= z_sorted[0]) & (reference_z <= z_sorted[-1])
    interpolated = np.interp(reference_z[inside], z_sorted, value_sorted)
    return interpolated, inside


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    profiles = read_profiles(args.profiles)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache_dir = args.output_dir / ".matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, len(COMPARISONS), figsize=(14.2, 5.2), sharey=True)

    for axis, comparison in zip(axes, COMPARISONS, strict=True):
        model_z = profiles["z_over_zi"]
        model_value = profiles[comparison.profile_column]
        if comparison.resolved_column is None:
            model_resolved = model_value
            model_sgs = np.zeros_like(model_value)
            axis.plot(model_value, model_z, color="black", linewidth=2.0, label="JAX resolved")
        else:
            model_resolved = profiles[comparison.resolved_column]
            model_sgs = profiles[comparison.sgs_column]
            axis.plot(model_value, model_z, color="black", linewidth=2.0, label="JAX total")
            axis.plot(
                model_resolved,
                model_z,
                color="0.4",
                linewidth=1.5,
                linestyle="--",
                label="JAX resolved",
            )
            axis.plot(
                model_sgs,
                model_z,
                color="#ff7f0e",
                linewidth=1.5,
                linestyle=":",
                label="JAX SGS contribution",
            )

        for suffix, label, marker, color in DATASETS:
            reference_path = args.reference_dir / f"{comparison.csv_prefix}_{suffix}.csv"
            reference_value, reference_z = read_digitized_csv(reference_path)
            interpolated, inside = interpolate_profile(model_value, model_z, reference_z)
            interpolated_resolved, resolved_inside = interpolate_profile(
                model_resolved, model_z, reference_z
            )
            interpolated_sgs, sgs_inside = interpolate_profile(model_sgs, model_z, reference_z)
            if not np.array_equal(inside, resolved_inside) or not np.array_equal(inside, sgs_inside):
                raise RuntimeError("Profile components do not share the same interpolation support")
            ref_inside = reference_value[inside]
            z_inside = reference_z[inside]
            error = interpolated - ref_inside
            ref_range = float(np.ptp(ref_inside))
            rmse = float(np.sqrt(np.mean(error * error)))
            metrics_rows.append(
                {
                    "figure": comparison.figure,
                    "quantity": comparison.quantity,
                    "dataset": suffix,
                    "dataset_label": label,
                    "n_points": int(error.size),
                    "rmse": rmse,
                    "mae": float(np.mean(np.abs(error))),
                    "bias_jax_minus_reference": float(np.mean(error)),
                    "max_abs_error": float(np.max(np.abs(error))),
                    "nrmse_by_reference_range": rmse / ref_range if ref_range > 0.0 else float("nan"),
                    "jax_peak_value": float(model_value[np.argmax(model_value)]),
                    "jax_peak_z_over_zi": float(model_z[np.argmax(model_value)]),
                    "reference_peak_value": float(reference_value[np.argmax(reference_value)]),
                    "reference_peak_z_over_zi": float(reference_z[np.argmax(reference_value)]),
                }
            )
            for z_value, reference, model, resolved, sgs, difference in zip(
                z_inside,
                ref_inside,
                interpolated,
                interpolated_resolved,
                interpolated_sgs,
                error,
                strict=True,
            ):
                point_rows.append(
                    {
                        "figure": comparison.figure,
                        "quantity": comparison.quantity,
                        "dataset": suffix,
                        "z_over_zi": float(z_value),
                        "reference_value": float(reference),
                        "jax_interpolated": float(model),
                        "jax_total_interpolated": float(model),
                        "jax_resolved_interpolated": float(resolved),
                        "jax_sgs_interpolated": float(sgs),
                        "jax_minus_reference": float(difference),
                    }
                )

            axis.scatter(
                reference_value,
                reference_z,
                marker=marker,
                facecolors="none" if suffix != "num" else color,
                edgecolors=color,
                s=30 if suffix != "num" else 18,
                linewidths=1.0,
                alpha=0.9,
                label=label,
            )

        axis.set_title(comparison.figure)
        axis.set_xlabel(comparison.xlabel)
        axis.set_ylim(0.0, 1.4)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$z/z_i$")
    legend_entries: dict[str, object] = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            legend_entries.setdefault(label, handle)
    fig.legend(
        legend_entries.values(),
        legend_entries.keys(),
        fontsize=8,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(f"Nieuwstadt et al. (1993): {args.run_label} vs digitized data")
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    fig.savefig(args.output_dir / "nieuwstadt1993_csv_comparison.png", dpi=200)
    plt.close(fig)

    metrics_fields = [
        "figure",
        "quantity",
        "dataset",
        "dataset_label",
        "n_points",
        "rmse",
        "mae",
        "bias_jax_minus_reference",
        "max_abs_error",
        "nrmse_by_reference_range",
        "jax_peak_value",
        "jax_peak_z_over_zi",
        "reference_peak_value",
        "reference_peak_z_over_zi",
    ]
    write_rows(args.output_dir / "csv_comparison_metrics.csv", metrics_fields, metrics_rows)
    point_fields = [
        "figure",
        "quantity",
        "dataset",
        "z_over_zi",
        "reference_value",
        "jax_interpolated",
        "jax_total_interpolated",
        "jax_resolved_interpolated",
        "jax_sgs_interpolated",
        "jax_minus_reference",
    ]
    write_rows(args.output_dir / "csv_comparison_points.csv", point_fields, point_rows)

    print(f"[comparison] wrote {args.output_dir / 'nieuwstadt1993_csv_comparison.png'}")
    for row in metrics_rows:
        if row["dataset"] == "num":
            print(
                f"  {row['figure']}: numerical-curve RMSE={row['rmse']:.4f}, "
                f"bias={row['bias_jax_minus_reference']:+.4f}, n={row['n_points']}"
            )


if __name__ == "__main__":
    main()
