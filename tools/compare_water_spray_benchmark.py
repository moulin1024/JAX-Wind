#!/usr/bin/env python3
"""Compare completed Montazeri-case-3 runs against digitized measurements."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def comparison(reference, directories, window):
    if "experimental_dbt_c" in reference:
        measured = np.asarray(reference["experimental_dbt_c"])
    else:
        pixels = np.asarray(reference["marker_x_pixels"])
        x0, x1 = reference["axis_x_pixels"]
        t0, t1 = reference["axis_temperature_c"]
        measured = t0 + (pixels - x0) / (x1 - x0) * (t1 - t0)
    result = {
        "reference": reference,
        "experimental_dbt_c_sorted": sorted(measured.tolist()),
        "experimental_sensor_mean_c": float(measured.mean()),
        "experimental_range_c": [float(measured.min()), float(measured.max())],
        "runs": [],
    }
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text())
        if summary["status"] != "complete":
            raise ValueError(f"incomplete run: {directory}")
        with (directory / "history.csv").open() as stream:
            rows = [
                {k: float(v) for k, v in row.items() if v}
                for row in csv.DictReader(stream)
                if row.get("sensor_0_dbt_c")
            ]
        end = summary["time_seconds"]
        samples = [r for r in rows if r["time_hours"] * 3600 > end - window - 1.0e-8]
        if len(samples) < 4:
            raise ValueError("comparison requires at least four diagnostic samples")
        sensors = np.array(
            [[r[f"sensor_{i}_dbt_c"] for i in range(9)] for r in samples]
        )
        means = sensors.mean(axis=1)
        predicted = sensors.mean(axis=0)
        half = len(means) // 2
        metrics = {
            "directory": str(directory),
            "cells": summary.get("cells"),
            "end_time_s": end,
            "window_s": window,
            "sample_count": len(samples),
            "predicted_sensor_dbt_c": predicted.tolist(),
            "sensor_mean_c": float(predicted.mean()),
            "sensor_range_c": [float(predicted.min()), float(predicted.max())],
            "mean_bias_k": float(predicted.mean() - measured.mean()),
            "sorted_distribution_rmse_k": float(
                np.sqrt(np.mean((np.sort(predicted) - np.sort(measured)) ** 2))
            ),
            "temporal_sensor_mean_range_k": float(np.ptp(means)),
            "late_minus_early_mean_k": float(means[half:].mean() - means[:half].mean()),
            "cooling_power_w": float(np.mean([r["cooling_power_w"] for r in samples])),
            "maximum_spray_mixing_ratio": max(
                r["maximum_spray_mixing_ratio"] for r in rows
            ),
            "minimum_water_mixing_ratio": min(
                r["minimum_water_mixing_ratio"] for r in rows
            ),
            "maximum_cfl": max(r["maximum_cfl"] for r in rows),
            "maximum_divergence_s": max(r["maximum_divergence_s"] for r in rows),
            "elapsed_seconds": summary["elapsed_seconds"],
            "devices": json.loads((directory / "run.json").read_text()).get("devices"),
        }
        metrics["mean_temperature_error_percent_celsius"] = float(
            100 * abs(predicted.mean() - measured.mean()) / abs(measured.mean())
        )
        metrics["mean_temperature_within_10_percent_celsius"] = (
            metrics["mean_temperature_error_percent_celsius"] <= 10
        )
        if reference.get("spatial_pairing_known", False):
            relative_errors = 100 * np.abs(predicted - measured) / np.abs(measured)
            metrics["sensor_temperature_errors_percent_celsius"] = (
                relative_errors.tolist()
            )
            metrics["maximum_sensor_error_percent_celsius"] = float(
                relative_errors.max()
            )
            metrics["sensors_exceeding_10_percent_celsius"] = int(
                np.sum(relative_errors > 10)
            )
            metrics["all_sensor_temperatures_within_10_percent_celsius"] = bool(
                np.all(relative_errors <= 10)
            )
            metrics["spatially_paired_rmse_k"] = float(
                np.sqrt(np.mean((predicted - measured) ** 2))
            )
            metrics["maximum_spatially_paired_error_k"] = float(
                np.max(np.abs(predicted - measured))
            )
        if "inlet_resolved_k_m2_s2" in samples[0]:
            metrics["mean_inlet_resolved_k_m2_s2"] = float(
                np.mean([r["inlet_resolved_k_m2_s2"] for r in samples])
            )
        from jaxwind.io.checkpoint import checkpoint_metadata

        header = checkpoint_metadata(directory / "checkpoint.npz")
        metrics["cells"] = header["resolved_case"]["mesh"]["cells"]
        metrics["devices"] = header["devices"]
        metrics["case_name"] = header["resolved_case"]["case"]["name"]
        metrics["spray_model"] = header["resolved_case"]["case"].get(
            "spray_model", "entrained"
        )
        metrics["dt_s"] = header["resolved_case"]["time"]["dt_seconds"]
        metrics["parcels_per_step"] = header["resolved_case"]["case"].get(
            "parcels_per_step"
        )
        metrics["sorted_distribution_within_1K_screening_threshold"] = (
            metrics["sorted_distribution_rmse_k"] <= 1.0
        )
        if metrics["spray_model"] == "inertial":
            metrics["maximum_parcel_volume_fraction"] = max(
                r["parcel_maximum_volume_fraction"] for r in rows
            )
            metrics["maximum_parcel_mass_balance_error_kg"] = max(
                abs(r["parcel_mass_balance_error_kg"]) for r in rows
            )
            metrics["maximum_parcel_count"] = max(r["parcel_count"] for r in rows)
            metrics["minimum_gas_temperature_k"] = min(
                r["minimum_gas_temperature_k"] for r in rows
            )
            metrics["parcel_evaporated_mass_kg"] = rows[-1]["parcel_evaporated_mass_kg"]
            metrics["parcel_injected_mass_kg"] = rows[-1]["parcel_injected_mass_kg"]
            metrics["parcel_dilute_volume_screening_exceeded"] = (
                metrics["maximum_parcel_volume_fraction"] > 0.001
            )
        metrics["within_1K_mean_screening_threshold"] = (
            abs(metrics["mean_bias_k"]) <= 1.0
        )
        metrics["boussinesq_loading_assumption_exceeded"] = (
            metrics["spray_model"] == "entrained"
            and metrics["maximum_spray_mixing_ratio"] > 0.1
        )
        result["runs"].append(metrics)
    result["adjacent_run_differences"] = []
    for before, after in zip(result["runs"][:-1], result["runs"][1:]):
        change = np.asarray(after["predicted_sensor_dbt_c"]) - np.asarray(
            before["predicted_sensor_dbt_c"]
        )
        result["adjacent_run_differences"].append(
            {
                "before": before["case_name"],
                "after": after["case_name"],
                "sensor_changes_k": change.tolist(),
                "sensor_mean_change_k": float(np.mean(change)),
                "sensor_rms_change_k": float(np.sqrt(np.mean(change**2))),
                "maximum_sensor_change_k": float(np.max(np.abs(change))),
                "interpretation": "Run differences; identify controlled inputs before attributing them to mesh or timestep refinement.",
            }
        )
    result["interpretation"] = (
        "Model-adequacy challenge, not validated air-assisted atomization. "
        "Sorted-distribution error is not a spatially paired error. "
        "1 K is a declared engineering screening threshold, not experimental uncertainty. "
        "10% metrics use the reported Celsius outlet temperatures, not Kelvin or temperature reduction; "
        "mean and individual-sensor acceptance are reported separately. "
        "No parameters fitted to the temperature targets."
    )
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("cases/WaterSprayMontazeri2015/reference.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cases/WaterSprayMontazeri2015/comparison")
    )
    parser.add_argument("--window", type=float, default=1.0)
    args = parser.parse_args()
    result = comparison(json.loads(args.reference.read_text()), args.runs, args.window)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    x = np.arange(1, 10)
    reference_is_table = "experimental_dbt_c" in result["reference"]
    ax.errorbar(
        x,
        result["experimental_dbt_c_sorted"],
        yerr=result["reference"].get(
            "temperature_uncertainty_c",
            result["reference"].get("digitization_uncertainty_c", 0.1),
        ),
        fmt="ko",
        label=(
            "Experiment (table; ±0.3 K reported uncertainty)"
            if reference_is_table
            else "Experiment (digitized; ±0.1 K reading uncertainty)"
        ),
    )
    for run in result["runs"]:
        parcel_label = (
            ""
            if run["parcels_per_step"] is None
            else f", {run['parcels_per_step']} parcels/step"
        )
        ax.plot(
            x,
            sorted(run["predicted_sensor_dbt_c"]),
            "o-",
            label=run["case_name"].replace("montazeri2015-case3-", "")
            + " "
            + "×".join(map(str, run["cells"]))
            + f" (dt={run['dt_s']:g} s{parcel_label})",
        )
    ax.set(
        xlabel="Rank within the nine-sensor set (not spatial pairing)",
        ylabel="Outlet dry-bulb temperature (°C)",
        title="Montazeri 2015 case 3: subgrid spray comparison",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(args.output / "comparison.svg")
    fig.savefig(args.output / "comparison.png", dpi=180)
    if result["reference"].get("spatial_pairing_known", False):
        measured = np.asarray(result["reference"]["experimental_dbt_c"])
        errors = [
            np.asarray(r["predicted_sensor_dbt_c"]) - measured for r in result["runs"]
        ]
        extent = max(float(np.max(np.abs(e))) for e in errors)
        paired_fig, axes = plt.subplots(
            1,
            len(errors),
            figsize=(3.3 * len(errors) + 1, 3.6),
            squeeze=False,
            constrained_layout=True,
        )
        for ax, error, run in zip(axes[0], errors, result["runs"]):
            values = error.reshape(3, 3)
            im = ax.imshow(
                values, origin="lower", cmap="coolwarm", vmin=-extent, vmax=extent
            )
            for row in range(3):
                for col in range(3):
                    ax.text(
                        col,
                        row,
                        f"{values[row, col]:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
            ax.set_xticks([0, 1, 2], ["Left", "Center", "Right"])
            ax.set_yticks([0, 1, 2], ["Bottom", "Middle", "Top"])
            ax.set_title(
                run["case_name"].replace("montazeri2015-case3-", "").replace("-", "\n"),
                fontsize=9,
            )
        paired_fig.colorbar(
            im, ax=axes[0].tolist(), label="Prediction − measurement (K)", shrink=0.8
        )
        paired_fig.savefig(args.output / "spatial_comparison.svg")
        paired_fig.savefig(args.output / "spatial_comparison.png", dpi=180)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
