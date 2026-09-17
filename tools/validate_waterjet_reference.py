#!/usr/bin/env python3
"""Recheck archived cooling runs against the uploaded Sureshkumar Table 2(b).

Run from the repository root with PYTHONPATH=src. No simulations or parameter
fitting are performed. The PDF transcription was visually checked on page 6.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_water_spray_energy import audit
from compare_water_spray_benchmark import comparison

PDF_SHA256 = "2e8d080d7944bea85d941328d40f0aafb85b906af1416c932847510dd132c95d"
RUN_NAMES = (
    "coarse",
    "fine",
    "inertial_energy_audit",
    "inertial_energy_half_dt",
    "inertial_refined",
    "inertial_smooth_walls",
    "inertial_wall_audit",
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("waterjet_referece.pdf"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cases/WaterSprayMontazeri2015/uploaded_reference_validation"),
    )
    args = parser.parse_args()
    if hashlib.sha256(args.pdf.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError("PDF differs from the visually verified source")

    reference = json.loads(
        Path("cases/WaterSprayMontazeri2015/reference_table.json").read_text()
    )
    # Bottom/middle/top; left/centre/right viewed from the tunnel exit.
    measured = np.array([30.7, 32.2, 32.2, 31.9, 31.4, 31.7, 31.0, 31.6, 31.7])
    np.testing.assert_array_equal(reference["experimental_dbt_c"], measured)
    reference.update(
        source_pdf=str(args.pdf),
        source_sha256=PDF_SHA256,
        source_location="PDF p.6, printed p.354, Table 2(b), last row; geometry: Fig.2, PDF p.3",
        method="Direct table transcription, visually verified against the uploaded PDF",
        experimental_wbt_c=[21.4, 20.8, 19.9, 20.5, 20.2, 20.2, 21.0, 20.8, 20.4],
        operating_conditions={
            "air_speed_m_s": 3.0,
            "inlet_dbt_c": 39.2,
            "inlet_wbt_c": 18.7,
            "nozzle_diameter_mm": 4.0,
            "water_gauge_pressure_bar": 3.0,
            "water_inlet_c": 35.2,
            "water_outlet_c": 26.1,
            "water_flow_l_min": 12.5,
            "configuration": "parallel flow, hot-dry",
            "tunnel_dimensions_m": [1.9, 0.585, 0.585],
        },
    )
    directories = [Path("outputs/water_spray_montazeri2015") / n for n in RUN_NAMES]
    result = comparison(reference, directories, window=1.0)
    relative_tolerance = result["acceptance"]["relative_tolerance"]
    threshold = 100 * relative_tolerance
    energy = audit(directories[-1])
    # Fingerprint the saved numerical inputs, not just the reference data.
    result["input_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for d in directories
        for p in (d / "history.csv", d / "summary.json", d / "resolved_case.toml")
    }
    result["scope"] = (
        "Archived cooling runs; current kernels tested separately. No new coupled simulation."
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("reference.json", reference),
        ("comparison.json", result),
        ("energy_audit.json", energy),
    ):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")

    labels = [
        f"{row}-{col}"
        for row in ("bottom", "middle", "top")
        for col in ("left", "centre", "right")
    ]
    latest = result["runs"][-1]
    predicted = np.asarray(latest["predicted_sensor_dbt_c"])
    errors = 100 * np.abs(predicted - measured) / np.abs(measured)
    with (args.output / "sensors.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sensor",
                "measured_dbt_c",
                "measured_wbt_c",
                "predicted_dbt_c",
                "error_k",
                "error_percent_celsius",
                "pass_acceptance",
            ]
        )
        for i, label in enumerate(labels):
            writer.writerow(
                [
                    label,
                    measured[i],
                    reference["experimental_wbt_c"][i],
                    predicted[i],
                    predicted[i] - measured[i],
                    errors[i],
                    errors[i] <= threshold,
                ]
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    x = np.arange(9)
    ax.fill_between(
        x,
        (1-relative_tolerance) * measured,
        (1+relative_tolerance) * measured,
        color="0.9",
        label=f"Current ±{threshold:g}% Celsius criterion",
    )
    ax.errorbar(
        x,
        measured,
        yerr=0.3,
        fmt="ko",
        capsize=3,
        label="Table 2(b): measured DBT ±0.3°C",
    )
    ax.plot(x, predicted, "o-", label="Archived inertial spray, smooth walls")
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Outlet dry-bulb temperature (°C)",
        title="Sureshkumar (2008): spatially paired sensors, simulation mean over 3–4 s",
    )
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(args.output / "sensors.svg")
    fig.savefig(args.output / "sensors.png", dpi=160)
    plt.close(fig)

    rows = [
        "| Run | Mean DBT °C | Paired RMSE K | Worst error % | Failed sensors |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for run in result["runs"]:
        rows.append(
            f"| {Path(run['directory']).name} | {run['sensor_mean_c']:.3f} | "
            f"{run['spatially_paired_rmse_k']:.3f} | "
            f"{run['maximum_sensor_error_percent_celsius']:.2f} | "
            f"{run['sensors_exceeding_acceptance']}/9 |"
        )
    rows += [
        "",
        "| Sensor | Measured °C | Predicted °C | Error % | Result |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for i, label in enumerate(labels):
        rows.append(
            f"| {label} | {measured[i]:.1f} | {predicted[i]:.3f} | "
            f"{errors[i]:.2f} | {'PASS' if errors[i] <= threshold else 'FAIL'} |"
        )
    (args.output / "tables.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
