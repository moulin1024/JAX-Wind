#!/usr/bin/env python3
"""Paired waterjet DBT/WBT assessment of a completed Fluent-DPM tunnel run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    f"{r}-{c}" for r in ("bottom", "middle", "top") for c in ("left", "centre", "right")
]


def window_values(times, values, start, end):
    if not times[0] <= start < end <= times[-1] + 1e-8:
        raise ValueError("history does not span requested averaging window")
    t = np.r_[start, times[(times > start) & (times < end)], end]
    v = np.stack([np.interp(t, times, a) for a in np.asarray(values).T], axis=1)
    return t, v


def average(times, values, start, end):
    t, v = window_values(times, values, start, end)
    return np.sum((v[1:] + v[:-1]) * 0.5 * np.diff(t)[:, None], axis=0) / (end - start)


def assess(directory, output, start=3.0, end=4.0):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete" or summary["time_seconds"] < end - 1e-8:
        raise ValueError(
            "comparison requires a complete run spanning the requested window"
        )
    import tomllib

    doc = tomllib.loads((directory / "resolved_case.toml").read_text())
    if doc["case"].get("spray_model") != "fluent-dpm":
        raise ValueError("this assessment requires the actual Fluent-DPM driver")
    reference_path = (
        ROOT
        / "cases/WaterSprayMontazeri2015/uploaded_reference_validation/reference.json"
    )
    reference = json.loads(reference_path.read_text())
    if (
        hashlib.sha256((ROOT / "waterjet_referece.pdf").read_bytes()).hexdigest()
        != reference["source_sha256"]
    ):
        raise ValueError("experimental PDF checksum mismatch")
    acceptance = json.loads(
        (ROOT / "cases/WaterSprayMontazeri2015/acceptance.json").read_text()
    )
    with (directory / "history.csv").open() as stream:
        records = [
            {k: float(v) for k, v in r.items() if v} for r in csv.DictReader(stream)
        ]
    times = np.array([r["time_hours"] * 3600 for r in records])
    if np.any(np.diff(times) <= 0):
        raise ValueError("history must have strictly increasing sample times")
    dbt = np.array([[r[f"sensor_{i}_dbt_c"] for i in range(9)] for r in records])
    wbt = np.array([[r[f"sensor_{i}_wbt_c"] for i in range(9)] for r in records])
    if not np.all(np.isfinite(dbt)):
        raise ValueError("nonfinite DBT history")
    predicted = average(times, dbt, start, end)
    predicted_wbt = average(times, wbt, start, end)
    measured = np.array(reference["experimental_dbt_c"])
    measured_wbt = np.array(reference["experimental_wbt_c"])
    errors = predicted - measured
    relative = 100 * abs(errors) / abs(measured)
    inlet = reference["operating_conditions"]["inlet_dbt_c"]
    threshold = 100 * acceptance["relative_tolerance"]
    mid = (start + end) / 2
    shift = average(times, dbt, mid, end) - average(times, dbt, start, mid)
    balances = {
        key: max(abs(r[key]) for r in records)
        for key in (
            "dpm_water_budget_error_kg",
            "dpm_carrier_vapor_balance_error_kg",
            "dpm_max_source_energy_error_J",
            "maximum_divergence_s",
            "maximum_cfl",
        )
    }

    def delta(key):
        v = np.array([r[key] for r in records])
        return float(np.interp(end, times, v) - np.interp(start, times, v))

    collected_mass = delta("dpm_collected_mass_kg")
    trapped_mass = delta("dpm_trapped_mass_kg")
    collected_h = delta("dpm_collected_enthalpy_J")
    trapped_h = delta("dpm_trapped_enthalpy_J")
    # Same explicit material as the benchmark, reference liquid enthalpy at 0 C.
    from jaxwind.physics.fluent_dpm import DPMWaterMaterial

    material = DPMWaterMaterial()

    def liquid_temperature(m, h):
        return (
            material.reference_temperature - 273.15 + h / (m * material.liquid_cp)
            if m > 0
            else None
        )

    result = {
        "run": str(directory),
        "window_seconds": [start, end],
        "averaging": "trapezoidal time average with interpolated window endpoints",
        "reference": reference,
        "acceptance": acceptance,
        "measured_mean_c": float(measured.mean()),
        "predicted_mean_c": float(predicted.mean()),
        "predicted_dbt_c": predicted.tolist(),
        "predicted_wbt_c": [
            float(v) if np.isfinite(v) else None for v in predicted_wbt
        ],
        "error_k": errors.tolist(),
        "error_percent_celsius": relative.tolist(),
        "maximum_error_percent_celsius": float(relative.max()),
        "maximum_error_k": float(abs(errors).max()),
        "worst_sensor": LABELS[int(relative.argmax())],
        "passing_sensors": int(np.sum(relative <= threshold)),
        "paired_rmse_k": float(np.sqrt(np.mean(errors**2))),
        "cooling_error_percent": (100 * abs(errors) / (inlet - measured)).tolist(),
        "maximum_half_window_shift_k": float(abs(shift).max()),
        "half_window_sensor_shift_k": shift.tolist(),
        "maximum_numerical_residuals": balances,
        "separator_collected_mass_in_window_kg": collected_mass,
        "wall_trapped_mass_in_window_kg": trapped_mass,
        "separator_water_temperature_c": liquid_temperature(
            collected_mass, collected_h
        ),
        "combined_drain_temperature_c": liquid_temperature(
            collected_mass + trapped_mass, collected_h + trapped_h
        ),
        "observation": doc["case"]["observation"],
        "limitations": [
            "Ideal perfect adiabatic separator; no pressure loss, mist carryover or plate/film heat exchange",
            "Assumed downstream sensor spacing and source size/angle/velocity distribution",
            "Single grid, timestep and parcel quadrature; no convergence claim",
        ],
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                directory / "history.csv",
                directory / "resolved_case.toml",
                directory / "summary.json",
                reference_path,
            ]
        },
    }
    wbt_error = predicted_wbt - measured_wbt
    result["wbt_error_k"] = [float(v) if np.isfinite(v) else None for v in wbt_error]
    result["maximum_wbt_error_k"] = (
        float(np.max(np.abs(wbt_error))) if np.all(np.isfinite(wbt_error)) else None
    )
    result["wbt_paired_rmse_k"] = (
        float(np.sqrt(np.mean(wbt_error**2)))
        if np.all(np.isfinite(wbt_error))
        else None
    )
    result["experimental_collected_water_c"] = reference["outlet_water_temperature_c"]
    if times[0] <= start - (end - start):
        previous = average(times, dbt, start - (end - start), start)
        result["previous_window_sensor_shift_k"] = (predicted - previous).tolist()
        result["maximum_previous_window_shift_k"] = float(
            abs(predicted - previous).max()
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (output / "sensors.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sensor",
                "measured_dbt_c",
                "predicted_dbt_c",
                "error_k",
                "error_percent_celsius",
                "pass_acceptance",
                "measured_wbt_c",
                "predicted_wbt_c",
                "wbt_error_k",
            ]
        )
        for i, name in enumerate(LABELS):
            writer.writerow(
                [
                    name,
                    measured[i],
                    predicted[i],
                    errors[i],
                    relative[i],
                    relative[i] <= threshold,
                    measured_wbt[i],
                    predicted_wbt[i],
                    predicted_wbt[i] - measured_wbt[i],
                ]
            )
    rows = [
        "| Sensor | Experiment °C | DPM °C | Error K | Error % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    rows += [
        f"| {name} | {measured[i]:.1f} | {predicted[i]:.3f} | {errors[i]:+.3f} | {relative[i]:.2f} |"
        for i, name in enumerate(LABELS)
    ]
    (output / "tables.md").write_text("\n".join(rows) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    x = np.arange(9)
    for ax, obs, pred, label in [
        (axes[0], measured, predicted, "Dry bulb"),
        (axes[1], measured_wbt, predicted_wbt, "Wet bulb"),
    ]:
        ax.plot(x, obs, "ko", label="Experiment")
        ax.plot(x, pred, "o-", label=f"DPM mean {start:g}–{end:g} s")
        ax.set(xticks=x, xticklabels=LABELS, ylabel=f"{label} (°C)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(output / "sensors.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.plot(times, dbt)
    ax.axvspan(start, end, color="0.8", alpha=0.4)
    ax.set(
        xlabel="Time (s)",
        ylabel="Sensor DBT (°C)",
        title="Nine DPM sensor histories; shaded averaging window",
    )
    ax.grid(alpha=0.2)
    fig.savefig(output / "history.png", dpi=160)
    plt.close(fig)
    print("\n".join(rows))
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "maximum_error_percent_celsius",
                    "worst_sensor",
                    "passing_sensors",
                    "maximum_half_window_shift_k",
                    "maximum_numerical_residuals",
                ]
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=3.0)
    parser.add_argument("--end", type=float, default=4.0)
    args = parser.parse_args()
    assess(args.directory, args.output, args.start, args.end)
