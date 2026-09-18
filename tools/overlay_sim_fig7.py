#!/usr/bin/env python3
"""Overlay paired inertial predictions on the original nine-panel Figure 7."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from jaxwind.physics.moisture import MoistureConfig, saturation_vapor_pressure_water
from jaxwind.waterjet_observation import wet_bulb_c

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
SENSORS = [
    f"{r}-{c}" for r in ("bottom", "middle", "top") for c in ("left", "centre", "right")
]
# Native image coordinates, verified against long black frame lines.
XFRAMES = [(80.5, 669.5), (800.5, 1387.5), (1512.0, 2099.0)]
YFRAMES = [(49.0, 534.0), (658.0, 1144.5), (1273.5, 1760.0)]
LIMITS = [(25.0, 35.0), (15.0, 25.0), (50.0, 70.0)]
NAMES = ["DBT (°C)", "WBT (°C)", "h (kJ/kg dry air)"]


def average(t, v, start, end):
    if not t[0] <= start < end <= t[-1] + 1e-8:
        raise ValueError("Averaging window outside completed history")
    times = np.r_[start, t[(t > start) & (t < end)], end]
    values = np.stack([np.interp(times, t, x) for x in np.asarray(v).T], axis=1)
    return np.sum(
        (values[1:] + values[:-1]) * 0.5 * np.diff(times)[:, None], axis=0
    ) / (end - start)


def humidity(td, tw, pressure=101325.0):
    cfg = MoistureConfig()
    e = np.asarray(saturation_vapor_pressure_water(jnp.asarray(tw) + 273.15))
    e = e - 0.00066 * (1 + 0.00115 * np.asarray(tw)) * pressure * (np.asarray(td) - tw)
    return cfg.dry_air_gas_constant / cfg.water_vapor_gas_constant * e / (pressure - e)


def enthalpy(td, q):
    return 1.005 * np.asarray(td) + np.asarray(q) * (2500.0 + 1.859 * np.asarray(td))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "outputs/sim_fig7")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/sim_fig7/comparison"
    )
    parser.add_argument("--start", type=float, default=3.0)
    parser.add_argument("--end", type=float, default=4.0)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    refpath = ROOT / "cases/WaterSprayMontazeri2015/sim_fig7/reference.json"
    reference = json.loads(refpath.read_text())
    for p, digest in reference["sha256"].items():
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source checksum mismatch: {p}")
    cfg = MoistureConfig()
    results, csvrows = [], []
    for case in reference["cases"]:
        n = case["case"]
        folder = args.runs / f"case{n}"
        summary = json.loads((folder / "summary.json").read_text())
        if summary["status"] != "complete" or summary["time_seconds"] < args.end - 1e-8:
            raise ValueError(f"Case {n} is not complete")
        with (folder / "history.csv").open() as stream:
            rows = [
                {k: float(v) for k, v in r.items() if v} for r in csv.DictReader(stream)
            ]
        t = np.array([r["time_hours"] * 3600 for r in rows])
        if np.any(np.diff(t) <= 0):
            raise ValueError("Nonmonotone history")
        td = np.array([[r[f"sensor_{i}_dbt_c"] for i in range(9)] for r in rows])
        q = np.array([[r[f"sensor_{i}_vapor_kg_kg"] for i in range(9)] for r in rows])
        tw = np.asarray(
            wet_bulb_c(
                jnp.asarray(td),
                jnp.asarray(q),
                101325.0,
                cfg.dry_air_gas_constant / cfg.water_vapor_gas_constant,
            )
        )
        obs_td, obs_tw = (
            np.array(case["experimental_dbt_c"]),
            np.array(case["experimental_wbt_c"]),
        )
        observations = np.array(
            [obs_td, obs_tw, enthalpy(obs_td, humidity(obs_td, obs_tw))]
        )
        predicted = np.array(
            [average(t, a, args.start, args.end) for a in (td, tw, enthalpy(td, q))]
        )
        if not np.all(np.isfinite(predicted)):
            raise ValueError("Nonfinite prediction; inspect saturation/temperature")
        qin = rows[-1]["inlet_vapor_mixing_ratio"]
        e = (
            101325.0
            * qin
            / (cfg.dry_air_gas_constant / cfg.water_vapor_gas_constant + qin)
        )
        rh = (
            100
            * e
            / float(
                saturation_vapor_pressure_water(
                    jnp.asarray(case["inlet_dbt_c"] + 273.15)
                )
            )
        )
        middle = (args.start + args.end) / 2
        shift = average(t, td, middle, args.end) - average(t, td, args.start, middle)
        result = {
            "case": n,
            "observations": observations.tolist(),
            "predicted": predicted.tolist(),
            "inlet_rh_percent": rh,
            "rmse": np.sqrt(np.mean((predicted - observations) ** 2, axis=1)).tolist(),
            "maximum_absolute_error": np.max(
                abs(predicted - observations), axis=1
            ).tolist(),
            "mean_dbt_bias_k": float(np.mean(predicted[0] - obs_td)),
            "maximum_dbt_half_window_shift_k": float(np.max(abs(shift))),
            "maximum_cfl": max(r["maximum_cfl"] for r in rows),
            "maximum_water_residual_kg": max(
                abs(r["parcel_mass_balance_error_kg"]) for r in rows
            ),
            "elapsed_advancement_seconds": summary["elapsed_seconds"],
            "history_sha256": hashlib.sha256(
                (folder / "history.csv").read_bytes()
            ).hexdigest(),
        }
        results.append(result)
        for i, name in enumerate(SENSORS):
            csvrows.append(
                [
                    n,
                    i + 1,
                    name,
                    *[
                        v
                        for j in range(3)
                        for v in (
                            observations[j, i],
                            predicted[j, i],
                            predicted[j, i] - observations[j, i],
                        )
                    ],
                ]
            )
    # Extract the native Figure 7 raster, retaining original CFD points and bands.
    import pymupdf

    pdf = pymupdf.open(ROOT / "sim.pdf")
    blocks = [b for b in pdf[6].get_text("dict")["blocks"] if b["type"] == 1]
    if len(blocks) != 1 or (blocks[0]["width"], blocks[0]["height"]) != (2120, 1835):
        raise ValueError("Unexpected PDF figure layout; recalibrate axes")
    original = out / ("figure7_original." + blocks[0]["ext"])
    original.write_bytes(blocks[0]["image"])
    artwork = np.asarray(Image.open(original))
    zoom, zoom_ax = plt.subplots(figsize=(10, 8))
    zoom_ax.imshow(artwork)
    zoom_ax.set_xlim(0, 710)
    zoom_ax.set_ylim(1220, 610)
    zoom_ax.axis("off")
    zoom_ax.scatter(
        [192.2], [993.6], s=350, facecolors="none", edgecolors="crimson", linewidths=2.5
    )
    zoom_ax.annotate(
        "Inconsistent published point\nExperimental x ≈ 26.9°C\nCFD y ≈ 28.1°C",
        xy=(192.2, 993.6),
        xytext=(345, 1060),
        color="crimson",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "crimson", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "crimson", "lw": 2},
    )
    zoom.suptitle("sim.pdf · Figure 7(b), case 2", fontsize=18, weight="bold")
    zoom.text(
        0.05,
        0.025,
        "Original experimental Table 2(b): case-2 DBT range = 28.4–30.6°C.\nThe plotted point falls outside that range; its sensor identity is not given.",
        fontsize=11,
    )
    for ext in ("png", "pdf"):
        zoom.savefig(
            out / f"figure7_inconsistent_point.{ext}", dpi=170, bbox_inches="tight"
        )
    plt.close(zoom)
    color = "#0077cc"
    fig, ax = plt.subplots(figsize=(15.5, 13.2))
    ax.imshow(artwork)
    ax.set_xlim(-30, 2490)
    ax.set_ylim(1980, -135)
    ax.axis("off")
    ax.text(
        10,
        -95,
        "Figure 7 · original paper + our inertial-parcel predictions",
        fontsize=16,
        weight="bold",
    )
    ax.text(
        10,
        -45,
        f"Brown: published CFD   |   Blue diamonds: our {args.start:g}–{args.end:g} s means   |   Rows: cases 1, 2, 3",
        fontsize=11,
    )
    overflow = []
    for row, result in enumerate(results):
        for col in range(3):
            lo, hi = LIMITS[col]
            xl, xr = XFRAMES[col]
            yt, yb = YFRAMES[row]
            x = np.array(result["observations"][col])
            y = np.array(result["predicted"][col])
            inside = (y >= lo) & (y <= hi) & (x >= lo) & (x <= hi)
            xp = xl + (x - lo) / (hi - lo) * (xr - xl)
            yp = yb - (y - lo) / (hi - lo) * (yb - yt)
            ax.scatter(
                xp[inside],
                yp[inside],
                marker="D",
                s=48,
                facecolors="none",
                edgecolors=color,
                linewidths=1.5,
                zorder=4,
            )
            for i in np.flatnonzero(~inside):
                edge = yt + 5 if y[i] > hi else yb - 5
                ax.scatter(
                    [xp[i]],
                    [edge],
                    marker="^" if y[i] > hi else "v",
                    s=65,
                    c=color,
                    zorder=5,
                )
                ax.text(
                    xp[i] + 8, edge + 20, f"{i + 1}", color=color, fontsize=8, zorder=5
                )
                overflow.append(
                    f"Case {row + 1}, {NAMES[col].split()[0]}, sensor {i + 1}: {y[i]:.2f}"
                )
    # The original panel-b marker has no published sensor identifier.
    ax.scatter(
        [192.2],
        [993.6],
        s=240,
        facecolors="none",
        edgecolors="crimson",
        linewidths=2,
        zorder=7,
    )
    ax.annotate(
        "Inconsistent experimental x ≈ 26.9°C\nOriginal table minimum: 28.4°C",
        xy=(192.2, 993.6),
        xytext=(285, 1080),
        fontsize=8,
        color="crimson",
        bbox={"facecolor": "white", "edgecolor": "crimson", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "crimson"},
        zorder=8,
    )
    ax.text(
        2150,
        85,
        "OUTSIDE ORIGINAL AXES\n(triangles mark direction)\n\n"
        + "\n".join(overflow or ["None"]),
        va="top",
        fontsize=9,
        color=color,
        linespacing=1.6,
    )
    ax.text(
        20,
        1910,
        "Experimental x-values: original Table 2(b), spatially paired; h reconstructed from DBT/WBT.\nOur solver: transient LES + inertial parcels; paper: steady RANS + DPM. No parameter fitting.",
        fontsize=10,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure7_overlay.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    # Expanded axes retain original plot interiors as background, showing every value.
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for row, result in enumerate(results):
        for col, ax in enumerate(axes[row]):
            lo, hi = LIMITS[col]
            xl, xr = XFRAMES[col]
            yt, yb = YFRAMES[row]
            ax.imshow(
                artwork[int(yt) + 2 : int(yb) - 1, int(xl) + 2 : int(xr) - 1],
                extent=(lo, hi, lo, hi),
                origin="upper",
                aspect="auto",
                zorder=0,
            )
            x = np.array(result["observations"][col])
            y = np.array(result["predicted"][col])
            upper = max(hi, float(y.max()) + 0.8)
            lower = min(lo, float(y.min()) - 0.8)
            ax.plot([lower, upper], [lower, upper], color="0.4", lw=0.8, zorder=1)
            ax.scatter(
                x,
                y,
                marker="D",
                facecolors="none",
                edgecolors=color,
                s=50,
                lw=1.5,
                zorder=3,
            )
            if row == 1 and col == 0:
                ax.scatter(
                    [26.888],
                    [28.109],
                    s=180,
                    facecolors="none",
                    edgecolors="crimson",
                    linewidths=1.5,
                    zorder=5,
                )
                ax.annotate(
                    "Published x ≈ 26.9°C\nTable minimum 28.4°C",
                    xy=(26.888, 28.109),
                    xytext=(28.2, 25.5),
                    fontsize=7,
                    color="crimson",
                    bbox={"facecolor": "white", "edgecolor": "crimson", "alpha": 0.95},
                    arrowprops={"arrowstyle": "->", "color": "crimson"},
                    zorder=6,
                )
            ax.set(
                xlim=(lo, hi),
                ylim=(lower, upper),
                xlabel="Experiment · " + NAMES[col],
                ylabel="Prediction · " + NAMES[col],
                title=f"Case {row + 1} · {NAMES[col].split()[0]}",
            )
    fig.suptitle(
        f"All predictions visible · {args.start:g}–{args.end:g} s averages\nBlue: our inertial parcels; brown/background: original Figure 7",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure7_expanded.{ext}", dpi=180)
    plt.close(fig)
    with (out / "sensors.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "sensor_id",
                "sensor",
                *[
                    f"{v}_{q}"
                    for v in ("dbt_c", "wbt_c", "enthalpy_kj_kg_dry_air")
                    for q in ("experiment", "prediction", "error")
                ],
            ]
        )
        writer.writerows(csvrows)
    report = {
        "window_seconds": [args.start, args.end],
        "enthalpy_formula": "h = 1.005 T_C + q (2500 + 1.859 T_C), kJ/kg dry air",
        "reference": reference,
        "results": results,
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
