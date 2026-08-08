#!/usr/bin/env python3
"""Plot MGM, LASD, and AMD pressure-driven profiles against the log law."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = {
    "MGM": ROOT / "outputs" / "pressure_driven_mgm_32x32x32_cpu_conservative_32padding",
    "LASD": ROOT / "outputs" / "pressure_driven_lasd_32x32x32_cpu_conservative_32padding",
    "AMD": ROOT / "outputs" / "pressure_driven_amd_32x32x32_cpu_conservative_32padding",
}
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "pressure_driven_mgm_lasd_amd_32x32x32_cpu_conservative_32padding_comparison.png"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULT_CASES.items():
        parser.add_argument(
            f"--{name.lower()}",
            type=Path,
            default=default,
            help=f"{name} output directory containing profiles.csv",
        )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _profile(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    with (directory / "profiles.csv").open(newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    return (
        np.asarray([float(row["z_m"]) for row in rows]),
        np.asarray([float(row["mean_u_m_s"]) for row in rows]),
    )


def _rmse(values: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((values[mask] - reference[mask]) ** 2)))


def main() -> int:
    args = _arguments()
    directories = {name: getattr(args, name.lower()) for name in DEFAULT_CASES}
    profiles = {name: _profile(path) for name, path in directories.items()}
    heights = profiles["MGM"][0]
    if any(not np.array_equal(z, heights) for z, _ in profiles.values()):
        raise ValueError("all profiles must use the same vertical grid")

    friction_velocity = 0.4
    roughness_length = 0.001
    von_karman = 0.4
    domain_height = 1000.0
    log_velocity = (
        friction_velocity
        / von_karman
        * np.log(heights / roughness_length)
    )
    masks = {
        "full": np.ones_like(heights, dtype=bool),
        "z_le_0.8H": heights <= 0.8 * domain_height,
        "z_le_0.5H": heights <= 0.5 * domain_height,
    }
    metrics = {
        name: {
            region: _rmse(velocity, log_velocity, mask)
            for region, mask in masks.items()
        }
        for name, (_z, velocity) in profiles.items()
    }

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
        }
    )
    figure, axis = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    styles = {
        "MGM": {"color": "#0072B2", "marker": "o"},
        "LASD": {"color": "#E69F00", "marker": "s"},
        "AMD": {"color": "#009E73", "marker": "^"},
    }
    height_plus = heights / roughness_length
    for name, (_z, velocity) in profiles.items():
        axis.plot(
            velocity / friction_velocity,
            height_plus,
            linewidth=2.0,
            markersize=4.6,
            marker=styles[name]["marker"],
            markevery=2,
            color=styles[name]["color"],
            label=f"{name}  (RMSE {metrics[name]['full']:.3f} m/s)",
        )
    axis.plot(
        log_velocity / friction_velocity,
        height_plus,
        color="#202020",
        linestyle="--",
        linewidth=2.2,
        label=r"Log law  $U^+=\ln(z/z_0)/\kappa$",
    )
    axis.set_yscale("log")
    axis.set_xlabel(r"Mean streamwise velocity  $U/u_*$")
    axis.set_ylabel(r"Normalized height  $z/z_0$")
    axis.set_title(
        r"Neutral ABL — conservative + 3/2 padding, $32^3$ CPU, 2.5 h"
    )
    axis.grid(True, which="major", color="#C7CCD1", linewidth=0.8, alpha=0.75)
    axis.grid(True, which="minor", color="#E2E5E8", linewidth=0.55, alpha=0.65)
    axis.legend(loc="lower right", frameon=True, framealpha=0.95)
    axis.text(
        0.025,
        0.975,
        r"$u_*=0.4$ m/s, $z_0=10^{-3}$ m, $\kappa=0.4$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="#343A40",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, facecolor="white")
    plt.close(figure)
    print(json.dumps({"output": str(args.output), "rmse_m_s": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
