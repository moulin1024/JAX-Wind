#!/usr/bin/env python3
"""Compare neutral log-law mean profiles from declarative run outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from plot_result import _filtered_log_denominator, _grid


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.result_dirs):
        raise ValueError("--labels must contain one label per result directory")

    configs = [
        json.loads((path / "resolved_config.json").read_text(encoding="utf-8"))
        for path in args.result_dirs
    ]
    profiles = [
        np.genfromtxt(path / "profiles.csv", delimiter=",", names=True)
        for path in args.result_dirs
    ]
    grid = _grid(configs[0])
    z = np.asarray(profiles[0]["z_m"], dtype=float)
    height = float(configs[0]["grid"]["extent"][2])
    ustar = float(configs[0]["momentum"]["friction_velocity"])
    roughness = float(configs[0]["momentum"]["roughness_length"])
    von_karman = float(configs[0]["momentum"].get("von_karman", 0.4))
    normalized_height = z / height
    lower = np.asarray(grid.z_faces[:-1], dtype=float) - grid.z_faces[0]
    upper = np.asarray(grid.z_faces[1:], dtype=float) - grid.z_faces[0]
    denominator = _filtered_log_denominator(lower, upper, roughness)
    target = ustar / von_karman * denominator
    fit = (normalized_height >= 0.05) & (normalized_height <= 0.3)

    for config, profile in zip(configs[1:], profiles[1:], strict=True):
        if config["grid"] != configs[0]["grid"]:
            raise ValueError("all comparisons must use the same grid")
        if not np.allclose(profile["z_m"], z, rtol=2.0e-6, atol=2.0e-6):
            raise ValueError("comparison profile heights do not match")

    labels = args.labels or [path.name for path in args.result_dirs]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 6.2), constrained_layout=True)
    styles = ("-", "--", "-.", ":")

    axes[0].plot(
        target / ustar,
        normalized_height,
        color="black",
        ls=":",
        lw=2.0,
        label="FV-filtered log law",
    )
    for index, (profile, label) in enumerate(zip(profiles, labels, strict=True)):
        error = np.asarray(profile["mean_u_m_s"], dtype=float) - target
        rmse = float(np.sqrt(np.mean(error[fit] ** 2)))
        line = axes[0].plot(
            profile["mean_u_m_s"] / ustar,
            normalized_height,
            ls=styles[index % len(styles)],
            lw=2.1,
            label=f"{label} (RMSE={rmse:.3f} m/s)",
        )[0]
        axes[1].plot(
            error / ustar,
            normalized_height,
            color=line.get_color(),
            ls=styles[index % len(styles)],
            lw=2.1,
            label=label,
        )

    lower_limit = 0.8 * normalized_height[0]
    for axis in axes:
        axis.set_yscale("log")
        axis.set_ylim(lower_limit, 0.5)
        axis.set_ylabel(r"$z/H$")
        axis.grid(alpha=0.25, which="both")
        axis.axhspan(0.05, 0.3, color="0.85", alpha=0.35)
    axes[0].set_xlabel(r"$\langle u\rangle/u_*$")
    axes[0].set_title("Mean velocity")
    axes[0].legend(fontsize=8)
    axes[1].axvline(0.0, color="black", ls=":", lw=1.2)
    axes[1].set_xlabel(r"$(\langle u\rangle-U_{log,FV})/u_*$")
    axes[1].set_title("Departure from log law")
    axes[1].legend(fontsize=8)
    figure.suptitle(
        f"Pressure-driven neutral log layer, {grid.shape[2]}³ z-stretched, "
        r"$t^*=20$"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
