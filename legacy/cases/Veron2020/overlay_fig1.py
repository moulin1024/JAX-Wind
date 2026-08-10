#!/usr/bin/env python3
"""Overlay the JAXWIND single-drop result on Veron (2020), figure 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from run_single_drop import simulate


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / "doc" / "reference" / "fig1.png"
DEFAULT_OUTPUT = ROOT / "doc" / "reference" / "fig1_jax_overlay.png"

# Pixel bounds of the shared logarithmic-time plotting rectangle in the
# 1288-by-854 reference raster: left, right, top, bottom.
REFERENCE_AXES = (235.0, 957.0, 58.0, 564.0)


def overlay(reference: Path, output: Path, *, dt: float) -> None:
    background = mpimg.imread(reference)
    height, width = background.shape[:2]
    left, right, top, bottom = REFERENCE_AXES
    fresh = simulate(120.0, dt)
    seawater_early = simulate(
        2.0,
        dt,
        salinity_mass_fraction=0.035,
        liquid_density=1025.0,
    )
    seawater_late = simulate(
        5000.0,
        max(dt, 0.01),
        salinity_mass_fraction=0.035,
        liquid_density=1025.0,
    )

    def merge_at(
        early: dict[str, np.ndarray],
        late: dict[str, np.ndarray],
        cutoff: float,
    ) -> dict[str, np.ndarray]:
        early_mask = early["time"] <= cutoff
        late_mask = late["time"] > cutoff
        return {
            name: np.concatenate((early[name][early_mask], late[name][late_mask]))
            for name in early
        }

    seawater = merge_at(seawater_early, seawater_late, 1.0)

    def logarithmic_time(data: dict[str, np.ndarray]) -> np.ndarray:
        time = np.asarray(data["time"], dtype=float).copy()
        time[0] = 1.0e-5
        return time

    fresh_time = logarithmic_time(fresh)
    seawater_time = logarithmic_time(seawater)

    dpi = 100
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    background_axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    background_axes.imshow(background)
    background_axes.axis("off")

    axes_rect = (
        left / width,
        (height - bottom) / height,
        (right - left) / width,
        (bottom - top) / height,
    )
    temperature_axes = figure.add_axes(
        axes_rect, facecolor="none", frameon=False
    )
    radius_axes = figure.add_axes(axes_rect, facecolor="none", frameon=False)
    velocity_axes = figure.add_axes(
        axes_rect, facecolor="none", frameon=False
    )
    for axis, limits in (
        (temperature_axes, (14.0, 21.0)),
        (radius_axes, (40.0, 110.0)),
        (velocity_axes, (-2.0, 16.0)),
    ):
        axis.set_xscale("log")
        axis.set_xlim(1.0e-5, 1.0e5)
        axis.set_ylim(*limits)
        axis.axis("off")

    halo = [
        path_effects.Stroke(linewidth=4.0, foreground="white"),
        path_effects.Normal(),
    ]
    line_style = dict(linewidth=2.0, zorder=10)
    fresh_temperature = temperature_axes.plot(
        fresh_time,
        fresh["temperature"] - 273.15,
        color="#0057ff",
        linestyle=(0, (5, 3)),
        **line_style,
    )[0]
    fresh_radius = radius_axes.plot(
        fresh_time,
        1.0e6 * fresh["radius"],
        color="#e00000",
        linestyle=(0, (5, 3)),
        **line_style,
    )[0]
    fresh_velocity = velocity_axes.plot(
        fresh_time,
        fresh["u"],
        color="#111111",
        linestyle=(0, (5, 3)),
        **line_style,
    )[0]
    sea_temperature = temperature_axes.plot(
        seawater_time,
        seawater["temperature"] - 273.15,
        color="#0057ff",
        linestyle=(0, (7, 2, 1.5, 2)),
        **line_style,
    )[0]
    sea_radius = radius_axes.plot(
        seawater_time,
        1.0e6 * seawater["radius"],
        color="#e00000",
        linestyle=(0, (7, 2, 1.5, 2)),
        **line_style,
    )[0]
    sea_velocity = velocity_axes.plot(
        seawater_time,
        seawater["u"],
        color="#111111",
        linestyle=(0, (7, 2, 1.5, 2)),
        **line_style,
    )[0]
    for line in (
        fresh_temperature,
        fresh_radius,
        fresh_velocity,
        sea_temperature,
        sea_radius,
        sea_velocity,
    ):
        line.set_path_effects(halo)

    temperature_axes.text(
        2.0e3,
        19.75,
        "JAX fresh: dashed\nJAX sea: dash-dot",
        color="#111111",
        fontsize=13,
        ha="right",
        va="top",
        path_effects=[
            path_effects.Stroke(linewidth=3.0, foreground="white"),
            path_effects.Normal(),
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, transparent=False)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()
    overlay(args.reference, args.output, dt=args.dt)


if __name__ == "__main__":
    main()
