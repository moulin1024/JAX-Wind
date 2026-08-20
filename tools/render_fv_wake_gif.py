#!/usr/bin/env python3
"""Render FV hub-height main-flow frames as an animated wake GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    arguments = parser.parse_args()
    if arguments.fps <= 0:
        raise ValueError("--fps must be positive")

    from applications.fv_abl.workflow import (
        _build_turbine_definition,
        load_workflow,
    )
    from jaxwind.domain import ScaleSystem

    workflow = load_workflow(arguments.config)
    source = workflow.options.output_directory / "main_flow_frames.npz"
    if not source.exists():
        raise FileNotFoundError(f"missing FV main frames: {source}")
    with np.load(source) as archive:
        fields = np.asarray(archive["u_hub_yx"])
        times = np.asarray(archive["time_seconds"])
        x = np.asarray(archive["x_m"])
        y = np.asarray(archive["y_m"])
    if fields.ndim != 3 or fields.shape[0] == 0:
        raise ValueError("u_hub_yx must contain at least one (y, x) frame")

    turbine = _build_turbine_definition(workflow)
    disk = (
        None
        if turbine is None
        else turbine.to_actuator_disk(scales=ScaleSystem(1.0, 1.0))
    )
    lower, upper = np.nanpercentile(fields, (1.0, 99.0))
    figure, axis = plt.subplots(figsize=(10.5, 3.8), constrained_layout=True)
    image = axis.imshow(
        fields[0],
        origin="lower",
        extent=(
            float(x[0] - 0.5 * (x[1] - x[0])),
            float(x[-1] + 0.5 * (x[-1] - x[-2])),
            float(y[0] - 0.5 * (y[1] - y[0])),
            float(y[-1] + 0.5 * (y[-1] - y[-2])),
        ),
        aspect="auto",
        cmap="turbo",
        vmin=float(lower),
        vmax=float(upper),
        interpolation="bilinear",
    )
    if disk is not None:
        axis.plot(
            [disk.x, disk.x],
            [disk.y - disk.tip_radius, disk.y + disk.tip_radius],
            color="white",
            linewidth=2.2,
        )
        axis.plot(
            disk.x,
            disk.y,
            marker="+",
            color="black",
            markersize=7,
            markeredgewidth=1.5,
        )
    title = axis.set_title("")
    axis.set(
        xlabel="x [m]",
        ylabel="y [m]",
        title="HITSZ R9 FV-GMG hub-height streamwise velocity",
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("u [m s$^{-1}$]")

    def update(index: int):
        image.set_data(fields[index])
        title.set_text(
            "HITSZ R9 FV-GMG hub-height streamwise velocity "
            f"(t={times[index]:.2f} s)"
        )
        return image, title

    output = (
        arguments.output
        if arguments.output is not None
        else workflow.options.output_directory / "main_flow.gif"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        figure,
        update,
        frames=fields.shape[0],
        interval=1000.0 / arguments.fps,
        blit=False,
    )
    movie.save(output, writer=animation.PillowWriter(fps=arguments.fps), dpi=120)
    plt.close(figure)
    print(f"wrote {output} ({fields.shape[0]} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
