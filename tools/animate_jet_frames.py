"""Render saved cryogenic centreplane snapshots to an H.264 MP4."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tomllib

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("fps must be positive")
    directory = args.directory.resolve()
    output = args.output or directory / "jet_flow_field.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    with (directory / "resolved_case.toml").open("rb") as stream:
        case = tomllib.load(stream)
    nozzle = case["physics"]["jet"]["position_m"]
    ambient = case["physics"]["ambient"]["temperature_k"]
    with np.load(directory / "flow_frames.npz") as saved:
        times = saved["time_seconds"].copy()
        steps = saved["step"].copy()
        temperature = saved["temperature_center_zx"].copy()
        velocity = saved["u_center_zx"].copy()
        x, y, z = (saved[name].copy() for name in ("x_m", "y_m", "z_m"))
        xf, zf = (saved[name].copy() for name in ("x_faces_m", "z_faces_m"))
    if temperature.shape != velocity.shape or temperature.shape != (len(times), len(z), len(x)):
        raise ValueError("snapshot dimensions do not match the mesh and times")
    if not np.isfinite(temperature).all() or not np.isfinite(velocity).all():
        raise ValueError("snapshots contain nonfinite fields")
    if not np.all(np.diff(times) > 0):
        raise ValueError("snapshot times must be strictly increasing")
    plane_y = y[np.argmin(abs(y - 0.5 * case["mesh"]["lengths_m"][1]))]
    cold = np.any(temperature < ambient - 1.0, axis=0)
    cold_x = x[np.any(cold, axis=0)]
    cold_z = z[np.any(cold, axis=1)]
    if len(cold_x):
        xmin = max(xf[0], math.floor((min(cold_x.min(), nozzle[0]) - .3) * 2) / 2)
        xmax = min(xf[-1], math.ceil((max(cold_x.max(), nozzle[0]) + .7) * 2) / 2)
        zmax = min(zf[-1], math.ceil((max(cold_z.max(), nozzle[2]) + .3) * 2) / 2)
    else:
        xmin, xmax, zmax = xf[0], xf[-1], zf[-1]
    limits = (xmin, xmax, zf[0], zmax)
    temperature_min = math.floor(float(temperature.min()) / 10) * 10
    tnorm = Normalize(temperature_min, ambient)
    umax = max(.1, math.ceil(float(np.abs(velocity).max()) * 2) / 2)
    unorm = TwoSlopeNorm(vmin=-umax, vcenter=0, vmax=umax)
    background, foreground, muted = "#101722", "#edf3fa", "#a9bacb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": foreground, "axes.labelcolor": foreground,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.edgecolor": muted, "savefig.facecolor": background})
    figure = plt.figure(figsize=(16, 9), dpi=120, facecolor=background)
    grid = figure.add_gridspec(2, 2, left=.06, right=.94, bottom=.10, top=.84,
                              hspace=.40, wspace=.30, height_ratios=(1, 1.6))
    overview = figure.add_subplot(grid[0, :])
    thermal = figure.add_subplot(grid[1, 0])
    flow = figure.add_subplot(grid[1, 1])
    axes = (overview, thermal, flow)
    extent = (xf[0], xf[-1], zf[0], zf[-1])
    images = [overview.imshow(temperature[0], origin="lower", extent=extent,
                               cmap="inferno_r", norm=tnorm, interpolation="nearest"),
              thermal.imshow(temperature[0], origin="lower", extent=extent,
                              cmap="inferno_r", norm=tnorm, interpolation="nearest"),
              flow.imshow(velocity[0], origin="lower", extent=extent,
                           cmap="RdBu_r", norm=unorm, interpolation="nearest")]
    for axis in axes:
        axis.set_facecolor(background)
        axis.set_aspect("equal")
        axis.set(xlabel="Streamwise distance x [m]", ylabel="Height z [m]")
        axis.plot(nozzle[0], nozzle[2], marker=">", markersize=6,
                  markerfacecolor="#55e2df", markeredgecolor="#101722", zorder=5)
    overview.set_title("Full tunnel domain · temperature", loc="left", pad=9, color=foreground)
    overview.add_patch(Rectangle((xmin, zf[0]), xmax-xmin, zmax-zf[0], fill=False,
                                edgecolor="#55e2df", linewidth=1.2, linestyle="--"))
    for axis in (thermal, flow):
        axis.set(xlim=limits[:2], ylim=limits[2:])
    thermal.set_title("Jet close-up · temperature", loc="left", pad=10, color=foreground)
    flow.set_title("Jet close-up · streamwise velocity", loc="left", pad=10, color=foreground)
    for axis, plotted, label in ((thermal, images[1], "Temperature [K]"),
                                 (flow, images[2], "u [m/s]")):
        bar = figure.colorbar(plotted, ax=axis, fraction=.04, pad=.025)
        bar.set_label(label)
        bar.ax.tick_params(colors=muted)
        bar.outline.set_edgecolor(muted)
    inlet_speed = case["physics"]["ambient"].get("streamwise_velocity_m_s", 0.0)
    airflow = f"{inlet_speed:g} m/s inflow" if inlet_speed else "still air"
    figure.text(.06, .946, f"HITSZ  |  Nitrogen jet in {airflow}", fontsize=23, weight="bold")
    cells_label = " × ".join(str(value) for value in case["mesh"]["cells"])
    figure.text(.06, .903,
                f"{cells_label} cells   ·   Low-Mach / GMG   ·   Vertical slice y = {plane_y:.3f} m",
                color=muted, fontsize=12)
    stamp = figure.text(.94, .947, "", ha="right", fontsize=17, weight="bold")
    figure.text(.06, .035,
                "Cyan marker: nozzle   ·   Dashed box: close-up   ·   Fixed colour scales   ·   Saved snapshots; no temporal interpolation",
                fontsize=10, color=muted)
    figure.text(.94, .066, f"{len(times)} snapshots  |  {args.fps} frames/s", ha="right", color=muted, fontsize=10)
    encoder = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams["animation.ffmpeg_path"] = encoder
    writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=4500,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                          metadata={"title": "HITSZ nitrogen jet: temperature and streamwise velocity"})
    with writer.saving(figure, str(output), dpi=120):
        for index, (time, step) in enumerate(zip(times, steps)):
            images[0].set_data(temperature[index])
            images[1].set_data(temperature[index])
            images[2].set_data(velocity[index])
            stamp.set_text(f"t = {time:.3f} s   |   step {step}")
            writer.grab_frame(facecolor=background)
            if index == len(times) - 1:
                figure.savefig(output.with_suffix(".png"), dpi=120, facecolor=background)
    plt.close(figure)
    subprocess.run([encoder, "-v", "error", "-i", str(output), "-f", "null", "-"], check=True)
    report = {"output": str(output), "frames": len(times), "fps": args.fps,
              "video_duration_seconds": len(times)/args.fps,
              "simulation_time_seconds": [float(times[0]), float(times[-1])],
              "dimensions": [1920, 1080], "slice_y_m": float(plane_y),
              "temperature_limits_k": [temperature_min, ambient],
              "streamwise_velocity_limits_m_s": [-umax, umax],
              "zoom_extent_m": list(limits), "decode_check": "passed"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
