#!/usr/bin/env python3
"""Render direct rigid-ALM rotor and wake slices as an animated GIF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "flow_slices",
        type=Path,
        help="flow_slices.npz written by the direct_rigid_alm runner",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def _symmetric_limit(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        raise ValueError("flow slices contain no finite values")
    return max(float(np.percentile(finite, 99.5)), 1.0e-5)


def _render(
    output: Path,
    *,
    metadata: dict,
    times: np.ndarray,
    rotor_planes: np.ndarray,
    hub_planes: np.ndarray,
    blade_positions_m: np.ndarray | None,
    fps: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the GIF")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    domain = metadata["domain"]
    turbine = metadata["turbine"]
    lx_m = float(domain["lx_m"])
    ly_m = float(domain["ly_m"])
    lz_m = float(domain["lz_m"])
    turbine_x = float(turbine["x_m"])
    turbine_y = float(turbine["y_m"])
    hub_z = float(turbine["hub_height_m"])
    hub_radius = float(turbine["hub_radius_m"])
    tip_radius = float(turbine["tip_radius_m"])
    omega = float(turbine["angular_velocity_rad_s"])
    initial_azimuth = np.deg2rad(
        float(turbine["initial_azimuth_degrees"])
    )
    precone_projection = np.cos(
        np.deg2rad(float(turbine["precone_degrees"]))
    )
    blade_count = int(turbine["blade_count"])
    model = str(turbine.get("model", "openfast_rigid_actuator_line"))
    title_model = (
        "Modal aeroelastic OpenFAST actuator line"
        if "aeroelastic" in model
        else "Rigid OpenFAST actuator line"
    )

    rotor_limit = _symmetric_limit(rotor_planes)
    hub_limit = _symmetric_limit(hub_planes)
    figure, (rotor_axis, wake_axis) = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.4),
        constrained_layout=True,
    )
    rotor_image = rotor_axis.imshow(
        rotor_planes[0],
        origin="lower",
        extent=(0.0, ly_m, 0.0, lz_m),
        cmap="coolwarm",
        vmin=-rotor_limit,
        vmax=rotor_limit,
        interpolation="bilinear",
        aspect="equal",
    )
    wake_image = wake_axis.imshow(
        hub_planes[0],
        origin="lower",
        extent=(0.0, lx_m, 0.0, ly_m),
        cmap="coolwarm",
        vmin=-hub_limit,
        vmax=hub_limit,
        interpolation="bilinear",
        aspect="equal",
    )

    crop_factor = 1.35
    rotor_axis.set(
        xlim=(
            turbine_y - crop_factor * tip_radius,
            turbine_y + crop_factor * tip_radius,
        ),
        ylim=(
            max(0.0, hub_z - crop_factor * tip_radius),
            hub_z + crop_factor * tip_radius,
        ),
        xlabel="y [m]",
        ylabel="z [m]",
        title="Rotor plane",
    )
    wake_axis.set(
        xlim=(
            max(0.0, turbine_x - 0.8 * tip_radius),
            min(lx_m, turbine_x + 3.2 * tip_radius),
        ),
        ylim=(
            turbine_y - crop_factor * tip_radius,
            turbine_y + crop_factor * tip_radius,
        ),
        xlabel="x [m]",
        ylabel="y [m]",
        title="Hub-height wake plane",
    )
    rotor_axis.add_patch(
        Circle(
            (turbine_y, hub_z),
            tip_radius * precone_projection,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.75,
        )
    )
    rotor_axis.plot(
        turbine_y,
        hub_z,
        marker="o",
        color="black",
        markersize=4,
    )
    blades = [
        rotor_axis.plot(
            (),
            (),
            color="black",
            linewidth=2.0,
            solid_capstyle="round",
        )[0]
        for _ in range(blade_count)
    ]
    wake_axis.plot(
        (turbine_x, turbine_x),
        (turbine_y - tip_radius, turbine_y + tip_radius),
        color="black",
        linewidth=2.2,
        solid_capstyle="round",
    )
    wake_axis.annotate(
        "mean flow",
        xy=(turbine_x + 2.6 * tip_radius, turbine_y),
        xytext=(turbine_x + 1.5 * tip_radius, turbine_y),
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "black"},
        ha="center",
        va="bottom",
        color="black",
    )
    rotor_colorbar = figure.colorbar(
        rotor_image,
        ax=rotor_axis,
        pad=0.02,
        shrink=0.84,
    )
    rotor_colorbar.set_label(r"$u-u_0$ [m s$^{-1}$]")
    wake_colorbar = figure.colorbar(
        wake_image,
        ax=wake_axis,
        pad=0.02,
        shrink=0.84,
    )
    wake_colorbar.set_label(r"$u-u_0$ [m s$^{-1}$]")
    title = figure.suptitle("")

    def update(frame: int) -> None:
        rotor_image.set_data(rotor_planes[frame])
        wake_image.set_data(hub_planes[frame])
        time_seconds = float(times[frame])
        for blade_index, blade in enumerate(blades):
            if blade_positions_m is None:
                theta = (
                    initial_azimuth
                    + omega * time_seconds
                    + 2.0 * np.pi * blade_index / blade_count
                )
                radii = np.asarray((hub_radius, tip_radius))
                blade.set_data(
                    turbine_y
                    + radii * precone_projection * np.sin(theta),
                    hub_z
                    + radii * precone_projection * np.cos(theta),
                )
            else:
                positions = blade_positions_m[frame, blade_index]
                blade.set_data(positions[:, 1], positions[:, 2])
        title.set_text(
            f"{title_model} — streamwise velocity perturbation\n"
            f"t = {time_seconds:.2f} s"
        )

    update(0)
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-filter_complex",
        (
            "[0:v]split[a][b];"
            "[a]palettegen=max_colors=192:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
        ),
        "-loop",
        "0",
        "-f",
        "gif",
        str(temporary),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for frame in range(len(times)):
            update(frame)
            figure.canvas.draw()
            process.stdin.write(figure.canvas.buffer_rgba())
        process.stdin.close()
        process.stdin = None
        _, error = process.communicate()
    except BaseException:
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    if process.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(error.decode("utf-8", errors="replace"))
    os.replace(temporary, output)


def main() -> None:
    args = _arguments()
    if args.fps <= 0.0:
        raise SystemExit("--fps must be positive")
    source = args.flow_slices.resolve()
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        times = np.asarray(data["times_seconds"])
        rotor_planes = np.asarray(data["rotor_plane_delta_u_m_s"])
        hub_planes = np.asarray(data["hub_plane_delta_u_m_s"])
        blade_positions_m = (
            np.asarray(data["blade_positions_m"])
            if "blade_positions_m" in data
            else None
        )
    if not (
        len(times) == len(rotor_planes) == len(hub_planes)
        and (
            blade_positions_m is None
            or len(blade_positions_m) == len(times)
        )
        and len(times) > 1
    ):
        raise ValueError("flow slices must contain at least two aligned frames")
    output = (
        source.with_name("flow_field.gif")
        if args.output is None
        else args.output.resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _render(
        output,
        metadata=metadata,
        times=times,
        rotor_planes=rotor_planes,
        hub_planes=hub_planes,
        blade_positions_m=blade_positions_m,
        fps=args.fps,
    )
    print(
        f"wrote {output} ({len(times)} frames, "
        f"{times[0]:.2f}–{times[-1]:.2f} s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
