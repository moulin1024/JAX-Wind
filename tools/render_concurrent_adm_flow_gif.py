#!/usr/bin/env python3
"""Render sampled concurrent-ADM streamwise-velocity fields as a GIF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the turbine-domain hub-height u field from "
            "main_velocity_*.npz snapshots."
        )
    )
    parser.add_argument(
        "fields_directory",
        type=Path,
        help="Directory containing main_velocity_*.npz snapshots",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--height-m", type=float)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    return parser.parse_args()


def _read_metadata(snapshot: Path) -> dict[str, Any]:
    with np.load(snapshot, allow_pickle=False) as data:
        return json.loads(str(data["metadata"]))


def _selected_snapshots(
    fields_directory: Path,
    *,
    stride: int,
    max_frames: int | None,
) -> list[Path]:
    snapshots = sorted(fields_directory.glob("main_velocity_*.npz"))
    snapshots = snapshots[::stride]
    if max_frames is not None:
        snapshots = snapshots[:max_frames]
    if not snapshots:
        raise FileNotFoundError(
            f"no main_velocity_*.npz snapshots in {fields_directory}"
        )
    return snapshots


def _load_hub_slices(
    snapshots: list[Path],
    *,
    height_m: float,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    first_metadata = _read_metadata(snapshots[0])
    grid = first_metadata["grid"]
    nz = int(grid["nz"])
    dz_m = float(grid["lz_m"]) / nz
    z_index = int(np.argmin(np.abs((np.arange(nz) + 0.5) * dz_m - height_m)))
    slices = []
    wake_times = []
    for index, snapshot in enumerate(snapshots, start=1):
        with np.load(snapshot, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            velocity = np.asarray(data["u_m_s"][z_index], dtype=np.float32)
        if metadata["grid"] != grid:
            raise ValueError(f"inconsistent grid metadata in {snapshot}")
        slices.append(velocity)
        wake_times.append(float(metadata["wake_time_seconds"]))
        if index % 50 == 0 or index == len(snapshots):
            print(f"loaded frame {index}/{len(snapshots)}", flush=True)
    return (
        np.stack(slices),
        np.asarray(wake_times),
        z_index,
        first_metadata,
    )


def _load_case(fields_directory: Path) -> dict[str, Any]:
    resolved = fields_directory.parent / "resolved_config.toml"
    if resolved.is_file():
        with resolved.open("rb") as stream:
            return tomllib.load(stream)

    legacy = fields_directory.parent / "resolved_config.json"
    if legacy.is_file():
        return json.loads(legacy.read_text())

    raise FileNotFoundError(
        f"{resolved} is required for turbine and fringe annotations"
    )


def _render(
    output: Path,
    fields: np.ndarray,
    wake_times: np.ndarray,
    *,
    z_m: float,
    metadata: dict[str, Any],
    case: dict[str, Any],
    fps: float,
    vmin: float | None,
    vmax: float | None,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the GIF")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if vmin is None or vmax is None:
        robust_min, robust_max = np.percentile(fields, (0.5, 99.5))
        vmin = float(robust_min) if vmin is None else vmin
        vmax = float(robust_max) if vmax is None else vmax
    if not np.isfinite(vmin + vmax) or vmax <= vmin:
        raise ValueError("the GIF color limits must be finite and increasing")

    grid = metadata["grid"]
    lx_m = float(grid["lx_m"])
    ly_m = float(grid["ly_m"])
    turbine = case["turbine"]
    turbine_x, turbine_y = map(float, turbine["location_m"])
    diameter = float(turbine["diameter_m"])
    fringe_x = float(case["fringe"]["start_x_m"])

    figure, axis = plt.subplots(figsize=(11.2, 6.2), constrained_layout=True)
    image = axis.imshow(
        fields[0],
        origin="lower",
        extent=(0.0, lx_m, 0.0, ly_m),
        aspect="equal",
        interpolation="bilinear",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    axis.axvspan(
        fringe_x,
        lx_m,
        color="white",
        alpha=0.10,
        linewidth=0.0,
    )
    axis.axvline(
        fringe_x,
        color="white",
        linestyle="--",
        linewidth=1.3,
        alpha=0.9,
    )
    axis.plot(
        (turbine_x, turbine_x),
        (turbine_y - 0.5 * diameter, turbine_y + 0.5 * diameter),
        color="white",
        linewidth=3.2,
        solid_capstyle="round",
    )
    axis.annotate(
        "ADM rotor",
        xy=(turbine_x, turbine_y + 0.52 * diameter),
        xytext=(12, 12),
        textcoords="offset points",
        color="white",
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": "white", "lw": 1.0},
        bbox={"facecolor": "black", "alpha": 0.42, "edgecolor": "none"},
    )
    axis.text(
        fringe_x + 0.5 * (lx_m - fringe_x),
        0.97 * ly_m,
        "precursor fringe",
        color="white",
        ha="center",
        va="top",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.42, "edgecolor": "none"},
    )
    axis.text(
        0.02 * lx_m,
        0.06 * ly_m,
        "mean flow →",
        color="white",
        fontsize=10,
        bbox={"facecolor": "black", "alpha": 0.42, "edgecolor": "none"},
    )
    axis.set(
        xlabel="x [m]",
        ylabel="y [m]",
        xlim=(0.0, lx_m),
        ylim=(0.0, ly_m),
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02, shrink=0.90)
    colorbar.set_label(r"streamwise velocity $u$ [m s$^{-1}$]")
    title = axis.set_title("")

    def update(frame: int) -> None:
        image.set_data(fields[frame])
        title.set_text(
            "DTU 10 MW concurrent precursor — "
            f"hub-height plane z = {z_m:.0f} m\n"
            f"wake time = {wake_times[frame] / 60.0:.1f} min"
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
        for frame in range(len(fields)):
            update(frame)
            figure.canvas.draw()
            process.stdin.write(figure.canvas.buffer_rgba())
            if (frame + 1) % 50 == 0 or frame + 1 == len(fields):
                print(f"rendered frame {frame + 1}/{len(fields)}", flush=True)
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
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    if args.fps <= 0.0:
        raise SystemExit("--fps must be positive")

    fields_directory = args.fields_directory.resolve()
    case = _load_case(fields_directory)
    height_m = (
        float(case["turbine"]["hub_height_m"])
        if args.height_m is None
        else args.height_m
    )
    snapshots = _selected_snapshots(
        fields_directory,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    fields, wake_times, z_index, metadata = _load_hub_slices(
        snapshots,
        height_m=height_m,
    )
    dz_m = float(metadata["grid"]["lz_m"]) / int(metadata["grid"]["nz"])
    sampled_z_m = (z_index + 0.5) * dz_m
    output = (
        fields_directory.parent / "main_u_hub_height.gif"
        if args.output is None
        else args.output.resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _render(
        output,
        fields,
        wake_times,
        z_m=sampled_z_m,
        metadata=metadata,
        case=case,
        fps=args.fps,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    print(
        f"wrote {output} ({len(snapshots)} frames, "
        f"{wake_times[0]:.1f}–{wake_times[-1]:.1f} s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
