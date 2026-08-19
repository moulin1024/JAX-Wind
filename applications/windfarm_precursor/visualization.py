"""Flow-frame sampling and rendering for offline precursor inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def evenly_spaced_frame_offsets(steps: int, frame_count: int) -> tuple[int, ...]:
    """Return exactly ``frame_count`` unique offsets including both endpoints."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("frame schedule steps must be a positive integer")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or not 1 <= frame_count <= steps + 1
    ):
        raise ValueError("frame count must lie in [1, steps + 1]")
    if frame_count == 1:
        return (steps,)
    offsets = np.rint(np.linspace(0, steps, frame_count)).astype(np.int64)
    if len(np.unique(offsets)) != frame_count:
        raise RuntimeError("evenly spaced frame schedule contains duplicates")
    return tuple(int(value) for value in offsets)


def capture_xz_velocity(
    state: Any,
    problem: Any,
    *,
    jax: Any,
    y_m: float | None = None,
) -> np.ndarray:
    """Gather one centre-y streamwise-velocity plane in physical SI units."""

    velocity = state.fields.velocity
    global_u = problem.solver.global_array(velocity.x.payload)
    physical_u = problem.scales.from_execution_velocity(global_u)
    if y_m is None:
        plane = physical_u[:, problem.physical_grid.ny // 2, :]
    else:
        position = y_m / problem.physical_grid.dy - 0.5
        lower = int(np.floor(position)) % problem.physical_grid.ny
        upper = (lower + 1) % problem.physical_grid.ny
        fraction = position - np.floor(position)
        plane = (1.0 - fraction) * physical_u[:, lower, :] + fraction * physical_u[:, upper, :]
    return np.asarray(jax.device_get(plane), dtype=np.float32)


def save_flow_frames(
    path: str | Path,
    frames: list[np.ndarray],
    elapsed_seconds: list[float],
) -> Path:
    if not frames or len(frames) != len(elapsed_seconds):
        raise ValueError("flow frames and elapsed times must be nonempty and aligned")
    target = Path(path)
    np.savez_compressed(
        target,
        u_xz_m_s=np.stack(frames),
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float64),
    )
    return target


def write_flow_gif(
    path: str | Path,
    frames: list[np.ndarray],
    elapsed_seconds: list[float],
    *,
    grid: Any,
    inlet_end_x_m: float,
    fps: int,
    turbine: Any | None = None,
    equal_physical_scale: bool = False,
) -> Path:
    """Render fixed-scale X--Z velocity with the legacy inlet annotated."""

    if not frames or len(frames) != len(elapsed_seconds):
        raise ValueError("flow frames and elapsed times must be nonempty and aligned")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("GIF frame rate must be a positive integer")
    values = np.stack(frames)
    vmin, vmax = (float(value) for value in np.percentile(values, (1.0, 99.0)))
    if not np.isfinite(vmin + vmax) or vmax <= vmin:
        raise ValueError("flow GIF requires a nonconstant finite velocity range")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    target = Path(path)
    figure_size = (12.8, 3.2) if equal_physical_scale else (10.8, 4.2)
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
    image = axis.imshow(
        values[0],
        origin="lower",
        extent=(0.0, grid.lx, 0.0, grid.lz),
        aspect="equal" if equal_physical_scale else "auto",
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )
    axis.axvspan(0.0, inlet_end_x_m, color="white", alpha=0.12)
    axis.axvline(
        inlet_end_x_m,
        color="white",
        linestyle="--",
        linewidth=1.2,
    )
    axis.text(
        0.5 * inlet_end_x_m,
        0.96 * grid.lz,
        "legacy precursor inlet",
        color="white",
        ha="center",
        va="top",
        bbox={"facecolor": "black", "alpha": 0.38, "edgecolor": "none"},
    )
    if turbine is not None:
        lower = turbine.hub_height_m - 0.5 * turbine.rotor_diameter_m
        upper = turbine.hub_height_m + 0.5 * turbine.rotor_diameter_m
        axis.plot(
            (turbine.x_m, turbine.x_m),
            (lower, upper),
            color="black",
            linewidth=3.0,
            solid_capstyle="round",
        )
        axis.scatter(
            (turbine.x_m,),
            (turbine.hub_height_m,),
            color="white",
            edgecolor="black",
            linewidth=1.0,
            s=34,
            zorder=4,
        )
        axis.text(
            turbine.x_m,
            upper + 0.025 * grid.lz,
            getattr(turbine, "model_name", "turbine"),
            color="black",
            ha="center",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    axis.set(xlabel="x [m]", ylabel="z [m]")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(r"streamwise velocity $u$ [m s$^{-1}$]")
    title = axis.set_title("")

    def update(index: int):
        image.set_data(values[index])
        title.set_text(
            "Legacy-inlet main domain, centre-y X–Z plane\n"
            f"elapsed time = {elapsed_seconds[index] / 60.0:.1f} min"
        )
        return image, title

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=1000.0 / fps,
        blit=False,
    )
    movie.save(target, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    return target


__all__ = [
    "capture_xz_velocity",
    "evenly_spaced_frame_offsets",
    "save_flow_frames",
    "write_flow_gif",
]
