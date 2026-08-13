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


def capture_xz_velocity(state: Any, problem: Any, *, jax: Any) -> np.ndarray:
    """Gather one centre-y streamwise-velocity plane in physical SI units."""

    velocity = state.fields.velocity
    global_u = problem.solver.global_array(velocity.x.payload)
    physical_u = problem.scales.from_execution_velocity(global_u)
    plane = physical_u[:, problem.physical_grid.ny // 2, :]
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
    fringe_start_x_m: float,
    fps: int,
) -> Path:
    """Render fixed-scale X--Z streamwise velocity with the fringe annotated."""

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
    figure, axis = plt.subplots(figsize=(10.8, 4.2), constrained_layout=True)
    image = axis.imshow(
        values[0],
        origin="lower",
        extent=(0.0, grid.lx, 0.0, grid.lz),
        aspect="auto",
        interpolation="bilinear",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    axis.axvspan(fringe_start_x_m, grid.lx, color="white", alpha=0.12)
    axis.axvline(
        fringe_start_x_m,
        color="white",
        linestyle="--",
        linewidth=1.2,
    )
    axis.text(
        0.5 * (fringe_start_x_m + grid.lx),
        0.96 * grid.lz,
        "offline precursor fringe",
        color="white",
        ha="center",
        va="top",
        bbox={"facecolor": "black", "alpha": 0.38, "edgecolor": "none"},
    )
    axis.set(xlabel="x [m]", ylabel="z [m]")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(r"streamwise velocity $u$ [m s$^{-1}$]")
    title = axis.set_title("")

    def update(index: int):
        image.set_data(values[index])
        title.set_text(
            "Fringe-enforced main domain, centre-y X–Z plane\n"
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
