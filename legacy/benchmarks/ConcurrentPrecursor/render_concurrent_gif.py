#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "jax"))

from run_single import RUN_DEFAULTS, load_config_file, params_from_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render matched-time precursor/main velocity slices from the "
            "JAX-native concurrent solver."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warmup-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="adjoint_concurrent_velocity.gif")
    parser.add_argument("--coordinator-address", default="127.0.0.1:12700")
    parser.add_argument("--num-processes", type=int)
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--local-device-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--height", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def rank_and_size(args: argparse.Namespace) -> tuple[int, int]:
    rank = args.process_id
    size = args.num_processes
    if rank is None:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    if size is None:
        size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    return rank, size


def local_horizontal_slice(state, z_index: int, role: int) -> np.ndarray | None:
    """Copy one 2-D slice only on the process that owns `(role, z_index)`."""

    shard = state.u.addressable_shards[0]
    role_slice = shard.index[0]
    z_slice = shard.index[3]
    role_start = 0 if role_slice.start is None else role_slice.start
    z_start = 0 if z_slice.start is None else z_slice.start
    z_stop = state.u.shape[3] if z_slice.stop is None else z_slice.stop
    if role_start != role or not z_start <= z_index < z_stop:
        return None
    local_z = z_index - z_start
    return np.asarray(shard.data[0, :, :, local_z])


def render_gif(
    output: Path,
    precursor: np.ndarray,
    main: np.ndarray,
    times: np.ndarray,
    params,
    height: float,
    fps: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle

    combined = np.concatenate((precursor.ravel(), main.ravel()))
    vmin, vmax = np.percentile(combined, (0.5, 99.5))
    if not np.isfinite(vmin + vmax) or vmax <= vmin:
        vmin = float(np.nanmin(combined))
        vmax = float(np.nanmax(combined))

    lx = params.lx * params.z_i
    ly = params.ly * params.z_i
    extent = (0.0, lx, 0.0, ly)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.4, 7.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    images = []
    labels = (
        "Precursor — no actuator disc",
        "Main simulation — actuator disc",
    )
    for axis, field, label in zip(axes, (precursor, main), labels, strict=True):
        image = axis.imshow(
            field[0].T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
        images.append(image)
        axis.add_patch(
            Rectangle(
                (params.fringe_start_x, 0.0),
                lx - params.fringe_start_x,
                ly,
                facecolor="none",
                edgecolor="white",
                linewidth=1.5,
                linestyle="--",
                hatch="///",
                alpha=0.8,
            )
        )
        axis.text(
            0.5 * (params.fringe_start_x + lx),
            0.97 * ly,
            "fringe zone",
            color="white",
            ha="center",
            va="top",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )
        axis.text(
            0.012 * lx,
            0.95 * ly,
            label,
            color="white",
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
        )
        axis.set_ylabel("y [m]")
    disk = axes[1].plot(
        [params.actuator_disk_x, params.actuator_disk_x],
        [
            params.actuator_disk_y - 0.5 * params.actuator_disk_diameter,
            params.actuator_disk_y + 0.5 * params.actuator_disk_diameter,
        ],
        color="white",
        linewidth=3.0,
        solid_capstyle="round",
        label="actuator disc",
    )[0]
    axes[1].text(
        params.actuator_disk_x + 0.15,
        params.actuator_disk_y + 0.6,
        "actuator disc",
        color="white",
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
    )
    axes[1].set_xlabel("x [m]")
    colorbar = fig.colorbar(images[0], ax=axes, location="right", shrink=0.88, pad=0.025)
    colorbar.set_label(r"streamwise velocity $u$ [m s$^{-1}$]")
    title = fig.suptitle(
        f"Matched time: t = {times[0]:.3f} s   |   z = {height:.3f} m",
        fontsize=12,
    )

    def update(frame: int):
        images[0].set_data(precursor[frame].T)
        images[1].set_data(main[frame].T)
        title.set_text(
            f"Matched time: t = {times[frame]:.3f} s   |   z = {height:.3f} m"
        )
        return images[0], images[1], title, disk

    animation = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=1000 / fps,
        blit=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rank, size = rank_and_size(args)
    if size < 4 or size % 2:
        raise SystemExit("Concurrent visualization requires an even rank count >= 4")
    if args.steps <= 0 or args.frames <= 0 or args.steps % args.frames:
        raise SystemExit("steps must be positive and divisible by frames")
    if args.height <= 0.0 or args.fps <= 0:
        raise SystemExit("height and fps must be positive")
    chunk_steps = args.steps // args.frames

    settings = dict(RUN_DEFAULTS)
    settings.update(load_config_file(args.config))
    if settings["precision"] == "float64" or settings["sgs_precision"] == "float64":
        from jax import config as jax_config

        jax_config.update("jax_enable_x64", True)

    import jax

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=size,
        process_id=rank,
        local_device_ids=[args.local_device_id],
    )
    import jax.numpy as jnp
    from jax.experimental import multihost_utils

    from wireles_jax.adjoint_sharded import (
        duplicate_state_for_adjoint,
        make_adjoint_chunk_step,
        make_empty_fringe_chunk,
        make_exchange_precursor_chunk,
    )
    from wireles_jax.checkpoint_sharded import load_sharded_checkpoint
    from wireles_jax.sharding import (
        make_adjoint_distributed_mesh,
        make_distributed_mesh,
    )
    from wireles_jax.timestep_sharded import make_sharded_operators

    configured = params_from_settings(settings, jnp)
    warmup_params = replace(
        configured,
        nsteps=0,
        actuator_disk_enabled=False,
        cold_source_enabled=False,
        fringe_enabled=False,
        horizontal_homogeneous=True,
        buoyancy_reference="plane_mean",
        sharded_pressure_solver="transpose",
    )
    warm_mesh = make_distributed_mesh(size)
    warm_state = load_sharded_checkpoint(
        args.warmup_checkpoint,
        warmup_params,
        warm_mesh,
        rank=rank,
    )
    params = replace(
        configured,
        nsteps=args.steps,
        horizontal_homogeneous=False,
        buoyancy_reference="ambient",
        sharded_pressure_solver="transpose",
    )
    height_internal = args.height / params.z_i
    z_index = int(round(height_internal / params.dz - 0.5))
    if not 0 <= z_index < params.nz:
        raise SystemExit(f"height={args.height:g} m is outside the domain")
    sampled_height = (z_index + 0.5) * params.dz * params.z_i

    mesh = make_adjoint_distributed_mesh(size)
    state = duplicate_state_for_adjoint(warm_state, mesh)
    ops = make_sharded_operators(params, mesh)
    empty = make_empty_fringe_chunk(params, mesh, chunk_steps)
    prime = jax.jit(
        make_adjoint_chunk_step(
            params,
            ops,
            mesh,
            chunk_steps=chunk_steps,
            advance_turbine=False,
        )
    )
    advance = jax.jit(
        make_adjoint_chunk_step(
            params,
            ops,
            mesh,
            chunk_steps=chunk_steps,
            advance_turbine=True,
        )
    )
    exchange = jax.jit(make_exchange_precursor_chunk(mesh))

    state, produced = prime(state, empty, ops.pressure, ops.pressure_spike)
    targets = exchange(produced)
    jax.block_until_ready(targets)

    precursor_frames: list[np.ndarray] = []
    main_frames: list[np.ndarray] = []
    for frame in range(args.frames):
        precursor_slice = local_horizontal_slice(state, z_index, role=0)
        if precursor_slice is not None:
            precursor_frames.append(precursor_slice)
        state, produced = advance(
            state, targets, ops.pressure, ops.pressure_spike
        )
        targets = exchange(produced)
        main_slice = local_horizontal_slice(state, z_index, role=1)
        if main_slice is not None:
            main_frames.append(main_slice)
        if rank == 0 and ((frame + 1) % 10 == 0 or frame + 1 == args.frames):
            print(f"[gif] sampled frame {frame + 1}/{args.frames}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if precursor_frames:
        np.savez_compressed(
            args.output_dir / "precursor_hub_slices.npz",
            u=np.stack(precursor_frames),
        )
    if main_frames:
        np.savez_compressed(
            args.output_dir / "main_hub_slices.npz",
            u=np.stack(main_frames),
        )
    multihost_utils.sync_global_devices("concurrent-gif-slices-written")
    jax.distributed.shutdown()

    if rank != 0:
        return
    with np.load(args.output_dir / "precursor_hub_slices.npz") as archive:
        precursor = np.asarray(archive["u"])
    with np.load(args.output_dir / "main_hub_slices.npz") as archive:
        main = np.asarray(archive["u"])
    times = (
        np.arange(1, args.frames + 1, dtype=np.float64)
        * chunk_steps
        * params.dt_physical
    )
    np.savez_compressed(
        args.output_dir / "matched_hub_slices.npz",
        precursor_u=precursor,
        main_u=main,
        elapsed_seconds=times,
        height_m=np.asarray(sampled_height),
        fringe_start_m=np.asarray(params.fringe_start_x),
    )
    output = args.output_dir / args.output_name
    render_gif(
        output,
        precursor,
        main,
        times,
        params,
        sampled_height,
        args.fps,
    )
    print(f"[gif] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
