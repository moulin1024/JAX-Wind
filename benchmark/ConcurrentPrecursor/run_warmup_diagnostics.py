#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "jax"))

from run_single import RUN_DEFAULTS, load_config_file, params_from_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume a distributed warm-up and create ABL profile "
            "and flow-slice diagnostics."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional warm-up checkpoint; omit to start from the configured initial field.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--flow-height", type=float, default=1.5)
    parser.add_argument("--coordinator-address", default="127.0.0.1:12680")
    parser.add_argument("--num-processes", type=int)
    parser.add_argument("--process-id", type=int)
    parser.add_argument(
        "--local-device-id",
        type=int,
        help=(
            "Local accelerator index. By default use the MPI or Slurm local "
            "rank, falling back to device 0 for a single process."
        ),
    )
    parser.add_argument("--copy-to", type=Path)
    parser.add_argument(
        "--liquid-nitrogen-nozzle",
        action="store_true",
        help=(
            "Enable the equivalent far-field LN2 hub nozzle: a localized "
            "streamwise momentum source and cooling-power sink with ambient "
            "thermal buoyancy."
        ),
    )
    parser.add_argument(
        "--ln2-multiphase",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use LN2 parcels plus independent nitrogen/water-fog/ice-fog "
            "transport. Defaults to [cryogenic].enabled."
        ),
    )
    parser.add_argument("--ln2-mass-flow-kg-s", type=float, default=0.020)
    parser.add_argument("--ln2-injection-speed", type=float, default=8.0)
    parser.add_argument("--ln2-x", type=float, default=12.15)
    parser.add_argument("--ln2-y", type=float, default=3.0)
    parser.add_argument("--ln2-z", type=float, default=0.876)
    parser.add_argument("--ln2-sigma-x", type=float, default=0.15)
    parser.add_argument("--ln2-sigma-r", type=float, default=0.15)
    parser.add_argument(
        "--ln2-specific-cooling-j-kg",
        type=float,
        default=383_675.0,
        help=(
            "Equivalent heat removed per kg LN2. The default includes "
            "vaporization plus 75%% of warming gaseous nitrogen from 77.34 K "
            "to 300 K."
        ),
    )
    parser.add_argument(
        "--ln2-cooling-power-w",
        type=float,
        help="Override mass_flow * specific_cooling for a sensitivity case.",
    )
    parser.add_argument("--ln2-carrier-density", type=float, default=1.225)
    parser.add_argument("--ln2-carrier-heat-capacity", type=float, default=1005.0)
    return parser.parse_args()


def rank_and_size(args: argparse.Namespace) -> tuple[int, int]:
    rank = args.process_id
    size = args.num_processes
    if rank is None:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    if size is None:
        size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    return rank, size


def local_device_id(args: argparse.Namespace) -> int:
    if args.local_device_id is not None:
        return args.local_device_id
    for variable in (
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MPI_LOCALRANKID",
    ):
        if variable in os.environ:
            return int(os.environ[variable])
    return 0


def liquid_nitrogen_cooling_power(args: argparse.Namespace) -> float:
    if args.ln2_cooling_power_w is not None:
        return args.ln2_cooling_power_w
    return args.ln2_mass_flow_kg_s * args.ln2_specific_cooling_j_kg


def configured_run_params(configured, args: argparse.Namespace, total_steps: int):
    """Build the baseline or equivalent-LN2 warmup experiment parameters."""

    baseline = replace(
        configured,
        nsteps=total_steps,
        actuator_disk_enabled=False,
        cold_source_enabled=False,
        fringe_enabled=False,
        horizontal_homogeneous=True,
        buoyancy_reference="plane_mean",
        sharded_pressure_solver="transpose",
    )
    if not args.liquid_nitrogen_nozzle:
        return baseline, baseline

    nozzle = replace(
        baseline,
        thermo_enabled=True,
        moisture_enabled=bool(args.ln2_multiphase),
        horizontal_homogeneous=False,
        buoyancy_reference="ambient",
        fringe_enabled=configured.fringe_enabled,
        fringe_start_x=configured.fringe_start_x,
        fringe_timescale=configured.fringe_timescale,
        fringe_target_u=configured.fringe_target_u,
        fringe_target_v=configured.fringe_target_v,
        fringe_target_theta=configured.fringe_target_theta,
        cold_source_enabled=True,
        cold_source_x=args.ln2_x,
        cold_source_y=args.ln2_y,
        cold_source_z=args.ln2_z,
        cold_source_sigma_x=args.ln2_sigma_x,
        cold_source_sigma_r=args.ln2_sigma_r,
        cold_source_momentum_flux=(
            args.ln2_mass_flow_kg_s * args.ln2_injection_speed
        ),
        cold_source_cooling_power=liquid_nitrogen_cooling_power(args),
        cold_source_density=args.ln2_carrier_density,
        cold_source_heat_capacity=args.ln2_carrier_heat_capacity,
    )
    return nozzle, baseline


def write_profile_csv(
    path: Path,
    z_center: np.ndarray,
    z_face: np.ndarray,
    averaged: np.ndarray,
    names: tuple[str, ...],
) -> None:
    columns = np.column_stack((z_center, z_face, averaged.T))
    np.savetxt(
        path,
        columns,
        delimiter=",",
        header=",".join(("z_center_m", "z_upper_face_m", *names)),
        comments="",
    )


def make_diagnostic_figure(
    path: Path,
    profiles: np.ndarray,
    names: tuple[str, ...],
    ustar_samples: np.ndarray,
    params,
    duration_seconds: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = {name: i for i, name in enumerate(names)}
    mean = profiles.mean(axis=0)
    z = (np.arange(params.nz) + 0.5) * params.dz * params.z_i
    z_face = (np.arange(params.nz) + 1.0) * params.dz * params.z_i
    height = params.lz * params.z_i
    zh = z / height
    zfh = z_face / height
    ustar = params.pressure_ustar
    ulog = (ustar / params.vonk) * np.log(np.maximum(z, params.zo * 1.01) / params.zo)
    log_cap = (ustar / params.vonk) * np.log(params.bl_height / params.zo)
    ulog = np.where(z >= params.bl_height, log_cap, ulog)
    pressure_driven = ustar > 1.0e-12
    velocity_scale = (
        ustar
        if pressure_driven
        else max(float(np.max(np.abs(mean[index["mean_u"]]))), 1.0e-6)
    )
    velocity_scale_symbol = r"u_*" if pressure_driven else r"U_{\rm ref}"
    sponge = params.sponge_start_height / height if params.sponge_enabled else None

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(
        mean[index["mean_u"]],
        zh,
        lw=2.2,
        label=f"LES {duration_seconds:g} s mean",
    )
    ax.plot(ulog, zh, "--", lw=1.8, label="target log law")
    ax.set(xlabel=r"$\langle u\rangle$ [m s$^{-1}$]", ylabel=r"$z/H$")
    ax.legend()

    ax = axes[0, 1]
    ax.semilogy(
        mean[index["mean_u"]] / velocity_scale,
        z / params.zo,
        lw=2.2,
        label="LES",
    )
    ax.semilogy(
        ulog / velocity_scale,
        z / params.zo,
        "--",
        lw=1.8,
        label=(
            r"$\kappa^{-1}\ln(z/z_0)$"
            if pressure_driven
            else "quiescent target"
        ),
    )
    ax.set(
        xlabel=rf"$U/{velocity_scale_symbol}$",
        ylabel=r"$z/z_0$",
    )
    ax.legend()

    ax = axes[0, 2]
    scale = velocity_scale * velocity_scale
    ax.plot(mean[index["var_u"]] / scale, zh, label=r"$\langle u'^2\rangle$")
    ax.plot(mean[index["var_v"]] / scale, zh, label=r"$\langle v'^2\rangle$")
    ax.plot(mean[index["var_w"]] / scale, zh, label=r"$\langle w'^2\rangle$")
    ax.set(xlabel=r"variance $/u_*^2$", ylabel=r"$z/H$")
    ax.legend()

    ax = axes[1, 0]
    resolved = -mean[index["resolved_uw_face"]] / scale
    sgs = -mean[index["sgs_txz_face"]] / scale
    total = resolved + sgs
    expected = (
        np.maximum(1.0 - zfh, 0.0)
        if pressure_driven
        else np.zeros_like(zfh)
    )
    ax.plot(resolved, zfh, label="resolved")
    ax.plot(sgs, zfh, label="SGS")
    ax.plot(total, zfh, lw=2.2, label="total")
    ax.plot(
        expected,
        zfh,
        "--",
        lw=1.5,
        label="pressure balance" if pressure_driven else "zero forcing",
    )
    ax.set(
        xlabel=rf"$-\langle u'w'+\tau_{{xz}}\rangle/"
        rf"{velocity_scale_symbol}^2$",
        ylabel=r"$z/H$",
    )
    ax.legend()

    ax = axes[1, 1]
    tke = 0.5 * (
        mean[index["var_u"]]
        + mean[index["var_v"]]
        + mean[index["var_w"]]
    ) / scale
    ax.plot(tke, zh, lw=2.2)
    ax.set(
        xlabel="resolved " + rf"$k/{velocity_scale_symbol}^2$",
        ylabel=r"$z/H$",
    )

    ax = axes[1, 2]
    ax.plot(mean[index["mean_cs"]], zh, lw=2.2)
    ax.set(xlabel=r"$\langle C_s\rangle$", ylabel=r"$z/H$")
    ax.text(
        0.98,
        0.03,
        rf"$\overline{{u_*}}={ustar_samples.mean():.3f}$ m s$^{{-1}}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )

    for panel in axes.flat:
        panel.grid(True, alpha=0.28, lw=0.6)
    for panel in axes.flat:
        if panel is axes[0, 1]:
            continue
        panel.set_ylim(0.0, 1.0)
        if sponge is not None:
            panel.axhspan(sponge, 1.0, color="0.5", alpha=0.08)
            panel.axhline(sponge, color="0.45", lw=0.8, ls=":")
    if sponge is not None:
        sponge_inner = params.sponge_start_height / params.zo
        axes[0, 1].axhspan(
            sponge_inner,
            z[-1] / params.zo,
            color="0.5",
            alpha=0.08,
        )
        axes[0, 1].axhline(
            sponge_inner, color="0.45", lw=0.8, ls=":"
        )
    case_name = (
        "Pressure-driven neutral ABL"
        if pressure_driven
        else "Pure jet in quiescent ambient"
    )
    fig.suptitle(
        f"{case_name}: {profiles.shape[0]} samples over "
        f"{duration_seconds:g} s",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_profile_gif(
    path: Path,
    profiles: np.ndarray,
    names: tuple[str, ...],
    elapsed_times: np.ndarray,
    params,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    index = {name: i for i, name in enumerate(names)}
    z = (np.arange(params.nz) + 0.5) * params.dz * params.z_i
    zh = z / (params.lz * params.z_i)
    ustar = params.pressure_ustar
    ulog = (ustar / params.vonk) * np.log(np.maximum(z, params.zo * 1.01) / params.zo)
    log_cap = (ustar / params.vonk) * np.log(params.bl_height / params.zo)
    ulog = np.where(z >= params.bl_height, log_cap, ulog)
    u_frames = profiles[:, index["mean_u"], :]
    running = np.cumsum(u_frames, axis=0) / np.arange(1, len(u_frames) + 1)[:, None]
    xmin = min(float(u_frames.min()), float(ulog.min())) - 0.15
    xmax = max(float(u_frames.max()), float(ulog.max())) + 0.15

    fig, ax = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
    ax.plot(
        ulog,
        zh,
        "--",
        lw=1.7,
        color="0.25",
        label=(
            "target log law"
            if params.pressure_ustar > 1.0e-12
            else "quiescent target"
        ),
    )
    instantaneous, = ax.plot(u_frames[0], zh, lw=1.6, label="instantaneous plane mean")
    averaged, = ax.plot(running[0], zh, lw=2.4, label="running time mean")
    time_text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")
    if params.sponge_enabled:
        sponge = params.sponge_start_height / (params.lz * params.z_i)
        ax.axhspan(sponge, 1.0, color="0.5", alpha=0.08)
    ax.set(
        xlim=(xmin, xmax),
        ylim=(0.0, 1.0),
        xlabel=r"$\langle u\rangle_{xy}$ [m s$^{-1}$]",
        ylabel=r"$z/H$",
    )
    ax.grid(True, alpha=0.28, lw=0.6)
    ax.legend(loc="lower right")

    def update(frame: int):
        instantaneous.set_xdata(u_frames[frame])
        averaged.set_xdata(running[frame])
        time_text.set_text(f"elapsed time = {elapsed_times[frame]:.1f} s")
        return instantaneous, averaged, time_text

    animation = FuncAnimation(
        fig, update, frames=len(u_frames), interval=100, blit=True
    )
    animation.save(path, writer=PillowWriter(fps=10), dpi=120)
    plt.close(fig)


def make_flow_slices_gif(
    path: Path,
    xy_frames: np.ndarray,
    xz_frames: np.ndarray,
    yz_frames: np.ndarray,
    elapsed_times: np.ndarray,
    params,
    horizontal_height: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    values = np.concatenate(
        (xy_frames.ravel(), xz_frames.ravel(), yz_frames.ravel())
    )
    vmin = float(values.min())
    vmax = float(values.max())
    x_length = params.lx * params.z_i
    y_length = params.ly * params.z_i
    z_length = params.lz * params.z_i
    x_mid = 0.5 * x_length
    y_mid = 0.5 * y_length

    fig = plt.figure(figsize=(10.0, 6.0))
    grid = fig.add_gridspec(2, 2, width_ratios=(2.0, 1.0))
    axes = (
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[:, 1]),
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.89,
        bottom=0.09,
        top=0.88,
        wspace=0.24,
        hspace=0.38,
    )
    images = (
        axes[0].imshow(
            xy_frames[0].T,
            origin="lower",
            extent=(0.0, x_length, 0.0, y_length),
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            interpolation="bilinear",
            aspect="equal",
        ),
        axes[1].imshow(
            xz_frames[0].T,
            origin="lower",
            extent=(0.0, x_length, 0.0, z_length),
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            interpolation="bilinear",
            aspect="equal",
        ),
        axes[2].imshow(
            yz_frames[0].T,
            origin="lower",
            extent=(0.0, y_length, 0.0, z_length),
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            interpolation="bilinear",
            aspect="equal",
        ),
    )
    axes[0].set(
        xlabel="x [m]",
        ylabel="y [m]",
        title=rf"$u(x,y)$ at $z={horizontal_height:g}$ m",
    )
    axes[1].set(
        xlabel="x [m]",
        ylabel="z [m]",
        title=rf"$u(x,z)$ at $y={y_mid:g}$ m",
    )
    axes[2].set(
        xlabel="y [m]",
        ylabel="z [m]",
        title=rf"$u(y,z)$ at $x={x_mid:g}$ m",
    )
    if params.sponge_enabled:
        for ax in axes[1:]:
            ax.axhline(
                params.sponge_start_height,
                color="white",
                lw=1.0,
                ls=":",
                alpha=0.9,
            )
    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label(r"streamwise velocity $u$ [m s$^{-1}$]")
    title = fig.suptitle("")

    def update(frame: int):
        images[0].set_data(xy_frames[frame].T)
        images[1].set_data(xz_frames[frame].T)
        images[2].set_data(yz_frames[frame].T)
        title.set_text(f"ABL flow field: t = {elapsed_times[frame]:.1f} s")
        return (*images, title)

    animation = FuncAnimation(
        fig, update, frames=len(elapsed_times), interval=100, blit=False
    )
    # Pillow retains encoded frames until finalization.  Keep the raster compact
    # enough for 100-frame runs without sacrificing the 64x32 slice resolution.
    animation.save(path, writer=PillowWriter(fps=10), dpi=80)
    plt.close(fig)


def cold_plume_centerline(
    theta_xz_frames: np.ndarray,
    params,
) -> tuple[np.ndarray, np.ndarray]:
    """Return center-plane cold-weighted height and peak deficit versus x."""

    cooling = np.maximum(params.theta0 - theta_xz_frames, 0.0)
    z = (np.arange(params.nz) + 0.5) * params.dz * params.z_i
    weight = cooling.sum(axis=-1)
    numerator = np.sum(cooling * z[None, None, :], axis=-1)
    centroid = np.full_like(weight, np.nan)
    frame_threshold = np.maximum(
        5.0e-3 * weight.max(axis=-1, keepdims=True),
        1.0e-8,
    )
    np.divide(
        numerator,
        weight,
        out=centroid,
        where=weight > frame_threshold,
    )
    return centroid, cooling.max(axis=-1)


def make_cold_plume_gif(
    path: Path,
    theta_xz_frames: np.ndarray,
    w_xz_frames: np.ndarray,
    elapsed_times: np.ndarray,
    params,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    cooling = np.maximum(params.theta0 - theta_xz_frames, 0.0)
    centroid, _ = cold_plume_centerline(theta_xz_frames, params)
    x_length = params.lx * params.z_i
    z_length = params.lz * params.z_i
    x = (np.arange(params.nx) + 0.5) * params.dx * params.z_i
    cooling_max = max(float(cooling.max()), 1.0e-6)
    w_limit = max(float(np.max(np.abs(w_xz_frames))), 1.0e-6)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    cooling_image = axes[0].imshow(
        cooling[0].T,
        origin="lower",
        extent=(0.0, x_length, 0.0, z_length),
        vmin=0.0,
        vmax=cooling_max,
        cmap="inferno",
        interpolation="bilinear",
        aspect="auto",
    )
    w_image = axes[1].imshow(
        w_xz_frames[0].T,
        origin="lower",
        extent=(0.0, x_length, 0.0, z_length),
        vmin=-w_limit,
        vmax=w_limit,
        cmap="RdBu_r",
        interpolation="bilinear",
        aspect="auto",
    )
    centerline = axes[0].plot(
        x,
        centroid[0],
        color="cyan",
        lw=1.5,
        label="cold-weighted center height",
    )[0]
    for axis in axes:
        axis.scatter(
            [params.cold_source_x],
            [params.cold_source_z],
            marker=">",
            s=55,
            color="lime",
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
            label="equivalent LN2 nozzle",
        )
        axis.set_ylabel("z [m]")
        axis.grid(False)
    axes[0].legend(loc="upper right")
    axes[0].set_title(r"center-plane cooling $T_0-\theta$ [K]")
    axes[1].set_title(r"center-plane vertical velocity $w$ [m s$^{-1}$]")
    axes[1].set_xlabel("x [m]")
    fig.colorbar(cooling_image, ax=axes[0], pad=0.015)
    fig.colorbar(w_image, ax=axes[1], pad=0.015)
    title = fig.suptitle("")

    def update(frame: int):
        cooling_image.set_data(cooling[frame].T)
        w_image.set_data(w_xz_frames[frame].T)
        centerline.set_data(x, centroid[frame])
        title.set_text(f"Equivalent LN2 cold plume: t = {elapsed_times[frame]:.2f} s")
        return cooling_image, w_image, centerline, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(elapsed_times),
        interval=100,
        blit=False,
    )
    animation.save(path, writer=PillowWriter(fps=10), dpi=90)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rank, size = rank_and_size(args)
    selected_local_device_id = local_device_id(args)
    if size <= 0:
        raise SystemExit("num-processes must be positive")
    if args.duration_seconds <= 0.0 or args.frames <= 0:
        raise SystemExit("duration and frame count must be positive")
    if args.liquid_nitrogen_nozzle:
        positive = {
            "--ln2-mass-flow-kg-s": args.ln2_mass_flow_kg_s,
            "--ln2-injection-speed": args.ln2_injection_speed,
            "--ln2-sigma-x": args.ln2_sigma_x,
            "--ln2-sigma-r": args.ln2_sigma_r,
            "--ln2-specific-cooling-j-kg": args.ln2_specific_cooling_j_kg,
            "--ln2-carrier-density": args.ln2_carrier_density,
            "--ln2-carrier-heat-capacity": args.ln2_carrier_heat_capacity,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise SystemExit(f"{', '.join(invalid)} must be positive")
        if (
            args.ln2_cooling_power_w is not None
            and args.ln2_cooling_power_w <= 0.0
        ):
            raise SystemExit("--ln2-cooling-power-w must be positive")

    settings = dict(RUN_DEFAULTS)
    settings.update(load_config_file(args.config))
    if args.ln2_multiphase is None:
        args.ln2_multiphase = bool(settings["cryogenic_enabled"])
    if args.ln2_multiphase and not args.liquid_nitrogen_nozzle:
        raise SystemExit("--ln2-multiphase requires --liquid-nitrogen-nozzle")
    if settings["precision"] == "float64" or settings["sgs_precision"] == "float64":
        from jax import config as jax_config

        jax_config.update("jax_enable_x64", True)

    import jax

    if size > 1:
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=size,
            process_id=rank,
            local_device_ids=[selected_local_device_id],
        )
    import jax.numpy as jnp
    from jax.experimental import multihost_utils

    from wireles_jax.checkpoint_sharded import (
        load_sharded_checkpoint,
        read_sharded_checkpoint_manifest,
        save_sharded_checkpoint,
    )
    from wireles_jax.cryogenic_microphysics import CryogenicMicrophysicsConfig
    from wireles_jax.cryogenic_sharded import (
        ShardedCryogenicState,
        initial_cryogenic_scalar_state,
        load_cryogenic_checkpoint_sidecar,
        make_step_cryogenic_sharded,
        make_prescribed_ln2_mass_outlet_sharded,
    )
    from wireles_jax.spray_dpm import SprayDPMConfig
    from wireles_jax.spray_dpm_sharded import initialize_sharded_spray
    from wireles_jax.sharding import make_distributed_mesh
    from wireles_jax.timestep_sharded import (
        ABL_PROFILE_NAMES,
        initial_sharded_state,
        make_abl_profiles_sharded,
        make_diagnostics_sharded,
        make_flow_slices_sharded,
        make_project_velocity_sharded,
        make_sharded_operators,
        make_step_ab2_sharded,
        validate_cfl,
        validate_lasd_cfl,
    )

    configured = params_from_settings(settings, jnp)
    total_steps = int(round(args.duration_seconds / configured.dt_physical))
    if not math.isclose(
        total_steps * configured.dt_physical,
        args.duration_seconds,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise SystemExit("duration must be an integer multiple of the physical dt")
    if total_steps % args.frames:
        raise SystemExit(
            f"{total_steps} steps cannot be divided into exactly {args.frames} frames"
        )
    sample_every = total_steps // args.frames
    params, baseline_params = configured_run_params(
        configured,
        args,
        total_steps,
    )
    multiphase_enabled = bool(
        args.liquid_nitrogen_nozzle and args.ln2_multiphase
    )
    if multiphase_enabled and not settings["mass_outlet_enabled"]:
        raise SystemExit(
            "multiphase LN2 requires [mass_outlet].enabled=true in the "
            "periodic rigid-lid domain"
        )
    if multiphase_enabled:
        # Parcel exchange supplies momentum, cooling, and nitrogen mass.
        params = replace(params, cold_source_enabled=False)
    mesh = make_distributed_mesh(size)
    ops = make_sharded_operators(params, mesh)
    checkpoint_transition = "new_initial_state"
    if args.checkpoint is None:
        if rank == 0:
            print("[initial] creating and projecting the configured initial field", flush=True)
        state = initial_sharded_state(params, mesh)
        project = jax.jit(
            make_project_velocity_sharded(
                params,
                ops.pressure,
                mesh,
                spike_ops=ops.pressure_spike,
            )
        )
        u, v, w, p = project(
            state.u,
            state.v,
            state.w,
            ops.pressure,
            ops.pressure_spike,
        )
        state = state._replace(u=u, v=v, w=w, p=p)
    else:
        checkpoint_params = params
        checkpoint_transition = "exact_restart"
        if args.liquid_nitrogen_nozzle:
            checkpoint_manifest = read_sharded_checkpoint_manifest(
                args.checkpoint
            )
            saved = checkpoint_manifest.get("restart_parameters", {})
            if (
                saved
                and not bool(saved.get("cold_source_enabled", False))
                and not bool(saved.get("thermo_enabled", False))
            ):
                checkpoint_params = baseline_params
                checkpoint_transition = "baseline_flow_to_ln2_thermo"
        state = load_sharded_checkpoint(
            args.checkpoint,
            checkpoint_params,
            mesh,
            rank=rank,
        )
    mass_outlet_enabled = bool(
        args.liquid_nitrogen_nozzle and settings["mass_outlet_enabled"]
    )
    outlet_end_x = settings["mass_outlet_end_x"]
    if outlet_end_x is None:
        outlet_end_x = params.lx * params.z_i
    outlet_config = CryogenicMicrophysicsConfig(
        pressure=params.surface_pressure,
        dry_air_density=args.ln2_carrier_density,
        dry_air_heat_capacity=args.ln2_carrier_heat_capacity,
        outlet_start_x=float(settings["mass_outlet_start_x"]),
        outlet_end_x=float(outlet_end_x),
        outlet_scalar_timescale=float(settings["mass_outlet_timescale"]),
        saturation_relaxation_timescale=float(
            settings["cryogenic_saturation_timescale"]
        ),
        freezing_timescale=float(
            settings["cryogenic_freezing_timescale"]
        ),
        melting_timescale=float(
            settings["cryogenic_melting_timescale"]
        ),
        liquid_fog_diameter=float(
            settings["cryogenic_liquid_fog_diameter"]
        ),
        ice_fog_diameter=float(
            settings["cryogenic_ice_fog_diameter"]
        ),
    )

    if multiphase_enabled:
        injection_end_time = (
            int(jax.device_get(state.step)) * params.dt_physical
            + args.duration_seconds
        )
        spray_config = SprayDPMConfig(
            material="nitrogen",
            max_parcels=int(settings["cryogenic_max_parcels_per_shard"]),
            parcels_per_step=int(settings["cryogenic_parcels_per_step"]),
            mass_flow_rate=args.ln2_mass_flow_kg_s,
            injection_end_time=injection_end_time,
            injection_x=params.cold_source_x,
            injection_y=params.cold_source_y,
            injection_z=params.cold_source_z,
            injection_radius=params.cold_source_sigma_r,
            injection_u=args.ln2_injection_speed,
            initial_diameter=float(
                settings["cryogenic_initial_diameter"]
            ),
            diameter_distribution="rosin_rammler",
            minimum_diameter=float(
                settings["cryogenic_minimum_diameter"]
            ),
            maximum_diameter=float(
                settings["cryogenic_maximum_diameter"]
            ),
            rosin_rammler_spread=float(
                settings["cryogenic_rosin_rammler_spread"]
            ),
            initial_temperature=outlet_config.nitrogen_boiling_temperature,
            boiling_temperature=outlet_config.nitrogen_boiling_temperature,
            substeps=int(settings["cryogenic_substeps"]),
            air_density=outlet_config.dry_air_density,
            air_heat_capacity=outlet_config.dry_air_heat_capacity,
            liquid_density=outlet_config.liquid_nitrogen_density,
            water_density=outlet_config.liquid_nitrogen_density,
            liquid_heat_capacity=outlet_config.liquid_nitrogen_heat_capacity,
            latent_heat=outlet_config.liquid_nitrogen_latent_heat,
            surface_tension=8.85e-3,
        )
        cryogenic_manifest = (
            None
            if args.checkpoint is None
            else args.checkpoint / "cryogenic_manifest.json"
        )
        if cryogenic_manifest is not None and cryogenic_manifest.exists():
            state = load_cryogenic_checkpoint_sidecar(
                args.checkpoint,
                state,
                spray_config,
                params,
                mesh,
                rank=rank,
            )
            checkpoint_transition = "exact_multiphase_restart"
        else:
            state = ShardedCryogenicState(
                flow=state,
                spray=initialize_sharded_spray(
                    spray_config, params, mesh
                ),
                scalars=initial_cryogenic_scalar_state(
                    state, outlet_config
                ),
            )
            if args.checkpoint is not None:
                checkpoint_transition += "_fresh_cryogenic_fields"
        start_step = int(jax.device_get(state.flow.step))
        step_fn = make_step_cryogenic_sharded(
            spray_config,
            outlet_config,
            params,
            ops,
            mesh,
        )
    else:
        start_step = int(jax.device_get(state.step))
        base_step_fn = make_step_ab2_sharded(params, ops, mesh)
        if not mass_outlet_enabled:
            step_fn = base_step_fn
        else:
            mass_outlet_target = make_prescribed_ln2_mass_outlet_sharded(
                params,
                mesh,
                mass_flow_rate=args.ln2_mass_flow_kg_s,
                source_x=params.cold_source_x,
                source_y=params.cold_source_y,
                source_z=params.cold_source_z,
                source_sigma_x=params.cold_source_sigma_x,
                source_sigma_r=params.cold_source_sigma_r,
                config=outlet_config,
                return_scalar_sink=True,
            )

            def step_fn(state, runtime_pressure_ops, runtime_spike_ops):
                target_divergence, scalar_sink = mass_outlet_target(
                    state.theta
                )
                return base_step_fn(
                    state,
                    runtime_pressure_ops,
                    runtime_spike_ops,
                    extra_rhs_theta=-scalar_sink * state.theta,
                    extra_rhs_qv=-scalar_sink * state.qv,
                    target_divergence=target_divergence,
                )
    profile_fn = make_abl_profiles_sharded(params, ops.horizontal, mesh)
    slice_fn = make_flow_slices_sharded(
        params, mesh, horizontal_height=args.flow_height
    )
    diag_fn = make_diagnostics_sharded(params, ops.horizontal, mesh)

    if rank == 0:
        if args.liquid_nitrogen_nozzle:
            if multiphase_enabled:
                print(
                    f"[ln2] multiphase parcel nozzle at "
                    f"({params.cold_source_x:g}, {params.cold_source_y:g}, "
                    f"{params.cold_source_z:g}) m, +x injection; "
                    f"mass_flow={args.ln2_mass_flow_kg_s:g} kg/s, "
                    f"speed={args.ln2_injection_speed:g} m/s, "
                    f"d50={spray_config.initial_diameter:.3e} m, "
                    f"substeps={spray_config.substeps}; "
                    f"mass_outlet={'on' if mass_outlet_enabled else 'off'}; "
                    f"checkpoint_transition={checkpoint_transition}",
                    flush=True,
                )
            else:
                print(
                    f"[ln2] equivalent nozzle at "
                    f"({params.cold_source_x:g}, {params.cold_source_y:g}, "
                    f"{params.cold_source_z:g}) m, +x injection; "
                    f"mass_flow={args.ln2_mass_flow_kg_s:g} kg/s, "
                    f"speed={args.ln2_injection_speed:g} m/s, "
                    f"momentum={params.cold_source_momentum_flux:g} N, "
                    f"cooling={params.cold_source_cooling_power:g} W; "
                    f"mass_outlet={'on' if mass_outlet_enabled else 'off'}; "
                    f"checkpoint_transition={checkpoint_transition}",
                    flush=True,
                )
        print(
            f"[run] start step={start_step}; advancing {total_steps} steps "
            f"({args.duration_seconds:g}s), sampling every {sample_every} steps",
            flush=True,
        )
    lowered = jax.jit(step_fn).lower(state, ops.pressure, ops.pressure_spike)
    step_fn = lowered.compile()
    profile_fn = jax.jit(profile_fn)
    slice_fn = jax.jit(slice_fn)
    diag_fn = jax.jit(diag_fn)

    sampled_profiles: list[np.ndarray] = []
    sampled_ustar: list[float] = []
    sampled_divergence: list[float] = []
    sampled_cfl: list[float] = []
    elapsed_times: list[float] = []
    sampled_xy: list[np.ndarray] = []
    sampled_xz: list[np.ndarray] = []
    sampled_yz: list[np.ndarray] = []
    sampled_theta_xz: list[np.ndarray] = []
    sampled_w_xz: list[np.ndarray] = []
    sampled_yn2_xz: list[np.ndarray] = []
    sampled_ql_xz: list[np.ndarray] = []
    sampled_qi_xz: list[np.ndarray] = []
    sampled_nitrogen_mass: list[float] = []
    sampled_liquid_fog_mass: list[float] = []
    sampled_ice_fog_mass: list[float] = []
    sampled_relative_humidity: list[float] = []
    sampled_evaporated_mass: list[float] = []
    sampled_nitrogen_sensible_cooling: list[float] = []
    sampled_nitrogen_outlet_mass: list[float] = []
    sampled_fog_condensed_mass: list[float] = []
    sampled_fog_evaporated_mass: list[float] = []
    wall_start = time.perf_counter()
    for local_step in range(1, total_steps + 1):
        if multiphase_enabled:
            state, cryogenic_diagnostics = step_fn(
                state, ops.pressure, ops.pressure_spike
            )
            flow_state = state.flow
        else:
            state = step_fn(state, ops.pressure, ops.pressure_spike)
            flow_state = state
        if local_step % sample_every:
            continue
        profile = jax.block_until_ready(
            profile_fn(
                flow_state.u,
                flow_state.v,
                flow_state.w,
                flow_state.cs2,
            )
        )
        diag = jax.block_until_ready(
            diag_fn(
                flow_state.u,
                flow_state.v,
                flow_state.w,
                flow_state.theta,
                flow_state.qv,
                flow_state.step,
            )
        )
        validate_cfl(diag)
        if params.sgs_model == "lasd":
            validate_lasd_cfl(diag, params)
        xy_slice, xz_slice, yz_slice = jax.block_until_ready(
            slice_fn(flow_state.u)
        )
        if args.liquid_nitrogen_nozzle:
            theta_slices, w_slices = jax.block_until_ready(
                (
                    slice_fn(flow_state.theta),
                    slice_fn(flow_state.w),
                )
            )
        if multiphase_enabled:
            yn2_slices, ql_slices, qi_slices = jax.block_until_ready(
                (
                    slice_fn(state.scalars.yn2),
                    slice_fn(state.scalars.ql),
                    slice_fn(state.scalars.qi),
                )
            )
        if rank == 0:
            sampled_profiles.append(np.asarray(profile))
            sampled_ustar.append(float(diag.ustar))
            sampled_divergence.append(float(diag.div_max))
            sampled_cfl.append(float(diag.cfl_x + diag.cfl_y + diag.cfl_z))
            elapsed_times.append(local_step * params.dt_physical)
            sampled_xy.append(np.asarray(xy_slice))
            sampled_xz.append(np.asarray(xz_slice))
            sampled_yz.append(np.asarray(yz_slice))
            if args.liquid_nitrogen_nozzle:
                theta_xz = np.asarray(theta_slices[1])
                w_xz = np.asarray(w_slices[1])
                sampled_theta_xz.append(theta_xz)
                sampled_w_xz.append(w_xz)
            if multiphase_enabled:
                sampled_yn2_xz.append(np.asarray(yn2_slices[1]))
                sampled_ql_xz.append(np.asarray(ql_slices[1]))
                sampled_qi_xz.append(np.asarray(qi_slices[1]))
                sampled_nitrogen_mass.append(
                    float(cryogenic_diagnostics.nitrogen_gas_mass)
                )
                sampled_liquid_fog_mass.append(
                    float(cryogenic_diagnostics.liquid_fog_mass)
                )
                sampled_ice_fog_mass.append(
                    float(cryogenic_diagnostics.ice_fog_mass)
                )
                sampled_relative_humidity.append(
                    float(cryogenic_diagnostics.max_relative_humidity)
                )
                sampled_evaporated_mass.append(
                    float(cryogenic_diagnostics.spray.evaporated_mass)
                )
                sampled_nitrogen_sensible_cooling.append(
                    float(
                        cryogenic_diagnostics.nitrogen_sensible_cooling
                    )
                )
                sampled_nitrogen_outlet_mass.append(
                    float(cryogenic_diagnostics.nitrogen_outlet_mass)
                )
                sampled_fog_condensed_mass.append(
                    float(cryogenic_diagnostics.fog_condensed_mass)
                )
                sampled_fog_evaporated_mass.append(
                    float(cryogenic_diagnostics.fog_evaporated_mass)
                )
            frame = len(sampled_profiles)
            if frame % 10 == 0 or frame == args.frames:
                message = (
                    f"[sample] frame={frame}/{args.frames}, "
                    f"t+={elapsed_times[-1]:.1f}s, "
                    f"ustar={sampled_ustar[-1]:.4f}, "
                    f"CFL={sampled_cfl[-1]:.4f}"
                )
                if args.liquid_nitrogen_nozzle:
                    centroid, peak = cold_plume_centerline(
                        theta_xz[None, ...],
                        params,
                    )
                    downstream = (
                        np.arange(params.nx) + 0.5
                    ) * params.dx * params.z_i >= params.cold_source_x
                    valid = downstream & np.isfinite(centroid[0])
                    minimum_height = (
                        float(np.min(centroid[0, valid]))
                        if np.any(valid)
                        else math.nan
                    )
                    message += (
                        f", max_dT={float(peak.max()):.3f}K, "
                        f"min_cold_z={minimum_height:.3f}m"
                    )
                if multiphase_enabled:
                    message += (
                        f", N2gas={sampled_nitrogen_mass[-1]:.4e}kg, "
                        f"fog={sampled_liquid_fog_mass[-1] + sampled_ice_fog_mass[-1]:.4e}kg, "
                        f"RHmax={sampled_relative_humidity[-1]:.3f}"
                    )
                print(message, flush=True)

    state = jax.block_until_ready(state)
    final_flow_state = state.flow if multiphase_enabled else state
    final_checkpoint = args.output_dir / "final_checkpoint"
    save_sharded_checkpoint(
        final_checkpoint,
        final_flow_state,
        params,
        mesh,
        rank=rank,
    )
    if multiphase_enabled:
        cryogenic_payload = {
            f"scalar_{name}": np.asarray(
                getattr(state.scalars, name).addressable_shards[0].data
            )
            for name in state.scalars._fields
        }
        cryogenic_payload.update(
            {
                f"spray_{name}": np.asarray(
                    getattr(state.spray, name).addressable_shards[0].data
                )
                for name in state.spray._fields
            }
        )
        np.savez_compressed(
            final_checkpoint / f"cryogenic_rank{rank:05d}.npz",
            **cryogenic_payload,
        )
        if rank == 0:
            cryogenic_manifest = {
                "format": "wireles-jax-cryogenic-zslab-v1",
                "source_parts": size,
                "global_shape": [params.nx, params.ny, params.nz],
                "step": int(jax.device_get(state.flow.step)),
                "scalar_fields": list(state.scalars._fields),
                "spray_fields": list(state.spray._fields),
                "material": spray_config.material,
                "mass_flow_rate_kg_s": spray_config.mass_flow_rate,
                "max_parcels_per_shard": spray_config.max_parcels,
            }
            (
                final_checkpoint / "cryogenic_manifest.json"
            ).write_text(
                json.dumps(cryogenic_manifest, indent=2) + "\n"
            )
    multihost_utils.sync_global_devices("abl-diagnostics-checkpoint-written")
    restored = load_sharded_checkpoint(
        final_checkpoint, params, mesh, rank=rank
    )
    local_exact = True
    for name in final_flow_state._fields:
        current = getattr(final_flow_state, name)
        reloaded = getattr(restored, name)
        if name == "step":
            local_exact &= int(jax.device_get(current)) == int(jax.device_get(reloaded))
        else:
            local_exact &= np.array_equal(
                np.asarray(current.addressable_shards[0].data),
                np.asarray(reloaded.addressable_shards[0].data),
            )
    exact_count = int(
        multihost_utils.process_allgather(np.asarray(int(local_exact))).sum()
    )
    multihost_utils.sync_global_devices("abl-diagnostics-checkpoint-validated")
    jax.distributed.shutdown()

    if rank != 0:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_array = np.stack(sampled_profiles)
    ustar_array = np.asarray(sampled_ustar)
    divergence_array = np.asarray(sampled_divergence)
    cfl_array = np.asarray(sampled_cfl)
    elapsed_array = np.asarray(elapsed_times)
    xy_array = np.stack(sampled_xy)
    xz_array = np.stack(sampled_xz)
    yz_array = np.stack(sampled_yz)
    if args.liquid_nitrogen_nozzle:
        theta_xz_array = np.stack(sampled_theta_xz)
        w_xz_array = np.stack(sampled_w_xz)
        cold_centroid, peak_cooling = cold_plume_centerline(
            theta_xz_array,
            params,
        )
        np.savez_compressed(
            args.output_dir / "ln2_cold_plume_centerplane_100frames.npz",
            theta_xz=theta_xz_array,
            w_xz=w_xz_array,
            cooling_xz=np.maximum(params.theta0 - theta_xz_array, 0.0),
            cold_centroid_z_m=cold_centroid,
            peak_cooling_k=peak_cooling,
            elapsed_seconds=elapsed_array,
            nozzle_xyz_m=np.asarray(
                (
                    params.cold_source_x,
                    params.cold_source_y,
                    params.cold_source_z,
                )
            ),
        )
    if multiphase_enabled:
        np.savez_compressed(
            args.output_dir / "ln2_multiphase_centerplane_100frames.npz",
            yn2_xz=np.stack(sampled_yn2_xz),
            liquid_fog_xz=np.stack(sampled_ql_xz),
            ice_fog_xz=np.stack(sampled_qi_xz),
            nitrogen_gas_mass_kg=np.asarray(sampled_nitrogen_mass),
            liquid_fog_mass_kg=np.asarray(sampled_liquid_fog_mass),
            ice_fog_mass_kg=np.asarray(sampled_ice_fog_mass),
            max_relative_humidity=np.asarray(sampled_relative_humidity),
            evaporated_nitrogen_mass_per_step_kg=np.asarray(
                sampled_evaporated_mass
            ),
            nitrogen_sensible_cooling_per_step_j=np.asarray(
                sampled_nitrogen_sensible_cooling
            ),
            nitrogen_outlet_mass_per_step_kg=np.asarray(
                sampled_nitrogen_outlet_mass
            ),
            fog_condensed_mass_per_step_kg=np.asarray(
                sampled_fog_condensed_mass
            ),
            fog_evaporated_mass_per_step_kg=np.asarray(
                sampled_fog_evaporated_mass
            ),
            elapsed_seconds=elapsed_array,
        )
    np.savez_compressed(
        args.output_dir / "abl_profile_samples.npz",
        profiles=profile_array,
        profile_names=np.asarray(ABL_PROFILE_NAMES),
        elapsed_seconds=elapsed_array,
        ustar=ustar_array,
        div_max=divergence_array,
        cfl_total=cfl_array,
    )
    np.savez_compressed(
        args.output_dir / "abl_flow_slices_100frames.npz",
        u_xy=xy_array,
        u_xz=xz_array,
        u_yz=yz_array,
        elapsed_seconds=elapsed_array,
        horizontal_height_m=np.asarray(args.flow_height),
        x_midpoint_m=np.asarray(0.5 * params.lx * params.z_i),
        y_midpoint_m=np.asarray(0.5 * params.ly * params.z_i),
    )
    z_center = (np.arange(params.nz) + 0.5) * params.dz * params.z_i
    z_face = (np.arange(params.nz) + 1.0) * params.dz * params.z_i
    write_profile_csv(
        args.output_dir / "abl_profile_time_mean.csv",
        z_center,
        z_face,
        profile_array.mean(axis=0),
        ABL_PROFILE_NAMES,
    )
    figure = args.output_dir / "abl_standard_profile_diagnostics.png"
    gif = args.output_dir / "abl_loglaw_profile_100frames.gif"
    flow_gif = args.output_dir / "abl_flow_three_slices_100frames.gif"
    cold_plume_gif = args.output_dir / "ln2_cold_plume_100frames.gif"
    make_diagnostic_figure(
        figure,
        profile_array,
        ABL_PROFILE_NAMES,
        ustar_array,
        params,
        args.duration_seconds,
    )
    make_profile_gif(
        gif, profile_array, ABL_PROFILE_NAMES, elapsed_array, params
    )
    make_flow_slices_gif(
        flow_gif,
        xy_array,
        xz_array,
        yz_array,
        elapsed_array,
        params,
        args.flow_height,
    )
    if args.liquid_nitrogen_nozzle:
        make_cold_plume_gif(
            cold_plume_gif,
            theta_xz_array,
            w_xz_array,
            elapsed_array,
            params,
        )
        x = (np.arange(params.nx) + 0.5) * params.dx * params.z_i
        final_rows = np.column_stack(
            (
                x,
                cold_centroid[-1],
                params.cold_source_z - cold_centroid[-1],
                peak_cooling[-1],
            )
        )
        np.savetxt(
            args.output_dir / "ln2_final_centerplane_descent.csv",
            final_rows,
            delimiter=",",
            header=(
                "x_m,cold_centroid_z_m,descent_from_nozzle_m,"
                "peak_cooling_k"
            ),
            comments="",
        )
    metadata = {
        "source_checkpoint": (
            None if args.checkpoint is None else str(args.checkpoint.resolve())
        ),
        "start_step": start_step,
        "final_step": int(start_step + total_steps),
        "dt_seconds": params.dt_physical,
        "duration_seconds": args.duration_seconds,
        "bl_height_m": params.bl_height,
        "pressure_force_height_m": params.forcing_height * params.z_i,
        "pressure_acceleration_m_s2": (
            params.driving_pressure_force / params.z_i
        ),
        "frames": args.frames,
        "sample_every_steps": sample_every,
        "time_mean_samples": args.frames,
        "mean_ustar": float(ustar_array.mean()),
        "max_divergence": float(divergence_array.max()),
        "max_total_cfl": float(cfl_array.max()),
        "restart_roundtrip_exact_ranks": exact_count,
        "restart_roundtrip_total_ranks": size,
        "local_device_id": selected_local_device_id,
        "checkpoint_transition": checkpoint_transition,
        "liquid_nitrogen_nozzle": (
            {
                "enabled": True,
                "model": (
                    "lagrangian_ln2_eulerian_n2_water_fog_ice_fog"
                    if multiphase_enabled
                    else "equivalent_streamwise_momentum_and_cooling_source"
                ),
                "multiphase_enabled": multiphase_enabled,
                "mass_flow_kg_s": args.ln2_mass_flow_kg_s,
                "injection_speed_m_s": args.ln2_injection_speed,
                "direction": "+x",
                "position_m": [
                    params.cold_source_x,
                    params.cold_source_y,
                    params.cold_source_z,
                ],
                "sigma_x_m": params.cold_source_sigma_x,
                "sigma_r_m": params.cold_source_sigma_r,
                "specific_cooling_j_kg": args.ln2_specific_cooling_j_kg,
                "momentum_flux_n": params.cold_source_momentum_flux,
                "cooling_power_w": params.cold_source_cooling_power,
                "carrier_density_kg_m3": params.cold_source_density,
                "carrier_heat_capacity_j_kg_k": (
                    params.cold_source_heat_capacity
                ),
                "buoyancy_reference": params.buoyancy_reference,
                "outflow_fringe_enabled": params.fringe_enabled,
                "outflow_fringe_start_x_m": (
                    params.fringe_start_x if params.fringe_enabled else None
                ),
                "outflow_fringe_timescale_s": (
                    params.fringe_timescale if params.fringe_enabled else None
                ),
                "mass_source_included": mass_outlet_enabled,
                "mass_only_outlet_enabled": mass_outlet_enabled,
                "mass_only_outlet_start_x_m": (
                    float(settings["mass_outlet_start_x"])
                    if mass_outlet_enabled
                    else None
                ),
                "mass_only_outlet_end_x_m": (
                    float(
                        settings["mass_outlet_end_x"]
                        if settings["mass_outlet_end_x"] is not None
                        else params.lx * params.z_i
                    )
                    if mass_outlet_enabled
                    else None
                ),
                "outlet_velocity_relaxation": False,
                "water_vapor_initial_kg_kg": params.qv0,
                "cryogenic_checkpoint_sidecars": (
                    "cryogenic_rankNNNNN.npz"
                    if multiphase_enabled
                    else None
                ),
            }
            if args.liquid_nitrogen_nozzle
            else {"enabled": False}
        ),
        "wall_seconds": time.perf_counter() - wall_start,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    if args.copy_to is not None:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure, args.copy_to / figure.name)
        shutil.copy2(gif, args.copy_to / gif.name)
        shutil.copy2(flow_gif, args.copy_to / flow_gif.name)
        if args.liquid_nitrogen_nozzle:
            shutil.copy2(
                cold_plume_gif,
                args.copy_to / cold_plume_gif.name,
            )
    print(f"[done] diagnostics: {figure}", flush=True)
    print(f"[done] animation: {gif}", flush=True)
    print(f"[done] flow animation: {flow_gif}", flush=True)
    if args.liquid_nitrogen_nozzle:
        print(f"[done] LN2 cold-plume animation: {cold_plume_gif}", flush=True)
    print(
        f"[done] restart round-trip exact on {exact_count}/{size} ranks",
        flush=True,
    )


if __name__ == "__main__":
    main()
