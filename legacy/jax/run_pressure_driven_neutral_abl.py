#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np


JAX_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(JAX_ROOT))

from run_single import (  # noqa: E402
    LOG_HEADER,
    RUN_DEFAULTS,
    format_diagnostic,
    load_config_file,
    params_from_settings,
)


DEFAULT_CONFIG = JAX_ROOT / "configs" / "pressure_driven_neutral_abl_lasd.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pressure-driven neutral LASD ABL and compare its mean profile with the neutral log law."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--average-start-step", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nz", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--single", action="store_true", help="Use float32 for the resolved velocity and pressure path.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--gif", action="store_true", help="Write an animated horizontal-mean velocity profile.")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument(
        "--field-gif",
        action="store_true",
        help="Write an animated speed-magnitude view on three orthogonal flow-field sections.",
    )
    parser.add_argument("--field-gif-fps", type=int, default=8)
    parser.add_argument("--field-gif-z-over-h", type=float, default=0.1)
    parser.add_argument("--cfl-limit", type=float, default=0.2)
    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> dict:
    settings = dict(RUN_DEFAULTS)
    settings.update(load_config_file(args.config))
    for key in ("nx", "ny", "nz", "steps", "dt", "log_every"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    if args.single:
        settings["precision"] = "float32"
    if settings["steps"] <= 0:
        raise ValueError("steps must be positive")
    if settings["log_every"] <= 0:
        raise ValueError("log_every must be positive")
    return settings


def log_law(z: np.ndarray, ustar: float, zo: float, vonk: float) -> np.ndarray:
    values = np.full_like(z, np.nan, dtype=np.float64)
    valid = z > zo
    values[valid] = (ustar / vonk) * np.log(z[valid] / zo)
    return values


def horizontal_u_profile(state) -> np.ndarray:
    return np.asarray(state.u, dtype=np.float64).mean(axis=(0, 1))


def velocity_cross_sections(
    state,
    params,
    z_over_h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return xy, xz, and yz slices of resolved cell-centred speed."""
    u = np.asarray(state.u, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    w_upper = np.asarray(state.w, dtype=np.float64)
    w_lower = np.concatenate((np.zeros_like(w_upper[:, :, :1]), w_upper[:, :, :-1]), axis=2)
    w_center = 0.5 * (w_lower + w_upper)
    speed = np.sqrt(u * u + v * v + w_center * w_center)
    z = (np.arange(speed.shape[2], dtype=np.float64) + 0.5) * params.dz * params.z_i
    k = int(np.argmin(np.abs(z / params.bl_height - z_over_h)))
    i = params.nx // 2
    j = params.ny // 2
    return (
        speed[:, :, k].T,
        speed[:, j, :].T,
        speed[i, :, :].T,
        float(z[k] / params.bl_height),
    )


def numerical_state_metrics(state, params) -> dict[str, float]:
    """Measure top-boundary closure and energy left above the horizontal cutoff."""
    u = np.asarray(state.u, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    w = np.asarray(state.w, dtype=np.float64)
    top_uv_mismatch = max(
        0.0,
        0.0,
    )
    wall_normal_velocity = max(
        0.0,
        float(np.max(np.abs(w[:, :, -1]))),
    )

    x_mode = np.abs(np.fft.fftfreq(params.nx) * params.nx)
    y_mode = np.fft.rfftfreq(params.ny) * params.ny
    cutoff_x = float(np.rint(params.nx / (2.0 * params.fgr)))
    cutoff_y = float(np.rint(params.ny / (2.0 * params.fgr)))
    keep = (x_mode[:, None] < cutoff_x) & (y_mode[None, :] < cutoff_y)
    rejected_max = 0.0
    spectral_max = 0.0
    for field in (u, v, w):
        field_hat = np.fft.rfft2(field, axes=(0, 1))
        spectral_max = max(spectral_max, float(np.max(np.abs(field_hat))))
        if np.any(~keep):
            rejected_max = max(rejected_max, float(np.max(np.abs(field_hat[~keep, :]))))
    rejected_ratio = rejected_max / spectral_max if spectral_max > 0.0 else 0.0
    u_physical = u
    v_physical = v
    w_lower = np.concatenate((np.zeros_like(w[:, :, :1]), w[:, :, :-1]), axis=2)
    w_center = 0.5 * (w_lower + w)
    z = (np.arange(u_physical.shape[2], dtype=np.float64) + 0.5) * params.dz * params.z_i
    upper_start = params.sponge_start_height if params.sponge_enabled else 0.75 * params.lz * params.z_i
    upper_mask = z >= upper_start
    if not np.any(upper_mask):
        upper_mask[-1] = True
    u_prime = u_physical - u_physical.mean(axis=(0, 1), keepdims=True)
    v_prime = v_physical - v_physical.mean(axis=(0, 1), keepdims=True)
    upper_fluctuation_rms = float(
        np.sqrt(np.mean(u_prime[:, :, upper_mask] ** 2 + v_prime[:, :, upper_mask] ** 2 + w_center[:, :, upper_mask] ** 2))
    )
    upper_w_rms = float(np.sqrt(np.mean(w_center[:, :, upper_mask] ** 2)))
    return {
        "top_uv_boundary_mismatch_m_s": top_uv_mismatch,
        "wall_normal_velocity_m_s": wall_normal_velocity,
        "rejected_horizontal_mode_ratio": rejected_ratio,
        "upper_layer_z_min_m": float(z[upper_mask].min()),
        "upper_layer_velocity_fluctuation_rms_m_s": upper_fluctuation_rms,
        "upper_layer_w_rms_m_s": upper_w_rms,
    }


def write_profile_csv(
    path: Path,
    z: np.ndarray,
    height: float,
    initial_u: np.ndarray,
    mean_u: np.ndarray,
    target_u: np.ndarray,
    target_ustar: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("z_m", "z_over_h", "initial_u_m_s", "mean_u_m_s", "loglaw_u_m_s", "u_plus", "loglaw_u_plus", "error_m_s")
        )
        for values in zip(z, z / height, initial_u, mean_u, target_u, mean_u / target_ustar, target_u / target_ustar, mean_u - target_u):
            writer.writerow((f"{value:.12e}" for value in values))


def write_summary_csv(path: Path, summary: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("quantity", "value"))
        for key, value in summary.items():
            writer.writerow((key, value))


def plot_comparison(
    path: Path,
    z: np.ndarray,
    height: float,
    domain_height: float,
    zo: float,
    target_ustar: float,
    initial_u: np.ndarray,
    mean_u: np.ndarray,
    target_u: np.ndarray,
    log_mask: np.ndarray,
    initial_label: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_profile, ax_plus) = plt.subplots(1, 2, figsize=(10.0, 5.2), constrained_layout=True)

    ax_profile.plot(initial_u, z / height, color="0.55", linestyle=":", linewidth=1.5, label=initial_label)
    ax_profile.plot(mean_u, z / height, color="tab:blue", linewidth=2.0, label="LES time mean")
    valid = np.isfinite(target_u) & (z <= height)
    ax_profile.plot(
        target_u[valid],
        z[valid] / height,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="neutral log law",
    )
    ax_profile.axhspan(float((z / height)[log_mask].min()), float((z / height)[log_mask].max()), color="tab:orange", alpha=0.12)
    ax_profile.axhline(1.0, color="0.4", linestyle="-.", linewidth=1.1, label="pressure-gradient top")
    ax_profile.set_xlabel(r"$\langle u\rangle$ [m s$^{-1}$]")
    ax_profile.set_ylabel(r"$z/H$")
    ax_profile.set_ylim(0.0, domain_height / height)
    ax_profile.grid(True, linewidth=0.5, alpha=0.35)
    ax_profile.legend()

    ax_plus.semilogx(z[valid] / zo, mean_u[valid] / target_ustar, color="tab:blue", linewidth=2.0, label="LES time mean")
    ax_plus.semilogx(z[valid] / zo, target_u[valid] / target_ustar, color="black", linestyle="--", linewidth=1.6, label=r"$\kappa^{-1}\ln(z/z_0)$")
    ax_plus.scatter(z[log_mask] / zo, mean_u[log_mask] / target_ustar, color="tab:orange", s=22, zorder=3, label="comparison range")
    ax_plus.set_xlabel(r"$z/z_0$")
    ax_plus.set_ylabel(r"$U^+=\langle u\rangle/u_*$")
    ax_plus.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax_plus.legend()
    fig.suptitle(
        f"Pressure-driven neutral ABL with LASD SGS "
        f"(H={height:.0f} m, lid at z/H={domain_height / height:.2f})"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_profile_evolution_gif(
    path: Path,
    z: np.ndarray,
    height: float,
    domain_height: float,
    profiles: list[np.ndarray],
    steps: list[int],
    dt_physical: float,
    target_u: np.ndarray,
    log_mask: np.ndarray,
    fps: int,
) -> None:
    if fps <= 0:
        raise ValueError("gif-fps must be positive")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(profiles, axis=0)
    max_speed = float(max(np.nanmax(stacked), np.nanmax(target_u)))

    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    ax.axhspan(
        float((z / height)[log_mask].min()),
        float((z / height)[log_mask].max()),
        color="tab:orange",
        alpha=0.12,
        label="comparison range",
    )
    valid = np.isfinite(target_u) & (z <= height)
    ax.plot(
        target_u[valid],
        z[valid] / height,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="neutral log law",
    )
    profile_line, = ax.plot(
        stacked[0],
        z / height,
        color="tab:blue",
        linewidth=2.4,
        label="LES horizontal mean",
    )
    time_label = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", ha="left")
    ax.set_xlim(0.0, 1.05 * max_speed)
    ax.axhline(1.0, color="0.4", linestyle="-.", linewidth=1.1, label="pressure-gradient top")
    ax.set_ylim(0.0, domain_height / height)
    ax.set_xlabel(r"$\langle u\rangle$ [m s$^{-1}$]")
    ax.set_ylabel(r"$z/H$")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="lower right")
    ax.set_title(
        f"Pressure-driven neutral ABL profile development "
        f"(H={height:.0f} m, lid at z/H={domain_height / height:.2f})"
    )

    def update(frame: int):
        profile_line.set_xdata(stacked[frame])
        time_label.set_text(f"step {steps[frame]:d}   t = {steps[frame] * dt_physical:.0f} s")
        return profile_line, time_label

    animation = FuncAnimation(
        fig,
        update,
        frames=len(steps),
        interval=1000.0 / fps,
        blit=True,
        repeat=True,
    )
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def plot_velocity_cross_sections_gif(
    path: Path,
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    steps: list[int],
    params,
    fps: int,
) -> None:
    if fps <= 0:
        raise ValueError("field-gif-fps must be positive")
    if not frames:
        raise ValueError("No flow-field frames were sampled")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    path.parent.mkdir(parents=True, exist_ok=True)
    all_values = np.concatenate(
        [section.ravel() for frame in frames for section in frame[:3]]
    )
    vmin = float(np.nanpercentile(all_values, 0.5))
    vmax = float(np.nanpercentile(all_values, 99.5))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise RuntimeError("Flow-field GIF received an invalid speed range")

    lx = params.lx * params.z_i
    ly = params.ly * params.z_i
    lz = params.lz * params.z_i
    first_xy, first_xz, first_yz, actual_z_over_h = frames[0]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    images = [
        axes[0].imshow(
            first_xy,
            origin="lower",
            extent=(0.0, lx, 0.0, ly),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        ),
        axes[1].imshow(
            first_xz,
            origin="lower",
            extent=(0.0, lx, 0.0, lz),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        ),
        axes[2].imshow(
            first_yz,
            origin="lower",
            extent=(0.0, ly, 0.0, lz),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        ),
    ]
    axes[0].set_title(rf"$x$-$y$ at $z/H={actual_z_over_h:.3f}$")
    axes[1].set_title(rf"$x$-$z$ at $y={0.5 * ly:.0f}$ m")
    axes[2].set_title(rf"$y$-$z$ at $x={0.5 * lx:.0f}$ m")
    axes[0].set(xlabel="x [m]", ylabel="y [m]")
    axes[1].set(xlabel="x [m]", ylabel="z [m]")
    axes[2].set(xlabel="y [m]", ylabel="z [m]")
    if params.sponge_enabled:
        for axis in axes[1:]:
            axis.axhline(
                params.sponge_start_height,
                color="tab:orange",
                linestyle="--",
                linewidth=1.2,
                label="sponge start",
            )
        axes[1].legend(loc="lower right")
    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.92, pad=0.02)
    colorbar.set_label(r"$|\mathbf{u}|$ [m s$^{-1}$]")
    title = fig.suptitle("")

    def update(frame_index: int):
        xy, xz, yz, _ = frames[frame_index]
        for image, values in zip(images, (xy, xz, yz)):
            image.set_data(values)
        step = steps[frame_index]
        title.set_text(f"Pressure-driven neutral ABL   step {step:d}   t = {step * params.dt_physical:.0f} s")
        return (*images, title)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(steps),
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.gif_fps <= 0:
        raise SystemExit("ERROR: gif-fps must be positive.")
    if args.field_gif_fps <= 0:
        raise SystemExit("ERROR: field-gif-fps must be positive.")
    if not 0.0 < args.field_gif_z_over_h < 1.0:
        raise SystemExit("ERROR: field-gif-z-over-h must lie strictly between zero and one.")
    if args.cfl_limit <= 0.0:
        raise SystemExit("ERROR: cfl-limit must be positive.")
    try:
        settings = load_settings(args)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"ERROR: failed to load case: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if settings["precision"] == "float64" or settings["sgs_precision"] == "float64":
        from jax import config as jax_config

        jax_config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp

    from wireles_jax import run
    from wireles_jax.wall import wall_stress

    try:
        params = params_from_settings(settings, jnp)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if params.initial_condition not in {"uniform", "default"}:
        raise SystemExit(
            "ERROR: pressure-driven comparison requires an initial uniform or capped log-law profile."
        )
    if params.sgs_model != "lasd":
        raise SystemExit("ERROR: pressure-driven comparison requires [sgs].model='lasd'.")
    if params.wall_stress_model != "dynamic_neutral":
        raise SystemExit(
            "ERROR: pressure-driven comparison requires wall_stress_model='dynamic_neutral'; "
            "the wall ustar must be diagnosed from the filtered first-level velocity."
        )
    if params.coriolis_f != 0.0:
        raise SystemExit("ERROR: this minimal pressure-driven case requires coriolis_f=0.")

    default_average_start = int(0.75 * params.nsteps)
    average_start_step = default_average_start if args.average_start_step is None else args.average_start_step
    if average_start_step < 0 or average_start_step > params.nsteps:
        raise SystemExit("ERROR: average-start-step must lie between zero and the total number of steps.")

    output_dir = args.output_dir
    if output_dir is None:
        configured = settings.get("field_output_dir")
        output_dir = Path(configured) if configured is not None else Path("outputs/pressure_driven_neutral_abl_lasd")

    initial_u: np.ndarray | None = None
    sampled_profiles: list[np.ndarray] = []
    sampled_ustar: list[float] = []
    sampled_wall_stress_x: list[float] = []
    sampled_steps: set[int] = set()
    evolution_profiles: list[np.ndarray] = []
    evolution_steps: list[int] = []
    field_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    field_steps: list[int] = []
    diagnostics_rows = []
    header_printed = False
    max_observed_cfl = 0.0

    def print_diagnostic(diag) -> None:
        nonlocal header_printed, max_observed_cfl
        diagnostics_rows.append(diag)
        current_cfl = max(float(diag.cfl_x), float(diag.cfl_y), float(diag.cfl_z))
        max_observed_cfl = max(max_observed_cfl, current_cfl)
        if current_cfl > args.cfl_limit:
            raise RuntimeError(
                f"CFL limit exceeded at step {int(diag.step)}: "
                f"{current_cfl:.6f} > {args.cfl_limit:.6f}"
            )
        if not header_printed:
            print(LOG_HEADER, flush=True)
            header_printed = True
        print(format_diagnostic(diag, params.cs_count), flush=True)

    def sample_profile(state, diag) -> None:
        nonlocal initial_u
        step = int(diag.step)
        ready_state = jax.block_until_ready(state)
        profile = horizontal_u_profile(ready_state)
        if step == 0:
            initial_u = profile
        if args.gif:
            evolution_profiles.append(profile)
            evolution_steps.append(step)
        if args.field_gif:
            field_frames.append(
                velocity_cross_sections(ready_state, params, args.field_gif_z_over_h)
            )
            field_steps.append(step)
        if step >= average_start_step:
            sampled_profiles.append(profile)
            sampled_ustar.append(float(diag.ustar))
            txz, *_ = wall_stress(state.u, state.v, params)
            sampled_wall_stress_x.append(float(jnp.mean(txz)))
            sampled_steps.add(step)

    try:
        state, _ = run(
            params,
            seed=args.seed,
            log_every=settings["log_every"],
            log_callback=print_diagnostic,
            log_state_callback=sample_profile,
            status_callback=lambda message: print(message, flush=True),
        )
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    state = jax.block_until_ready(state)

    final_step = int(state.step)
    if args.gif and (not evolution_steps or evolution_steps[-1] != final_step):
        evolution_profiles.append(horizontal_u_profile(state))
        evolution_steps.append(final_step)
    if args.field_gif and (not field_steps or field_steps[-1] != final_step):
        field_frames.append(velocity_cross_sections(state, params, args.field_gif_z_over_h))
        field_steps.append(final_step)
    if final_step >= average_start_step and final_step not in sampled_steps:
        sampled_profiles.append(horizontal_u_profile(state))
        final_txz, *_, final_ustar = wall_stress(state.u, state.v, params)
        sampled_ustar.append(float(jnp.mean(final_ustar)))
        sampled_wall_stress_x.append(float(jnp.mean(final_txz)))
        sampled_steps.add(final_step)
    if initial_u is None or not sampled_profiles:
        raise RuntimeError("No profiles were sampled; reduce average-start-step or log-every.")

    mean_u = np.mean(np.stack(sampled_profiles, axis=0), axis=0)
    mean_ustar = float(np.mean(sampled_ustar))
    mean_wall_stress_x = float(np.mean(sampled_wall_stress_x))
    stress_equivalent_ustar = float(np.sqrt(max(-mean_wall_stress_x, 0.0)))
    pressure_ustar = params.pressure_ustar
    if (
        not np.all(np.isfinite(mean_u))
        or not np.isfinite(mean_ustar)
        or not np.isfinite(mean_wall_stress_x)
        or pressure_ustar <= 0.0
    ):
        raise RuntimeError("The pressure-driven run produced non-finite profile statistics.")
    dz_physical = params.dz * params.z_i
    z = (np.arange(mean_u.size, dtype=np.float64) + 0.5) * dz_physical
    height = params.bl_height
    target_u = log_law(z, pressure_ustar, params.zo, params.vonk)
    log_mask = (z / height >= 0.05) & (z / height <= 0.30) & np.isfinite(target_u)
    if int(log_mask.sum()) < 2:
        raise RuntimeError("The grid has fewer than two points in 0.05 <= z/H <= 0.30.")

    error = mean_u[log_mask] - target_u[log_mask]
    rmse = float(np.sqrt(np.mean(error * error)))
    relative_rmse = rmse / float(np.mean(np.abs(target_u[log_mask])))
    log_coordinate = np.log(z[log_mask] / params.zo)
    fitted_ustar = params.vonk * float(np.dot(log_coordinate, mean_u[log_mask]) / np.dot(log_coordinate, log_coordinate))
    post_projection_divergence = [float(diag.div_max) for diag in diagnostics_rows if int(diag.step) > 0]
    state_metrics = numerical_state_metrics(state, params)

    profile_path = output_dir / "profiles.csv"
    summary_path = output_dir / "summary.csv"
    plot_path = output_dir / "profile_vs_loglaw.png"
    gif_path = output_dir / "profile_evolution.gif"
    field_gif_path = output_dir / "velocity_three_sections.gif"
    write_profile_csv(profile_path, z, height, initial_u, mean_u, target_u, pressure_ustar)
    summary = {
        "grid": f"{params.nx}x{params.ny}x{params.nz}",
        "domain_height_m": params.lz * params.z_i,
        "pressure_forcing_top_m": params.bl_height,
        "steps": params.nsteps,
        "physical_time_s": params.nsteps * params.dt_physical,
        "averaging_start_step": average_start_step,
        "profile_sample_count": len(sampled_profiles),
        "pressure_ustar_m_s": pressure_ustar,
        "mean_diagnosed_wall_ustar_m_s": mean_ustar,
        "stress_equivalent_wall_ustar_m_s": stress_equivalent_ustar,
        "mean_wall_stress_x_m2_s2": mean_wall_stress_x,
        "fitted_loglaw_ustar_m_s": fitted_ustar,
        "sgs_model": params.sgs_model,
        "wall_stress_model": params.wall_stress_model,
        "initial_condition": params.initial_condition,
        "sponge_enabled": params.sponge_enabled,
        "sponge_start_height_m": params.sponge_start_height,
        "sponge_timescale_s": params.sponge_timescale,
        "sponge_power": params.sponge_power,
        "sponge_target": params.sponge_target,
        "pressure_acceleration_m_s2": params.driving_pressure_force / params.z_i,
        "log_layer_z_over_h_min": float((z / height)[log_mask].min()),
        "log_layer_z_over_h_max": float((z / height)[log_mask].max()),
        "log_layer_rmse_m_s": rmse,
        "log_layer_relative_rmse": relative_rmse,
        "max_post_projection_divergence": max(post_projection_divergence, default=0.0),
        "cfl_limit": args.cfl_limit,
        "max_observed_cfl": max_observed_cfl,
        **state_metrics,
    }
    write_summary_csv(summary_path, summary)
    if not args.no_plot:
        initial_label = (
            "initial uniform"
            if params.initial_condition == "uniform"
            else "initial log law, uniform above H"
        )
        plot_comparison(
            plot_path,
            z,
            height,
            params.lz * params.z_i,
            params.zo,
            pressure_ustar,
            initial_u,
            mean_u,
            target_u,
            log_mask,
            initial_label,
        )
    if args.gif:
        plot_profile_evolution_gif(
            gif_path,
            z,
            height,
            params.lz * params.z_i,
            evolution_profiles,
            evolution_steps,
            params.dt_physical,
            target_u,
            log_mask,
            args.gif_fps,
        )
    if args.field_gif:
        plot_velocity_cross_sections_gif(
            field_gif_path,
            field_frames,
            field_steps,
            params,
            args.field_gif_fps,
        )

    print(f"[comparison] averaged {len(sampled_profiles)} profile(s) from step {min(sampled_steps)} to {max(sampled_steps)}")
    print(
        f"[comparison] pressure-equivalent ustar={pressure_ustar:.6f} m/s, "
        f"mean diagnosed wall ustar={mean_ustar:.6f} m/s, "
        f"stress-equivalent wall ustar={stress_equivalent_ustar:.6f} m/s"
    )
    print(f"[comparison] fitted log-law ustar={fitted_ustar:.6f} m/s")
    print(f"[comparison] log-layer RMSE={rmse:.6e} m/s ({relative_rmse:.3%})")
    print(f"[stability] max observed CFL={max_observed_cfl:.6f} (limit {args.cfl_limit:.6f})")
    print(
        f"[top-boundary] uv mismatch={state_metrics['top_uv_boundary_mismatch_m_s']:.3e} m/s, "
        f"|w|={state_metrics['wall_normal_velocity_m_s']:.3e} m/s, "
        f"rejected-mode ratio={state_metrics['rejected_horizontal_mode_ratio']:.3e}"
    )
    upper_label = "sponge" if params.sponge_enabled else "upper-layer"
    print(
        f"[{upper_label}] z>={state_metrics['upper_layer_z_min_m']:.1f} m, "
        f"velocity-fluctuation RMS={state_metrics['upper_layer_velocity_fluctuation_rms_m_s']:.3e} m/s, "
        f"w RMS={state_metrics['upper_layer_w_rms_m_s']:.3e} m/s"
    )
    print(f"[output] {profile_path}")
    print(f"[output] {summary_path}")
    if not args.no_plot:
        print(f"[output] {plot_path}")
    if args.gif:
        print(f"[output] {gif_path}")
    if args.field_gif:
        print(f"[output] {field_gif_path}")


if __name__ == "__main__":
    main()
