#!/usr/bin/env python3
"""Run the zero-yaw WiRE-01 case with a concurrent precursor on ``src/wireles``."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
PRESSURE_SOURCE = Path(
    os.environ.get(
        "WIRELES_SPECTRAL_FD_SOURCE",
        ROOT / "external" / "bw1000_benchmark",
    )
)
for source in (ROOT, SOURCE, PRESSURE_SOURCE):
    if source.exists() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from spectral_fd import runtime_from_initialized_jax  # noqa: E402

from benchmark.LinPorteAgel2019.case import (  # noqa: E402
    PAPER_CASE,
    local_thrust_coefficient,
)
from wireles.domain import (  # noqa: E402
    Accepted,
    AcceptedClock,
    AddressableField,
    Candidate,
    Cell,
    DistributionSpec,
    EqualZSlab,
    MeshAxis,
    MeshTopology,
    PassiveScalarConcentration,
    ScaleSystem,
    UniformGrid,
    VerticalBoundary,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from wireles.effects import (  # noqa: E402
    SideBySideStreamLauncher,
    ZSlabCheckpointLayout,
    load_boussinesq_checkpoint,
)
from wireles.integrators import (  # noqa: E402
    AB2Config,
    ConcurrentPrecursorState,
    cold_start_boussinesq,
    serial_pair,
    step_boussinesq,
    step_concurrent_boussinesq_precursor,
)
from wireles.interpreters.jax_zslab import (  # noqa: E402
    ZFaceFieldContext,
    build_zslab_interpreter,
)
from wireles.operators import VelocityVector, project  # noqa: E402
from wireles.physics import (  # noqa: E402
    BoussinesqFields,
    BoussinesqModel,
    BoussinesqVectorField,
    ConcurrentPrecursorFringe,
    ConservativeAdvection,
    ConservativeScalarAdvection,
    DryFlowModel,
    FilteredNeutralLogWall,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LasdAcceptedStepEvent,
    KinematicPressureGradient,
    NeutralLogWall,
    NoBuoyancy,
    NoActuatorDisk,
    NoFringe,
    NoRayleighDamping,
    NoRotation,
    PureThrustActuatorDisk,
    ScalarFluxBoundary,
    WindTunnelBoussinesqVectorField,
    WindTunnelModel,
)
from wireles.pressure import build_spectral_fd_pressure_adapter  # noqa: E402


HERE = Path(__file__).resolve().parent
NON_YAW_THRUST_COEFFICIENT = 0.78
DT_SECONDS = 2.5e-4
FRINGE_START_X = 5.4
FRINGE_RELAXATION_TIME = 0.1
INITIAL_TURBULENCE_CORRELATION_LENGTH = 0.05


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=PAPER_CASE.grid[0])
    parser.add_argument("--ny", type=int, default=PAPER_CASE.grid[1])
    parser.add_argument("--nz", type=int, default=PAPER_CASE.grid[2])
    parser.add_argument("--dt", type=float, default=DT_SECONDS)
    parser.add_argument("--precursor-steps", type=int, default=19_000)
    parser.add_argument(
        "--precursor-restart",
        type=Path,
        help="developed standalone Boussinesq/LASD precursor checkpoint",
    )
    parser.add_argument("--concurrent-steps", type=int, default=19_000)
    parser.add_argument("--lasd-update-interval", type=int, default=10)
    parser.add_argument("--sample-start", type=int, default=4_000)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument(
        "--initial-perturbation",
        choices=("correlated", "white"),
        default="correlated",
    )
    parser.add_argument(
        "--wall-model",
        choices=("filtered", "local"),
        default="filtered",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--method", choices=("transpose", "spike"), default="spike")
    parser.add_argument(
        "--execution",
        choices=("auto", "serial", "threads", "cuda-streams"),
        default="auto",
        help="one-device paired launch policy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "LinPorteAgel2019_nonyawed",
    )
    parser.add_argument(
        "--flow-gif",
        action="store_true",
        help="record orthogonal instantaneous velocity slices",
    )
    parser.add_argument("--gif-start", type=int, default=9_000)
    parser.add_argument("--gif-every", type=int, default=50)
    parser.add_argument("--gif-frames", type=int, default=100)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.nx, args.ny, args.nz = 32, 16, 16
        args.precursor_steps = 8
        args.concurrent_steps = 16
        args.sample_start = 0
        args.sample_every = 1
        args.log_every = 4
        args.gif_start = 0
        args.gif_every = 1
        args.gif_frames = 16
    if min(args.nx, args.ny, args.nz) <= 1:
        parser.error("all grid dimensions must exceed one")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.precursor_steps < 0 or args.concurrent_steps <= 0:
        parser.error(
            "precursor steps must be nonnegative and concurrent steps positive"
        )
    if args.precursor_steps == 0 and args.precursor_restart is None:
        parser.error("zero precursor steps requires --precursor-restart")
    if args.lasd_update_interval <= 0:
        parser.error("--lasd-update-interval must be positive")
    if args.sample_start < 0 or args.sample_start >= args.concurrent_steps:
        parser.error("--sample-start must be within the concurrent run")
    if min(args.sample_every, args.log_every) <= 0:
        parser.error("sampling and logging intervals must be positive")
    if args.gif_start < 0 or args.gif_start >= args.concurrent_steps:
        parser.error("--gif-start must be within the concurrent run")
    if min(args.gif_every, args.gif_frames, args.gif_fps) <= 0:
        parser.error("GIF interval, frame count, and frame rate must be positive")
    return args


def _initial_velocity(
    *,
    args: argparse.Namespace,
    physical_grid: UniformGrid,
    decomposition: EqualZSlab,
    scales: ScaleSystem,
    algebra,
    pressure_solver,
    config: AB2Config,
) -> VelocityVector:
    dtype = getattr(jnp, args.dtype)
    shape = (1, physical_grid.nz, physical_grid.ny, physical_grid.nx)
    z = (jnp.arange(physical_grid.nz, dtype=dtype) + 0.5) * physical_grid.dz
    log_velocity = (
        PAPER_CASE.friction_velocity
        / 0.4
        * jnp.log(z / PAPER_CASE.roughness_length)
    )
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 3)

    def correlated_noise(key):
        noise = jax.random.normal(key, shape, dtype=dtype)
        spectrum = jnp.fft.rfftn(noise, axes=(1, 2, 3))
        kz = 2.0 * jnp.pi * jnp.fft.fftfreq(
            physical_grid.nz,
            d=physical_grid.dz,
        )
        ky = 2.0 * jnp.pi * jnp.fft.fftfreq(
            physical_grid.ny,
            d=physical_grid.dy,
        )
        kx = 2.0 * jnp.pi * jnp.fft.rfftfreq(
            physical_grid.nx,
            d=physical_grid.dx,
        )
        length = INITIAL_TURBULENCE_CORRELATION_LENGTH
        low_pass = jnp.exp(
            -0.5
            * length**2
            * (
                kz[:, None, None] ** 2
                + ky[None, :, None] ** 2
                + kx[None, None, :] ** 2
            )
        )
        filtered = jnp.fft.irfftn(
            spectrum * low_pass[None],
            s=(physical_grid.nz, physical_grid.ny, physical_grid.nx),
            axes=(1, 2, 3),
        ).astype(dtype)
        filtered = filtered - jnp.mean(filtered, axis=(2, 3), keepdims=True)
        rms = jnp.sqrt(jnp.mean(filtered * filtered))
        return filtered / jnp.maximum(rms, jnp.finfo(dtype).tiny)

    noise = (
        [correlated_noise(key) for key in keys]
        if args.initial_perturbation == "correlated"
        else [jax.random.normal(key, shape, dtype=dtype) for key in keys]
    )
    fluctuation = PAPER_CASE.hub_turbulence_intensity * PAPER_CASE.hub_velocity
    taper = jnp.sin(jnp.pi * z / physical_grid.lz) ** 0.25
    u = (
        log_velocity[None, :, None, None]
        + fluctuation
        * taper[None, :, None, None]
        * noise[0]
    )
    v = (
        fluctuation
        * taper[None, :, None, None]
        * noise[1]
    )
    z_faces = (jnp.arange(physical_grid.nz, dtype=dtype) + 1.0) * physical_grid.dz
    face_taper = jnp.sin(jnp.pi * z_faces / physical_grid.lz)
    w = (
        fluctuation
        * face_taper[None, :, None, None]
        * noise[2]
    ).at[:, -1].set(0.0)
    candidate = VelocityVector(
        AddressableField(
            XVelocity,
            Cell,
            decomposition.regions(Cell),
            Candidate,
            scales.to_execution_velocity(u),
        ),
        AddressableField(
            YVelocity,
            Cell,
            decomposition.regions(Cell),
            Candidate,
            scales.to_execution_velocity(v),
        ),
        ZFaceFieldContext(
            AddressableField(
                VerticalVelocity,
                ZFace,
                decomposition.regions(ZFace),
                Candidate,
                scales.to_execution_velocity(w),
            ),
            jnp.zeros((physical_grid.ny, physical_grid.nx), dtype=dtype),
        ),
    )
    return project(
        candidate,
        dt=config.dt,
        normal_boundary=VerticalBoundary(0.0, 0.0),
        algebra=algebra,
        pressure_solver=pressure_solver,
    ).velocity


def _w_at_cells(velocity: VelocityVector):
    upper = velocity.z.owned.payload
    lower_plane = jnp.broadcast_to(
        jnp.asarray(velocity.z.lower_boundary, dtype=upper.dtype),
        upper.shape[2:],
    )
    lower = jnp.concatenate((lower_plane[None, None], upper[:, :-1]), axis=1)
    return 0.5 * (lower + upper)


def _hub_plane(values, physical_grid: UniformGrid):
    fractional = PAPER_CASE.hub_height / physical_grid.dz - 0.5
    lower = max(0, min(int(math.floor(fractional)), physical_grid.nz - 2))
    weight = fractional - lower
    return (1.0 - weight) * values[0, lower] + weight * values[0, lower + 1]


class HubStatistics:
    def __init__(self, physical_grid: UniformGrid, scales: ScaleSystem) -> None:
        self.physical_grid = physical_grid
        self.scales = scales
        self.count = 0
        self.sums: dict[str, list] = {}
        self.squares: dict[str, list] = {}

    def sample(self, paired: ConcurrentPrecursorState) -> None:
        for name, state in (("precursor", paired.precursor), ("main", paired.main)):
            velocity = state.fields.velocity
            planes = (
                _hub_plane(
                    self.scales.from_execution_velocity(velocity.x.payload),
                    self.physical_grid,
                ),
                _hub_plane(
                    self.scales.from_execution_velocity(velocity.y.payload),
                    self.physical_grid,
                ),
                _hub_plane(
                    self.scales.from_execution_velocity(_w_at_cells(velocity)),
                    self.physical_grid,
                ),
            )
            if name not in self.sums:
                self.sums[name] = [jnp.zeros_like(value) for value in planes]
                self.squares[name] = [jnp.zeros_like(value) for value in planes]
            self.sums[name] = [
                total + value
                for total, value in zip(self.sums[name], planes, strict=True)
            ]
            self.squares[name] = [
                total + value * value
                for total, value in zip(self.squares[name], planes, strict=True)
            ]
        self.count += 1

    def finish(self) -> dict[str, dict[str, np.ndarray]]:
        if self.count == 0:
            raise RuntimeError("no hub-height statistics were sampled")
        result = {}
        for name in ("precursor", "main"):
            mean = [value / self.count for value in self.sums[name]]
            variance = [
                jnp.maximum(square / self.count - average * average, 0.0)
                for square, average in zip(
                    self.squares[name], mean, strict=True
                )
            ]
            result[name] = {
                "mean_u": np.asarray(jax.device_get(mean[0]), dtype=np.float64),
                "tke": np.asarray(
                    jax.device_get(0.5 * sum(variance)),
                    dtype=np.float64,
                ),
                "streamwise_variance": np.asarray(
                    jax.device_get(variance[0]), dtype=np.float64
                ),
            }
        return result


def _linear_plane(values, coordinate: float, spacing: float, axis: int):
    fractional = coordinate / spacing - 0.5
    lower = max(0, min(int(math.floor(fractional)), values.shape[axis] - 2))
    weight = fractional - lower
    first = jnp.take(values, lower, axis=axis)
    second = jnp.take(values, lower + 1, axis=axis)
    return (1.0 - weight) * first + weight * second


def _capture_flow_frame(
    state,
    *,
    paired_step: int,
    physical_grid: UniformGrid,
    scales: ScaleSystem,
    dt: float,
) -> dict[str, np.ndarray | float]:
    velocity = state.fields.velocity if hasattr(state, "fields") else state.velocity
    u = scales.from_execution_velocity(velocity.x.payload)[0]
    v = scales.from_execution_velocity(velocity.y.payload)[0]
    w = scales.from_execution_velocity(_w_at_cells(velocity))[0]
    station_x = PAPER_CASE.turbine_x + 6.0 * PAPER_CASE.rotor_diameter
    values = {
        "time_seconds": paired_step * dt,
        "xy_u": _linear_plane(u, PAPER_CASE.hub_height, physical_grid.dz, 0),
        "xy_v": _linear_plane(v, PAPER_CASE.hub_height, physical_grid.dz, 0),
        "xz_u": _linear_plane(u, PAPER_CASE.turbine_y, physical_grid.dy, 1),
        "xz_w": _linear_plane(w, PAPER_CASE.turbine_y, physical_grid.dy, 1),
        "yz_u": _linear_plane(u, station_x, physical_grid.dx, 2),
        "yz_v": _linear_plane(v, station_x, physical_grid.dx, 2),
        "yz_w": _linear_plane(w, station_x, physical_grid.dx, 2),
    }
    return {
        name: float(value)
        if name == "time_seconds"
        else np.asarray(jax.device_get(value), dtype=np.float32)
        for name, value in values.items()
    }


def _capture_concurrent_flow_frame(
    paired: ConcurrentPrecursorState,
    *,
    paired_step: int,
    physical_grid: UniformGrid,
    scales: ScaleSystem,
    dt: float,
) -> dict[str, np.ndarray | float]:
    main = _capture_flow_frame(
        paired.main,
        paired_step=paired_step,
        physical_grid=physical_grid,
        scales=scales,
        dt=dt,
    )
    precursor = _capture_flow_frame(
        paired.precursor,
        paired_step=paired_step,
        physical_grid=physical_grid,
        scales=scales,
        dt=dt,
    )

    def fringe_cross_plane(state):
        velocity = state.fields.velocity
        u = scales.from_execution_velocity(velocity.x.payload)[0]
        v = scales.from_execution_velocity(velocity.y.payload)[0]
        w = scales.from_execution_velocity(_w_at_cells(velocity))[0]
        station = 0.5 * (FRINGE_START_X + physical_grid.lx)
        return (
            _linear_plane(u, station, physical_grid.dx, 2),
            _linear_plane(v, station, physical_grid.dx, 2),
            _linear_plane(w, station, physical_grid.dx, 2),
        )

    main_fringe = fringe_cross_plane(paired.main)
    precursor_fringe = fringe_cross_plane(paired.precursor)
    frame = dict(main)
    frame.update(
        {
            f"precursor_{name}": value
            for name, value in precursor.items()
            if name != "time_seconds"
        }
    )
    for prefix, values in (
        ("fringe", main_fringe),
        ("precursor_fringe", precursor_fringe),
    ):
        for component, value in zip(("u", "v", "w"), values, strict=True):
            frame[f"{prefix}_yz_{component}"] = np.asarray(
                jax.device_get(value),
                dtype=np.float32,
            )
    return frame


def _write_flow_gif(
    path: Path,
    frames: list[dict[str, np.ndarray | float]],
    physical_grid: UniformGrid,
    *,
    fps: int,
    show_turbine: bool = True,
) -> None:
    if not frames:
        raise RuntimeError("no flow frames were captured")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    diameter = PAPER_CASE.rotor_diameter
    x = (
        (np.arange(physical_grid.nx) + 0.5) * physical_grid.dx
        - PAPER_CASE.turbine_x
    ) / diameter
    y = (
        (np.arange(physical_grid.ny) + 0.5) * physical_grid.dy
        - PAPER_CASE.turbine_y
    ) / diameter
    z = (np.arange(physical_grid.nz) + 0.5) * physical_grid.dz / diameter
    x_window = (x >= -4.0) & (x <= 18.0)
    x_indices = np.flatnonzero(x_window)
    x_extent = (x[x_indices[0]], x[x_indices[-1]])
    y_extent = (y[0], y[-1])
    z_extent = (z[0], z[-1])
    initial = frames[0]
    uh = PAPER_CASE.hub_velocity

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    images = (
        axes[0].imshow(
            initial["xy_u"][:, x_indices] / uh,
            origin="lower",
            aspect="auto",
            extent=(*x_extent, *y_extent),
            cmap="turbo",
            vmin=0.35,
            vmax=1.25,
        ),
        axes[1].imshow(
            initial["xz_u"][:, x_indices] / uh,
            origin="lower",
            aspect="auto",
            extent=(*x_extent, *z_extent),
            cmap="turbo",
            vmin=0.35,
            vmax=1.25,
        ),
        axes[2].imshow(
            initial["yz_u"] / uh,
            origin="lower",
            aspect="auto",
            extent=(*y_extent, *z_extent),
            cmap="turbo",
            vmin=0.35,
            vmax=1.25,
        ),
    )
    xy_x_stride, xy_y_stride = 8, 4
    xz_x_stride, xz_z_stride = 8, 2
    yz_y_stride, yz_z_stride = 4, 2
    xy_x = x[x_indices][::xy_x_stride]
    xy_y = y[::xy_y_stride]
    xz_x = x[x_indices][::xz_x_stride]
    xz_z = z[::xz_z_stride]
    yz_y = y[::yz_y_stride]
    yz_z = z[::yz_z_stride]
    quivers = (
        axes[0].quiver(
            xy_x,
            xy_y,
            initial["xy_u"][::xy_y_stride, x_indices][..., ::xy_x_stride] / uh,
            initial["xy_v"][::xy_y_stride, x_indices][..., ::xy_x_stride] / uh,
            color="white",
            alpha=0.7,
            scale=20.0,
            width=0.002,
        ),
        axes[1].quiver(
            xz_x,
            xz_z,
            initial["xz_u"][::xz_z_stride, x_indices][..., ::xz_x_stride] / uh,
            initial["xz_w"][::xz_z_stride, x_indices][..., ::xz_x_stride] / uh,
            color="white",
            alpha=0.7,
            scale=20.0,
            width=0.002,
        ),
        axes[2].quiver(
            yz_y,
            yz_z,
            initial["yz_v"][::yz_z_stride, ::yz_y_stride] / uh,
            initial["yz_w"][::yz_z_stride, ::yz_y_stride] / uh,
            color="white",
            alpha=0.8,
            scale=2.0,
            width=0.003,
        ),
    )
    hub_over_d = PAPER_CASE.hub_height / diameter
    if show_turbine:
        axes[0].plot((0.0, 0.0), (-0.5, 0.5), color="black", lw=2.0)
        axes[1].plot(
            (0.0, 0.0),
            (hub_over_d - 0.5, hub_over_d + 0.5),
            color="black",
            lw=2.0,
        )
        rotor = plt.Circle(
            (0.0, hub_over_d),
            0.5,
            fill=False,
            color="black",
            lw=2.0,
        )
        axes[2].add_patch(rotor)
    reference = "t" if show_turbine else "ref"
    axes[0].set(
        title="top: hub-height x-y",
        xlabel=f"(x-x_{reference})/D",
        ylabel=f"(y-y_{reference})/D",
    )
    axes[1].set(
        title=("side: turbine-centre x-z" if show_turbine else "side: centre x-z"),
        xlabel=f"(x-x_{reference})/D",
        ylabel="z/D",
    )
    axes[2].set(
        title=(
            "cross-stream: y-z at x/D=6"
            if show_turbine
            else "cross-stream: y-z reference plane"
        ),
        xlabel=f"(y-y_{reference})/D",
        ylabel="z/D",
    )
    for axis in axes:
        axis.grid(False)
    colorbar = figure.colorbar(images[0], ax=axes, shrink=0.84, pad=0.02)
    colorbar.set_label(r"instantaneous $u/u_h$")
    title = figure.suptitle("")

    def update(index: int):
        frame = frames[index]
        images[0].set_data(frame["xy_u"][:, x_indices] / uh)
        images[1].set_data(frame["xz_u"][:, x_indices] / uh)
        images[2].set_data(frame["yz_u"] / uh)
        quivers[0].set_UVC(
            frame["xy_u"][::xy_y_stride, x_indices][..., ::xy_x_stride] / uh,
            frame["xy_v"][::xy_y_stride, x_indices][..., ::xy_x_stride] / uh,
        )
        quivers[1].set_UVC(
            frame["xz_u"][::xz_z_stride, x_indices][..., ::xz_x_stride] / uh,
            frame["xz_w"][::xz_z_stride, x_indices][..., ::xz_x_stride] / uh,
        )
        quivers[2].set_UVC(
            frame["yz_v"][::yz_z_stride, ::yz_y_stride] / uh,
            frame["yz_w"][::yz_z_stride, ::yz_y_stride] / uh,
        )
        prefix = "zero-yaw pure-thrust ADM" if show_turbine else "neutral precursor"
        title.set_text(f"{prefix}, time = {frame['time_seconds']:.3f} s")
        return (*images, *quivers, title)

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=1000.0 / fps,
        blit=False,
    )
    movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(figure)


def _write_fringe_gif(
    path: Path,
    frames: list[dict[str, np.ndarray | float]],
    physical_grid: UniformGrid,
    *,
    fps: int,
) -> None:
    """Animate the main-minus-precursor field through the fringe zone."""
    if not frames:
        raise RuntimeError("no fringe frames were captured")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    diameter = PAPER_CASE.rotor_diameter
    x = (
        (np.arange(physical_grid.nx) + 0.5) * physical_grid.dx
        - PAPER_CASE.turbine_x
    ) / diameter
    y = (
        (np.arange(physical_grid.ny) + 0.5) * physical_grid.dy
        - PAPER_CASE.turbine_y
    ) / diameter
    z = (np.arange(physical_grid.nz) + 0.5) * physical_grid.dz / diameter
    fringe_start = (FRINGE_START_X - PAPER_CASE.turbine_x) / diameter
    fringe_station = (
        0.5 * (FRINGE_START_X + physical_grid.lx) - PAPER_CASE.turbine_x
    ) / diameter

    def differences(frame):
        return (
            (frame["xy_u"] - frame["precursor_xy_u"]) / PAPER_CASE.hub_velocity,
            (frame["xz_u"] - frame["precursor_xz_u"]) / PAPER_CASE.hub_velocity,
            (
                frame["fringe_yz_u"] - frame["precursor_fringe_yz_u"]
            )
            / PAPER_CASE.hub_velocity,
        )

    all_differences = [value for frame in frames for value in differences(frame)]
    color_limit = max(
        0.05,
        min(
            0.6,
            float(
                np.percentile(
                    np.concatenate([np.ravel(v) for v in all_differences]),
                    99.5,
                )
            ),
        ),
    )
    color_limit = max(
        color_limit,
        min(
            0.6,
            float(
                np.percentile(
                    np.concatenate([np.ravel(-v) for v in all_differences]),
                    99.5,
                )
            ),
        ),
    )
    initial = differences(frames[0])
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    images = (
        axes[0].imshow(
            initial[0],
            origin="lower",
            aspect="auto",
            extent=(x[0], x[-1], y[0], y[-1]),
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
        ),
        axes[1].imshow(
            initial[1],
            origin="lower",
            aspect="auto",
            extent=(x[0], x[-1], z[0], z[-1]),
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
        ),
        axes[2].imshow(
            initial[2],
            origin="lower",
            aspect="auto",
            extent=(y[0], y[-1], z[0], z[-1]),
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
        ),
    )
    for axis in axes[:2]:
        axis.axvspan(fringe_start, x[-1], color="gold", alpha=0.12)
        axis.axvline(fringe_start, color="gold", lw=1.8, ls="--")
    axes[0].plot((0.0, 0.0), (-0.5, 0.5), color="black", lw=2.0)
    hub_over_d = PAPER_CASE.hub_height / diameter
    axes[1].plot(
        (0.0, 0.0),
        (hub_over_d - 0.5, hub_over_d + 0.5),
        color="black",
        lw=2.0,
    )
    axes[0].set(
        title="hub-height x-y difference",
        xlabel="(x-x_t)/D",
        ylabel="(y-y_t)/D",
    )
    axes[1].set(
        title="turbine-centre x-z difference",
        xlabel="(x-x_t)/D",
        ylabel="z/D",
    )
    axes[2].set(
        title=rf"fringe y-z difference at $(x-x_t)/D={fringe_station:.1f}$",
        xlabel="(y-y_t)/D",
        ylabel="z/D",
    )
    colorbar = figure.colorbar(images[0], ax=axes, shrink=0.84, pad=0.02)
    colorbar.set_label(r"$(u_{main}-u_{precursor})/u_h$")
    title = figure.suptitle("")

    def update(index: int):
        values = differences(frames[index])
        for image, value in zip(images, values, strict=True):
            image.set_data(value)
        title.set_text(
            "concurrent precursor fringe diagnostic, "
            f"time = {frames[index]['time_seconds']:.3f} s"
        )
        return (*images, title)

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=1000.0 / fps,
        blit=False,
    )
    movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(figure)


def _write_fringe_diagnostic(
    path: Path,
    frames: list[dict[str, np.ndarray | float]],
    physical_grid: UniformGrid,
) -> dict[str, float]:
    """Plot the imposed fringe mask and the evolving hub-plane mismatch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = (np.arange(physical_grid.nx) + 0.5) * physical_grid.dx
    half_width = 0.5 * (physical_grid.lx - FRINGE_START_X)

    def cinf_step(coordinate):
        safe = np.clip(coordinate, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        exponent = np.clip(
            1.0 / (1.0 - safe) - 1.0 / safe,
            -700.0,
            700.0,
        )
        interior = 1.0 / (1.0 + np.exp(-exponent))
        return np.where(
            coordinate <= 0.0,
            0.0,
            np.where(coordinate >= 1.0, 1.0, interior),
        )

    mask = cinf_step((x - FRINGE_START_X) / half_width) * cinf_step(
        (physical_grid.lx - x) / half_width
    )
    mismatch = np.asarray(
        [
            np.sqrt(
                np.mean(
                    (
                        (frame["xy_u"] - frame["precursor_xy_u"])
                        / PAPER_CASE.hub_velocity
                    )
                    ** 2,
                    axis=0,
                )
            )
            for frame in frames
        ]
    )
    times = np.asarray([frame["time_seconds"] for frame in frames])
    x_over_d = (x - PAPER_CASE.turbine_x) / PAPER_CASE.rotor_diameter
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 6.5), sharex=True)
    axes[0].plot(x_over_d, mask, color="black")
    axes[0].fill_between(x_over_d, 0.0, mask, color="gold", alpha=0.3)
    axes[0].set_ylabel("fringe mask")
    axes[0].set_ylim(-0.03, 1.03)
    image = axes[1].pcolormesh(
        x_over_d,
        times,
        mismatch,
        shading="auto",
        cmap="magma",
    )
    axes[1].set(
        xlabel="(x-x_t)/D",
        ylabel="paired time [s]",
    )
    figure.colorbar(
        image,
        ax=axes[1],
        label=r"RMS$_y[(u_{main}-u_{precursor})/u_h]$",
    )
    fringe_start_over_d = (
        FRINGE_START_X - PAPER_CASE.turbine_x
    ) / PAPER_CASE.rotor_diameter
    for axis in axes:
        axis.axvline(fringe_start_over_d, color="goldenrod", ls="--")
        axis.grid(alpha=0.2)
    figure.suptitle("Concurrent-precursor fringe activation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)

    final = mismatch[-1]
    pre_turbine = x < (
        PAPER_CASE.turbine_x - PAPER_CASE.rotor_diameter
    )
    wake_to_fringe = (x >= PAPER_CASE.turbine_x) & (x < FRINGE_START_X)
    fringe = x >= FRINGE_START_X
    peak_index = int(np.argmax(mask))
    final_difference = (
        frames[-1]["xy_u"] - frames[-1]["precursor_xy_u"]
    ) / PAPER_CASE.hub_velocity
    return {
        "mask_peak": float(mask[peak_index]),
        "mask_peak_x_m": float(x[peak_index]),
        "final_pre_turbine_rms_delta_u_over_uh": float(
            np.sqrt(np.mean(final[pre_turbine] ** 2))
        ),
        "final_wake_to_fringe_rms_delta_u_over_uh": float(
            np.sqrt(np.mean(final[wake_to_fringe] ** 2))
        ),
        "final_fringe_rms_delta_u_over_uh": float(
            np.sqrt(np.mean(final[fringe] ** 2))
        ),
        "final_peak_mask_delta_u_over_uh": float(final[peak_index]),
        "final_periodic_seam_jump_rms_delta_u_over_uh": float(
            np.sqrt(
                np.mean(
                    (final_difference[:, -1] - final_difference[:, 0]) ** 2
                )
            )
        ),
        "maximum_recorded_hub_rms_delta_u_over_uh": float(np.max(mismatch)),
    }


def _sample_x(plane: np.ndarray, coordinate: float, dx: float) -> np.ndarray:
    fractional = coordinate / dx - 0.5
    lower = int(math.floor(fractional)) % plane.shape[-1]
    upper = (lower + 1) % plane.shape[-1]
    weight = fractional - math.floor(fractional)
    return (1.0 - weight) * plane[:, lower] + weight * plane[:, upper]


def _write_profiles(
    output: Path,
    statistics: dict[str, dict[str, np.ndarray]],
    physical_grid: UniformGrid,
) -> list[dict[str, float]]:
    y = (np.arange(physical_grid.ny) + 0.5) * physical_grid.dy
    y_over_d = (y - PAPER_CASE.turbine_y) / PAPER_CASE.rotor_diameter
    station_profiles = []
    for x_over_d in PAPER_CASE.profile_x_over_d:
        coordinate = PAPER_CASE.turbine_x + x_over_d * PAPER_CASE.rotor_diameter
        values = {name: {} for name in ("precursor", "main")}
        for name in values:
            for quantity in ("mean_u", "tke", "streamwise_variance"):
                values[name][quantity] = _sample_x(
                    statistics[name][quantity], coordinate, physical_grid.dx
                )
        station_profiles.append((x_over_d, values))

    columns = ["y_over_d"]
    quantities = (
        "precursor_u_over_uh",
        "wake_u_over_uh",
        "velocity_deficit",
        "precursor_tke_over_uh2",
        "wake_tke_over_uh2",
        "added_tke_over_uh2",
        "precursor_iu",
        "wake_iu",
    )
    for x_over_d, _ in station_profiles:
        columns.extend(f"{quantity}_{x_over_d:g}d" for quantity in quantities)
    with (output / "wake_profiles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, transverse in enumerate(y_over_d):
            row = {"y_over_d": float(transverse)}
            for x_over_d, values in station_profiles:
                suffix = f"{x_over_d:g}d"
                precursor_u = values["precursor"]["mean_u"][index]
                main_u = values["main"]["mean_u"][index]
                precursor_tke = values["precursor"]["tke"][index]
                main_tke = values["main"]["tke"][index]
                row.update(
                    {
                        f"precursor_u_over_uh_{suffix}": precursor_u
                        / PAPER_CASE.hub_velocity,
                        f"wake_u_over_uh_{suffix}": main_u
                        / PAPER_CASE.hub_velocity,
                        f"velocity_deficit_{suffix}": (precursor_u - main_u)
                        / PAPER_CASE.hub_velocity,
                        f"precursor_tke_over_uh2_{suffix}": precursor_tke
                        / PAPER_CASE.hub_velocity**2,
                        f"wake_tke_over_uh2_{suffix}": main_tke
                        / PAPER_CASE.hub_velocity**2,
                        f"added_tke_over_uh2_{suffix}": (main_tke - precursor_tke)
                        / PAPER_CASE.hub_velocity**2,
                        f"precursor_iu_{suffix}": math.sqrt(
                            values["precursor"]["streamwise_variance"][index]
                        )
                        / PAPER_CASE.hub_velocity,
                        f"wake_iu_{suffix}": math.sqrt(
                            values["main"]["streamwise_variance"][index]
                        )
                        / PAPER_CASE.hub_velocity,
                    }
                )
            writer.writerow(row)

    rotor_window = np.abs(y_over_d) <= 1.5
    summary = []
    for x_over_d, values in station_profiles:
        deficit = (
            values["precursor"]["mean_u"] - values["main"]["mean_u"]
        ) / PAPER_CASE.hub_velocity
        added_tke = (
            values["main"]["tke"] - values["precursor"]["tke"]
        ) / PAPER_CASE.hub_velocity**2
        summary.append(
            {
                "x_over_d": x_over_d,
                "maximum_velocity_deficit": float(np.max(deficit[rotor_window])),
                "mean_velocity_deficit": float(np.mean(deficit[rotor_window])),
                "maximum_added_tke_over_uh2": float(
                    np.max(added_tke[rotor_window])
                ),
                "mean_added_tke_over_uh2": float(
                    np.mean(added_tke[rotor_window])
                ),
            }
        )
    _plot_profiles(output / "wake_velocity_tke.png", y_over_d, station_profiles)
    np.savez_compressed(
        output / "hub_statistics.npz",
        y_over_d=y_over_d,
        **{
            f"{name}_{quantity}": values
            for name, fields in statistics.items()
            for quantity, values in fields.items()
        },
    )
    return summary


def _plot_profiles(path: Path, y_over_d: np.ndarray, profiles) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, len(profiles), figsize=(12.0, 6.5), sharey=True)
    for column, (x_over_d, values) in enumerate(profiles):
        axes[0, column].plot(
            values["precursor"]["mean_u"] / PAPER_CASE.hub_velocity,
            y_over_d,
            color="0.55",
            label="precursor",
        )
        axes[0, column].plot(
            values["main"]["mean_u"] / PAPER_CASE.hub_velocity,
            y_over_d,
            color="black",
            label="wake",
        )
        axes[1, column].plot(
            values["precursor"]["tke"] / PAPER_CASE.hub_velocity**2,
            y_over_d,
            color="0.55",
        )
        axes[1, column].plot(
            values["main"]["tke"] / PAPER_CASE.hub_velocity**2,
            y_over_d,
            color="black",
        )
        axes[0, column].set_title(rf"$x/D={x_over_d:g}$")
        axes[0, column].set_xlabel(r"$\overline{u}/u_h$")
        axes[1, column].set_xlabel(r"$k/u_h^2$")
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
            axes[row, column].set_ylim(-1.5, 1.5)
    axes[0, 0].set_ylabel(r"$y/D$")
    axes[1, 0].set_ylabel(r"$y/D$")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Zero-yaw pure-thrust ADM: concurrent precursor vs wake")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _block_pair(state: ConcurrentPrecursorState) -> None:
    state.precursor.fields.velocity.x.payload.block_until_ready()
    state.main.fields.velocity.x.payload.block_until_ready()


def _resolved_execution(args: argparse.Namespace) -> str:
    if args.execution != "auto":
        return args.execution
    return "cuda-streams" if jax.default_backend() == "gpu" else "serial"


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if jax.device_count() != 1:
        raise RuntimeError(
            "this side-by-side benchmark requires exactly one JAX device"
        )
    dtype = getattr(jnp, args.dtype)
    physical_grid = UniformGrid(
        args.nx,
        args.ny,
        args.nz,
        *PAPER_CASE.domain,
    )
    scales = ScaleSystem(PAPER_CASE.boundary_layer_height, PAPER_CASE.hub_velocity)
    grid = scales.to_execution_grid(physical_grid)
    decomposition = EqualZSlab(
        grid,
        MeshTopology((MeshAxis("z", 1),)),
        DistributionSpec.z_slab(),
    )
    algebra = build_zslab_interpreter(decomposition, addressable_shards=(0,))
    runtime = runtime_from_initialized_jax(jax)
    precursor_pressure = build_spectral_fd_pressure_adapter(
        decomposition,
        addressable_shards=(0,),
        runtime=runtime,
        dtype=args.dtype,
        method=args.method,
    )
    main_pressure = build_spectral_fd_pressure_adapter(
        decomposition,
        addressable_shards=(0,),
        runtime=runtime,
        dtype=args.dtype,
        method=args.method,
    )
    pressure_acceleration = (
        PAPER_CASE.friction_velocity**2 / PAPER_CASE.boundary_layer_height
    )
    wall = (
        FilteredNeutralLogWall(
            scales.to_execution_length(PAPER_CASE.roughness_length)
        )
        if args.wall_model == "filtered"
        else NeutralLogWall(
            scales.to_execution_length(PAPER_CASE.roughness_length)
        )
    )
    momentum_sgs = LagrangianScaleDependentDynamic(
        update_interval=args.lasd_update_interval
    )
    scalar_sgs = LagrangianScaleDependentScalarFlux()
    boussinesq_model = BoussinesqModel(
        DryFlowModel(
            ConservativeAdvection(),
            KinematicPressureGradient(
                scales.to_execution_acceleration(pressure_acceleration)
            ),
            wall,
            momentum_sgs,
            NoRotation(),
        ),
        ConservativeScalarAdvection(),
        scalar_sgs,
        NoBuoyancy(),
        NoRayleighDamping(),
        ScalarFluxBoundary(),
    )
    base_vector_field = BoussinesqVectorField(algebra, boussinesq_model)
    precursor_vector_field = WindTunnelBoussinesqVectorField(
        algebra,
        base_vector_field,
        WindTunnelModel(NoActuatorDisk(), NoFringe()),
    )
    main_vector_field = WindTunnelBoussinesqVectorField(
        algebra,
        base_vector_field,
        WindTunnelModel(
            PureThrustActuatorDisk(
                scales.to_execution_length(PAPER_CASE.turbine_x),
                scales.to_execution_length(PAPER_CASE.turbine_y),
                scales.to_execution_length(PAPER_CASE.hub_height),
                scales.to_execution_length(PAPER_CASE.rotor_diameter),
                local_thrust_coefficient(NON_YAW_THRUST_COEFFICIENT),
                scales.to_execution_length(2.0 * physical_grid.dx),
                scales.to_execution_length(
                    2.0 * max(physical_grid.dy, physical_grid.dz)
                ),
                yaw_degrees=0.0,
                filtered_velocity_correction=True,
            ),
            ConcurrentPrecursorFringe(
                scales.to_execution_length(FRINGE_START_X),
                scales.to_execution_time(FRINGE_RELAXATION_TIME),
            ),
        ),
    )
    config = AB2Config(scales.to_execution_time(args.dt))
    closure_fingerprint = momentum_sgs.fingerprint + "|" + scalar_sgs.fingerprint
    if args.precursor_restart is None:
        velocity = _initial_velocity(
            args=args,
            physical_grid=physical_grid,
            decomposition=decomposition,
            scales=scales,
            algebra=algebra,
            pressure_solver=precursor_pressure,
            config=config,
        )
        scalar = AddressableField(
            PassiveScalarConcentration,
            Cell,
            decomposition.regions(Cell),
            Accepted,
            jnp.zeros_like(velocity.x.payload),
        )
        fields = BoussinesqFields(velocity, scalar)
        fields = algebra.initialize_lasd_closure(fields, boussinesq_model)
        precursor = cold_start_boussinesq(
            fields,
            clock=AcceptedClock(0.0, 0),
            config=config,
        )
    else:
        precursor = load_boussinesq_checkpoint(
            args.precursor_restart,
            layout=ZSlabCheckpointLayout(
                decomposition,
                (0,),
                jnp.asarray,
            ),
            config=config,
            closure_fingerprint=closure_fingerprint,
        )
    precursor_closure_event = LasdAcceptedStepEvent(
        algebra,
        boussinesq_model,
        config.dt,
    )
    main_closure_event = LasdAcceptedStepEvent(
        algebra,
        boussinesq_model,
        config.dt,
    )
    precursor_initial_step = precursor.clock.step
    boundary = lambda _clock, _environment: VerticalBoundary(0.0, 0.0)
    started = time.perf_counter()
    for iteration in range(args.precursor_steps):
        result = step_boussinesq(
            precursor,
            config=config,
            environment=None,
            vector_field=precursor_vector_field,
            normal_boundary=boundary,
            algebra=algebra,
            pressure_solver=precursor_pressure,
            closure_event=precursor_closure_event,
        )
        precursor = result.state
        if (
            (iteration + 1) % args.log_every == 0
            or iteration + 1 == args.precursor_steps
        ):
            precursor.fields.velocity.x.payload.block_until_ready()
            print(
                f"precursor step={iteration + 1}/{args.precursor_steps} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    main = cold_start_boussinesq(
        precursor.fields,
        clock=precursor.clock,
        config=config,
    )
    paired = ConcurrentPrecursorState(precursor, main)
    execution = _resolved_execution(args)
    launcher_context = None
    launch_pair = serial_pair
    if execution in ("threads", "cuda-streams"):
        launcher_context = SideBySideStreamLauncher(
            execution_streams=execution == "cuda-streams"
        )
        launch_pair = launcher_context
    statistics = HubStatistics(physical_grid, scales)
    flow_frames: list[dict[str, np.ndarray | float]] = []
    maximum_cfl = 0.0
    maximum_divergence = 0.0
    paired_started = time.perf_counter()
    try:
        for iteration in range(args.concurrent_steps):
            result = step_concurrent_boussinesq_precursor(
                paired,
                config=config,
                precursor_vector_field=precursor_vector_field,
                main_vector_field=main_vector_field,
                normal_boundary=boundary,
                algebra=algebra,
                precursor_pressure_solver=precursor_pressure,
                main_pressure_solver=main_pressure,
                precursor_closure_event=precursor_closure_event,
                main_closure_event=main_closure_event,
                launch_pair=launch_pair,
            )
            paired = result.state
            final = iteration + 1 == args.concurrent_steps
            paired_step = iteration + 1
            if (
                args.flow_gif
                and paired_step >= args.gif_start
                and (paired_step - args.gif_start) % args.gif_every == 0
                and len(flow_frames) < args.gif_frames
            ):
                flow_frames.append(
                    _capture_concurrent_flow_frame(
                        paired,
                        paired_step=paired_step,
                        physical_grid=physical_grid,
                        scales=scales,
                        dt=args.dt,
                    )
                )
            if iteration >= args.sample_start and (
                (iteration - args.sample_start) % args.sample_every == 0 or final
            ):
                statistics.sample(paired)
                velocity = paired.main.fields.velocity
                cfl = max(
                    float(config.dt * jnp.max(jnp.abs(velocity.x.payload)) / grid.dx),
                    float(config.dt * jnp.max(jnp.abs(velocity.y.payload)) / grid.dy),
                    float(
                        config.dt
                        * jnp.max(jnp.abs(velocity.z.owned.payload))
                        / grid.dz
                    ),
                )
                divergence = float(
                    jnp.max(
                        jnp.abs(
                            result.diagnostic.main.projection.divergence.payload
                        )
                    )
                )
                maximum_cfl = max(maximum_cfl, cfl)
                maximum_divergence = max(maximum_divergence, divergence)
            if (iteration + 1) % args.log_every == 0 or final:
                _block_pair(paired)
                print(
                    f"paired step={iteration + 1}/{args.concurrent_steps} "
                    f"mode={execution} CFL={maximum_cfl:.3f} "
                    f"elapsed={time.perf_counter() - paired_started:.1f}s",
                    flush=True,
                )
    finally:
        if launcher_context is not None:
            launcher_context.close()
    _block_pair(paired)
    paired_runtime = time.perf_counter() - paired_started
    fields = statistics.finish()
    wake = _write_profiles(args.output_dir, fields, physical_grid)
    flow_gif = None
    fringe_gif = None
    fringe_diagnostic = None
    fringe_metrics = None
    if args.flow_gif:
        flow_gif = args.output_dir / "flow_three_plane.gif"
        fringe_gif = args.output_dir / "fringe_three_plane.gif"
        _write_flow_gif(
            flow_gif,
            flow_frames,
            physical_grid,
            fps=args.gif_fps,
        )
        _write_fringe_gif(
            fringe_gif,
            flow_frames,
            physical_grid,
            fps=args.gif_fps,
        )
        fringe_diagnostic = args.output_dir / "fringe_diagnostic.png"
        fringe_metrics = _write_fringe_diagnostic(
            fringe_diagnostic,
            flow_frames,
            physical_grid,
        )
        np.savez_compressed(
            args.output_dir / "flow_slices.npz",
            **{
                name: np.asarray([frame[name] for frame in flow_frames])
                for name in flow_frames[0]
            },
        )
    summary = {
        "paper": "Lin and Porte-Agel (2019), doi:10.3390/en12234574",
        "case": "zero-yaw WiRE-01 first-stage benchmark",
        "model": "uniform pure-thrust actuator disk",
        "thrust_coefficient": NON_YAW_THRUST_COEFFICIENT,
        "local_thrust_coefficient": local_thrust_coefficient(
            NON_YAW_THRUST_COEFFICIENT
        ),
        "actuator_normal_smoothing_width_m": 2.0 * physical_grid.dx,
        "actuator_transverse_smoothing_width_m": 2.0
        * max(physical_grid.dy, physical_grid.dz),
        "actuator_filtered_velocity_correction": True,
        "grid": [args.nx, args.ny, args.nz],
        "precursor_steps": args.precursor_steps,
        "precursor_initial_step": precursor_initial_step,
        "precursor_paired_start_step": precursor_initial_step + args.precursor_steps,
        "precursor_restart": (
            None
            if args.precursor_restart is None
            else str(args.precursor_restart)
        ),
        "concurrent_steps": args.concurrent_steps,
        "concurrent_physical_time_seconds": args.concurrent_steps * args.dt,
        "sgs_model": "lasd",
        "lasd_update_interval": args.lasd_update_interval,
        "samples": statistics.count,
        "execution": execution,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "paired_runtime_seconds": paired_runtime,
        "paired_domain_steps_per_second": 2.0 * args.concurrent_steps / paired_runtime,
        "maximum_cfl": maximum_cfl,
        "maximum_lasd_cfl": maximum_cfl * args.lasd_update_interval,
        "maximum_divergence": maximum_divergence,
        "flow_gif": None if flow_gif is None else str(flow_gif),
        "fringe_gif": None if fringe_gif is None else str(fringe_gif),
        "fringe_diagnostic": (
            None if fringe_diagnostic is None else str(fringe_diagnostic)
        ),
        "fringe": fringe_metrics,
        "flow_gif_frames": len(flow_frames),
        "wake": wake,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.output_dir / "configuration.json").write_text(
        json.dumps(
            {
                **vars(args),
                "output_dir": str(args.output_dir),
                "precursor_restart": (
                    None
                    if args.precursor_restart is None
                    else str(args.precursor_restart)
                ),
                "domain_m": PAPER_CASE.domain,
                "dt_seconds": args.dt,
                "fringe_start_x_m": FRINGE_START_X,
                "fringe_relaxation_time_s": FRINGE_RELAXATION_TIME,
                "side_by_side_resident": True,
                "precursor_sample_time": "t_n",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"results: {args.output_dir}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
