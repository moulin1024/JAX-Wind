#!/usr/bin/env python3
"""Run and diagnose the Lin--Porte-Agel neutral precursor by itself."""

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
        "JAXWIND_SPECTRAL_FD_SOURCE",
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

from benchmark.LinPorteAgel2019.case import PAPER_CASE  # noqa: E402
from benchmark.LinPorteAgel2019.run_nonyawed import (  # noqa: E402
    DT_SECONDS,
    _capture_flow_frame,
    _hub_plane,
    _initial_velocity,
    _w_at_cells,
    _write_flow_gif,
)
from jaxwind.domain import (  # noqa: E402
    Accepted,
    AcceptedClock,
    AddressableField,
    Cell,
    DistributionSpec,
    EqualZSlab,
    MeshAxis,
    MeshTopology,
    ScaleSystem,
    UniformGrid,
    VerticalBoundary,
    PassiveScalarConcentration,
)
from jaxwind.effects import (  # noqa: E402
    ZSlabCheckpointLayout,
    load_boussinesq_checkpoint,
    save_boussinesq_checkpoint,
)
from jaxwind.integrators import (  # noqa: E402
    AB2Config,
    cold_start_boussinesq,
    step_boussinesq,
)
from jaxwind.interpreters.jax_zslab import build_zslab_interpreter  # noqa: E402
from jaxwind.physics import (  # noqa: E402
    BoussinesqFields,
    BoussinesqModel,
    BoussinesqVectorField,
    ConservativeAdvection,
    ConservativeScalarAdvection,
    DryFlowModel,
    FilteredNeutralLogWall,
    IdentityClosureEvent,
    KinematicPressureGradient,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LasdAcceptedStepEvent,
    NoBuoyancy,
    NeutralLogWall,
    NoRayleighDamping,
    NoRotation,
    ScalarFluxBoundary,
    StaticSmagorinsky,
    StaticSmagorinskyScalarFlux,
)
from jaxwind.pressure import build_spectral_fd_pressure_adapter  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=PAPER_CASE.grid[0])
    parser.add_argument("--ny", type=int, default=PAPER_CASE.grid[1])
    parser.add_argument("--nz", type=int, default=PAPER_CASE.grid[2])
    parser.add_argument("--dt", type=float, default=DT_SECONDS)
    parser.add_argument("--steps", type=int, default=19_000)
    parser.add_argument("--sample-start", type=int, default=12_000)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--method", choices=("transpose", "spike"), default="spike")
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
    parser.add_argument("--sgs", choices=("lasd", "static"), default="lasd")
    parser.add_argument("--lasd-update-interval", type=int, default=10)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--flow-gif", action="store_true")
    parser.add_argument("--gif-start", type=int, default=14_000)
    parser.add_argument("--gif-every", type=int, default=50)
    parser.add_argument("--gif-frames", type=int, default=100)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "LinPorteAgel2019_precursor",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.nx, args.ny, args.nz = 32, 16, 16
        args.steps = 16
        args.sample_start = 0
        args.sample_every = 1
        args.log_every = 4
        args.gif_start = 1
        args.gif_every = 1
        args.gif_frames = 16
    if min(args.nx, args.ny, args.nz) <= 1:
        parser.error("all grid dimensions must exceed one")
    if args.dt <= 0.0 or args.steps <= 0:
        parser.error("time step and step count must be positive")
    if args.sample_start < 0 or args.sample_start >= args.steps:
        parser.error("--sample-start must be within the run")
    if min(
        args.sample_every,
        args.log_every,
        args.lasd_update_interval,
    ) <= 0:
        parser.error("sampling and logging intervals must be positive")
    if args.flow_gif and not 0 <= args.gif_start < args.steps:
        parser.error("--gif-start must be within the run")
    if min(args.gif_every, args.gif_frames, args.gif_fps) <= 0:
        parser.error("GIF controls must be positive")
    return args


class PrecursorStatistics:
    def __init__(self, grid: UniformGrid, scales: ScaleSystem) -> None:
        self.grid = grid
        self.scales = scales
        self.count = 0
        self.sum_u = np.zeros(grid.nz)
        self.sum_v = np.zeros(grid.nz)
        self.sum_w = np.zeros(grid.nz)
        self.sum_u2 = np.zeros(grid.nz)
        self.sum_v2 = np.zeros(grid.nz)
        self.sum_w2 = np.zeros(grid.nz)
        self.spectrum_x = np.zeros(grid.nx // 2 + 1)
        self.spectrum_y = np.zeros(grid.ny // 2 + 1)

    def sample(self, state) -> None:
        velocity = state.fields.velocity
        u = self.scales.from_execution_velocity(velocity.x.payload)
        v = self.scales.from_execution_velocity(velocity.y.payload)
        w = self.scales.from_execution_velocity(_w_at_cells(velocity))
        profiles = jnp.stack(
            tuple(
                jnp.mean(value, axis=(0, 2, 3))
                for value in (u, v, w, u * u, v * v, w * w)
            )
        )
        values = np.asarray(jax.device_get(profiles), dtype=np.float64)
        for target, value in zip(
            (
                self.sum_u,
                self.sum_v,
                self.sum_w,
                self.sum_u2,
                self.sum_v2,
                self.sum_w2,
            ),
            values,
            strict=True,
        ):
            target += value

        hub = _hub_plane(u, self.grid)
        hub = hub - jnp.mean(hub)
        coefficient_x = jnp.fft.rfft(hub, axis=1) / self.grid.nx
        coefficient_y = jnp.fft.rfft(hub, axis=0) / self.grid.ny
        energy_x = jnp.mean(jnp.abs(coefficient_x) ** 2, axis=0)
        energy_y = jnp.mean(jnp.abs(coefficient_y) ** 2, axis=1)
        factor_x = jnp.full_like(energy_x, 2.0).at[0].set(1.0)
        factor_y = jnp.full_like(energy_y, 2.0).at[0].set(1.0)
        if self.grid.nx % 2 == 0:
            factor_x = factor_x.at[-1].set(1.0)
        if self.grid.ny % 2 == 0:
            factor_y = factor_y.at[-1].set(1.0)
        self.spectrum_x += np.asarray(
            jax.device_get(energy_x * factor_x), dtype=np.float64
        )
        self.spectrum_y += np.asarray(
            jax.device_get(energy_y * factor_y), dtype=np.float64
        )
        self.count += 1

    def finish(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("no precursor samples were collected")
        mean_u = self.sum_u / self.count
        mean_v = self.sum_v / self.count
        mean_w = self.sum_w / self.count
        var_u = np.maximum(self.sum_u2 / self.count - mean_u**2, 0.0)
        var_v = np.maximum(self.sum_v2 / self.count - mean_v**2, 0.0)
        var_w = np.maximum(self.sum_w2 / self.count - mean_w**2, 0.0)
        return {
            "mean_u": mean_u,
            "mean_v": mean_v,
            "mean_w": mean_w,
            "var_u": var_u,
            "var_v": var_v,
            "var_w": var_w,
            "tke": 0.5 * (var_u + var_v + var_w),
            "spectrum_x": self.spectrum_x / self.count,
            "spectrum_y": self.spectrum_y / self.count,
        }


def _linear_profile(profile: np.ndarray, coordinate: float, dz: float) -> float:
    fractional = coordinate / dz - 0.5
    lower = max(0, min(int(math.floor(fractional)), profile.size - 2))
    weight = fractional - lower
    return float((1.0 - weight) * profile[lower] + weight * profile[lower + 1])


def _filtered_wall_velocity(u0, v0, grid: UniformGrid):
    values = jnp.stack((u0, v0))
    spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
    x_mode = jnp.arange(grid.nx // 2 + 1)
    y_mode = jnp.abs(jnp.fft.fftfreq(grid.ny) * grid.ny)
    filter_width = 1.5 * 2.0
    cutoff_x = jnp.floor(grid.nx / (2.0 * filter_width))
    cutoff_y = jnp.floor(grid.ny / (2.0 * filter_width))
    keep = (y_mode[:, None] < cutoff_y) & (x_mode[None, :] < cutoff_x)
    filtered = jnp.fft.irfftn(
        spectrum * keep,
        s=(grid.ny, grid.nx),
        axes=(-2, -1),
    )
    return filtered[0], filtered[1]


def _instantaneous_hub_and_ustar(
    state,
    grid: UniformGrid,
    scales: ScaleSystem,
    wall_model: str,
):
    velocity = state.fields.velocity
    u = scales.from_execution_velocity(velocity.x.payload)
    v = scales.from_execution_velocity(velocity.y.payload)
    hub = _hub_plane(u, grid)
    hub_mean = float(jnp.mean(hub))
    hub_iu = float(jnp.std(hub) / jnp.maximum(jnp.abs(jnp.mean(hub)), 1.0e-12))
    u0 = u[:, 0]
    v0 = v[:, 0]
    if wall_model == "filtered":
        u0, v0 = _filtered_wall_velocity(u0, v0, grid)
    speed = jnp.hypot(u0, v0)
    local_ustar = (
        0.4
        * speed
        / math.log(0.5 * grid.dz / PAPER_CASE.roughness_length)
    )
    mean_ustar = float(jnp.mean(local_ustar))
    stress_ustar = float(jnp.sqrt(jnp.mean(local_ustar * local_ustar)))
    return hub_mean, hub_iu, mean_ustar, stress_ustar


def _write_outputs(
    output: Path,
    fields: dict[str, np.ndarray],
    grid: UniformGrid,
) -> dict[str, float]:
    z = (np.arange(grid.nz) + 0.5) * grid.dz
    iu = np.sqrt(fields["var_u"]) / PAPER_CASE.hub_velocity
    tke = fields["tke"] / PAPER_CASE.hub_velocity**2
    with (output / "precursor_profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("z_m", "z_over_h", "mean_u_over_uh", "iu", "tke_over_uh2")
        )
        writer.writerows(
            zip(
                z,
                z / PAPER_CASE.boundary_layer_height,
                fields["mean_u"] / PAPER_CASE.hub_velocity,
                iu,
                tke,
                strict=True,
            )
        )

    kx = 2.0 * np.pi * np.fft.rfftfreq(grid.nx, d=grid.dx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(grid.ny, d=grid.dy)
    np.savez_compressed(
        output / "precursor_statistics.npz",
        z=z,
        kx=kx,
        ky=ky,
        **fields,
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), sharey=True)
    axes[0].plot(fields["mean_u"] / PAPER_CASE.hub_velocity, z)
    axes[0].axvline(1.0, color="0.5", ls="--")
    axes[0].set(xlabel=r"$\overline{u}/u_h$", ylabel="z [m]")
    axes[1].plot(iu, z)
    axes[1].axvline(PAPER_CASE.hub_turbulence_intensity, color="0.5", ls="--")
    axes[1].set(xlabel=r"$I_u$")
    axes[2].plot(tke, z)
    axes[2].set(xlabel=r"$k/u_h^2$")
    for axis in axes:
        axis.axhline(PAPER_CASE.hub_height, color="0.5", ls=":")
        axis.grid(alpha=0.25)
    figure.suptitle("Neutral precursor statistics")
    figure.tight_layout()
    figure.savefig(output / "precursor_profiles.png", dpi=180)
    plt.close(figure)

    hub_mean = _linear_profile(fields["mean_u"], PAPER_CASE.hub_height, grid.dz)
    hub_iu = _linear_profile(iu, PAPER_CASE.hub_height, grid.dz)
    hub_tke = _linear_profile(tke, PAPER_CASE.hub_height, grid.dz)
    x_cut = kx >= 0.5 * np.max(kx)
    y_cut = ky >= 0.5 * np.max(ky)
    high_x = float(
        np.sum(fields["spectrum_x"][x_cut]) / np.sum(fields["spectrum_x"])
    )
    high_y = float(
        np.sum(fields["spectrum_y"][y_cut]) / np.sum(fields["spectrum_y"])
    )
    return {
        "hub_mean_velocity_m_s": hub_mean,
        "hub_mean_velocity_relative_error": (
            hub_mean - PAPER_CASE.hub_velocity
        )
        / PAPER_CASE.hub_velocity,
        "hub_streamwise_turbulence_intensity": hub_iu,
        "hub_tke_over_uh2": hub_tke,
        "high_wavenumber_energy_fraction_x": high_x,
        "high_wavenumber_energy_fraction_y": high_y,
    }


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if jax.device_count() != 1:
        raise RuntimeError("the precursor diagnostic requires exactly one JAX device")
    physical_grid = UniformGrid(args.nx, args.ny, args.nz, *PAPER_CASE.domain)
    scales = ScaleSystem(PAPER_CASE.boundary_layer_height, PAPER_CASE.hub_velocity)
    grid = scales.to_execution_grid(physical_grid)
    decomposition = EqualZSlab(
        grid,
        MeshTopology((MeshAxis("z", 1),)),
        DistributionSpec.z_slab(),
    )
    algebra = build_zslab_interpreter(decomposition, addressable_shards=(0,))
    pressure_solver = build_spectral_fd_pressure_adapter(
        decomposition,
        addressable_shards=(0,),
        runtime=runtime_from_initialized_jax(jax),
        dtype=args.dtype,
        method=args.method,
    )
    wall = (
        FilteredNeutralLogWall(
            scales.to_execution_length(PAPER_CASE.roughness_length)
        )
        if args.wall_model == "filtered"
        else NeutralLogWall(scales.to_execution_length(PAPER_CASE.roughness_length))
    )
    momentum_sgs = (
        LagrangianScaleDependentDynamic(
            update_interval=args.lasd_update_interval
        )
        if args.sgs == "lasd"
        else StaticSmagorinsky(0.16)
    )
    scalar_sgs = (
        LagrangianScaleDependentScalarFlux()
        if args.sgs == "lasd"
        else StaticSmagorinskyScalarFlux()
    )
    model = BoussinesqModel(
        DryFlowModel(
            ConservativeAdvection(),
            KinematicPressureGradient(
                scales.to_execution_acceleration(
                    PAPER_CASE.friction_velocity**2
                    / PAPER_CASE.boundary_layer_height
                )
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
    config = AB2Config(scales.to_execution_time(args.dt))
    if args.restart is None:
        velocity = _initial_velocity(
            args=args,
            physical_grid=physical_grid,
            decomposition=decomposition,
            scales=scales,
            algebra=algebra,
            pressure_solver=pressure_solver,
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
        if args.sgs == "lasd":
            fields = algebra.initialize_lasd_closure(fields, model)
        state = cold_start_boussinesq(
            fields,
            clock=AcceptedClock(0.0, 0),
            config=config,
        )
    else:
        closure_fingerprint = (
            momentum_sgs.fingerprint + "|" + scalar_sgs.fingerprint
            if args.sgs == "lasd"
            else None
        )
        state = load_boussinesq_checkpoint(
            args.restart,
            layout=ZSlabCheckpointLayout(
                decomposition,
                (0,),
                jnp.asarray,
            ),
            config=config,
            closure_fingerprint=closure_fingerprint,
        )
    vector_field = BoussinesqVectorField(algebra, model)
    closure_event = (
        LasdAcceptedStepEvent(algebra, model, config.dt)
        if args.sgs == "lasd"
        else IdentityClosureEvent()
    )
    boundary = lambda _clock, _environment: VerticalBoundary(0.0, 0.0)
    statistics = PrecursorStatistics(physical_grid, scales)
    frames: list[dict[str, np.ndarray | float]] = []
    history = []
    maximum_cfl = 0.0
    maximum_lasd_cfl = 0.0
    maximum_divergence = 0.0
    started = time.perf_counter()
    for iteration in range(args.steps):
        result = step_boussinesq(
            state,
            config=config,
            environment=None,
            vector_field=vector_field,
            normal_boundary=boundary,
            algebra=algebra,
            pressure_solver=pressure_solver,
            closure_event=closure_event,
        )
        state = result.state
        run_step = iteration + 1
        accepted_step = state.clock.step
        if iteration >= args.sample_start and (
            (iteration - args.sample_start) % args.sample_every == 0
            or run_step == args.steps
        ):
            statistics.sample(state)
            velocity = state.fields.velocity
            maximum_cfl = max(
                maximum_cfl,
                float(config.dt * jnp.max(jnp.abs(velocity.x.payload)) / grid.dx),
                float(config.dt * jnp.max(jnp.abs(velocity.y.payload)) / grid.dy),
                float(
                    config.dt
                    * jnp.max(jnp.abs(velocity.z.owned.payload))
                    / grid.dz
                ),
            )
            if args.sgs == "lasd":
                maximum_lasd_cfl = max(
                    maximum_lasd_cfl,
                    maximum_cfl * args.lasd_update_interval,
                )
            maximum_divergence = max(
                maximum_divergence,
                float(
                    jnp.max(jnp.abs(result.diagnostic.projection.divergence.payload))
                ),
            )
        if (
            args.flow_gif
            and run_step >= args.gif_start
            and (run_step - args.gif_start) % args.gif_every == 0
            and len(frames) < args.gif_frames
        ):
            frames.append(
                _capture_flow_frame(
                    state,
                    paired_step=accepted_step,
                    physical_grid=physical_grid,
                    scales=scales,
                    dt=args.dt,
                )
            )
        if run_step % args.log_every == 0 or run_step == args.steps:
            (
                hub_mean,
                hub_iu,
                mean_ustar,
                stress_ustar,
            ) = _instantaneous_hub_and_ustar(
                state,
                physical_grid,
                scales,
                args.wall_model,
            )
            row = {
                "step": accepted_step,
                "time_seconds": accepted_step * args.dt,
                "hub_mean_velocity_m_s": hub_mean,
                "hub_spatial_iu": hub_iu,
                "mean_local_friction_velocity_m_s": mean_ustar,
                "stress_equivalent_friction_velocity_m_s": stress_ustar,
            }
            history.append(row)
            print(
                f"step={run_step}/{args.steps} global={accepted_step} "
                f"uh={hub_mean:.3f}m/s Iu={hub_iu:.3f} "
                f"u*={stress_ustar:.3f}m/s "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    fields = statistics.finish()
    summary = _write_outputs(args.output_dir, fields, physical_grid)
    summary.update(
        {
            "samples": statistics.count,
            "steps_run": args.steps,
            "initial_step": state.clock.step - args.steps,
            "final_step": state.clock.step,
            "maximum_cfl": maximum_cfl,
            "maximum_lasd_cfl": maximum_lasd_cfl,
            "maximum_divergence": maximum_divergence,
            "initial_perturbation": args.initial_perturbation,
            "wall_model": args.wall_model,
            "sgs_model": args.sgs,
            "lasd_update_interval": args.lasd_update_interval,
            "restart": None if args.restart is None else str(args.restart),
            "jax_backend": jax.default_backend(),
            "final_mean_local_friction_velocity_m_s": history[-1][
                "mean_local_friction_velocity_m_s"
            ],
            "final_stress_equivalent_friction_velocity_m_s": history[-1][
                "stress_equivalent_friction_velocity_m_s"
            ],
        }
    )
    with (args.output_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_boussinesq_checkpoint(
        args.output_dir / "precursor_final.npz",
        state,
    )
    if args.flow_gif:
        _write_flow_gif(
            args.output_dir / "precursor_three_plane.gif",
            frames,
            physical_grid,
            fps=args.gif_fps,
            show_turbine=False,
        )
        np.savez_compressed(
            args.output_dir / "precursor_flow_slices.npz",
            **{
                key: np.asarray([frame[key] for frame in frames])
                for key in frames[0]
            },
        )
        summary["flow_gif_frames"] = len(frames)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
