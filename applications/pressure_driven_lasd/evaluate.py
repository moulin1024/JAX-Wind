"""Evaluate pressure-driven neutral-ABL case data through JAX-Wind."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from .config import CaseConfig, load_case
from .problem import build_pressure_driven_problem
from .reporting import write_log_law_svg


HERE = Path(__file__).resolve().parent
SOURCE_CHECKOUT_ROOT = HERE.parents[1]
DEFAULT_CONFIG = (
    SOURCE_CHECKOUT_ROOT / "cases" / "PressureDrivenLASD" / "config.toml"
)


class ProfileStatistics:
    """Restartable horizontal-plane statistics accumulated on the host."""

    NAMES = ("u", "v", "w", "u2", "v2", "w2")

    def __init__(self, nz: int) -> None:
        self.count = 0
        self.sums = {name: np.zeros(nz, dtype=np.float64) for name in self.NAMES}

    def sample(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> None:
        values = {
            "u": np.mean(u, axis=(1, 2)),
            "v": np.mean(v, axis=(1, 2)),
            "w": np.mean(w, axis=(1, 2)),
            "u2": np.mean(u * u, axis=(1, 2)),
            "v2": np.mean(v * v, axis=(1, 2)),
            "w2": np.mean(w * w, axis=(1, 2)),
        }
        for name, value in values.items():
            self.sums[name] += value
        self.count += 1

    def save(self, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez(stream, count=np.asarray(self.count), **self.sums)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path, nz: int) -> "ProfileStatistics":
        result = cls(nz)
        with np.load(path, allow_pickle=False) as archive:
            result.count = int(archive["count"])
            for name in cls.NAMES:
                value = np.asarray(archive[name], dtype=np.float64)
                if value.shape != (nz,):
                    raise ValueError("statistics profile shape does not match the grid")
                result.sums[name] = value.copy()
        return result

    def profiles(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("no profile samples have been collected")
        means = {name: value / self.count for name, value in self.sums.items()}
        return {
            "mean_u": means["u"],
            "mean_v": means["v"],
            "mean_w": means["w"],
            "var_u": np.maximum(means["u2"] - means["u"] ** 2, 0.0),
            "var_v": np.maximum(means["v2"] - means["v"] ** 2, 0.0),
            "var_w": np.maximum(means["w2"] - means["w"] ** 2, 0.0),
        }


def _configure_source_paths() -> None:
    pressure_source = Path(
        os.environ.get(
            "JAXWIND_SPECTRAL_FD_SOURCE",
            SOURCE_CHECKOUT_ROOT / "external" / "bw1000_benchmark",
        )
    )
    if pressure_source.exists() and str(pressure_source) not in sys.path:
        sys.path.insert(0, str(pressure_source))


def _correlated_noise(jax, jnp, key, case: CaseConfig):
    domain = case.domain
    shape = (domain.nz, domain.ny, domain.nx)
    dtype = getattr(jnp, case.numerics.dtype)
    noise = jax.random.normal(key, shape, dtype=dtype)
    spectrum = jnp.fft.rfftn(noise, axes=(0, 1, 2))
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(domain.nz, d=domain.dz_m)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(domain.ny, d=domain.dy_m)
    kx = 2.0 * jnp.pi * jnp.fft.rfftfreq(domain.nx, d=domain.dx_m)
    length = case.flow.initial_correlation_length_m
    low_pass = jnp.exp(
        -0.5
        * length**2
        * (kz[:, None, None] ** 2 + ky[None, :, None] ** 2 + kx[None, None, :] ** 2)
    )
    filtered = jnp.fft.irfftn(
        spectrum * low_pass,
        s=shape,
        axes=(0, 1, 2),
    ).astype(dtype)
    filtered -= jnp.mean(filtered, axis=(1, 2), keepdims=True)
    rms = jnp.sqrt(jnp.mean(filtered * filtered))
    return filtered / jnp.maximum(rms, jnp.finfo(dtype).tiny)


def _initial_fields(
    *,
    jax,
    jnp,
    case: CaseConfig,
    physical_grid,
    scales,
    solver,
):
    from jaxwind.domain import (
        Accepted,
        PassiveScalarConcentration,
    )
    from jaxwind.physics import BoussinesqFields

    dtype = getattr(jnp, case.numerics.dtype)
    domain = case.domain
    z = (jnp.arange(domain.nz, dtype=dtype) + 0.5) * domain.dz_m
    log_velocity = (
        case.flow.friction_velocity_m_s
        / case.flow.von_karman
        * jnp.log(z / case.flow.roughness_length_m)
    )
    keys = jax.random.split(jax.random.PRNGKey(case.numerics.seed), 3)
    noise = tuple(_correlated_noise(jax, jnp, key, case) for key in keys)
    amplitude = case.flow.initial_perturbation_rms_m_s
    cell_taper = jnp.sin(jnp.pi * z / domain.lz_m) ** 0.25
    face_z = (jnp.arange(domain.nz, dtype=dtype) + 1.0) * domain.dz_m
    face_taper = jnp.maximum(jnp.sin(jnp.pi * face_z / domain.lz_m), 0.0)

    u = log_velocity[:, None, None] + amplitude * cell_taper[:, None, None] * noise[0]
    v = amplitude * cell_taper[:, None, None] * noise[1]
    w = (amplitude * face_taper[:, None, None] * noise[2]).at[-1].set(0.0)
    u = jnp.broadcast_to(u, (domain.nz, domain.ny, domain.nx))
    v = jnp.broadcast_to(v, (domain.nz, domain.ny, domain.nx))
    w = jnp.broadcast_to(w, (domain.nz, domain.ny, domain.nx))

    candidate = solver.candidate_velocity(
        scales.to_execution_velocity(u),
        scales.to_execution_velocity(v),
        scales.to_execution_velocity(w),
        lower_boundary=jnp.zeros((domain.ny, domain.nx), dtype=dtype),
    )
    velocity = solver.project_initial_velocity(candidate)
    scalar = solver.cell_field(
        PassiveScalarConcentration,
        Accepted,
        jnp.zeros((domain.nz, domain.ny, domain.nx), dtype=dtype),
    )
    return BoussinesqFields(velocity, scalar)


def _physical_velocity(state, case: CaseConfig, scales, jnp, solver):
    velocity = state.fields.velocity
    u = scales.from_execution_velocity(solver.global_array(velocity.x.payload))
    v = scales.from_execution_velocity(solver.global_array(velocity.y.payload))
    w_upper = scales.from_execution_velocity(
        solver.global_array(velocity.z.owned.payload)
    )
    lower = jnp.concatenate((jnp.zeros_like(w_upper[:1]), w_upper[:-1]), axis=0)
    return u, v, 0.5 * (lower + w_upper), w_upper


def _filtered_first_level(u0, v0, case: CaseConfig, jnp):
    values = jnp.stack((u0, v0))
    spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
    x_mode = jnp.arange(case.domain.nx // 2 + 1)
    y_mode = jnp.abs(jnp.fft.fftfreq(case.domain.ny) * case.domain.ny)
    width = case.wall.filter_grid_ratio * case.wall.test_filter_ratio
    cutoff_x = jnp.floor(case.domain.nx / (2.0 * width))
    cutoff_y = jnp.floor(case.domain.ny / (2.0 * width))
    keep = (y_mode[:, None] < cutoff_y) & (x_mode[None, :] < cutoff_x)
    filtered = jnp.fft.irfftn(
        spectrum * keep,
        s=(case.domain.ny, case.domain.nx),
        axes=(-2, -1),
    )
    return filtered[0], filtered[1]


def _diagnostics(
    state, divergence, case: CaseConfig, scales, jnp, solver
) -> dict[str, float]:
    u, v, _w, w_upper = _physical_velocity(state, case, scales, jnp, solver)
    cfl_x = float(jnp.max(jnp.abs(u))) * case.time.dt_seconds / case.domain.dx_m
    cfl_y = float(jnp.max(jnp.abs(v))) * case.time.dt_seconds / case.domain.dy_m
    cfl_z = float(jnp.max(jnp.abs(w_upper))) * case.time.dt_seconds / case.domain.dz_m
    u0, v0 = _filtered_first_level(u[0], v[0], case, jnp)
    local_ustar = (
        case.flow.von_karman
        * jnp.hypot(u0, v0)
        / math.log(0.5 * case.domain.dz_m / case.flow.roughness_length_m)
    )
    maximum_cfl = max(cfl_x, cfl_y, cfl_z)
    return {
        "cfl_x": cfl_x,
        "cfl_y": cfl_y,
        "cfl_z": cfl_z,
        "maximum_cfl": maximum_cfl,
        # Horizontal departure interpolation is periodic and supports travel over
        # multiple cells.  The vertical interpolation has one neighboring plane,
        # so its trajectory limit applies only to vertical displacement.
        "lasd_trajectory_cfl": cfl_z * case.sgs.update_interval_steps,
        "stress_equivalent_ustar_m_s": float(
            jnp.sqrt(jnp.mean(local_ustar * local_ustar))
        ),
        "maximum_execution_divergence": float(
            jnp.max(jnp.abs(solver.global_array(divergence)))
        ),
    }


def _write_profiles(
    path: Path,
    case: CaseConfig,
    statistics: ProfileStatistics,
) -> None:
    fields = statistics.profiles()
    z = (np.arange(case.domain.nz) + 0.5) * case.domain.dz_m
    tke = 0.5 * (fields["var_u"] + fields["var_v"] + fields["var_w"])
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "z_m",
                "mean_u_m_s",
                "mean_v_m_s",
                "mean_w_m_s",
                "var_u_m2_s2",
                "var_v_m2_s2",
                "var_w_m2_s2",
                "resolved_tke_m2_s2",
            )
        )
        writer.writerows(
            zip(
                z,
                fields["mean_u"],
                fields["mean_v"],
                fields["mean_w"],
                fields["var_u"],
                fields["var_v"],
                fields["var_w"],
                tke,
                strict=True,
            )
        )


def evaluate(
    case: CaseConfig,
    *,
    output_dir: Path,
    restart: Path | None,
    max_steps: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    _configure_source_paths()
    import jax

    jax.config.update("jax_enable_x64", case.numerics.dtype == "float64")
    import jax.numpy as jnp
    from jaxwind.domain import AcceptedClock
    from jaxwind.effects import (
        JaxRuntime,
        load_boussinesq_checkpoint,
        save_boussinesq_checkpoint,
    )
    runtime = JaxRuntime.from_initialized_jax(jax)

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = output_dir / "checkpoint_latest.npz"
    statistics_path = output_dir / "statistics_latest.npz"
    if restart is None and latest_checkpoint.exists() and not overwrite:
        raise FileExistsError(
            f"{latest_checkpoint} already exists; configure "
            "execution.restart_checkpoint or execution.overwrite"
        )
    if restart is not None and not runtime.checkpoint_path(restart).exists():
        raise FileNotFoundError(runtime.checkpoint_path(restart))
    if runtime.is_primary:
        (output_dir / "resolved_config.toml").write_text(case.resolved_toml())
    runtime.synchronize("jaxwind-pressure-driven-output-ready")

    problem = build_pressure_driven_problem(case, runtime=runtime)
    physical_grid = problem.physical_grid
    scales = problem.scales
    momentum_sgs = problem.momentum_sgs
    scalar_sgs = problem.scalar_sgs
    physics_fingerprint = problem.physics_fingerprint
    integrator_config = problem.integrator
    solver = problem.solver
    checkpoint_layout = solver.checkpoint_layout(jnp.asarray)
    local_latest_checkpoint = runtime.checkpoint_path(latest_checkpoint)

    if restart is None:
        fields = _initial_fields(
            jax=jax,
            jnp=jnp,
            case=case,
            physical_grid=physical_grid,
            scales=scales,
            solver=solver,
        )
        fields = solver.initialize_fields(fields)
        state = solver.cold_start(fields, clock=AcceptedClock(0.0, 0))
        statistics = ProfileStatistics(case.domain.nz)
    else:
        state = load_boussinesq_checkpoint(
            runtime.checkpoint_path(restart),
            layout=checkpoint_layout,
            config=integrator_config,
            scale_fingerprint=scales.fingerprint,
            closure_fingerprint=momentum_sgs.fingerprint + "|" + scalar_sgs.fingerprint,
            physics_fingerprint=physics_fingerprint,
        )
        statistics = (
            ProfileStatistics.load(statistics_path, case.domain.nz)
            if statistics_path.exists()
            else ProfileStatistics(case.domain.nz)
        )

    if state.clock.step > case.time.steps:
        raise ValueError("restart step is beyond the configured final time")
    remaining_steps = case.time.steps - state.clock.step
    steps_to_run = (
        remaining_steps if max_steps is None else min(remaining_steps, max_steps)
    )
    if steps_to_run == 0:
        if runtime.is_primary:
            print("configured final time is already present in the checkpoint")

    advance = solver.advance

    history_path = output_dir / "history.csv"
    history_exists = (
        runtime.is_primary and history_path.exists() and restart is not None
    )
    history_stream = (
        history_path.open("a" if history_exists else "w", newline="")
        if runtime.is_primary
        else io.StringIO()
    )
    fieldnames = (
        "step",
        "time_hours",
        "cfl_x",
        "cfl_y",
        "cfl_z",
        "maximum_cfl",
        "lasd_trajectory_cfl",
        "stress_equivalent_ustar_m_s",
        "maximum_execution_divergence",
        "elapsed_seconds",
    )
    history_writer = csv.DictWriter(history_stream, fieldnames=fieldnames)
    if not history_exists:
        history_writer.writeheader()

    started = time.perf_counter()
    latest_diagnostic: dict[str, float] = {}
    try:
        for local_step in range(1, steps_to_run + 1):
            next_accepted_step = state.clock.step + 1
            should_log = (
                next_accepted_step % case.output.log_every_steps == 0
                or local_step == steps_to_run
            )
            result = advance(
                state,
                compute_projection_residual=should_log,
            )
            state = result.state
            accepted_step = state.clock.step
            if (
                accepted_step >= case.sample_start_step
                and (accepted_step - case.sample_start_step)
                % case.output.sample_every_steps
                == 0
            ):
                u, v, w, _w_upper = _physical_velocity(
                    state, case, scales, jnp, solver
                )
                statistics.sample(
                    np.asarray(jax.device_get(u), dtype=np.float64),
                    np.asarray(jax.device_get(v), dtype=np.float64),
                    np.asarray(jax.device_get(w), dtype=np.float64),
                )

            if should_log:
                latest_diagnostic = _diagnostics(
                    state,
                    result.diagnostic.projection.divergence.payload,
                    case,
                    scales,
                    jnp,
                    solver,
                )
                if latest_diagnostic["maximum_cfl"] >= case.numerics.cfl_abort:
                    raise RuntimeError(
                        "CFL abort limit reached: "
                        f"{latest_diagnostic['maximum_cfl']:.4f} >= "
                        f"{case.numerics.cfl_abort:.4f}"
                    )
                if (
                    latest_diagnostic["lasd_trajectory_cfl"]
                    >= case.numerics.lasd_trajectory_cfl_abort
                ):
                    raise RuntimeError(
                        "LASD trajectory CFL abort limit reached: "
                        f"{latest_diagnostic['lasd_trajectory_cfl']:.4f} >= "
                        f"{case.numerics.lasd_trajectory_cfl_abort:.4f}"
                    )
                row = {
                    "step": accepted_step,
                    "time_hours": (accepted_step * case.time.dt_seconds / 3600.0),
                    **latest_diagnostic,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                if runtime.is_primary:
                    history_writer.writerow(row)
                    history_stream.flush()
                    print(
                        f"step={accepted_step}/{case.time.steps} "
                        f"time={row['time_hours']:.3f}h "
                        f"CFL={latest_diagnostic['maximum_cfl']:.3f} "
                        f"trajectory-CFL="
                        f"{latest_diagnostic['lasd_trajectory_cfl']:.3f} "
                        f"u*={latest_diagnostic['stress_equivalent_ustar_m_s']:.3f}m/s "
                        f"elapsed={row['elapsed_seconds']:.1f}s",
                        flush=True,
                    )

            should_checkpoint = (
                accepted_step % case.output.checkpoint_every_steps == 0
                or local_step == steps_to_run
            )
            if should_checkpoint:
                save_boussinesq_checkpoint(
                    local_latest_checkpoint,
                    state,
                    scale_fingerprint=scales.fingerprint,
                    physics_fingerprint=physics_fingerprint,
                )
                if runtime.is_primary:
                    statistics.save(statistics_path)
    finally:
        history_stream.close()

    if steps_to_run == 0:
        save_boussinesq_checkpoint(
            local_latest_checkpoint,
            state,
            scale_fingerprint=scales.fingerprint,
            physics_fingerprint=physics_fingerprint,
        )
        if runtime.is_primary:
            statistics.save(statistics_path)
    if runtime.is_primary and statistics.count:
        profile_path = output_dir / "profiles.csv"
        _write_profiles(profile_path, case, statistics)
        write_log_law_svg(
            profile_path,
            output_dir / "loglaw_velocity_profile.svg",
            friction_velocity_m_s=case.flow.friction_velocity_m_s,
            roughness_length_m=case.flow.roughness_length_m,
            von_karman=case.flow.von_karman,
            model_label="LASD",
            statistics_label=(
                f"{case.output.sample_start_hours:g}--"
                f"{case.time.duration_hours:g} h"
            ),
        )
    if state.clock.step == case.time.steps:
        save_boussinesq_checkpoint(
            runtime.checkpoint_path(output_dir / "checkpoint_final.npz"),
            state,
            scale_fingerprint=scales.fingerprint,
            physics_fingerprint=physics_fingerprint,
        )

    summary = {
        **case.resolved(),
        "physics": {
            "momentum_advection": "conservative",
            "dealiasing": "three-halves-padding",
            "nonlinear_padding_ratio": 1.5,
            "fingerprint": physics_fingerprint,
        },
        "runtime": {
            "jax_backend": jax.default_backend(),
            "process_count": runtime.process_count,
            "global_device_count": runtime.global_devices,
            "local_device_count": runtime.local_devices,
            "restart": None if restart is None else str(restart),
            "initial_step": state.clock.step - steps_to_run,
            "steps_run": steps_to_run,
            "final_step": state.clock.step,
            "final_time_hours": (state.clock.step * case.time.dt_seconds / 3600.0),
            "profile_samples": statistics.count,
            "reached_configured_final_time": state.clock.step == case.time.steps,
            **latest_diagnostic,
        },
    }
    if runtime.is_primary:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case = load_case(args.config)
    if args.dry_run:
        print(case.resolved_toml(), end="")
        return 0
    evaluate(
        case,
        output_dir=Path(case.output.directory),
        restart=args.restart,
        max_steps=args.max_steps,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
