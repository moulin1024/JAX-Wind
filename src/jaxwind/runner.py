"""Benchmark-independent execution engine for declarative ABL cases."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import CaseConfig, load_case


PROFILE_COLUMNS = (
    "z_m",
    "mean_u_m_s",
    "mean_v_m_s",
    "mean_w_m_s",
    "var_u_m2_s2",
    "var_v_m2_s2",
    "var_w_m2_s2",
    "mean_scalar",
    "var_scalar",
    "resolved_uw_m2_s2",
    "resolved_vw_m2_s2",
    "resolved_wscalar",
    "sgs_viscosity_m2_s",
)
HISTORY_COLUMNS = (
    "step",
    "time_s",
    "timestep_s",
    "advective_cfl",
    "diffusive_cfl",
    "divergence_l2",
    "kinetic_energy_m2_s2",
    "mean_scalar",
)


@dataclass(slots=True)
class Simulation:
    config: CaseConfig
    grid: Any
    pressure: Any
    solver: Any
    dtype: Any

    @property
    def momentum(self) -> Any:
        return self.solver.momentum

    @property
    def scalar(self) -> Any | None:
        return self.solver.scalar

    @property
    def thermodynamics_enabled(self) -> bool:
        return bool(self.config.section("thermodynamics")["enabled"])

    def initial_state(self) -> Any:
        import jax.numpy as jnp
        from jaxwind.pressure import MACVelocity

        initial = self.config.section("initial")
        velocity_spec = initial["velocity"]
        seed = int(initial.get("seed", 0))
        nz, ny, nx = self.grid.shape

        if velocity_spec["kind"] == "table":
            z = np.asarray(self.grid.z_centers, dtype=float)
            if "z" in velocity_spec:
                source_z = np.asarray(velocity_spec["z"], dtype=float)
            else:
                source_z = float(velocity_spec["z_origin"]) + np.arange(
                    len(velocity_spec["u"]), dtype=float
                ) * float(velocity_spec["z_spacing"])

            def profile(name: str) -> np.ndarray:
                source = np.asarray(velocity_spec[name], dtype=float)
                values = np.interp(z, source_z, source)
                if velocity_spec.get("lower_extrapolation") == "log":
                    roughness = self.config.section("momentum")["roughness_length"]
                    below = z < source_z[0]
                    ratio = np.log(np.maximum(z, 1.001 * roughness) / roughness)
                    ratio /= math.log(source_z[0] / roughness)
                    values = np.where(below, source[0] * ratio, values)
                return values

            tke = profile("tke") if "tke" in velocity_spec else None
            velocity = self.momentum.initial_profile(
                jnp.asarray(profile("u"), dtype=self.dtype),
                jnp.asarray(profile("v"), dtype=self.dtype),
                perturbation_tke=(
                    None if tke is None else jnp.asarray(tke, dtype=self.dtype)
                ),
                seed=seed,
            )
        else:
            value = velocity_spec.get("value", [0.0, 0.0, 0.0])
            velocity = MACVelocity(
                jnp.full((nz, ny, nx + 1), value[0], dtype=self.dtype),
                jnp.full((nz, ny + 1, nx), value[1], dtype=self.dtype),
                jnp.full((nz + 1, ny, nx), value[2], dtype=self.dtype),
            )

        potential_temperature, random = self._initial_potential_temperature(seed)
        temperature_spec = initial["potential_temperature"]
        w_noise = float(temperature_spec.get("vertical_velocity_noise", 0.0))
        if w_noise != 0.0:
            z_faces = jnp.asarray(self.grid.z_faces[1:], dtype=self.dtype)
            top = float(temperature_spec["inversion_height"])
            envelope = (
                jnp.maximum(1.0 - z_faces / top, 0.0)
                if temperature_spec.get("taper", False)
                else (z_faces < float(temperature_spec.get("noise_below", top)))
            )
            w = velocity.z.at[1:].add(w_noise * random * envelope[:, None, None])
            velocity = MACVelocity(velocity.x, velocity.y, w)

        projected = self.momentum.projector.project_velocity_and_pressure(
            self.momentum.enforce_boundaries(velocity),
            timestep=1.0,
        )
        velocity = projected.velocity
        self.momentum.restore_pressure(projected.pressure)
        if self.momentum.lasd_closure is not None:
            self.momentum.reset_lasd(velocity)

        return self.solver.initial_state(
            velocity,
            potential_temperature,
            pressure=projected.pressure,
        )

    def _initial_potential_temperature(self, seed: int) -> tuple[Any | None, Any]:
        import jax
        import jax.numpy as jnp

        spec = self.config.section("initial")["potential_temperature"]
        nz, ny, nx = self.grid.shape
        random = jax.random.uniform(
            jax.random.PRNGKey(seed),
            (nz, ny, nx),
            minval=float(spec.get("noise_minimum", -0.5)),
            maxval=float(spec.get("noise_maximum", 0.5)),
            dtype=self.dtype,
        )
        if spec.get("plane_mean_zero", False):
            random -= jnp.mean(random, axis=(1, 2), keepdims=True)
        if not self.thermodynamics_enabled:
            return None, random

        z = jnp.asarray(self.grid.z_centers, dtype=self.dtype)
        inversion = float(spec["inversion_height"])
        base = float(spec.get("base", 0.0))
        gradient = float(spec.get("gradient_above", 0.0))
        profile = jnp.where(z <= inversion, base, base + gradient * (z - inversion))
        amplitude = float(spec.get("noise_amplitude", 0.0))
        noise_below = float(spec.get("noise_below", inversion))
        envelope = (
            jnp.maximum(1.0 - z / inversion, 0.0)
            if spec.get("taper", False)
            else (z < noise_below)
        )
        potential_temperature = (
            profile[:, None, None] + amplitude * random * envelope[:, None, None]
        )
        return potential_temperature, random

    def step(self, state: Any, timestep: float) -> Any:
        return self.solver.step(state, timestep=timestep)

    def timestep(self, state: Any) -> float:
        numerics = self.config.section("numerics")
        target_cfl = float(numerics["target_cfl"])
        target_diffusive = float(numerics["target_diffusive_cfl"])
        value = self.solver.timestep_for_cfl(
            state,
            target_cfl,
            target_diffusive,
        )
        return min(value, float(self.config.section("time")["maximum_step"]))


def build_simulation(config: CaseConfig) -> Simulation:
    from jax import config as jax_config

    numerics = config.section("numerics")
    if numerics["dtype"] == "float64":
        jax_config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    from jaxwind.momentum import (
        ABLSolver,
        AMDModel,
        LASDModel,
        MomentumConfig,
        MomentumOperators,
        ScalarConfig,
        ScalarOperators,
        ThermodynamicsConfig,
    )
    from jaxwind.pressure import (
        BoundaryCondition,
        GMGConfig,
        MatrixFreePoissonSolver,
        PCGConfig,
        PoissonBoundaryConditions,
        RectilinearGrid,
    )

    grid_spec = config.section("grid")
    nx, ny, nz = (int(value) for value in grid_spec["shape"])
    lx, ly, lz = (float(value) for value in grid_spec["extent"])
    grid = RectilinearGrid.uniform(nx, ny, nz, lx=lx, ly=ly, lz=lz)
    dtype = jnp.float32 if numerics["dtype"] == "float32" else jnp.float64
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions(
            periodic,
            periodic,
            periodic,
            periodic,
            neumann,
            neumann,
        ),
        dtype=dtype,
        gmg=GMGConfig(
            smoother="auto",
            coarsening=str(numerics.get("pressure_coarsening", "auto")),
            pre_smooth=int(numerics.get("pressure_smooth", 2)),
            post_smooth=int(numerics.get("pressure_smooth", 2)),
            coarse_smooth=int(numerics["pressure_coarse_smooth"]),
        ),
        krylov=PCGConfig(
            max_iterations=int(numerics["pressure_max_iterations"]),
            relative_tolerance=float(numerics["pressure_relative_tolerance"]),
            execution="jax",
        ),
    )

    momentum_spec = config.section("momentum")
    sgs = config.section("sgs")
    geostrophic = momentum_spec.get("geostrophic_wind")
    lasd = None
    if sgs["model"] == "multilevel_lasd":
        lasd = LASDModel(
            update_interval=int(sgs.get("update_interval", 1)),
            initial_coefficient=float(sgs.get("initial_coefficient", 0.03)),
            maximum_coefficient=float(sgs.get("maximum_coefficient", 0.81)),
            sgs_delta_scale=(
                None if sgs.get("delta_scale") is None else float(sgs["delta_scale"])
            ),
        )
    momentum = MomentumOperators(
        grid,
        pressure,
        MomentumConfig(
            friction_velocity=float(momentum_spec["friction_velocity"]),
            roughness_length=float(momentum_spec["roughness_length"]),
            pressure_acceleration=(
                None
                if momentum_spec.get("pressure_acceleration") is None
                else float(momentum_spec["pressure_acceleration"])
            ),
            geostrophic_wind=(
                None
                if geostrophic is None
                else (float(geostrophic[0]), float(geostrophic[1]))
            ),
            coriolis_vertical=float(momentum_spec.get("coriolis_vertical", 0.0)),
            coriolis_horizontal=float(momentum_spec.get("coriolis_horizontal", 0.0)),
            wall_matching_height=momentum_spec.get("wall_matching_height"),
            mp5_dissipation_strength=float(numerics.get("mp5_strength", 1.0)),
            amd=AMDModel(coefficient=float(sgs["coefficient"])),
            lasd=lasd,
            sgs_time_integration=str(numerics["sgs_time_integration"]),
        ),
    )

    thermodynamics = config.section("thermodynamics")
    surface = config.section("surface")
    scalar = None
    thermal_config = None
    if thermodynamics["enabled"]:
        thermal_boundary = str(surface["thermal_boundary"])
        scalar = ScalarOperators(
            grid,
            ScalarConfig(
                coefficient=float(thermodynamics["sgs_coefficient"]),
                lower_surface_flux=(
                    float(surface["heat_flux"]) if thermal_boundary == "flux" else 0.0
                ),
                upper_surface_flux=0.0,
                mp5_dissipation_strength=float(numerics.get("mp5_strength", 1.0)),
            ),
        )
        sponge = config.data.get("sponge", {})
        thermal_config = ThermodynamicsConfig(
            gravity=float(thermodynamics["gravity"]),
            reference_potential_temperature=float(
                thermodynamics["reference_potential_temperature"]
            ),
            surface_potential_temperature=(
                float(surface["potential_temperature"])
                if thermal_boundary == "temperature"
                else None
            ),
            surface_temperature_tendency=float(
                surface.get("temperature_tendency", 0.0)
            ),
            thermal_roughness_length=surface.get("thermal_roughness_length"),
            rayleigh_sponge_start_height=sponge.get("start_height"),
            rayleigh_sponge_maximum_rate=float(sponge.get("maximum_rate", 0.0)),
            rayleigh_reference_temperature_at_zero=sponge.get(
                "reference_temperature_at_zero"
            ),
            rayleigh_reference_temperature_gradient=float(
                sponge.get("reference_temperature_gradient", 0.0)
            ),
        )
    solver = ABLSolver(momentum, scalar, thermal_config)
    return Simulation(config, grid, pressure, solver, dtype)


def _physics_fingerprint(config: CaseConfig) -> str:
    sections = {
        key: config.data[key]
        for key in (
            "grid",
            "numerics",
            "momentum",
            "sgs",
            "thermodynamics",
            "surface",
            "sponge",
            "initial",
        )
        if key in config.data
    }
    encoded = json.dumps(sections, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _snapshot(simulation: Simulation, state: Any) -> np.ndarray:
    cells = np.asarray(simulation.momentum.cell_centered_velocity(state.velocity))
    mean = cells.mean(axis=(1, 2))
    fluctuation = cells - mean[:, None, None, :]
    variance = np.mean(fluctuation * fluctuation, axis=(1, 2))
    scalar = state.potential_temperature
    if scalar is None:
        scalar_mean = np.full(simulation.grid.shape[0], np.nan)
        scalar_variance = np.full_like(scalar_mean, np.nan)
        wscalar = np.full_like(scalar_mean, np.nan)
    else:
        scalar_array = np.asarray(scalar) + float(
            simulation.config.data.get("display", {}).get("scalar_offset", 0.0)
        )
        scalar_mean = scalar_array.mean(axis=(1, 2))
        scalar_fluctuation = scalar_array - scalar_mean[:, None, None]
        scalar_variance = np.mean(scalar_fluctuation**2, axis=(1, 2))
        wscalar = np.mean(fluctuation[..., 2] * scalar_fluctuation, axis=(1, 2))
    viscosity = np.asarray(simulation.momentum.sgs_viscosity(cells)).mean(axis=(1, 2))
    return np.column_stack(
        (
            np.asarray(simulation.grid.z_centers),
            mean,
            variance,
            scalar_mean,
            scalar_variance,
            np.mean(fluctuation[..., 0] * fluctuation[..., 2], axis=(1, 2)),
            np.mean(fluctuation[..., 1] * fluctuation[..., 2], axis=(1, 2)),
            wscalar,
            viscosity,
        )
    )


def _diagnostic_row(
    simulation: Simulation,
    state: Any,
    timestep: float,
) -> tuple[float, ...]:
    import jax.numpy as jnp
    from jaxwind.pressure import mac_divergence

    cells = simulation.momentum.cell_centered_velocity(state.velocity)
    advective_rate = float(simulation.momentum.cfl_rate(state.velocity))
    viscosity = simulation.momentum.sgs_viscosity(cells)
    momentum_rate = float(
        simulation.momentum.explicit_sgs_diffusion_rate(
            viscosity,
            include_vertical=(
                simulation.momentum.config.sgs_time_integration == "explicit"
            ),
        )
    )
    scalar_rate = 0.0
    scalar = state.potential_temperature
    if scalar is not None:
        scalar_rate = float(simulation.scalar.diffusive_rate(scalar, state.velocity))
    divergence = float(
        simulation.pressure.operator.norm(
            mac_divergence(state.velocity, simulation.grid)
        )
    )
    kinetic = float(0.5 * jnp.mean(jnp.sum(cells * cells, axis=-1)))
    scalar_mean = (
        math.nan
        if scalar is None
        else float(jnp.mean(scalar))
        + float(simulation.config.data.get("display", {}).get("scalar_offset", 0.0))
    )
    return (
        float(state.step),
        float(state.time),
        timestep,
        timestep * advective_rate,
        timestep * max(momentum_rate, scalar_rate),
        divergence,
        kinetic,
        scalar_mean,
    )


def _atomic_checkpoint(
    destination: Path,
    simulation: Simulation,
    state: Any,
    samples: list[np.ndarray],
    sample_times: list[float],
    history: list[tuple[float, ...]],
) -> None:
    payload: dict[str, Any] = {
        "schema": "jaxwind.checkpoint.v1",
        "physics_fingerprint": _physics_fingerprint(simulation.config),
        "velocity_x": np.asarray(state.velocity.x),
        "velocity_y": np.asarray(state.velocity.y),
        "velocity_z": np.asarray(state.velocity.z),
        "pressure": np.asarray(state.pressure),
        "time": state.time,
        "step": state.step,
        "sample_times": np.asarray(sample_times),
        "samples": np.stack(samples) if samples else np.empty((0,)),
        "history": np.asarray(history),
    }
    scalar = state.potential_temperature
    if scalar is not None:
        payload["scalar"] = np.asarray(scalar)
    if simulation.momentum.lasd_state is not None:
        lasd = simulation.momentum.lasd_state
        for name, value in zip(lasd._fields, lasd, strict=True):
            payload[f"lasd_{name}"] = np.asarray(value)
        step, interval_time = simulation.momentum.lasd_progress
        payload["lasd_step"] = step
        payload["lasd_interval_time"] = interval_time
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, destination)


def _restore(
    path: Path,
    simulation: Simulation,
) -> tuple[Any, list[np.ndarray], list[float], list[tuple[float, ...]]]:
    import jax.numpy as jnp
    from jaxwind.momentum import LASDState
    from jaxwind.pressure import MACVelocity

    checkpoint = np.load(path, allow_pickle=False)
    if str(checkpoint["schema"]) != "jaxwind.checkpoint.v1":
        raise ValueError("unsupported checkpoint schema")
    if str(checkpoint["physics_fingerprint"]) != _physics_fingerprint(
        simulation.config
    ):
        raise ValueError("checkpoint physics does not match the case configuration")
    velocity = MACVelocity(
        jnp.asarray(checkpoint["velocity_x"], dtype=simulation.dtype),
        jnp.asarray(checkpoint["velocity_y"], dtype=simulation.dtype),
        jnp.asarray(checkpoint["velocity_z"], dtype=simulation.dtype),
    )
    pressure = jnp.asarray(checkpoint["pressure"], dtype=simulation.dtype)
    scalar = (
        jnp.asarray(checkpoint["scalar"], dtype=simulation.dtype)
        if "scalar" in checkpoint
        else None
    )
    state = simulation.solver.initial_state(
        velocity,
        scalar,
        pressure=pressure,
        time=float(checkpoint["time"]),
        step=int(checkpoint["step"]),
    )
    if simulation.momentum.lasd_closure is not None:
        lasd = LASDState(
            *(
                jnp.asarray(checkpoint[f"lasd_{name}"], dtype=simulation.dtype)
                for name in LASDState._fields
            )
        )
        simulation.momentum.restore_lasd(
            lasd,
            accepted_step=int(checkpoint["lasd_step"]),
            interval_time=float(checkpoint["lasd_interval_time"]),
        )
    sample_array = np.asarray(checkpoint["samples"])
    samples = [sample_array[index] for index in range(sample_array.shape[0])]
    sample_times = list(np.asarray(checkpoint["sample_times"], dtype=float))
    history_array = np.asarray(checkpoint["history"], dtype=float)
    history = [tuple(row) for row in history_array]
    return state, samples, sample_times, history


def _write_outputs(
    output_dir: Path,
    simulation: Simulation,
    state: Any,
    samples: list[np.ndarray],
    sample_times: list[float],
    history: list[tuple[float, ...]],
    runtime: float,
    stopped_early: bool,
) -> dict[str, Any]:
    time_spec = simulation.config.section("time")
    selected = [
        sample
        for sample_time, sample in zip(sample_times, samples, strict=True)
        if sample_time >= float(time_spec["sample_start"])
    ]
    if not selected:
        selected = samples
    profile = np.mean(np.stack(selected), axis=0)
    with (output_dir / "profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(PROFILE_COLUMNS)
        writer.writerows(profile)
    with (output_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HISTORY_COLUMNS)
        writer.writerows(history)
    summary = {
        "schema": "jaxwind.summary.v1",
        "case": simulation.config.name,
        "final_step": state.step,
        "final_time_s": state.time,
        "samples": len(samples),
        "runtime_s": runtime,
        "stopped_early": stopped_early,
        "sgs_model": simulation.config.section("sgs")["model"],
        "shape_zyx": list(simulation.grid.shape),
        "domain_m": list(simulation.config.section("grid")["extent"]),
        "friction_velocity_m_s": simulation.config.section("momentum")[
            "friction_velocity"
        ],
        "geostrophic_wind_m_s": simulation.config.section("momentum").get(
            "geostrophic_wind"
        ),
        "projection_method": "full",
        "pressure_solver": "gmg_pcg",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "resolved_config.json").write_text(simulation.config.resolved_json())
    return summary


def run_case(
    config: CaseConfig,
    *,
    output_dir: Path,
    restart: Path | None = None,
    max_steps: int | None = None,
    max_run_seconds: float | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation = build_simulation(config)
    if restart is None:
        state = simulation.initial_state()
        samples: list[np.ndarray] = []
        sample_times: list[float] = []
        history: list[tuple[float, ...]] = []
    else:
        state, samples, sample_times, history = _restore(restart, simulation)

    time_spec = config.section("time")
    end_time = float(time_spec["end"])
    sample_start = float(time_spec["sample_start"])
    sample_basis = str(time_spec["sample_basis"])
    sample_interval = float(time_spec["sample_interval"])
    if sample_basis == "time":
        if sample_times:
            next_sample_time = sample_times[-1] + sample_interval
        elif state.time < sample_start:
            next_sample_time = sample_start
        else:
            completed_intervals = math.floor(
                (state.time - sample_start) / sample_interval
            )
            next_sample_time = (
                sample_start + (completed_intervals + 1) * sample_interval
            )
    else:
        next_sample_time = math.inf
    log_interval = int(time_spec["log_interval"])
    checkpoint_interval = int(time_spec["checkpoint_interval"])
    checkpoint = output_dir / "checkpoint.npz"
    start = time.perf_counter()
    stopped_early = False

    while state.time < end_time:
        timestep = min(simulation.timestep(state), end_time - state.time)
        state = simulation.step(state, timestep)
        row = _diagnostic_row(simulation, state, timestep)
        history.append(row)
        sample_due = state.time + 1.0e-12 >= sample_start and (
            state.step % int(sample_interval) == 0
            if sample_basis == "step"
            else state.time + 1.0e-12 >= next_sample_time
        )
        final = state.time >= end_time
        if sample_due or final:
            samples.append(_snapshot(simulation, state))
            sample_times.append(state.time)
            while next_sample_time <= state.time + 1.0e-12:
                next_sample_time += sample_interval
        if state.step % log_interval == 0 or final:
            scale = float(config.data.get("display", {}).get("time_scale", 1.0))
            label = str(config.data.get("display", {}).get("time_label", "t"))
            print(
                f"step={state.step} {label}={state.time / scale:.5g}/"
                f"{end_time / scale:.5g} CFL={row[3]:.4f} "
                f"CFLnu={row[4]:.4f} divL2={row[5]:.3e}",
                flush=True,
            )
        if state.step % checkpoint_interval == 0 or final:
            _atomic_checkpoint(
                checkpoint,
                simulation,
                state,
                samples,
                sample_times,
                history,
            )
        if max_steps is not None and state.step >= max_steps:
            stopped_early = state.time < end_time
            break
        if (
            max_run_seconds is not None
            and time.perf_counter() - start >= max_run_seconds
        ):
            stopped_early = True
            break

    if not samples:
        samples.append(_snapshot(simulation, state))
        sample_times.append(state.time)
    _atomic_checkpoint(
        checkpoint,
        simulation,
        state,
        samples,
        sample_times,
        history,
    )
    summary = _write_outputs(
        output_dir,
        simulation,
        state,
        samples,
        sample_times,
        history,
        time.perf_counter() - start,
        stopped_early,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-run-seconds", type=float)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.max_run_seconds is not None and args.max_run_seconds <= 0.0:
        parser.error("--max-run-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_case(args.config).with_overrides(args.overrides)
    if args.quick:
        config = config.with_overrides(
            [
                "grid.shape=[8, 8, 8]",
                'numerics.dtype="float32"',
                "time.sample_start=0.0",
                'time.sample_basis="step"',
                "time.sample_interval=1",
                "time.log_interval=1",
                "time.checkpoint_interval=2",
                "time.maximum_step=0.25",
            ]
        )
        if args.max_steps is None:
            args.max_steps = 4
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(
            config.data.get("output", {}).get("directory", "benchmark_results/run")
        )
    )
    run_case(
        config,
        output_dir=output_dir,
        restart=args.restart,
        max_steps=args.max_steps,
        max_run_seconds=args.max_run_seconds,
    )
    return 0


__all__ = [
    "HISTORY_COLUMNS",
    "PROFILE_COLUMNS",
    "Simulation",
    "build_simulation",
    "main",
    "parse_args",
    "run_case",
]
