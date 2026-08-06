"""Benchmark-independent execution engine for declarative ABL cases."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
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

        if velocity_spec["kind"] == "log_law":
            velocity = self.momentum.initial_log_profile(
                perturbation_amplitude=float(
                    velocity_spec.get("perturbation_amplitude", 0.05)
                ),
                project=False,
            )
        elif velocity_spec["kind"] == "table":
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
                project=False,
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
        if self.momentum.config.wall_temporal_filter_timescale is not None:
            self.momentum.reset_wall_model(velocity)

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
            random -= self.momentum.horizontal_mean(random, keepdims=True)
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

    def step(self, state: Any, timestep: float, prepared: Any | None = None) -> Any:
        return self.solver.step(
            state,
            timestep=timestep,
            prepared=prepared,
        )

    def prepare_step(self, state: Any) -> Any:
        return self.solver.prepare_step(state)

    def runtime_rates(self, state: Any) -> Any:
        return self.solver.runtime_rates(state)

    def diagnostic_metrics(self, state: Any) -> Any:
        return self.solver.diagnostic_metrics(state)

    def timestep_from_metrics(self, metrics: np.ndarray) -> float:
        numerics = self.config.section("numerics")
        target_cfl = float(numerics["target_cfl"])
        target_diffusive = float(numerics["target_diffusive_cfl"])
        advective, momentum_diffusive, scalar_diffusive = (
            float(value) for value in metrics[:3]
        )
        if not math.isfinite(advective) or advective <= 0.0:
            raise ValueError("cannot choose a CFL step for zero or invalid velocity")
        candidates = [target_cfl / advective]
        for rate in (momentum_diffusive, scalar_diffusive):
            if not math.isfinite(rate) or rate < 0.0:
                raise ValueError("diffusive stability rates must be finite and nonnegative")
            if rate > 0.0:
                candidates.append(target_diffusive / rate)
        candidates.append(float(self.config.section("time")["maximum_step"]))
        return min(candidates)

    def timestep(self, state: Any) -> float:
        return self.timestep_from_metrics(np.asarray(self.runtime_rates(state)))


def build_simulation(config: CaseConfig) -> Simulation:
    from jax import config as jax_config

    numerics = config.section("numerics")
    if numerics["dtype"] == "float64":
        jax_config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    from jaxwind.domain import AnalyticAxisMapping
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
    mapping_spec = grid_spec.get("mapping", {})

    def axis_mapping(name: str) -> AnalyticAxisMapping:
        specification = mapping_spec.get(name)
        if specification is None:
            return AnalyticAxisMapping()
        return AnalyticAxisMapping(
            function=str(specification["function"]),
            focus=(
                None
                if specification.get("focus") is None
                else float(specification["focus"])
            ),
            strength=float(specification.get("strength", 0.0)),
        )

    grid = RectilinearGrid.analytic(
        nx,
        ny,
        nz,
        lx=lx,
        ly=ly,
        lz=lz,
        x=axis_mapping("x"),
        y=axis_mapping("y"),
        z=axis_mapping("z"),
    )
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
    von_karman = float(momentum_spec.get("von_karman", 0.4))
    wall_filter_width = momentum_spec.get("wall_filter_width")
    wall_temporal_filter_gamma = momentum_spec.get("wall_temporal_filter_gamma")
    wall_temporal_filter_timescale = (
        None
        if wall_temporal_filter_gamma is None
        else float(grid.z_widths[0])
        / (
            float(wall_temporal_filter_gamma)
            * von_karman
            * float(momentum_spec["friction_velocity"])
        )
    )
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
            von_karman=von_karman,
            wall_filter_width=(
                None if wall_filter_width is None else float(wall_filter_width)
            ),
            wall_temporal_filter_timescale=wall_temporal_filter_timescale,
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
    sections["_solver_physics"] = {
        "wall_closure": "finite_volume_filtered_most_v1"
    }
    if config.section("sgs")["model"] == "multilevel_lasd":
        sections["_solver_physics"]["lasd_discretization"] = (
            "nested_finite_volume_v1"
        )
    encoded = json.dumps(sections, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _snapshot(simulation: Simulation, state: Any) -> np.ndarray:
    profile = np.asarray(simulation.solver.profile(state))
    if state.potential_temperature is not None:
        profile = profile.copy()
        profile[:, 7] += float(
            simulation.config.data.get("display", {}).get("scalar_offset", 0.0)
        )
    return profile


def _diagnostic_row_from_metrics(
    config: CaseConfig,
    state: Any,
    timestep: float,
    rates: np.ndarray,
    diagnostics: np.ndarray,
) -> tuple[float, ...]:
    scalar_mean = float(diagnostics[3])
    if not math.isnan(scalar_mean):
        scalar_mean += float(config.data.get("display", {}).get("scalar_offset", 0.0))
    return (
        float(state.step),
        float(state.time),
        timestep,
        timestep * float(diagnostics[0]),
        timestep * max(float(rates[1]), float(rates[2])),
        float(diagnostics[1]),
        float(diagnostics[2]),
        scalar_mean,
    )


def _format_estimated_completion(remaining_seconds: float) -> str:
    completion = datetime.now().astimezone() + timedelta(seconds=remaining_seconds)
    total_seconds = max(0, int(round(remaining_seconds)))
    days, remainder = divmod(total_seconds, 24 * 3600)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        duration = f"{days}d{hours:02d}h{minutes:02d}m"
    elif hours:
        duration = f"{hours}h{minutes:02d}m{seconds:02d}s"
    elif minutes:
        duration = f"{minutes}m{seconds:02d}s"
    else:
        duration = f"{seconds}s"
    return f"ETA={completion:%Y-%m-%d %H:%M:%S %Z} remaining={duration}"


class _RollingCompletionEstimator:
    """Estimate completion from recent, post-compilation synchronized progress."""

    def __init__(self, window: int = 6) -> None:
        if window < 2:
            raise ValueError("ETA rolling window must contain at least two observations")
        self._window = window
        self._observations: list[tuple[float, int, float]] = []

    def observe(self, *, wall_time: float, step: int, physical_time: float) -> None:
        observation = (float(wall_time), int(step), float(physical_time))
        if self._observations:
            previous = self._observations[-1]
            if observation[0] <= previous[0]:
                raise ValueError("ETA observation wall time must increase")
            if observation[1] <= previous[1]:
                raise ValueError("ETA observation step must increase")
            if observation[2] <= previous[2]:
                raise ValueError("ETA observation physical time must increase")
        self._observations.append(observation)
        if len(self._observations) > self._window:
            del self._observations[: -self._window]

    def estimate(self, *, remaining: float) -> str:
        if remaining <= 0.0:
            return "ETA=done"
        if len(self._observations) < 2:
            return "ETA=warming-up"

        first_wall, first_step, _ = self._observations[0]
        last_wall, last_step, last_physical = self._observations[-1]
        _, previous_step, previous_physical = self._observations[-2]
        elapsed = last_wall - first_wall
        completed_steps = last_step - first_step
        recent_steps = last_step - previous_step
        recent_physical_advance = last_physical - previous_physical
        if (
            elapsed <= 0.0
            or completed_steps <= 0
            or recent_steps <= 0
            or recent_physical_advance <= 0.0
        ):
            return "ETA=unknown"

        steps_per_second = completed_steps / elapsed
        physical_time_per_step = recent_physical_advance / recent_steps
        physical_time_per_second = steps_per_second * physical_time_per_step
        remaining_seconds = remaining / physical_time_per_second
        if not math.isfinite(remaining_seconds):
            return "ETA=unknown"
        return _format_estimated_completion(remaining_seconds)


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
    if simulation.momentum.wall_model_state is not None:
        payload["wall_filtered_velocity"] = np.asarray(
            simulation.momentum.wall_model_state.filtered_velocity
        )
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, destination)


def _restore(
    path: Path,
    simulation: Simulation,
) -> tuple[Any, list[np.ndarray], list[float], list[tuple[float, ...]]]:
    import jax.numpy as jnp
    from jaxwind.momentum import LASDState, WallModelState
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
    if simulation.momentum.config.wall_temporal_filter_timescale is not None:
        if "wall_filtered_velocity" not in checkpoint:
            raise ValueError("temporal wall-model checkpoint is missing memory")
        simulation.momentum.restore_wall_model(
            WallModelState(
                jnp.asarray(
                    checkpoint["wall_filtered_velocity"],
                    dtype=simulation.dtype,
                )
            )
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
    # Height is geometry, not a sampled statistic.  Repeatedly accumulating a
    # float32 copy can move the nominal centres enough to fail strict grid
    # consistency checks over a long averaging window.
    profile[:, 0] = np.asarray(
        simulation.grid.z_centers,
        dtype=profile.dtype,
    )
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
        "lasd_discretization": (
            "nested_finite_volume_v1"
            if simulation.config.section("sgs")["model"] == "multilevel_lasd"
            else None
        ),
        "shape_zyx": list(simulation.grid.shape),
        "domain_m": list(simulation.config.section("grid")["extent"]),
        "minimum_spacing_m": [
            min(simulation.grid.x_widths),
            min(simulation.grid.y_widths),
            min(simulation.grid.z_widths),
        ],
        "maximum_spacing_m": [
            max(simulation.grid.x_widths),
            max(simulation.grid.y_widths),
            max(simulation.grid.z_widths),
        ],
        "friction_velocity_m_s": simulation.config.section("momentum")[
            "friction_velocity"
        ],
        "wall_closure": "finite_volume_filtered_most_v1",
        "wall_first_cell_height_m": simulation.momentum.wall_cell_height,
        "wall_filter_width_grid_cells": (
            simulation.momentum.config.wall_filter_width
        ),
        "wall_temporal_filter_gamma": simulation.config.section("momentum").get(
            "wall_temporal_filter_gamma"
        ),
        "wall_temporal_filter_timescale_s": (
            simulation.momentum.config.wall_temporal_filter_timescale
        ),
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
    import jax

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
    history_interval = int(time_spec["history_interval"])
    log_interval = int(time_spec["log_interval"])
    checkpoint_interval = int(time_spec["checkpoint_interval"])
    checkpoint = output_dir / "checkpoint.npz"
    start = time.perf_counter()
    completion_estimator = _RollingCompletionEstimator()
    stopped_early = False
    steps_this_run = 0
    last_timestep: float | None = None
    pending_step = simulation.prepare_step(state)
    ready_rates: np.ndarray | None = None

    while state.time < end_time:
        prepared_step = pending_step
        if ready_rates is None:
            rates = np.asarray(prepared_step.rates)
        else:
            rates = ready_rates
            ready_rates = None
        if (
            steps_this_run > 0
            and max_run_seconds is not None
            and time.perf_counter() - start >= max_run_seconds
        ):
            stopped_early = True
            if not history or int(history[-1][0]) != state.step:
                assert last_timestep is not None
                diagnostics = np.asarray(simulation.diagnostic_metrics(state))
                history.append(
                    _diagnostic_row_from_metrics(
                        config,
                        state,
                        last_timestep,
                        rates,
                        diagnostics,
                    )
                )
            break

        timestep = min(
            simulation.timestep_from_metrics(rates),
            end_time - state.time,
        )
        state = simulation.step(state, timestep, prepared_step)
        steps_this_run += 1
        last_timestep = timestep
        pending_step = simulation.prepare_step(state)

        sample_due = state.time + 1.0e-12 >= sample_start and (
            state.step % int(sample_interval) == 0
            if sample_basis == "step"
            else state.time + 1.0e-12 >= next_sample_time
        )
        final = state.time >= end_time
        step_limit = max_steps is not None and state.step >= max_steps
        history_due = state.step % history_interval == 0
        log_due = state.step % log_interval == 0 or final
        row: tuple[float, ...] | None = None
        if history_due or log_due or final or step_limit:
            device_rates, device_diagnostics = jax.device_get(
                (pending_step.rates, simulation.diagnostic_metrics(state))
            )
            ready_rates = np.asarray(device_rates)
            diagnostics = np.asarray(device_diagnostics)
            row = _diagnostic_row_from_metrics(
                config,
                state,
                timestep,
                ready_rates,
                diagnostics,
            )
            completion_estimator.observe(
                wall_time=time.perf_counter(),
                step=state.step,
                physical_time=state.time,
            )
            if (history_due or final or step_limit) and (
                not history or int(history[-1][0]) != state.step
            ):
                history.append(row)

        if sample_due or final:
            samples.append(_snapshot(simulation, state))
            sample_times.append(state.time)
            while next_sample_time <= state.time + 1.0e-12:
                next_sample_time += sample_interval
        if log_due:
            if row is None:
                raise RuntimeError("log metrics were not materialized")
            scale = float(config.data.get("display", {}).get("time_scale", 1.0))
            label = str(config.data.get("display", {}).get("time_label", "t"))
            completion = completion_estimator.estimate(remaining=end_time - state.time)
            print(
                f"step={state.step} {label}={state.time / scale:.5g}/"
                f"{end_time / scale:.5g} CFL={row[3]:.4f} "
                f"CFLnu={row[4]:.4f} divL2={row[5]:.3e} {completion}",
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
        if step_limit:
            stopped_early = state.time < end_time
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
                "time.history_interval=1",
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
