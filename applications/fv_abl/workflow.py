"""Run FFT warmup/precursor and open-boundary GMG main FV stages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from .config import FiniteVolumeCase, load_fv_abl
from .diagnostics import initial_fields
from .evaluate import resolved


@dataclass(frozen=True, slots=True)
class WorkflowOptions:
    """Stage lengths and storage for the one-plane offline precursor path."""

    warmup_steps: int
    precursor_steps: int
    main_steps: int
    record_plane: int
    chunk_steps: int
    output_directory: Path
    precursor_dt_seconds: float | None = None
    main_dt_seconds: float | None = None
    main_frame_count: int = 0
    main_pressure_force: bool = True
    evolve_scalar: bool = True


@dataclass(frozen=True, slots=True)
class OpenFastAdbemOptions:
    """Physical placement and fixed operating point for an OpenFAST rotor."""

    model: str
    model_environment: str | None
    x_m: float
    y_m: float
    hub_height_m: float
    rotor_speed_rpm: float
    blade_pitch_degrees: float
    smoothing_width_m: float
    smearing_azimuthal_elements: int
    body_smoothing_width_m: float
    nacelle_drag_coefficient: float
    tower_drag_coefficient: float


@dataclass(frozen=True, slots=True)
class FiniteVolumeWorkflow:
    case: FiniteVolumeCase
    options: WorkflowOptions
    turbine: OpenFastAdbemOptions | None = None

    def resolved(self) -> dict[str, Any]:
        grid = self.case.physical.physical_grid
        return {
            "schema": "jaxwind.fv-precursor-main.v1",
            "case": resolved(self.case),
            "warmup": {
                "pressure_backend": "fft",
                "periodic_x": True,
                "steps": self.options.warmup_steps,
                "duration_seconds": (
                    self.options.warmup_steps * self.case.physical.dt_seconds
                ),
                "maximum_dt_seconds": self.case.physical.dt_seconds,
                "cfl_ceiling": self.case.options.cfl_ceiling,
                "checkpoint": "warmup_checkpoint.npz",
            },
            "precursor": {
                "pressure_backend": "fft",
                "periodic_x": True,
                "steps": self.options.precursor_steps,
                "duration_seconds": (
                    self.options.precursor_steps
                    * (
                        self.options.precursor_dt_seconds
                        or self.case.physical.dt_seconds
                    )
                ),
                "dt_seconds": self.options.precursor_dt_seconds,
                "adaptive_cfl_ceiling": (
                    self.case.options.cfl_ceiling
                    if self.options.precursor_dt_seconds is None
                    else None
                ),
                "record_plane": self.options.record_plane,
                "stored_x_layers_per_sample": 1,
                "sample_every_steps": 1,
                "directory": "precursor_inflow",
            },
            "main": {
                "pressure_backend": "gmg",
                "periodic_x": False,
                "steps": self.options.main_steps,
                "duration_seconds": (
                    self.options.main_steps
                    * (
                        self.options.main_dt_seconds
                        or self.options.precursor_dt_seconds
                        or self.case.physical.dt_seconds
                    )
                ),
                "dt_seconds": (
                    self.options.main_dt_seconds
                    or self.options.precursor_dt_seconds
                    or self.case.physical.dt_seconds
                ),
                "time_integration": self.case.options.time_integration,
                "frame_count": self.options.main_frame_count,
                "pressure_force": self.options.main_pressure_force,
                "evolve_scalar": self.options.evolve_scalar,
                "inflow": "one recorded yz layer per step",
                "outflow": "second-order zero-gradient transported fields",
                "pressure_boundary": "inlet Neumann, outlet Dirichlet",
                "x_velocity_faces": grid.nx + 1,
            },
            "chunk_steps": self.options.chunk_steps,
            "output_directory": str(self.options.output_directory),
            "turbine": (
                None
                if self.turbine is None
                else {
                    "model": self.turbine.model,
                    "model_environment": self.turbine.model_environment,
                    "x_m": self.turbine.x_m,
                    "y_m": self.turbine.y_m,
                    "hub_height_m": self.turbine.hub_height_m,
                    "rotor_speed_rpm": self.turbine.rotor_speed_rpm,
                    "blade_pitch_degrees": self.turbine.blade_pitch_degrees,
                    "smearing_azimuthal_elements": (
                        self.turbine.smearing_azimuthal_elements
                    ),
                    "nacelle_and_tower": True,
                }
            ),
        }


def _positive_integer(table: dict[str, Any], key: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"finite_volume_workflow.{key} must be positive")
    return value


def _finite_number(table: dict[str, Any], key: str) -> float:
    import math

    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"finite_volume_turbine.{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"finite_volume_turbine.{key} must be finite")
    return result


def _load_turbine(document: dict[str, Any]) -> OpenFastAdbemOptions | None:
    table = document.get("finite_volume_turbine")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError("[finite_volume_turbine] must be a table")
    model = table.get("model")
    if model not in ("openfast-ad-bem", "hitsz-r9-ad-bem"):
        raise ValueError(
            "finite_volume_turbine.model must be openfast-ad-bem "
            "or hitsz-r9-ad-bem"
        )
    common = {
        "model",
        "x_m",
        "y_m",
        "hub_height_m",
        "rotor_speed_rpm",
        "blade_pitch_degrees",
        "smoothing_width_m",
        "smearing_azimuthal_elements",
        "body_smoothing_width_m",
        "nacelle_drag_coefficient",
        "tower_drag_coefficient",
    }
    expected = (
        common | {"openfast_model_environment"}
        if model == "openfast-ad-bem"
        else common
    )
    missing = expected - table.keys()
    unknown = table.keys() - expected
    if missing:
        raise ValueError(
            "[finite_volume_turbine] is missing: "
            + ", ".join(sorted(missing))
        )
    if unknown:
        raise ValueError(
            "[finite_volume_turbine] has unknown keys: "
            + ", ".join(sorted(unknown))
        )
    environment = table.get("openfast_model_environment")
    if model == "openfast-ad-bem" and (
        not isinstance(environment, str) or not environment
    ):
        raise ValueError(
            "finite_volume_turbine.openfast_model_environment must be a string"
        )
    smearing = table["smearing_azimuthal_elements"]
    if (
        isinstance(smearing, bool)
        or not isinstance(smearing, int)
        or smearing <= 0
    ):
        raise ValueError(
            "finite_volume_turbine.smearing_azimuthal_elements must be positive"
        )
    result = OpenFastAdbemOptions(
        model=model,
        model_environment=environment,
        x_m=_finite_number(table, "x_m"),
        y_m=_finite_number(table, "y_m"),
        hub_height_m=_finite_number(table, "hub_height_m"),
        rotor_speed_rpm=_finite_number(table, "rotor_speed_rpm"),
        blade_pitch_degrees=_finite_number(table, "blade_pitch_degrees"),
        smoothing_width_m=_finite_number(table, "smoothing_width_m"),
        smearing_azimuthal_elements=smearing,
        body_smoothing_width_m=_finite_number(table, "body_smoothing_width_m"),
        nacelle_drag_coefficient=_finite_number(
            table, "nacelle_drag_coefficient"
        ),
        tower_drag_coefficient=_finite_number(
            table, "tower_drag_coefficient"
        ),
    )
    if min(
        result.hub_height_m,
        result.rotor_speed_rpm,
        result.smoothing_width_m,
        result.body_smoothing_width_m,
    ) <= 0.0:
        raise ValueError("finite-volume turbine dimensions and speed must be positive")
    return result

def load_workflow(path: str | Path) -> FiniteVolumeWorkflow:
    """Load the physical case, FV numerics, and strict stage workflow."""
    source = Path(path)
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    table = document.get("finite_volume_workflow")
    if not isinstance(table, dict):
        raise ValueError("missing [finite_volume_workflow] table")
    expected = {
        "warmup_steps",
        "precursor_steps",
        "main_steps",
        "record_plane",
        "chunk_steps",
        "output_directory",
    }
    missing = expected - table.keys()
    unknown = table.keys() - expected - {
        "precursor_dt_seconds",
        "main_dt_seconds",
        "main_frame_count",
        "main_pressure_force",
        "evolve_scalar",
    }
    if missing:
        raise ValueError(
            "[finite_volume_workflow] is missing: "
            + ", ".join(sorted(missing))
        )
    if unknown:
        raise ValueError(
            "[finite_volume_workflow] has unknown keys: "
            + ", ".join(sorted(unknown))
        )
    output = table["output_directory"]
    if not isinstance(output, str) or not output:
        raise ValueError(
            "finite_volume_workflow.output_directory must be a non-empty string"
        )
    record_plane = table["record_plane"]
    if isinstance(record_plane, bool) or not isinstance(record_plane, int):
        raise ValueError("finite_volume_workflow.record_plane must be an integer")
    case = load_fv_abl(source)
    grid = case.physical.physical_grid
    if not 0 <= record_plane < grid.nx:
        raise ValueError("finite_volume_workflow.record_plane is outside the mesh")
    options = WorkflowOptions(
        warmup_steps=_positive_integer(table, "warmup_steps"),
        precursor_steps=_positive_integer(table, "precursor_steps"),
        main_steps=_positive_integer(table, "main_steps"),
        record_plane=record_plane,
        chunk_steps=_positive_integer(table, "chunk_steps"),
        output_directory=Path(output),
        precursor_dt_seconds=(
            _finite_number(table, "precursor_dt_seconds")
            if "precursor_dt_seconds" in table
            else None
        ),
        main_dt_seconds=(
            _finite_number(table, "main_dt_seconds")
            if "main_dt_seconds" in table
            else None
        ),
        main_frame_count=table.get("main_frame_count", 0),
        main_pressure_force=table.get("main_pressure_force", True),
        evolve_scalar=table.get("evolve_scalar", True),
    )
    if (
        options.precursor_dt_seconds is not None
        and options.precursor_dt_seconds <= 0.0
    ):
        raise ValueError(
            "finite_volume_workflow.precursor_dt_seconds must be positive"
        )
    if options.main_dt_seconds is not None and options.main_dt_seconds <= 0.0:
        raise ValueError(
            "finite_volume_workflow.main_dt_seconds must be positive"
        )
    if (
        isinstance(options.main_frame_count, bool)
        or not isinstance(options.main_frame_count, int)
        or options.main_frame_count < 0
        or options.main_frame_count > options.main_steps
    ):
        raise ValueError(
            "finite_volume_workflow.main_frame_count must be between "
            "zero and main_steps"
        )
    if not isinstance(options.main_pressure_force, bool):
        raise ValueError(
            "finite_volume_workflow.main_pressure_force must be boolean"
        )
    if not isinstance(options.evolve_scalar, bool):
        raise ValueError(
            "finite_volume_workflow.evolve_scalar must be boolean"
        )
    if options.main_steps > options.precursor_steps:
        raise ValueError("main_steps cannot exceed recorded precursor_steps")
    return FiniteVolumeWorkflow(case, options, _load_turbine(document))


def _models(
    configured: FiniteVolumeCase,
    *,
    periodic_x: bool,
    forcing=None,
    pressure_force_enabled: bool = True,
    evolve_scalar: bool = True,
):
    """Compose identical physical closures for each workflow stage."""
    from jaxwind.fv import (
        AnisotropicMinimumDissipation,
        CELL_CENTRE,
        LOCAL,
        OPEN,
        Boundaries,
        CoriolisGeostrophic,
        FlowModel,
        LinearBoussinesqBuoyancy,
        MoninObukhovSurface,
        MoninObukhovWall,
        PassiveScalar,
        monin_obukhov_boundaries,
    )

    case = configured.physical
    configuration = resolved(configured)
    offset_u, offset_v = case.advection_frame_velocity_m_s
    vertical_f = configuration["coriolis_vertical_s"]
    rotation = None
    if vertical_f != 0.0:
        evolved = configuration["evolved_geostrophic_velocity_m_s"]
        rotation = CoriolisGeostrophic(
            vertical_f,
            evolved[0],
            evolved[1],
            configuration["coriolis_horizontal_s"],
        )
    coupled = case.model.surface_transfer
    if not hasattr(coupled, "scalar_roughness_length"):
        coupled = None
    wall = None
    surface = None
    if coupled is None:
        wall = MoninObukhovWall(
            configuration["roughness_length_m"],
            von_karman=case.model.momentum.wall.von_karman,
            sampling=CELL_CENTRE,
            averaging=LOCAL,
        )
    else:
        coefficient = configuration["buoyancy_acceleration_per_scalar"]
        surface = MoninObukhovSurface(
            momentum_roughness=configuration["momentum_roughness_m"],
            scalar_roughness=configuration["scalar_roughness_m"],
            surface_scalar_initial=configuration["surface_scalar_initial"],
            surface_scalar_rate=configuration["surface_scalar_rate_per_second"],
            x_velocity_offset=offset_u,
            y_velocity_offset=offset_v,
            buoyancy_coefficient=coefficient,
            von_karman=case.model.momentum.wall.von_karman,
            positive_zeta_momentum_slope=coupled.positive_zeta_momentum_slope,
            positive_zeta_scalar_slope=coupled.positive_zeta_scalar_slope,
            negative_zeta_momentum_coefficient=(
                coupled.negative_zeta_momentum_coefficient
            ),
            negative_zeta_scalar_coefficient=(
                coupled.negative_zeta_scalar_coefficient
            ),
            iterations=coupled.iterations,
            relaxation=coupled.relaxation,
            maximum_abs_zeta=coupled.maximum_abs_zeta,
        )
    pressure_force = configuration["pressure_acceleration_m_s2"]
    if not pressure_force_enabled:
        pressure_force = (0.0, 0.0)
    momentum = FlowModel(
        body_force=(pressure_force[0], pressure_force[1], 0.0),
        forcing=forcing,
        subfilter=AnisotropicMinimumDissipation(),
        surface=wall,
        rotation=rotation,
    )
    scalar = (
        PassiveScalar(
            lower_flux=configuration["scalar_surface_flux"],
            turbulent_prandtl=configured.options.turbulent_prandtl,
        )
        if evolve_scalar
        else None
    )
    coefficient = configuration["buoyancy_acceleration_per_scalar"]
    buoyancy = (
        LinearBoussinesqBuoyancy(coefficient) if coefficient != 0.0 else None
    )
    boundaries = monin_obukhov_boundaries()
    if not periodic_x:
        boundaries = Boundaries(
            boundaries.lower,
            boundaries.upper,
            streamwise=OPEN,
        )
    return boundaries, momentum, scalar, buoyancy, surface


def _openfast_path(options: OpenFastAdbemOptions) -> Path:
    environment = options.model_environment
    if environment is None:
        raise ValueError("the native HITSZ rotor does not use an OpenFAST path")
    value = os.environ.get(environment)
    if not value:
        raise ValueError(f"set {environment} to the OpenFAST .fst model")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"OpenFAST model does not exist: {path}")
    return path


def _build_turbine_definition(workflow: FiniteVolumeWorkflow):
    options = workflow.turbine
    if options is None:
        return None
    from jaxwind.windfarm import (
        HITSZR9BladeElementDisk,
        RigidBladeElementDisk,
        load_openfast_rigid_turbine,
    )

    common = {
        "x_m": options.x_m,
        "y_m": options.y_m,
        "smoothing_width_m": options.smoothing_width_m,
        "hub_height_m": options.hub_height_m,
        "rotor_speed_rpm": options.rotor_speed_rpm,
        "pitch_degrees": options.blade_pitch_degrees,
        "smearing_azimuthal_elements": options.smearing_azimuthal_elements,
        "body_smoothing_width_m": options.body_smoothing_width_m,
        "nacelle_drag_coefficient": options.nacelle_drag_coefficient,
        "tower_drag_coefficient": options.tower_drag_coefficient,
    }
    if options.model == "hitsz-r9-ad-bem":
        turbine = HITSZR9BladeElementDisk(**common)
    else:
        rotor = load_openfast_rigid_turbine(_openfast_path(options))
        turbine = RigidBladeElementDisk(rotor=rotor, **common)

    from jaxwind.domain import ScaleSystem

    grid = workflow.case.physical.physical_grid
    disk = turbine.to_actuator_disk(scales=ScaleSystem(1.0, 1.0))
    if not 0.0 < disk.x < grid.lx:
        raise ValueError("finite-volume turbine x position is outside the domain")
    if not 0.0 < disk.y < grid.ly:
        raise ValueError("finite-volume turbine y position is outside the domain")
    if disk.z + disk.tip_radius >= grid.lz:
        raise ValueError("finite-volume turbine rotor intersects the upper boundary")
    return turbine


def _build_turbine_forcing(workflow: FiniteVolumeWorkflow):
    turbine = _build_turbine_definition(workflow)
    if turbine is None:
        return None
    from jaxwind.domain import ScaleSystem
    from jaxwind.fv import build_adbem_forcing

    scales = ScaleSystem(1.0, 1.0)
    return build_adbem_forcing(
        workflow.case.physical.physical_grid,
        turbine.to_actuator_disk(scales=scales),
        turbine.to_nacelle_tower(scales=scales),
    )

def _initial_periodic(configured: FiniteVolumeCase, jax, jnp):
    from jaxwind.fv import (
        StaggeredVelocity,
        build_pressure_poisson,
        initial_atmospheric_solution,
        project,
    )

    case = configured.physical
    grid = case.physical_grid
    u, v, w, scalar = initial_fields(case, jax, jnp)
    offset_u, offset_v = case.advection_frame_velocity_m_s
    velocity = StaggeredVelocity(u - offset_u, v - offset_v, w)
    poisson = build_pressure_poisson(
        grid,
        backend="fft",
        dtype=case.pressure.dtype,
    )
    velocity, _ = project(velocity, poisson, 1.0)
    return initial_atmospheric_solution(
        grid,
        velocity,
        scalar,
        dtype=case.pressure.dtype,
    )


def _periodic_advance(configured: FiniteVolumeCase):
    from jaxwind.fv import (
        build_atmospheric_run,
        build_atmospheric_step,
        build_pressure_poisson,
    )

    case = configured.physical
    grid = case.physical_grid
    boundaries, momentum, scalar, buoyancy, surface = _models(
        configured, periodic_x=True
    )
    poisson = build_pressure_poisson(
        grid,
        backend="fft",
        dtype=case.pressure.dtype,
    )
    step = build_atmospheric_step(
        grid,
        boundaries,
        poisson,
        momentum,
        scalar,
        buoyancy,
        surface,
        scheme=configured.options.time_integration,
    )
    return step, build_atmospheric_run(step)


def _save_solution(path: Path, solution) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        velocity_x=np.asarray(solution.velocity.x),
        velocity_y=np.asarray(solution.velocity.y),
        velocity_z=np.asarray(solution.velocity.z),
        pressure=np.asarray(solution.pressure),
        momentum_tendency_x=np.asarray(solution.momentum_tendency.x),
        momentum_tendency_y=np.asarray(solution.momentum_tendency.y),
        momentum_tendency_z=np.asarray(solution.momentum_tendency.z),
        scalar=np.asarray(solution.scalar),
        scalar_tendency=np.asarray(solution.scalar_tendency),
        time=np.asarray(solution.time),
        step=np.asarray(solution.step),
    )


def _load_solution(path: Path, jnp):
    from jaxwind.fv import AtmosphericSolution, StaggeredVelocity

    if not path.exists():
        raise FileNotFoundError(f"missing workflow checkpoint: {path}")
    with np.load(path) as data:
        return AtmosphericSolution(
            StaggeredVelocity(
                jnp.asarray(data["velocity_x"]),
                jnp.asarray(data["velocity_y"]),
                jnp.asarray(data["velocity_z"]),
            ),
            jnp.asarray(data["pressure"]),
            StaggeredVelocity(
                jnp.asarray(data["momentum_tendency_x"]),
                jnp.asarray(data["momentum_tendency_y"]),
                jnp.asarray(data["momentum_tendency_z"]),
            ),
            jnp.asarray(data["scalar"]),
            jnp.asarray(data["scalar_tendency"]),
            jnp.asarray(data["time"]),
            jnp.asarray(data["step"]),
        )


def _run_periodic_blocks(
    solution,
    advance,
    *,
    grid,
    dt: float,
    steps: int,
    chunk: int,
):
    import jax
    from jaxwind.fv import courant_number

    completed = 0
    started = time.perf_counter()
    maximum_cfl = 0.0
    final_cfl = 0.0
    while completed < steps:
        count = min(chunk, steps - completed)
        solution = advance(solution, dt, count)
        jax.block_until_ready(solution.velocity.x)
        completed += count
        final_cfl = float(courant_number(solution.velocity, grid, dt))
        maximum_cfl = max(maximum_cfl, final_cfl)
        print(
            f"periodic {completed:8d}/{steps} CFL {final_cfl:.3f}",
            flush=True,
        )
    return solution, time.perf_counter() - started, final_cfl, maximum_cfl


def _run_adaptive_periodic_blocks(
    solution,
    advance,
    *,
    grid,
    maximum_dt: float,
    cfl_ceiling: float,
    duration_seconds: float,
    chunk: int,
):
    import jax
    import jax.numpy as jnp
    from jaxwind.fv import courant_number, stable_timestep

    initial_time = float(solution.time)
    target_time = initial_time + duration_seconds
    started = time.perf_counter()
    initial_step = int(solution.step)
    maximum_sampled_cfl = 0.0
    minimum_block_dt = float("inf")
    maximum_block_dt = 0.0
    tolerance = 8.0 * np.finfo(np.float32).eps * max(1.0, target_time)
    next_dt = maximum_dt
    next_cfl = 0.0
    while float(solution.time) < target_time - tolerance:
        before_time = float(solution.time)
        before_step = int(solution.step)
        solution = advance(solution, target_time, chunk)
        jax.block_until_ready(solution.velocity.x)
        after_time = float(solution.time)
        after_step = int(solution.step)
        active_steps = after_step - before_step
        if active_steps <= 0 or after_time <= before_time:
            raise RuntimeError("adaptive RK block made no progress")
        block_dt = (after_time - before_time) / active_steps
        minimum_block_dt = min(minimum_block_dt, block_dt)
        maximum_block_dt = max(maximum_block_dt, block_dt)
        next_dt = float(
            jnp.minimum(
                maximum_dt,
                stable_timestep(
                    solution.velocity,
                    grid,
                    0.0,
                    courant=cfl_ceiling,
                ),
            )
        )
        next_cfl = float(courant_number(solution.velocity, grid, next_dt))
        maximum_sampled_cfl = max(maximum_sampled_cfl, next_cfl)
        print(
            f"periodic step {after_step:8d} "
            f"time {after_time:9.3f}/{target_time:.3f} s "
            f"dt {next_dt:.6f} CFL {next_cfl:.3f}",
            flush=True,
        )
    statistics = {
        "steps": int(solution.step) - initial_step,
        "duration_seconds": float(solution.time) - initial_time,
        "minimum_block_mean_dt_seconds": minimum_block_dt,
        "maximum_block_mean_dt_seconds": maximum_block_dt,
        "final_candidate_dt_seconds": next_dt,
        "final_candidate_cfl": next_cfl,
        "maximum_sampled_candidate_cfl": maximum_sampled_cfl,
    }
    return solution, time.perf_counter() - started, statistics


def run_warmup(workflow: FiniteVolumeWorkflow, *, steps: int) -> dict[str, Any]:
    import jax

    case = workflow.case.physical
    options = workflow.case.options
    jax.config.update("jax_enable_x64", case.pressure.dtype == "float64")
    import jax.numpy as jnp

    solution = _initial_periodic(workflow.case, jax, jnp)
    step, fixed_advance = _periodic_advance(workflow.case)
    duration_seconds = steps * case.dt_seconds
    if options.cfl_ceiling is None:
        solution, elapsed, final_cfl, maximum_cfl = _run_periodic_blocks(
            solution,
            fixed_advance,
            grid=case.physical_grid,
            dt=case.dt_seconds,
            steps=steps,
            chunk=workflow.options.chunk_steps,
        )
        statistics = {
            "steps": steps,
            "duration_seconds": duration_seconds,
            "fixed_dt_seconds": case.dt_seconds,
            "final_cfl": final_cfl,
            "maximum_sampled_cfl": maximum_cfl,
        }
    else:
        from jaxwind.fv import build_adaptive_atmospheric_run

        adaptive_advance = build_adaptive_atmospheric_run(
            step,
            case.physical_grid,
            cfl_ceiling=options.cfl_ceiling,
            maximum_dt=case.dt_seconds,
        )
        solution, elapsed, statistics = _run_adaptive_periodic_blocks(
            solution,
            adaptive_advance,
            grid=case.physical_grid,
            maximum_dt=case.dt_seconds,
            cfl_ceiling=options.cfl_ceiling,
            duration_seconds=duration_seconds,
            chunk=workflow.options.chunk_steps,
        )
    path = workflow.options.output_directory / "warmup_checkpoint.npz"
    _save_solution(path, solution)
    actual_steps = int(statistics["steps"])
    return {
        **statistics,
        "elapsed_seconds": elapsed,
        "steps_per_second": actual_steps / elapsed,
        "maximum_dt_seconds": case.dt_seconds,
        "cfl_ceiling": options.cfl_ceiling,
        "pressure_backend": "fft",
        "time_integration": options.time_integration,
        "checkpoint": str(path),
    }

def run_precursor(workflow: FiniteVolumeWorkflow, *, steps: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from jaxwind.fv import (
        build_adaptive_atmospheric_run,
        extract_inflow_plane,
        stable_timestep,
    )

    case = workflow.case.physical
    options = workflow.case.options
    grid = case.physical_grid
    checkpoint = workflow.options.output_directory / "warmup_checkpoint.npz"
    warm = _load_solution(checkpoint, jnp)
    step, _fixed_advance = _periodic_advance(workflow.case)
    fixed_dt = workflow.options.precursor_dt_seconds
    stage_dt = case.dt_seconds if fixed_dt is None else fixed_dt
    adaptive = fixed_dt is None and options.cfl_ceiling is not None
    duration_seconds = steps * stage_dt
    initial_time = float(warm.time)
    target_time = initial_time + duration_seconds

    preflight_elapsed = 0.0
    if not adaptive:
        samples = steps
    else:
        adaptive_advance = build_adaptive_atmospheric_run(
            step,
            grid,
            cfl_ceiling=options.cfl_ceiling,
            maximum_dt=case.dt_seconds,
        )
        _, preflight_elapsed, preflight = _run_adaptive_periodic_blocks(
            warm,
            adaptive_advance,
            grid=grid,
            maximum_dt=case.dt_seconds,
            cfl_ceiling=options.cfl_ceiling,
            duration_seconds=duration_seconds,
            chunk=workflow.options.chunk_steps,
        )
        samples = int(preflight["steps"])
        print(
            f"precursor allocation samples={samples} "
            f"duration={duration_seconds:.3f}s",
            flush=True,
        )

    directory = workflow.options.output_directory / "precursor_inflow"
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x_velocity": np.lib.format.open_memmap(
            directory / "x_velocity.npy",
            mode="w+",
            dtype=case.pressure.dtype,
            shape=(samples, grid.nz, grid.ny),
        ),
        "y_velocity": np.lib.format.open_memmap(
            directory / "y_velocity.npy",
            mode="w+",
            dtype=case.pressure.dtype,
            shape=(samples, grid.nz, grid.ny),
        ),
        "z_velocity": np.lib.format.open_memmap(
            directory / "z_velocity.npy",
            mode="w+",
            dtype=case.pressure.dtype,
            shape=(samples, grid.nz + 1, grid.ny),
        ),
        "scalar": np.lib.format.open_memmap(
            directory / "scalar.npy",
            mode="w+",
            dtype=case.pressure.dtype,
            shape=(samples, grid.nz, grid.ny),
        ),
    }
    timesteps = np.lib.format.open_memmap(
        directory / "dt_seconds.npy",
        mode="w+",
        dtype=case.pressure.dtype,
        shape=(samples,),
    )
    compiled: dict[int, Any] = {}

    def block(count: int):
        if count not in compiled:
            def scan(current, first_step):
                def advance(state, local_step):
                    if not adaptive:
                        exact_time = initial_time + local_step * stage_dt
                        state = state._replace(
                            time=jnp.asarray(exact_time, state.time.dtype)
                        )
                    plane = extract_inflow_plane(
                        state,
                        grid,
                        workflow.options.record_plane,
                    )
                    if not adaptive:
                        dt = jnp.asarray(stage_dt, state.time.dtype)
                    else:
                        remaining = jnp.asarray(target_time, state.time.dtype) - state.time
                        dt = jnp.minimum(
                            jnp.minimum(
                                case.dt_seconds,
                                stable_timestep(
                                    state.velocity,
                                    grid,
                                    0.0,
                                    courant=options.cfl_ceiling,
                                ),
                            ),
                            remaining,
                        )
                    advanced = step(state, dt)
                    if not adaptive:
                        exact_end_time = (
                            initial_time + (local_step + 1) * stage_dt
                        )
                        advanced = advanced._replace(
                            time=jnp.asarray(exact_end_time, state.time.dtype)
                        )
                    return advanced, (plane, dt)

                scan_steps = (
                    None
                    if adaptive
                    else first_step + jnp.arange(count, dtype=jnp.int32)
                )
                return jax.lax.scan(
                    advance, current, scan_steps, length=count
                )

            compiled[count] = jax.jit(scan)
        return compiled[count]

    solution = warm
    completed = 0
    started = time.perf_counter()
    while completed < samples:
        count = min(workflow.options.chunk_steps, samples - completed)
        solution, recorded = block(count)(
            solution, jnp.asarray(completed, jnp.int32)
        )
        planes, block_timesteps = jax.device_get(recorded)
        stop = completed + count
        for name in arrays:
            arrays[name][completed:stop] = np.asarray(getattr(planes, name))
        timesteps[completed:stop] = np.asarray(block_timesteps)
        completed = stop
        print(
            f"precursor {completed:8d}/{samples} "
            f"time {float(solution.time) - initial_time:8.3f}/{duration_seconds:.3f}s",
            flush=True,
        )
    recording_elapsed = time.perf_counter() - started
    for array in arrays.values():
        array.flush()
    timesteps.flush()
    actual_duration = float(solution.time) - initial_time
    if not np.isclose(actual_duration, duration_seconds, rtol=0.0, atol=1.0e-4):
        raise RuntimeError(
            f"precursor ended at {actual_duration}s, expected {duration_seconds}s"
        )
    timestep_values = np.asarray(timesteps, dtype=np.float64)
    metadata = {
        "schema": "jaxwind.fv-inflow-plane.v2",
        "samples": samples,
        "sample_every_steps": 1,
        "variable_timestep": adaptive,
        "dt_seconds": None if adaptive else stage_dt,
        "dt_seconds_file": "dt_seconds.npy",
        "minimum_dt_seconds": float(np.min(timestep_values)),
        "maximum_dt_seconds": float(np.max(timestep_values)),
        "duration_seconds": actual_duration,
        "start_time_seconds": initial_time,
        "end_time_seconds": float(solution.time),
        "cfl_ceiling": options.cfl_ceiling if adaptive else None,
        "configured_cfl_ceiling": options.cfl_ceiling,
        "record_plane": workflow.options.record_plane,
        "stored_x_layers_per_sample": 1,
        "components": list(arrays),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    _save_solution(
        workflow.options.output_directory / "precursor_final.npz",
        solution,
    )
    return {
        "steps": samples,
        "duration_seconds": actual_duration,
        "preflight_elapsed_seconds": preflight_elapsed,
        "recording_elapsed_seconds": recording_elapsed,
        "elapsed_seconds": preflight_elapsed + recording_elapsed,
        "recording_steps_per_second": samples / recording_elapsed,
        "recording": str(directory),
        "stored_x_layers_per_sample": 1,
        "variable_timestep": adaptive,
        "minimum_dt_seconds": metadata["minimum_dt_seconds"],
        "maximum_dt_seconds": metadata["maximum_dt_seconds"],
        "cfl_ceiling": options.cfl_ceiling if adaptive else None,
        "configured_cfl_ceiling": options.cfl_ceiling,
    }

def _load_inflow_block(directory: Path, start: int, stop: int, jnp):
    from jaxwind.fv import InflowPlane

    return InflowPlane(
        jnp.asarray(np.load(directory / "x_velocity.npy", mmap_mode="r")[start:stop]),
        jnp.asarray(np.load(directory / "y_velocity.npy", mmap_mode="r")[start:stop]),
        jnp.asarray(np.load(directory / "z_velocity.npy", mmap_mode="r")[start:stop]),
        jnp.asarray(np.load(directory / "scalar.npy", mmap_mode="r")[start:stop]),
    )


def _main_frame_steps(steps: int, count: int) -> tuple[int, ...]:
    """Return unique, evenly spaced one-based capture steps."""
    if count == 0:
        return ()
    return tuple(
        int(value)
        for value in ((np.arange(count, dtype=np.int64) + 1) * steps // count)
    )


def _build_main_frame_capture(grid, *, y_m: float, z_m: float):
    """Compile extraction of the two saved streamwise slices on the device."""
    import jax

    z_index = np.clip(z_m / grid.dz - 0.5, 0.0, grid.nz - 1.0)
    z_lower = int(np.floor(z_index))
    z_upper = min(z_lower + 1, grid.nz - 1)
    z_weight = z_index - z_lower
    y_index = y_m / grid.dy - 0.5
    y_floor = np.floor(y_index)
    y_lower = int(y_floor) % grid.ny
    y_upper = (y_lower + 1) % grid.ny
    y_weight = y_index - y_floor

    def capture(x_faces):
        hub_faces = (
            (1.0 - z_weight) * x_faces[z_lower]
            + z_weight * x_faces[z_upper]
        )
        centre_faces = (
            (1.0 - y_weight) * x_faces[:, y_lower]
            + y_weight * x_faces[:, y_upper]
        )
        return (
            0.5 * (hub_faces[:, :-1] + hub_faces[:, 1:]),
            0.5 * (centre_faces[:, :-1] + centre_faces[:, 1:]),
        )

    return jax.jit(capture)


def _capture_main_frame(
    solution,
    grid,
    *,
    y_m: float,
    z_m: float,
    capture=None,
) -> dict[str, Any]:
    """Copy only the two saved streamwise slices from device to host."""
    if capture is None:
        capture = _build_main_frame_capture(grid, y_m=y_m, z_m=z_m)
    hub, centre = capture(solution.velocity.x)
    return {
        "u_hub_yx": np.asarray(hub),
        "u_center_zx": np.asarray(centre),
        "time_seconds": float(solution.time),
        "step": int(solution.step),
    }


def run_main(workflow: FiniteVolumeWorkflow, *, steps: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from jaxwind.fv import (
        build_open_atmospheric_run,
        build_open_atmospheric_step,
        build_pressure_poisson,
        divergence,
        enforce_open_scalar,
        enforce_open_velocity,
        initial_atmospheric_solution,
        periodic_to_open_velocity,
    )

    case = workflow.case.physical
    grid = case.physical_grid
    recording = workflow.options.output_directory / "precursor_inflow"
    metadata_path = recording / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing precursor recording: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["stored_x_layers_per_sample"] != 1:
        raise ValueError("FV main requires exactly one recorded inflow layer")
    if steps > int(metadata["samples"]):
        raise ValueError("main steps exceed the precursor recording")
    if bool(metadata.get("variable_timestep", False)):
        raise ValueError("fixed-step FV main requires fixed-step precursor data")
    recorded_dt = metadata.get("dt_seconds")
    stage_dt = (
        workflow.options.main_dt_seconds
        if workflow.options.main_dt_seconds is not None
        else recorded_dt
    )
    if stage_dt is None:
        stage_dt = case.dt_seconds
    stage_dt = float(stage_dt)
    if recorded_dt is not None and not np.isclose(
        stage_dt, float(recorded_dt), rtol=0.0, atol=1.0e-9
    ):
        raise ValueError(
            "main_dt_seconds must match the fixed precursor recording cadence"
        )

    warm = _load_solution(
        workflow.options.output_directory / "warmup_checkpoint.npz",
        jnp,
    )
    first = _load_inflow_block(recording, 0, 1, jnp)
    first = type(first)(*(component[0] for component in first))
    velocity = periodic_to_open_velocity(warm.velocity, grid)
    velocity = enforce_open_velocity(velocity, first, grid)
    scalar_field = enforce_open_scalar(warm.scalar, first, grid)
    solution = initial_atmospheric_solution(
        grid,
        velocity,
        scalar_field,
        dtype=case.pressure.dtype,
    )
    forcing = _build_turbine_forcing(workflow)
    boundaries, momentum, scalar, buoyancy, surface = _models(
        workflow.case,
        periodic_x=False,
        forcing=forcing,
        pressure_force_enabled=workflow.options.main_pressure_force,
        evolve_scalar=workflow.options.evolve_scalar,
    )
    gmg_config = {
        "presweeps": workflow.case.options.gmg_presweeps,
        "postsweeps": workflow.case.options.gmg_postsweeps,
        **(
            {}
            if workflow.case.options.gmg_tolerance is None
            else {"tolerance": workflow.case.options.gmg_tolerance}
        ),
    }
    poisson = build_pressure_poisson(
        grid,
        backend="gmg",
        periodic_x=False,
        dtype=case.pressure.dtype,
        config=gmg_config,
    )
    step = build_open_atmospheric_step(
        grid,
        boundaries,
        poisson,
        momentum,
        scalar,
        buoyancy,
        surface,
        scheme=workflow.case.options.time_integration,
    )
    advance = build_open_atmospheric_run(step)

    frame_steps = _main_frame_steps(
        steps, workflow.options.main_frame_count
    )
    next_frame = 0
    frames: list[dict[str, Any]] = []
    turbine = workflow.turbine
    frame_y = 0.5 * grid.ly if turbine is None else turbine.y_m
    frame_z = 0.5 * grid.lz if turbine is None else turbine.hub_height_m
    capture_main_frame = (
        None
        if not frame_steps
        else _build_main_frame_capture(grid, y_m=frame_y, z_m=frame_z)
    )

    completed = 0
    started = time.perf_counter()
    block_elapsed: list[float] = []
    block_steps: list[int] = []
    while completed < steps:
        block_started = time.perf_counter()
        block_start = completed
        stop = min(completed + workflow.options.chunk_steps, steps)
        if next_frame < len(frame_steps):
            stop = min(stop, frame_steps[next_frame])
        inflows = _load_inflow_block(recording, completed, stop, jnp)
        solution = advance(solution, stage_dt, inflows)
        jax.block_until_ready(solution.velocity.x)
        completed = stop
        if next_frame < len(frame_steps) and completed == frame_steps[next_frame]:
            frames.append(
                _capture_main_frame(
                    solution,
                    grid,
                    y_m=frame_y,
                    z_m=frame_z,
                    capture=capture_main_frame,
                )
            )
            next_frame += 1
        maximum_divergence = float(
            jnp.max(jnp.abs(divergence(solution.velocity, grid)))
        )
        block_elapsed.append(time.perf_counter() - block_started)
        block_steps.append(stop - block_start)
        print(
            f"main {completed:8d}/{steps} div {maximum_divergence:.3e} "
            f"rate {block_steps[-1] / block_elapsed[-1]:.1f} step/s "
            f"frames {len(frames)}/{len(frame_steps)}",
            flush=True,
        )
    elapsed = time.perf_counter() - started

    frame_path = workflow.options.output_directory / "main_flow_frames.npz"
    if frames:
        np.savez_compressed(
            frame_path,
            u_hub_yx=np.stack([frame["u_hub_yx"] for frame in frames]),
            u_center_zx=np.stack([frame["u_center_zx"] for frame in frames]),
            time_seconds=np.asarray(
                [frame["time_seconds"] for frame in frames], dtype=np.float64
            ),
            step=np.asarray([frame["step"] for frame in frames], dtype=np.int64),
            x_m=(np.arange(grid.nx) + 0.5) * grid.dx,
            y_m=(np.arange(grid.ny) + 0.5) * grid.dy,
            z_m=(np.arange(grid.nz) + 0.5) * grid.dz,
            hub_height_m=np.asarray(frame_z),
            center_y_m=np.asarray(frame_y),
            dt_seconds=np.asarray(stage_dt),
        )

    startup_blocks = min(2, len(block_elapsed))
    steady_steps = sum(block_steps[startup_blocks:])
    steady_elapsed = sum(block_elapsed[startup_blocks:])
    steady_rate = (
        steady_steps / steady_elapsed if steady_elapsed > 0.0 else None
    )
    _save_solution(
        workflow.options.output_directory / "main_final.npz",
        solution,
    )
    maximum_divergence = float(
        jnp.max(jnp.abs(divergence(solution.velocity, grid)))
    )
    return {
        "steps": steps,
        "duration_seconds": steps * stage_dt,
        "dt_seconds": stage_dt,
        "time_integration": workflow.case.options.time_integration,
        "elapsed_seconds": elapsed,
        "compile_and_first_block_seconds": block_elapsed[0],
        "startup_blocks": startup_blocks,
        "compile_and_startup_seconds": sum(block_elapsed[:startup_blocks]),
        "steady_steps": steady_steps,
        "steady_elapsed_seconds": steady_elapsed,
        "steady_steps_per_second": steady_rate,
        "steady_cell_updates_per_second": (
            None if steady_rate is None else steady_rate * grid.cell_count
        ),
        "pressure_backend": "gmg",
        "periodic_x": False,
        "maximum_divergence_s": maximum_divergence,
        "inflow_enforcement": "one-layer direct Dirichlet",
        "outflow": "second-order zero-gradient plus fixed pressure",
        "frames": {
            "count": len(frames),
            "path": str(frame_path) if frames else None,
            "fields": ["u_hub_yx", "u_center_zx"],
        },
    }

def execute(
    workflow: FiniteVolumeWorkflow,
    *,
    stage: str = "all",
    max_steps: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute one stage or the complete warmup/precursor/main chain."""
    if stage not in ("warmup", "precursor", "main", "all"):
        raise ValueError("stage must be warmup, precursor, main, or all")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if workflow.turbine is not None and stage in ("main", "all"):
        _build_turbine_definition(workflow)
    output = workflow.options.output_directory
    if output.exists() and any(output.iterdir()) and not overwrite and stage in (
        "warmup",
        "all",
    ):
        raise FileExistsError(
            f"workflow output is not empty: {output}; use --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest = workflow.resolved()
    (output / "resolved_workflow.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    requested = {
        "warmup": workflow.options.warmup_steps,
        "precursor": workflow.options.precursor_steps,
        "main": workflow.options.main_steps,
    }
    effective = {
        name: count if max_steps is None else min(count, max_steps)
        for name, count in requested.items()
    }
    if effective["main"] > effective["precursor"] and stage == "all":
        effective["main"] = effective["precursor"]
    summary: dict[str, Any] = {"schema": manifest["schema"]}
    if stage in ("warmup", "all"):
        summary["warmup"] = run_warmup(workflow, steps=effective["warmup"])
    if stage in ("precursor", "all"):
        summary["precursor"] = run_precursor(
            workflow,
            steps=effective["precursor"],
        )
    if stage in ("main", "all"):
        summary["main"] = run_main(workflow, steps=effective["main"])
    (output / "workflow_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("config", type=Path)
    result.add_argument(
        "--stage",
        choices=("warmup", "precursor", "main", "all"),
        default="all",
    )
    result.add_argument("--max-steps", type=int)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    workflow = load_workflow(arguments.config)
    if arguments.dry_run:
        print(json.dumps(workflow.resolved(), indent=2))
        return 0
    execute(
        workflow,
        stage=arguments.stage,
        max_steps=arguments.max_steps,
        overwrite=arguments.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FiniteVolumeWorkflow",
    "WorkflowOptions",
    "execute",
    "load_workflow",
    "run_main",
    "run_precursor",
    "run_warmup",
]
