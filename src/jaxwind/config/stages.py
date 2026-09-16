"""Run FFT warmup/precursor and open-boundary GMG main FV stages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .abl import FiniteVolumeCase, load_fv_abl
from .abl_resolved import resolved
from .moisture import AtmosphericMoistureOptions, WaterSprayOptions, load_moisture


@dataclass(frozen=True, slots=True)
class WorkflowOptions:
    """Stage lengths and storage for the one-plane offline precursor path."""

    warmup_steps: int
    precursor_steps: int
    main_steps: int
    record_plane: int
    chunk_steps: int
    output_directory: Path
    input_directory: Path | None = None
    precursor_dt_seconds: float | None = None
    precursor_frame_count: int = 0
    main_dt_seconds: float | None = None
    main_frame_count: int = 0
    main_pressure_force: bool = True
    evolve_scalar: bool = True
    main_substeps_per_inflow: int = 1
    warmup_restart_checkpoint: Path | None = None


@dataclass(frozen=True, slots=True)
class TurbineOptions:
    """Physical placement and fixed operating point for AD-BEM or ALM."""

    model: str
    model_environment: str | None
    x_m: float
    y_m: float
    hub_height_m: float
    rotor_speed_rpm: float
    blade_pitch_degrees: float
    smoothing_width_m: float
    smoothing_width_chord_factor: float | None
    smearing_azimuthal_elements: int
    initial_azimuth_degrees: float
    body_smoothing_width_m: float
    nacelle_drag_coefficient: float
    tower_drag_coefficient: float
    minimum_normal_smoothing_width_m: float = 0.0
    momentum_stabilization_coefficient: float = 0.0



@dataclass(frozen=True, slots=True)
class NacelleCoolingOptions:
    """Thermodynamic calibration and resolved support of the LN2 spray."""

    mass_flow_rate_kg_s: float
    exit_vapor_quality: float
    injection_temperature_k: float
    ambient_temperature_k: float
    liquid_latent_heat_j_kg: float
    nitrogen_heat_capacity_j_kg_k: float
    air_density_kg_m3: float
    air_heat_capacity_j_kg_k: float
    thermal_coupling_efficiency: float
    streamwise_offset_m: float
    standard_deviation_m: tuple[float, float, float]
    ramp_time_s: float
    nozzle_diameter_m: float | None = None
    injection_speed_m_s: float | None = None
    cone_half_angle_degrees: float = 0.0
    pre_nozzle_vapor_quality: float = 0.0

    @property
    def cooling_power_w(self) -> float:
        latent = (
            (1.0 - self.exit_vapor_quality)
            * self.liquid_latent_heat_j_kg
        )
        sensible = self.nitrogen_heat_capacity_j_kg_k * (
            self.ambient_temperature_k - self.injection_temperature_k
        )
        return (
            self.thermal_coupling_efficiency
            * self.mass_flow_rate_kg_s
            * (latent + sensible)
        )

    @property
    def has_momentum_jet(self) -> bool:
        return (
            self.nozzle_diameter_m is not None
            and self.injection_speed_m_s is not None
        )

    @property
    def axial_momentum_flux_n(self) -> float:
        if self.injection_speed_m_s is None:
            return 0.0
        return self.mass_flow_rate_kg_s * self.injection_speed_m_s

    @property
    def exit_vapor_mass_flow_rate_kg_s(self) -> float:
        return self.mass_flow_rate_kg_s * self.exit_vapor_quality

    @property
    def exit_liquid_mass_flow_rate_kg_s(self) -> float:
        return self.mass_flow_rate_kg_s * (1.0 - self.exit_vapor_quality)

@dataclass(frozen=True, slots=True)
class FiniteVolumeWorkflow:
    case: FiniteVolumeCase
    options: WorkflowOptions
    turbine: TurbineOptions | None = None
    cooling: NacelleCoolingOptions | None = None
    moisture: AtmosphericMoistureOptions | None = None
    water_spray: WaterSprayOptions | None = None

    def resolved(self) -> dict[str, Any]:
        from dataclasses import asdict
        grid = self.case.physical.physical_grid
        return {
            "moisture": None if self.moisture is None else asdict(self.moisture),
            "water_spray": None if self.water_spray is None else asdict(self.water_spray),
            "schema": "jaxwind.precursor-main.v1",
            "case": resolved(self.case),
            "warmup": {
                "pressure_backend": "fft",
                "fft_method": self.case.options.fft_method,
                "periodic_x": True,
                "steps": self.options.warmup_steps,
                "duration_seconds": (
                    self.options.warmup_steps * self.case.physical.dt_seconds
                ),
                "maximum_dt_seconds": self.case.physical.dt_seconds,
                "cfl_ceiling": self.case.options.cfl_ceiling,
                "checkpoint": "warmup_checkpoint.npz",
                "restart_checkpoint": (
                    None
                    if self.options.warmup_restart_checkpoint is None
                    else str(self.options.warmup_restart_checkpoint)
                ),
            },
            "precursor": {
                "pressure_backend": "fft",
                "fft_method": self.case.options.fft_method,
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
                "frame_count": self.options.precursor_frame_count,
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
                "inflow_dt_seconds": (
                    self.options.main_dt_seconds
                    or self.options.precursor_dt_seconds
                    or self.case.physical.dt_seconds
                ),
                "substeps_per_inflow": self.options.main_substeps_per_inflow,
                "integration_steps": (
                    self.options.main_steps
                    * self.options.main_substeps_per_inflow
                ),
                "dt_seconds": (
                    (
                        self.options.main_dt_seconds
                        or self.options.precursor_dt_seconds
                        or self.case.physical.dt_seconds
                    )
                    / self.options.main_substeps_per_inflow
                ),
                "time_integration": self.case.options.time_integration,
                "frame_count": self.options.main_frame_count,
                "pressure_force": self.options.main_pressure_force,
                "evolve_scalar": self.options.evolve_scalar,
                "inflow": "one recorded yz layer held across configured substeps",
                "outflow": "second-order zero-gradient transported fields",
                "pressure_boundary": "inlet Neumann, outlet Dirichlet",
                "x_velocity_faces": grid.nx + 1,
            },
            "chunk_steps": self.options.chunk_steps,
            "output_directory": str(self.options.output_directory),
            "input_directory": str(
                self.options.input_directory or self.options.output_directory
            ),
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
                    "smoothing_width_chord_factor": (
                        self.turbine.smoothing_width_chord_factor
                    ),
                    "minimum_normal_smoothing_width_m": self.turbine.minimum_normal_smoothing_width_m,
                    "momentum_stabilization_coefficient": self.turbine.momentum_stabilization_coefficient,
                    "smearing_azimuthal_elements": (
                        self.turbine.smearing_azimuthal_elements
                        if self.turbine.model.endswith("ad-bem")
                        else None
                    ),
                    "initial_azimuth_degrees": (
                        self.turbine.initial_azimuth_degrees
                    ),
                    "nacelle_and_tower": True,
                }
            ),
            "cooling": (
                None
                if self.cooling is None
                else {
                    "model": (
                        "conservative-unresolved-round-jet"
                        if self.cooling.has_momentum_jet
                        and self.cooling.cone_half_angle_degrees == 0.0
                        else "conservative-gaussian-temperature-sink"
                    ),
                    "mass_flow_rate_kg_s": self.cooling.mass_flow_rate_kg_s,
                    "exit_vapor_quality": self.cooling.exit_vapor_quality,
                    "pre_nozzle_vapor_quality": (
                        self.cooling.pre_nozzle_vapor_quality
                    ),
                    "exit_vapor_mass_flow_rate_kg_s": (
                        self.cooling.exit_vapor_mass_flow_rate_kg_s
                    ),
                    "exit_liquid_mass_flow_rate_kg_s": (
                        self.cooling.exit_liquid_mass_flow_rate_kg_s
                    ),
                    "cooling_power_w": self.cooling.cooling_power_w,
                    "nozzle_diameter_m": self.cooling.nozzle_diameter_m,
                    "injection_speed_m_s": self.cooling.injection_speed_m_s,
                    "axial_momentum_flux_n": self.cooling.axial_momentum_flux_n,
                    "cone_half_angle_degrees": self.cooling.cone_half_angle_degrees,
                    "thermal_coupling_efficiency": (
                        self.cooling.thermal_coupling_efficiency
                    ),
                    "streamwise_offset_m": self.cooling.streamwise_offset_m,
                    "standard_deviation_m": (
                        self.cooling.standard_deviation_m
                    ),
                    "ramp_time_s": self.cooling.ramp_time_s,
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


def _load_turbine(document: dict[str, Any]) -> TurbineOptions | None:
    table = document.get("finite_volume_turbine")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError("[finite_volume_turbine] must be a table")
    models = (
        "openfast-ad-bem",
        "hitsz-r9-ad-bem",
        "openfast-alm",
        "hitsz-r9-alm",
    )
    model = table.get("model")
    if model not in models:
        raise ValueError(
            "finite_volume_turbine.model must be one of: "
            + ", ".join(models)
        )
    required = {
        "model",
        "x_m",
        "y_m",
        "hub_height_m",
        "rotor_speed_rpm",
        "blade_pitch_degrees",
        "smoothing_width_m",
        "body_smoothing_width_m",
        "nacelle_drag_coefficient",
        "tower_drag_coefficient",
    }
    allowed = required | {
        "smearing_azimuthal_elements",
        "initial_azimuth_degrees",
        "smoothing_width_chord_factor",
        "minimum_normal_smoothing_width_m",
        "momentum_stabilization_coefficient",
    }
    if model.startswith("openfast-"):
        required.add("openfast_model_environment")
        allowed.add("openfast_model_environment")
    if model.endswith("ad-bem"):
        required.add("smearing_azimuthal_elements")
    missing = required - table.keys()
    unknown = table.keys() - allowed
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
    if model.startswith("openfast-") and (
        not isinstance(environment, str) or not environment
    ):
        raise ValueError(
            "finite_volume_turbine.openfast_model_environment must be a string"
        )
    smearing = table.get("smearing_azimuthal_elements", 64)
    if (
        isinstance(smearing, bool)
        or not isinstance(smearing, int)
        or smearing <= 0
    ):
        raise ValueError(
            "finite_volume_turbine.smearing_azimuthal_elements must be positive"
        )
    initial_azimuth = table.get("initial_azimuth_degrees", 0.0)
    if isinstance(initial_azimuth, bool) or not isinstance(
        initial_azimuth, (int, float)
    ):
        raise ValueError(
            "finite_volume_turbine.initial_azimuth_degrees must be a number"
        )
    import math

    initial_azimuth = float(initial_azimuth)
    if not math.isfinite(initial_azimuth):
        raise ValueError(
            "finite_volume_turbine.initial_azimuth_degrees must be finite"
        )
    chord_factor = table.get("smoothing_width_chord_factor")
    if chord_factor is not None:
        if isinstance(chord_factor, bool) or not isinstance(
            chord_factor, (int, float)
        ):
            raise ValueError(
                "finite_volume_turbine.smoothing_width_chord_factor must "
                "be a number"
            )
        chord_factor = float(chord_factor)
        if not math.isfinite(chord_factor) or chord_factor <= 0.0:
            raise ValueError(
                "finite_volume_turbine.smoothing_width_chord_factor must "
                "be positive and finite"
            )
        if model.endswith("ad-bem"):
            raise ValueError(
                "finite_volume_turbine.smoothing_width_chord_factor is "
                "only valid for an ALM"
            )
    normal_width = table.get("minimum_normal_smoothing_width_m", 0.0)
    if isinstance(normal_width, bool) or not isinstance(normal_width, (int, float)) or not math.isfinite(normal_width) or normal_width < 0.:
        raise ValueError("minimum_normal_smoothing_width_m must be finite and nonnegative")
    if normal_width and not model.endswith("ad-bem"):
        raise ValueError("minimum_normal_smoothing_width_m is only valid for AD-BEM")
    stabilization = table.get("momentum_stabilization_coefficient", 0.0)
    if isinstance(stabilization, bool) or not isinstance(stabilization, (int, float)) or not math.isfinite(stabilization) or not 0. <= stabilization <= 1./16.:
        raise ValueError("momentum_stabilization_coefficient must be finite and between 0 and 1/16")
    if stabilization and not model.endswith("ad-bem"):
        raise ValueError("momentum_stabilization_coefficient is only valid for AD-BEM")
    result = TurbineOptions(
        momentum_stabilization_coefficient=float(stabilization),
        minimum_normal_smoothing_width_m=float(normal_width),
        model=model,
        model_environment=environment,
        x_m=_finite_number(table, "x_m"),
        y_m=_finite_number(table, "y_m"),
        hub_height_m=_finite_number(table, "hub_height_m"),
        rotor_speed_rpm=_finite_number(table, "rotor_speed_rpm"),
        blade_pitch_degrees=_finite_number(table, "blade_pitch_degrees"),
        smoothing_width_m=_finite_number(table, "smoothing_width_m"),
        smoothing_width_chord_factor=chord_factor,
        smearing_azimuthal_elements=smearing,
        initial_azimuth_degrees=initial_azimuth,
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


def _load_cooling(document: dict[str, Any]) -> NacelleCoolingOptions | None:
    table = document.get("finite_volume_cooling")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError("[finite_volume_cooling] must be a table")
    expected = {
        "mass_flow_rate_kg_s",
        "exit_vapor_quality",
        "injection_temperature_k",
        "ambient_temperature_k",
        "liquid_latent_heat_j_kg",
        "nitrogen_heat_capacity_j_kg_k",
        "air_density_kg_m3",
        "air_heat_capacity_j_kg_k",
        "thermal_coupling_efficiency",
        "streamwise_offset_m",
        "standard_deviation_m",
        "ramp_time_s",
    }
    optional = {
        "nozzle_diameter_m",
        "injection_speed_m_s",
        "cone_half_angle_degrees",
        "pre_nozzle_vapor_quality",
    }
    missing = expected - table.keys()
    unknown = table.keys() - expected - optional
    if missing:
        raise ValueError(
            "[finite_volume_cooling] is missing: "
            + ", ".join(sorted(missing))
        )
    if unknown:
        raise ValueError(
            "[finite_volume_cooling] has unknown keys: "
            + ", ".join(sorted(unknown))
        )

    def number(key: str) -> float:
        import math

        value = table[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"finite_volume_cooling.{key} must be a number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"finite_volume_cooling.{key} must be finite")
        return result

    def optional_number(key: str, default: float | None) -> float | None:
        if key not in table:
            return default
        return number(key)

    has_diameter = "nozzle_diameter_m" in table
    has_speed = "injection_speed_m_s" in table
    if has_diameter != has_speed:
        raise ValueError(
            "cooling nozzle diameter and speed must be specified together"
        )

    raw_widths = table["standard_deviation_m"]
    if not isinstance(raw_widths, list) or len(raw_widths) != 3:
        raise ValueError(
            "finite_volume_cooling.standard_deviation_m must have three values"
        )
    widths = tuple(float(value) for value in raw_widths)
    result = NacelleCoolingOptions(
        mass_flow_rate_kg_s=number("mass_flow_rate_kg_s"),
        exit_vapor_quality=number("exit_vapor_quality"),
        injection_temperature_k=number("injection_temperature_k"),
        ambient_temperature_k=number("ambient_temperature_k"),
        liquid_latent_heat_j_kg=number("liquid_latent_heat_j_kg"),
        nitrogen_heat_capacity_j_kg_k=number(
            "nitrogen_heat_capacity_j_kg_k"
        ),
        air_density_kg_m3=number("air_density_kg_m3"),
        air_heat_capacity_j_kg_k=number("air_heat_capacity_j_kg_k"),
        thermal_coupling_efficiency=number(
            "thermal_coupling_efficiency"
        ),
        streamwise_offset_m=number("streamwise_offset_m"),
        standard_deviation_m=widths,
        ramp_time_s=number("ramp_time_s"),
        nozzle_diameter_m=optional_number("nozzle_diameter_m", None),
        injection_speed_m_s=optional_number("injection_speed_m_s", None),
        cone_half_angle_degrees=float(
            optional_number("cone_half_angle_degrees", 0.0)
        ),
        pre_nozzle_vapor_quality=float(
            optional_number("pre_nozzle_vapor_quality", 0.0)
        ),
    )
    positive = (
        result.mass_flow_rate_kg_s,
        result.injection_temperature_k,
        result.ambient_temperature_k,
        result.liquid_latent_heat_j_kg,
        result.nitrogen_heat_capacity_j_kg_k,
        result.air_density_kg_m3,
        result.air_heat_capacity_j_kg_k,
        result.streamwise_offset_m,
        *result.standard_deviation_m,
    )
    if not all(value > 0.0 for value in positive):
        raise ValueError(
            "finite-volume cooling properties, offset, and widths must be positive"
        )
    if result.injection_temperature_k >= result.ambient_temperature_k:
        raise ValueError(
            "finite_volume_cooling injection temperature must be below ambient"
        )
    if not 0.0 <= result.exit_vapor_quality <= 1.0:
        raise ValueError(
            "finite_volume_cooling.exit_vapor_quality must lie in [0, 1]"
        )
    if not 0.0 < result.thermal_coupling_efficiency <= 1.0:
        raise ValueError(
            "finite_volume_cooling.thermal_coupling_efficiency "
            "must lie in (0, 1]"
        )
    if result.ramp_time_s < 0.0:
        raise ValueError("finite_volume_cooling.ramp_time_s must be nonnegative")
    if result.has_momentum_jet and not (
        result.nozzle_diameter_m > 0.0 and result.injection_speed_m_s > 0.0
    ):
        raise ValueError("cooling nozzle diameter and speed must be positive")
    if not 0.0 <= result.cone_half_angle_degrees < 90.0:
        raise ValueError(
            "finite_volume_cooling.cone_half_angle_degrees must lie in [0, 90)"
        )
    if not 0.0 <= result.pre_nozzle_vapor_quality <= result.exit_vapor_quality:
        raise ValueError(
            "finite_volume_cooling.pre_nozzle_vapor_quality must lie between "
            "zero and exit_vapor_quality"
        )
    return result


def load_workflow(path: str | Path) -> FiniteVolumeWorkflow:
    """Load the physical case, FV numerics, and strict stage workflow."""
    source = Path(path)
    from .document import native_document
    document = native_document(path)
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
        "precursor_frame_count",
        "main_dt_seconds",
        "main_frame_count",
        "main_pressure_force",
        "evolve_scalar",
        "input_directory",
        "main_substeps_per_inflow",
        "warmup_restart_checkpoint",
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
    input_value = table.get("input_directory")
    if input_value is not None and (
        not isinstance(input_value, str) or not input_value
    ):
        raise ValueError(
            "finite_volume_workflow.input_directory must be a non-empty string"
        )
    restart_value = table.get("warmup_restart_checkpoint")
    if restart_value is not None and (
        not isinstance(restart_value, str) or not restart_value
    ):
        raise ValueError(
            "finite_volume_workflow.warmup_restart_checkpoint must be a "
            "non-empty string"
        )
    output = table["output_directory"]
    if not isinstance(output, str) or not output:
        raise ValueError(
            "finite_volume_workflow.output_directory must be a non-empty string"
        )
    record_plane = table["record_plane"]
    if isinstance(record_plane, bool) or not isinstance(record_plane, int):
        raise ValueError("finite_volume_workflow.record_plane must be an integer")
    case = load_fv_abl(path)
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
        input_directory=(
            None if input_value is None else Path(input_value)
        ),
        precursor_dt_seconds=(
            _finite_number(table, "precursor_dt_seconds")
            if "precursor_dt_seconds" in table
            else None
        ),
        precursor_frame_count=table.get("precursor_frame_count", 0),
        main_dt_seconds=(
            _finite_number(table, "main_dt_seconds")
            if "main_dt_seconds" in table
            else None
        ),
        main_frame_count=table.get("main_frame_count", 0),
        main_pressure_force=table.get("main_pressure_force", True),
        evolve_scalar=table.get("evolve_scalar", True),
        main_substeps_per_inflow=(
            _positive_integer(table, "main_substeps_per_inflow")
            if "main_substeps_per_inflow" in table
            else 1
        ),
        warmup_restart_checkpoint=(
            None if restart_value is None else Path(restart_value)
        ),
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
        isinstance(options.precursor_frame_count, bool)
        or not isinstance(options.precursor_frame_count, int)
        or options.precursor_frame_count < 0
        or options.precursor_frame_count > options.precursor_steps
    ):
        raise ValueError(
            "finite_volume_workflow.precursor_frame_count must be between "
            "zero and precursor_steps"
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
    turbine = _load_turbine(document)
    cooling = _load_cooling(document)
    if cooling is not None:
        if turbine is None:
            raise ValueError("finite-volume cooling requires a turbine")
        if not options.evolve_scalar:
            raise ValueError("finite-volume cooling requires evolve_scalar=true")
        source_x = turbine.x_m + cooling.streamwise_offset_m
        if source_x >= grid.lx:
            raise ValueError("the nacelle cooling source lies outside the domain")
    moisture, water_spray = load_moisture(document)
    if moisture is not None:
        if not options.evolve_scalar:
            raise ValueError("moisture requires evolve_scalar=true")
        if case.options.time_integration != "fast-rk3":
            raise ValueError("moisture currently requires fast-rk3 integration")
        if cooling is not None:
            raise ValueError("prescribed LN2 cooling and atmospheric moisture cannot be combined")
    if water_spray is not None:
        if turbine is None:
            raise ValueError("water spray requires a turbine")
        source_x = turbine.x_m + water_spray.streamwise_offset_m
        if not grid.x_centers[0] < source_x < grid.x_centers[-1]:
            raise ValueError("water spray source must lie inside the transported domain")
    return FiniteVolumeWorkflow(case, options, turbine, cooling, moisture, water_spray)
