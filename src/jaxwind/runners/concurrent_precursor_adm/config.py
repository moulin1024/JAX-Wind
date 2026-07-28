"""Configuration for concurrent-precursor wind-turbine cases."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from jaxwind.openfast import (
    OpenFASTInputError,
    OpenFASTRigidTurbine,
    load_openfast_rigid_turbine,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 development fallback
    import tomli as tomllib

from .._toml import dumps as toml_dumps
from ..pressure_driven_warmup.config import (
    CaseConfig as WarmupCaseConfig,
    load_case as load_warmup_case,
)


class ConfigError(ValueError):
    """The concurrent case is incomplete or internally inconsistent."""


def _table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing [{name}] table")
    return value


def _string(table: dict[str, Any], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _integer(table: dict[str, Any], key: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    return value


def _optional_integer(table: dict[str, Any], key: str) -> int | None:
    if key not in table:
        return None
    return _integer(table, key)


def _number(table: dict[str, Any], key: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{key} must be finite")
    return result


def _optional_number(table: dict[str, Any], key: str) -> float | None:
    if key not in table:
        return None
    return _number(table, key)


def _location(table: dict[str, Any]) -> tuple[float, float]:
    value = table.get("location_m")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        raise ConfigError("turbine.location_m must contain numeric [x, y]")
    location = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in location):
        raise ConfigError("turbine.location_m must be finite")
    return location


def _resolved_path(source: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (source.parent / path).resolve()


@dataclass(frozen=True, slots=True)
class WarmupConfig:
    case_config: Path
    checkpoint: Path


@dataclass(frozen=True, slots=True)
class ConcurrentConfig:
    steps: int
    launch: str
    lasd_update_interval_steps: int | None = None

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ConfigError("concurrent.steps must be positive")
        if self.launch not in ("auto", "serial", "threads", "cuda-streams"):
            raise ConfigError(
                "concurrent.launch must be auto, serial, threads, or cuda-streams"
            )
        if (
            self.lasd_update_interval_steps is not None
            and self.lasd_update_interval_steps <= 0
        ):
            raise ConfigError(
                "concurrent.lasd_update_interval_steps must be positive"
            )


@dataclass(frozen=True, slots=True)
class FringeConfig:
    start_x_m: float
    relaxation_time_seconds: float
    rise_width_m: float
    fall_width_m: float
    maximum_residual_fraction: float

    def __post_init__(self) -> None:
        if self.start_x_m < 0.0:
            raise ConfigError("fringe.start_x_m must be nonnegative")
        if self.relaxation_time_seconds <= 0.0:
            raise ConfigError("fringe relaxation time must be positive")
        if min(self.rise_width_m, self.fall_width_m) <= 0.0:
            raise ConfigError("fringe rise and fall widths must be positive")
        if not 0.0 < self.maximum_residual_fraction < 1.0:
            raise ConfigError(
                "fringe.maximum_residual_fraction must lie in (0, 1)"
            )


@dataclass(frozen=True, slots=True)
class TurbineConfig:
    location_m: tuple[float, float]
    diameter_m: float
    hub_height_m: float
    thrust_coefficient: float

    def __post_init__(self) -> None:
        if self.diameter_m <= 0.0:
            raise ConfigError("turbine diameter must be positive")
        if self.hub_height_m <= 0.0:
            raise ConfigError("turbine hub height must be positive")
        if not 0.0 <= self.thrust_coefficient < 1.0:
            raise ConfigError("turbine thrust coefficient must lie in [0, 1)")

    @property
    def local_thrust_coefficient(self) -> float:
        induction = 0.5 * (1.0 - math.sqrt(1.0 - self.thrust_coefficient))
        return self.thrust_coefficient / (1.0 - induction) ** 2


@dataclass(frozen=True, slots=True)
class RigidActuatorLineTurbineConfig:
    """OpenFAST-backed, fixed-operating-point actuator-line configuration."""

    location_m: tuple[float, float]
    openfast_input_file: Path
    openfast: OpenFASTRigidTurbine
    smoothing_width_m: float
    hub_height_override_m: float | None = None
    rotor_speed_override_rpm: float | None = None
    pitch_override_degrees: float | None = None
    yaw_override_degrees: float | None = None
    initial_azimuth_override_degrees: float | None = None

    def __post_init__(self) -> None:
        if self.smoothing_width_m <= 0.0:
            raise ConfigError("turbine.smoothing_width_m must be positive")
        if (
            self.hub_height_override_m is not None
            and self.hub_height_override_m <= 0.0
        ):
            raise ConfigError("turbine.hub_height_m must be positive")

    @property
    def diameter_m(self) -> float:
        return 2.0 * self.openfast.tip_radius_m

    @property
    def hub_height_m(self) -> float:
        return (
            self.openfast.hub_height_m
            if self.hub_height_override_m is None
            else self.hub_height_override_m
        )

    @property
    def rotor_speed_rpm(self) -> float:
        return (
            self.openfast.rotor_speed_rpm
            if self.rotor_speed_override_rpm is None
            else self.rotor_speed_override_rpm
        )

    @property
    def pitch_degrees(self) -> float:
        return (
            self.openfast.pitch_degrees
            if self.pitch_override_degrees is None
            else self.pitch_override_degrees
        )

    @property
    def yaw_degrees(self) -> float:
        return (
            self.openfast.yaw_degrees
            if self.yaw_override_degrees is None
            else self.yaw_override_degrees
        )

    @property
    def initial_azimuth_degrees(self) -> float:
        return (
            self.openfast.initial_azimuth_degrees
            if self.initial_azimuth_override_degrees is None
            else self.initial_azimuth_override_degrees
        )


@dataclass(frozen=True, slots=True)
class OutputConfig:
    directory: str
    log_every_steps: int
    checkpoint_every_steps: int
    field_sample_every_steps: int | None

    def __post_init__(self) -> None:
        if not self.directory:
            raise ConfigError("output.directory must be non-empty")
        if min(self.log_every_steps, self.checkpoint_every_steps) <= 0:
            raise ConfigError("output step intervals must be positive")
        if (
            self.field_sample_every_steps is not None
            and self.field_sample_every_steps <= 0
        ):
            raise ConfigError("output.field_sample_every_steps must be positive")


@dataclass(frozen=True, slots=True)
class CaseConfig:
    runner: str
    name: str
    warmup: WarmupConfig
    base: WarmupCaseConfig
    concurrent: ConcurrentConfig
    fringe: FringeConfig
    turbine: TurbineConfig | RigidActuatorLineTurbineConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        if self.runner != "concurrent_precursor_adm":
            raise ConfigError(
                "case.runner must be 'concurrent_precursor_adm' for this runner"
            )
        if not self.name:
            raise ConfigError("case.name must be non-empty")
        domain = self.base.domain
        x, y = self.turbine.location_m
        if not 0.0 <= x < domain.lx_m or not 0.0 <= y < domain.ly_m:
            raise ConfigError("turbine location must lie inside the horizontal domain")
        radius = 0.5 * self.turbine.diameter_m
        if self.turbine.hub_height_m - radius <= 0.0:
            raise ConfigError("turbine rotor intersects the lower wall")
        if self.turbine.hub_height_m + radius >= domain.lz_m:
            raise ConfigError("turbine rotor intersects the upper wall")
        if not x + radius < self.fringe.start_x_m < domain.lx_m:
            raise ConfigError(
                "fringe must start downstream of the rotor and before the periodic seam"
            )
        available = domain.lx_m - self.fringe.start_x_m
        if self.fringe.rise_width_m + self.fringe.fall_width_m > available:
            raise ConfigError(
                "fringe rise and fall widths exceed the fringe region"
            )
        if (
            self.predicted_fringe_residual_fraction
            > self.fringe.maximum_residual_fraction
        ):
            raise ConfigError(
                "fringe attenuation is too weak: predicted residual "
                f"{self.predicted_fringe_residual_fraction:.4g} exceeds "
                f"{self.fringe.maximum_residual_fraction:.4g}; reduce the "
                "relaxation time or widen the plateau"
            )
        if (
            self.estimated_lasd_trajectory_cfl
            >= self.base.numerics.lasd_trajectory_cfl_abort
        ):
            raise ConfigError(
                "concurrent LASD trajectory CFL estimate crosses the abort limit"
            )

    @property
    def lasd_update_interval_steps(self) -> int:
        configured = self.concurrent.lasd_update_interval_steps
        return (
            self.base.sgs.update_interval_steps
            if configured is None
            else configured
        )

    @property
    def estimated_lasd_trajectory_cfl(self) -> float:
        return (
            self.base.estimated_startup_cfl
            * self.lasd_update_interval_steps
        )

    @property
    def normal_smoothing_width_m(self) -> float:
        return 2.0 * self.base.domain.dx_m

    @property
    def transverse_smoothing_width_m(self) -> float:
        return 2.0 * max(self.base.domain.dy_m, self.base.domain.dz_m)

    @property
    def rotor_cells_y(self) -> float:
        return self.turbine.diameter_m / self.base.domain.dy_m

    @property
    def rotor_cells_z(self) -> float:
        return self.turbine.diameter_m / self.base.domain.dz_m

    @property
    def fringe_plateau_width_m(self) -> float:
        return (
            self.base.domain.lx_m
            - self.fringe.start_x_m
            - self.fringe.rise_width_m
            - self.fringe.fall_width_m
        )

    @property
    def effective_fringe_damping_length_m(self) -> float:
        available = self.base.domain.lx_m - self.fringe.start_x_m
        return available - 0.5 * (
            self.fringe.rise_width_m + self.fringe.fall_width_m
        )

    @property
    def predicted_fringe_residual_fraction(self) -> float:
        exponent = self.effective_fringe_damping_length_m / (
            self.base.top_log_velocity_m_s
            * self.fringe.relaxation_time_seconds
        )
        return math.exp(-exponent)

    def resolved(self) -> dict[str, Any]:
        base = self.base.resolved()
        sgs = dict(base["sgs"])
        sgs["update_interval_steps"] = self.lasd_update_interval_steps
        numerics = dict(base["numerics"])
        numerics["estimated_lasd_trajectory_cfl"] = (
            self.estimated_lasd_trajectory_cfl
        )
        if isinstance(self.turbine, TurbineConfig):
            turbine = {
                "model": "uniform_pure_thrust_adm",
                "location_m": list(self.turbine.location_m),
                "diameter_m": self.turbine.diameter_m,
                "hub_height_m": self.turbine.hub_height_m,
                "thrust_coefficient": self.turbine.thrust_coefficient,
                "local_thrust_coefficient": (
                    self.turbine.local_thrust_coefficient
                ),
                "normal_smoothing_width_m": self.normal_smoothing_width_m,
                "transverse_smoothing_width_m": (
                    self.transverse_smoothing_width_m
                ),
                "filtered_velocity_correction": True,
                "cells_across_rotor_y": self.rotor_cells_y,
                "cells_across_rotor_z": self.rotor_cells_z,
                "under_resolved_for_science": min(
                    self.rotor_cells_y, self.rotor_cells_z
                )
                < 8.0,
            }
        else:
            turbine = {
                "model": "openfast_rigid_actuator_line",
                "location_m": list(self.turbine.location_m),
                "openfast_input_file": str(
                    self.turbine.openfast_input_file
                ),
                "aerodyn_input_file": str(
                    self.turbine.openfast.aerodyn_source
                ),
                "elastodyn_input_file": str(
                    self.turbine.openfast.elastodyn_source
                ),
                "diameter_m": self.turbine.diameter_m,
                "hub_height_m": self.turbine.hub_height_m,
                "rotor_speed_rpm": self.turbine.rotor_speed_rpm,
                "pitch_degrees": self.turbine.pitch_degrees,
                "yaw_degrees": self.turbine.yaw_degrees,
                "initial_azimuth_degrees": (
                    self.turbine.initial_azimuth_degrees
                ),
                "smoothing_width_m": self.turbine.smoothing_width_m,
                "blade_count": self.turbine.openfast.blade_count,
                "blade_element_count": len(
                    self.turbine.openfast.element_radii_m
                ),
                "airfoil_count": len(
                    self.turbine.openfast.airfoil_sources
                ),
                "tip_loss": self.turbine.openfast.tip_loss,
                "root_loss": self.turbine.openfast.root_loss,
                "compatibility_notes": list(
                    self.turbine.openfast.compatibility_notes
                ),
                "cells_across_rotor_y": self.rotor_cells_y,
                "cells_across_rotor_z": self.rotor_cells_z,
                "under_resolved_for_science": min(
                    self.rotor_cells_y, self.rotor_cells_z
                )
                < 8.0,
            }
        return {
            "runner": self.runner,
            "case": self.name,
            "warmup": {
                "case_config": str(self.warmup.case_config),
                "checkpoint": str(self.warmup.checkpoint),
                "base_case": self.base.name,
            },
            "domain": base["domain"],
            "flow": base["flow"],
            "sgs": sgs,
            "time": {
                "integrator": self.base.time.integrator,
                "dt_seconds": self.base.time.dt_seconds,
                "additional_steps": self.concurrent.steps,
                "additional_duration_seconds": (
                    self.concurrent.steps * self.base.time.dt_seconds
                ),
            },
            "concurrent": {
                "launch": self.concurrent.launch,
                "warmup_lasd_update_interval_steps": (
                    self.base.sgs.update_interval_steps
                ),
            },
            "fringe": {
                "start_x_m": self.fringe.start_x_m,
                "relaxation_time_seconds": self.fringe.relaxation_time_seconds,
                "rise_width_m": self.fringe.rise_width_m,
                "plateau_width_m": self.fringe_plateau_width_m,
                "fall_width_m": self.fringe.fall_width_m,
                "effective_damping_length_m": (
                    self.effective_fringe_damping_length_m
                ),
                "attenuation_reference_velocity_m_s": (
                    self.base.top_log_velocity_m_s
                ),
                "predicted_residual_fraction": (
                    self.predicted_fringe_residual_fraction
                ),
                "maximum_residual_fraction": (
                    self.fringe.maximum_residual_fraction
                ),
                "relaxes_lasd_closure_memory": True,
            },
            "turbine": turbine,
            "numerics": numerics,
            "output": {
                "directory": self.output.directory,
                "log_every_steps": self.output.log_every_steps,
                "checkpoint_every_steps": self.output.checkpoint_every_steps,
                "field_sample_every_steps": (
                    self.output.field_sample_every_steps
                ),
            },
        }

    def resolved_toml(self) -> str:
        return toml_dumps(self.resolved())


def load_case(path: str | Path) -> CaseConfig:
    source = Path(path).resolve()
    with source.open("rb") as stream:
        document = tomllib.load(stream)

    case = _table(document, "case")
    warmup = _table(document, "warmup")
    concurrent = _table(document, "concurrent")
    fringe = _table(document, "fringe")
    turbine = _table(document, "turbine")
    output = _table(document, "output")

    warmup_case_path = _resolved_path(source, _string(warmup, "case_config"))
    if warmup_case_path.is_dir():
        warmup_case_path = warmup_case_path / "config.toml"
    warmup_checkpoint = _resolved_path(source, _string(warmup, "checkpoint"))
    base = load_warmup_case(warmup_case_path)
    turbine_model = turbine.get("model", "uniform_pure_thrust_adm")
    if turbine_model == "uniform_pure_thrust_adm":
        turbine_config: TurbineConfig | RigidActuatorLineTurbineConfig = (
            TurbineConfig(
                location_m=_location(turbine),
                diameter_m=_number(turbine, "diameter_m"),
                hub_height_m=_number(turbine, "hub_height_m"),
                thrust_coefficient=_number(
                    turbine,
                    "thrust_coefficient",
                ),
            )
        )
    elif turbine_model == "openfast_rigid_actuator_line":
        openfast_input_file = _resolved_path(
            source,
            _string(turbine, "openfast_input_file"),
        )
        try:
            openfast = load_openfast_rigid_turbine(openfast_input_file)
        except OpenFASTInputError as error:
            raise ConfigError(str(error)) from error
        smoothing_width_m = (
            2.0 * max(base.domain.dy_m, base.domain.dz_m)
            if "smoothing_width_m" not in turbine
            else _number(turbine, "smoothing_width_m")
        )
        turbine_config = RigidActuatorLineTurbineConfig(
            location_m=_location(turbine),
            openfast_input_file=openfast_input_file,
            openfast=openfast,
            smoothing_width_m=smoothing_width_m,
            hub_height_override_m=_optional_number(
                turbine,
                "hub_height_m",
            ),
            rotor_speed_override_rpm=_optional_number(
                turbine,
                "rotor_speed_rpm",
            ),
            pitch_override_degrees=_optional_number(
                turbine,
                "pitch_degrees",
            ),
            yaw_override_degrees=_optional_number(
                turbine,
                "yaw_degrees",
            ),
            initial_azimuth_override_degrees=_optional_number(
                turbine,
                "initial_azimuth_degrees",
            ),
        )
    else:
        raise ConfigError(
            "turbine.model must be 'uniform_pure_thrust_adm' or "
            "'openfast_rigid_actuator_line'"
        )

    return CaseConfig(
        runner=_string(case, "runner"),
        name=_string(case, "name"),
        warmup=WarmupConfig(warmup_case_path, warmup_checkpoint),
        base=base,
        concurrent=ConcurrentConfig(
            steps=_integer(concurrent, "steps"),
            launch=_string(concurrent, "launch"),
            lasd_update_interval_steps=_optional_integer(
                concurrent,
                "lasd_update_interval_steps",
            ),
        ),
        fringe=FringeConfig(
            start_x_m=_number(fringe, "start_x_m"),
            relaxation_time_seconds=_number(
                fringe, "relaxation_time_seconds"
            ),
            rise_width_m=_number(fringe, "rise_width_m"),
            fall_width_m=_number(fringe, "fall_width_m"),
            maximum_residual_fraction=_number(
                fringe, "maximum_residual_fraction"
            ),
        ),
        turbine=turbine_config,
        output=OutputConfig(
            directory=_string(output, "directory"),
            log_every_steps=_integer(output, "log_every_steps"),
            checkpoint_every_steps=_integer(
                output, "checkpoint_every_steps"
            ),
            field_sample_every_steps=_optional_integer(
                output, "field_sample_every_steps"
            ),
        ),
    )
