"""Configuration for a concurrent precursor and pure-thrust ADM case."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 development fallback
    import tomli as tomllib

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


def _number(table: dict[str, Any], key: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{key} must be finite")
    return result


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

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ConfigError("concurrent.steps must be positive")
        if self.launch not in ("auto", "serial", "threads", "cuda-streams"):
            raise ConfigError(
                "concurrent.launch must be auto, serial, threads, or cuda-streams"
            )


@dataclass(frozen=True, slots=True)
class FringeConfig:
    start_x_m: float
    relaxation_time_seconds: float

    def __post_init__(self) -> None:
        if self.start_x_m < 0.0:
            raise ConfigError("fringe.start_x_m must be nonnegative")
        if self.relaxation_time_seconds <= 0.0:
            raise ConfigError("fringe relaxation time must be positive")


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
class OutputConfig:
    directory: str
    log_every_steps: int
    checkpoint_every_steps: int

    def __post_init__(self) -> None:
        if not self.directory:
            raise ConfigError("output.directory must be non-empty")
        if min(self.log_every_steps, self.checkpoint_every_steps) <= 0:
            raise ConfigError("output step intervals must be positive")


@dataclass(frozen=True, slots=True)
class CaseConfig:
    runner: str
    name: str
    warmup: WarmupConfig
    base: WarmupCaseConfig
    concurrent: ConcurrentConfig
    fringe: FringeConfig
    turbine: TurbineConfig
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

    def resolved(self) -> dict[str, Any]:
        base = self.base.resolved()
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
            "sgs": base["sgs"],
            "time": {
                "integrator": self.base.time.integrator,
                "dt_seconds": self.base.time.dt_seconds,
                "additional_steps": self.concurrent.steps,
                "additional_duration_seconds": (
                    self.concurrent.steps * self.base.time.dt_seconds
                ),
            },
            "concurrent": {"launch": self.concurrent.launch},
            "fringe": {
                "start_x_m": self.fringe.start_x_m,
                "relaxation_time_seconds": self.fringe.relaxation_time_seconds,
            },
            "turbine": {
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
            },
            "numerics": base["numerics"],
            "output": {
                "directory": self.output.directory,
                "log_every_steps": self.output.log_every_steps,
                "checkpoint_every_steps": self.output.checkpoint_every_steps,
            },
        }

    def resolved_json(self) -> str:
        return json.dumps(self.resolved(), indent=2) + "\n"


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

    return CaseConfig(
        runner=_string(case, "runner"),
        name=_string(case, "name"),
        warmup=WarmupConfig(warmup_case_path, warmup_checkpoint),
        base=base,
        concurrent=ConcurrentConfig(
            steps=_integer(concurrent, "steps"),
            launch=_string(concurrent, "launch"),
        ),
        fringe=FringeConfig(
            start_x_m=_number(fringe, "start_x_m"),
            relaxation_time_seconds=_number(
                fringe, "relaxation_time_seconds"
            ),
        ),
        turbine=TurbineConfig(
            location_m=_location(turbine),
            diameter_m=_number(turbine, "diameter_m"),
            hub_height_m=_number(turbine, "hub_height_m"),
            thrust_coefficient=_number(turbine, "thrust_coefficient"),
        ),
        output=OutputConfig(
            directory=_string(output, "directory"),
            log_every_steps=_integer(output, "log_every_steps"),
            checkpoint_every_steps=_integer(
                output, "checkpoint_every_steps"
            ),
        ),
    )
