"""TOML configuration for the built-in pressure-driven neutral warmup runner."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 development fallback
    import tomli as tomllib

from .._toml import dumps as toml_dumps


class ConfigError(ValueError):
    """The case file is incomplete or internally inconsistent."""


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


def _boolean(
    table: dict[str, Any],
    key: str,
    *,
    default: bool | None = None,
) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class DomainConfig:
    nx: int
    ny: int
    nz: int
    lx_m: float
    ly_m: float
    lz_m: float

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) <= 1:
            raise ConfigError("all grid dimensions must exceed one")
        if min(self.lx_m, self.ly_m, self.lz_m) <= 0.0:
            raise ConfigError("all domain lengths must be positive")

    @property
    def dx_m(self) -> float:
        return self.lx_m / self.nx

    @property
    def dy_m(self) -> float:
        return self.ly_m / self.ny

    @property
    def dz_m(self) -> float:
        return self.lz_m / self.nz


@dataclass(frozen=True, slots=True)
class FlowConfig:
    friction_velocity_m_s: float
    roughness_length_m: float
    forcing_height_m: float
    von_karman: float
    initial_perturbation_rms_m_s: float
    initial_correlation_length_m: float

    def __post_init__(self) -> None:
        positive = (
            self.friction_velocity_m_s,
            self.roughness_length_m,
            self.forcing_height_m,
            self.von_karman,
            self.initial_correlation_length_m,
        )
        if not all(value > 0.0 for value in positive):
            raise ConfigError("flow scales and wall constants must be positive")
        if self.initial_perturbation_rms_m_s < 0.0:
            raise ConfigError("initial perturbation RMS must be nonnegative")

    @property
    def pressure_acceleration_m_s2(self) -> float:
        return self.friction_velocity_m_s**2 / self.forcing_height_m


@dataclass(frozen=True, slots=True)
class WallConfig:
    model: str
    filter_grid_ratio: float
    test_filter_ratio: float
    porte_agel_correction: bool

    def __post_init__(self) -> None:
        if self.model != "filtered_neutral_log":
            raise ConfigError("wall.model must be 'filtered_neutral_log'")
        if self.filter_grid_ratio <= 0.0 or self.test_filter_ratio <= 1.0:
            raise ConfigError("wall filter ratios are invalid")


@dataclass(frozen=True, slots=True)
class SgsConfig:
    model: str
    update_interval_steps: int
    filter_grid_ratio: float
    test_filter_ratio: float
    timescale_coefficient: float
    initial_coefficient: float
    minimum_coefficient: float
    maximum_coefficient: float
    dissipation_coefficient: float
    fallback_coefficient: float
    gradient_norm_epsilon_s2: float
    kinematic_viscosity_m2_s: float
    scalar_turbulent_prandtl: float

    def __post_init__(self) -> None:
        if self.model not in ("lasd", "mgm", "amd"):
            raise ConfigError("sgs.model must be 'lasd', 'mgm', or 'amd'")
        if self.filter_grid_ratio <= 0.0:
            raise ConfigError("SGS filter-grid ratio must be positive")
        if self.scalar_turbulent_prandtl <= 0.0:
            raise ConfigError("scalar turbulent Prandtl number must be positive")
        if self.model == "lasd":
            if self.update_interval_steps <= 0:
                raise ConfigError("LASD update interval must be positive")
            if self.test_filter_ratio <= 1.0:
                raise ConfigError("LASD test-filter ratio must exceed one")
            if self.timescale_coefficient <= 0.0:
                raise ConfigError("LASD timescale coefficient must be positive")
            if not (
                0.0
                <= self.minimum_coefficient
                <= self.initial_coefficient
                <= self.maximum_coefficient
            ):
                raise ConfigError("LASD coefficient bounds are inconsistent")
        elif self.model == "mgm":
            if self.dissipation_coefficient <= 0.0:
                raise ConfigError("MGM dissipation coefficient must be positive")
            if self.fallback_coefficient < 0.0:
                raise ConfigError("MGM fallback coefficient must be nonnegative")
            if self.gradient_norm_epsilon_s2 <= 0.0:
                raise ConfigError("MGM gradient-norm epsilon must be positive")
            if self.kinematic_viscosity_m2_s < 0.0:
                raise ConfigError("MGM kinematic viscosity must be nonnegative")


@dataclass(frozen=True, slots=True)
class TimeConfig:
    integrator: str
    dt_seconds: float
    duration_hours: float

    def __post_init__(self) -> None:
        if self.integrator != "ab2":
            raise ConfigError("time.integrator must be 'ab2'")
        if self.dt_seconds <= 0.0 or self.duration_hours <= 0.0:
            raise ConfigError("time step and duration must be positive")
        _ = self.steps

    @property
    def duration_seconds(self) -> float:
        return self.duration_hours * 3600.0

    @property
    def steps(self) -> int:
        raw_steps = self.duration_seconds / self.dt_seconds
        steps = int(round(raw_steps))
        if not math.isclose(raw_steps, steps, rel_tol=0.0, abs_tol=1.0e-10):
            raise ConfigError("duration must contain an integer number of time steps")
        return steps


@dataclass(frozen=True, slots=True)
class NumericsConfig:
    dtype: str
    pressure_method: str
    seed: int
    cfl_abort: float
    lasd_trajectory_cfl_abort: float

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "float64"):
            raise ConfigError("numerics.dtype must be float32 or float64")
        if self.pressure_method not in ("transpose", "spike", "spike-adaptive"):
            raise ConfigError("unsupported pressure method")
        if self.seed < 0:
            raise ConfigError("random seed must be nonnegative")
        if self.cfl_abort <= 0.0 or self.lasd_trajectory_cfl_abort <= 0.0:
            raise ConfigError("CFL abort limits must be positive")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    directory: str
    log_every_steps: int
    sample_start_hours: float
    sample_every_steps: int
    checkpoint_every_steps: int

    def __post_init__(self) -> None:
        if not self.directory:
            raise ConfigError("output.directory must be non-empty")
        if min(
            self.log_every_steps,
            self.sample_every_steps,
            self.checkpoint_every_steps,
        ) <= 0:
            raise ConfigError("output step intervals must be positive")
        if self.sample_start_hours < 0.0:
            raise ConfigError("sample start must be nonnegative")


@dataclass(frozen=True, slots=True)
class CaseConfig:
    runner: str
    name: str
    domain: DomainConfig
    flow: FlowConfig
    wall: WallConfig
    sgs: SgsConfig
    time: TimeConfig
    numerics: NumericsConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        if self.runner != "pressure_driven_warmup":
            raise ConfigError(
                "case.runner must be 'pressure_driven_warmup' for this runner"
            )
        if not self.name:
            raise ConfigError("case.name must be non-empty")
        if self.flow.forcing_height_m > self.domain.lz_m:
            raise ConfigError("forcing height cannot exceed the domain height")
        if self.flow.roughness_length_m >= 0.5 * self.domain.dz_m:
            raise ConfigError("roughness length must lie below the first cell center")
        if self.output.sample_start_hours >= self.time.duration_hours:
            raise ConfigError("sample start must be earlier than the final time")
        if self.estimated_startup_cfl >= self.numerics.cfl_abort:
            raise ConfigError(
                "estimated startup CFL exceeds numerics.cfl_abort; reduce dt"
            )
        if self.sgs.model == "lasd" and (
            self.estimated_lasd_trajectory_cfl
            >= self.numerics.lasd_trajectory_cfl_abort
        ):
            raise ConfigError(
                "estimated LASD trajectory CFL exceeds its abort limit; "
                "reduce dt or the update interval"
            )

    @property
    def top_cell_height_m(self) -> float:
        return self.domain.lz_m - 0.5 * self.domain.dz_m

    @property
    def top_log_velocity_m_s(self) -> float:
        return (
            self.flow.friction_velocity_m_s
            / self.flow.von_karman
            * math.log(self.top_cell_height_m / self.flow.roughness_length_m)
        )

    @property
    def estimated_startup_cfl(self) -> float:
        conservative_speed = (
            self.top_log_velocity_m_s
            + 3.0 * self.flow.initial_perturbation_rms_m_s
        )
        return conservative_speed * self.time.dt_seconds / self.domain.dx_m

    @property
    def estimated_lasd_trajectory_cfl(self) -> float:
        if self.sgs.model != "lasd":
            return 0.0
        return self.estimated_startup_cfl * self.sgs.update_interval_steps

    @property
    def sample_start_step(self) -> int:
        raw_step = self.output.sample_start_hours * 3600.0 / self.time.dt_seconds
        step = int(round(raw_step))
        if not math.isclose(raw_step, step, rel_tol=0.0, abs_tol=1.0e-10):
            raise ConfigError("sample start must fall on an accepted time step")
        return step

    def resolved(self) -> dict[str, Any]:
        sgs = {"model": self.sgs.model}
        if self.sgs.model == "lasd":
            sgs["update_interval_steps"] = self.sgs.update_interval_steps
        elif self.sgs.model == "mgm":
            sgs.update(
                {
                    "filter_grid_ratio": self.sgs.filter_grid_ratio,
                    "dissipation_coefficient": self.sgs.dissipation_coefficient,
                    "fallback_coefficient": self.sgs.fallback_coefficient,
                    "gradient_norm_epsilon_s2": (
                        self.sgs.gradient_norm_epsilon_s2
                    ),
                    "kinematic_viscosity_m2_s": (
                        self.sgs.kinematic_viscosity_m2_s
                    ),
                    "scalar_turbulent_prandtl": (
                        self.sgs.scalar_turbulent_prandtl
                    ),
                }
            )
        return {
            "runner": self.runner,
            "case": self.name,
            "domain": {
                "cells": [self.domain.nx, self.domain.ny, self.domain.nz],
                "lengths_m": [
                    self.domain.lx_m,
                    self.domain.ly_m,
                    self.domain.lz_m,
                ],
                "spacing_m": [
                    self.domain.dx_m,
                    self.domain.dy_m,
                    self.domain.dz_m,
                ],
            },
            "flow": {
                "friction_velocity_m_s": self.flow.friction_velocity_m_s,
                "roughness_length_m": self.flow.roughness_length_m,
                "forcing_height_m": self.flow.forcing_height_m,
                "pressure_acceleration_m_s2": (
                    self.flow.pressure_acceleration_m_s2
                ),
                "top_log_velocity_m_s": self.top_log_velocity_m_s,
            },
            "wall": {
                "model": self.wall.model,
                "filter_grid_ratio": self.wall.filter_grid_ratio,
                "test_filter_ratio": self.wall.test_filter_ratio,
                "porte_agel_correction": self.wall.porte_agel_correction,
            },
            "sgs": sgs,
            "time": {
                "integrator": self.time.integrator,
                "dt_seconds": self.time.dt_seconds,
                "duration_hours": self.time.duration_hours,
                "steps": self.time.steps,
            },
            "numerics": {
                "dtype": self.numerics.dtype,
                "pressure_method": self.numerics.pressure_method,
                "estimated_startup_cfl": self.estimated_startup_cfl,
                "estimated_lasd_trajectory_cfl": (
                    self.estimated_lasd_trajectory_cfl
                ),
                "cfl_abort": self.numerics.cfl_abort,
                "lasd_trajectory_cfl_abort": (
                    self.numerics.lasd_trajectory_cfl_abort
                ),
            },
            "output": {
                "directory": self.output.directory,
                "sample_start_step": self.sample_start_step,
                "sample_every_steps": self.output.sample_every_steps,
                "checkpoint_every_steps": self.output.checkpoint_every_steps,
            },
        }

    def resolved_toml(self) -> str:
        return toml_dumps(self.resolved())


def load_case(
    path: str | Path,
    *,
    dt_seconds: float | None = None,
    duration_hours: float | None = None,
    statistics_fraction: float | None = None,
) -> CaseConfig:
    source = Path(path)
    with source.open("rb") as stream:
        document = tomllib.load(stream)

    case = _table(document, "case")
    domain = _table(document, "domain")
    flow = _table(document, "flow")
    wall = _table(document, "wall")
    sgs = _table(document, "sgs")
    time = _table(document, "time")
    numerics = _table(document, "numerics")
    output = _table(document, "output")
    configured_dt = _number(time, "dt_seconds")
    configured_duration = _number(time, "duration_hours")
    resolved_dt = (
        configured_dt
        if dt_seconds is None
        else _number({"dt_seconds": dt_seconds}, "dt_seconds")
    )
    resolved_duration = (
        configured_duration
        if duration_hours is None
        else _number({"duration_hours": duration_hours}, "duration_hours")
    )
    if statistics_fraction is None:
        sample_start_hours = _number(output, "sample_start_hours")
    else:
        fraction = _number(
            {"statistics_fraction": statistics_fraction},
            "statistics_fraction",
        )
        if not 0.0 < fraction <= 1.0:
            raise ConfigError("statistics_fraction must lie in (0, 1]")
        sample_start_hours = (1.0 - fraction) * resolved_duration

    return CaseConfig(
        runner=_string(case, "runner"),
        name=_string(case, "name"),
        domain=DomainConfig(
            nx=_integer(domain, "nx"),
            ny=_integer(domain, "ny"),
            nz=_integer(domain, "nz"),
            lx_m=_number(domain, "lx_m"),
            ly_m=_number(domain, "ly_m"),
            lz_m=_number(domain, "lz_m"),
        ),
        flow=FlowConfig(
            friction_velocity_m_s=_number(flow, "friction_velocity_m_s"),
            roughness_length_m=_number(flow, "roughness_length_m"),
            forcing_height_m=_number(flow, "forcing_height_m"),
            von_karman=_number(flow, "von_karman"),
            initial_perturbation_rms_m_s=_number(
                flow, "initial_perturbation_rms_m_s"
            ),
            initial_correlation_length_m=_number(
                flow, "initial_correlation_length_m"
            ),
        ),
        wall=WallConfig(
            model=_string(wall, "model"),
            filter_grid_ratio=_number(wall, "filter_grid_ratio"),
            test_filter_ratio=_number(wall, "test_filter_ratio"),
            porte_agel_correction=_boolean(
                wall,
                "porte_agel_correction",
                default=True,
            ),
        ),
        sgs=SgsConfig(
            model=_string(sgs, "model"),
            update_interval_steps=_integer(sgs, "update_interval_steps"),
            filter_grid_ratio=_number(sgs, "filter_grid_ratio"),
            test_filter_ratio=_number(sgs, "test_filter_ratio"),
            timescale_coefficient=_number(sgs, "timescale_coefficient"),
            initial_coefficient=_number(sgs, "initial_coefficient"),
            minimum_coefficient=_number(sgs, "minimum_coefficient"),
            maximum_coefficient=_number(sgs, "maximum_coefficient"),
            dissipation_coefficient=_number(sgs, "dissipation_coefficient"),
            fallback_coefficient=_number(sgs, "fallback_coefficient"),
            gradient_norm_epsilon_s2=_number(
                sgs,
                "gradient_norm_epsilon_s2",
            ),
            kinematic_viscosity_m2_s=_number(
                sgs,
                "kinematic_viscosity_m2_s",
            ),
            scalar_turbulent_prandtl=_number(
                sgs,
                "scalar_turbulent_prandtl",
            ),
        ),
        time=TimeConfig(
            integrator=_string(time, "integrator"),
            dt_seconds=resolved_dt,
            duration_hours=resolved_duration,
        ),
        numerics=NumericsConfig(
            dtype=_string(numerics, "dtype"),
            pressure_method=_string(numerics, "pressure_method"),
            seed=_integer(numerics, "seed"),
            cfl_abort=_number(numerics, "cfl_abort"),
            lasd_trajectory_cfl_abort=_number(
                numerics, "lasd_trajectory_cfl_abort"
            ),
        ),
        output=OutputConfig(
            directory=_string(output, "directory"),
            log_every_steps=_integer(output, "log_every_steps"),
            sample_start_hours=sample_start_hours,
            sample_every_steps=_integer(output, "sample_every_steps"),
            checkpoint_every_steps=_integer(output, "checkpoint_every_steps"),
        ),
    )
