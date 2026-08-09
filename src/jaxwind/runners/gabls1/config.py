"""Validated TOML configuration for the GABLS1 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from .._toml import dumps as toml_dumps


class ConfigError(ValueError):
    """The GABLS1 case file is incomplete or inconsistent."""


def _table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing [{name}] table")
    return value


def _number(table: dict[str, Any], key: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{key} must be finite")
    return result


def _integer(table: dict[str, Any], key: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    return value


def _string(table: dict[str, Any], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
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
            raise ConfigError("domain lengths must be positive")

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
    geostrophic_u_m_s: float
    geostrophic_v_m_s: float
    coriolis_s: float
    roughness_length_m: float
    von_karman: float

    def __post_init__(self) -> None:
        if self.coriolis_s <= 0.0:
            raise ConfigError("Coriolis parameter must be positive")
        if self.roughness_length_m <= 0.0 or self.von_karman <= 0.0:
            raise ConfigError("wall constants must be positive")


@dataclass(frozen=True, slots=True)
class ThermalConfig:
    initial_temperature_k: float
    reference_temperature_k: float
    inversion_height_m: float
    inversion_gradient_k_m: float
    perturbation_amplitude_k: float
    perturbation_height_m: float
    surface_cooling_k_s: float
    thermal_roughness_length_m: float
    gravity_m_s2: float

    def __post_init__(self) -> None:
        positive = (
            self.initial_temperature_k,
            self.reference_temperature_k,
            self.inversion_height_m,
            self.inversion_gradient_k_m,
            self.thermal_roughness_length_m,
            self.gravity_m_s2,
        )
        if not all(value > 0.0 for value in positive):
            raise ConfigError("thermal scales must be positive")
        if self.perturbation_amplitude_k < 0.0 or self.perturbation_height_m < 0.0:
            raise ConfigError("thermal perturbation controls must be nonnegative")
        if self.surface_cooling_k_s >= 0.0:
            raise ConfigError("GABLS1 surface cooling rate must be negative")


@dataclass(frozen=True, slots=True)
class SgsConfig:
    model: str
    scalar_turbulent_prandtl: float
    lasd_update_interval: int

    def __post_init__(self) -> None:
        if self.model not in ("amd", "lasd"):
            raise ConfigError("GABLS1 sgs.model must be 'amd' or 'lasd'")
        if self.scalar_turbulent_prandtl <= 0.0:
            raise ConfigError("scalar turbulent Prandtl number must be positive")
        if self.lasd_update_interval <= 0:
            raise ConfigError("LASD update interval must be positive")


@dataclass(frozen=True, slots=True)
class TimeConfig:
    dt_seconds: float
    duration_hours: float
    sample_start_hours: float

    def __post_init__(self) -> None:
        if self.dt_seconds <= 0.0 or self.duration_hours <= 0.0:
            raise ConfigError("time step and duration must be positive")
        if not 0.0 <= self.sample_start_hours < self.duration_hours:
            raise ConfigError("sample start must lie inside the run")
        _ = self.steps

    @property
    def steps(self) -> int:
        value = self.duration_hours * 3600.0 / self.dt_seconds
        result = int(round(value))
        if not math.isclose(value, result, abs_tol=1.0e-10):
            raise ConfigError("duration must contain an integer number of steps")
        return result

    @property
    def sample_start_step(self) -> int:
        return int(round(self.sample_start_hours * 3600.0 / self.dt_seconds))


@dataclass(frozen=True, slots=True)
class NumericsConfig:
    dtype: str
    pressure_method: str
    seed: int
    cfl_warning: float

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "float64"):
            raise ConfigError("dtype must be float32 or float64")
        if self.pressure_method not in ("transpose", "spike", "spike-adaptive"):
            raise ConfigError("unsupported pressure method")
        if self.cfl_warning <= 0.0:
            raise ConfigError("CFL warning level must be positive")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    directory: str
    sample_every_steps: int
    log_every_steps: int
    checkpoint_every_steps: int

    def __post_init__(self) -> None:
        if not self.directory:
            raise ConfigError("output directory must be non-empty")
        if min(
            self.sample_every_steps,
            self.log_every_steps,
            self.checkpoint_every_steps,
        ) <= 0:
            raise ConfigError("output intervals must be positive")


@dataclass(frozen=True, slots=True)
class CaseConfig:
    name: str
    runner: str
    domain: DomainConfig
    flow: FlowConfig
    thermal: ThermalConfig
    sgs: SgsConfig
    time: TimeConfig
    numerics: NumericsConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        if self.runner != "gabls1":
            raise ConfigError("case.runner must be 'gabls1'")
        if self.thermal.inversion_height_m >= self.domain.lz_m:
            raise ConfigError("inversion height must be below the domain top")
        if self.thermal.perturbation_height_m > self.domain.lz_m:
            raise ConfigError("perturbation height must lie inside the domain")
        if self.flow.roughness_length_m >= 0.5 * self.domain.dz_m:
            raise ConfigError("momentum roughness must be below the first cell")
        if self.thermal.thermal_roughness_length_m >= 0.5 * self.domain.dz_m:
            raise ConfigError("thermal roughness must be below the first cell")

    def resolved(self) -> dict[str, Any]:
        return {
            "case": {"name": self.name, "runner": self.runner},
            "domain": {
                "nx": self.domain.nx,
                "ny": self.domain.ny,
                "nz": self.domain.nz,
                "lx_m": self.domain.lx_m,
                "ly_m": self.domain.ly_m,
                "lz_m": self.domain.lz_m,
            },
            "flow": {
                "geostrophic_u_m_s": self.flow.geostrophic_u_m_s,
                "geostrophic_v_m_s": self.flow.geostrophic_v_m_s,
                "coriolis_s": self.flow.coriolis_s,
                "roughness_length_m": self.flow.roughness_length_m,
                "von_karman": self.flow.von_karman,
            },
            "thermal": {
                "initial_temperature_k": self.thermal.initial_temperature_k,
                "reference_temperature_k": self.thermal.reference_temperature_k,
                "inversion_height_m": self.thermal.inversion_height_m,
                "inversion_gradient_k_m": self.thermal.inversion_gradient_k_m,
                "perturbation_amplitude_k": self.thermal.perturbation_amplitude_k,
                "perturbation_height_m": self.thermal.perturbation_height_m,
                "surface_cooling_k_s": self.thermal.surface_cooling_k_s,
                "thermal_roughness_length_m": (
                    self.thermal.thermal_roughness_length_m
                ),
                "gravity_m_s2": self.thermal.gravity_m_s2,
            },
            "sgs": {
                "model": self.sgs.model,
                "scalar_turbulent_prandtl": self.sgs.scalar_turbulent_prandtl,
                "lasd_update_interval": self.sgs.lasd_update_interval,
            },
            "time": {
                "dt_seconds": self.time.dt_seconds,
                "duration_hours": self.time.duration_hours,
                "sample_start_hours": self.time.sample_start_hours,
            },
            "numerics": {
                "dtype": self.numerics.dtype,
                "pressure_method": self.numerics.pressure_method,
                "seed": self.numerics.seed,
                "cfl_warning": self.numerics.cfl_warning,
            },
            "output": {
                "directory": self.output.directory,
                "sample_every_steps": self.output.sample_every_steps,
                "log_every_steps": self.output.log_every_steps,
                "checkpoint_every_steps": self.output.checkpoint_every_steps,
            },
        }

    def resolved_toml(self) -> str:
        return toml_dumps(self.resolved())


def load_case(path: str | Path) -> CaseConfig:
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    case = _table(document, "case")
    domain = _table(document, "domain")
    flow = _table(document, "flow")
    thermal = _table(document, "thermal")
    sgs = _table(document, "sgs")
    time = _table(document, "time")
    numerics = _table(document, "numerics")
    output = _table(document, "output")
    return CaseConfig(
        _string(case, "name"),
        _string(case, "runner"),
        DomainConfig(
            _integer(domain, "nx"),
            _integer(domain, "ny"),
            _integer(domain, "nz"),
            _number(domain, "lx_m"),
            _number(domain, "ly_m"),
            _number(domain, "lz_m"),
        ),
        FlowConfig(
            _number(flow, "geostrophic_u_m_s"),
            _number(flow, "geostrophic_v_m_s"),
            _number(flow, "coriolis_s"),
            _number(flow, "roughness_length_m"),
            _number(flow, "von_karman"),
        ),
        ThermalConfig(
            _number(thermal, "initial_temperature_k"),
            _number(thermal, "reference_temperature_k"),
            _number(thermal, "inversion_height_m"),
            _number(thermal, "inversion_gradient_k_m"),
            _number(thermal, "perturbation_amplitude_k"),
            _number(thermal, "perturbation_height_m"),
            _number(thermal, "surface_cooling_k_s"),
            _number(thermal, "thermal_roughness_length_m"),
            _number(thermal, "gravity_m_s2"),
        ),
        SgsConfig(
            _string(sgs, "model"),
            _number(sgs, "scalar_turbulent_prandtl"),
            _integer(sgs, "lasd_update_interval"),
        ),
        TimeConfig(
            _number(time, "dt_seconds"),
            _number(time, "duration_hours"),
            _number(time, "sample_start_hours"),
        ),
        NumericsConfig(
            _string(numerics, "dtype"),
            _string(numerics, "pressure_method"),
            _integer(numerics, "seed"),
            _number(numerics, "cfl_warning"),
        ),
        OutputConfig(
            _string(output, "directory"),
            _integer(output, "sample_every_steps"),
            _integer(output, "log_every_steps"),
            _integer(output, "checkpoint_every_steps"),
        ),
    )
