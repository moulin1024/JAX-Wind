"""Declarative configuration for the shared ABL workflow runner."""

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
    """The ABL warmup configuration is incomplete or inconsistent."""


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


def _number(table: dict[str, Any], key: str, default: float | None = None) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{key} must be finite")
    return result


def _integer(table: dict[str, Any], key: str, default: int | None = None) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    return value


def _boolean(table: dict[str, Any], key: str, default: bool) -> bool:
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
    momentum_forcing: str
    friction_velocity_m_s: float
    forcing_height_m: float
    geostrophic_u_m_s: float
    geostrophic_v_m_s: float
    coriolis_s: float
    roughness_length_m: float
    von_karman: float
    initial_velocity_profile: str
    initial_perturbation_rms_m_s: float
    initial_correlation_length_m: float

    def __post_init__(self) -> None:
        if self.momentum_forcing not in (
            "pressure_gradient",
            "geostrophic",
            "none",
        ):
            raise ConfigError("unsupported flow.momentum_forcing")
        if self.initial_velocity_profile not in (
            "log_law",
            "zero",
            "andren1994_table",
        ):
            raise ConfigError(
                "unsupported flow.initial_velocity_profile"
            )
        if min(
            self.friction_velocity_m_s,
            self.forcing_height_m,
            self.coriolis_s,
            self.initial_perturbation_rms_m_s,
        ) < 0.0:
            raise ConfigError(
                "flow forcing and perturbation values must be nonnegative"
            )
        if min(
            self.roughness_length_m,
            self.von_karman,
            self.initial_correlation_length_m,
        ) <= 0.0:
            raise ConfigError("flow wall and correlation scales must be positive")

    @property
    def pressure_acceleration_m_s2(self) -> float:
        if self.momentum_forcing != "pressure_gradient":
            return 0.0
        return self.friction_velocity_m_s**2 / self.forcing_height_m


@dataclass(frozen=True, slots=True)
class WallConfig:
    model: str
    filter_grid_ratio: float
    test_filter_ratio: float
    porte_agel_correction: bool

    def __post_init__(self) -> None:
        if self.model not in (
            "filtered_neutral_log",
            "neutral_log",
            "monin_obukhov",
        ):
            raise ConfigError("unsupported wall.model")
        if self.filter_grid_ratio <= 0.0 or self.test_filter_ratio <= 1.0:
            raise ConfigError("wall filter ratios are invalid")


@dataclass(frozen=True, slots=True)
class ThermalConfig:
    boundary_condition: str
    initial_temperature_k: float
    reference_temperature_k: float
    inversion_height_m: float
    inversion_gradient_k_m: float
    perturbation_amplitude_k: float
    perturbation_height_m: float
    surface_cooling_k_s: float
    surface_heat_flux_k_m_s: float
    thermal_roughness_length_m: float
    gravity_m_s2: float

    def __post_init__(self) -> None:
        if self.boundary_condition not in (
            "none",
            "prescribed_surface_temperature",
            "fixed_surface_flux",
        ):
            raise ConfigError("unsupported thermal.boundary_condition")
        if min(
            self.initial_temperature_k,
            self.reference_temperature_k,
            self.thermal_roughness_length_m,
            self.gravity_m_s2,
        ) <= 0.0:
            raise ConfigError("thermal reference values must be positive")
        if min(
            self.inversion_height_m,
            self.inversion_gradient_k_m,
            self.perturbation_amplitude_k,
            self.perturbation_height_m,
        ) < 0.0:
            raise ConfigError("thermal profile values must be nonnegative")


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

    def __post_init__(self) -> None:
        if self.model != "lasd":
            raise ConfigError("sgs.model must be 'lasd'")
        if self.update_interval_steps <= 0:
            raise ConfigError("LASD update interval must be positive")
        if self.filter_grid_ratio <= 0.0 or self.test_filter_ratio <= 1.0:
            raise ConfigError("LASD filter ratios are invalid")
        if self.timescale_coefficient <= 0.0:
            raise ConfigError("LASD timescale coefficient must be positive")
        if not (
            0.0
            <= self.minimum_coefficient
            <= self.initial_coefficient
            <= self.maximum_coefficient
        ):
            raise ConfigError("LASD coefficient bounds are inconsistent")

    @property
    def lasd_update_interval(self) -> int:
        """Compatibility spelling used by the stratified warmup engine."""

        return self.update_interval_steps


@dataclass(frozen=True, slots=True)
class TimeConfig:
    integrator: str
    dt_seconds: float
    duration_hours: float
    sample_start_hours: float

    def __post_init__(self) -> None:
        if self.integrator != "ab2":
            raise ConfigError("time.integrator must be 'ab2'")
        if self.dt_seconds <= 0.0 or self.duration_hours <= 0.0:
            raise ConfigError("time step and duration must be positive")
        if not 0.0 <= self.sample_start_hours < self.duration_hours:
            raise ConfigError("sample start must lie inside the run")
        _ = self.steps

    @property
    def steps(self) -> int:
        raw = self.duration_hours * 3600.0 / self.dt_seconds
        result = int(round(raw))
        if not math.isclose(raw, result, abs_tol=1.0e-10):
            raise ConfigError("duration must contain an integer number of steps")
        return result

    @property
    def sample_start_step(self) -> int:
        raw = self.sample_start_hours * 3600.0 / self.dt_seconds
        result = int(round(raw))
        if not math.isclose(raw, result, abs_tol=1.0e-10):
            raise ConfigError("sample start must fall on an accepted step")
        return result


@dataclass(frozen=True, slots=True)
class NumericsConfig:
    dtype: str
    pressure_method: str
    seed: int
    cfl_abort: float
    cfl_warning: float
    lasd_trajectory_cfl_abort: float

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "float64"):
            raise ConfigError("numerics.dtype must be float32 or float64")
        if self.pressure_method not in ("transpose", "spike", "spike-adaptive"):
            raise ConfigError("unsupported pressure method")
        if self.seed < 0:
            raise ConfigError("random seed must be nonnegative")
        if min(
            self.cfl_abort,
            self.cfl_warning,
            self.lasd_trajectory_cfl_abort,
        ) <= 0.0:
            raise ConfigError("CFL thresholds must be positive")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    directory: str
    sample_start_hours: float
    sample_every_steps: int
    log_every_steps: int
    checkpoint_every_steps: int

    def __post_init__(self) -> None:
        if not self.directory:
            raise ConfigError("output.directory must be non-empty")
        if min(
            self.sample_every_steps,
            self.log_every_steps,
            self.checkpoint_every_steps,
        ) <= 0:
            raise ConfigError("output intervals must be positive")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    name: str
    initial_profiles: Path
    reference_results: Path
    horizontal_coriolis_s: float
    passive_scalar_surface_flux_kg_m2_s: float
    air_density_kg_m3: float
    passive_scalar_reference_kg_m3: float
    thomas_chunk: int
    fig13_budget: bool

    def __post_init__(self) -> None:
        if self.name != "andren1994":
            raise ConfigError("unsupported benchmark.name")
        if not self.initial_profiles.is_file():
            raise ConfigError(
                f"benchmark initial profiles do not exist: {self.initial_profiles}"
            )
        if not self.reference_results.is_file():
            raise ConfigError(
                f"benchmark reference results do not exist: {self.reference_results}"
            )
        if self.horizontal_coriolis_s <= 0.0:
            raise ConfigError("benchmark horizontal Coriolis must be positive")
        if min(
            self.passive_scalar_surface_flux_kg_m2_s,
            self.air_density_kg_m3,
            self.passive_scalar_reference_kg_m3,
        ) <= 0.0:
            raise ConfigError("benchmark passive-scalar scales must be positive")
        if self.thomas_chunk <= 0:
            raise ConfigError("benchmark.thomas_chunk must be positive")


@dataclass(frozen=True, slots=True)
class CaseConfig:
    name: str
    runner: str
    workflow: str
    domain: DomainConfig
    flow: FlowConfig
    wall: WallConfig
    thermal: ThermalConfig
    sgs: SgsConfig
    time: TimeConfig
    numerics: NumericsConfig
    output: OutputConfig
    benchmark: BenchmarkConfig | None

    def __post_init__(self) -> None:
        if self.runner != "abl":
            raise ConfigError("case.runner must be 'abl'")
        if self.workflow != "warmup":
            raise ConfigError("only case.workflow = 'warmup' is implemented")
        if self.flow.roughness_length_m >= 0.5 * self.domain.dz_m:
            raise ConfigError("momentum roughness must be below the first cell")
        if self.thermal.thermal_roughness_length_m >= 0.5 * self.domain.dz_m:
            raise ConfigError("thermal roughness must be below the first cell")
        if self.thermal.inversion_height_m > self.domain.lz_m:
            raise ConfigError("thermal inversion must not exceed the domain height")
        if self.thermal.perturbation_height_m > self.domain.lz_m:
            raise ConfigError("thermal perturbations must lie inside the domain")
        if self.output.sample_start_hours != self.time.sample_start_hours:
            raise ConfigError("time and output sample-start values must match")
        self._validate_physics()

    def _validate_physics(self) -> None:
        if self.benchmark is not None:
            expected = (
                "geostrophic",
                "none",
                "neutral_log",
                "andren1994_table",
            )
            actual = (
                self.flow.momentum_forcing,
                self.thermal.boundary_condition,
                self.wall.model,
                self.flow.initial_velocity_profile,
            )
            if actual != expected:
                raise ConfigError(
                    "Andren1994 requires geostrophic/no-thermal/"
                    f"neutral-log/table choices {expected}, received {actual}"
                )
            if self.flow.coriolis_s <= 0.0:
                raise ConfigError("Andren1994 requires vertical Coriolis rotation")
            if math.hypot(
                self.flow.geostrophic_u_m_s,
                self.flow.geostrophic_v_m_s,
            ) <= 0.0:
                raise ConfigError("Andren1994 requires geostrophic wind")
            return
        boundary = self.thermal.boundary_condition
        expected = {
            "none": (
                "pressure_gradient",
                "filtered_neutral_log",
                "log_law",
            ),
            "prescribed_surface_temperature": (
                "geostrophic",
                "monin_obukhov",
                "zero",
            ),
            "fixed_surface_flux": (
                "none",
                "neutral_log",
                "zero",
            ),
        }[boundary]
        actual = (
            self.flow.momentum_forcing,
            self.wall.model,
            self.flow.initial_velocity_profile,
        )
        if actual != expected:
            raise ConfigError(
                f"thermal boundary {boundary!r} requires forcing/wall/initial "
                f"choices {expected}, received {actual}"
            )
        if boundary == "none":
            if min(
                self.flow.friction_velocity_m_s,
                self.flow.forcing_height_m,
            ) <= 0.0:
                raise ConfigError("unstratified warmup requires pressure forcing")
        elif boundary == "prescribed_surface_temperature":
            if self.flow.coriolis_s <= 0.0 or self.flow.geostrophic_u_m_s == 0.0:
                raise ConfigError("prescribed surface temperature requires rotation")
            if self.thermal.surface_cooling_k_s == 0.0:
                raise ConfigError("prescribed surface temperature must evolve")
        elif self.thermal.surface_heat_flux_k_m_s == 0.0:
            raise ConfigError("fixed surface heat flux must be nonzero")

    @property
    def stability(self) -> str:
        """Derive the ABL stability class from configured thermal forcing."""

        boundary = self.thermal.boundary_condition
        if boundary == "none":
            return "neutral"
        forcing = (
            self.thermal.surface_heat_flux_k_m_s
            if boundary == "fixed_surface_flux"
            else self.thermal.surface_cooling_k_s
        )
        return "unstable" if forcing > 0.0 else "stable"

    @property
    def configured_surface_buoyancy_flux_m2_s3(self) -> float | None:
        if self.thermal.boundary_condition != "fixed_surface_flux":
            return None
        return (
            self.thermal.gravity_m_s2
            * self.thermal.surface_heat_flux_k_m_s
            / self.thermal.reference_temperature_k
        )

    @property
    def estimated_startup_cfl(self) -> float:
        if self.benchmark is not None:
            speed = math.hypot(
                self.flow.geostrophic_u_m_s,
                self.flow.geostrophic_v_m_s,
            )
        elif self.thermal.boundary_condition == "none":
            top = self.domain.lz_m - 0.5 * self.domain.dz_m
            speed = self.flow.friction_velocity_m_s / self.flow.von_karman
            speed *= math.log(top / self.flow.roughness_length_m)
            speed += 3.0 * self.flow.initial_perturbation_rms_m_s
        elif self.thermal.boundary_condition == "prescribed_surface_temperature":
            speed = math.hypot(
                self.flow.geostrophic_u_m_s,
                self.flow.geostrophic_v_m_s,
            )
        else:
            speed = (
                self.thermal.gravity_m_s2
                * abs(self.thermal.surface_heat_flux_k_m_s)
                * max(self.thermal.inversion_height_m, self.domain.dz_m)
                / self.thermal.reference_temperature_k
            ) ** (1.0 / 3.0)
        spacing = (
            min(self.domain.dx_m, self.domain.dy_m)
            if self.benchmark is not None
            else min(
                self.domain.dx_m,
                self.domain.dy_m,
                self.domain.dz_m,
            )
        )
        return speed * self.time.dt_seconds / spacing

    @property
    def estimated_lasd_trajectory_cfl(self) -> float:
        return self.estimated_startup_cfl * self.sgs.update_interval_steps

    @property
    def sample_start_step(self) -> int:
        """Accepted step at which restart-continuous sampling begins."""

        return self.time.sample_start_step

    def resolved(self) -> dict[str, Any]:
        resolved = {
            "case": {
                "name": self.name,
                "runner": self.runner,
                "workflow": self.workflow,
            },
            "derived": {
                "stability": self.stability,
                "configured_surface_buoyancy_flux_m2_s3": (
                    self.configured_surface_buoyancy_flux_m2_s3
                ),
            },
            "domain": {
                "nx": self.domain.nx,
                "ny": self.domain.ny,
                "nz": self.domain.nz,
                "lx_m": self.domain.lx_m,
                "ly_m": self.domain.ly_m,
                "lz_m": self.domain.lz_m,
            },
            "flow": {
                "momentum_forcing": self.flow.momentum_forcing,
                "friction_velocity_m_s": self.flow.friction_velocity_m_s,
                "forcing_height_m": self.flow.forcing_height_m,
                "pressure_acceleration_m_s2": (
                    self.flow.pressure_acceleration_m_s2
                ),
                "geostrophic_u_m_s": self.flow.geostrophic_u_m_s,
                "geostrophic_v_m_s": self.flow.geostrophic_v_m_s,
                "coriolis_s": self.flow.coriolis_s,
                "roughness_length_m": self.flow.roughness_length_m,
                "von_karman": self.flow.von_karman,
                "initial_velocity_profile": self.flow.initial_velocity_profile,
                "initial_perturbation_rms_m_s": (
                    self.flow.initial_perturbation_rms_m_s
                ),
                "initial_correlation_length_m": (
                    self.flow.initial_correlation_length_m
                ),
            },
            "wall": {
                "model": self.wall.model,
                "filter_grid_ratio": self.wall.filter_grid_ratio,
                "test_filter_ratio": self.wall.test_filter_ratio,
                "porte_agel_correction": self.wall.porte_agel_correction,
            },
            "thermal": {
                "boundary_condition": self.thermal.boundary_condition,
                "initial_temperature_k": self.thermal.initial_temperature_k,
                "reference_temperature_k": self.thermal.reference_temperature_k,
                "inversion_height_m": self.thermal.inversion_height_m,
                "inversion_gradient_k_m": self.thermal.inversion_gradient_k_m,
                "perturbation_amplitude_k": self.thermal.perturbation_amplitude_k,
                "perturbation_height_m": self.thermal.perturbation_height_m,
                "surface_cooling_k_s": self.thermal.surface_cooling_k_s,
                "surface_heat_flux_k_m_s": (
                    self.thermal.surface_heat_flux_k_m_s
                ),
                "thermal_roughness_length_m": (
                    self.thermal.thermal_roughness_length_m
                ),
                "gravity_m_s2": self.thermal.gravity_m_s2,
            },
            "sgs": {
                "model": self.sgs.model,
                "update_interval_steps": self.sgs.update_interval_steps,
                "filter_grid_ratio": self.sgs.filter_grid_ratio,
                "test_filter_ratio": self.sgs.test_filter_ratio,
                "timescale_coefficient": self.sgs.timescale_coefficient,
                "initial_coefficient": self.sgs.initial_coefficient,
                "minimum_coefficient": self.sgs.minimum_coefficient,
                "maximum_coefficient": self.sgs.maximum_coefficient,
            },
            "time": {
                "integrator": self.time.integrator,
                "dt_seconds": self.time.dt_seconds,
                "duration_hours": self.time.duration_hours,
                "steps": self.time.steps,
                "sample_start_hours": self.time.sample_start_hours,
            },
            "numerics": {
                "dtype": self.numerics.dtype,
                "pressure_method": self.numerics.pressure_method,
                "seed": self.numerics.seed,
                "cfl_abort": self.numerics.cfl_abort,
                "cfl_warning": self.numerics.cfl_warning,
                "lasd_trajectory_cfl_abort": (
                    self.numerics.lasd_trajectory_cfl_abort
                ),
                "estimated_startup_cfl": self.estimated_startup_cfl,
                "estimated_lasd_trajectory_cfl": (
                    self.estimated_lasd_trajectory_cfl
                ),
            },
            "output": {
                "directory": self.output.directory,
                "sample_start_hours": self.output.sample_start_hours,
                "sample_every_steps": self.output.sample_every_steps,
                "log_every_steps": self.output.log_every_steps,
                "checkpoint_every_steps": self.output.checkpoint_every_steps,
                "checkpoint_latest": "checkpoint_latest.npz",
                "checkpoint_final": "checkpoint_final.npz",
                "checkpoint_layout": "z_slab_boussinesq.v1",
                "history": "history.csv",
                "profiles": "profiles.csv",
                "manifest": "warmup_manifest.json",
            },
        }
        if self.benchmark is not None:
            resolved["benchmark"] = {
                "name": self.benchmark.name,
                "initial_profiles": str(self.benchmark.initial_profiles),
                "reference_results": str(self.benchmark.reference_results),
                "horizontal_coriolis_s": (
                    self.benchmark.horizontal_coriolis_s
                ),
                "passive_scalar_surface_flux_kg_m2_s": (
                    self.benchmark.passive_scalar_surface_flux_kg_m2_s
                ),
                "air_density_kg_m3": self.benchmark.air_density_kg_m3,
                "passive_scalar_reference_kg_m3": (
                    self.benchmark.passive_scalar_reference_kg_m3
                ),
                "thomas_chunk": self.benchmark.thomas_chunk,
                "fig13_budget": self.benchmark.fig13_budget,
            }
        return resolved

    def resolved_toml(self) -> str:
        return toml_dumps(self.resolved())


def load_case(path: str | Path) -> CaseConfig:
    source = Path(path)
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    case = _table(document, "case")
    categorical_keys = {"regime", "stability"}.intersection(case)
    if categorical_keys:
        key = sorted(categorical_keys)[0]
        raise ConfigError(
            f"case.{key} is derived from thermal forcing and must not be set"
        )
    domain = _table(document, "domain")
    flow = _table(document, "flow")
    wall = _table(document, "wall")
    thermal = _table(document, "thermal")
    sgs = _table(document, "sgs")
    time = _table(document, "time")
    numerics = _table(document, "numerics")
    output = _table(document, "output")
    benchmark = document.get("benchmark")
    if benchmark is not None and not isinstance(benchmark, dict):
        raise ConfigError("[benchmark] must be a table")
    sample_start = _number(output, "sample_start_hours")
    return CaseConfig(
        name=_string(case, "name"),
        runner=_string(case, "runner"),
        workflow=_string(case, "workflow"),
        domain=DomainConfig(
            _integer(domain, "nx"),
            _integer(domain, "ny"),
            _integer(domain, "nz"),
            _number(domain, "lx_m"),
            _number(domain, "ly_m"),
            _number(domain, "lz_m"),
        ),
        flow=FlowConfig(
            _string(flow, "momentum_forcing"),
            _number(flow, "friction_velocity_m_s", 0.0),
            _number(flow, "forcing_height_m", 0.0),
            _number(flow, "geostrophic_u_m_s", 0.0),
            _number(flow, "geostrophic_v_m_s", 0.0),
            _number(flow, "coriolis_s", 0.0),
            _number(wall, "roughness_length_m"),
            _number(wall, "von_karman", 0.4),
            _string(flow, "initial_velocity_profile"),
            _number(flow, "initial_perturbation_rms_m_s", 0.0),
            _number(flow, "initial_correlation_length_m", 1.0),
        ),
        wall=WallConfig(
            _string(wall, "model"),
            _number(wall, "filter_grid_ratio", 1.5),
            _number(wall, "test_filter_ratio", 2.0),
            _boolean(wall, "porte_agel_correction", True),
        ),
        thermal=ThermalConfig(
            _string(thermal, "boundary_condition"),
            _number(thermal, "initial_temperature_k", 300.0),
            _number(thermal, "reference_temperature_k", 300.0),
            _number(thermal, "inversion_height_m", 0.0),
            _number(thermal, "inversion_gradient_k_m", 0.0),
            _number(thermal, "perturbation_amplitude_k", 0.0),
            _number(thermal, "perturbation_height_m", 0.0),
            _number(thermal, "surface_cooling_k_s", 0.0),
            _number(thermal, "surface_heat_flux_k_m_s", 0.0),
            _number(wall, "thermal_roughness_length_m", 0.1),
            _number(thermal, "gravity_m_s2", 9.81),
        ),
        sgs=SgsConfig(
            _string(sgs, "model"),
            _integer(sgs, "update_interval_steps"),
            _number(sgs, "filter_grid_ratio", 1.5),
            _number(sgs, "test_filter_ratio", 2.0),
            _number(sgs, "timescale_coefficient", 1.5),
            _number(sgs, "initial_coefficient", 0.03),
            _number(sgs, "minimum_coefficient", 1.0e-6),
            _number(sgs, "maximum_coefficient", 0.81),
        ),
        time=TimeConfig(
            _string(time, "integrator"),
            _number(time, "dt_seconds"),
            _number(time, "duration_hours"),
            sample_start,
        ),
        numerics=NumericsConfig(
            _string(numerics, "dtype"),
            _string(numerics, "pressure_method"),
            _integer(numerics, "seed"),
            _number(numerics, "cfl_abort", 1.0),
            _number(numerics, "cfl_warning", 0.8),
            _number(numerics, "lasd_trajectory_cfl_abort", 1.0),
        ),
        output=OutputConfig(
            _string(output, "directory"),
            sample_start,
            _integer(output, "sample_every_steps"),
            _integer(output, "log_every_steps"),
            _integer(output, "checkpoint_every_steps"),
        ),
        benchmark=(
            None
            if benchmark is None
            else BenchmarkConfig(
                _string(benchmark, "name"),
                (source.parent / _string(benchmark, "initial_profiles")).resolve(),
                (source.parent / _string(benchmark, "reference_results")).resolve(),
                _number(benchmark, "horizontal_coriolis_s"),
                _number(
                    benchmark,
                    "passive_scalar_surface_flux_kg_m2_s",
                ),
                _number(benchmark, "air_density_kg_m3", 1.0),
                _number(benchmark, "passive_scalar_reference_kg_m3", 1.0),
                _integer(benchmark, "thomas_chunk", 20),
                _boolean(benchmark, "fig13_budget", False),
            )
        ),
    )


__all__ = ["BenchmarkConfig", "CaseConfig", "ConfigError", "load_case"]
