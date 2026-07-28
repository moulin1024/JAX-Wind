"""Validated configuration models for direct ALM cases."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from jaxwind.openfast import OpenFASTModalTurbine, OpenFASTRigidTurbine

from .._toml import dumps as toml_dumps


class ConfigError(ValueError):
    """The direct rigid-ALM case is incomplete or inconsistent."""


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

    def __post_init__(self) -> None:
        if min(
            self.friction_velocity_m_s,
            self.roughness_length_m,
            self.forcing_height_m,
            self.von_karman,
        ) <= 0.0:
            raise ConfigError("flow scales and wall constants must be positive")

    @property
    def pressure_acceleration_m_s2(self) -> float:
        return self.friction_velocity_m_s**2 / self.forcing_height_m


@dataclass(frozen=True, slots=True)
class StaticSgsConfig:
    coefficient: float

    def __post_init__(self) -> None:
        if self.coefficient < 0.0:
            raise ConfigError("sgs.coefficient must be nonnegative")


@dataclass(frozen=True, slots=True)
class TimeConfig:
    dt_seconds: float
    steps: int

    def __post_init__(self) -> None:
        if self.dt_seconds <= 0.0:
            raise ConfigError("time.dt_seconds must be positive")
        if self.steps <= 0:
            raise ConfigError("time.steps must be positive")


@dataclass(frozen=True, slots=True)
class NumericsConfig:
    dtype: str
    pressure_method: str
    cfl_abort: float

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "float64"):
            raise ConfigError("numerics.dtype must be float32 or float64")
        if self.pressure_method not in (
            "transpose",
            "spike",
            "spike-adaptive",
        ):
            raise ConfigError("unsupported pressure method")
        if self.cfl_abort <= 0.0:
            raise ConfigError("numerics.cfl_abort must be positive")


@dataclass(frozen=True, slots=True)
class AeroelasticConfig:
    enabled: bool = False
    air_density_kg_m3: float = 1.225
    gravity_m_s2: float = 9.80665
    maximum_tip_deflection_m: float = 20.0

    def __post_init__(self) -> None:
        if self.air_density_kg_m3 <= 0.0:
            raise ConfigError("aeroelastic.air_density_kg_m3 must be positive")
        if self.gravity_m_s2 < 0.0:
            raise ConfigError("aeroelastic.gravity_m_s2 must be nonnegative")
        if self.maximum_tip_deflection_m <= 0.0:
            raise ConfigError(
                "aeroelastic.maximum_tip_deflection_m must be positive"
            )


@dataclass(frozen=True, slots=True)
class TurbineConfig:
    location_m: tuple[float, float]
    openfast_input_file: Path
    openfast: OpenFASTRigidTurbine
    modal_openfast: OpenFASTModalTurbine | None
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
    flow_slice_every_steps: int | None = None

    def __post_init__(self) -> None:
        if not self.directory:
            raise ConfigError("output.directory must be non-empty")
        if self.log_every_steps <= 0:
            raise ConfigError("output.log_every_steps must be positive")
        if (
            self.flow_slice_every_steps is not None
            and self.flow_slice_every_steps <= 0
        ):
            raise ConfigError(
                "output.flow_slice_every_steps must be positive"
            )


@dataclass(frozen=True, slots=True)
class CaseConfig:
    runner: str
    name: str
    domain: DomainConfig
    flow: FlowConfig
    sgs: StaticSgsConfig
    time: TimeConfig
    numerics: NumericsConfig
    aeroelastic: AeroelasticConfig
    turbine: TurbineConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        if self.runner not in (
            "direct_rigid_alm",
            "direct_aeroelastic_alm",
        ):
            raise ConfigError(
                "case.runner must be 'direct_rigid_alm', "
                "or 'direct_aeroelastic_alm' for this runner"
            )
        if (
            self.runner == "direct_aeroelastic_alm"
            and not self.aeroelastic.enabled
        ):
            raise ConfigError(
                "direct_aeroelastic_alm requires aeroelastic.enabled = true"
            )
        if self.aeroelastic.enabled and self.turbine.modal_openfast is None:
            raise ConfigError(
                "enabled aeroelastic coupling requires modal OpenFAST data"
            )
        if not self.name:
            raise ConfigError("case.name must be non-empty")
        if self.flow.forcing_height_m > self.domain.lz_m:
            raise ConfigError("forcing height cannot exceed the domain height")
        if self.flow.roughness_length_m >= 0.5 * self.domain.dz_m:
            raise ConfigError(
                "roughness length must lie below the first cell center"
            )
        x, y = self.turbine.location_m
        if not 0.0 <= x < self.domain.lx_m:
            raise ConfigError("turbine x location lies outside the domain")
        if not 0.0 <= y < self.domain.ly_m:
            raise ConfigError("turbine y location lies outside the domain")
        radius = self.turbine.openfast.tip_radius_m
        if self.turbine.hub_height_m - radius <= 0.0:
            raise ConfigError("turbine rotor intersects the lower wall")
        if self.turbine.hub_height_m + radius >= self.domain.lz_m:
            raise ConfigError("turbine rotor intersects the upper wall")
        if self.estimated_initial_cfl >= self.numerics.cfl_abort:
            raise ConfigError(
                "estimated initial CFL exceeds numerics.cfl_abort; reduce dt"
            )

    @property
    def top_cell_height_m(self) -> float:
        return self.domain.lz_m - 0.5 * self.domain.dz_m

    @property
    def top_log_velocity_m_s(self) -> float:
        return (
            self.flow.friction_velocity_m_s
            / self.flow.von_karman
            * math.log(
                self.top_cell_height_m / self.flow.roughness_length_m
            )
        )

    @property
    def estimated_initial_cfl(self) -> float:
        return (
            self.top_log_velocity_m_s
            * self.time.dt_seconds
            / min(self.domain.dx_m, self.domain.dy_m, self.domain.dz_m)
        )

    @property
    def cell_count(self) -> int:
        return self.domain.nx * self.domain.ny * self.domain.nz

    def resolved(self) -> dict[str, Any]:
        turbine = self.turbine
        compatibility_notes = list(turbine.openfast.compatibility_notes)
        if turbine.modal_openfast is not None:
            compatibility_notes = [
                note
                for note in compatibility_notes
                if "ElastoDyn structural flexibility" not in note
            ]
            compatibility_notes.extend(
                turbine.modal_openfast.compatibility_notes
            )
        return {
            "runner": self.runner,
            "case": self.name,
            "domain": {
                "cells": [
                    self.domain.nx,
                    self.domain.ny,
                    self.domain.nz,
                ],
                "cell_count": self.cell_count,
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
                "initial_condition": "neutral_log_profile",
                "friction_velocity_m_s": (
                    self.flow.friction_velocity_m_s
                ),
                "roughness_length_m": self.flow.roughness_length_m,
                "forcing_height_m": self.flow.forcing_height_m,
                "pressure_acceleration_m_s2": (
                    self.flow.pressure_acceleration_m_s2
                ),
                "top_log_velocity_m_s": self.top_log_velocity_m_s,
            },
            "sgs": {
                "model": "static_smagorinsky",
                "coefficient": self.sgs.coefficient,
            },
            "time": {
                "integrator": "ab2",
                "dt_seconds": self.time.dt_seconds,
                "steps": self.time.steps,
                "duration_seconds": (
                    self.time.dt_seconds * self.time.steps
                ),
            },
            "numerics": {
                "dtype": self.numerics.dtype,
                "pressure_method": self.numerics.pressure_method,
                "estimated_initial_cfl": self.estimated_initial_cfl,
                "cfl_abort": self.numerics.cfl_abort,
            },
            "aeroelastic": {
                "enabled": self.aeroelastic.enabled,
                "coupling": (
                    "explicit_partitioned_newmark"
                    if self.aeroelastic.enabled
                    else "disabled"
                ),
                "air_density_kg_m3": (
                    self.aeroelastic.air_density_kg_m3
                ),
                "gravity_m_s2": self.aeroelastic.gravity_m_s2,
                "maximum_tip_deflection_m": (
                    self.aeroelastic.maximum_tip_deflection_m
                ),
                "enabled_blade_modes": (
                    []
                    if turbine.modal_openfast is None
                    else [
                        name
                        for name, enabled in zip(
                            ("flap1", "flap2", "edge1"),
                            turbine.modal_openfast.enabled_modes,
                        )
                        if enabled
                    ]
                ),
                "blade_structure_file": (
                    None
                    if turbine.modal_openfast is None
                    else str(
                        turbine.modal_openfast.blade_structure.source
                    )
                ),
            },
            "turbine": {
                "model": (
                    "openfast_modal_aeroelastic_actuator_line"
                    if self.aeroelastic.enabled
                    else "openfast_rigid_actuator_line"
                ),
                "location_m": list(turbine.location_m),
                "hub_height_m": turbine.hub_height_m,
                "tip_radius_m": turbine.openfast.tip_radius_m,
                "openfast_input_file": str(
                    turbine.openfast_input_file
                ),
                "rotor_speed_rpm": turbine.rotor_speed_rpm,
                "pitch_degrees": turbine.pitch_degrees,
                "yaw_degrees": turbine.yaw_degrees,
                "initial_azimuth_degrees": (
                    turbine.initial_azimuth_degrees
                ),
                "smoothing_width_m": turbine.smoothing_width_m,
                "blade_count": turbine.openfast.blade_count,
                "blade_element_count": len(
                    turbine.openfast.element_radii_m
                ),
                "airfoil_count": len(turbine.openfast.airfoil_sources),
                "compatibility_notes": compatibility_notes,
            },
            "output": {
                "directory": self.output.directory,
                "log_every_steps": self.output.log_every_steps,
                "flow_slice_every_steps": (
                    self.output.flow_slice_every_steps
                ),
                "writes_full_field_checkpoint": False,
            },
        }

    def resolved_toml(self) -> str:
        return toml_dumps(self.resolved())
