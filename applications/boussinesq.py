"""Data products used to declare a Boussinesq case."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from jaxwind.domain import ScaleSystem, UniformGrid
from jaxwind.integrators import AB2Config
from jaxwind.physics import BoussinesqModel


@dataclass(frozen=True, slots=True)
class PressureProjection:
    """Numerical parameters for the generic pressure adapter."""

    dtype: str
    method: str
    thomas_chunk: int = 1

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "float64"):
            raise ValueError("pressure dtype must be float32 or float64")
        if self.method not in ("transpose", "spike", "spike-adaptive"):
            raise ValueError("unsupported pressure method")
        if self.thomas_chunk <= 0:
            raise ValueError("Thomas chunk must be positive")


@dataclass(frozen=True, slots=True)
class TabulatedBoussinesqState:
    """Tabulated means and perturbation amplitudes for every evolved field."""

    path: Path
    seed: int

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("initial-condition seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class ScalarScaleSystem:
    """Unit-aware scalar scale independent of its buoyancy coupling."""

    mechanical: ScaleSystem
    magnitude: float
    quantity: str
    reference_value: float = 0.0
    version: str = "jaxwind.application-scalar-scales.v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.magnitude) or self.magnitude <= 0.0:
            raise ValueError("scalar scale must be finite and positive")
        if self.quantity not in (
            "passive_concentration",
            "potential_temperature",
        ):
            raise ValueError("unsupported scalar quantity")
        if not math.isfinite(self.reference_value):
            raise ValueError("scalar reference value must be finite")

    @property
    def fingerprint(self) -> str:
        return (
            f"{self.version}|mechanical={self.mechanical.fingerprint}"
            f"|magnitude={float(self.magnitude).hex()}"
            f"|quantity={self.quantity}"
            f"|reference={float(self.reference_value).hex()}"
        )

    @property
    def field_quantity(self):
        from jaxwind.domain import (
            PassiveScalarConcentration,
            PotentialTemperaturePerturbation,
        )

        return (
            PotentialTemperaturePerturbation
            if self.quantity == "potential_temperature"
            else PassiveScalarConcentration
        )

    def to_execution_scalar(self, value):
        return (value - self.reference_value) / self.magnitude

    def from_execution_scalar(self, value):
        return value * self.magnitude + self.reference_value

    def to_execution_flux(self, value):
        return value / (self.magnitude * self.mechanical.velocity)

    def from_execution_flux(self, value):
        return value * self.magnitude * self.mechanical.velocity

    def to_execution_buoyancy_coefficient(self, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("buoyancy acceleration per scalar must be finite")
        return value * self.magnitude / self.mechanical.acceleration

    def from_execution_buoyancy_coefficient(self, value: float) -> float:
        return value * self.mechanical.acceleration / self.magnitude


@dataclass(frozen=True, slots=True)
class SurfaceScalarEvolution:
    """Physical scalar value prescribed at a rough surface over time."""

    initial_value: float
    rate_per_second: float
    roughness_length_m: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.initial_value, self.rate_per_second)
        ):
            raise ValueError("surface-scalar values must be finite")
        if (
            not math.isfinite(self.roughness_length_m)
            or self.roughness_length_m <= 0.0
        ):
            raise ValueError("surface-scalar roughness must be finite and positive")


@dataclass(frozen=True, slots=True)
class DiagnosticReference:
    """Case-data normalization for generic profiles, spectra, and bulk metrics."""

    length_m: float
    velocity_m_s: float
    scalar: float
    inversion_search_max_height_m: float
    spectrum_heights_m: tuple[float, ...]

    def __post_init__(self) -> None:
        if min(self.length_m, self.velocity_m_s, self.scalar) <= 0.0:
            raise ValueError("diagnostic scales must be positive")
        if self.inversion_search_max_height_m <= 0.0:
            raise ValueError("inversion search height must be positive")
        if not self.spectrum_heights_m or any(
            value <= 0.0 for value in self.spectrum_heights_m
        ):
            raise ValueError("spectrum heights must be positive")


@dataclass(frozen=True, slots=True)
class OutputSchedule:
    directory: Path
    sample_start_step: int
    sample_every_steps: int
    log_every_steps: int
    checkpoint_every_steps: int

    def __post_init__(self) -> None:
        if self.sample_start_step < 0:
            raise ValueError("sample start must be nonnegative")
        if min(
            self.sample_every_steps,
            self.log_every_steps,
            self.checkpoint_every_steps,
        ) <= 0:
            raise ValueError("output intervals must be positive")


@dataclass(frozen=True, slots=True)
class BoussinesqCase:
    """A fully composed case consumed without implementation dispatch."""

    name: str
    citation: str
    physical_grid: UniformGrid
    mechanical_scales: ScaleSystem
    scalar_scales: ScalarScaleSystem
    model: BoussinesqModel
    integrator: AB2Config
    initial_condition: TabulatedBoussinesqState
    diagnostic_reference: DiagnosticReference
    reference_results: Path
    pressure: PressureProjection
    output: OutputSchedule
    steps: int
    cfl_warning: float
    cfl_abort: float
    trajectory_cfl_abort: float
    advection_frame_velocity_m_s: tuple[float, float] = (0.0, 0.0)
    nonlinear_padding_ratio: float = 1.5
    nonlinear_dealiasing: str = "three_halves"

    def __post_init__(self) -> None:
        if not self.name or not self.citation:
            raise ValueError("case name and citation must be non-empty")
        if self.scalar_scales.mechanical != self.mechanical_scales:
            raise ValueError("scalar and mechanical scales must share a basis")
        if self.steps <= 0:
            raise ValueError("case steps must be positive")
        if self.output.sample_start_step >= self.steps:
            raise ValueError("sample start must precede the final step")
        if not 0.0 < self.cfl_warning < self.cfl_abort:
            raise ValueError("CFL limits are inconsistent")
        if self.trajectory_cfl_abort <= 0.0:
            raise ValueError("trajectory CFL limit must be positive")
        if not all(
            math.isfinite(value) for value in self.advection_frame_velocity_m_s
        ):
            raise ValueError("advection-frame velocity must be finite")
        if self.nonlinear_dealiasing not in (
            "three_halves",
            "two_thirds",
            "legacy_two_thirds",
        ):
            raise ValueError(
                "nonlinear dealiasing must be three_halves, two_thirds, "
                "or legacy_two_thirds"
            )
        if (
            self.nonlinear_dealiasing == "three_halves"
            and self.nonlinear_padding_ratio < 1.5
        ):
            raise ValueError("nonlinear padding ratio must be at least 1.5")

    @property
    def dt_seconds(self) -> float:
        return float(
            self.mechanical_scales.from_execution_time(self.integrator.dt)
        )

    @property
    def duration_seconds(self) -> float:
        return self.steps * self.dt_seconds


__all__ = [
    "BoussinesqCase",
    "DiagnosticReference",
    "OutputSchedule",
    "PressureProjection",
    "ScalarScaleSystem",
    "SurfaceScalarEvolution",
    "TabulatedBoussinesqState",
]
