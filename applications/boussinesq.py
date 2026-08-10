"""Data products used to declare a Boussinesq case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jaxwind.domain import PassiveScalarScaleSystem, ScaleSystem, UniformGrid
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
class TabulatedVelocityTKE:
    """Horizontally homogeneous means and isotropic TKE perturbations."""

    path: Path
    seed: int

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("initial-condition seed must be nonnegative")


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
    scalar_scales: PassiveScalarScaleSystem
    model: BoussinesqModel
    integrator: AB2Config
    initial_condition: TabulatedVelocityTKE
    reference_results: Path
    pressure: PressureProjection
    output: OutputSchedule
    steps: int
    cfl_warning: float
    cfl_abort: float
    trajectory_cfl_abort: float
    nonlinear_padding_ratio: float = 1.5

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
        if self.nonlinear_padding_ratio < 1.0:
            raise ValueError("nonlinear padding ratio cannot be below one")

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
    "OutputSchedule",
    "PressureProjection",
    "TabulatedVelocityTKE",
]
