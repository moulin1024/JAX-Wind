"""A projection program interpreted by reference and production algebras."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Generic, Protocol, TypeVar


X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")
P = TypeVar("P")
D = TypeVar("D")


@dataclass(frozen=True, slots=True)
class VelocityVector(Generic[X, Y, Z]):
    """The hybrid velocity product: cell x/y and face-normal z."""

    x: X
    y: Y
    z: Z


@dataclass(frozen=True, slots=True)
class PressureGradient(Generic[X, Y, Z]):
    """The three compatible components of a pressure gradient."""

    x: X
    y: Y
    z: Z


@dataclass(frozen=True, slots=True)
class ProjectionResult(Generic[X, P, D]):
    """Projected velocity, its gauge-fixed pressure, and residual divergence."""

    velocity: X
    pressure: P
    divergence: D


class PressureSolver(Protocol):
    def solve(self, rhs: Any) -> Any: ...


class ProjectionAlgebra(Protocol):
    def enforce_normal_boundary(self, velocity: Any, boundary: Any) -> Any: ...

    def velocity_divergence(self, velocity: Any) -> Any: ...

    def pressure_rhs(self, divergence: Any, inverse_dt: float) -> Any: ...

    def pressure_gradient(self, pressure: Any) -> Any: ...

    def correct_velocity(
        self,
        velocity: Any,
        gradient: Any,
        dt: float,
    ) -> Any: ...


def project(
    velocity: VelocityVector,
    *,
    dt: float,
    normal_boundary: Any,
    algebra: ProjectionAlgebra,
    pressure_solver: PressureSolver,
    compute_residual: bool = True,
) -> ProjectionResult:
    """Interpret the same compatible projection program in any admitted algebra.

    The residual divergence is diagnostic-only and may be omitted on steps
    where the caller will not inspect it. The projected velocity and pressure
    are independent of this choice.
    """
    if isinstance(dt, Real) and (not math.isfinite(dt) or dt <= 0.0):
        raise ValueError("projection dt must be finite and positive")
    if not isinstance(compute_residual, bool):
        raise TypeError("compute_residual must be a bool")
    prepared = algebra.enforce_normal_boundary(velocity, normal_boundary)
    candidate_divergence = algebra.velocity_divergence(prepared)
    rhs = algebra.pressure_rhs(candidate_divergence, 1.0 / dt)
    pressure = pressure_solver.solve(rhs)
    gradient = algebra.pressure_gradient(pressure)
    projected = algebra.correct_velocity(prepared, gradient, dt)
    residual = (
        algebra.velocity_divergence(projected) if compute_residual else None
    )
    return ProjectionResult(projected, pressure, residual)
