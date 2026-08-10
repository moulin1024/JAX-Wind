"""Fixed-step AB2 as a higher-order interpretation of a pure vector field."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Generic, Protocol, TypeAlias, TypeVar

from jaxwind.domain import AcceptedClock, EvaluationTime
from jaxwind.operators import ProjectionResult, project


V = TypeVar("V")
T = TypeVar("T")
E = TypeVar("E")
D = TypeVar("D")
P = TypeVar("P")


@dataclass(frozen=True, slots=True)
class AB2Config:
    """Static fixed-step method configuration and restart fingerprint."""

    dt: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("AB2 dt must be finite and positive")

    @property
    def fingerprint(self) -> str:
        return (
            "jaxwind.ab2.fixed.v1"
            f"|dt={float(self.dt).hex()}"
            "|evaluation=t_n"
            "|projection=terminal-compatible-v1"
            "|normal-boundary=t_next"
        )


@dataclass(frozen=True, slots=True)
class ColdStart:
    """Explicit absence of a previous tendency."""


@dataclass(frozen=True, slots=True)
class PreviousTendency(Generic[T]):
    """The vector field evaluated at the preceding accepted state."""

    value: T


AB2History: TypeAlias = ColdStart | PreviousTendency[T]


@dataclass(frozen=True, slots=True)
class AB2PersistentState(Generic[V, T]):
    """Every method-specific value required for accepted-boundary restart."""

    velocity: V
    clock: AcceptedClock
    history: AB2History[T]
    integrator_fingerprint: str

    def __post_init__(self) -> None:
        if not self.integrator_fingerprint:
            raise ValueError("integrator fingerprint must be non-empty")


@dataclass(frozen=True, slots=True)
class Evaluation(Generic[V, E]):
    """Read-only state, explicit time, and explicit environment for physics."""

    velocity: V
    time: EvaluationTime
    environment: E


@dataclass(frozen=True, slots=True)
class VectorFieldResult(Generic[T, D]):
    tendency: T
    diagnostic: D


@dataclass(frozen=True, slots=True)
class _AB2Stage(Generic[V, T, D]):
    value: V
    evaluated: VectorFieldResult[T, D]
    previous_tendency: T
    evaluation_time: EvaluationTime
    current_weight: float
    previous_weight: float
    used_euler_startup: bool
    preparation_diagnostic: Any


@dataclass(frozen=True, slots=True)
class AB2StepDiagnostic(Generic[D, P]):
    evaluation_time: EvaluationTime
    accepted_clock: AcceptedClock
    used_euler_startup: bool
    vector_field: D
    projection: P


@dataclass(frozen=True, slots=True)
class AB2StepResult(Generic[V, T, D, P]):
    state: AB2PersistentState[V, T]
    diagnostic: AB2StepDiagnostic[D, P]


class AB2VelocityAlgebra(Protocol):
    def ab2_candidate_velocity(
        self,
        velocity: Any,
        current_tendency: Any,
        previous_tendency: Any,
        *,
        dt: float,
        current_weight: float,
        previous_weight: float,
    ) -> Any: ...


class VectorField(Protocol):
    def __call__(self, evaluation: Evaluation) -> VectorFieldResult: ...


class NormalBoundaryLaw(Protocol):
    def __call__(self, clock: AcceptedClock, environment: Any) -> Any: ...


def cold_start(
    velocity: V,
    *,
    clock: AcceptedClock,
    config: AB2Config,
) -> AB2PersistentState[V, Any]:
    return AB2PersistentState(velocity, clock, ColdStart(), config.fingerprint)


def _stage_ab2(
    state: Any,
    *,
    value: V,
    config: AB2Config,
    environment: E,
    vector_field: VectorField,
    prepare: Any = None,
) -> _AB2Stage:
    """Evaluate the control law shared by every AB2 prognostic product."""

    if state.integrator_fingerprint != config.fingerprint:
        raise ValueError("AB2 state fingerprint does not match the configuration")
    if prepare is None:
        prepared = value
        preparation_diagnostic = None
    else:
        prepared, preparation_diagnostic = prepare(
            value,
            state.clock,
            environment,
        )
    evaluation_time = EvaluationTime(
        state.clock.time,
        state.clock.step,
        "ab2-current",
    )
    evaluated = vector_field(Evaluation(prepared, evaluation_time, environment))
    if isinstance(state.history, ColdStart):
        previous = evaluated.tendency
        current_weight = 1.0
        previous_weight = 0.0
        startup = True
    else:
        previous = state.history.value
        current_weight = 1.5
        previous_weight = -0.5
        startup = False
    return _AB2Stage(
        prepared,
        evaluated,
        previous,
        evaluation_time,
        current_weight,
        previous_weight,
        startup,
        preparation_diagnostic,
    )


def step(
    state: AB2PersistentState,
    *,
    config: AB2Config,
    environment: E,
    vector_field: VectorField,
    normal_boundary: NormalBoundaryLaw,
    algebra: AB2VelocityAlgebra,
    pressure_solver: Any,
    compute_projection_residual: bool = True,
) -> AB2StepResult:
    """Advance one accepted step with one vector evaluation and one projection."""
    staged = _stage_ab2(
        state,
        value=state.velocity,
        config=config,
        environment=environment,
        vector_field=vector_field,
    )
    candidate = algebra.ab2_candidate_velocity(
        staged.value,
        staged.evaluated.tendency,
        staged.previous_tendency,
        dt=config.dt,
        current_weight=staged.current_weight,
        previous_weight=staged.previous_weight,
    )
    accepted_clock = state.clock.advance(config.dt)
    projected: ProjectionResult = project(
        candidate,
        dt=config.dt,
        normal_boundary=normal_boundary(accepted_clock, environment),
        algebra=algebra,
        pressure_solver=pressure_solver,
        compute_residual=compute_projection_residual,
    )
    accepted = AB2PersistentState(
        projected.velocity,
        accepted_clock,
        PreviousTendency(staged.evaluated.tendency),
        config.fingerprint,
    )
    diagnostic = AB2StepDiagnostic(
        staged.evaluation_time,
        accepted_clock,
        staged.used_euler_startup,
        staged.evaluated.diagnostic,
        projected,
    )
    return AB2StepResult(accepted, diagnostic)
