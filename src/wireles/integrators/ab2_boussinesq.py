"""Fixed-step AB2 interpretation for a velocity--scalar product state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from wireles.domain import AcceptedClock, EvaluationTime
from wireles.operators import ProjectionResult, project
from wireles.physics import (
    BoussinesqFields,
    BoussinesqTendency,
    IdentityClosureEvent,
)

from .ab2 import AB2Config, ColdStart, Evaluation, PreviousTendency


X = TypeVar("X")
T = TypeVar("T")
D = TypeVar("D")
P = TypeVar("P")


@dataclass(frozen=True, slots=True)
class AB2BoussinesqState(Generic[X, T]):
    fields: X
    clock: AcceptedClock
    history: ColdStart | PreviousTendency[T]
    integrator_fingerprint: str

    def __post_init__(self) -> None:
        if not self.integrator_fingerprint:
            raise ValueError("integrator fingerprint must be non-empty")


@dataclass(frozen=True, slots=True)
class AB2BoussinesqStepDiagnostic(Generic[D, P]):
    evaluation_time: EvaluationTime
    accepted_clock: AcceptedClock
    used_euler_startup: bool
    closure_event: Any
    vector_field: D
    projection: P


@dataclass(frozen=True, slots=True)
class AB2BoussinesqStepResult(Generic[X, T, D, P]):
    state: AB2BoussinesqState[X, T]
    diagnostic: AB2BoussinesqStepDiagnostic[D, P]


class BoussinesqAB2Algebra(Protocol):
    def ab2_candidate_velocity(self, *args, **kwargs) -> Any: ...

    def ab2_candidate_scalar(self, *args, **kwargs) -> Any: ...

    def accept_scalar(self, scalar: Any) -> Any: ...


def cold_start_boussinesq(
    fields: X,
    *,
    clock: AcceptedClock,
    config: AB2Config,
) -> AB2BoussinesqState[X, Any]:
    return AB2BoussinesqState(fields, clock, ColdStart(), config.fingerprint)


def step_boussinesq(
    state: AB2BoussinesqState,
    *,
    config: AB2Config,
    environment: Any,
    vector_field: Any,
    normal_boundary: Any,
    algebra: BoussinesqAB2Algebra,
    pressure_solver: Any,
    closure_event: Any = IdentityClosureEvent(),
) -> AB2BoussinesqStepResult:
    """Advance both fields and apply the terminal projection only to velocity."""
    if state.integrator_fingerprint != config.fingerprint:
        raise ValueError("AB2 state fingerprint does not match the configuration")
    evaluation_time = EvaluationTime(
        state.clock.time,
        state.clock.step,
        "ab2-current",
    )
    prepared_fields, closure_diagnostic = closure_event(
        state.fields,
        state.clock,
        environment,
    )
    evaluated = vector_field(Evaluation(prepared_fields, evaluation_time, environment))
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
    candidate_velocity = algebra.ab2_candidate_velocity(
        prepared_fields.velocity,
        evaluated.tendency.velocity,
        previous.velocity,
        dt=config.dt,
        current_weight=current_weight,
        previous_weight=previous_weight,
    )
    candidate_scalar = algebra.ab2_candidate_scalar(
        prepared_fields.potential_temperature,
        evaluated.tendency.potential_temperature,
        previous.potential_temperature,
        dt=config.dt,
        current_weight=current_weight,
        previous_weight=previous_weight,
    )
    accepted_clock = state.clock.advance(config.dt)
    projected: ProjectionResult = project(
        candidate_velocity,
        dt=config.dt,
        normal_boundary=normal_boundary(accepted_clock, environment),
        algebra=algebra,
        pressure_solver=pressure_solver,
    )
    accepted_fields = BoussinesqFields(
        projected.velocity,
        algebra.accept_scalar(candidate_scalar),
        prepared_fields.closure,
    )
    accepted = AB2BoussinesqState(
        accepted_fields,
        accepted_clock,
        PreviousTendency(
            BoussinesqTendency(
                evaluated.tendency.velocity,
                evaluated.tendency.potential_temperature,
            )
        ),
        config.fingerprint,
    )
    return AB2BoussinesqStepResult(
        accepted,
        AB2BoussinesqStepDiagnostic(
            evaluation_time,
            accepted_clock,
            startup,
            closure_diagnostic,
            evaluated.diagnostic,
            projected,
        ),
    )
