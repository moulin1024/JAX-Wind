"""Pure solver composition over JAX-Wind numerical transitions."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from .integrators import step_boussinesq


State = TypeVar("State")
StepResult = TypeVar("StepResult")


class Advance(Protocol[State, StepResult]):
    """One accepted numerical transition."""

    def __call__(
        self,
        state: State,
        *,
        environment: Any = None,
        compute_projection_residual: bool = True,
    ) -> StepResult: ...


def build_solver(
    *,
    config: Any,
    vector_field: Any,
    normal_boundary: Any,
    algebra: Any,
    pressure_solver: Any,
    closure_event: Any,
    environment: Any = None,
) -> Advance:
    """Close one step with a default, per-call-overridable environment."""

    def advance(
        state: Any,
        *,
        environment: Any = environment,
        compute_projection_residual: bool = True,
    ) -> Any:
        return step_boussinesq(
            state,
            config=config,
            environment=environment,
            vector_field=vector_field,
            normal_boundary=normal_boundary,
            algebra=algebra,
            pressure_solver=pressure_solver,
            closure_event=closure_event,
            compute_projection_residual=compute_projection_residual,
        )

    return advance


def solve(
    state: State,
    *,
    steps: int,
    advance: Advance[State, StepResult],
    compute_projection_residual: bool = False,
) -> State:
    """Apply an accepted transition repeatedly without performing effects."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("solver steps must be a nonnegative integer")
    current = state
    for _ in range(steps):
        result = advance(
            current,
            compute_projection_residual=compute_projection_residual,
        )
        current = result.state
    return current


__all__ = ["Advance", "build_solver", "solve"]
