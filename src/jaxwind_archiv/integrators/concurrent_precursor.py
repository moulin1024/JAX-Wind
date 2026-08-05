"""Accepted-step coupling of side-by-side precursor and main domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from jaxwind_archiv.physics.wind_tunnel import ConcurrentPrecursorEnvironment

from .ab2 import AB2PersistentState, AB2StepResult, step
from .ab2_boussinesq import step_boussinesq


@dataclass(frozen=True, slots=True)
class ConcurrentPrecursorState:
    """Two complete states resident side by side on the same device mesh."""

    precursor: AB2PersistentState
    main: AB2PersistentState

    def __post_init__(self) -> None:
        if self.precursor.clock != self.main.clock:
            raise ValueError("precursor and main clocks must be synchronized")


@dataclass(frozen=True, slots=True)
class ConcurrentPrecursorStepDiagnostic:
    precursor: Any
    main: Any


@dataclass(frozen=True, slots=True)
class ConcurrentPrecursorStepResult:
    state: ConcurrentPrecursorState
    diagnostic: ConcurrentPrecursorStepDiagnostic


def serial_pair(left: Callable[[], Any], right: Callable[[], Any]) -> tuple[Any, Any]:
    """Reference launcher for deterministic single-threaded execution."""
    return left(), right()


def step_concurrent_precursor(
    state: ConcurrentPrecursorState,
    *,
    config: Any,
    precursor_vector_field: Any,
    main_vector_field: Any,
    normal_boundary: Any,
    algebra: Any,
    precursor_pressure_solver: Any,
    main_pressure_solver: Any,
    launch_pair: Callable[
        [Callable[[], Any], Callable[[], Any]], tuple[Any, Any]
    ] = serial_pair,
    compute_projection_residual: bool = True,
) -> ConcurrentPrecursorStepResult:
    """Advance both domains from one synchronized accepted boundary.

    The main-domain fringe sees the precursor at exactly ``t_n``.  Both
    advances therefore have no data dependency and a runtime may launch them
    on independent GPU streams.  Arrays remain device resident; only the
    small Python product that names the two states is reconstructed.
    """
    if state.precursor.clock != state.main.clock:
        raise ValueError("precursor and main clocks must be synchronized")
    environment = ConcurrentPrecursorEnvironment(state.precursor.velocity)

    def advance_precursor() -> AB2StepResult:
        return step(
            state.precursor,
            config=config,
            environment=None,
            vector_field=precursor_vector_field,
            normal_boundary=normal_boundary,
            algebra=algebra,
            pressure_solver=precursor_pressure_solver,
            compute_projection_residual=compute_projection_residual,
        )

    def advance_main() -> AB2StepResult:
        return step(
            state.main,
            config=config,
            environment=environment,
            vector_field=main_vector_field,
            normal_boundary=normal_boundary,
            algebra=algebra,
            pressure_solver=main_pressure_solver,
            compute_projection_residual=compute_projection_residual,
        )

    precursor, main = launch_pair(advance_precursor, advance_main)
    paired = ConcurrentPrecursorState(precursor.state, main.state)
    return ConcurrentPrecursorStepResult(
        paired,
        ConcurrentPrecursorStepDiagnostic(
            precursor.diagnostic,
            main.diagnostic,
        ),
    )


def step_concurrent_boussinesq_precursor(
    state: ConcurrentPrecursorState,
    *,
    config: Any,
    precursor_vector_field: Any,
    main_vector_field: Any,
    normal_boundary: Any,
    algebra: Any,
    precursor_pressure_solver: Any,
    main_pressure_solver: Any,
    precursor_closure_event: Any,
    main_closure_event: Any,
    launch_pair: Callable[
        [Callable[[], Any], Callable[[], Any]], tuple[Any, Any]
    ] = serial_pair,
    compute_projection_residual: bool = True,
) -> ConcurrentPrecursorStepResult:
    """Advance synchronized velocity--scalar precursor and main domains.

    Each domain owns independent closure memory.  The main fringe samples the
    precursor velocity at ``t_n``, so the two device-resident advances remain
    independent and may be dispatched on separate execution streams.
    """
    if state.precursor.clock != state.main.clock:
        raise ValueError("precursor and main clocks must be synchronized")
    environment = ConcurrentPrecursorEnvironment(
        state.precursor.fields.velocity,
        state.precursor.fields.closure,
    )

    def advance_precursor():
        return step_boussinesq(
            state.precursor,
            config=config,
            environment=None,
            vector_field=precursor_vector_field,
            normal_boundary=normal_boundary,
            algebra=algebra,
            pressure_solver=precursor_pressure_solver,
            closure_event=precursor_closure_event,
            compute_projection_residual=compute_projection_residual,
        )

    def advance_main():
        return step_boussinesq(
            state.main,
            config=config,
            environment=environment,
            vector_field=main_vector_field,
            normal_boundary=normal_boundary,
            algebra=algebra,
            pressure_solver=main_pressure_solver,
            closure_event=main_closure_event,
            compute_projection_residual=compute_projection_residual,
        )

    precursor, main = launch_pair(advance_precursor, advance_main)
    paired = ConcurrentPrecursorState(precursor.state, main.state)
    return ConcurrentPrecursorStepResult(
        paired,
        ConcurrentPrecursorStepDiagnostic(
            precursor.diagnostic,
            main.diagnostic,
        ),
    )
