from __future__ import annotations

from dataclasses import dataclass

import pytest

import jaxwind.solver as solver_module
from jaxwind import build_solver, solve


@dataclass(frozen=True)
class Result:
    state: int


def test_solve_repeats_a_pure_transition() -> None:
    seen: list[bool] = []

    def advance(state: int, *, compute_projection_residual: bool = True) -> Result:
        seen.append(compute_projection_residual)
        return Result(state + 2)

    assert solve(1, steps=3, advance=advance) == 7
    assert seen == [False, False, False]


def test_built_solver_accepts_a_dynamic_environment_override(monkeypatch) -> None:
    seen = []

    def step(state, **kwargs):
        seen.append(kwargs["environment"])
        return Result(state + 1)

    monkeypatch.setattr(solver_module, "step_boussinesq", step)
    advance = build_solver(
        config=object(),
        vector_field=object(),
        normal_boundary=object(),
        algebra=object(),
        pressure_solver=object(),
        closure_event=object(),
        environment="default",
    )

    advance(0)
    advance(1, environment="offline-plane")

    assert seen == ["default", "offline-plane"]


@pytest.mark.parametrize("steps", [-1, True, 1.5])
def test_solve_rejects_invalid_step_counts(steps) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        solve(0, steps=steps, advance=lambda state, **_: Result(state))
