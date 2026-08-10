from __future__ import annotations

from dataclasses import dataclass

import pytest

from jaxwind import solve


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


@pytest.mark.parametrize("steps", [-1, True, 1.5])
def test_solve_rejects_invalid_step_counts(steps) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        solve(0, steps=steps, advance=lambda state, **_: Result(state))
