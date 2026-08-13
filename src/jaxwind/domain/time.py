"""Array-independent accepted and evaluation clock values."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real


def _is_scalar_array(value: object) -> bool:
    """Recognize backend scalar arrays without importing a numerical backend."""

    return getattr(value, "shape", None) == () and hasattr(value, "dtype")


@dataclass(frozen=True, slots=True)
class AcceptedClock:
    """Physical time and index of the last accepted full-step state."""

    time: float
    step: int

    def __post_init__(self) -> None:
        if isinstance(self.time, Real) and not math.isfinite(self.time):
            raise ValueError("accepted physical time must be finite")
        if not isinstance(self.time, Real) and not _is_scalar_array(self.time):
            raise TypeError("accepted physical time must be a real scalar")
        if isinstance(self.step, bool) or (
            not isinstance(self.step, Integral) and not _is_scalar_array(self.step)
        ):
            raise ValueError("accepted step must be a nonnegative integer")
        if isinstance(self.step, Integral) and self.step < 0:
            raise ValueError("accepted step must be a nonnegative integer")

    def advance(self, dt: float) -> "AcceptedClock":
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("clock dt must be finite and positive")
        return AcceptedClock(self.time + dt, self.step + 1)


@dataclass(frozen=True, slots=True)
class EvaluationTime:
    """Explicit physical time observed by one vector-field evaluation."""

    time: float
    accepted_step: int
    identity: str

    def __post_init__(self) -> None:
        if isinstance(self.time, Real) and not math.isfinite(self.time):
            raise ValueError("evaluation time must be finite")
        if not isinstance(self.time, Real) and not _is_scalar_array(self.time):
            raise TypeError("evaluation time must be a real scalar")
        if (
            isinstance(self.accepted_step, bool)
            or (
                not isinstance(self.accepted_step, Integral)
                and not _is_scalar_array(self.accepted_step)
            )
            or (
                isinstance(self.accepted_step, Integral)
                and self.accepted_step < 0
            )
        ):
            raise ValueError("evaluation accepted step must be nonnegative")
        if not self.identity:
            raise ValueError("evaluation identity must be non-empty")
