"""Array-independent accepted and evaluation clock values."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class AcceptedClock:
    """Physical time and index of the last accepted full-step state."""

    time: float
    step: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.time):
            raise ValueError("accepted physical time must be finite")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
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
        if not math.isfinite(self.time):
            raise ValueError("evaluation time must be finite")
        if (
            isinstance(self.accepted_step, bool)
            or not isinstance(self.accepted_step, int)
            or self.accepted_step < 0
        ):
            raise ValueError("evaluation accepted step must be nonnegative")
        if not self.identity:
            raise ValueError("evaluation identity must be non-empty")
