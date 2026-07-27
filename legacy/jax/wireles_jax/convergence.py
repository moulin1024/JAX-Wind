from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class UStarSlidingWindow:
    """Stationarity criterion based on a moving mean of wall friction velocity."""

    target: float
    relative_tolerance: float
    window_samples: int
    minimum_step: int = 0
    _values: deque[float] = field(init=False, repr=False)
    converged: bool = field(default=False, init=False)
    mean: float | None = field(default=None, init=False)
    last_step: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.target <= 0.0:
            raise ValueError("target ustar must be positive")
        if self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive")
        if self.window_samples <= 0:
            raise ValueError("window_samples must be positive")
        if self.minimum_step < 0:
            raise ValueError("minimum_step must be nonnegative")
        self._values = deque(maxlen=self.window_samples)

    @property
    def sample_count(self) -> int:
        return len(self._values)

    @property
    def lower_bound(self) -> float:
        return self.target * (1.0 - self.relative_tolerance)

    @property
    def upper_bound(self) -> float:
        return self.target * (1.0 + self.relative_tolerance)

    def update(self, step: int, instantaneous_ustar: float) -> bool:
        self.last_step = int(step)
        self._values.append(float(instantaneous_ustar))
        self.mean = sum(self._values) / len(self._values)
        self.converged = (
            self.last_step >= self.minimum_step
            and len(self._values) == self.window_samples
            and self.lower_bound <= self.mean <= self.upper_bound
        )
        return self.converged

