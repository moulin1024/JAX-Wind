"""Explicit physical-boundary values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class VerticalBoundary(Generic[T]):
    """Lower and upper values at genuine physical vertical faces."""

    lower: T
    upper: T

