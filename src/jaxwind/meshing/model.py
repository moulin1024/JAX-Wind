"""Validated semantic inputs and outputs for analytic meshing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from jaxwind.domain import RectilinearGrid


AxisClustering = Literal["uniform", "single", "double"]


@dataclass(frozen=True, slots=True)
class AxisMeshSpec:
    """Analytic clustering controls for one physical coordinate axis.

    ``single`` clusters toward one domain boundary, selected by ``point``.
    ``double`` clusters from both sides toward one interior ``point``.
    ``strength=0`` is the exact uniform-grid limit for every mode.
    """

    lower: float
    upper: float
    cells: int
    clustering: AxisClustering = "uniform"
    point: float | None = None
    strength: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.lower, self.upper)):
            raise ValueError("axis limits must be finite")
        if self.upper <= self.lower:
            raise ValueError("axis upper limit must exceed its lower limit")
        if isinstance(self.cells, bool) or not isinstance(self.cells, int):
            raise ValueError("axis cell count must be an integer")
        if self.cells <= 0:
            raise ValueError("axis cell count must be positive")
        if self.clustering not in {"uniform", "single", "double"}:
            raise ValueError("axis clustering must be 'uniform', 'single', or 'double'")
        if not math.isfinite(self.strength) or not 0.0 <= self.strength <= 50.0:
            raise ValueError("axis clustering strength must be in [0, 50]")

        if self.clustering == "uniform":
            if self.point is not None:
                raise ValueError("uniform clustering does not accept a point")
            if self.strength != 0.0:
                raise ValueError("uniform clustering requires zero strength")
            return

        if self.point is None or not math.isfinite(self.point):
            raise ValueError("clustered axes require a finite point")
        tolerance = 1.0e-12 * max(1.0, abs(self.lower), abs(self.upper))
        at_lower = math.isclose(self.point, self.lower, abs_tol=tolerance)
        at_upper = math.isclose(self.point, self.upper, abs_tol=tolerance)
        if self.clustering == "single":
            if not (at_lower or at_upper):
                raise ValueError(
                    "single-sided clustering point must be an axis boundary"
                )
            object.__setattr__(self, "point", self.lower if at_lower else self.upper)
            return

        if self.cells < 2:
            raise ValueError("double-sided clustering requires at least two cells")
        if not self.lower < self.point < self.upper:
            raise ValueError(
                "double-sided clustering point must be strictly inside the axis"
            )


@dataclass(frozen=True, slots=True)
class MeshSpec:
    """Independent clustering controls for all three Cartesian axes."""

    x: AxisMeshSpec
    y: AxisMeshSpec
    z: AxisMeshSpec


@dataclass(frozen=True, slots=True)
class AxisMeshStatistics:
    """Geometry-only quality summary for one generated axis."""

    minimum_spacing: float
    maximum_spacing: float
    maximum_adjacent_ratio: float


@dataclass(frozen=True, slots=True)
class GeneratedMesh:
    """A generated physical grid together with its reproducible specification."""

    specification: MeshSpec
    grid: RectilinearGrid
