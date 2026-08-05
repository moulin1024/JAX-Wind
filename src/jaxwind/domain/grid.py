"""Uniform and rectilinear physical-grid metadata in canonical SI units."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


MappingFunction = Literal["uniform", "exponential"]


@dataclass(frozen=True, slots=True)
class AnalyticAxisMapping:
    """Normalized analytic map from computational to physical coordinates.

    ``exponential`` clusters cells toward ``focus``.  A focus at zero or one
    selects a domain boundary; an interior focus joins two exponential maps at
    that physical point.  ``strength=0`` is the exact uniform-grid limit.
    """

    function: MappingFunction = "uniform"
    focus: float | None = None
    strength: float = 0.0

    def __post_init__(self) -> None:
        if self.function not in {"uniform", "exponential"}:
            raise ValueError("axis mapping function must be uniform or exponential")
        if not math.isfinite(self.strength) or not 0.0 <= self.strength <= 50.0:
            raise ValueError("axis mapping strength must lie in [0, 50]")
        if self.function == "uniform":
            if self.focus is not None:
                raise ValueError("uniform axis mapping does not accept a focus")
            if self.strength != 0.0:
                raise ValueError("uniform axis mapping requires zero strength")
            return
        if self.focus is None or not math.isfinite(self.focus):
            raise ValueError("exponential axis mapping requires a finite focus")
        if not 0.0 <= self.focus <= 1.0:
            raise ValueError("axis mapping focus must lie in [0, 1]")


def _uniform_faces(start: float, length: float, count: int) -> tuple[float, ...]:
    return tuple(start + length * index / count for index in range(count + 1))


def _clustered_interval(
    lower: float,
    upper: float,
    cells: int,
    strength: float,
    *,
    toward: Literal["lower", "upper"],
) -> tuple[float, ...]:
    denominator = math.expm1(strength)
    length = upper - lower
    faces = []
    for index in range(cells + 1):
        coordinate = index / cells
        if toward == "lower":
            mapped = math.expm1(strength * coordinate) / denominator
        else:
            mapped = 1.0 - math.expm1(strength * (1.0 - coordinate)) / denominator
        faces.append(lower + length * mapped)
    faces[0] = lower
    faces[-1] = upper
    return tuple(faces)


def analytic_axis_faces(
    count: int,
    length: float,
    mapping: AnalyticAxisMapping = AnalyticAxisMapping(),
    *,
    start: float = 0.0,
) -> tuple[float, ...]:
    """Generate one strictly increasing axis from an analytic mapping."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("axis cell count must be a positive integer")
    if not math.isfinite(start) or not math.isfinite(length) or length <= 0.0:
        raise ValueError("axis origin and length must be finite with positive length")
    if mapping.function == "uniform" or mapping.strength == 0.0:
        return _uniform_faces(start, length, count)

    assert mapping.focus is not None
    lower = start
    upper = start + length
    if mapping.focus == 0.0:
        faces = _clustered_interval(
            lower,
            upper,
            count,
            mapping.strength,
            toward="lower",
        )
    elif mapping.focus == 1.0:
        faces = _clustered_interval(
            lower,
            upper,
            count,
            mapping.strength,
            toward="upper",
        )
    else:
        focus = lower + length * mapping.focus
        left_cells = min(count - 1, max(1, round(count * mapping.focus)))
        right_cells = count - left_cells
        left = _clustered_interval(
            lower,
            focus,
            left_cells,
            mapping.strength,
            toward="upper",
        )
        right = _clustered_interval(
            focus,
            upper,
            right_cells,
            mapping.strength,
            toward="lower",
        )
        faces = left[:-1] + right

    if not all(math.isfinite(value) for value in faces):
        raise ValueError("analytic axis mapping produced non-finite faces")
    if not all(right > left for left, right in zip(faces, faces[1:])):
        raise ValueError(
            "analytic axis mapping collapsed neighboring faces; reduce strength"
        )
    return faces


@dataclass(frozen=True, slots=True)
class RectilinearGrid:
    """Tensor-product finite-volume grid described by physical cell faces.

    The grid is an array-independent domain value.  It deliberately stores
    physical coordinates rather than the analytic mapping that produced them,
    so solvers consume one stable representation regardless of the meshing
    application used upstream.
    """

    x_faces: tuple[float, ...]
    y_faces: tuple[float, ...]
    z_faces: tuple[float, ...]

    def __post_init__(self) -> None:
        for faces, name in (
            (self.x_faces, "x"),
            (self.y_faces, "y"),
            (self.z_faces, "z"),
        ):
            if len(faces) < 2:
                raise ValueError(f"{name} requires at least one cell")
            if not all(math.isfinite(value) for value in faces):
                raise ValueError(f"{name} faces must be finite")
            if not all(right > left for left, right in zip(faces, faces[1:])):
                raise ValueError(f"{name} faces must be strictly increasing")

    @classmethod
    def uniform(
        cls,
        nx: int,
        ny: int,
        nz: int,
        *,
        lx: float = 1.0,
        ly: float = 1.0,
        lz: float = 1.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> RectilinearGrid:
        if min(nx, ny, nz) <= 0:
            raise ValueError("cell counts must be positive")
        if min(lx, ly, lz) <= 0.0:
            raise ValueError("domain lengths must be positive")

        return cls(
            _uniform_faces(origin[0], lx, nx),
            _uniform_faces(origin[1], ly, ny),
            _uniform_faces(origin[2], lz, nz),
        )

    @classmethod
    def analytic(
        cls,
        nx: int,
        ny: int,
        nz: int,
        *,
        lx: float = 1.0,
        ly: float = 1.0,
        lz: float = 1.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        x: AnalyticAxisMapping = AnalyticAxisMapping(),
        y: AnalyticAxisMapping = AnalyticAxisMapping(),
        z: AnalyticAxisMapping = AnalyticAxisMapping(),
    ) -> RectilinearGrid:
        """Build a tensor grid from independent normalized axis mappings."""

        return cls(
            analytic_axis_faces(nx, lx, x, start=origin[0]),
            analytic_axis_faces(ny, ly, y, start=origin[1]),
            analytic_axis_faces(nz, lz, z, start=origin[2]),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            len(self.z_faces) - 1,
            len(self.y_faces) - 1,
            len(self.x_faces) - 1,
        )

    @property
    def cell_count(self) -> int:
        nz, ny, nx = self.shape
        return nx * ny * nz

    @staticmethod
    def _widths(faces: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(right - left for left, right in zip(faces, faces[1:]))

    @staticmethod
    def _centers(faces: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(0.5 * (left + right) for left, right in zip(faces, faces[1:]))

    @property
    def x_widths(self) -> tuple[float, ...]:
        return self._widths(self.x_faces)

    @property
    def y_widths(self) -> tuple[float, ...]:
        return self._widths(self.y_faces)

    @property
    def z_widths(self) -> tuple[float, ...]:
        return self._widths(self.z_faces)

    @property
    def x_centers(self) -> tuple[float, ...]:
        return self._centers(self.x_faces)

    @property
    def y_centers(self) -> tuple[float, ...]:
        return self._centers(self.y_faces)

    @property
    def z_centers(self) -> tuple[float, ...]:
        return self._centers(self.z_faces)

    @staticmethod
    def _is_uniform(widths: tuple[float, ...]) -> bool:
        reference = sum(widths) / len(widths)
        tolerance = 1.0e-12 * max(1.0, abs(reference))
        return all(
            math.isclose(
                width,
                reference,
                rel_tol=1.0e-12,
                abs_tol=tolerance,
            )
            for width in widths
        )

    @property
    def uniform_axes(self) -> tuple[bool, bool, bool]:
        """Return uniformity in physical ``(x, y, z)`` order."""

        return (
            self._is_uniform(self.x_widths),
            self._is_uniform(self.y_widths),
            self._is_uniform(self.z_widths),
        )
