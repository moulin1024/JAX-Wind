"""Pure analytic face-coordinate generation without solver dependencies."""

from __future__ import annotations

import math

from jaxwind.domain import RectilinearGrid

from .model import AxisMeshSpec, AxisMeshStatistics, GeneratedMesh, MeshSpec


def _uniform_faces(lower: float, upper: float, cells: int) -> tuple[float, ...]:
    length = upper - lower
    return tuple(lower + length * index / cells for index in range(cells + 1))


def _clustered_interval(
    lower: float,
    upper: float,
    cells: int,
    strength: float,
    *,
    toward: str,
) -> tuple[float, ...]:
    """Generate an exponential map clustered toward one interval endpoint."""

    denominator = math.expm1(strength)
    length = upper - lower
    values = []
    for index in range(cells + 1):
        coordinate = index / cells
        if toward == "lower":
            mapped = math.expm1(strength * coordinate) / denominator
        elif toward == "upper":
            mapped = 1.0 - math.expm1(strength * (1.0 - coordinate)) / denominator
        else:  # pragma: no cover - private programming error
            raise ValueError(f"unsupported clustering endpoint: {toward!r}")
        values.append(lower + length * mapped)
    values[0] = lower
    values[-1] = upper
    return tuple(values)


def _validate_generated_faces(
    faces: tuple[float, ...],
    specification: AxisMeshSpec,
) -> tuple[float, ...]:
    if not all(math.isfinite(value) for value in faces):
        raise ValueError("analytic mapping produced non-finite faces")
    if not all(right > left for left, right in zip(faces, faces[1:])):
        raise ValueError(
            "analytic mapping produced indistinguishable faces; reduce the "
            f"clustering strength below {specification.strength:g}"
        )
    return faces


def generate_axis_faces(specification: AxisMeshSpec) -> tuple[float, ...]:
    """Generate physical face coordinates for one validated axis."""

    if specification.clustering == "uniform" or specification.strength == 0.0:
        return _uniform_faces(
            specification.lower,
            specification.upper,
            specification.cells,
        )

    assert specification.point is not None
    if specification.clustering == "single":
        toward = "lower" if specification.point == specification.lower else "upper"
        faces = _clustered_interval(
            specification.lower,
            specification.upper,
            specification.cells,
            specification.strength,
            toward=toward,
        )
        return _validate_generated_faces(faces, specification)

    fraction = (specification.point - specification.lower) / (
        specification.upper - specification.lower
    )
    left_cells = min(
        specification.cells - 1,
        max(1, round(specification.cells * fraction)),
    )
    right_cells = specification.cells - left_cells
    left = _clustered_interval(
        specification.lower,
        specification.point,
        left_cells,
        specification.strength,
        toward="upper",
    )
    right = _clustered_interval(
        specification.point,
        specification.upper,
        right_cells,
        specification.strength,
        toward="lower",
    )
    return _validate_generated_faces(left[:-1] + right, specification)


def generate_mesh(specification: MeshSpec) -> GeneratedMesh:
    """Generate one solver-independent physical rectilinear mesh."""

    grid = RectilinearGrid(
        generate_axis_faces(specification.x),
        generate_axis_faces(specification.y),
        generate_axis_faces(specification.z),
    )
    return GeneratedMesh(specification, grid)


def axis_statistics(faces: tuple[float, ...]) -> AxisMeshStatistics:
    """Return spacing extrema and the worst neighboring-cell size jump."""

    spacing = tuple(right - left for left, right in zip(faces, faces[1:]))
    adjacent = tuple(
        max(left / right, right / left) for left, right in zip(spacing, spacing[1:])
    )
    return AxisMeshStatistics(
        min(spacing),
        max(spacing),
        max(adjacent, default=1.0),
    )
