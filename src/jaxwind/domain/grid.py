"""Uniform and rectilinear physical-grid metadata in canonical SI units."""

from __future__ import annotations

from dataclasses import dataclass
import math

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

        def faces(start: float, length: float, count: int) -> tuple[float, ...]:
            return tuple(start + length * index / count for index in range(count + 1))

        return cls(
            faces(origin[0], lx, nx),
            faces(origin[1], ly, ny),
            faces(origin[2], lz, nz),
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
