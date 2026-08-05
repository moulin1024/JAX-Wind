"""Shared geometric hierarchy for pressure multigrid and LES closures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .grid import RectilinearGrid


Array = jax.Array
CoarseningMode = Literal["auto", "full", "z_semi"]


def _coarsen_faces(faces: tuple[float, ...], factor: int) -> tuple[float, ...]:
    if factor == 1:
        return faces
    if factor != 2 or (len(faces) - 1) % 2:
        raise ValueError("multigrid supports factor-two cell aggregation")
    return faces[::2]


def _coarsening_factor(cell_count: int, minimum: int) -> int:
    return 2 if cell_count > minimum and cell_count % 2 == 0 else 1


def z_anisotropy_ratio(grid: RectilinearGrid) -> float:
    """Return horizontal-to-smallest-vertical grid-spacing ratio."""
    dx = np.diff(np.asarray(grid.x_faces))
    dy = np.diff(np.asarray(grid.y_faces))
    dz = np.diff(np.asarray(grid.z_faces))
    horizontal = min(float(np.median(dx)), float(np.median(dy)))
    return horizontal / float(np.min(dz))


def _cell_volume(grid: RectilinearGrid, dtype: jnp.dtype) -> Array:
    dx = jnp.diff(jnp.asarray(grid.x_faces, dtype=dtype))
    dy = jnp.diff(jnp.asarray(grid.y_faces, dtype=dtype))
    dz = jnp.diff(jnp.asarray(grid.z_faces, dtype=dtype))
    return dz[:, None, None] * dy[None, :, None] * dx[None, None, :]


def _block_sum(values: Array, axis: int, factor: int) -> Array:
    if factor == 1:
        return values
    shape = list(values.shape)
    if shape[axis] % factor:
        raise ValueError("field shape is incompatible with hierarchy transfer")
    shape[axis : axis + 1] = (shape[axis] // factor, factor)
    return jnp.reshape(values, shape).sum(axis=axis + 1)


@dataclass(frozen=True, slots=True)
class MultigridHierarchy:
    """Nested grids and factor-two aggregation shared across subsystems.

    ``restrict`` is a conservative finite-volume block average.  Pressure
    uses its own volume-adjoint Galerkin restriction, but both pressure and
    LES consume this exact same set of grids and coarsening decisions.
    """

    grids: tuple[RectilinearGrid, ...]
    coarsening_factors: tuple[tuple[int, int, int], ...]

    @classmethod
    def build(
        cls,
        grid: RectilinearGrid,
        *,
        max_levels: int,
        min_coarse_cells: int,
        coarsening: CoarseningMode,
        anisotropy_threshold: float,
    ) -> MultigridHierarchy:
        grids = [grid]
        factors: list[tuple[int, int, int]] = []
        for _ in range(max_levels - 1):
            fine = grids[-1]
            nz, ny, nx = fine.shape
            fx = _coarsening_factor(nx, min_coarse_cells)
            fy = _coarsening_factor(ny, min_coarse_cells)
            hold_z = coarsening == "z_semi" or (
                coarsening == "auto"
                and z_anisotropy_ratio(fine) >= anisotropy_threshold
            )
            fz = 1 if hold_z else _coarsening_factor(nz, min_coarse_cells)
            factor = (fz, fy, fx)
            if factor == (1, 1, 1):
                break
            factors.append(factor)
            grids.append(
                RectilinearGrid(
                    _coarsen_faces(fine.x_faces, fx),
                    _coarsen_faces(fine.y_faces, fy),
                    _coarsen_faces(fine.z_faces, fz),
                )
            )
        return cls(tuple(grids), tuple(factors))

    @property
    def level_shapes(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(grid.shape for grid in self.grids)

    def restrict(self, field: Array, *, fine_level: int = 0) -> Array:
        """Conservatively average a scalar or trailing-component FV field."""
        if not 0 <= fine_level < len(self.coarsening_factors):
            raise ValueError("restriction level is outside the hierarchy")
        fine = self.grids[fine_level]
        if tuple(field.shape[:3]) != fine.shape:
            raise ValueError(
                f"expected leading field shape {fine.shape}, got {field.shape[:3]}"
            )
        trailing_rank = field.ndim - 3
        fine_volume = _cell_volume(fine, field.dtype).reshape(
            (*fine.shape, *((1,) * trailing_rank))
        )
        weighted = field * fine_volume
        for axis, factor in enumerate(self.coarsening_factors[fine_level]):
            weighted = _block_sum(weighted, axis, factor)
        coarse = self.grids[fine_level + 1]
        coarse_volume = _cell_volume(coarse, field.dtype).reshape(
            (*coarse.shape, *((1,) * trailing_rank))
        )
        return weighted / coarse_volume


__all__ = ["CoarseningMode", "MultigridHierarchy", "z_anisotropy_ratio"]
