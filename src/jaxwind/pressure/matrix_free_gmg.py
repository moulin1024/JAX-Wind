"""Standalone matrix-free finite-volume Poisson solver.

The solver uses cell-centred pressure unknowns in canonical ``(z, y, x)``
layout.  The positive operator is ``-div(grad(p))``.  A geometric multigrid
V-cycle with volume-adjoint transfers is used as a symmetric positive
preconditioner for PCG, with flexible GMRES retained as an alternative.

This module deliberately has no dependency on the time integrators or the
distributed pressure facade.  It is the single-process reference backend for
developing non-periodic and rectilinear-grid pressure projection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .device_gmres import build_device_gmres_solver
from .device_pcg import build_device_pcg_solver
from .fgmres import FGMRESConfig, FGMRESResult, fgmres
from .pcg import PCGConfig, PCGResult, pcg


BoundaryKind = Literal["periodic", "dirichlet", "neumann"]
Array = jax.Array


@dataclass(frozen=True, slots=True)
class BoundaryCondition:
    """A scalar condition on one outward domain face.

    For ``neumann``, ``value`` is the outward normal derivative
    ``grad(p) dot n``.  Periodic values are ignored.
    """

    kind: BoundaryKind
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"periodic", "dirichlet", "neumann"}:
            raise ValueError(f"unsupported boundary kind: {self.kind!r}")
        if not math.isfinite(self.value):
            raise ValueError("boundary value must be finite")


@dataclass(frozen=True, slots=True)
class PoissonBoundaryConditions:
    """Boundary conditions ordered by physical coordinate axis."""

    x_lower: BoundaryCondition
    x_upper: BoundaryCondition
    y_lower: BoundaryCondition
    y_upper: BoundaryCondition
    z_lower: BoundaryCondition
    z_upper: BoundaryCondition

    def __post_init__(self) -> None:
        for lower, upper, name in self.axis_pairs():
            periodic_count = int(lower.kind == "periodic") + int(
                upper.kind == "periodic"
            )
            if periodic_count == 1:
                raise ValueError(
                    f"{name} boundaries must either both be periodic or neither"
                )

    @classmethod
    def periodic(cls) -> PoissonBoundaryConditions:
        condition = BoundaryCondition("periodic")
        return cls(*(condition for _ in range(6)))

    @classmethod
    def homogeneous_neumann(cls) -> PoissonBoundaryConditions:
        condition = BoundaryCondition("neumann")
        return cls(*(condition for _ in range(6)))

    @classmethod
    def homogeneous_dirichlet(cls) -> PoissonBoundaryConditions:
        condition = BoundaryCondition("dirichlet")
        return cls(*(condition for _ in range(6)))

    def axis_pairs(
        self,
    ) -> tuple[
        tuple[BoundaryCondition, BoundaryCondition, str],
        tuple[BoundaryCondition, BoundaryCondition, str],
        tuple[BoundaryCondition, BoundaryCondition, str],
    ]:
        return (
            (self.x_lower, self.x_upper, "x"),
            (self.y_lower, self.y_upper, "y"),
            (self.z_lower, self.z_upper, "z"),
        )

    @property
    def has_constant_nullspace(self) -> bool:
        return not any(
            condition.kind == "dirichlet"
            for pair in self.axis_pairs()
            for condition in pair[:2]
        )


@dataclass(frozen=True, slots=True)
class RectilinearGrid:
    """Tensor-product finite-volume grid described by physical cell faces."""

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
            return tuple(
                start + length * index / count for index in range(count + 1)
            )

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


@dataclass(frozen=True, slots=True)
class _Level:
    grid: RectilinearGrid
    widths: tuple[Array, Array, Array]
    centers: tuple[Array, Array, Array]
    volume: Array
    diagonal: Array


def _axis_data(faces: tuple[float, ...], dtype: jnp.dtype) -> tuple[Array, Array]:
    face_array = jnp.asarray(faces, dtype=dtype)
    widths = face_array[1:] - face_array[:-1]
    centers = 0.5 * (face_array[1:] + face_array[:-1])
    return widths, centers


def _apply_axis(
    field: Array,
    *,
    axis: int,
    widths: Array,
    centers: Array,
    lower: BoundaryCondition,
    upper: BoundaryCondition,
) -> Array:
    values = jnp.moveaxis(field, axis, -1)
    result = jnp.zeros_like(values)
    count = values.shape[-1]

    if count > 1:
        distance = centers[1:] - centers[:-1]
        difference = values[..., :-1] - values[..., 1:]
        result = result.at[..., :-1].add(
            difference / (widths[:-1] * distance)
        )
        result = result.at[..., 1:].add(
            -difference / (widths[1:] * distance)
        )

    if lower.kind == "periodic":
        if count > 1:
            distance = 0.5 * (widths[-1] + widths[0])
            difference = values[..., -1] - values[..., 0]
            result = result.at[..., -1].add(
                difference / (widths[-1] * distance)
            )
            result = result.at[..., 0].add(
                -difference / (widths[0] * distance)
            )
    else:
        if lower.kind == "dirichlet":
            result = result.at[..., 0].add(
                values[..., 0] * 2.0 / (widths[0] * widths[0])
            )
        if upper.kind == "dirichlet":
            result = result.at[..., -1].add(
                values[..., -1] * 2.0 / (widths[-1] * widths[-1])
            )

    return jnp.moveaxis(result, -1, axis)


def _axis_diagonal(
    widths: Array,
    centers: Array,
    lower: BoundaryCondition,
    upper: BoundaryCondition,
) -> Array:
    diagonal = jnp.zeros_like(widths)
    count = widths.shape[0]
    if count > 1:
        distance = centers[1:] - centers[:-1]
        diagonal = diagonal.at[:-1].add(1.0 / (widths[:-1] * distance))
        diagonal = diagonal.at[1:].add(1.0 / (widths[1:] * distance))
    if lower.kind == "periodic":
        if count > 1:
            distance = 0.5 * (widths[-1] + widths[0])
            diagonal = diagonal.at[-1].add(1.0 / (widths[-1] * distance))
            diagonal = diagonal.at[0].add(1.0 / (widths[0] * distance))
    else:
        if lower.kind == "dirichlet":
            diagonal = diagonal.at[0].add(2.0 / (widths[0] * widths[0]))
        if upper.kind == "dirichlet":
            diagonal = diagonal.at[-1].add(2.0 / (widths[-1] * widths[-1]))
    return diagonal


def _make_level(
    grid: RectilinearGrid,
    boundaries: PoissonBoundaryConditions,
    dtype: jnp.dtype,
) -> _Level:
    wx, cx = _axis_data(grid.x_faces, dtype)
    wy, cy = _axis_data(grid.y_faces, dtype)
    wz, cz = _axis_data(grid.z_faces, dtype)
    dx = _axis_diagonal(wx, cx, boundaries.x_lower, boundaries.x_upper)
    dy = _axis_diagonal(wy, cy, boundaries.y_lower, boundaries.y_upper)
    dz = _axis_diagonal(wz, cz, boundaries.z_lower, boundaries.z_upper)
    volume = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    diagonal = (
        dz[:, None, None] + dy[None, :, None] + dx[None, None, :]
    )
    return _Level(grid, (wx, wy, wz), (cx, cy, cz), volume, diagonal)


class MatrixFreePoissonOperator:
    """Matrix-free cell-centred finite-volume ``-div(grad)`` operator."""

    def __init__(
        self,
        grid: RectilinearGrid,
        boundaries: PoissonBoundaryConditions,
        *,
        dtype: jnp.dtype = jnp.float64,
    ) -> None:
        self.grid = grid
        self.boundaries = boundaries
        self.dtype = jnp.dtype(dtype)
        self._level = _make_level(grid, boundaries, self.dtype)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape

    @property
    def volume(self) -> Array:
        return self._level.volume

    @property
    def diagonal(self) -> Array:
        return self._level.diagonal

    @property
    def has_constant_nullspace(self) -> bool:
        return self.boundaries.has_constant_nullspace

    def _check_shape(self, field: Array) -> None:
        if tuple(field.shape) != self.shape:
            raise ValueError(
                f"expected a z-y-x field with shape {self.shape}, "
                f"got {tuple(field.shape)}"
            )

    def apply(self, pressure: Array) -> Array:
        """Apply the homogeneous-boundary linear operator."""
        self._check_shape(pressure)
        wx, wy, wz = self._level.widths
        cx, cy, cz = self._level.centers
        result = _apply_axis(
            pressure,
            axis=-1,
            widths=wx,
            centers=cx,
            lower=self.boundaries.x_lower,
            upper=self.boundaries.x_upper,
        )
        result = result + _apply_axis(
            pressure,
            axis=-2,
            widths=wy,
            centers=cy,
            lower=self.boundaries.y_lower,
            upper=self.boundaries.y_upper,
        )
        return result + _apply_axis(
            pressure,
            axis=-3,
            widths=wz,
            centers=cz,
            lower=self.boundaries.z_lower,
            upper=self.boundaries.z_upper,
        )

    def boundary_rhs(self) -> Array:
        """Return the source induced by non-homogeneous boundary data."""
        result = jnp.zeros(self.shape, dtype=self.dtype)
        wx, wy, wz = self._level.widths
        axis_data = (
            (-1, wx, self.boundaries.x_lower, self.boundaries.x_upper),
            (-2, wy, self.boundaries.y_lower, self.boundaries.y_upper),
            (-3, wz, self.boundaries.z_lower, self.boundaries.z_upper),
        )
        for axis, widths, lower, upper in axis_data:
            lower_slice = [slice(None)] * 3
            upper_slice = [slice(None)] * 3
            lower_slice[axis] = 0
            upper_slice[axis] = -1
            if lower.kind == "dirichlet":
                result = result.at[tuple(lower_slice)].add(
                    2.0 * lower.value / (widths[0] * widths[0])
                )
            elif lower.kind == "neumann":
                result = result.at[tuple(lower_slice)].add(
                    lower.value / widths[0]
                )
            if upper.kind == "dirichlet":
                result = result.at[tuple(upper_slice)].add(
                    2.0 * upper.value / (widths[-1] * widths[-1])
                )
            elif upper.kind == "neumann":
                result = result.at[tuple(upper_slice)].add(
                    upper.value / widths[-1]
                )
        return result

    def volume_mean(self, field: Array) -> Array:
        self._check_shape(field)
        return jnp.sum(field * self.volume) / jnp.sum(self.volume)

    def project_nullspace(self, field: Array) -> Array:
        if not self.has_constant_nullspace:
            return field
        return field - self.volume_mean(field)

    def inner(self, left: Array, right: Array) -> Array:
        self._check_shape(left)
        self._check_shape(right)
        return jnp.sum(self.volume * left * right)

    def norm(self, field: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(field, field), 0.0))

    def prepare_rhs(self, physical_rhs: Array) -> tuple[Array, Array]:
        """Add boundary forcing and enforce pure-Neumann compatibility."""
        self._check_shape(physical_rhs)
        effective = jnp.asarray(physical_rhs, dtype=self.dtype) + self.boundary_rhs()
        shift = jnp.asarray(0.0, dtype=self.dtype)
        if self.has_constant_nullspace:
            shift = self.volume_mean(effective)
            effective = effective - shift
        return effective, shift


@dataclass(frozen=True, slots=True)
class GMGConfig:
    """Geometric multigrid V-cycle configuration."""

    max_levels: int = 20
    min_coarse_cells: int = 2
    pre_smooth: int = 2
    post_smooth: int = 2
    coarse_smooth: int = 40
    jacobi_omega: float = 0.72
    line_omega: float = 0.9
    smoother: Literal["auto", "jacobi", "z_line"] = "auto"
    coarsening: Literal["auto", "full", "z_semi"] = "auto"
    anisotropy_threshold: float = 4.0

    def __post_init__(self) -> None:
        if self.max_levels <= 0:
            raise ValueError("max_levels must be positive")
        if self.min_coarse_cells <= 0:
            raise ValueError("min_coarse_cells must be positive")
        if min(self.pre_smooth, self.post_smooth, self.coarse_smooth) < 0:
            raise ValueError("smoothing counts must be nonnegative")
        if self.pre_smooth != self.post_smooth:
            raise ValueError(
                "symmetric GMG requires equal pre_smooth and post_smooth"
            )
        if not 0.0 < self.jacobi_omega < 1.0:
            raise ValueError("jacobi_omega must lie between zero and one")
        if not 0.0 < self.line_omega <= 1.0:
            raise ValueError("line_omega must lie in (0, 1]")
        if self.smoother not in {"auto", "jacobi", "z_line"}:
            raise ValueError(f"unsupported smoother: {self.smoother!r}")
        if self.coarsening not in {"auto", "full", "z_semi"}:
            raise ValueError(f"unsupported coarsening: {self.coarsening!r}")
        if self.anisotropy_threshold <= 1.0:
            raise ValueError("anisotropy_threshold must exceed one")


@dataclass(frozen=True, slots=True)
class _Interpolation:
    lower_index: Array
    upper_index: Array
    lower_weight: Array
    upper_weight: Array


@dataclass(frozen=True, slots=True)
class _Transfer:
    fine_volume: Array
    coarse_volume: Array
    factors_zyx: tuple[int, int, int]
    x: _Interpolation
    y: _Interpolation
    z: _Interpolation


def _coarsen_faces(faces: tuple[float, ...], factor: int) -> tuple[float, ...]:
    if factor == 1:
        return faces
    if factor != 2 or (len(faces) - 1) % 2:
        raise ValueError("GMG currently supports factor-two cell aggregation")
    return faces[::2]


def _coarsening_factor(cell_count: int, minimum: int) -> int:
    return 2 if cell_count > minimum and cell_count % 2 == 0 else 1


def _build_interpolation(
    fine_faces: tuple[float, ...],
    coarse_faces: tuple[float, ...],
    lower: BoundaryCondition,
    upper: BoundaryCondition,
    dtype: jnp.dtype,
) -> _Interpolation:
    fine_faces_np = np.asarray(fine_faces, dtype=float)
    coarse_faces_np = np.asarray(coarse_faces, dtype=float)
    fine = 0.5 * (fine_faces_np[1:] + fine_faces_np[:-1])
    coarse = 0.5 * (coarse_faces_np[1:] + coarse_faces_np[:-1])
    count = coarse.size
    lower_index = np.empty(fine.size, dtype=np.int32)
    upper_index = np.empty(fine.size, dtype=np.int32)
    lower_weight = np.empty(fine.size, dtype=float)
    upper_weight = np.empty(fine.size, dtype=float)
    periodic = lower.kind == "periodic"
    domain_length = coarse_faces_np[-1] - coarse_faces_np[0]

    for index, coordinate in enumerate(fine):
        if count == 1:
            lower_index[index] = upper_index[index] = 0
            if lower.kind == "dirichlet" and coordinate < coarse[0]:
                weight = (coordinate - coarse_faces_np[0]) / (
                    coarse[0] - coarse_faces_np[0]
                )
            elif upper.kind == "dirichlet" and coordinate > coarse[0]:
                weight = (coarse_faces_np[-1] - coordinate) / (
                    coarse_faces_np[-1] - coarse[0]
                )
            else:
                weight = 1.0
            lower_weight[index] = weight
            upper_weight[index] = 0.0
            continue

        high = int(np.searchsorted(coarse, coordinate, side="right"))
        low = high - 1
        if 0 <= low and high < count:
            left_coordinate = coarse[low]
            right_coordinate = coarse[high]
            fraction = (coordinate - left_coordinate) / (
                right_coordinate - left_coordinate
            )
            lower_index[index], upper_index[index] = low, high
            lower_weight[index], upper_weight[index] = 1.0 - fraction, fraction
        elif coordinate < coarse[0]:
            if periodic:
                left_coordinate = coarse[-1] - domain_length
                right_coordinate = coarse[0]
                fraction = (coordinate - left_coordinate) / (
                    right_coordinate - left_coordinate
                )
                lower_index[index], upper_index[index] = count - 1, 0
                lower_weight[index], upper_weight[index] = (
                    1.0 - fraction,
                    fraction,
                )
            elif lower.kind == "dirichlet":
                lower_index[index] = upper_index[index] = 0
                lower_weight[index] = (coordinate - coarse_faces_np[0]) / (
                    coarse[0] - coarse_faces_np[0]
                )
                upper_weight[index] = 0.0
            else:
                lower_index[index] = upper_index[index] = 0
                lower_weight[index], upper_weight[index] = 1.0, 0.0
        else:
            if periodic:
                left_coordinate = coarse[-1]
                right_coordinate = coarse[0] + domain_length
                fraction = (coordinate - left_coordinate) / (
                    right_coordinate - left_coordinate
                )
                lower_index[index], upper_index[index] = count - 1, 0
                lower_weight[index], upper_weight[index] = (
                    1.0 - fraction,
                    fraction,
                )
            elif upper.kind == "dirichlet":
                lower_index[index] = upper_index[index] = count - 1
                lower_weight[index] = (coarse_faces_np[-1] - coordinate) / (
                    coarse_faces_np[-1] - coarse[-1]
                )
                upper_weight[index] = 0.0
            else:
                lower_index[index] = upper_index[index] = count - 1
                lower_weight[index], upper_weight[index] = 1.0, 0.0

    return _Interpolation(
        jnp.asarray(lower_index),
        jnp.asarray(upper_index),
        jnp.asarray(lower_weight, dtype=dtype),
        jnp.asarray(upper_weight, dtype=dtype),
    )


def _interpolate_axis(field: Array, interpolation: _Interpolation, axis: int) -> Array:
    lower = jnp.take(field, interpolation.lower_index, axis=axis)
    upper = jnp.take(field, interpolation.upper_index, axis=axis)
    shape = [1] * field.ndim
    shape[axis % field.ndim] = interpolation.lower_index.shape[0]
    lower_weight = jnp.reshape(interpolation.lower_weight, shape)
    upper_weight = jnp.reshape(interpolation.upper_weight, shape)
    return lower * lower_weight + upper * upper_weight


def _transpose_interpolate_axis(
    field: Array,
    interpolation: _Interpolation,
    axis: int,
    coarse_count: int,
) -> Array:
    """Apply the Euclidean transpose of one interpolation axis."""
    values = jnp.moveaxis(field, axis, -1)
    result = jnp.zeros(
        (*values.shape[:-1], coarse_count),
        dtype=values.dtype,
    )
    result = result.at[..., interpolation.lower_index].add(
        values * interpolation.lower_weight
    )
    result = result.at[..., interpolation.upper_index].add(
        values * interpolation.upper_weight
    )
    return jnp.moveaxis(result, -1, axis)


def _make_transfer(
    fine: _Level,
    coarse: _Level,
    boundaries: PoissonBoundaryConditions,
    dtype: jnp.dtype,
) -> _Transfer:
    fine_shape = fine.grid.shape
    coarse_shape = coarse.grid.shape
    factors = tuple(
        fine_count // coarse_count
        for fine_count, coarse_count in zip(fine_shape, coarse_shape)
    )
    return _Transfer(
        fine.volume,
        coarse.volume,
        factors,
        _build_interpolation(
            fine.grid.x_faces,
            coarse.grid.x_faces,
            boundaries.x_lower,
            boundaries.x_upper,
            dtype,
        ),
        _build_interpolation(
            fine.grid.y_faces,
            coarse.grid.y_faces,
            boundaries.y_lower,
            boundaries.y_upper,
            dtype,
        ),
        _build_interpolation(
            fine.grid.z_faces,
            coarse.grid.z_faces,
            boundaries.z_lower,
            boundaries.z_upper,
            dtype,
        ),
    )


def _restrict(field: Array, transfer: _Transfer) -> Array:
    """Apply ``V_c^-1 P^T V_f``, the volume-weighted adjoint of prolongation."""
    coarse_shape = transfer.coarse_volume.shape
    weighted = field * transfer.fine_volume
    result = _transpose_interpolate_axis(
        weighted,
        transfer.z,
        -3,
        coarse_shape[-3],
    )
    result = _transpose_interpolate_axis(
        result,
        transfer.y,
        -2,
        coarse_shape[-2],
    )
    result = _transpose_interpolate_axis(
        result,
        transfer.x,
        -1,
        coarse_shape[-1],
    )
    return result / transfer.coarse_volume


def _prolong(field: Array, transfer: _Transfer) -> Array:
    result = _interpolate_axis(field, transfer.x, -1)
    result = _interpolate_axis(result, transfer.y, -2)
    return _interpolate_axis(result, transfer.z, -3)


def _z_anisotropy_ratio(grid: RectilinearGrid) -> float:
    dx = np.diff(np.asarray(grid.x_faces))
    dy = np.diff(np.asarray(grid.y_faces))
    dz = np.diff(np.asarray(grid.z_faces))
    horizontal = min(float(np.median(dx)), float(np.median(dy)))
    return horizontal / float(np.min(dz))


def _solve_z_lines(
    operator: MatrixFreePoissonOperator,
    rhs: Array,
) -> Array:
    """Solve the block-Jacobi tridiagonal system along every z column."""
    wx, wy, wz = operator._level.widths
    cx, cy, cz = operator._level.centers
    dx = _axis_diagonal(
        wx,
        cx,
        operator.boundaries.x_lower,
        operator.boundaries.x_upper,
    )
    dy = _axis_diagonal(
        wy,
        cy,
        operator.boundaries.y_lower,
        operator.boundaries.y_upper,
    )
    dz = _axis_diagonal(
        wz,
        cz,
        operator.boundaries.z_lower,
        operator.boundaries.z_upper,
    )
    diagonal = (
        dz[:, None, None] + dy[None, :, None] + dx[None, None, :]
    )
    count = rhs.shape[0]
    if count == 1:
        safe_diagonal = jnp.where(diagonal != 0.0, diagonal, 1.0)
        return rhs / safe_diagonal

    distance = cz[1:] - cz[:-1]
    lower = jnp.zeros_like(wz).at[1:].set(
        -1.0 / (wz[1:] * distance)
    )
    upper = jnp.zeros_like(wz).at[:-1].set(
        -1.0 / (wz[:-1] * distance)
    )
    first_denominator = jnp.where(diagonal[0] != 0.0, diagonal[0], 1.0)
    first_upper = upper[0] / first_denominator
    first_rhs = rhs[0] / first_denominator

    def forward(
        carry: tuple[Array, Array],
        values: tuple[Array, Array, Array, Array],
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        previous_upper, previous_rhs = carry
        lower_value, diagonal_value, upper_value, rhs_value = values
        denominator = diagonal_value - lower_value * previous_upper
        denominator = jnp.where(denominator != 0.0, denominator, 1.0)
        reduced_upper = upper_value / denominator
        reduced_rhs = (
            rhs_value - lower_value * previous_rhs
        ) / denominator
        return (
            (reduced_upper, reduced_rhs),
            (reduced_upper, reduced_rhs),
        )

    _, (upper_tail, rhs_tail) = jax.lax.scan(
        forward,
        (first_upper, first_rhs),
        (lower[1:], diagonal[1:], upper[1:], rhs[1:]),
    )
    reduced_upper = jnp.concatenate(
        (first_upper[None, ...], upper_tail),
        axis=0,
    )
    reduced_rhs = jnp.concatenate(
        (first_rhs[None, ...], rhs_tail),
        axis=0,
    )

    def backward(next_value: Array, values: tuple[Array, Array]) -> tuple[Array, Array]:
        rhs_value, upper_value = values
        value = rhs_value - upper_value * next_value
        return value, value

    _, prefix_reverse = jax.lax.scan(
        backward,
        reduced_rhs[-1],
        (reduced_rhs[:-1][::-1], reduced_upper[:-1][::-1]),
    )
    return jnp.concatenate(
        (prefix_reverse[::-1], reduced_rhs[-1:]),
        axis=0,
    )


class MatrixFreeGMG:
    """Matrix-free geometric multigrid V-cycle preconditioner."""

    def __init__(
        self,
        operator: MatrixFreePoissonOperator,
        config: GMGConfig = GMGConfig(),
    ) -> None:
        self.boundaries = operator.boundaries
        self.dtype = operator.dtype
        self.config = config
        operators = [operator]
        transfers: list[_Transfer] = []
        level_smoothers: list[str] = []
        coarsening_factors: list[tuple[int, int, int]] = []

        if (
            config.smoother == "z_line"
            and operator.boundaries.z_lower.kind == "periodic"
        ):
            raise ValueError("z-line smoothing does not support periodic z")

        for _ in range(config.max_levels - 1):
            fine = operators[-1]
            anisotropic = (
                _z_anisotropy_ratio(fine.grid)
                >= config.anisotropy_threshold
            )
            use_z_line = config.smoother == "z_line" or (
                config.smoother == "auto"
                and anisotropic
                and fine.boundaries.z_lower.kind != "periodic"
            )
            nz, ny, nx = fine.shape
            fx = _coarsening_factor(nx, config.min_coarse_cells)
            fy = _coarsening_factor(ny, config.min_coarse_cells)
            hold_z = config.coarsening == "z_semi" or (
                config.coarsening == "auto" and anisotropic
            )
            fz = (
                1
                if hold_z
                else _coarsening_factor(nz, config.min_coarse_cells)
            )
            if (fz, fy, fx) == (1, 1, 1):
                break
            level_smoothers.append("z_line" if use_z_line else "jacobi")
            coarsening_factors.append((fz, fy, fx))
            coarse_grid = RectilinearGrid(
                _coarsen_faces(fine.grid.x_faces, fx),
                _coarsen_faces(fine.grid.y_faces, fy),
                _coarsen_faces(fine.grid.z_faces, fz),
            )
            coarse = MatrixFreePoissonOperator(
                coarse_grid,
                self.boundaries,
                dtype=self.dtype,
            )
            transfers.append(
                _make_transfer(
                    fine._level,
                    coarse._level,
                    self.boundaries,
                    self.dtype,
                )
            )
            operators.append(coarse)

        coarse = operators[-1]
        coarse_anisotropic = (
            _z_anisotropy_ratio(coarse.grid)
            >= config.anisotropy_threshold
        )
        coarse_z_line = config.smoother == "z_line" or (
            config.smoother == "auto"
            and coarse_anisotropic
            and coarse.boundaries.z_lower.kind != "periodic"
        )
        level_smoothers.append("z_line" if coarse_z_line else "jacobi")
        self.operators = tuple(operators)
        self.transfers = tuple(transfers)
        self.level_smoothers = tuple(level_smoothers)
        self.coarsening_factors = tuple(coarsening_factors)

    @property
    def level_shapes(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(operator.shape for operator in self.operators)

    def _smooth(self, level: int, solution: Array, rhs: Array, count: int) -> Array:
        operator = self.operators[level]
        for _ in range(count):
            residual = rhs - operator.apply(solution)
            if self.level_smoothers[level] == "z_line":
                correction = _solve_z_lines(operator, residual)
                solution = solution + self.config.line_omega * correction
            else:
                solution = solution + (
                    self.config.jacobi_omega * residual / operator.diagonal
                )
            solution = operator.project_nullspace(solution)
        return solution

    def _cycle(self, level: int, solution: Array, rhs: Array) -> Array:
        operator = self.operators[level]
        rhs = operator.project_nullspace(rhs)
        if level == len(self.operators) - 1:
            return self._smooth(
                level,
                solution,
                rhs,
                self.config.coarse_smooth,
            )

        solution = self._smooth(
            level,
            solution,
            rhs,
            self.config.pre_smooth,
        )
        residual = operator.project_nullspace(rhs - operator.apply(solution))
        coarse_rhs = _restrict(residual, self.transfers[level])
        coarse_error = jnp.zeros_like(coarse_rhs)
        coarse_error = self._cycle(level + 1, coarse_error, coarse_rhs)
        solution = solution + _prolong(coarse_error, self.transfers[level])
        solution = operator.project_nullspace(solution)
        return self._smooth(
            level,
            solution,
            rhs,
            self.config.post_smooth,
        )

    def apply(self, rhs: Array) -> Array:
        """Apply one zero-initialized V-cycle."""
        if tuple(rhs.shape) != self.operators[0].shape:
            raise ValueError(
                f"expected preconditioner RHS shape {self.operators[0].shape}, "
                f"got {tuple(rhs.shape)}"
            )
        rhs = self.operators[0].project_nullspace(rhs)
        return self._cycle(0, jnp.zeros_like(rhs), rhs)


class MatrixFreePoissonSolver:
    """Self-contained FV Poisson + symmetric GMG + Krylov solver."""

    def __init__(
        self,
        grid: RectilinearGrid,
        boundaries: PoissonBoundaryConditions,
        *,
        dtype: jnp.dtype = jnp.float64,
        gmg: GMGConfig = GMGConfig(),
        krylov: FGMRESConfig | PCGConfig = FGMRESConfig(),
    ) -> None:
        self.operator = MatrixFreePoissonOperator(
            grid,
            boundaries,
            dtype=dtype,
        )
        self.preconditioner = MatrixFreeGMG(self.operator, gmg)
        self.krylov = krylov
        if krylov.jit_kernels:
            self._apply_kernel = jax.jit(self.operator.apply)
            self._preconditioner_kernel = jax.jit(
                self.preconditioner.apply
            )
            self._python_krylov_config = replace(
                krylov,
                jit_kernels=False,
            )
        else:
            self._apply_kernel = self.operator.apply
            self._preconditioner_kernel = self.preconditioner.apply
            self._python_krylov_config = krylov
        self._device_solve_kernel = (
            self._build_device_solver()
            if self.krylov.execution == "jax"
            else None
        )

    def _build_device_solver(self):
        common = dict(
            apply=self.operator.apply,
            precondition=self.preconditioner.apply,
            volume=self.operator.volume,
            project=self.operator.project_nullspace,
            max_iterations=self.krylov.max_iterations,
            relative_tolerance=self.krylov.relative_tolerance,
            absolute_tolerance=self.krylov.absolute_tolerance,
        )
        if isinstance(self.krylov, PCGConfig):
            return build_device_pcg_solver(**common)
        return build_device_gmres_solver(
            **common,
            restart=self.krylov.restart,
            solve_method=self.krylov.jax_solve_method,
        )

    def solve(
        self,
        physical_rhs: Array,
        *,
        initial: Array | None = None,
    ) -> FGMRESResult | PCGResult:
        effective_rhs, compatibility_shift = self.operator.prepare_rhs(
            physical_rhs
        )
        if self.krylov.execution == "jax":
            solution = self._device_solution(effective_rhs, initial)
            residual = self.operator.project_nullspace(
                effective_rhs - self._apply_kernel(solution)
            )
            residual_norm = float(self.operator.norm(residual))
            rhs_norm = float(self.operator.norm(effective_rhs))
            relative_residual = (
                0.0 if rhs_norm == 0.0 else residual_norm / rhs_norm
            )
            target = max(
                self.krylov.absolute_tolerance,
                self.krylov.relative_tolerance * rhs_norm,
            )
            result_type = (
                PCGResult
                if isinstance(self.krylov, PCGConfig)
                else FGMRESResult
            )
            return result_type(
                solution,
                residual_norm <= target,
                self.krylov.max_iterations,
                residual_norm,
                relative_residual,
                (residual_norm,),
                float(compatibility_shift),
            )
        solve = (
            pcg
            if isinstance(self._python_krylov_config, PCGConfig)
            else fgmres
        )
        result = solve(
            self._apply_kernel,
            effective_rhs,
            preconditioner=self._preconditioner_kernel,
            initial=initial,
            inner=self.operator.inner,
            project=self.operator.project_nullspace,
            config=self._python_krylov_config,
        )
        return replace(result, compatibility_shift=float(compatibility_shift))

    def _device_solution(
        self, effective_rhs: Array, initial: Array | None
    ) -> Array:
        if self._device_solve_kernel is None:
            raise RuntimeError("device GMRES kernel was not initialized")
        starting_value = (
            jnp.zeros_like(effective_rhs)
            if initial is None
            else jnp.asarray(initial, dtype=self.operator.dtype)
        )
        return self._device_solve_kernel(effective_rhs, starting_value)

    def solve_array(
        self, physical_rhs: Array, *, initial: Array | None = None
    ) -> Array:
        """Return only the solution, keeping device GMRES free of host syncs."""
        if self.krylov.execution != "jax":
            return self.solve(physical_rhs, initial=initial).solution
        effective_rhs, _ = self.operator.prepare_rhs(physical_rhs)
        return self._device_solution(effective_rhs, initial)

__all__ = [
    "BoundaryCondition",
    "FGMRESConfig",
    "FGMRESResult",
    "GMGConfig",
    "MatrixFreeGMG",
    "MatrixFreePoissonOperator",
    "MatrixFreePoissonSolver",
    "PCGConfig",
    "PCGResult",
    "PoissonBoundaryConditions",
    "RectilinearGrid",
    "fgmres",
    "pcg",
]
