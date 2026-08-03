"""Discrete metrics and variable-spacing operators for one rectilinear axis.

The solver consumes physical face coordinates instead of an analytic mapping,
so every spacing-dependent operator is rebuilt from those coordinates.  An axis
that is uniform to floating-point precision keeps the constant-spacing kernel it
has always used: that preserves existing uniform-grid results bit for bit and
avoids the gather and scatter traffic the variable-spacing stencils need.

Three properties are what the stretched-axis operators are built around.

:meth:`AxisMetric.derivative` reproduces a constant exactly.  Every stencil is
rebalanced so its weights sum to zero, because the wall-normal stress
reconstruction integrates this operator and relies on ``D 1 = 0``.

:meth:`AxisMetric.negative_derivative_transpose` is the exact width-weighted
adjoint ``-W^-1 D^T W``.  On a uniform periodic axis the fourth-order difference
is antisymmetric and this collapses to ``D``, which is what the uniform kernels
use.  A stretched axis loses that antisymmetry, so the adjoint is assembled
explicitly; otherwise the variational SGS operator stops being energy
dissipative and the skew-symmetric advection split stops being energy neutral.

:meth:`AxisMetric.interface_states` reconstructs face values from cell averages
through the primitive function, so the fifth-order MP5 stencil keeps its formal
order on a stretched axis.  The Suresh-Huynh limiter around it is rewritten in
terms of slopes and physical lengths instead of raw differences, and reduces
term by term to the classical form when the spacing is constant.
"""

from __future__ import annotations

import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array

ReconstructionScheme = Literal["mp5", "muscl-mc"]

_MP5_ALPHA = 4.0
_FOURTH_ORDER_WIDTH = 5
_SECOND_ORDER_WIDTH = 3
_RECONSTRUCTION_WIDTH = 5


def _derivative_weights(nodes: np.ndarray, target: float) -> np.ndarray:
    """Return weights whose contraction with node values is ``d/dx`` at target.

    These are the weights of the interpolating polynomial through ``nodes``,
    obtained by requiring exactness on the shifted monomial basis.  Scaling the
    offsets keeps the small Vandermonde system well conditioned at the strong
    clustering ratios an atmospheric mesh uses near the ground.
    """

    count = int(nodes.size)
    if count < 2:
        raise ValueError("a derivative stencil needs at least two nodes")
    offsets = np.asarray(nodes, dtype=np.float64) - float(target)
    scale = float(np.max(np.abs(offsets)))
    if scale <= 0.0:
        raise ValueError("derivative nodes must not all coincide with the target")
    scaled = offsets / scale
    matrix = scaled[None, :] ** np.arange(count, dtype=np.float64)[:, None]
    right_hand_side = np.zeros(count, dtype=np.float64)
    right_hand_side[1] = 1.0
    return np.linalg.solve(matrix, right_hand_side) / scale


def minmod(*values: Array) -> Array:
    """Return the common-sign minimum magnitude, or zero on a sign change."""

    if not values:
        raise ValueError("minmod requires at least one value")
    magnitude = jnp.abs(values[0])
    all_positive = values[0] > 0.0
    all_negative = values[0] < 0.0
    for value in values[1:]:
        magnitude = jnp.minimum(magnitude, jnp.abs(value))
        all_positive &= value > 0.0
        all_negative &= value < 0.0
    return jnp.where(
        all_positive,
        magnitude,
        jnp.where(all_negative, -magnitude, 0.0),
    )


def wall_normal_derivative(field: Array, centers: Array) -> Array:
    """Differentiate leading-axis cell values on arbitrary centre coordinates."""

    if field.shape[0] == 1:
        return jnp.zeros_like(field)
    derivative = jnp.zeros_like(field)
    if field.shape[0] > 2:
        lower_distance = centers[1:-1] - centers[:-2]
        upper_distance = centers[2:] - centers[1:-1]
        shape = (lower_distance.shape[0],) + (1,) * (field.ndim - 1)
        lower_distance = jnp.reshape(lower_distance, shape)
        upper_distance = jnp.reshape(upper_distance, shape)
        lower_slope = (field[1:-1] - field[:-2]) / lower_distance
        upper_slope = (field[2:] - field[1:-1]) / upper_distance
        derivative = derivative.at[1:-1].set(
            (upper_distance * lower_slope + lower_distance * upper_slope)
            / (lower_distance + upper_distance)
        )
    lower_spacing = centers[1] - centers[0]
    upper_spacing = centers[-1] - centers[-2]
    derivative = derivative.at[0].set((field[1] - field[0]) / lower_spacing)
    derivative = derivative.at[-1].set((field[-1] - field[-2]) / upper_spacing)
    return derivative


def wall_normal_derivative_transpose(field: Array, centers: Array) -> Array:
    """Apply the Euclidean transpose of :func:`wall_normal_derivative`."""

    if field.shape[0] == 1:
        return jnp.zeros_like(field)
    result = jnp.zeros_like(field)
    lower_spacing = centers[1] - centers[0]
    upper_spacing = centers[-1] - centers[-2]
    result = result.at[0].add(-field[0] / lower_spacing)
    result = result.at[1].add(field[0] / lower_spacing)
    if field.shape[0] > 2:
        lower_distance = centers[1:-1] - centers[:-2]
        upper_distance = centers[2:] - centers[1:-1]
        shape = (lower_distance.shape[0],) + (1,) * (field.ndim - 1)
        lower_distance = jnp.reshape(lower_distance, shape)
        upper_distance = jnp.reshape(upper_distance, shape)
        lower_weight = -upper_distance / (
            lower_distance * (lower_distance + upper_distance)
        )
        center_weight = (upper_distance - lower_distance) / (
            lower_distance * upper_distance
        )
        upper_weight = lower_distance / (
            upper_distance * (lower_distance + upper_distance)
        )
        result = result.at[:-2].add(lower_weight * field[1:-1])
        result = result.at[1:-1].add(center_weight * field[1:-1])
        result = result.at[2:].add(upper_weight * field[1:-1])
    result = result.at[-2].add(-field[-1] / upper_spacing)
    result = result.at[-1].add(field[-1] / upper_spacing)
    return result


def _limit_interface_jump(
    left: Array,
    right: Array,
    cell_jump: Array,
) -> tuple[Array, Array]:
    """Project an interface jump onto the sign of the cell jump.

    Independent one-sided extrapolations can cross at a strongly curved face.
    A full Riemann flux may tolerate that crossing, but this solver adds the
    reconstructed jump as a correction to a kinetic-energy-neutral centred
    flux, so the correction must not be able to inject scalar variance or
    resolved kinetic energy.
    """

    orientation = jnp.sign(cell_jump)
    jump_magnitude = jnp.minimum(
        jnp.abs(cell_jump),
        jnp.maximum(orientation * (right - left), 0.0),
    )
    limited_jump = orientation * jump_magnitude
    midpoint = 0.5 * (left + right)
    return midpoint - 0.5 * limited_jump, midpoint + 0.5 * limited_jump


def _uniform_mp5_reconstruct(
    vm2: Array,
    vm1: Array,
    value: Array,
    vp1: Array,
    vp2: Array,
) -> Array:
    """Suresh-Huynh MP5 value on the face to the right of ``value``.

    Kept as the literal constant-spacing form so a uniform axis evaluates the
    same floating-point expression it always has.  :func:`_variable_mp5_limit`
    is its arbitrary-spacing generalization and agrees with it to roundoff.
    """

    unlimited = (2.0 * vm2 - 13.0 * vm1 + 47.0 * value + 27.0 * vp1 - 3.0 * vp2) / 60.0
    monotone = value + minmod(
        vp1 - value,
        _MP5_ALPHA * (value - vm1),
    )

    dm1 = vm2 - 2.0 * vm1 + value
    d0 = vm1 - 2.0 * value + vp1
    dp1 = value - 2.0 * vp1 + vp2
    curvature_plus = minmod(
        4.0 * d0 - dp1,
        4.0 * dp1 - d0,
        d0,
        dp1,
    )
    curvature_minus = minmod(
        4.0 * d0 - dm1,
        4.0 * dm1 - d0,
        d0,
        dm1,
    )
    upper_left = value + _MP5_ALPHA * (value - vm1)
    average = 0.5 * (value + vp1)
    median = average - 0.5 * curvature_plus
    large_curvature = value + 0.5 * (value - vm1) + (4.0 / 3.0) * curvature_minus
    return _mp5_select(
        value,
        vp1,
        unlimited,
        monotone,
        median,
        upper_left,
        large_curvature,
    )


def _mp5_select(
    value: Array,
    vp1: Array,
    unlimited: Array,
    monotone: Array,
    median: Array,
    upper_left: Array,
    large_curvature: Array,
) -> Array:
    """Accept the unlimited value or clip it into the MP5 admissible interval."""

    lower = jnp.maximum(
        jnp.minimum(jnp.minimum(value, vp1), median),
        jnp.minimum(jnp.minimum(value, upper_left), large_curvature),
    )
    upper = jnp.minimum(
        jnp.maximum(jnp.maximum(value, vp1), median),
        jnp.maximum(jnp.maximum(value, upper_left), large_curvature),
    )
    limited = unlimited + minmod(
        lower - unlimited,
        upper - unlimited,
    )
    tolerance = (
        10.0
        * jnp.finfo(value.dtype).eps
        * jnp.maximum(
            1.0,
            jnp.maximum(jnp.abs(value), jnp.abs(unlimited)),
        )
    )
    admissible = (unlimited - value) * (unlimited - monotone) <= tolerance
    return jnp.where(admissible, unlimited, limited)


def _sample_offset(values: Array, offset: int, *, periodic: bool) -> Array:
    if periodic:
        return jnp.roll(values, -offset, axis=-1)
    indices = jnp.clip(
        jnp.arange(values.shape[-1]) + offset,
        0,
        values.shape[-1] - 1,
    )
    return jnp.take(values, indices, axis=-1)


def mp5_interface_states(
    field: Array,
    *,
    axis: int,
    periodic: bool,
) -> tuple[Array, Array]:
    """Return left and right MP5 states on every face after a cell.

    Constant-spacing form.  A bounded axis clamps its outermost samples, which
    collapses the limiter to its one-sided branch at the first and last cell.
    """

    values = jnp.moveaxis(field, axis, -1)

    def sample(offset: int) -> Array:
        return _sample_offset(values, offset, periodic=periodic)

    vm2 = sample(-2)
    vm1 = sample(-1)
    value = sample(0)
    vp1 = sample(1)
    vp2 = sample(2)
    vp3 = sample(3)
    left = _uniform_mp5_reconstruct(vm2, vm1, value, vp1, vp2)
    right = _uniform_mp5_reconstruct(vp3, vp2, vp1, value, vm1)
    return jnp.moveaxis(left, -1, axis), jnp.moveaxis(right, -1, axis)


def muscl_mc_interface_states(
    field: Array,
    *,
    axis: int,
    periodic: bool,
) -> tuple[Array, Array]:
    """Return bounded, sign-preserving MUSCL-MC states on constant spacing."""

    values = jnp.moveaxis(field, axis, -1)

    def sample(offset: int) -> Array:
        return _sample_offset(values, offset, periodic=periodic)

    vm1 = sample(-1)
    value = sample(0)
    vp1 = sample(1)
    vp2 = sample(2)
    slope = minmod(
        2.0 * (value - vm1),
        0.5 * (vp1 - vm1),
        2.0 * (vp1 - value),
    )
    slope_p1 = minmod(
        2.0 * (vp1 - value),
        0.5 * (vp2 - value),
        2.0 * (vp2 - vp1),
    )
    left, right = _limit_interface_jump(
        value + 0.5 * slope,
        vp1 - 0.5 * slope_p1,
        vp1 - value,
    )
    return jnp.moveaxis(left, -1, axis), jnp.moveaxis(right, -1, axis)


def _variable_mp5_limit(
    vm2: Array,
    vm1: Array,
    value: Array,
    vp1: Array,
    vp2: Array,
    unlimited: Array,
    *,
    gap_m2: Array,
    gap_m1: Array,
    gap_0: Array,
    gap_p1: Array,
    theta: Array,
) -> Array:
    """Apply the Suresh-Huynh bounds on arbitrary spacing.

    ``gap_k`` are the positive centre distances of the stencil ordered along the
    direction of the reconstruction, and ``theta`` locates the face between the
    home cell centre and its downstream neighbour.  Each bound is a slope times
    a physical length, so constant spacing recovers the classical
    difference-based term exactly.
    """

    slope_minus = (value - vm1) / gap_m1
    slope_plus = (vp1 - value) / gap_0
    slope_far_minus = (vm1 - vm2) / gap_m2
    slope_far_plus = (vp2 - vp1) / gap_p1

    # Curvature scaled by the half-span of its own three-point stencil, so that
    # constant spacing gives back the classical second difference.
    dm1 = 0.5 * (gap_m2 + gap_m1) * (slope_minus - slope_far_minus)
    d0 = 0.5 * (gap_m1 + gap_0) * (slope_plus - slope_minus)
    dp1 = 0.5 * (gap_0 + gap_p1) * (slope_far_plus - slope_plus)

    monotone = value + gap_0 * minmod(slope_plus, _MP5_ALPHA * slope_minus)
    curvature_plus = minmod(4.0 * d0 - dp1, 4.0 * dp1 - d0, d0, dp1)
    curvature_minus = minmod(4.0 * d0 - dm1, 4.0 * dm1 - d0, d0, dm1)
    upper_left = value + _MP5_ALPHA * gap_0 * slope_minus
    average = value + theta * gap_0 * slope_plus
    median = average - 0.5 * curvature_plus
    large_curvature = value + 0.5 * gap_0 * slope_minus + (4.0 / 3.0) * curvature_minus
    return _mp5_select(
        value,
        vp1,
        unlimited,
        monotone,
        median,
        upper_left,
        large_curvature,
    )


class AxisMetric:
    """Geometry and discrete operators of one storage axis of a cell field.

    ``axis`` is the position of this coordinate in canonical ``z-y-x`` storage,
    so ``0`` is the wall-normal axis and ``2`` the first horizontal one.  Fields
    carrying trailing component axes are handled by broadcasting rather than by
    a separate code path.

    ``derivative_width`` selects the centred stencil of :meth:`derivative`: the
    horizontal axes use the five-point fourth-order stencil and the wall-normal
    axis the three-point second-order one, which is the accuracy split the wall
    model and the vertical line solves are built around.
    """

    def __init__(
        self,
        faces,
        *,
        axis: int,
        periodic: bool,
        dtype,
        derivative_width: int = _FOURTH_ORDER_WIDTH,
    ) -> None:
        host_faces = np.asarray(faces, dtype=np.float64).reshape(-1)
        if host_faces.size < 2:
            raise ValueError("an axis metric needs at least one cell")
        if not np.all(np.diff(host_faces) > 0.0):
            raise ValueError("axis faces must be strictly increasing")
        if derivative_width not in {_SECOND_ORDER_WIDTH, _FOURTH_ORDER_WIDTH}:
            raise ValueError("derivative width must be three or five")
        if derivative_width == _SECOND_ORDER_WIDTH and periodic:
            raise ValueError("the three-point stencil is for bounded axes only")

        self.axis = int(axis)
        self.periodic = bool(periodic)
        self.dtype = dtype
        self.count = int(host_faces.size - 1)
        self.derivative_width = int(derivative_width)

        host_widths = np.diff(host_faces)
        host_centers = 0.5 * (host_faces[:-1] + host_faces[1:])
        self.length = float(host_faces[-1] - host_faces[0])
        self.spacing = self.length / self.count
        tolerance = 1.0e-12 * max(1.0, abs(self.spacing))
        self.uniform = all(
            math.isclose(
                float(width),
                self.spacing,
                rel_tol=1.0e-12,
                abs_tol=tolerance,
            )
            for width in host_widths
        )

        self._host_faces = host_faces
        self._host_widths = host_widths
        self._host_centers = host_centers
        self.faces = jnp.asarray(host_faces, dtype=dtype)
        self.widths = jnp.asarray(host_widths, dtype=dtype)
        self.centers = jnp.asarray(host_centers, dtype=dtype)

        host_center_gaps = (
            0.5 * (host_widths + np.roll(host_widths, -1))
            if self.periodic
            else np.diff(host_centers)
        )
        self._host_center_gaps = host_center_gaps
        self.center_gaps = jnp.asarray(host_center_gaps, dtype=dtype)

        self._host_offset_centers: dict[int, np.ndarray] = {}
        self._derivative_stencil: tuple[Array, Array] | None = None
        self._reconstruction: dict[str, tuple[Array, Array]] = {}

        needs_derivative_stencil = self.derivative_width == _FOURTH_ORDER_WIDTH and not (
            self.uniform and self.periodic
        )
        if needs_derivative_stencil:
            self._derivative_stencil = self._build_derivative_stencil()
        if not self.uniform:
            self._reconstruction["left"] = self._build_reconstruction_stencil(-2)
            self._reconstruction["right"] = self._build_reconstruction_stencil(-1)

    # ---------------------------------------------------------------- geometry

    def _host_center_at(self, index: int) -> float:
        """Return the unwrapped centre coordinate of cell ``index``.

        Periodic indices are unwrapped through the axis length.  A bounded axis
        extrapolates linearly past its first and last centre while sampled
        *values* are clamped, which keeps every stencil distance positive and
        finite and degrades the limiter to its one-sided form at the boundary.
        """

        if self.periodic:
            return float(
                self._host_centers[index % self.count]
                + self.length * (index // self.count)
            )
        if index < 0:
            gap = float(self._host_centers[1] - self._host_centers[0])
            return float(self._host_centers[0] + index * gap)
        if index > self.count - 1:
            gap = float(self._host_centers[-1] - self._host_centers[-2])
            return float(self._host_centers[-1] + (index - (self.count - 1)) * gap)
        return float(self._host_centers[index])

    def _host_face_at(self, index: int) -> float:
        if self.periodic:
            return float(
                self._host_faces[index % self.count]
                + self.length * (index // self.count)
            )
        return float(self._host_faces[int(np.clip(index, 0, self.count))])

    def _host_width_at(self, index: int) -> float:
        if self.periodic:
            return float(self._host_widths[index % self.count])
        return float(self._host_widths[int(np.clip(index, 0, self.count - 1))])

    def _sample_index(self, index: int) -> int:
        if self.periodic:
            return index % self.count
        return int(np.clip(index, 0, self.count - 1))

    def _offset_centers(self, offset: int) -> np.ndarray:
        cached = self._host_offset_centers.get(offset)
        if cached is None:
            cached = np.array(
                [self._host_center_at(cell + offset) for cell in range(self.count)],
                dtype=np.float64,
            )
            self._host_offset_centers[offset] = cached
        return cached

    def _stencil_start(self, index: int, width: int, offset: int) -> int:
        """Return the first stencil cell, shifted inward on a bounded axis."""

        start = index + offset
        if self.periodic:
            return start
        return int(np.clip(start, 0, self.count - width))

    # ------------------------------------------------------- stencil assembly

    def _build_derivative_stencil(self) -> tuple[Array, Array]:
        width = min(self.derivative_width, self.count)
        offset = -(width // 2)
        weights = np.zeros((self.count, width), dtype=np.float64)
        indices = np.zeros((self.count, width), dtype=np.int64)
        for cell in range(self.count):
            first = self._stencil_start(cell, width, offset)
            stencil = [first + step for step in range(width)]
            nodes = np.array(
                [self._host_center_at(entry) for entry in stencil],
                dtype=np.float64,
            )
            row = _derivative_weights(nodes, self._host_center_at(cell))
            # A constant must differentiate to zero to roundoff.
            row[cell - first] -= row.sum()
            weights[cell] = row
            indices[cell] = [self._sample_index(entry) for entry in stencil]
        return (
            jnp.asarray(indices, dtype=jnp.int32),
            jnp.asarray(weights, dtype=self.dtype),
        )

    def _build_reconstruction_stencil(self, offset: int) -> tuple[Array, Array]:
        """Return primitive-function face weights for one stencil placement.

        Interpolating the primitive function ``V(x) = int v dx`` through the
        ``width + 1`` faces bounding the stencil cells and differentiating that
        interpolant at the target face turns the known cell averages into a face
        value that keeps fifth-order accuracy on arbitrary spacing.  Constant
        spacing reproduces the classical ``(2, -13, 47, 27, -3) / 60`` weights.
        """

        width = min(_RECONSTRUCTION_WIDTH, self.count)
        weights = np.zeros((self.count, width), dtype=np.float64)
        indices = np.zeros((self.count, width), dtype=np.int64)
        for cell in range(self.count):
            first = self._stencil_start(cell, width, offset)
            nodes = np.array(
                [self._host_face_at(first + step) for step in range(width + 1)],
                dtype=np.float64,
            )
            primitive = _derivative_weights(nodes, self._host_face_at(cell + 1))
            # v_face = sum_m d_m V_m with V_m the primitive at face m.  Writing
            # V_m as the partial sum of h_j v_j leaves one weight per cell.
            suffix = np.cumsum(primitive[::-1])[::-1]
            row = np.array(
                [
                    self._host_width_at(first + step) * suffix[step + 1]
                    for step in range(width)
                ],
                dtype=np.float64,
            )
            # A constant field must be reconstructed exactly.
            row /= row.sum()
            weights[cell] = row
            indices[cell] = [
                self._sample_index(first + step) for step in range(width)
            ]
        return (
            jnp.asarray(indices, dtype=jnp.int32),
            jnp.asarray(weights, dtype=self.dtype),
        )

    # ------------------------------------------------------------ broadcasting

    def broadcast(self, values: Array, ndim: int, dtype=None) -> Array:
        """Reshape a per-cell vector for broadcasting over an ``ndim`` field."""

        if ndim <= self.axis:
            raise ValueError("field rank does not reach this metric axis")
        shape = [1] * ndim
        shape[self.axis] = -1
        reshaped = jnp.reshape(values, tuple(shape))
        if dtype is not None and reshaped.dtype != dtype:
            return reshaped.astype(dtype)
        return reshaped

    def cell_widths(self, ndim: int, dtype=None) -> Array:
        """Return the cell widths broadcast over an ``ndim`` field."""

        return self.broadcast(self.widths, ndim, dtype)

    @staticmethod
    def _leading(values: Array, ndim: int, dtype=None) -> Array:
        """Shape a per-cell vector for a leading-axis field of rank ``ndim``.

        ``dtype`` follows the field rather than the metric.  Geometry is held at
        the dtype the grid was built with, which need not be the dtype a caller
        integrates in, and letting the wider one win would silently promote
        whole fields.
        """

        reshaped = jnp.reshape(values, (values.shape[0],) + (1,) * (ndim - 1))
        if dtype is not None and reshaped.dtype != dtype:
            return reshaped.astype(dtype)
        return reshaped

    @property
    def diffusion_diagonal(self) -> Array:
        """Return the finite-volume diffusion diagonal per unit diffusivity.

        Cell ``i`` contributes ``1 / (h_i d_-) + 1 / (h_i d_+)`` with ``d`` the
        centre distances across its two faces.  A bounded axis drops its two
        wall faces, which is the zero-flux natural boundary the vertical
        operators use.  Uniform spacing gives back ``2 / h**2``.
        """

        if self.periodic:
            lower = self._host_center_gaps[np.arange(self.count) - 1]
            upper = self._host_center_gaps
        else:
            infinite = np.array([np.inf])
            lower = np.concatenate((infinite, self._host_center_gaps))
            upper = np.concatenate((self._host_center_gaps, infinite))
        diagonal = 1.0 / (self._host_widths * lower) + 1.0 / (
            self._host_widths * upper
        )
        return jnp.asarray(diagonal, dtype=self.dtype)

    # --------------------------------------------------------------- operators

    def _sample(self, values: Array, offset: int) -> Array:
        """Return leading-axis values shifted by ``offset`` cells."""

        if offset == 0:
            return values
        if self.periodic:
            return jnp.roll(values, -offset, axis=0)
        indices = np.clip(np.arange(self.count) + offset, 0, self.count - 1)
        return jnp.take(values, jnp.asarray(indices, dtype=jnp.int32), axis=0)

    def derivative(self, field: Array) -> Array:
        """Return the centred derivative of a cell field along this axis."""

        if self.derivative_width == _SECOND_ORDER_WIDTH:
            return self._three_point_derivative(field)
        if self.uniform and self.periodic:
            return self._uniform_fourth_derivative(field)
        values = jnp.moveaxis(field, self.axis, 0)
        return jnp.moveaxis(
            self._apply_stencil(values, self._derivative_stencil),
            0,
            self.axis,
        )

    def _apply_stencil(
        self,
        values: Array,
        stencil: tuple[Array, Array],
    ) -> Array:
        indices, weights = stencil
        result = jnp.zeros_like(values)
        for step in range(weights.shape[1]):
            row = self._leading(weights[:, step], values.ndim, values.dtype)
            result += row * jnp.take(values, indices[:, step], axis=0)
        return result

    def _uniform_fourth_derivative(self, field: Array) -> Array:
        axis = self.axis
        return (
            -jnp.roll(field, -2, axis=axis)
            + 8.0 * jnp.roll(field, -1, axis=axis)
            - 8.0 * jnp.roll(field, 1, axis=axis)
            + jnp.roll(field, 2, axis=axis)
        ) / (12.0 * self.spacing)

    def _three_point_derivative(self, field: Array) -> Array:
        values = jnp.moveaxis(field, self.axis, 0)
        return jnp.moveaxis(
            wall_normal_derivative(values, self.centers.astype(values.dtype)),
            0,
            self.axis,
        )

    def _derivative_transpose(self, field: Array) -> Array:
        """Apply the Euclidean transpose of :meth:`derivative`."""

        if self.derivative_width == _SECOND_ORDER_WIDTH:
            return self._three_point_derivative_transpose(field)
        values = jnp.moveaxis(field, self.axis, 0)
        indices, weights = self._derivative_stencil
        result = jnp.zeros_like(values)
        for step in range(weights.shape[1]):
            row = self._leading(weights[:, step], values.ndim, values.dtype)
            result = result.at[indices[:, step]].add(row * values)
        return jnp.moveaxis(result, 0, self.axis)

    def _three_point_derivative_transpose(self, field: Array) -> Array:
        values = jnp.moveaxis(field, self.axis, 0)
        return jnp.moveaxis(
            wall_normal_derivative_transpose(
                values,
                self.centers.astype(values.dtype),
            ),
            0,
            self.axis,
        )

    def negative_derivative_transpose(self, field: Array) -> Array:
        """Return ``-W^-1 D^T W`` with ``W`` the cell widths of this axis.

        This is minus the adjoint of :meth:`derivative` in the cell-volume
        inner product.  The widths of the other two axes are constant along
        this one and cancel, so a separable volume needs only this axis.  Used
        as a divergence it keeps the variational SGS operator dissipative and
        the skew-symmetric advection split energy neutral on any spacing.
        """

        if (
            self.uniform
            and self.periodic
            and self.derivative_width == _FOURTH_ORDER_WIDTH
        ):
            # The uniform periodic fourth-order difference is antisymmetric.
            return self.derivative(field)
        widths = self.cell_widths(field.ndim, field.dtype)
        return -self._derivative_transpose(field * widths) / widths

    def upper_face_flux_divergence(self, flux: Array) -> Array:
        """Return ``(F_i - F_{i-1}) / h_i`` for fluxes on upper cell faces.

        A bounded axis takes a vanishing flux below its first cell, matching the
        zero-flux natural boundary of the wall-normal reconstruction.
        """

        axis = self.axis
        if self.periodic:
            previous = jnp.roll(flux, 1, axis=axis)
        else:
            first = [slice(None)] * flux.ndim
            first[axis] = slice(0, 1)
            preceding = [slice(None)] * flux.ndim
            preceding[axis] = slice(0, -1)
            previous = jnp.concatenate(
                (
                    jnp.zeros_like(flux[tuple(first)]),
                    flux[tuple(preceding)],
                ),
                axis=axis,
            )
        return (flux - previous) / self.cell_widths(flux.ndim, flux.dtype)

    # ---------------------------------------------------------- reconstruction

    def interface_states(
        self,
        field: Array,
        scheme: ReconstructionScheme,
    ) -> tuple[Array, Array]:
        """Return one-sided states on the upper face of every cell."""

        if scheme == "mp5":
            if self.uniform:
                return mp5_interface_states(
                    field,
                    axis=self.axis,
                    periodic=self.periodic,
                )
            return self._variable_mp5_states(field)
        if scheme == "muscl-mc":
            if self.uniform and self.periodic:
                return muscl_mc_interface_states(
                    field,
                    axis=self.axis,
                    periodic=True,
                )
            return self._variable_muscl_states(field)
        raise ValueError("reconstruction scheme must be 'mp5' or 'muscl-mc'")

    def _variable_mp5_states(self, field: Array) -> tuple[Array, Array]:
        values = jnp.moveaxis(field, self.axis, 0)
        ndim = values.ndim
        sample = {offset: self._sample(values, offset) for offset in range(-2, 4)}
        gap = {
            offset: self._leading(self._center_gap(offset), ndim, values.dtype)
            for offset in range(-2, 3)
        }
        theta = self._leading(self._face_position(), ndim, values.dtype)
        left = _variable_mp5_limit(
            sample[-2],
            sample[-1],
            values,
            sample[1],
            sample[2],
            self._apply_stencil(values, self._reconstruction["left"]),
            gap_m2=gap[-2],
            gap_m1=gap[-1],
            gap_0=gap[0],
            gap_p1=gap[1],
            theta=theta,
        )
        right = _variable_mp5_limit(
            sample[3],
            sample[2],
            sample[1],
            values,
            sample[-1],
            self._apply_stencil(values, self._reconstruction["right"]),
            gap_m2=gap[2],
            gap_m1=gap[1],
            gap_0=gap[0],
            gap_p1=gap[-1],
            theta=1.0 - theta,
        )
        return jnp.moveaxis(left, 0, self.axis), jnp.moveaxis(right, 0, self.axis)

    def _center_gap(self, offset: int) -> Array:
        """Return the centre distance from cell ``i + offset`` to ``i + offset + 1``."""

        lower = self._offset_centers(offset)
        upper = self._offset_centers(offset + 1)
        return jnp.asarray(upper - lower, dtype=self.dtype)

    def _face_position(self) -> Array:
        """Return the face position normalized by the bracketing centre gap."""

        home = self._offset_centers(0)
        upper = self._offset_centers(1)
        face = np.array(
            [self._host_face_at(cell + 1) for cell in range(self.count)],
            dtype=np.float64,
        )
        return jnp.asarray((face - home) / (upper - home), dtype=self.dtype)

    def _monotonicity_factors(self) -> tuple[Array, Array]:
        """Return the per-cell MC limiter factors on the lower and upper side.

        The factor of two in the classical monotonized-central limiter is the
        ratio of the neighbour centre distance to the half width of the cell,
        which is what bounds a one-sided extrapolation by the neighbouring cell
        value.  Where the widths change that ratio is no longer two, and keeping
        it explicit is what stops the reconstruction from overshooting into a
        refining region.  Constant spacing gives back two on both sides.
        """

        lower_gap = self._offset_centers(0) - self._offset_centers(-1)
        upper_gap = self._offset_centers(1) - self._offset_centers(0)
        half_width = 0.5 * self._host_widths
        return (
            jnp.asarray(lower_gap / half_width, dtype=self.dtype),
            jnp.asarray(upper_gap / half_width, dtype=self.dtype),
        )

    def _variable_muscl_states(self, field: Array) -> tuple[Array, Array]:
        values = jnp.moveaxis(field, self.axis, 0)
        ndim = values.ndim
        dtype = values.dtype
        lower_factor, upper_factor = self._monotonicity_factors()
        slope = jnp.zeros_like(values)
        if self.periodic:
            lower = self._leading(self._center_gap(-1), ndim, dtype)
            upper = self._leading(self._center_gap(0), ndim, dtype)
            lower_slope = (values - self._sample(values, -1)) / lower
            upper_slope = (self._sample(values, 1) - values) / upper
            centered = (upper * lower_slope + lower * upper_slope) / (lower + upper)
            slope = minmod(
                self._leading(lower_factor, ndim, dtype) * lower_slope,
                centered,
                self._leading(upper_factor, ndim, dtype) * upper_slope,
            )
            next_value = self._sample(values, 1)
            next_slope = self._sample(slope, 1)
        else:
            if values.shape[0] > 2:
                lower = self._leading(self.center_gaps[:-1], ndim, dtype)
                upper = self._leading(self.center_gaps[1:], ndim, dtype)
                lower_slope = (values[1:-1] - values[:-2]) / lower
                upper_slope = (values[2:] - values[1:-1]) / upper
                centered = (upper * lower_slope + lower * upper_slope) / (
                    lower + upper
                )
                slope = slope.at[1:-1].set(
                    minmod(
                        self._leading(lower_factor[1:-1], ndim, dtype) * lower_slope,
                        centered,
                        self._leading(upper_factor[1:-1], ndim, dtype) * upper_slope,
                    )
                )
            next_value = jnp.concatenate((values[1:], values[-1:]), axis=0)
            next_slope = jnp.concatenate((slope[1:], slope[-1:]), axis=0)

        upper_offset = self._leading(self._upper_face_offset(), ndim, dtype)
        next_lower_offset = self._leading(
            self._next_lower_face_offset(),
            ndim,
            dtype,
        )
        left, right = _limit_interface_jump(
            values + slope * upper_offset,
            next_value - next_slope * next_lower_offset,
            next_value - values,
        )
        return jnp.moveaxis(left, 0, self.axis), jnp.moveaxis(right, 0, self.axis)

    def _upper_face_offset(self) -> Array:
        """Return the distance from every cell centre up to its upper face."""

        offsets = np.array(
            [
                self._host_face_at(cell + 1) - self._host_centers[cell]
                for cell in range(self.count)
            ],
            dtype=np.float64,
        )
        return jnp.asarray(offsets, dtype=self.dtype)

    def _next_lower_face_offset(self) -> Array:
        """Return the distance from the next cell centre down to that face."""

        offsets = np.zeros(self.count, dtype=np.float64)
        for cell in range(self.count):
            if not self.periodic and cell == self.count - 1:
                continue
            offsets[cell] = self._host_center_at(cell + 1) - self._host_face_at(
                cell + 1
            )
        return jnp.asarray(offsets, dtype=self.dtype)


def reconstruction_dissipation(
    field: Array,
    directions: tuple[tuple[Array, AxisMetric], ...],
    strength: float,
    scheme: ReconstructionScheme,
) -> Array:
    """Return the conservative Rusanov correction of a reconstruction."""

    tendency = jnp.zeros_like(field)
    for face_speed, metric in directions:
        left, right = metric.interface_states(field, scheme)
        speed = jnp.abs(face_speed)
        if speed.ndim < field.ndim:
            speed = speed[..., None]
        flux = -0.5 * strength * speed * (right - left)
        tendency -= metric.upper_face_flux_divergence(flux)
    return tendency


__all__ = [
    "AxisMetric",
    "ReconstructionScheme",
    "minmod",
    "mp5_interface_states",
    "muscl_mc_interface_states",
    "reconstruction_dissipation",
    "wall_normal_derivative",
    "wall_normal_derivative_transpose",
]
