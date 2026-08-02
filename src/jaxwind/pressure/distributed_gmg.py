"""Multi-device y-slab matrix-free GMG with replicated coarse levels."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
from jax import lax
import jax.numpy as jnp

from .fgmres import FGMRESConfig, FGMRESResult, fgmres
from .pcg import PCGConfig, PCGResult, pcg
from .matrix_free_gmg import (
    GMGConfig,
    MatrixFreeGMG,
    MatrixFreePoissonOperator,
    PoissonBoundaryConditions,
    RectilinearGrid,
    _apply_axis,
    _axis_diagonal,
    _interpolate_axis,
    _transpose_interpolate_axis,
)


Array = jax.Array


@dataclass(frozen=True, slots=True)
class YSlabConfig:
    """Controls for the pmap y-slab reference backend."""

    coarse_cells_per_device: int = 4
    axis_name: str = "poisson_y"

    def __post_init__(self) -> None:
        if self.coarse_cells_per_device <= 0:
            raise ValueError("coarse_cells_per_device must be positive")
        if not self.axis_name:
            raise ValueError("axis_name must be nonempty")


def shard_y_field(field: Array, device_count: int) -> Array:
    """Convert global ``(z,y,x)`` storage to device-leading y slabs."""
    if field.ndim != 3:
        raise ValueError("global pressure field must have three dimensions")
    nz, ny, nx = field.shape
    if ny % device_count:
        raise ValueError("global y extent must divide across devices")
    local_y = ny // device_count
    return jnp.transpose(
        jnp.reshape(field, (nz, device_count, local_y, nx)),
        (1, 0, 2, 3),
    )


def gather_y_field(field: Array) -> Array:
    """Convert device-leading y slabs back to global z-y-x storage."""
    if field.ndim != 4:
        raise ValueError("sharded pressure field must have four dimensions")
    devices, nz, local_y, nx = field.shape
    return jnp.reshape(
        jnp.transpose(field, (1, 0, 2, 3)),
        (nz, devices * local_y, nx),
    )


class YSlabMatrixFreePoissonSolver:
    """Y-slab halo GMG that keeps complete vertical columns local."""

    def __init__(
        self,
        grid: RectilinearGrid,
        boundaries: PoissonBoundaryConditions,
        *,
        devices: tuple[jax.Device, ...] | None = None,
        dtype: jnp.dtype = jnp.float64,
        gmg: GMGConfig = GMGConfig(),
        krylov: FGMRESConfig | PCGConfig = FGMRESConfig(),
        distribution: YSlabConfig = YSlabConfig(),
    ) -> None:
        selected = tuple(jax.local_devices()) if devices is None else tuple(devices)
        if not selected:
            raise ValueError("at least one device is required")
        process_count = jax.process_count()
        if process_count > 1 and devices is not None:
            expected = tuple(jax.local_devices())
            if selected != expected:
                raise ValueError(
                    "multi-process pmap must use every local JAX device"
                )
        self.local_device_count = len(selected)
        self.device_count = (
            jax.device_count() if process_count > 1 else len(selected)
        )
        if grid.shape[1] % self.device_count:
            raise ValueError("finest-grid y cells must divide across devices")
        self.devices = selected
        self.distribution = distribution
        self.operator = MatrixFreePoissonOperator(
            grid,
            boundaries,
            dtype=dtype,
        )
        self.serial_gmg = MatrixFreeGMG(self.operator, gmg)
        self.krylov = replace(krylov, jit_kernels=False)
        if (
            self.krylov.execution == "jax"
            and not isinstance(self.krylov, PCGConfig)
        ):
            raise ValueError(
                "device-resident y-slab solves currently require PCG"
            )
        self.replication_level = self._choose_replication_level()
        self._validate_sharded_levels()
        mapped: dict[str, object] = dict(
            axis_name=distribution.axis_name,
            axis_size=self.device_count,
        )
        if process_count == 1:
            mapped["devices"] = self.devices
        self.pmap_options = mapped
        self._mapped_apply = jax.pmap(self._apply_local, **mapped)
        self._mapped_precondition = jax.pmap(
            self._precondition_local,
            **mapped,
        )
        self._mapped_project = jax.pmap(
            self._project_finest_local,
            **mapped,
        )
        self._mapped_inner = jax.pmap(self._inner_finest_local, **mapped)
        self._mapped_prepare_rhs = jax.pmap(
            self._prepare_rhs_local,
            **mapped,
        )
        self._mapped_device_solve = (
            jax.pmap(
                self._device_pcg_local,
                in_axes=(0, 0),
                **mapped,
            )
            if self.krylov.execution == "jax"
            else None
        )

    @property
    def level_shapes(self) -> tuple[tuple[int, int, int], ...]:
        return self.serial_gmg.level_shapes

    @property
    def replicated_shape(self) -> tuple[int, int, int]:
        return self.level_shapes[self.replication_level]

    def _choose_replication_level(self) -> int:
        threshold = (
            self.distribution.coarse_cells_per_device * self.device_count
        )
        for level, shape in enumerate(self.level_shapes):
            if shape[1] <= threshold:
                return level
            if (
                level + 1 < len(self.level_shapes)
                and self.level_shapes[level + 1][1] % self.device_count
            ):
                return level
        return len(self.level_shapes) - 1

    def _validate_sharded_levels(self) -> None:
        for level in range(self.replication_level + 1):
            if self.level_shapes[level][1] % self.device_count:
                raise ValueError(f"level {level} y extent does not divide devices")
        for level in range(self.replication_level):
            local_y = self.level_shapes[level][1] // self.device_count
            if local_y % self.serial_gmg.coarsening_factors[level][1]:
                raise ValueError("restriction would cross a y-slab boundary")

    def _rank(self) -> Array:
        return lax.axis_index(self.distribution.axis_name)

    def _slice_y(self, field: Array, level: int) -> Array:
        global_y = self.level_shapes[level][1]
        local_y = global_y // self.device_count
        return lax.dynamic_slice_in_dim(
            field,
            self._rank() * local_y,
            local_y,
            axis=1,
        )

    def _exchange_y(self, field: Array, periodic: bool) -> tuple[Array, Array]:
        if self.device_count == 1:
            if periodic:
                return field[:, -1, :], field[:, 0, :]
            zeros = jnp.zeros_like(field[:, 0, :])
            return zeros, zeros
        if periodic:
            send_right = [
                (source, (source + 1) % self.device_count)
                for source in range(self.device_count)
            ]
            send_left = [
                (source, (source - 1) % self.device_count)
                for source in range(self.device_count)
            ]
        else:
            send_right = [
                (source, source + 1)
                for source in range(self.device_count - 1)
            ]
            send_left = [
                (source, source - 1)
                for source in range(1, self.device_count)
            ]
        left = lax.ppermute(
            field[:, -1, :],
            self.distribution.axis_name,
            send_right,
        )
        right = lax.ppermute(
            field[:, 0, :],
            self.distribution.axis_name,
            send_left,
        )
        return left, right

    def _apply_local_at_level(self, pressure: Array, level: int) -> Array:
        operator = self.serial_gmg.operators[level]
        wx, wy, wz = operator._level.widths
        cx, cy, cz = operator._level.centers
        result = _apply_axis(
            pressure,
            axis=-1,
            widths=wx,
            centers=cx,
            lower=operator.boundaries.x_lower,
            upper=operator.boundaries.x_upper,
        )
        result += _apply_axis(
            pressure,
            axis=-3,
            widths=wz,
            centers=cz,
            lower=operator.boundaries.z_lower,
            upper=operator.boundaries.z_upper,
        )
        global_y = operator.shape[1]
        local_y = global_y // self.device_count
        start = self._rank() * local_y
        widths = lax.dynamic_slice_in_dim(wy, start, local_y)
        centers = lax.dynamic_slice_in_dim(cy, start, local_y)
        y_result = jnp.zeros_like(pressure)
        if local_y > 1:
            distance = centers[1:] - centers[:-1]
            difference = pressure[:, :-1, :] - pressure[:, 1:, :]
            y_result = y_result.at[:, :-1, :].add(
                difference / widths[:-1][None, :, None] / distance[None, :, None]
            )
            y_result = y_result.at[:, 1:, :].add(
                -difference / widths[1:][None, :, None] / distance[None, :, None]
            )

        periodic = operator.boundaries.y_lower.kind == "periodic"
        left, right = self._exchange_y(pressure, periodic)
        rank = self._rank()
        previous = lax.dynamic_index_in_dim(
            cy,
            jnp.maximum(start - 1, 0),
            keepdims=False,
        )
        following = lax.dynamic_index_in_dim(
            cy,
            jnp.minimum(start + local_y, global_y - 1),
            keepdims=False,
        )
        lower_internal = (
            (pressure[:, 0, :] - left)
            / widths[0]
            / (centers[0] - previous)
        )
        upper_internal = (
            (pressure[:, -1, :] - right)
            / widths[-1]
            / (following - centers[-1])
        )
        if periodic:
            wrap = 0.5 * (wy[0] + wy[-1])
            lower_physical = (
                (pressure[:, 0, :] - left) / widths[0] / wrap
            )
            upper_physical = (
                (pressure[:, -1, :] - right) / widths[-1] / wrap
            )
        else:
            lower_physical = jnp.zeros_like(left)
            upper_physical = jnp.zeros_like(right)
            if operator.boundaries.y_lower.kind == "dirichlet":
                lower_physical = (
                    2.0 * pressure[:, 0, :] / (widths[0] * widths[0])
                )
            if operator.boundaries.y_upper.kind == "dirichlet":
                upper_physical = (
                    2.0 * pressure[:, -1, :] / (widths[-1] * widths[-1])
                )
        y_result = y_result.at[:, 0, :].add(
            jnp.where(rank > 0, lower_internal, lower_physical)
        )
        y_result = y_result.at[:, -1, :].add(
            jnp.where(
                rank < self.device_count - 1,
                upper_internal,
                upper_physical,
            )
        )
        return result + y_result

    def _apply_local(self, pressure: Array) -> Array:
        return self._apply_local_at_level(pressure, 0)

    def _project_local(self, field: Array, level: int) -> Array:
        operator = self.serial_gmg.operators[level]
        if not operator.has_constant_nullspace:
            return field
        volume = self._slice_y(operator.volume, level)
        total = lax.psum(
            jnp.sum(volume * field),
            self.distribution.axis_name,
        )
        return field - total * operator.inverse_total_volume

    def _project_finest_local(self, field: Array) -> Array:
        return self._project_local(field, 0)

    def _inner_finest_local(self, left: Array, right: Array) -> Array:
        volume = self._slice_y(self.operator.volume, 0)
        value = jnp.sum(volume * left * right)
        return lax.psum(value, self.distribution.axis_name)

    def _line_solve_local(self, rhs: Array, level: int) -> Array:
        operator = self.serial_gmg.operators[level]
        wx, wy, wz = operator._level.widths
        cx, cy, cz = operator._level.centers
        dx = _axis_diagonal(
            wx,
            cx,
            operator.boundaries.x_lower,
            operator.boundaries.x_upper,
        )
        global_dy = _axis_diagonal(
            wy,
            cy,
            operator.boundaries.y_lower,
            operator.boundaries.y_upper,
        )
        local_dy = self._slice_y(global_dy[None, :, None], level)[0, :, 0]
        dz = _axis_diagonal(
            wz,
            cz,
            operator.boundaries.z_lower,
            operator.boundaries.z_upper,
        )
        diagonal = (
            dz[:, None, None]
            + local_dy[None, :, None]
            + dx[None, None, :]
        )
        if rhs.shape[0] == 1:
            return rhs / jnp.where(diagonal != 0.0, diagonal, 1.0)
        distance = cz[1:] - cz[:-1]
        lower = jnp.zeros_like(wz).at[1:].set(
            -1.0 / (wz[1:] * distance)
        )
        upper = jnp.zeros_like(wz).at[:-1].set(
            -1.0 / (wz[:-1] * distance)
        )
        denominator = jnp.where(diagonal[0] != 0.0, diagonal[0], 1.0)
        first_upper = upper[0] / denominator
        first_rhs = rhs[0] / denominator

        def forward(carry, values):
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

        _, (upper_tail, rhs_tail) = lax.scan(
            forward,
            (first_upper, first_rhs),
            (lower[1:], diagonal[1:], upper[1:], rhs[1:]),
        )
        reduced_upper = jnp.concatenate(
            (first_upper[None], upper_tail),
            axis=0,
        )
        reduced_rhs = jnp.concatenate((first_rhs[None], rhs_tail), axis=0)

        def backward(next_value, values):
            rhs_value, upper_value = values
            value = rhs_value - upper_value * next_value
            return value, value

        _, prefix_reverse = lax.scan(
            backward,
            reduced_rhs[-1],
            (reduced_rhs[:-1][::-1], reduced_upper[:-1][::-1]),
        )
        return jnp.concatenate(
            (prefix_reverse[::-1], reduced_rhs[-1:]),
            axis=0,
        )

    def _smooth_local(
        self,
        level: int,
        solution: Array,
        rhs: Array,
        count: int,
    ) -> Array:
        operator = self.serial_gmg.operators[level]
        diagonal = self._slice_y(operator.diagonal, level)
        for _ in range(count):
            residual = rhs - self._apply_local_at_level(solution, level)
            if self.serial_gmg.level_smoothers[level] == "z_line":
                correction = self._line_solve_local(residual, level)
                solution += self.serial_gmg.config.line_omega * correction
            else:
                solution += (
                    self.serial_gmg.config.jacobi_omega
                    * residual
                    / diagonal
                )
            # Constant pressure offsets do not change the operator residual.
            # Fix the gauge once at the preconditioner exit instead of issuing
            # a cross-rank reduction after every smoothing update.
        return solution

    def _restrict_local(self, field: Array, level: int) -> Array:
        fine = self.serial_gmg.operators[level]
        coarse = self.serial_gmg.operators[level + 1]
        transfer = self.serial_gmg.transfers[level]
        _, local_y, _ = field.shape
        fine_volume = self._slice_y(fine.volume, level)
        coarse_volume = self._slice_y(coarse.volume, level + 1)
        weighted = _transpose_interpolate_axis(
            field * fine_volume,
            transfer.z,
            -3,
            coarse.shape[0],
        )
        weighted = _transpose_interpolate_axis(
            weighted,
            transfer.x,
            -1,
            coarse.shape[2],
        )

        fine_local_y = fine.shape[1] // self.device_count
        coarse_local_y = coarse.shape[1] // self.device_count
        rank = self._rank()
        fine_start = rank * fine_local_y
        coarse_start = rank * coarse_local_y

        def local_slice(array):
            return lax.dynamic_slice_in_dim(
                array,
                fine_start,
                local_y,
            )

        lower_index = local_slice(transfer.y.lower_index)
        upper_index = local_slice(transfer.y.upper_index)
        lower_weight = local_slice(transfer.y.lower_weight)
        upper_weight = local_slice(transfer.y.upper_weight)
        periodic = fine.boundaries.y_lower.kind == "periodic"

        def extended_index(global_index):
            index = global_index - coarse_start + 1
            if periodic:
                index = jnp.where(
                    (rank == 0) & (global_index == coarse.shape[1] - 1),
                    0,
                    index,
                )
                index = jnp.where(
                    (rank == self.device_count - 1) & (global_index == 0),
                    coarse_local_y + 1,
                    index,
                )
            return index

        extended = jnp.zeros(
            (coarse.shape[0], coarse_local_y + 2, coarse.shape[2]),
            dtype=field.dtype,
        )
        shape = (1, local_y, 1)
        extended = extended.at[:, extended_index(lower_index), :].add(
            weighted * lower_weight.reshape(shape)
        )
        extended = extended.at[:, extended_index(upper_index), :].add(
            weighted * upper_weight.reshape(shape)
        )
        interior = extended[:, 1:-1, :]
        left_ghost = extended[:, 0, :]
        right_ghost = extended[:, -1, :]

        if self.device_count == 1:
            if periodic:
                interior = interior.at[:, -1, :].add(left_ghost)
                interior = interior.at[:, 0, :].add(right_ghost)
        else:
            if periodic:
                send_right = [
                    (source, (source + 1) % self.device_count)
                    for source in range(self.device_count)
                ]
                send_left = [
                    (source, (source - 1) % self.device_count)
                    for source in range(self.device_count)
                ]
            else:
                send_right = [
                    (source, source + 1)
                    for source in range(self.device_count - 1)
                ]
                send_left = [
                    (source, source - 1)
                    for source in range(1, self.device_count)
                ]
            from_left = lax.ppermute(
                right_ghost,
                self.distribution.axis_name,
                send_right,
            )
            from_right = lax.ppermute(
                left_ghost,
                self.distribution.axis_name,
                send_left,
            )
            interior = interior.at[:, 0, :].add(from_left)
            interior = interior.at[:, -1, :].add(from_right)
        return interior / coarse_volume

    def _prolong_local(self, field: Array, level: int) -> Array:
        transfer = self.serial_gmg.transfers[level]
        fine_global_y = self.level_shapes[level][1]
        coarse_global_y = self.level_shapes[level + 1][1]
        fine_local_y = fine_global_y // self.device_count
        coarse_local_y = coarse_global_y // self.device_count
        rank = self._rank()
        result = _interpolate_axis(field, transfer.x, -1)
        periodic = (
            self.serial_gmg.operators[level].boundaries.y_lower.kind
            == "periodic"
        )
        left, right = self._exchange_y(result, periodic)
        padded = jnp.concatenate(
            (left[:, None, :], result, right[:, None, :]),
            axis=1,
        )
        fine_start = rank * fine_local_y
        coarse_start = rank * coarse_local_y

        def local_slice(array):
            return lax.dynamic_slice_in_dim(
                array,
                fine_start,
                fine_local_y,
            )

        lower_index = local_slice(transfer.y.lower_index)
        upper_index = local_slice(transfer.y.upper_index)
        lower_weight = local_slice(transfer.y.lower_weight)
        upper_weight = local_slice(transfer.y.upper_weight)

        def local_index(global_index):
            mapped = jnp.where(
                global_index < coarse_start,
                0,
                jnp.where(
                    global_index >= coarse_start + coarse_local_y,
                    coarse_local_y + 1,
                    global_index - coarse_start + 1,
                ),
            )
            if periodic:
                mapped = jnp.where(
                    (rank == 0) & (global_index == coarse_global_y - 1),
                    0,
                    mapped,
                )
                mapped = jnp.where(
                    (rank == self.device_count - 1) & (global_index == 0),
                    coarse_local_y + 1,
                    mapped,
                )
            return mapped

        shape = (1, fine_local_y, 1)
        result = (
            jnp.take(padded, local_index(lower_index), axis=1)
            * lower_weight.reshape(shape)
            + jnp.take(padded, local_index(upper_index), axis=1)
            * upper_weight.reshape(shape)
        )
        return _interpolate_axis(result, transfer.z, -3)

    def _replicated_cycle(self, rhs: Array, level: int) -> Array:
        global_rhs = lax.all_gather(
            rhs,
            self.distribution.axis_name,
            axis=1,
            tiled=True,
        )
        error = self.serial_gmg._cycle(
            level,
            jnp.zeros_like(global_rhs),
            global_rhs,
        )
        return self._slice_y(error, level)

    def _cycle_local(self, level: int, solution: Array, rhs: Array) -> Array:
        if level == self.replication_level:
            return self._replicated_cycle(rhs, level)
        config = self.serial_gmg.config
        rhs = self._project_local(rhs, level)
        solution = self._smooth_local(
            level,
            solution,
            rhs,
            config.pre_smooth,
        )
        residual = self._project_local(
            rhs - self._apply_local_at_level(solution, level),
            level,
        )
        coarse_rhs = self._restrict_local(residual, level)
        coarse_error = self._cycle_local(
            level + 1,
            jnp.zeros_like(coarse_rhs),
            coarse_rhs,
        )
        solution += self._prolong_local(coarse_error, level)
        return self._smooth_local(
            level,
            solution,
            rhs,
            config.post_smooth,
        )

    def _precondition_local(self, rhs: Array) -> Array:
        result = self._cycle_local(0, jnp.zeros_like(rhs), rhs)
        return self._project_local(result, 0)

    def _prepare_rhs_local(self, physical_rhs: Array) -> tuple[Array, Array]:
        effective = physical_rhs + self._slice_y(
            self.operator.boundary_rhs(),
            0,
        )
        shift = jnp.asarray(0.0, dtype=effective.dtype)
        if self.operator.has_constant_nullspace:
            volume = self._slice_y(self.operator.volume, 0)
            total = lax.psum(
                jnp.sum(volume * effective),
                self.distribution.axis_name,
            )
            shift = total * self.operator.inverse_total_volume
            effective -= shift
        return effective, shift

    def _device_pcg_local(
        self,
        physical_rhs: Array,
        initial: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Run one complete PCG solve inside the global pmap program.

        Keeping the Krylov loop on device is particularly important for a
        multi-process CPU pmap: a Python loop otherwise performs several host
        synchronizations and launches several cross-rank collectives for every
        Krylov iteration.
        """
        if not isinstance(self.krylov, PCGConfig):
            raise TypeError("device y-slab PCG requires a PCG configuration")

        rhs, compatibility_shift = self._prepare_rhs_local(physical_rhs)

        def inner(left: Array, right: Array) -> Array:
            return self._inner_finest_local(left, right)

        def norm(value: Array) -> Array:
            return jnp.sqrt(jnp.maximum(inner(value, value), 0.0))

        solution = self._project_finest_local(initial)
        rhs_norm = norm(rhs)
        target = jnp.maximum(
            self.krylov.absolute_tolerance,
            self.krylov.relative_tolerance * rhs_norm,
        )
        residual = rhs - self._apply_local(solution)
        residual_norm = norm(residual)
        preconditioned = self._precondition_local(residual)
        residual_dot_preconditioned = inner(residual, preconditioned)
        active = (
            jnp.isfinite(residual_norm)
            & (residual_norm > target)
            & jnp.isfinite(residual_dot_preconditioned)
            & (residual_dot_preconditioned > 0.0)
        )
        iteration = jnp.asarray(0, dtype=jnp.int32)

        def condition(state) -> Array:
            current_iteration = state[0]
            current_active = state[-1]
            return (
                current_iteration < self.krylov.max_iterations
            ) & current_active

        def body(state):
            (
                current_iteration,
                current_solution,
                current_residual,
                direction,
                current_dot,
                _,
                _,
            ) = state
            action = self._apply_local(direction)
            curvature = inner(direction, action)
            valid_curvature = jnp.isfinite(curvature) & (curvature > 0.0)
            safe_curvature = jnp.where(valid_curvature, curvature, 1.0)
            step = jnp.where(
                valid_curvature,
                current_dot / safe_curvature,
                0.0,
            )
            next_solution = current_solution + step * direction
            next_residual = current_residual - step * action
            next_residual_norm = norm(next_residual)
            next_preconditioned = self._precondition_local(next_residual)
            next_dot = inner(next_residual, next_preconditioned)
            valid_next = (
                valid_curvature
                & jnp.isfinite(next_residual_norm)
                & jnp.isfinite(next_dot)
                & (next_dot > 0.0)
            )
            safe_dot = jnp.where(current_dot > 0.0, current_dot, 1.0)
            direction = (
                next_preconditioned
                + (next_dot / safe_dot) * direction
            )
            next_active = valid_next & (next_residual_norm > target)
            return (
                current_iteration + 1,
                next_solution,
                next_residual,
                direction,
                next_dot,
                next_residual_norm,
                next_active,
            )

        state = lax.while_loop(
            condition,
            body,
            (
                iteration,
                solution,
                residual,
                preconditioned,
                residual_dot_preconditioned,
                residual_norm,
                active,
            ),
        )
        iterations, solution, _, _, _, residual_norm, active = state
        solution = self._project_finest_local(solution)
        converged = (~active) & jnp.isfinite(residual_norm) & (
            residual_norm <= target
        )
        return (
            solution,
            compatibility_shift,
            iterations,
            residual_norm,
            rhs_norm,
            converged,
        )

    def _check_sharded_shape(self, field: Array) -> None:
        nz, ny, nx = self.operator.shape
        expected = (
            self.local_device_count,
            nz,
            ny // self.device_count,
            nx,
        )
        if tuple(field.shape) != expected:
            raise ValueError(f"expected y-slab shape {expected}, got {field.shape}")

    def apply(self, pressure: Array) -> Array:
        self._check_sharded_shape(pressure)
        return self._mapped_apply(pressure)

    def precondition(self, rhs: Array) -> Array:
        self._check_sharded_shape(rhs)
        return self._mapped_precondition(rhs)

    def project_nullspace(self, field: Array) -> Array:
        self._check_sharded_shape(field)
        return self._mapped_project(field)

    def inner(self, left: Array, right: Array) -> Array:
        self._check_sharded_shape(left)
        self._check_sharded_shape(right)
        return self._mapped_inner(left, right)[0]

    def solve(
        self,
        physical_rhs: Array,
        *,
        initial: Array | None = None,
    ) -> FGMRESResult | PCGResult:
        self._check_sharded_shape(physical_rhs)
        if self.krylov.execution == "jax":
            if self._mapped_device_solve is None:
                raise RuntimeError("device y-slab solver was not initialized")
            starting_value = (
                jnp.zeros_like(physical_rhs)
                if initial is None
                else jnp.asarray(initial, dtype=self.operator.dtype)
            )
            self._check_sharded_shape(starting_value)
            (
                solution,
                shifts,
                iterations,
                residual_norms,
                rhs_norms,
                converged,
            ) = self._mapped_device_solve(physical_rhs, starting_value)
            residual_norm = float(residual_norms[0])
            rhs_norm = float(rhs_norms[0])
            relative_residual = (
                0.0 if rhs_norm == 0.0 else residual_norm / rhs_norm
            )
            return PCGResult(
                solution,
                bool(converged[0]),
                int(iterations[0]),
                residual_norm,
                relative_residual,
                (residual_norm,),
                float(shifts[0]),
            )
        effective_rhs, shifts = self._mapped_prepare_rhs(physical_rhs)
        solve = pcg if isinstance(self.krylov, PCGConfig) else fgmres
        result = solve(
            self.apply,
            effective_rhs,
            preconditioner=self.precondition,
            initial=initial,
            inner=self.inner,
            project=self.project_nullspace,
            config=self.krylov,
        )
        return replace(result, compatibility_shift=float(shifts[0]))

    def solve_array(
        self,
        physical_rhs: Array,
        *,
        initial: Array | None = None,
    ) -> Array:
        """Solve without transferring convergence scalars to the host."""
        if self.krylov.execution != "jax":
            return self.solve(physical_rhs, initial=initial).solution
        self._check_sharded_shape(physical_rhs)
        if self._mapped_device_solve is None:
            raise RuntimeError("device y-slab solver was not initialized")
        starting_value = (
            jnp.zeros_like(physical_rhs)
            if initial is None
            else jnp.asarray(initial, dtype=self.operator.dtype)
        )
        self._check_sharded_shape(starting_value)
        solution, *_ = self._mapped_device_solve(
            physical_rhs,
            starting_value,
        )
        return solution


__all__ = [
    "YSlabConfig",
    "YSlabMatrixFreePoissonSolver",
    "gather_y_field",
    "shard_y_field",
]
