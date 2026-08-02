"""Face-staggered MAC pressure projection on rectilinear FV grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .fgmres import FGMRESResult
from .kep4_operators import (
    divergence_axis as kep4_divergence_axis,
    gradient_axis as kep4_gradient_axis,
    validate_kep4_pressure_grid,
)
from .matrix_free_gmg import (
    MatrixFreePoissonSolver,
    PoissonBoundaryConditions,
    RectilinearGrid,
)
from .pcg import PCGResult


Array = jax.Array


class MACVelocity(NamedTuple):
    """Three face-normal velocities in canonical z-y-x storage."""

    x: Array  # (nz, ny, nx + 1)
    y: Array  # (nz, ny + 1, nx)
    z: Array  # (nz + 1, ny, nx)


@dataclass(frozen=True, slots=True)
class MACProjectionResult:
    velocity: MACVelocity
    pressure_correction: Array
    divergence_before: Array
    divergence_after: Array
    target_divergence: Array
    linear_result: FGMRESResult | PCGResult


@dataclass(frozen=True, slots=True)
class SSPRK3ProjectionResult:
    velocity: MACVelocity
    stages: tuple[MACProjectionResult, MACProjectionResult, MACProjectionResult]


class VelocityPressureProjection(NamedTuple):
    """Projected velocity and its gauge-fixed pseudo-pressure."""

    velocity: MACVelocity
    pressure: Array


def _widths(faces: tuple[float, ...], dtype: jnp.dtype) -> Array:
    values = jnp.asarray(faces, dtype=dtype)
    return values[1:] - values[:-1]


def _centers(faces: tuple[float, ...], dtype: jnp.dtype) -> Array:
    values = jnp.asarray(faces, dtype=dtype)
    return 0.5 * (values[1:] + values[:-1])


def _check_velocity_shape(velocity: MACVelocity, grid: RectilinearGrid) -> None:
    nz, ny, nx = grid.shape
    expected = (
        (nz, ny, nx + 1),
        (nz, ny + 1, nx),
        (nz + 1, ny, nx),
    )
    actual = tuple(tuple(component.shape) for component in velocity)
    if actual != expected:
        raise ValueError(f"expected MAC velocity shapes {expected}, got {actual}")


def mac_divergence(velocity: MACVelocity, grid: RectilinearGrid) -> Array:
    """Return conservative cell divergence from face-normal velocities."""
    _check_velocity_shape(velocity, grid)
    dtype = jnp.result_type(velocity.x, velocity.y, velocity.z)
    dx = _widths(grid.x_faces, dtype)
    dy = _widths(grid.y_faces, dtype)
    dz = _widths(grid.z_faces, dtype)
    return (
        (velocity.x[..., 1:] - velocity.x[..., :-1]) / dx[None, None, :]
        + (velocity.y[:, 1:, :] - velocity.y[:, :-1, :]) / dy[None, :, None]
        + (velocity.z[1:, ...] - velocity.z[:-1, ...]) / dz[:, None, None]
    )


def _gradient_axis(
    pressure: Array,
    *,
    axis: int,
    faces: tuple[float, ...],
    lower_kind: str,
    lower_value: float,
    upper_kind: str,
    upper_value: float,
) -> Array:
    dtype = pressure.dtype
    face_coordinates = jnp.asarray(faces, dtype=dtype)
    centers = 0.5 * (face_coordinates[1:] + face_coordinates[:-1])
    values = jnp.moveaxis(pressure, axis, -1)
    face_shape = values.shape[:-1] + (values.shape[-1] + 1,)
    gradient = jnp.zeros(face_shape, dtype=dtype)
    if values.shape[-1] > 1:
        gradient = gradient.at[..., 1:-1].set(
            (values[..., 1:] - values[..., :-1]) / (centers[1:] - centers[:-1])
        )

    if lower_kind == "periodic":
        distance = 0.5 * (
            face_coordinates[1]
            - face_coordinates[0]
            + face_coordinates[-1]
            - face_coordinates[-2]
        )
        periodic_gradient = (values[..., 0] - values[..., -1]) / distance
        gradient = gradient.at[..., 0].set(periodic_gradient)
        gradient = gradient.at[..., -1].set(periodic_gradient)
    else:
        if lower_kind == "dirichlet":
            lower_gradient = (values[..., 0] - lower_value) / (
                centers[0] - face_coordinates[0]
            )
        else:
            lower_gradient = jnp.full(
                values.shape[:-1],
                -lower_value,
                dtype=dtype,
            )
        if upper_kind == "dirichlet":
            upper_gradient = (upper_value - values[..., -1]) / (
                face_coordinates[-1] - centers[-1]
            )
        else:
            upper_gradient = jnp.full(
                values.shape[:-1],
                upper_value,
                dtype=dtype,
            )
        gradient = gradient.at[..., 0].set(lower_gradient)
        gradient = gradient.at[..., -1].set(upper_gradient)
    return jnp.moveaxis(gradient, -1, axis)


def mac_pressure_gradient(
    pressure: Array,
    grid: RectilinearGrid,
    boundaries: PoissonBoundaryConditions,
) -> MACVelocity:
    """Map cell pressure to the three physical face gradients."""
    if tuple(pressure.shape) != grid.shape:
        raise ValueError(
            f"expected pressure shape {grid.shape}, got {tuple(pressure.shape)}"
        )
    return MACVelocity(
        _gradient_axis(
            pressure,
            axis=-1,
            faces=grid.x_faces,
            lower_kind=boundaries.x_lower.kind,
            lower_value=boundaries.x_lower.value,
            upper_kind=boundaries.x_upper.kind,
            upper_value=boundaries.x_upper.value,
        ),
        _gradient_axis(
            pressure,
            axis=-2,
            faces=grid.y_faces,
            lower_kind=boundaries.y_lower.kind,
            lower_value=boundaries.y_lower.value,
            upper_kind=boundaries.y_upper.kind,
            upper_value=boundaries.y_upper.value,
        ),
        _gradient_axis(
            pressure,
            axis=-3,
            faces=grid.z_faces,
            lower_kind=boundaries.z_lower.kind,
            lower_value=boundaries.z_lower.value,
            upper_kind=boundaries.z_upper.kind,
            upper_value=boundaries.z_upper.value,
        ),
    )


def kep4_mac_divergence(
    velocity: MACVelocity,
    grid: RectilinearGrid,
    boundaries: PoissonBoundaryConditions,
) -> Array:
    """Return the fourth-order divergence compatible with KEP4 pressure."""
    _check_velocity_shape(velocity, grid)
    dx, dy, dz = validate_kep4_pressure_grid(grid, boundaries)
    return (
        kep4_divergence_axis(
            velocity.x,
            spacing=dx,
            axis=-1,
            lower_kind=boundaries.x_lower.kind,
            upper_kind=boundaries.x_upper.kind,
        )
        + kep4_divergence_axis(
            velocity.y,
            spacing=dy,
            axis=-2,
            lower_kind=boundaries.y_lower.kind,
            upper_kind=boundaries.y_upper.kind,
        )
        + kep4_divergence_axis(
            velocity.z,
            spacing=dz,
            axis=-3,
            lower_kind=boundaries.z_lower.kind,
            upper_kind=boundaries.z_upper.kind,
        )
    )


def kep4_mac_pressure_gradient(
    pressure: Array,
    grid: RectilinearGrid,
    boundaries: PoissonBoundaryConditions,
) -> MACVelocity:
    """Return the cell-to-face gradient whose negative transpose is D4."""
    if tuple(pressure.shape) != grid.shape:
        raise ValueError(
            f"expected pressure shape {grid.shape}, got {tuple(pressure.shape)}"
        )
    dx, dy, dz = validate_kep4_pressure_grid(grid, boundaries)
    return MACVelocity(
        kep4_gradient_axis(
            pressure,
            spacing=dx,
            axis=-1,
            lower_kind=boundaries.x_lower.kind,
            upper_kind=boundaries.x_upper.kind,
        ),
        kep4_gradient_axis(
            pressure,
            spacing=dy,
            axis=-2,
            lower_kind=boundaries.y_lower.kind,
            upper_kind=boundaries.y_upper.kind,
        ),
        kep4_gradient_axis(
            pressure,
            spacing=dz,
            axis=-3,
            lower_kind=boundaries.z_lower.kind,
            upper_kind=boundaries.z_upper.kind,
        ),
    )


def _velocity_sum(*terms: tuple[float, MACVelocity]) -> MACVelocity:
    return MACVelocity(
        sum(weight * velocity.x for weight, velocity in terms),
        sum(weight * velocity.y for weight, velocity in terms),
        sum(weight * velocity.z for weight, velocity in terms),
    )


class MACStageProjector:
    """Apply the pressure constraint to a provisional MAC stage."""

    def __init__(self, solver: MatrixFreePoissonSolver) -> None:
        self.solver = solver
        self.grid = solver.operator.grid
        self.boundaries = solver.operator.boundaries
        self.discretization = solver.discretization
        if self.discretization == "kep4":

            def divergence_operator(velocity):
                return kep4_mac_divergence(
                    velocity,
                    self.grid,
                    self.boundaries,
                )

            def gradient_operator(pressure):
                return kep4_mac_pressure_gradient(
                    pressure,
                    self.grid,
                    self.boundaries,
                )
        else:

            def divergence_operator(velocity):
                return mac_divergence(velocity, self.grid)

            def gradient_operator(pressure):
                return mac_pressure_gradient(
                    pressure,
                    self.grid,
                    self.boundaries,
                )

        self._divergence_operator = divergence_operator
        self._gradient_operator = gradient_operator

        def project_velocity_kernel(
            velocity: MACVelocity,
            timestep: Array,
            target_divergence: Array,
        ) -> MACVelocity:
            divergence = divergence_operator(velocity)
            target = jnp.broadcast_to(
                target_divergence,
                self.grid.shape,
            )
            pressure = self.solver.solve_array((target - divergence) / timestep)
            gradient = gradient_operator(pressure)
            return _velocity_sum(
                (1.0, velocity),
                (-timestep, gradient),
            )

        def project_velocity_pressure_kernel(
            velocity: MACVelocity,
            timestep: Array,
            target_divergence: Array,
            initial_pressure: Array,
        ) -> VelocityPressureProjection:
            divergence = divergence_operator(velocity)
            target = jnp.broadcast_to(
                target_divergence,
                self.grid.shape,
            )
            pressure = self.solver.solve_array(
                (target - divergence) / timestep,
                initial=initial_pressure,
            )
            gradient = gradient_operator(pressure)
            return VelocityPressureProjection(
                _velocity_sum(
                    (1.0, velocity),
                    (-timestep, gradient),
                ),
                pressure,
            )

        self._project_velocity_kernel = jax.jit(project_velocity_kernel)
        self._project_velocity_pressure_kernel = jax.jit(
            project_velocity_pressure_kernel
        )
        self._pressure_gradient_kernel = jax.jit(gradient_operator)
        self._divergence_kernel = jax.jit(divergence_operator)

    def divergence(self, velocity: MACVelocity) -> Array:
        """Return the constraint divergence used by this projector."""
        _check_velocity_shape(velocity, self.grid)
        return self._divergence_kernel(velocity)

    def pressure_gradient(self, pressure: Array) -> MACVelocity:
        """Return a compiled pressure gradient for predicted-stage methods."""
        if tuple(pressure.shape) != self.grid.shape:
            raise ValueError("pressure shape does not match the projector grid")
        return self._pressure_gradient_kernel(pressure)

    def project_velocity(
        self,
        velocity: MACVelocity,
        *,
        timestep: float,
        target_divergence: Array | float = 0.0,
    ) -> MACVelocity:
        """Project a stage without materializing host-side diagnostics."""
        if self.solver.krylov.execution != "jax":
            return self.project(
                velocity,
                timestep=timestep,
                target_divergence=target_divergence,
            ).velocity
        return self._project_velocity_kernel(
            velocity,
            jnp.asarray(timestep, dtype=velocity.x.dtype),
            jnp.asarray(target_divergence, dtype=velocity.x.dtype),
        )

    def project_velocity_and_pressure(
        self,
        velocity: MACVelocity,
        *,
        timestep: float,
        target_divergence: Array | float = 0.0,
        initial_pressure: Array | None = None,
    ) -> VelocityPressureProjection:
        """Project without host synchronization and retain the pseudo-pressure."""
        if self.solver.krylov.execution != "jax":
            result = self.project(
                velocity,
                timestep=timestep,
                target_divergence=target_divergence,
                initial_pressure=initial_pressure,
            )
            return VelocityPressureProjection(
                result.velocity,
                result.pressure_correction,
            )
        starting_pressure = (
            jnp.zeros(self.grid.shape, dtype=velocity.x.dtype)
            if initial_pressure is None
            else jnp.asarray(initial_pressure, dtype=velocity.x.dtype)
        )
        return self._project_velocity_pressure_kernel(
            velocity,
            jnp.asarray(timestep, dtype=velocity.x.dtype),
            jnp.asarray(target_divergence, dtype=velocity.x.dtype),
            starting_pressure,
        )

    def project(
        self,
        velocity: MACVelocity,
        *,
        timestep: float,
        target_divergence: Array | float = 0.0,
        initial_pressure: Array | None = None,
    ) -> MACProjectionResult:
        if not math_is_positive_finite(timestep):
            raise ValueError("projection timestep must be positive and finite")
        divergence_before = self._divergence_operator(velocity)
        target = jnp.broadcast_to(
            jnp.asarray(target_divergence, dtype=divergence_before.dtype),
            self.grid.shape,
        )
        physical_rhs = (target - divergence_before) / timestep
        linear_result = self.solver.solve(
            physical_rhs,
            initial=initial_pressure,
        )
        gradient = self._gradient_operator(linear_result.solution)
        corrected = _velocity_sum(
            (1.0, velocity),
            (-timestep, gradient),
        )
        divergence_after = self._divergence_operator(corrected)
        return MACProjectionResult(
            corrected,
            linear_result.solution,
            divergence_before,
            divergence_after,
            target,
            linear_result,
        )


def math_is_positive_finite(value: float) -> bool:
    return value > 0.0 and value < float("inf")


def projected_ssprk3_step(
    initial: MACVelocity,
    *,
    tendency: Callable[[MACVelocity, float], MACVelocity],
    projector: MACStageProjector,
    timestep: float,
    time: float = 0.0,
    target_divergence: Callable[[float], Array | float] | None = None,
) -> SSPRK3ProjectionResult:
    """Advance one SSPRK3 step and project after every explicit stage."""
    target = target_divergence or (lambda _: 0.0)

    first_tendency = tendency(initial, time)
    first_provisional = _velocity_sum(
        (1.0, initial),
        (timestep, first_tendency),
    )
    first = projector.project(
        first_provisional,
        timestep=timestep,
        target_divergence=target(time + timestep),
    )

    second_tendency = tendency(first.velocity, time + timestep)
    second_provisional = _velocity_sum(
        (0.75, initial),
        (0.25, first.velocity),
        (0.25 * timestep, second_tendency),
    )
    second = projector.project(
        second_provisional,
        timestep=0.25 * timestep,
        target_divergence=target(time + 0.5 * timestep),
    )

    third_tendency = tendency(second.velocity, time + 0.5 * timestep)
    third_provisional = _velocity_sum(
        (1.0 / 3.0, initial),
        (2.0 / 3.0, second.velocity),
        (2.0 * timestep / 3.0, third_tendency),
    )
    third = projector.project(
        third_provisional,
        timestep=2.0 * timestep / 3.0,
        target_divergence=target(time + timestep),
    )
    return SSPRK3ProjectionResult(third.velocity, (first, second, third))


def projected_ssprk3_velocity_step(
    initial: MACVelocity,
    *,
    tendency: Callable[[MACVelocity, float], MACVelocity],
    projector: MACStageProjector,
    timestep: float,
    time: float = 0.0,
) -> MACVelocity:
    """Fast SSPRK3 path that avoids stage diagnostic synchronization."""
    first_tendency = tendency(initial, time)
    first = projector.project_velocity(
        _velocity_sum(
            (1.0, initial),
            (timestep, first_tendency),
        ),
        timestep=timestep,
    )
    second_tendency = tendency(first, time + timestep)
    second = projector.project_velocity(
        _velocity_sum(
            (0.75, initial),
            (0.25, first),
            (0.25 * timestep, second_tendency),
        ),
        timestep=0.25 * timestep,
    )
    third_tendency = tendency(second, time + 0.5 * timestep)
    return projector.project_velocity(
        _velocity_sum(
            (1.0 / 3.0, initial),
            (2.0 / 3.0, second),
            (2.0 * timestep / 3.0, third_tendency),
        ),
        timestep=2.0 * timestep / 3.0,
    )


def projected_ssprk3_velocity_pressure_step(
    initial: MACVelocity,
    *,
    tendency: Callable[[MACVelocity, float], MACVelocity],
    projector: MACStageProjector,
    timestep: float,
    time: float = 0.0,
    initial_pressure: Array | None = None,
) -> VelocityPressureProjection:
    """Full three-PPE SSPRK3 in Butcher form, retaining the final pressure.

    This is the startup and reference path for FPJ-2.  Writing SSPRK3 in
    Butcher form makes the stage abscissae explicit: ``c=(0, 1, 1/2)``.
    """
    first_tendency = tendency(initial, time)
    second = projector.project_velocity_and_pressure(
        _velocity_sum(
            (1.0, initial),
            (timestep, first_tendency),
        ),
        timestep=timestep,
        initial_pressure=initial_pressure,
    )
    second_tendency = tendency(second.velocity, time + timestep)
    third = projector.project_velocity_and_pressure(
        _velocity_sum(
            (1.0, initial),
            (0.25 * timestep, first_tendency),
            (0.25 * timestep, second_tendency),
        ),
        timestep=0.5 * timestep,
        initial_pressure=second.pressure,
    )
    third_tendency = tendency(third.velocity, time + 0.5 * timestep)
    return projector.project_velocity_and_pressure(
        _velocity_sum(
            (1.0, initial),
            (timestep / 6.0, first_tendency),
            (timestep / 6.0, second_tendency),
            (2.0 * timestep / 3.0, third_tendency),
        ),
        timestep=timestep,
        initial_pressure=third.pressure,
    )


def fpj2_pressure_prediction(
    current_pressure: Array,
    previous_pressure: Array,
    *,
    current_timestep: float,
    previous_timestep: float,
    next_timestep: float,
    stage_abscissa: float,
) -> Array:
    """Variable-step FPJ-2 extrapolation of a stage pseudo-pressure."""
    if min(current_timestep, previous_timestep, next_timestep) <= 0.0:
        raise ValueError("FPJ-2 timesteps must be positive")
    if not 0.0 <= stage_abscissa <= 1.0:
        raise ValueError("FPJ-2 stage abscissa must lie in [0, 1]")
    extrapolation = (current_timestep + stage_abscissa * next_timestep) / (
        current_timestep + previous_timestep
    )
    return current_pressure + extrapolation * (current_pressure - previous_pressure)


def fpj2_ssprk3_velocity_step(
    initial: MACVelocity,
    *,
    tendency: Callable[[MACVelocity, float], MACVelocity],
    projector: MACStageProjector,
    timestep: float,
    current_pressure: Array,
    previous_pressure: Array,
    current_timestep: float,
    previous_timestep: float,
    time: float = 0.0,
) -> VelocityPressureProjection:
    """Third-order FPJ-2 SSPRK3 with one final pressure Poisson solve."""
    second_pressure = fpj2_pressure_prediction(
        current_pressure,
        previous_pressure,
        current_timestep=current_timestep,
        previous_timestep=previous_timestep,
        next_timestep=timestep,
        stage_abscissa=1.0,
    )
    third_pressure = fpj2_pressure_prediction(
        current_pressure,
        previous_pressure,
        current_timestep=current_timestep,
        previous_timestep=previous_timestep,
        next_timestep=timestep,
        stage_abscissa=0.5,
    )
    second_gradient = projector.pressure_gradient(second_pressure)
    first_tendency = tendency(initial, time)
    second = _velocity_sum(
        (1.0, initial),
        (timestep, first_tendency),
        (-timestep, second_gradient),
    )
    second_tendency = tendency(second, time + timestep)
    third_gradient = projector.pressure_gradient(third_pressure)
    third = _velocity_sum(
        (1.0, initial),
        (0.25 * timestep, first_tendency),
        (0.25 * timestep, second_tendency),
        (-0.5 * timestep, third_gradient),
    )
    third_tendency = tendency(third, time + 0.5 * timestep)
    provisional = _velocity_sum(
        (1.0, initial),
        (timestep / 6.0, first_tendency),
        (timestep / 6.0, second_tendency),
        (2.0 * timestep / 3.0, third_tendency),
    )
    return projector.project_velocity_and_pressure(
        provisional,
        timestep=timestep,
        initial_pressure=second_pressure,
    )


__all__ = [
    "VelocityPressureProjection",
    "fpj2_pressure_prediction",
    "fpj2_ssprk3_velocity_step",
    "MACProjectionResult",
    "MACStageProjector",
    "MACVelocity",
    "SSPRK3ProjectionResult",
    "kep4_mac_divergence",
    "kep4_mac_pressure_gradient",
    "mac_divergence",
    "mac_pressure_gradient",
    "projected_ssprk3_step",
    "projected_ssprk3_velocity_pressure_step",
    "projected_ssprk3_velocity_step",
]
