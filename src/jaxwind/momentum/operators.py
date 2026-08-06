"""Reusable momentum and scalar operators on rectilinear MAC grids.

Every axis carries its own :class:`~jaxwind.momentum.metrics.AxisMetric`, so
clustering is independent per direction and an axis that is uniform keeps the
constant-spacing kernels unchanged.  Spacing enters through three places only:
gradients use :meth:`AxisMetric.derivative`, divergences of a modeled flux use
:meth:`AxisMetric.negative_derivative_transpose` so the SGS operator stays
dissipative, and face reconstructions use :meth:`AxisMetric.interface_states`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp

from jaxwind.pressure import (
    _velocity_sum,
    MACStageProjector,
    MACVelocity,
    MatrixFreePoissonSolver,
    RectilinearGrid,
    VelocityPressureProjection,
    mac_divergence,
    projected_ssprk3_velocity_pressure_step,
)

from .lasd import LASDModel, LASDState, MultilevelLASD
from .metrics import (
    AxisMetric,
    reconstruction_dissipation,
    reconstruction_flux,
)
from .physical_filter import physical_top_hat_filter
from .surface_layer import NeutralLogWallLaw, SurfaceLayerFluxes


Array = jax.Array


@dataclass(frozen=True, slots=True)
class AMDModel:
    """Filter-free anisotropic minimum-dissipation eddy viscosity."""

    coefficient: float = 0.212
    molecular_viscosity: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise ValueError("AMD coefficient must be finite and nonnegative")
        if (
            not math.isfinite(self.molecular_viscosity)
            or self.molecular_viscosity < 0.0
        ):
            raise ValueError("molecular viscosity must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ScalarConfig:
    """Conservative scalar AMD controls."""

    coefficient: float = 0.212
    molecular_diffusivity: float = 0.0
    lower_surface_flux: float = 1.0e-3
    upper_surface_flux: float = 0.0
    mp5_dissipation_strength: float = 1.0

    def __post_init__(self) -> None:
        nonnegative = {
            "AMD scalar coefficient": self.coefficient,
            "molecular diffusivity": self.molecular_diffusivity,
            "scalar MP5 strength": self.mp5_dissipation_strength,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("lower scalar flux", self.lower_surface_flux),
            ("upper scalar flux", self.upper_surface_flux),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class MomentumConfig:
    """Physical and temporal controls for ABL momentum."""

    friction_velocity: float = 0.1
    roughness_length: float = 1.0e-3
    von_karman: float = 0.4
    wall_filter_width: float | None = None
    wall_temporal_filter_timescale: float | None = None
    mp5_dissipation_strength: float = 1.0
    pressure_acceleration: float | None = None
    geostrophic_wind: tuple[float, float] | None = None
    coriolis_vertical: float = 0.0
    coriolis_horizontal: float = 0.0
    amd: AMDModel = AMDModel()
    lasd: LASDModel | None = None
    sgs_time_integration: Literal["explicit", "imex_ark3"] = "explicit"

    def __post_init__(self) -> None:
        if self.friction_velocity <= 0.0:
            raise ValueError("friction velocity must be positive")
        if self.roughness_length <= 0.0:
            raise ValueError("roughness length must be positive")
        if self.von_karman <= 0.0:
            raise ValueError("von Karman constant must be positive")
        if self.wall_filter_width is not None and (
            not math.isfinite(self.wall_filter_width) or self.wall_filter_width <= 0.0
        ):
            raise ValueError("wall filter width must be positive and finite")
        if self.wall_temporal_filter_timescale is not None and (
            not math.isfinite(self.wall_temporal_filter_timescale)
            or self.wall_temporal_filter_timescale <= 0.0
        ):
            raise ValueError(
                "wall temporal filter timescale must be positive and finite"
            )
        if (
            not math.isfinite(self.mp5_dissipation_strength)
            or self.mp5_dissipation_strength < 0.0
        ):
            raise ValueError("MP5 dissipation strength must be finite and nonnegative")
        if self.pressure_acceleration is not None and not math.isfinite(
            self.pressure_acceleration
        ):
            raise ValueError("pressure acceleration must be finite")
        if self.geostrophic_wind is not None and not all(
            math.isfinite(component) for component in self.geostrophic_wind
        ):
            raise ValueError("geostrophic wind components must be finite")
        if not math.isfinite(self.coriolis_vertical):
            raise ValueError("vertical Coriolis parameter must be finite")
        if not math.isfinite(self.coriolis_horizontal):
            raise ValueError("horizontal Coriolis parameter must be finite")
        if self.sgs_time_integration not in {"explicit", "imex_ark3"}:
            raise ValueError("SGS time integration must be 'explicit' or 'imex_ark3'")


class WallModelState(NamedTuple):
    """Accepted-step memory for the temporally filtered wall input."""

    filtered_velocity: Array


class PreparedIMEXStep(NamedTuple):
    """Reusable work prepared from one accepted neutral IMEX state."""

    initial_explicit: MACVelocity
    initial_implicit: MACVelocity
    frozen_viscosity: Array
    lasd_coefficient: Array
    wall_velocity: Array


@dataclass(frozen=True, slots=True)
class MomentumDiagnostic:
    time: float
    kinetic_energy: float
    maximum_cfl: float
    maximum_diffusive_cfl: float
    divergence_norm: float
    mean_wall_ustar: float
    mean_sgs_viscosity: float
    maximum_sgs_viscosity: float
    mean_sgs_coefficient: float
    maximum_sgs_coefficient: float
    clipped_sgs_coefficient_fraction: float

    @property
    def mean_amd_viscosity(self) -> float:
        """Backward-compatible name for AMD-specific callers."""
        return self.mean_sgs_viscosity

    @property
    def maximum_amd_viscosity(self) -> float:
        """Backward-compatible name for AMD-specific callers."""
        return self.maximum_sgs_viscosity


def _build_axis_metrics(
    grid: RectilinearGrid,
    dtype,
) -> tuple[AxisMetric, AxisMetric, AxisMetric]:
    """Return the ``(x, y, z)`` metrics of a rectilinear ABL grid.

    The horizontal axes are periodic and carry the five-point fourth-order
    centred stencil; the wall-normal axis is bounded and carries the three-point
    stencil the wall model and the vertical line solves are built around.
    """

    return (
        AxisMetric(grid.x_faces, axis=2, periodic=True, dtype=dtype),
        AxisMetric(grid.y_faces, axis=1, periodic=True, dtype=dtype),
        AxisMetric(
            grid.z_faces, axis=0, periodic=False, dtype=dtype, derivative_width=3
        ),
    )


def _cell_length_scales(
    metrics: tuple[AxisMetric, AxisMetric, AxisMetric],
    dtype=None,
) -> Array:
    """Return the local ``(dx, dy, dz)`` triple of every cell.

    The trailing axis carries the three directions, and leading axes stay
    singleton wherever the corresponding coordinate is uniform so the closure
    length scale never materializes a field-sized constant.
    """

    x_metric, y_metric, z_metric = metrics
    if x_metric.uniform and y_metric.uniform:
        stacked = jnp.stack(
            (
                jnp.full_like(z_metric.widths, x_metric.spacing),
                jnp.full_like(z_metric.widths, y_metric.spacing),
                z_metric.widths,
            ),
            axis=-1,
        )[:, None, None, :]
    else:
        stacked = jnp.stack(
            jnp.broadcast_arrays(
                x_metric.broadcast(x_metric.widths, 3),
                y_metric.broadcast(y_metric.widths, 3),
                z_metric.broadcast(z_metric.widths, 3),
            ),
            axis=-1,
        )
    if dtype is not None and stacked.dtype != dtype:
        return stacked.astype(dtype)
    return stacked


def _horizontal_mean(
    field: Array,
    x_metric: AxisMetric,
    y_metric: AxisMetric,
    *,
    keepdims: bool = False,
) -> Array:
    """Area-average a ``(z,y,x,...)`` field without changing uniform results."""

    if x_metric.uniform and y_metric.uniform:
        return jnp.mean(field, axis=(1, 2), keepdims=keepdims)
    trailing = (1,) * (field.ndim - 3)
    area = (y_metric.widths[:, None] * x_metric.widths[None, :]).astype(
        field.dtype
    )
    weights = jnp.reshape(area, (1, *area.shape, *trailing))
    mean = jnp.sum(field * weights, axis=(1, 2), keepdims=keepdims)
    return mean / jnp.asarray(x_metric.length * y_metric.length, field.dtype)


def _surface_mean(field: Array, x_metric: AxisMetric, y_metric: AxisMetric) -> Array:
    """Area-average a ``(y,x,...)`` surface field."""

    if x_metric.uniform and y_metric.uniform:
        return jnp.mean(field, axis=(0, 1))
    trailing = (1,) * (field.ndim - 2)
    area = (y_metric.widths[:, None] * x_metric.widths[None, :]).astype(
        field.dtype
    )
    weights = jnp.reshape(area, (*area.shape, *trailing))
    return jnp.sum(field * weights, axis=(0, 1)) / jnp.asarray(
        x_metric.length * y_metric.length,
        field.dtype,
    )


def _volume_mean(
    field: Array,
    metrics: tuple[AxisMetric, AxisMetric, AxisMetric],
) -> Array:
    """Volume-average a cell scalar, retaining the uniform fast path."""

    x_metric, y_metric, z_metric = metrics
    if x_metric.uniform and y_metric.uniform and z_metric.uniform:
        return jnp.mean(field)
    plane_mean = _horizontal_mean(field, x_metric, y_metric)
    normalized_height = z_metric.widths.astype(field.dtype) / jnp.asarray(
        z_metric.length,
        field.dtype,
    )
    return jnp.sum(plane_mean * normalized_height)


def _z_vector(values: Array, ndim: int, dtype=None) -> Array:
    """Reshape one z vector for broadcasting over a z-first field.

    Grid geometry is stored at the precision the grid was built with, which need
    not be the precision a caller integrates in.  Passing the field dtype keeps
    the wider one from promoting whole fields on the way into a scatter.
    """

    reshaped = jnp.reshape(values, (values.shape[0],) + (1,) * (ndim - 1))
    if dtype is not None and reshaped.dtype != dtype:
        return reshaped.astype(dtype)
    return reshaped


def _interpolate_to_vertical_faces(
    values: Array,
    centers: Array,
    faces: Array,
) -> Array:
    """Linearly interpolate cell values to interior physical z faces."""

    center_distance = centers[1:] - centers[:-1]
    lower_weight = (centers[1:] - faces[1:-1]) / center_distance
    upper_weight = (faces[1:-1] - centers[:-1]) / center_distance
    shape = (center_distance.shape[0],) + (1,) * (values.ndim - 1)
    lower_weight = jnp.reshape(lower_weight, shape).astype(values.dtype)
    upper_weight = jnp.reshape(upper_weight, shape).astype(values.dtype)
    return lower_weight * values[:-1] + upper_weight * values[1:]


def _interpolate_to_periodic_faces(
    values: Array,
    centers: Array,
    faces: Array,
    *,
    axis: int,
) -> Array:
    """Linearly interpolate periodic cell values to physical faces.

    Unlike an arithmetic mean, the weights account for the two adjacent cell
    widths.  The duplicated first/last MAC face receives the same interpolation
    across the unwrapped periodic seam.
    """

    axis %= values.ndim
    moved = jnp.moveaxis(values, axis, -1)
    center_distance = centers[1:] - centers[:-1]
    lower_weight = (centers[1:] - faces[1:-1]) / center_distance
    upper_weight = (faces[1:-1] - centers[:-1]) / center_distance
    weight_shape = (1,) * (moved.ndim - 1) + (center_distance.shape[0],)
    interior = (
        jnp.reshape(lower_weight, weight_shape).astype(values.dtype)
        * moved[..., :-1]
        + jnp.reshape(upper_weight, weight_shape).astype(values.dtype)
        * moved[..., 1:]
    )

    length = faces[-1] - faces[0]
    left_center = centers[-1] - length
    right_center = centers[0]
    seam_distance = right_center - left_center
    seam = (
        ((right_center - faces[0]) / seam_distance).astype(values.dtype)
        * moved[..., -1]
        + ((faces[0] - left_center) / seam_distance).astype(values.dtype)
        * moved[..., 0]
    )
    result = jnp.concatenate(
        (seam[..., None], interior, seam[..., None]),
        axis=-1,
    )
    return jnp.moveaxis(result, -1, axis)


def _cell_velocity(velocity: MACVelocity) -> Array:
    return jnp.stack(
        (
            0.5 * (velocity.x[..., 1:] + velocity.x[..., :-1]),
            0.5 * (velocity.y[:, 1:, :] + velocity.y[:, :-1, :]),
            0.5 * (velocity.z[1:, ...] + velocity.z[:-1, ...]),
        ),
        axis=-1,
    )


def _cells_to_faces(tendency: Array) -> MACVelocity:
    nz, ny, nx, _ = tendency.shape
    x = jnp.zeros((nz, ny, nx + 1), dtype=tendency.dtype)
    x = x.at[..., 1:-1].set(0.5 * (tendency[..., :-1, 0] + tendency[..., 1:, 0]))
    x_boundary = 0.5 * (tendency[..., -1, 0] + tendency[..., 0, 0])
    x = x.at[..., 0].set(x_boundary)
    x = x.at[..., -1].set(x_boundary)

    y = jnp.zeros((nz, ny + 1, nx), dtype=tendency.dtype)
    y = y.at[:, 1:-1, :].set(0.5 * (tendency[:, :-1, :, 1] + tendency[:, 1:, :, 1]))
    y_boundary = 0.5 * (tendency[:, -1, :, 1] + tendency[:, 0, :, 1])
    y = y.at[:, 0, :].set(y_boundary)
    y = y.at[:, -1, :].set(y_boundary)

    z = jnp.zeros((nz + 1, ny, nx), dtype=tendency.dtype)
    if nz > 1:
        z = z.at[1:-1].set(0.5 * (tendency[:-1, ..., 2] + tendency[1:, ..., 2]))
    return MACVelocity(x, y, z)


# Ascher-Ruuth-Spiteri ARS(2,3,3).  This third-order pair needs the same
# three explicit tendency evaluations as SSPRK3 and two vertical line solves.
_ARK3_GAMMA = (3.0 + math.sqrt(3.0)) / 6.0
_ARK3_C = (0.0, _ARK3_GAMMA, 1.0 - _ARK3_GAMMA)
_ARK3_EXPLICIT_A = (
    (0.0, 0.0, 0.0),
    (_ARK3_GAMMA, 0.0, 0.0),
    (
        _ARK3_GAMMA - 1.0,
        2.0 - 2.0 * _ARK3_GAMMA,
        0.0,
    ),
)
_ARK3_IMPLICIT_A = (
    (0.0, 0.0, 0.0),
    (0.0, _ARK3_GAMMA, 0.0),
    (
        0.0,
        1.0 - 2.0 * _ARK3_GAMMA,
        _ARK3_GAMMA,
    ),
)
_ARK3_EXPLICIT_B = (0.0, 0.5, 0.5)
_ARK3_IMPLICIT_B = (0.0, 0.5, 0.5)


class MomentumOperators:
    """ABL momentum operators with projected explicit and IMEX time steps."""

    def __init__(
        self,
        grid: RectilinearGrid,
        pressure_solver: MatrixFreePoissonSolver,
        config: MomentumConfig = MomentumConfig(),
    ) -> None:
        if pressure_solver.operator.grid != grid:
            raise ValueError("pressure and momentum grids must match")
        if grid.shape[0] < 2 or min(grid.shape[1:]) < 4:
            raise ValueError("momentum operators require nz>=2 and nx,ny>=4")
        self.grid = grid
        self.pressure_solver = pressure_solver
        self.projector = MACStageProjector(pressure_solver)
        self.config = config
        dtype = pressure_solver.operator.dtype
        self.x_metric, self.y_metric, self.z_metric = _build_axis_metrics(grid, dtype)
        self.metrics = (self.x_metric, self.y_metric, self.z_metric)
        self.uniform_axes = tuple(metric.uniform for metric in self.metrics)
        self.uniform_z = self.z_metric.uniform
        # Nominal reference lengths.  These are the exact spacings of a uniform
        # axis and its mean otherwise; the operators use the metrics.
        self.dx = self.x_metric.spacing
        self.dy = self.y_metric.spacing
        self.dz = self.z_metric.spacing
        self.z_faces = self.z_metric.faces
        self.z_centers = self.z_metric.centers
        self.dz_cell = self.z_metric.widths
        self.dz_center = self.z_metric.center_gaps
        if config.lasd is not None and not all(self.uniform_axes):
            raise ValueError(
                "stretched grids support the AMD closure, not LASD: the LASD test"
                " filter and its Lagrangian trajectory advection are both defined"
                " on constant spacing"
            )

        self._wall_cell_height = float(grid.z_faces[1] - grid.z_faces[0])
        self.height = grid.z_faces[-1] - grid.z_faces[0]
        self.wall_law = NeutralLogWallLaw(
            config.roughness_length,
            config.von_karman,
        )
        self.wall_law.cell_average_log_denominator(self._wall_cell_height)
        self.pressure_acceleration = (
            config.pressure_acceleration
            if config.pressure_acceleration is not None
            else (
                0.0
                if config.geostrophic_wind is not None
                else config.friction_velocity**2 / self.height
            )
        )
        self.lasd_closure = (
            None
            if config.lasd is None
            else MultilevelLASD(
                multigrid=pressure_solver.preconditioner,
                model=config.lasd,
            )
        )
        self._lasd_state: LASDState | None = None
        self._lasd_step = 0
        self._lasd_interval_time = 0.0
        self._pressure = jnp.zeros(
            grid.shape,
            dtype=self.pressure_solver.operator.dtype,
        )
        self._wall_model_state: WallModelState | None = None

        def compiled_tendency(
            velocity: MACVelocity,
            lasd_coefficient: Array,
            wall_velocity: Array,
        ) -> MACVelocity:
            if self.x_metric.uniform and self.y_metric.uniform:
                return _cells_to_faces(
                    self.cell_tendency(
                        velocity,
                        lasd_coefficient,
                        wall_velocity=wall_velocity,
                    )
                )
            cells = _cell_velocity(velocity)
            wall_stress = self.wall_stress(
                cells,
                wall_velocity=wall_velocity,
            )
            return self._add_wall_stress_face_tendency(
                _cells_to_faces(
                    self.cell_tendency(
                        velocity,
                        lasd_coefficient,
                        cell_velocity=cells,
                        wall_stress=jnp.zeros_like(wall_stress),
                    )
                ),
                wall_stress,
            )

        self._compiled_tendency = jax.jit(compiled_tendency)

        def compiled_tendency_with_wall_stress(
            velocity: MACVelocity,
            lasd_coefficient: Array,
            wall_stress: Array,
        ) -> MACVelocity:
            if self.x_metric.uniform and self.y_metric.uniform:
                return _cells_to_faces(
                    self.cell_tendency(
                        velocity,
                        lasd_coefficient,
                        wall_stress=wall_stress,
                    )
                )
            cells = _cell_velocity(velocity)
            return self._add_wall_stress_face_tendency(
                _cells_to_faces(
                    self.cell_tendency(
                        velocity,
                        lasd_coefficient,
                        cell_velocity=cells,
                        wall_stress=jnp.zeros_like(wall_stress),
                    )
                ),
                wall_stress,
            )

        self._compiled_tendency_with_wall_stress = jax.jit(
            compiled_tendency_with_wall_stress
        )

        def imex_tendencies_from_gradient(
            velocity: MACVelocity,
            cells: Array,
            gradient: Array,
            frozen_viscosity: Array,
            lasd_coefficient: Array,
            *,
            wall_velocity: Array | None = None,
            wall_stress: Array | None = None,
        ) -> tuple[MACVelocity, MACVelocity]:
            if self.x_metric.uniform and self.y_metric.uniform:
                principal, cross = self.sgs_split_tendency(
                    cells,
                    frozen_viscosity,
                    lasd_coefficient,
                    gradient=gradient,
                    wall_velocity=wall_velocity,
                    wall_stress=wall_stress,
                )
                explicit = (
                    self.conservative_advection(velocity, cells)
                    + cross
                    + self.forcing_tendency(cells)
                )
                if self.config.mp5_dissipation_strength > 0.0:
                    explicit += self.advection_dissipation(velocity, cells)
                return _cells_to_faces(explicit), _cells_to_faces(principal)

            active_wall_stress = (
                self.wall_stress(cells, wall_velocity=wall_velocity)
                if wall_stress is None
                else wall_stress
            )
            principal, cross = self.sgs_split_tendency(
                cells,
                frozen_viscosity,
                lasd_coefficient,
                gradient=gradient,
                wall_stress=jnp.zeros_like(active_wall_stress),
            )
            explicit = (
                self.conservative_advection(velocity, cells)
                + cross
                + self.forcing_tendency(cells)
            )
            if self.config.mp5_dissipation_strength > 0.0:
                explicit += self.advection_dissipation(velocity, cells)
            return (
                self._add_wall_stress_face_tendency(
                    _cells_to_faces(explicit),
                    active_wall_stress,
                ),
                _cells_to_faces(principal),
            )

        def compiled_imex_initial_tendencies(
            velocity: MACVelocity,
            lasd_coefficient: Array,
            wall_velocity: Array,
        ) -> tuple[MACVelocity, MACVelocity, Array]:
            cells = _cell_velocity(velocity)
            gradient = self.velocity_gradient(cells)
            frozen_viscosity = self._sgs_viscosity_from_gradient(
                gradient,
                lasd_coefficient,
            )
            explicit, implicit = imex_tendencies_from_gradient(
                velocity,
                cells,
                gradient,
                frozen_viscosity,
                lasd_coefficient,
                wall_velocity=wall_velocity,
            )
            return explicit, implicit, frozen_viscosity

        self._compiled_imex_initial_tendencies = jax.jit(
            compiled_imex_initial_tendencies
        )

        def compiled_imex_prepare(
            velocity: MACVelocity,
            lasd_coefficient: Array,
            wall_velocity: Array,
        ) -> tuple[MACVelocity, MACVelocity, Array, Array]:
            initial_explicit, initial_implicit, frozen_viscosity = (
                compiled_imex_initial_tendencies(
                    velocity,
                    lasd_coefficient,
                    wall_velocity,
                )
            )
            cells = _cell_velocity(velocity)
            rates = jnp.stack(
                (
                    jnp.maximum(
                        self.cfl_rate(velocity),
                        self.wall_stability_rate(cells),
                    ),
                    self.explicit_sgs_diffusion_rate(
                        frozen_viscosity,
                        include_vertical=False,
                    ),
                )
            )
            return initial_explicit, initial_implicit, frozen_viscosity, rates

        self._compiled_imex_prepare = jax.jit(compiled_imex_prepare)

        def compiled_imex_tendencies(
            velocity: MACVelocity,
            frozen_viscosity: Array,
            lasd_coefficient: Array,
            wall_velocity: Array,
        ) -> tuple[MACVelocity, MACVelocity]:
            cells = _cell_velocity(velocity)
            gradient = self.velocity_gradient(cells)
            return imex_tendencies_from_gradient(
                velocity,
                cells,
                gradient,
                frozen_viscosity,
                lasd_coefficient,
                wall_velocity=wall_velocity,
            )

        self._compiled_imex_tendencies = jax.jit(compiled_imex_tendencies)

        def compiled_imex_initial_tendencies_with_wall_stress(
            velocity: MACVelocity,
            lasd_coefficient: Array,
            wall_stress: Array,
        ) -> tuple[MACVelocity, MACVelocity, Array]:
            cells = _cell_velocity(velocity)
            gradient = self.velocity_gradient(cells)
            frozen_viscosity = self._sgs_viscosity_from_gradient(
                gradient,
                lasd_coefficient,
            )
            explicit, implicit = imex_tendencies_from_gradient(
                velocity,
                cells,
                gradient,
                frozen_viscosity,
                lasd_coefficient,
                wall_stress=wall_stress,
            )
            return explicit, implicit, frozen_viscosity

        self._compiled_imex_initial_tendencies_with_wall_stress = jax.jit(
            compiled_imex_initial_tendencies_with_wall_stress
        )

        def compiled_imex_tendencies_with_wall_stress(
            velocity: MACVelocity,
            frozen_viscosity: Array,
            lasd_coefficient: Array,
            wall_stress: Array,
        ) -> tuple[MACVelocity, MACVelocity]:
            cells = _cell_velocity(velocity)
            gradient = self.velocity_gradient(cells)
            return imex_tendencies_from_gradient(
                velocity,
                cells,
                gradient,
                frozen_viscosity,
                lasd_coefficient,
                wall_stress=wall_stress,
            )

        self._compiled_imex_tendencies_with_wall_stress = jax.jit(
            compiled_imex_tendencies_with_wall_stress
        )

        def compiled_implicit_diffusion(
            velocity: MACVelocity,
            frozen_viscosity: Array,
            implicit_timestep: Array,
        ) -> MACVelocity:
            cells = _cell_velocity(velocity)
            solved = self.solve_vertical_sgs_diffusion(
                cells,
                frozen_viscosity,
                implicit_timestep,
            )
            correction = _cells_to_faces(solved - cells)
            return self.enforce_boundaries(
                _velocity_sum(
                    (1.0, velocity),
                    (1.0, correction),
                )
            )

        self._compiled_implicit_diffusion = jax.jit(compiled_implicit_diffusion)

        def compiled_diagnostic(
            velocity: MACVelocity,
            timestep: Array,
            lasd_coefficient: Array,
            wall_velocity: Array,
        ) -> tuple[Array, ...]:
            cells = _cell_velocity(velocity)
            viscosity = self.sgs_viscosity(cells, lasd_coefficient)
            energy = 0.5 * self.volume_mean(jnp.sum(cells * cells, axis=-1))
            divergence = mac_divergence(velocity, self.grid)
            if self.lasd_closure is None:
                mean_coefficient = jnp.asarray(
                    self.config.amd.coefficient,
                    dtype=cells.dtype,
                )
                maximum_coefficient = mean_coefficient
                clipped_fraction = jnp.asarray(0.0, dtype=cells.dtype)
            else:
                mean_coefficient = self.volume_mean(lasd_coefficient)
                maximum_coefficient = jnp.max(lasd_coefficient)
                clipped_fraction = self.volume_mean(
                    (
                        lasd_coefficient
                        >= 0.999 * self.config.lasd.maximum_coefficient
                    ).astype(cells.dtype)
                )
            return (
                energy,
                timestep
                * jnp.maximum(
                    self.cfl_rate(velocity),
                    self.wall_stability_rate(
                        cells,
                        wall_velocity=wall_velocity,
                    ),
                ),
                timestep * self.explicit_sgs_diffusion_rate(viscosity),
                self.pressure_solver.operator.norm(divergence),
                self.surface_mean(
                    self.wall_ustar(cells, wall_velocity=wall_velocity)
                ),
                self.volume_mean(viscosity),
                jnp.max(viscosity),
                mean_coefficient,
                maximum_coefficient,
                clipped_fraction,
            )

        self._compiled_diagnostic = jax.jit(compiled_diagnostic)
        self._compiled_cfl_rate = jax.jit(self.cfl_rate)

        def compiled_timestep_rates(
            velocity: MACVelocity,
            lasd_coefficient: Array,
        ) -> tuple[Array, Array]:
            cells = _cell_velocity(velocity)
            viscosity = self.sgs_viscosity(cells, lasd_coefficient)
            diffusion_rate = self.explicit_sgs_diffusion_rate(viscosity)
            return (
                jnp.maximum(
                    self.cfl_rate(velocity),
                    self.wall_stability_rate(cells),
                ),
                diffusion_rate,
            )

        self._compiled_timestep_rates = jax.jit(compiled_timestep_rates)

        def compiled_imex_timestep_rates(
            velocity: MACVelocity,
            lasd_coefficient: Array,
        ) -> tuple[Array, Array]:
            cells = _cell_velocity(velocity)
            viscosity = self.sgs_viscosity(cells, lasd_coefficient)
            explicit_horizontal_diffusion_rate = self.explicit_sgs_diffusion_rate(
                viscosity,
                include_vertical=False,
            )
            return (
                jnp.maximum(
                    self.cfl_rate(velocity),
                    self.wall_stability_rate(cells),
                ),
                explicit_horizontal_diffusion_rate,
            )

        self._compiled_imex_timestep_rates = jax.jit(compiled_imex_timestep_rates)
        if self.lasd_closure is not None:
            self._compiled_lasd_accumulate = jax.jit(
                self.lasd_closure.accumulate,
                inline=False,
            )

            def compiled_lasd_statistics(
                state: LASDState,
                cells: Array,
            ) -> tuple[LASDState, Array, Array, Array, Array]:
                accumulated = self.lasd_closure.accumulate(state, cells)
                gradient = self.velocity_gradient(cells)
                fields = self.lasd_closure.contraction_inputs(
                    cells,
                    gradient,
                )
                return (
                    accumulated,
                    *self.lasd_closure.contractions_from_inputs(fields),
                )

            def compiled_lasd_finalize(
                state: LASDState,
                lm: Array,
                mm: Array,
                qn: Array,
                nn: Array,
                interval_dt: Array,
                first_update: Array,
            ) -> LASDState:
                return self.lasd_closure.update_from_contractions(
                    state,
                    lm,
                    mm,
                    qn,
                    nn,
                    interval_dt=interval_dt,
                    first_update=first_update,
                )

            # Keep a hard executable boundary between local Germano
            # statistics and Lagrangian trajectory/history work.  This lets
            # gradient construction feed the dual-scale filter without a
            # field-sized spill, but does not build one giant timestep graph.
            self._compiled_lasd_statistics = jax.jit(
                compiled_lasd_statistics,
                inline=False,
            )
            self._compiled_lasd_finalize = jax.jit(
                compiled_lasd_finalize,
                inline=False,
            )

    def velocity_gradient(self, cell_velocity: Array) -> Array:
        derivatives = []
        for component in range(3):
            value = cell_velocity[..., component]
            derivatives.append(
                jnp.stack(
                    (
                        self.x_metric.derivative(value),
                        self.y_metric.derivative(value),
                        self.z_metric.derivative(value),
                    ),
                    axis=-1,
                )
            )
        return jnp.stack(derivatives, axis=-2)

    def _negative_derivative_transpose(
        self,
        field: Array,
        direction: int,
    ) -> Array:
        """Return minus the volume-weighted adjoint of the axis derivative.

        Applied to a modeled flux this is its energy-consistent divergence, so
        the SGS operator stays dissipative however the axis is stretched.
        """

        return self.metrics[direction].negative_derivative_transpose(field)

    def principal_sgs_tendency(
        self,
        cell_velocity: Array,
        frozen_viscosity: Array,
    ) -> Array:
        """Return conservative vertical principal SGS diffusion."""
        if cell_velocity.shape[0] == 1:
            return jnp.zeros_like(cell_velocity)
        face_viscosity = _interpolate_to_vertical_faces(
            frozen_viscosity,
            self.z_centers,
            self.z_faces,
        )
        flux = (
            face_viscosity[..., None]
            * (cell_velocity[1:] - cell_velocity[:-1])
            / _z_vector(self.dz_center, cell_velocity.ndim, cell_velocity.dtype)
        )
        result = jnp.zeros_like(cell_velocity)
        rank, dtype = cell_velocity.ndim, cell_velocity.dtype
        result = result.at[:-1].add(flux / _z_vector(self.dz_cell[:-1], rank, dtype))
        result = result.at[1:].add(-flux / _z_vector(self.dz_cell[1:], rank, dtype))
        return result

    def _vertical_stress_faces_from_cell_stress(
        self,
        vertical_cell_stress: Array,
    ) -> Array:
        """Return the exact telescoping face representation of ``-D_z^T tau``.

        The variational SGS operator stores strain and stress at cell centres.
        Its wall-normal divergence is therefore not a conventional two-point
        face stencil.  Cumulatively integrating the conservative tendency
        exposes the unique face flux with zero stress on the lower natural
        boundary.  The upper face is zero to roundoff because ``D_z 1 = 0``.
        """
        vertical_tendency = self._negative_derivative_transpose(
            vertical_cell_stress,
            2,
        )
        lower = jnp.zeros_like(vertical_tendency[:1])
        return jnp.concatenate(
            (
                lower,
                jnp.cumsum(
                    _z_vector(
                        self.dz_cell,
                        vertical_tendency.ndim,
                        vertical_tendency.dtype,
                    )
                    * vertical_tendency,
                    axis=0,
                ),
            ),
            axis=0,
        )

    def _vertical_stress_divergence(self, face_stress: Array) -> Array:
        return (face_stress[1:] - face_stress[:-1]) / _z_vector(
            self.dz_cell,
            face_stress.ndim,
            face_stress.dtype,
        )

    def variational_sgs_tendency(
        self,
        cell_velocity: Array,
        frozen_viscosity: Array,
        *,
        gradient: Array | None = None,
        wall_stress: Array | None = None,
    ) -> Array:
        """Return the energy-dissipative SGS divergence plus wall traction."""
        if gradient is None:
            gradient = self.velocity_gradient(cell_velocity)
        stress = frozen_viscosity[..., None, None] * (
            gradient + jnp.swapaxes(gradient, -1, -2)
        )
        vertical_faces = self._vertical_stress_faces_from_cell_stress(stress[..., :, 2])
        if wall_stress is not None:
            vertical_faces = vertical_faces.at[0].add(wall_stress)
        vertical = self._vertical_stress_divergence(vertical_faces)
        components = []
        for component in range(3):
            value = vertical[..., component]
            for direction in range(2):
                value += self._negative_derivative_transpose(
                    stress[..., component, direction],
                    direction,
                )
            components.append(value)
        return jnp.stack(components, axis=-1)

    def sgs_split_tendency(
        self,
        cell_velocity: Array,
        frozen_viscosity: Array,
        lasd_coefficient: Array,
        *,
        gradient: Array | None = None,
        wall_velocity: Array | None = None,
        wall_stress: Array | None = None,
    ) -> tuple[Array, Array]:
        """Split a frozen vertical reference from the full nonlinear SGS."""
        if gradient is None:
            gradient = self.velocity_gradient(cell_velocity)
        principal = self.principal_sgs_tendency(
            cell_velocity,
            frozen_viscosity,
        )
        dynamic_viscosity = self._sgs_viscosity_from_gradient(
            gradient,
            lasd_coefficient,
        )
        full = self.variational_sgs_tendency(
            cell_velocity,
            dynamic_viscosity,
            gradient=gradient,
            wall_stress=(
                self.wall_stress(
                    cell_velocity,
                    wall_velocity=wall_velocity,
                )
                if wall_stress is None
                else wall_stress
            ),
        )
        return principal, full - principal

    def solve_vertical_sgs_diffusion(
        self,
        rhs: Array,
        frozen_viscosity: Array,
        implicit_timestep: Array,
    ) -> Array:
        """Solve ``(I-dt Lz) u=rhs`` independently along every column."""
        if rhs.shape[0] == 1:
            return rhs
        face_viscosity = _interpolate_to_vertical_faces(
            frozen_viscosity,
            self.z_centers,
            self.z_faces,
        )
        lower = jnp.zeros_like(frozen_viscosity)
        upper = jnp.zeros_like(frozen_viscosity)
        lower = lower.at[1:].set(
            -implicit_timestep
            * face_viscosity
            / _z_vector(self.dz_cell[1:] * self.dz_center, lower.ndim, lower.dtype)
        )
        upper = upper.at[:-1].set(
            -implicit_timestep
            * face_viscosity
            / _z_vector(self.dz_cell[:-1] * self.dz_center, upper.ndim, upper.dtype)
        )
        diagonal = 1.0 - lower - upper

        first_upper = upper[0] / diagonal[0]
        first_rhs = rhs[0] / diagonal[0][..., None]

        def forward(carry, values):
            previous_upper, previous_rhs = carry
            lower_value, diagonal_value, upper_value, rhs_value = values
            denominator = diagonal_value - lower_value * previous_upper
            reduced_upper = upper_value / denominator
            reduced_rhs = (
                rhs_value - lower_value[..., None] * previous_rhs
            ) / denominator[..., None]
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
            (first_upper[None], upper_tail),
            axis=0,
        )
        reduced_rhs = jnp.concatenate(
            (first_rhs[None], rhs_tail),
            axis=0,
        )

        def backward(next_value, values):
            rhs_value, upper_value = values
            value = rhs_value - upper_value[..., None] * next_value
            return value, value

        _, prefix_reverse = jax.lax.scan(
            backward,
            reduced_rhs[-1],
            (
                reduced_rhs[:-1][::-1],
                reduced_upper[:-1][::-1],
            ),
        )
        return jnp.concatenate(
            (prefix_reverse[::-1], reduced_rhs[-1:]),
            axis=0,
        )

    def _amd_viscosity_from_gradient(self, gradient: Array) -> Array:
        strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
        delta = _cell_length_scales(self.metrics, gradient.dtype)
        weighted_gradient = gradient * delta[..., None, :]
        gradient_tensor = jnp.einsum(
            "...ik,...jk->...ij",
            weighted_gradient,
            weighted_gradient,
        )
        production = -jnp.einsum(
            "...ij,...ij->...",
            gradient_tensor,
            strain,
        )
        denominator = jnp.einsum("...ij,...ij->...", gradient, gradient)
        epsilon = jnp.finfo(gradient.dtype).eps
        eddy = (
            self.config.amd.coefficient
            * jnp.maximum(
                production,
                0.0,
            )
            / jnp.maximum(denominator, epsilon)
        )
        return eddy + self.config.amd.molecular_viscosity

    def _amd_stress_from_gradient(self, gradient: Array) -> Array:
        strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
        viscosity = self._amd_viscosity_from_gradient(gradient)
        return 2.0 * viscosity[..., None, None] * strain

    def amd_viscosity(self, cell_velocity: Array) -> Array:
        return self._amd_viscosity_from_gradient(self.velocity_gradient(cell_velocity))

    def conservative_advection(
        self,
        velocity: MACVelocity,
        cell_velocity: Array | None = None,
    ) -> Array:
        """Return compatible centered MAC momentum-flux divergence.

        Centering transported momentum on the normal velocity faces makes the
        finite-volume flux telescope exactly. For a projected MAC velocity,
        the same construction is also kinetic-energy neutral because its
        residual work is proportional to the cellwise MAC divergence.

        Both properties follow from the arithmetic face mean and the telescoping
        difference alone, so they survive stretching in any direction.  What a
        stretched axis costs is interpolation accuracy at its faces, which is
        the usual price of a symmetry-preserving flux on a non-uniform mesh.
        """
        cells = _cell_velocity(velocity) if cell_velocity is None else cell_velocity

        x_boundary = 0.5 * (cells[..., -1, :] + cells[..., 0, :])
        x_faces = jnp.concatenate(
            (
                x_boundary[..., None, :],
                0.5 * (cells[..., :-1, :] + cells[..., 1:, :]),
                x_boundary[..., None, :],
            ),
            axis=2,
        )
        x_flux = velocity.x[..., None] * x_faces

        y_boundary = 0.5 * (cells[:, -1, ...] + cells[:, 0, ...])
        y_faces = jnp.concatenate(
            (
                y_boundary[:, None, ...],
                0.5 * (cells[:, :-1, ...] + cells[:, 1:, ...]),
                y_boundary[:, None, ...],
            ),
            axis=1,
        )
        y_flux = velocity.y[..., None] * y_faces

        z_flux = self.vertical_advective_flux(velocity, cells)
        rank, dtype = z_flux.ndim, z_flux.dtype
        return -(
            (x_flux[:, :, 1:, :] - x_flux[:, :, :-1, :])
            / self.x_metric.cell_widths(rank, dtype)
            + (y_flux[:, 1:, :, :] - y_flux[:, :-1, :, :])
            / self.y_metric.cell_widths(rank, dtype)
            + (z_flux[1:] - z_flux[:-1]) / self.z_metric.cell_widths(rank, dtype)
        )

    def vertical_advective_flux(
        self,
        velocity: MACVelocity,
        cell_velocity: Array | None = None,
    ) -> Array:
        """Return the conservative vertical flux of all momentum components."""
        cells = _cell_velocity(velocity) if cell_velocity is None else cell_velocity
        flux = jnp.zeros(
            (cells.shape[0] + 1, *cells.shape[1:]),
            dtype=cells.dtype,
        )
        return flux.at[1:-1].set(
            velocity.z[1:-1, ..., None] * 0.5 * (cells[:-1] + cells[1:])
        )

    def amd_tendency(self, cell_velocity: Array) -> Array:
        """Return the non-variational AMD stress divergence."""

        stress = self.amd_stress(cell_velocity)
        result = []
        for component in range(3):
            divergence = sum(
                self._negative_derivative_transpose(
                    stress[..., component, direction],
                    direction,
                )
                for direction in range(3)
            )
            result.append(divergence)
        return jnp.stack(result, axis=-1)

    def sgs_viscosity(
        self,
        cell_velocity: Array,
        lasd_coefficient: Array | None = None,
        *,
        gradient: Array | None = None,
    ) -> Array:
        if gradient is None:
            gradient = self.velocity_gradient(cell_velocity)
        return self._sgs_viscosity_from_gradient(
            gradient,
            lasd_coefficient,
        )

    def explicit_sgs_diffusion_rate(
        self,
        viscosity: Array,
        *,
        include_vertical: bool = True,
    ) -> Array:
        """Return the maximum local FV diffusion diagonal magnitude."""

        rank, dtype = viscosity.ndim, viscosity.dtype
        horizontal_diagonal = self.x_metric.broadcast(
            self.x_metric.diffusion_diagonal,
            rank,
            dtype,
        ) + self.y_metric.broadcast(
            self.y_metric.diffusion_diagonal,
            rank,
            dtype,
        )
        rate = viscosity * horizontal_diagonal
        if not include_vertical or viscosity.shape[0] == 1:
            return jnp.max(rate)
        face_viscosity = _interpolate_to_vertical_faces(
            viscosity,
            self.z_centers,
            self.z_faces,
        )
        vertical = jnp.zeros_like(viscosity)
        vertical = vertical.at[:-1].add(
            face_viscosity / _z_vector(self.dz_cell[:-1] * self.dz_center, rank, dtype)
        )
        vertical = vertical.at[1:].add(
            face_viscosity / _z_vector(self.dz_cell[1:] * self.dz_center, rank, dtype)
        )
        return jnp.max(rate + vertical)

    def _sgs_viscosity_from_gradient(
        self,
        gradient: Array,
        lasd_coefficient: Array | None = None,
    ) -> Array:
        if self.lasd_closure is None:
            return self._amd_viscosity_from_gradient(gradient)
        coefficient = (
            self._require_lasd_state().coefficient
            if lasd_coefficient is None
            else lasd_coefficient
        )
        return self.lasd_closure.viscosity(
            coefficient,
            gradient,
        )

    def sgs_stress(
        self,
        cell_velocity: Array,
        lasd_coefficient: Array | None = None,
        *,
        gradient: Array | None = None,
    ) -> Array:
        if gradient is None:
            gradient = self.velocity_gradient(cell_velocity)
        return self._sgs_stress_from_gradient(
            gradient,
            lasd_coefficient,
        )

    def _sgs_stress_from_gradient(
        self,
        gradient: Array,
        lasd_coefficient: Array | None = None,
    ) -> Array:
        if self.lasd_closure is None:
            return self._amd_stress_from_gradient(gradient)
        coefficient = (
            self._require_lasd_state().coefficient
            if lasd_coefficient is None
            else lasd_coefficient
        )
        return self.lasd_closure.stress(
            coefficient,
            gradient,
        )

    def sgs_tendency(
        self,
        cell_velocity: Array,
        lasd_coefficient: Array | None = None,
        *,
        gradient: Array | None = None,
        wall_velocity: Array | None = None,
        wall_stress: Array | None = None,
    ) -> Array:
        if gradient is None:
            gradient = self.velocity_gradient(cell_velocity)
        viscosity = self._sgs_viscosity_from_gradient(
            gradient,
            lasd_coefficient,
        )
        return self.variational_sgs_tendency(
            cell_velocity,
            viscosity,
            gradient=gradient,
            wall_stress=(
                self.wall_stress(
                    cell_velocity,
                    wall_velocity=wall_velocity,
                )
                if wall_stress is None
                else wall_stress
            ),
        )

    def vertical_sgs_stress_flux(
        self,
        cell_velocity: Array,
        lasd_coefficient: Array | None = None,
        *,
        gradient: Array | None = None,
        wall_velocity: Array | None = None,
        wall_stress: Array | None = None,
    ) -> Array:
        """Return SGS momentum stress on every vertical face.

        Its discrete divergence is exactly the wall-normal part of
        :meth:`sgs_tendency`, including the modeled lower-wall traction.
        """
        if gradient is None:
            gradient = self.velocity_gradient(cell_velocity)
        stress = self._sgs_stress_from_gradient(
            gradient,
            lasd_coefficient,
        )
        faces = self._vertical_stress_faces_from_cell_stress(stress[..., :, 2])
        return faces.at[0].add(
            self.wall_stress(
                cell_velocity,
                wall_velocity=wall_velocity,
            )
            if wall_stress is None
            else wall_stress
        )

    def _upper_face_speeds(
        self,
        velocity: MACVelocity,
    ) -> tuple[tuple[Array, AxisMetric], ...]:
        """Pair the upper-face normal speed of each axis with its metric."""

        return (
            (velocity.x[..., 1:], self.x_metric),
            (velocity.y[:, 1:, :], self.y_metric),
            (velocity.z[1:, ...], self.z_metric),
        )

    def _reconstruction_dissipation(
        self,
        velocity: MACVelocity,
        cell_velocity: Array | None,
    ) -> Array:
        cells = _cell_velocity(velocity) if cell_velocity is None else cell_velocity
        return reconstruction_dissipation(
            cells,
            self._upper_face_speeds(velocity),
            self.config.mp5_dissipation_strength,
        )

    def mp5_dissipation(
        self,
        velocity: MACVelocity,
        cell_velocity: Array | None = None,
    ) -> Array:
        """Return local MP5/Rusanov face dissipation for all momenta."""
        return self._reconstruction_dissipation(velocity, cell_velocity)

    def advection_dissipation(
        self,
        velocity: MACVelocity,
        cell_velocity: Array | None = None,
    ) -> Array:
        """Return the conservative MP5 advection stabilization."""
        return self._reconstruction_dissipation(velocity, cell_velocity)

    def vertical_advection_dissipation_flux(
        self,
        velocity: MACVelocity,
        cell_velocity: Array | None = None,
    ) -> Array:
        """Return the configured numerical momentum flux on vertical faces."""
        cells = _cell_velocity(velocity) if cell_velocity is None else cell_velocity
        flux = jnp.zeros(
            (cells.shape[0] + 1, *cells.shape[1:]),
            dtype=cells.dtype,
        )
        upper_flux = reconstruction_flux(
            cells,
            velocity.z[1:],
            self.z_metric,
            self.config.mp5_dissipation_strength,
        )
        return flux.at[1:].set(upper_flux)

    @property
    def wall_cell_height(self) -> float:
        """Physical height of the finite volume filtered by the wall law."""
        return self._wall_cell_height

    def instantaneous_wall_velocity(self, cell_velocity: Array) -> Array:
        """Return the spatially filtered first-cell mean wall-model velocity."""
        horizontal = cell_velocity[0, ..., :2]
        width = self.config.wall_filter_width
        if width is None:
            return horizontal
        return physical_top_hat_filter(
            horizontal,
            width,
            axes=(-3, -2),
            boundaries=("periodic", "periodic"),
        )

    def wall_velocity(
        self,
        cell_velocity: Array,
        *,
        filtered_velocity: Array | None = None,
    ) -> Array:
        """Return the horizontal velocity supplied to the wall law."""
        if filtered_velocity is not None:
            return filtered_velocity
        if (
            self.config.wall_temporal_filter_timescale is not None
            and self._wall_model_state is not None
        ):
            return self._wall_model_state.filtered_velocity
        return self.instantaneous_wall_velocity(cell_velocity)

    def wall_ustar(
        self,
        cell_velocity: Array,
        *,
        wall_velocity: Array | None = None,
    ) -> Array:
        horizontal = self.wall_velocity(
            cell_velocity,
            filtered_velocity=wall_velocity,
        )
        return self.wall_law.friction_velocity(
            horizontal,
            self.wall_cell_height,
        )

    def wall_fluxes(
        self,
        cell_velocity: Array,
        *,
        wall_velocity: Array | None = None,
    ) -> SurfaceLayerFluxes:
        """Evaluate the closure independently of its conservative coupling."""
        horizontal = self.wall_velocity(
            cell_velocity,
            filtered_velocity=wall_velocity,
        )
        return self.wall_law.surface_fluxes(
            horizontal,
            self.wall_cell_height,
        )

    def surface_momentum_stability_rate(
        self,
        horizontal_velocity: Array,
        momentum_stress: Array,
    ) -> Array:
        """Return the explicit lower-wall drag linearization rate."""
        speed = jnp.linalg.norm(horizontal_velocity, axis=-1)
        stress = jnp.linalg.norm(momentum_stress, axis=-1)
        epsilon = jnp.finfo(horizontal_velocity.dtype).tiny
        local_rate = jnp.where(
            speed > epsilon,
            2.0 * stress / (jnp.maximum(speed, epsilon) * self.dz_cell[0]),
            0.0,
        )
        return jnp.max(local_rate)

    def wall_stability_rate(
        self,
        cell_velocity: Array,
        *,
        wall_velocity: Array | None = None,
    ) -> Array:
        """Return the active neutral-log wall drag rate."""
        horizontal = self.wall_velocity(
            cell_velocity,
            filtered_velocity=wall_velocity,
        )
        fluxes = self.wall_law.surface_fluxes(
            horizontal,
            self.wall_cell_height,
        )
        return self.surface_momentum_stability_rate(
            horizontal,
            fluxes.momentum_stress,
        )

    def wall_stress(
        self,
        cell_velocity: Array,
        *,
        wall_velocity: Array | None = None,
    ) -> Array:
        """Return positive inward-normal SGS stress on the lower wall face."""
        fluxes = self.wall_fluxes(
            cell_velocity,
            wall_velocity=wall_velocity,
        )
        tangential = fluxes.momentum_stress
        return jnp.concatenate(
            (tangential, jnp.zeros_like(tangential[..., :1])),
            axis=-1,
        )

    def _wall_stress_tangential_faces(
        self,
        wall_stress: Array,
    ) -> tuple[Array, Array]:
        expected = (*self.grid.shape[1:], 3)
        if tuple(wall_stress.shape) != expected:
            raise ValueError(
                f"expected wall stress shape {expected}, got {tuple(wall_stress.shape)}"
            )
        stress_x = _interpolate_to_periodic_faces(
            wall_stress[..., 0],
            self.x_metric.centers,
            self.x_metric.faces,
            axis=1,
        )
        stress_y = _interpolate_to_periodic_faces(
            wall_stress[..., 1],
            self.y_metric.centers,
            self.y_metric.faces,
            axis=0,
        )
        return stress_x, stress_y

    def _add_wall_stress_face_tendency(
        self,
        tendency: MACVelocity,
        wall_stress: Array,
    ) -> MACVelocity:
        stress_x, stress_y = self._wall_stress_tangential_faces(wall_stress)
        inverse_first_width = jnp.asarray(
            -1.0,
            dtype=wall_stress.dtype,
        ) / self.dz_cell[0].astype(wall_stress.dtype)
        return MACVelocity(
            tendency.x.at[0].add(inverse_first_width * stress_x),
            tendency.y.at[0].add(inverse_first_width * stress_y),
            tendency.z,
        )

    def wall_stress_face_tendency(self, wall_stress: Array) -> MACVelocity:
        """Map cell-centred wall traction to its physical MAC u/v faces."""
        nz, ny, nx = self.grid.shape
        return self._add_wall_stress_face_tendency(
            MACVelocity(
                jnp.zeros((nz, ny, nx + 1), dtype=wall_stress.dtype),
                jnp.zeros((nz, ny + 1, nx), dtype=wall_stress.dtype),
                jnp.zeros((nz + 1, ny, nx), dtype=wall_stress.dtype),
            ),
            wall_stress,
        )

    def forcing_tendency(self, cell_velocity: Array) -> Array:
        tendency = jnp.zeros_like(cell_velocity)
        tendency = tendency.at[..., 0].add(self.pressure_acceleration)
        if self.config.geostrophic_wind is not None:
            geostrophic_u, geostrophic_v = self.config.geostrophic_wind
            u = cell_velocity[..., 0]
            v = cell_velocity[..., 1]
            w = cell_velocity[..., 2]
            vertical = self.config.coriolis_vertical
            horizontal = self.config.coriolis_horizontal
            tendency = tendency.at[..., 0].add(
                vertical * (v - geostrophic_v) - horizontal * w
            )
            tendency = tendency.at[..., 1].add(-vertical * (u - geostrophic_u))
            tendency = tendency.at[..., 2].add(horizontal * u)
        return tendency

    def cell_tendency(
        self,
        velocity: MACVelocity,
        lasd_coefficient: Array | None = None,
        *,
        cell_velocity: Array | None = None,
        gradient: Array | None = None,
        wall_velocity: Array | None = None,
        wall_stress: Array | None = None,
    ) -> Array:
        cells = _cell_velocity(velocity) if cell_velocity is None else cell_velocity
        if gradient is None:
            gradient = self.velocity_gradient(cells)
        tendency = (
            self.conservative_advection(velocity, cells)
            + self.sgs_tendency(
                cells,
                lasd_coefficient,
                gradient=gradient,
                wall_velocity=wall_velocity,
                wall_stress=wall_stress,
            )
            + self.forcing_tendency(cells)
        )
        if self.config.mp5_dissipation_strength > 0.0:
            tendency += self.advection_dissipation(velocity, cells)
        return tendency

    def tendency(self, velocity: MACVelocity, _time: float) -> MACVelocity:
        return self._compiled_tendency(
            velocity,
            self._active_lasd_coefficient(velocity),
            self.active_wall_velocity(velocity),
        )

    def tendency_with_wall_stress(
        self,
        velocity: MACVelocity,
        wall_stress: Array,
    ) -> MACVelocity:
        """Return the tendency with a prescribed lower-wall stress plane."""
        expected = (*self.grid.shape[1:], 3)
        if tuple(wall_stress.shape) != expected:
            raise ValueError(
                f"expected wall stress shape {expected}, got {tuple(wall_stress.shape)}"
            )
        return self._compiled_tendency_with_wall_stress(
            velocity,
            self._active_lasd_coefficient(velocity),
            wall_stress,
        )

    def _active_lasd_coefficient(self, velocity: MACVelocity) -> Array:
        if self.lasd_closure is None:
            return jnp.zeros((1,), dtype=velocity.x.dtype)
        if self._lasd_state is None:
            return jnp.full(
                self.grid.shape,
                self.config.lasd.initial_coefficient,
                dtype=velocity.x.dtype,
            )
        return self._require_lasd_state().coefficient

    def _require_lasd_state(self) -> LASDState:
        if self._lasd_state is None:
            raise RuntimeError("LASD state has not been initialized")
        return self._lasd_state

    def reset_lasd(self, velocity: MACVelocity) -> LASDState | None:
        """Initialize accepted-step LASD memory from ``velocity``."""
        if self.lasd_closure is None:
            return None
        self._lasd_state = self.lasd_closure.initialize(_cell_velocity(velocity))
        self._lasd_step = 0
        self._lasd_interval_time = 0.0
        return self._lasd_state

    @property
    def lasd_state(self) -> LASDState | None:
        return self._lasd_state

    @property
    def wall_model_state(self) -> WallModelState | None:
        return self._wall_model_state

    def reset_wall_model(self, velocity: MACVelocity) -> WallModelState | None:
        """Initialize temporal wall-model memory from ``velocity``."""
        if self.config.wall_temporal_filter_timescale is None:
            self._wall_model_state = None
            return None
        self._wall_model_state = WallModelState(
            self.instantaneous_wall_velocity(_cell_velocity(velocity))
        )
        return self._wall_model_state

    def restore_wall_model(self, state: WallModelState) -> None:
        expected = (
            self.grid.shape[1],
            self.grid.shape[2],
            2,
        )
        if state.filtered_velocity.shape != expected:
            raise ValueError("wall-model state shape does not match the grid")
        self._wall_model_state = WallModelState(
            jnp.asarray(
                state.filtered_velocity,
                dtype=self.pressure_solver.operator.dtype,
            )
        )

    def active_wall_velocity(self, velocity: MACVelocity) -> Array:
        """Return the frozen wall input to use throughout one time step."""
        cells = _cell_velocity(velocity)
        if self.config.wall_temporal_filter_timescale is None:
            return self.instantaneous_wall_velocity(cells)
        if self._wall_model_state is None:
            return self.instantaneous_wall_velocity(cells)
        return self._wall_model_state.filtered_velocity

    def _advance_wall_model(
        self,
        velocity: MACVelocity,
        timestep: float,
    ) -> None:
        timescale = self.config.wall_temporal_filter_timescale
        if timescale is None:
            return
        if self._wall_model_state is None:
            self.reset_wall_model(velocity)
            return
        epsilon = min(timestep / timescale, 1.0)
        instantaneous = self.instantaneous_wall_velocity(_cell_velocity(velocity))
        filtered = (
            1.0 - epsilon
        ) * self._wall_model_state.filtered_velocity + epsilon * instantaneous
        self._wall_model_state = WallModelState(filtered)

    @property
    def lasd_progress(self) -> tuple[int, float]:
        """Return accepted steps and elapsed time in the current LASD interval."""
        return self._lasd_step, self._lasd_interval_time

    def restore_lasd(
        self,
        state: LASDState,
        *,
        accepted_step: int,
        interval_time: float,
    ) -> None:
        """Restore checkpointed LASD memory and accepted-step counters."""
        if self.lasd_closure is None:
            raise RuntimeError("cannot restore LASD on an AMD solver")
        if accepted_step < 0 or interval_time < 0.0:
            raise ValueError("LASD checkpoint progress must be nonnegative")
        fine_shape = self.grid.shape
        coarse_shape = self.lasd_closure.hierarchy.grids[1].shape
        expected_shapes = (fine_shape,) + (coarse_shape,) * 7
        if tuple(field.shape for field in state) != expected_shapes:
            raise ValueError(
                "LASD checkpoint fields do not match the fine/coarse hierarchy"
            )
        self._lasd_state = state
        self._lasd_step = accepted_step
        self._lasd_interval_time = interval_time

    @property
    def pressure(self) -> Array:
        """Return the pressure accepted by the last full-projection step."""
        return self._pressure

    def restore_pressure(self, pressure: Array) -> None:
        """Restore a gauge-fixed full-projection pressure from a checkpoint."""
        pressure = jnp.asarray(pressure, dtype=self.pressure_solver.operator.dtype)
        if tuple(pressure.shape) != self.grid.shape:
            raise ValueError("pressure checkpoint shape does not match grid")
        self._pressure = self.pressure_solver.operator.project_nullspace(pressure)

    def reset_pressure(self) -> None:
        """Reset the full-projection pressure initial guess to zero."""
        self._pressure = jnp.zeros_like(self._pressure)

    def implicit_diffusion_solve(
        self,
        velocity: MACVelocity,
        frozen_viscosity: Array,
        implicit_timestep: float,
    ) -> MACVelocity:
        """Solve one frozen-viscosity vertical SGS diffusion stage."""
        if not math.isfinite(implicit_timestep) or implicit_timestep < 0.0:
            raise ValueError("implicit timestep must be finite and nonnegative")
        return self._compiled_implicit_diffusion(
            velocity,
            jnp.asarray(frozen_viscosity, dtype=velocity.x.dtype),
            jnp.asarray(implicit_timestep, dtype=velocity.x.dtype),
        )

    def prepare_imex_step(
        self,
        velocity: MACVelocity,
    ) -> tuple[PreparedIMEXStep, Array]:
        """Prepare initial IMEX tendencies and the two stability rates.

        The accepted LASD and wall-model states are explicit inputs so the
        compiled kernel can be launched immediately after an accepted step and
        safely consumed on the following iteration.
        """
        if self.config.sgs_time_integration != "imex_ark3":
            raise RuntimeError("IMEX preparation requires imex_ark3 integration")
        coefficient = self._active_lasd_coefficient(velocity)
        wall_velocity = self.active_wall_velocity(velocity)
        initial_explicit, initial_implicit, frozen_viscosity, rates = (
            self._compiled_imex_prepare(
                velocity,
                coefficient,
                wall_velocity,
            )
        )
        return (
            PreparedIMEXStep(
                initial_explicit,
                initial_implicit,
                frozen_viscosity,
                coefficient,
                wall_velocity,
            ),
            rates,
        )

    def _imex_ark3_step(
        self,
        velocity: MACVelocity,
        *,
        timestep: float,
        time: float,
        lasd_coefficient: Array,
        wall_velocity: Array,
        initial_pressure: Array | None = None,
        explicit_forcing: MACVelocity | None = None,
        explicit_forcing_provider: Callable[[MACVelocity, float], MACVelocity]
        | None = None,
        wall_stress_provider: Callable[[MACVelocity, float], Array] | None = None,
        prepared: PreparedIMEXStep | None = None,
    ) -> VelocityPressureProjection:
        """Advance ARS(2,3,3) with frozen vertical SGS diffusion implicit.

        ``explicit_forcing`` is a stage-independent acceleration, such as
        buoyancy frozen at the midpoint of a Strang-coupled scalar step.  A
        state-dependent ``explicit_forcing_provider`` and the
        ``wall_stress_provider`` is evaluated with each ARK stage velocity and
        physical stage time, keeping nonlinear surface-layer coupling explicit.
        """
        if prepared is not None:
            if wall_stress_provider is not None:
                raise ValueError("prepared IMEX work cannot supply prescribed stress")
            initial_explicit = prepared.initial_explicit
            initial_implicit = prepared.initial_implicit
            frozen_viscosity = prepared.frozen_viscosity
        elif wall_stress_provider is None:
            initial_explicit, initial_implicit, frozen_viscosity = (
                self._compiled_imex_initial_tendencies(
                    velocity,
                    lasd_coefficient,
                    wall_velocity,
                )
            )
        else:
            initial_explicit, initial_implicit, frozen_viscosity = (
                self._compiled_imex_initial_tendencies_with_wall_stress(
                    velocity,
                    lasd_coefficient,
                    wall_stress_provider(
                        velocity,
                        time + _ARK3_C[0] * timestep,
                    ),
                )
            )
        if explicit_forcing is not None:
            initial_explicit = _velocity_sum(
                (1.0, initial_explicit),
                (1.0, explicit_forcing),
            )
        if explicit_forcing_provider is not None:
            initial_explicit = _velocity_sum(
                (1.0, initial_explicit),
                (
                    1.0,
                    explicit_forcing_provider(
                        velocity,
                        time + _ARK3_C[0] * timestep,
                    ),
                ),
            )
        explicit_tendencies: list[MACVelocity] = [initial_explicit]
        implicit_tendencies: list[MACVelocity] = [initial_implicit]

        def evaluate(stage_velocity: MACVelocity, stage_index: int) -> None:
            if wall_stress_provider is None:
                explicit, implicit = self._compiled_imex_tendencies(
                    stage_velocity,
                    frozen_viscosity,
                    lasd_coefficient,
                    wall_velocity,
                )
            else:
                explicit, implicit = self._compiled_imex_tendencies_with_wall_stress(
                    stage_velocity,
                    frozen_viscosity,
                    lasd_coefficient,
                    wall_stress_provider(
                        stage_velocity,
                        time + _ARK3_C[stage_index] * timestep,
                    ),
                )
            if explicit_forcing is not None:
                explicit = _velocity_sum(
                    (1.0, explicit),
                    (1.0, explicit_forcing),
                )
            if explicit_forcing_provider is not None:
                explicit = _velocity_sum(
                    (1.0, explicit),
                    (
                        1.0,
                        explicit_forcing_provider(
                            stage_velocity,
                            time + _ARK3_C[stage_index] * timestep,
                        ),
                    ),
                )
            explicit_tendencies.append(explicit)
            implicit_tendencies.append(implicit)

        pressure_guess = initial_pressure

        for stage_index in range(1, len(_ARK3_C)):
            terms: list[tuple[float, MACVelocity]] = [(1.0, velocity)]
            for previous in range(stage_index):
                explicit_weight = timestep * _ARK3_EXPLICIT_A[stage_index][previous]
                implicit_weight = timestep * _ARK3_IMPLICIT_A[stage_index][previous]
                if explicit_weight != 0.0:
                    terms.append(
                        (
                            explicit_weight,
                            explicit_tendencies[previous],
                        )
                    )
                if implicit_weight != 0.0:
                    terms.append(
                        (
                            implicit_weight,
                            implicit_tendencies[previous],
                        )
                    )
            stage = self.implicit_diffusion_solve(
                _velocity_sum(*terms),
                frozen_viscosity,
                _ARK3_GAMMA * timestep,
            )
            projection_timestep = _ARK3_C[stage_index] * timestep
            projected = self.projector.project_velocity_and_pressure(
                stage,
                timestep=projection_timestep,
                initial_pressure=pressure_guess,
            )
            stage = projected.velocity
            pressure_guess = projected.pressure
            evaluate(stage, stage_index)

        final_terms: list[tuple[float, MACVelocity]] = [(1.0, velocity)]
        for stage_index in range(len(_ARK3_C)):
            final_terms.append(
                (
                    timestep * _ARK3_EXPLICIT_B[stage_index],
                    explicit_tendencies[stage_index],
                )
            )
            final_terms.append(
                (
                    timestep * _ARK3_IMPLICIT_B[stage_index],
                    implicit_tendencies[stage_index],
                )
            )
        return self.projector.project_velocity_and_pressure(
            _velocity_sum(*final_terms),
            timestep=timestep,
            initial_pressure=pressure_guess,
        )

    def _advance_lasd(
        self,
        velocity: MACVelocity,
        timestep: float,
    ) -> None:
        if self.lasd_closure is None:
            return
        cells = _cell_velocity(velocity)
        self._lasd_interval_time += timestep
        accepted_step = self._lasd_step + 1
        interval = self.config.lasd.update_interval
        if accepted_step % interval == 0:
            state, lm, mm, qn, nn = self._compiled_lasd_statistics(
                self._require_lasd_state(),
                cells,
            )
            state = self._compiled_lasd_finalize(
                state,
                lm,
                mm,
                qn,
                nn,
                jnp.asarray(self._lasd_interval_time, dtype=cells.dtype),
                jnp.asarray(accepted_step == interval),
            )
            self._lasd_interval_time = 0.0
        else:
            state = self._compiled_lasd_accumulate(
                self._require_lasd_state(),
                cells,
            )
        self._lasd_state = state
        self._lasd_step = accepted_step

    @staticmethod
    @jax.jit
    def enforce_boundaries(velocity: MACVelocity) -> MACVelocity:
        """Restore the periodic seam and the impermeable walls.

        Compiled because it is six scatters over face arrays and runs twice per
        step; uncompiled each one was its own dispatch and memory pass.
        """

        x_boundary = 0.5 * (velocity.x[..., 0] + velocity.x[..., -1])
        y_boundary = 0.5 * (velocity.y[:, 0, :] + velocity.y[:, -1, :])
        return MACVelocity(
            velocity.x.at[..., 0].set(x_boundary).at[..., -1].set(x_boundary),
            velocity.y.at[:, 0, :].set(y_boundary).at[:, -1, :].set(y_boundary),
            velocity.z.at[0].set(0.0).at[-1].set(0.0),
        )

    def initial_log_profile(
        self,
        *,
        perturbation_amplitude: float = 0.05,
        project: bool = True,
    ) -> MACVelocity:
        nz, ny, nx = self.grid.shape
        z = self.z_centers - self.z_faces[0]
        lower = self.z_faces[:-1] - self.z_faces[0]
        upper = self.z_faces[1:] - self.z_faces[0]
        mean_u = (
            self.config.friction_velocity
            / self.config.von_karman
            * self.wall_law.cell_average_log_denominators(lower, upper)
        )
        # Phases follow the physical centres so a stretched horizontal axis
        # still receives a smooth periodic perturbation.
        xx = (
            2.0
            * jnp.pi
            * (self.x_metric.centers - self.x_metric.faces[0])
            / self.x_metric.length
        )
        yy = (
            2.0
            * jnp.pi
            * (self.y_metric.centers - self.y_metric.faces[0])
            / self.y_metric.length
        )
        envelope = jnp.sin(jnp.pi * z / self.height)
        perturbation = (
            perturbation_amplitude
            * self.config.friction_velocity
            * envelope[:, None, None]
        )
        cells = jnp.zeros((nz, ny, nx, 3), dtype=z.dtype)
        cells = cells.at[..., 0].set(
            mean_u[:, None, None]
            + perturbation
            * jnp.sin(2.0 * yy)[None, :, None]
            * jnp.cos(xx)[None, None, :]
        )
        cells = cells.at[..., 1].set(
            perturbation * jnp.cos(yy)[None, :, None] * jnp.sin(2.0 * xx)[None, None, :]
        )
        velocity = self.enforce_boundaries(_cells_to_faces(cells))
        if not project:
            return velocity
        if self.pressure_solver.krylov.execution == "jax":
            return self.projector.project_velocity(velocity, timestep=1.0)
        return self.projector.project(velocity, timestep=1.0).velocity

    def initial_profile(
        self,
        mean_u: Array,
        mean_v: Array,
        *,
        perturbation_tke: Array | None = None,
        seed: int = 0,
        project: bool = True,
    ) -> MACVelocity:
        """Build a horizontally homogeneous tabulated velocity profile."""
        nz, ny, nx = self.grid.shape
        dtype = self.pressure_solver.operator.dtype
        mean_u = jnp.asarray(mean_u, dtype=dtype)
        mean_v = jnp.asarray(mean_v, dtype=dtype)
        if mean_u.shape != (nz,) or mean_v.shape != (nz,):
            raise ValueError("mean profiles must contain one value per z cell")
        cells = jnp.zeros((nz, ny, nx, 3), dtype=dtype)
        cells = cells.at[..., 0].set(mean_u[:, None, None])
        cells = cells.at[..., 1].set(mean_v[:, None, None])
        if perturbation_tke is not None:
            target_tke = jnp.asarray(perturbation_tke, dtype=dtype)
            if target_tke.shape != (nz,):
                raise ValueError("perturbation TKE must contain one value per z cell")
            if bool(jnp.any(target_tke < 0.0)):
                raise ValueError("perturbation TKE must be nonnegative")
            random = jax.random.uniform(
                jax.random.PRNGKey(seed),
                cells.shape,
                dtype=dtype,
                minval=-0.5,
                maxval=0.5,
            )
            random -= self.horizontal_mean(random, keepdims=True)
            current_tke = 0.5 * self.horizontal_mean(
                jnp.sum(random * random, axis=-1),
            )
            scale = jnp.sqrt(
                target_tke / jnp.maximum(current_tke, jnp.finfo(dtype).eps)
            )
            random *= scale[:, None, None, None]
            cells += random
        velocity = self.enforce_boundaries(_cells_to_faces(cells))
        if not project:
            return velocity
        if self.pressure_solver.krylov.execution == "jax":
            return self.projector.project_velocity(velocity, timestep=1.0)
        return self.projector.project(velocity, timestep=1.0).velocity

    @staticmethod
    def cell_centered_velocity(velocity: MACVelocity) -> Array:
        return _cell_velocity(velocity)

    def diagnostic_wall_consistent_gradient(
        self,
        cell_velocity: Array,
        *,
        gradient: Array | None = None,
    ) -> Array:
        """Return a gradient with neutral-log shear at the first cell.

        The replacement is diagnostic only.  It supplies the unresolved-energy
        equilibrium with the same wall shear used by the momentum boundary
        traction without changing the prognostic AMD operator.
        """
        if gradient is None:
            gradient = self.velocity_gradient(cell_velocity)
        height = self.grid.z_centers[0] - self.grid.z_faces[0]
        horizontal = self.wall_velocity(cell_velocity)
        speed = jnp.linalg.norm(horizontal, axis=-1)
        epsilon = jnp.finfo(cell_velocity.dtype).tiny
        direction = horizontal / jnp.maximum(speed[..., None], epsilon)
        wall_shear = self.wall_law.friction_velocity(
            horizontal,
            self.wall_cell_height,
        ) / (self.config.von_karman * height)
        gradient = gradient.at[0, ..., 0, 2].set(wall_shear * direction[..., 0])
        return gradient.at[0, ..., 1, 2].set(wall_shear * direction[..., 1])

    def cell_length_scale(self, dtype=None) -> Array:
        """Return the local closure length ``(dx dy dz)^(1/3)`` of every cell.

        Diagnostics need the same length the closure uses, and on a stretched
        grid that varies cell by cell, so it is published here rather than
        reconstructed from a nominal spacing by each caller.
        """

        return jnp.prod(_cell_length_scales(self.metrics, dtype), axis=-1) ** (
            1.0 / 3.0
        )

    def diagnostic_sgs_tke(
        self,
        cell_velocity: Array,
        lasd_coefficient: Array | None = None,
        *,
        gradient: Array | None = None,
        dissipation_coefficient: float = 0.93,
    ) -> Array:
        """Diagnose unresolved TKE from local production-dissipation balance.

        This is an observation of an eddy-viscosity closure, not prognostic SGS
        energy.  The same labeled diagnostic is used for like-for-like Andrén
        resolved-plus-SGS statistics in the LASD benchmark path.
        """
        if not math.isfinite(dissipation_coefficient) or dissipation_coefficient <= 0.0:
            raise ValueError("SGS dissipation coefficient must be positive")
        if gradient is None:
            gradient = self.velocity_gradient(cell_velocity)
        diagnostic_gradient = self.diagnostic_wall_consistent_gradient(
            cell_velocity,
            gradient=gradient,
        )
        strain = 0.5 * (diagnostic_gradient + jnp.swapaxes(diagnostic_gradient, -1, -2))
        strain_magnitude_squared = 2.0 * jnp.einsum(
            "...ij,...ij->...",
            strain,
            strain,
        )
        viscosity = self._sgs_viscosity_from_gradient(
            gradient,
            lasd_coefficient,
        )
        if self.lasd_closure is None:
            viscosity = jnp.maximum(
                viscosity - self.config.amd.molecular_viscosity,
                0.0,
            )
        delta = self.cell_length_scale(viscosity.dtype)
        return jnp.maximum(
            viscosity * strain_magnitude_squared * delta / dissipation_coefficient,
            0.0,
        ) ** (2.0 / 3.0)

    def resolved_tke_sgs_dissipation(
        self,
        cell_velocity: Array,
        lasd_coefficient: Array | None = None,
        *,
        gradient: Array | None = None,
    ) -> Array:
        """Return the local SGS term in the resolved-TKE equation."""
        if gradient is None:
            gradient = self.velocity_gradient(cell_velocity)
        stress = self._sgs_stress_from_gradient(
            gradient,
            lasd_coefficient,
        )
        return -jnp.einsum("...ij,...ij->...", stress, gradient)

    def amd_stress(self, cell_velocity: Array) -> Array:
        """Return the modeled viscous momentum flux ``2 nu S``."""
        gradient = self.velocity_gradient(cell_velocity)
        return self._amd_stress_from_gradient(gradient)

    def step(
        self,
        velocity: MACVelocity,
        *,
        timestep: float,
        time: float,
        prepared: PreparedIMEXStep | None = None,
    ) -> MACVelocity:
        if prepared is not None and self.config.sgs_time_integration != "imex_ark3":
            raise ValueError("prepared work can only be used by IMEX integration")
        if self.lasd_closure is not None and self._lasd_state is None:
            self.reset_lasd(velocity)
        if (
            self.config.wall_temporal_filter_timescale is not None
            and self._wall_model_state is None
        ):
            self.reset_wall_model(velocity)
        coefficient = (
            self._active_lasd_coefficient(velocity)
            if prepared is None
            else prepared.lasd_coefficient
        )
        wall_velocity = (
            self.active_wall_velocity(velocity)
            if prepared is None
            else prepared.wall_velocity
        )

        if self.config.sgs_time_integration == "imex_ark3":
            projected = self._imex_ark3_step(
                velocity,
                timestep=timestep,
                time=time,
                lasd_coefficient=coefficient,
                wall_velocity=wall_velocity,
                initial_pressure=self._pressure,
                prepared=prepared,
            )
            advanced = self.enforce_boundaries(projected.velocity)
            self._pressure = projected.pressure
            self._advance_lasd(advanced, timestep)
            self._advance_wall_model(advanced, timestep)
            return advanced

        def stage_tendency(
            stage_velocity: MACVelocity,
            _stage_time: float,
        ) -> MACVelocity:
            return self._compiled_tendency(
                stage_velocity,
                coefficient,
                wall_velocity,
            )

        projected = projected_ssprk3_velocity_pressure_step(
            velocity,
            tendency=stage_tendency,
            projector=self.projector,
            timestep=timestep,
            time=time,
            initial_pressure=self._pressure,
        )
        advanced = projected.velocity
        self._pressure = projected.pressure
        advanced = self.enforce_boundaries(advanced)
        self._advance_lasd(advanced, timestep)
        self._advance_wall_model(advanced, timestep)
        return advanced

    def diagnostic(
        self,
        velocity: MACVelocity,
        *,
        timestep: float,
        time: float,
        lasd_coefficient: Array | None = None,
    ) -> MomentumDiagnostic:
        """Diagnose a state, optionally using the coefficient from its last step."""
        coefficient = (
            self._active_lasd_coefficient(velocity)
            if lasd_coefficient is None
            else lasd_coefficient
        )
        values = self._compiled_diagnostic(
            velocity,
            jnp.asarray(timestep, dtype=velocity.x.dtype),
            coefficient,
            self.active_wall_velocity(velocity),
        )
        return MomentumDiagnostic(
            time,
            *(float(value) for value in values),
        )

    def cfl_rate(self, velocity: MACVelocity) -> Array:
        """Return a conservative cell-local face-envelope CFL rate.

        Each cell uses the larger speed on its two faces in each direction.
        Taking the maximum only after summing those local contributions avoids
        combining three unrelated domain-wide extrema while remaining safe for
        the face-flux update.
        """
        rank, dtype = velocity.z[:-1].ndim, velocity.z.dtype
        local_rate = (
            jnp.maximum(
                jnp.abs(velocity.x[..., :-1]),
                jnp.abs(velocity.x[..., 1:]),
            )
            / self.x_metric.cell_widths(rank, dtype)
            + jnp.maximum(
                jnp.abs(velocity.y[:, :-1, :]),
                jnp.abs(velocity.y[:, 1:, :]),
            )
            / self.y_metric.cell_widths(rank, dtype)
            + jnp.maximum(
                jnp.abs(velocity.z[:-1, ...]),
                jnp.abs(velocity.z[1:, ...]),
            )
            / self.z_metric.cell_widths(rank, dtype)
        )
        return jnp.max(local_rate)

    def timestep_for_cfl(
        self,
        velocity: MACVelocity,
        target_cfl: float,
        target_diffusive_cfl: float = 0.5,
    ) -> float:
        """Choose a step satisfying active explicit stability limits."""
        if not math.isfinite(target_cfl) or target_cfl <= 0.0:
            raise ValueError("target CFL must be positive and finite")
        if not math.isfinite(target_diffusive_cfl) or target_diffusive_cfl <= 0.0:
            raise ValueError("target diffusive CFL must be positive and finite")
        if self.config.sgs_time_integration == "imex_ark3":
            advective_rate, diffusive_rate = self._compiled_imex_timestep_rates(
                velocity,
                self._active_lasd_coefficient(velocity),
            )
            advective_rate = float(advective_rate)
            diffusive_rate = float(diffusive_rate)
        else:
            advective_rate, diffusive_rate = self._compiled_timestep_rates(
                velocity,
                self._active_lasd_coefficient(velocity),
            )
            advective_rate = float(advective_rate)
            diffusive_rate = float(diffusive_rate)
        if advective_rate <= 0.0:
            raise ValueError("cannot choose a CFL step for zero velocity")
        advective_step = target_cfl / advective_rate
        if diffusive_rate <= 0.0:
            return advective_step
        return min(
            advective_step,
            target_diffusive_cfl / diffusive_rate,
        )

    def horizontal_mean(self, field: Array, *, keepdims: bool = False) -> Array:
        """Return a physical-area horizontal mean at each z level."""

        return _horizontal_mean(
            field,
            self.x_metric,
            self.y_metric,
            keepdims=keepdims,
        )

    def surface_mean(self, field: Array) -> Array:
        """Return a physical-area mean of a horizontal surface field."""

        return _surface_mean(field, self.x_metric, self.y_metric)

    def volume_mean(self, field: Array) -> Array:
        """Return a physical-volume mean of a cell scalar."""

        return _volume_mean(field, self.metrics)

    def plane_mean_profile(self, velocity: MACVelocity) -> Array:
        return self.horizontal_mean(_cell_velocity(velocity)[..., 0])

    def plane_statistics(
        self,
        velocity: MACVelocity,
    ) -> tuple[Array, Array, Array]:
        """Return mean velocity, resolved TKE and minus-uw profiles."""
        cells = _cell_velocity(velocity)
        mean = self.horizontal_mean(cells)
        fluctuations = cells - mean[:, None, None, :]
        resolved_tke = 0.5 * self.horizontal_mean(
            jnp.sum(fluctuations * fluctuations, axis=-1)
        )
        minus_uw = -self.horizontal_mean(
            fluctuations[..., 0] * fluctuations[..., 2]
        )
        return mean, resolved_tke, minus_uw


class ScalarOperators:
    """Cell-centred passive scalar transported by a projected MAC velocity.

    Advection is conservative and uses the same centered flux plus configured
    local nonlinear correction as the neutral momentum path. The scalar AMD
    diffusivity is the minimum nonnegative diffusivity obtained from the local
    velocity and scalar gradients. A prescribed lower flux and zero-flux top
    close the vertical finite-volume balance exactly.
    """

    def __init__(
        self,
        grid: RectilinearGrid,
        model: ScalarConfig = ScalarConfig(),
    ) -> None:
        if grid.shape[0] < 2 or min(grid.shape[1:]) < 4:
            raise ValueError("scalar AMD requires nz>=2 and nx,ny>=4")
        self.grid = grid
        self.model = model
        dtype = jnp.asarray(grid.z_faces).dtype
        self.x_metric, self.y_metric, self.z_metric = _build_axis_metrics(grid, dtype)
        self.metrics = (self.x_metric, self.y_metric, self.z_metric)
        self.uniform_axes = tuple(metric.uniform for metric in self.metrics)
        self.uniform_z = self.z_metric.uniform
        self.dx = self.x_metric.spacing
        self.dy = self.y_metric.spacing
        self.dz = self.z_metric.spacing
        self.z_faces = self.z_metric.faces
        self.z_centers = self.z_metric.centers
        self.dz_cell = self.z_metric.widths
        self.dz_center = self.z_metric.center_gaps
        self._compiled_step = jax.jit(self._ssprk3_step)
        self._compiled_diffusive_rate = jax.jit(self.diffusive_rate)

    def gradient(self, scalar: Array) -> Array:
        self._validate_scalar(scalar)
        return jnp.stack(
            tuple(metric.derivative(scalar) for metric in self.metrics),
            axis=-1,
        )

    def amd_diffusivity(
        self,
        scalar: Array,
        velocity_gradient: Array,
        *,
        scalar_gradient: Array | None = None,
    ) -> Array:
        """Return filter-free minimum-dissipation scalar diffusivity."""
        if scalar_gradient is None:
            scalar_gradient = self.gradient(scalar)
        delta = _cell_length_scales(self.metrics, scalar.dtype)
        # The velocity gradient carries a further component axis, so the length
        # scale has to be inserted on the derivative direction explicitly rather
        # than left to trailing-axis broadcasting.
        weighted_velocity_gradient = velocity_gradient * delta[..., None, :]
        weighted_scalar_gradient = scalar_gradient * delta
        production = -jnp.einsum(
            "...ik,...k,...i->...",
            weighted_velocity_gradient,
            weighted_scalar_gradient,
            scalar_gradient,
        )
        denominator = jnp.einsum(
            "...i,...i->...",
            scalar_gradient,
            scalar_gradient,
        )
        epsilon = jnp.finfo(scalar.dtype).eps
        return (
            self.model.coefficient
            * jnp.maximum(production, 0.0)
            / jnp.maximum(denominator, epsilon)
            + self.model.molecular_diffusivity
        )

    def advective_tendency(
        self,
        scalar: Array,
        velocity: MACVelocity,
    ) -> Array:
        """Return centered advection plus the configured nonlinear correction."""
        return self.centered_advective_tendency(
            scalar,
            velocity,
        ) + self.advection_dissipation(scalar, velocity)

    def centered_advective_tendency(
        self,
        scalar: Array,
        velocity: MACVelocity,
    ) -> Array:
        """Return the conservative centered scalar-flux divergence."""
        self._validate_scalar(scalar)
        x_boundary = 0.5 * (scalar[..., -1] + scalar[..., 0])
        x_faces = jnp.concatenate(
            (
                x_boundary[..., None],
                0.5 * (scalar[..., :-1] + scalar[..., 1:]),
                x_boundary[..., None],
            ),
            axis=2,
        )
        y_boundary = 0.5 * (scalar[:, -1, :] + scalar[:, 0, :])
        y_faces = jnp.concatenate(
            (
                y_boundary[:, None, :],
                0.5 * (scalar[:, :-1, :] + scalar[:, 1:, :]),
                y_boundary[:, None, :],
            ),
            axis=1,
        )
        z_faces = jnp.zeros(
            (scalar.shape[0] + 1, *scalar.shape[1:]),
            dtype=scalar.dtype,
        )
        z_faces = z_faces.at[1:-1].set(0.5 * (scalar[:-1] + scalar[1:]))
        rank, dtype = scalar.ndim, scalar.dtype
        return -(
            (
                velocity.x[..., 1:] * x_faces[..., 1:]
                - velocity.x[..., :-1] * x_faces[..., :-1]
            )
            / self.x_metric.cell_widths(rank, dtype)
            + (
                velocity.y[:, 1:, :] * y_faces[:, 1:, :]
                - velocity.y[:, :-1, :] * y_faces[:, :-1, :]
            )
            / self.y_metric.cell_widths(rank, dtype)
            + (velocity.z[1:] * z_faces[1:] - velocity.z[:-1] * z_faces[:-1])
            / self.z_metric.cell_widths(rank, dtype)
        )

    def _reconstruction_dissipation(
        self,
        scalar: Array,
        velocity: MACVelocity,
    ) -> Array:
        self._validate_scalar(scalar)
        strength = self.model.mp5_dissipation_strength
        if strength <= 0.0:
            return jnp.zeros_like(scalar)
        directions = (
            (velocity.x[..., 1:], self.x_metric),
            (velocity.y[:, 1:, :], self.y_metric),
            (velocity.z[1:], self.z_metric),
        )
        return reconstruction_dissipation(scalar, directions, strength)

    def mp5_dissipation(
        self,
        scalar: Array,
        velocity: MACVelocity,
    ) -> Array:
        """Return only the conservative MP5/Rusanov scalar dissipation."""
        return self._reconstruction_dissipation(scalar, velocity)

    def advection_dissipation(
        self,
        scalar: Array,
        velocity: MACVelocity,
    ) -> Array:
        """Return the conservative MP5 scalar stabilization."""
        return self._reconstruction_dissipation(scalar, velocity)

    def vertical_advection_dissipation_flux(
        self,
        scalar: Array,
        velocity: MACVelocity,
    ) -> Array:
        """Return the configured numerical scalar flux on vertical faces."""
        self._validate_scalar(scalar)
        flux = jnp.zeros(
            (scalar.shape[0] + 1, *scalar.shape[1:]),
            dtype=scalar.dtype,
        )
        upper_flux = reconstruction_flux(
            scalar,
            velocity.z[1:],
            self.z_metric,
            self.model.mp5_dissipation_strength,
        )
        return flux.at[1:].set(upper_flux)

    def sgs_fluxes(
        self,
        scalar: Array,
        velocity_gradient: Array,
        *,
        lower_surface_flux: Array | float | None = None,
        upper_surface_flux: Array | float | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Return cell diffusivity, cell gradients, and three SGS fluxes."""
        gradient = self.gradient(scalar)
        diffusivity = self.amd_diffusivity(
            scalar,
            velocity_gradient,
            scalar_gradient=gradient,
        )
        flux_x = -diffusivity * gradient[..., 0]
        flux_y = -diffusivity * gradient[..., 1]
        face_diffusivity = jnp.zeros(
            (scalar.shape[0] + 1, *scalar.shape[1:]),
            dtype=scalar.dtype,
        )
        face_diffusivity = face_diffusivity.at[1:-1].set(
            _interpolate_to_vertical_faces(
                diffusivity,
                self.z_centers,
                self.z_faces,
            )
        )
        face_diffusivity = face_diffusivity.at[0].set(diffusivity[0])
        face_diffusivity = face_diffusivity.at[-1].set(diffusivity[-1])
        flux_z = jnp.zeros_like(face_diffusivity)
        flux_z = flux_z.at[1:-1].set(
            -face_diffusivity[1:-1]
            * (scalar[1:] - scalar[:-1])
            / _z_vector(self.dz_center, scalar.ndim, scalar.dtype)
        )
        lower_flux = (
            self.model.lower_surface_flux
            if lower_surface_flux is None
            else lower_surface_flux
        )
        upper_flux = (
            self.model.upper_surface_flux
            if upper_surface_flux is None
            else upper_surface_flux
        )
        flux_z = flux_z.at[0].set(lower_flux)
        flux_z = flux_z.at[-1].set(upper_flux)
        return diffusivity, gradient, flux_x, flux_y, flux_z

    def sgs_tendency(
        self,
        scalar: Array,
        velocity_gradient: Array,
        *,
        lower_surface_flux: Array | float | None = None,
        upper_surface_flux: Array | float | None = None,
    ) -> Array:
        _, _, flux_x, flux_y, flux_z = self.sgs_fluxes(
            scalar,
            velocity_gradient,
            lower_surface_flux=lower_surface_flux,
            upper_surface_flux=upper_surface_flux,
        )
        # The horizontal flux lives at cell centres, so its divergence uses the
        # volume-weighted adjoint of the same gradient that produced it.  That
        # is what keeps the AMD flux variance dissipative on a stretched axis;
        # on a uniform periodic axis it is the fourth-order difference.
        return -(
            self.x_metric.negative_derivative_transpose(flux_x)
            + self.y_metric.negative_derivative_transpose(flux_y)
            + (flux_z[1:] - flux_z[:-1])
            / self.z_metric.cell_widths(scalar.ndim, scalar.dtype)
        )

    def _cell_velocity_gradient(self, cells: Array) -> Array:
        return jnp.stack(
            tuple(
                jnp.stack(
                    tuple(
                        metric.derivative(cells[..., component])
                        for metric in self.metrics
                    ),
                    axis=-1,
                )
                for component in range(3)
            ),
            axis=-2,
        )

    def tendency(
        self,
        scalar: Array,
        velocity: MACVelocity,
        *,
        lower_surface_flux: Array | float | None = None,
        upper_surface_flux: Array | float | None = None,
    ) -> Array:
        cells = _cell_velocity(velocity)
        velocity_gradient = self._cell_velocity_gradient(cells)
        return self.advective_tendency(
            scalar,
            velocity,
        ) + self.sgs_tendency(
            scalar,
            velocity_gradient,
            lower_surface_flux=lower_surface_flux,
            upper_surface_flux=upper_surface_flux,
        )

    def _ssprk3_step(
        self,
        scalar: Array,
        velocity: MACVelocity,
        timestep: Array,
    ) -> Array:
        first_tendency = self.tendency(scalar, velocity)
        first = scalar + timestep * first_tendency
        second_tendency = self.tendency(first, velocity)
        second = scalar + 0.25 * timestep * (first_tendency + second_tendency)
        third_tendency = self.tendency(second, velocity)
        return scalar + timestep * (
            first_tendency / 6.0 + second_tendency / 6.0 + (2.0 / 3.0) * third_tendency
        )

    def step(
        self,
        scalar: Array,
        velocity: MACVelocity,
        timestep: float,
    ) -> Array:
        """Advance one frozen-velocity SSPRK3 scalar step."""
        if not math.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("scalar timestep must be positive and finite")
        self._validate_scalar(scalar)
        return self._compiled_step(
            scalar,
            velocity,
            jnp.asarray(timestep, dtype=scalar.dtype),
        )

    def diffusive_rate(self, scalar: Array, velocity: MACVelocity) -> Array:
        velocity_gradient = self._cell_velocity_gradient(_cell_velocity(velocity))
        diffusivity = self.amd_diffusivity(scalar, velocity_gradient)
        return self.explicit_diffusion_rate(diffusivity)

    def explicit_diffusion_rate(self, diffusivity: Array) -> Array:
        """Return the maximum local finite-volume diffusion diagonal."""
        rank, dtype = diffusivity.ndim, diffusivity.dtype
        rate = diffusivity * (
            self.x_metric.broadcast(self.x_metric.diffusion_diagonal, rank, dtype)
            + self.y_metric.broadcast(self.y_metric.diffusion_diagonal, rank, dtype)
        )
        face_diffusivity = _interpolate_to_vertical_faces(
            diffusivity,
            self.z_centers,
            self.z_faces,
        )
        vertical = jnp.zeros_like(diffusivity)
        vertical = vertical.at[:-1].add(
            face_diffusivity
            / _z_vector(self.dz_cell[:-1] * self.dz_center, rank, dtype)
        )
        vertical = vertical.at[1:].add(
            face_diffusivity / _z_vector(self.dz_cell[1:] * self.dz_center, rank, dtype)
        )
        return jnp.max(rate + vertical)

    def volume_mean(self, scalar: Array) -> Array:
        """Return the finite-volume mean on a rectilinear mesh."""
        self._validate_scalar(scalar)
        plane_mean = _horizontal_mean(scalar, self.x_metric, self.y_metric)
        normalized_volume = self.dz_cell / jnp.sum(self.dz_cell)
        return jnp.sum(normalized_volume * plane_mean)

    def timestep_for_diffusive_cfl(
        self,
        scalar: Array,
        velocity: MACVelocity,
        target_diffusive_cfl: float,
    ) -> float:
        if not math.isfinite(target_diffusive_cfl) or target_diffusive_cfl <= 0.0:
            raise ValueError("scalar diffusive CFL target must be positive")
        rate = float(self._compiled_diffusive_rate(scalar, velocity))
        return math.inf if rate <= 0.0 else target_diffusive_cfl / rate

    def _validate_scalar(self, scalar: Array) -> None:
        if tuple(scalar.shape) != self.grid.shape:
            raise ValueError(
                f"expected scalar shape {self.grid.shape}, got {tuple(scalar.shape)}"
            )


__all__ = [
    "AMDModel",
    "MomentumConfig",
    "MomentumDiagnostic",
    "MomentumOperators",
    "PreparedIMEXStep",
    "ScalarConfig",
    "ScalarOperators",
    "WallModelState",
]
