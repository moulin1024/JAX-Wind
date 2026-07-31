"""Neutral pressure-driven ABL momentum on a uniform MAC grid."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from jaxwind.pressure import (
    fpj2_pressure_prediction,
    fpj2_ssprk3_velocity_step,
    MACStageProjector,
    MACVelocity,
    MatrixFreePoissonSolver,
    RectilinearGrid,
    VelocityPressureProjection,
    mac_divergence,
    mac_pressure_gradient,
    projected_ssprk3_velocity_pressure_step,
    projected_ssprk3_step,
    projected_ssprk3_velocity_step,
)

from .lasd import LASDModel, LASDState, PhysicalSpaceLASD
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
            raise ValueError(
                "molecular viscosity must be finite and nonnegative"
            )


@dataclass(frozen=True, slots=True)
class NeutralABLConfig:
    """Physical and temporal controls for the minimal neutral ABL."""

    friction_velocity: float = 0.1
    roughness_length: float = 1.0e-3
    von_karman: float = 0.4
    wall_matching_level: int = 0
    wall_filter_width: float | None = None
    wall_temporal_filter_timescale: float | None = None
    mp5_dissipation_strength: float = 1.0
    geostrophic_wind: tuple[float, float] | None = None
    coriolis_vertical: float = 0.0
    coriolis_horizontal: float = 0.0
    amd: AMDModel = AMDModel()
    lasd: LASDModel | None = None
    sgs_time_integration: Literal["explicit", "imex_ark3"] = "explicit"
    projection_method: Literal["full", "fpj2"] = "full"
    fpj2_timestep_ratio_limit: float = 2.0

    def __post_init__(self) -> None:
        if self.friction_velocity <= 0.0:
            raise ValueError("friction velocity must be positive")
        if self.roughness_length <= 0.0:
            raise ValueError("roughness length must be positive")
        if self.von_karman <= 0.0:
            raise ValueError("von Karman constant must be positive")
        if (
            isinstance(self.wall_matching_level, bool)
            or not isinstance(self.wall_matching_level, int)
            or self.wall_matching_level < 0
        ):
            raise ValueError("wall matching level must be a nonnegative integer")
        if self.wall_filter_width is not None and (
            not math.isfinite(self.wall_filter_width)
            or self.wall_filter_width <= 0.0
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
            raise ValueError(
                "MP5 dissipation strength must be finite and nonnegative"
            )
        if self.geostrophic_wind is not None and not all(
            math.isfinite(component) for component in self.geostrophic_wind
        ):
            raise ValueError("geostrophic wind components must be finite")
        if not math.isfinite(self.coriolis_vertical):
            raise ValueError("vertical Coriolis parameter must be finite")
        if not math.isfinite(self.coriolis_horizontal):
            raise ValueError("horizontal Coriolis parameter must be finite")
        if self.projection_method not in {"full", "fpj2"}:
            raise ValueError("projection method must be 'full' or 'fpj2'")
        if self.sgs_time_integration not in {"explicit", "imex_ark3"}:
            raise ValueError(
                "SGS time integration must be 'explicit' or 'imex_ark3'"
            )
        if (
            not math.isfinite(self.fpj2_timestep_ratio_limit)
            or self.fpj2_timestep_ratio_limit < 1.0
        ):
            raise ValueError("FPJ-2 timestep ratio limit must be at least one")


class FPJ2State(NamedTuple):
    """Two accepted pseudo-pressures and their actual step sizes."""

    current_pressure: Array
    previous_pressure: Array
    current_timestep: float
    previous_timestep: float
    history_count: int


class WallModelState(NamedTuple):
    """Accepted-step memory for the temporally filtered wall input."""

    filtered_velocity: Array


@dataclass(frozen=True, slots=True)
class NeutralABLDiagnostic:
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


def _uniform_spacing(faces: tuple[float, ...], name: str) -> float:
    reference = (faces[-1] - faces[0]) / (len(faces) - 1)
    tolerance = 1.0e-12 * max(1.0, abs(reference))
    if not all(
        math.isclose(
            right - left,
            reference,
            rel_tol=1.0e-12,
            abs_tol=tolerance,
        )
        for left, right in zip(faces[:-1], faces[1:], strict=True)
    ):
        raise ValueError(f"KEP4 momentum currently requires uniform {name}")
    return reference


def _periodic_d4(field: Array, spacing: float, axis: int) -> Array:
    return (
        -jnp.roll(field, -2, axis=axis)
        + 8.0 * jnp.roll(field, -1, axis=axis)
        - 8.0 * jnp.roll(field, 1, axis=axis)
        + jnp.roll(field, 2, axis=axis)
    ) / (12.0 * spacing)


def _wall_normal_derivative(field: Array, spacing: float) -> Array:
    if field.shape[0] == 1:
        return jnp.zeros_like(field)
    derivative = jnp.zeros_like(field)
    if field.shape[0] > 2:
        derivative = derivative.at[1:-1].set(
            (field[2:] - field[:-2]) / (2.0 * spacing)
        )
    derivative = derivative.at[0].set((field[1] - field[0]) / spacing)
    derivative = derivative.at[-1].set(
        (field[-1] - field[-2]) / spacing
    )
    return derivative


def _wall_normal_derivative_transpose(
    field: Array,
    spacing: float,
) -> Array:
    """Apply the Euclidean transpose of ``_wall_normal_derivative``."""
    if field.shape[0] == 1:
        return jnp.zeros_like(field)
    result = jnp.zeros_like(field)
    result = result.at[0].add(-field[0] / spacing)
    result = result.at[1].add(field[0] / spacing)
    if field.shape[0] > 2:
        result = result.at[:-2].add(-field[1:-1] / (2.0 * spacing))
        result = result.at[2:].add(field[1:-1] / (2.0 * spacing))
    result = result.at[-2].add(-field[-1] / spacing)
    result = result.at[-1].add(field[-1] / spacing)
    return result


def _minmod(*values: Array) -> Array:
    stacked = jnp.stack(values)
    magnitude = jnp.min(jnp.abs(stacked), axis=0)
    all_positive = jnp.all(stacked > 0.0, axis=0)
    all_negative = jnp.all(stacked < 0.0, axis=0)
    return jnp.where(
        all_positive,
        magnitude,
        jnp.where(all_negative, -magnitude, 0.0),
    )


def _mp5_reconstruct(
    vm2: Array,
    vm1: Array,
    value: Array,
    vp1: Array,
    vp2: Array,
) -> Array:
    """Suresh-Huynh MP5 value on the face to the right of ``value``."""
    unlimited = (
        2.0 * vm2
        - 13.0 * vm1
        + 47.0 * value
        + 27.0 * vp1
        - 3.0 * vp2
    ) / 60.0
    alpha = 4.0
    monotone = value + _minmod(
        vp1 - value,
        alpha * (value - vm1),
    )

    dm1 = vm2 - 2.0 * vm1 + value
    d0 = vm1 - 2.0 * value + vp1
    dp1 = value - 2.0 * vp1 + vp2
    curvature_plus = _minmod(
        4.0 * d0 - dp1,
        4.0 * dp1 - d0,
        d0,
        dp1,
    )
    curvature_minus = _minmod(
        4.0 * d0 - dm1,
        4.0 * dm1 - d0,
        d0,
        dm1,
    )
    upper_left = value + alpha * (value - vm1)
    average = 0.5 * (value + vp1)
    median = average - 0.5 * curvature_plus
    large_curvature = (
        value + 0.5 * (value - vm1) + (4.0 / 3.0) * curvature_minus
    )
    lower = jnp.maximum(
        jnp.minimum(jnp.minimum(value, vp1), median),
        jnp.minimum(jnp.minimum(value, upper_left), large_curvature),
    )
    upper = jnp.minimum(
        jnp.maximum(jnp.maximum(value, vp1), median),
        jnp.maximum(jnp.maximum(value, upper_left), large_curvature),
    )
    limited = unlimited + _minmod(
        lower - unlimited,
        upper - unlimited,
    )
    tolerance = 10.0 * jnp.finfo(value.dtype).eps * jnp.maximum(
        1.0,
        jnp.maximum(jnp.abs(value), jnp.abs(unlimited)),
    )
    admissible = (
        (unlimited - value) * (unlimited - monotone) <= tolerance
    )
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


def _mp5_interface_states(
    field: Array,
    *,
    axis: int,
    periodic: bool,
) -> tuple[Array, Array]:
    """Return left/right MP5 states on every face after a cell."""
    values = jnp.moveaxis(field, axis, -1)

    def sample(offset: int) -> Array:
        return _sample_offset(values, offset, periodic=periodic)

    left = _mp5_reconstruct(
        sample(-2),
        sample(-1),
        sample(0),
        sample(1),
        sample(2),
    )
    right = _mp5_reconstruct(
        sample(3),
        sample(2),
        sample(1),
        sample(0),
        sample(-1),
    )
    return jnp.moveaxis(left, -1, axis), jnp.moveaxis(right, -1, axis)


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
    x = x.at[..., 1:-1].set(
        0.5 * (tendency[..., :-1, 0] + tendency[..., 1:, 0])
    )
    x_boundary = 0.5 * (
        tendency[..., -1, 0] + tendency[..., 0, 0]
    )
    x = x.at[..., 0].set(x_boundary)
    x = x.at[..., -1].set(x_boundary)

    y = jnp.zeros((nz, ny + 1, nx), dtype=tendency.dtype)
    y = y.at[:, 1:-1, :].set(
        0.5 * (tendency[:, :-1, :, 1] + tendency[:, 1:, :, 1])
    )
    y_boundary = 0.5 * (
        tendency[:, -1, :, 1] + tendency[:, 0, :, 1]
    )
    y = y.at[:, 0, :].set(y_boundary)
    y = y.at[:, -1, :].set(y_boundary)

    z = jnp.zeros((nz + 1, ny, nx), dtype=tendency.dtype)
    if nz > 1:
        z = z.at[1:-1].set(
            0.5 * (tendency[:-1, ..., 2] + tendency[1:, ..., 2])
        )
    return MACVelocity(x, y, z)


def _velocity_sum(*terms: tuple[float, MACVelocity]) -> MACVelocity:
    return MACVelocity(
        sum(weight * velocity.x for weight, velocity in terms),
        sum(weight * velocity.y for weight, velocity in terms),
        sum(weight * velocity.z for weight, velocity in terms),
    )


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


class NeutralABLMomentum:
    """Neutral ABL momentum with projected explicit and IMEX time steps."""

    def __init__(
        self,
        grid: RectilinearGrid,
        pressure_solver: MatrixFreePoissonSolver,
        config: NeutralABLConfig = NeutralABLConfig(),
    ) -> None:
        if pressure_solver.operator.grid != grid:
            raise ValueError("pressure and momentum grids must match")
        if grid.shape[0] < 2 or min(grid.shape[1:]) < 4:
            raise ValueError("KEP4 requires nz>=2 and nx,ny>=4")
        self.grid = grid
        self.pressure_solver = pressure_solver
        self.projector = MACStageProjector(pressure_solver)
        self.config = config
        self.dx = _uniform_spacing(grid.x_faces, "x spacing")
        self.dy = _uniform_spacing(grid.y_faces, "y spacing")
        self.dz = _uniform_spacing(grid.z_faces, "z spacing")
        if config.wall_matching_level >= grid.shape[0]:
            raise ValueError("wall matching level must lie inside the grid")
        if config.roughness_length >= 0.5 * self.dz:
            raise ValueError("roughness must lie below the first cell centre")
        self.height = grid.z_faces[-1] - grid.z_faces[0]
        self.wall_law = NeutralLogWallLaw(
            config.roughness_length,
            config.von_karman,
        )
        self.pressure_acceleration = (
            0.0
            if config.geostrophic_wind is not None
            else config.friction_velocity**2 / self.height
        )
        self.lasd_closure = (
            None
            if config.lasd is None
            else PhysicalSpaceLASD(
                dx=self.dx,
                dy=self.dy,
                dz=self.dz,
                model=config.lasd,
            )
        )
        self._lasd_state: LASDState | None = None
        self._lasd_step = 0
        self._lasd_interval_time = 0.0
        self._fpj2_state: FPJ2State | None = None
        self._wall_model_state: WallModelState | None = None

        def compiled_tendency(
            velocity: MACVelocity,
            lasd_coefficient: Array,
            wall_velocity: Array,
        ) -> MACVelocity:
            return _cells_to_faces(
                self.cell_tendency(
                    velocity,
                    lasd_coefficient,
                    wall_velocity=wall_velocity,
                )
            )

        self._compiled_tendency = jax.jit(compiled_tendency)

        def imex_tendencies_from_gradient(
            velocity: MACVelocity,
            cells: Array,
            gradient: Array,
            frozen_viscosity: Array,
            lasd_coefficient: Array,
            wall_velocity: Array,
        ) -> tuple[MACVelocity, MACVelocity]:
            principal, cross = self.sgs_split_tendency(
                cells,
                frozen_viscosity,
                lasd_coefficient,
                gradient=gradient,
                wall_velocity=wall_velocity,
            )
            explicit = (
                self.conservative_advection(velocity, cells)
                + cross
                + self.forcing_tendency(cells)
            )
            if self.config.mp5_dissipation_strength > 0.0:
                explicit += self.mp5_dissipation(velocity, cells)
            return _cells_to_faces(explicit), _cells_to_faces(principal)

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
                wall_velocity,
            )
            return explicit, implicit, frozen_viscosity

        self._compiled_imex_initial_tendencies = jax.jit(
            compiled_imex_initial_tendencies
        )

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
                wall_velocity,
            )

        self._compiled_imex_tendencies = jax.jit(
            compiled_imex_tendencies
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

        self._compiled_implicit_diffusion = jax.jit(
            compiled_implicit_diffusion
        )

        def compiled_diagnostic(
            velocity: MACVelocity,
            timestep: Array,
            lasd_coefficient: Array,
            wall_velocity: Array,
        ) -> tuple[Array, ...]:
            cells = _cell_velocity(velocity)
            viscosity = self.sgs_viscosity(cells, lasd_coefficient)
            energy = 0.5 * jnp.mean(
                jnp.sum(cells * cells, axis=-1)
            )
            divergence = mac_divergence(velocity, self.grid)
            if self.lasd_closure is None:
                mean_coefficient = jnp.asarray(
                    self.config.amd.coefficient,
                    dtype=cells.dtype,
                )
                maximum_coefficient = mean_coefficient
                clipped_fraction = jnp.asarray(0.0, dtype=cells.dtype)
            else:
                mean_coefficient = jnp.mean(lasd_coefficient)
                maximum_coefficient = jnp.max(lasd_coefficient)
                clipped_fraction = jnp.mean(
                    lasd_coefficient
                    >= 0.999 * self.config.lasd.maximum_coefficient
                )
            return (
                energy,
                timestep * self.cfl_rate(velocity),
                timestep
                * 2.0
                * jnp.max(viscosity)
                * (
                    1.0 / self.dx**2
                    + 1.0 / self.dy**2
                    + 1.0 / self.dz**2
                ),
                self.pressure_solver.operator.norm(divergence),
                jnp.mean(
                    self.wall_ustar(cells, wall_velocity=wall_velocity)
                ),
                jnp.mean(viscosity),
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
            diffusion_rate = (
                2.0
                * jnp.max(viscosity)
                * (
                    1.0 / self.dx**2
                    + 1.0 / self.dy**2
                    + 1.0 / self.dz**2
                )
            )
            return self.cfl_rate(velocity), diffusion_rate

        self._compiled_timestep_rates = jax.jit(compiled_timestep_rates)

        def compiled_imex_timestep_rates(
            velocity: MACVelocity,
            lasd_coefficient: Array,
        ) -> tuple[Array, Array]:
            cells = _cell_velocity(velocity)
            viscosity = self.sgs_viscosity(cells, lasd_coefficient)
            explicit_horizontal_diffusion_rate = (
                2.0
                * jnp.max(viscosity)
                * (1.0 / self.dx**2 + 1.0 / self.dy**2)
            )
            return self.cfl_rate(velocity), explicit_horizontal_diffusion_rate

        self._compiled_imex_timestep_rates = jax.jit(
            compiled_imex_timestep_rates
        )
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
                        _periodic_d4(value, self.dx, -1),
                        _periodic_d4(value, self.dy, -2),
                        _wall_normal_derivative(value, self.dz),
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
        if direction == 0:
            return _periodic_d4(field, self.dx, -1)
        if direction == 1:
            return _periodic_d4(field, self.dy, -2)
        return -_wall_normal_derivative_transpose(field, self.dz)

    def principal_sgs_tendency(
        self,
        cell_velocity: Array,
        frozen_viscosity: Array,
    ) -> Array:
        """Return conservative vertical principal SGS diffusion."""
        if cell_velocity.shape[0] == 1:
            return jnp.zeros_like(cell_velocity)
        face_viscosity = 0.5 * (
            frozen_viscosity[:-1] + frozen_viscosity[1:]
        )
        flux = (
            face_viscosity[..., None]
            * (cell_velocity[1:] - cell_velocity[:-1])
            / self.dz
        )
        result = jnp.zeros_like(cell_velocity)
        result = result.at[:-1].add(flux / self.dz)
        result = result.at[1:].add(-flux / self.dz)
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
        vertical_tendency = -_wall_normal_derivative_transpose(
            vertical_cell_stress,
            self.dz,
        )
        lower = jnp.zeros_like(vertical_tendency[:1])
        return jnp.concatenate(
            (
                lower,
                self.dz * jnp.cumsum(vertical_tendency, axis=0),
            ),
            axis=0,
        )

    def _vertical_stress_divergence(self, face_stress: Array) -> Array:
        return (face_stress[1:] - face_stress[:-1]) / self.dz

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
        vertical_faces = self._vertical_stress_faces_from_cell_stress(
            stress[..., :, 2]
        )
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
            wall_stress=self.wall_stress(
                cell_velocity,
                wall_velocity=wall_velocity,
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
        factor = implicit_timestep / (self.dz * self.dz)
        face_viscosity = 0.5 * (
            frozen_viscosity[:-1] + frozen_viscosity[1:]
        )
        lower = jnp.zeros_like(frozen_viscosity)
        upper = jnp.zeros_like(frozen_viscosity)
        lower = lower.at[1:].set(-factor * face_viscosity)
        upper = upper.at[:-1].set(-factor * face_viscosity)
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
        delta = jnp.asarray(
            (self.dx, self.dy, self.dz),
            dtype=gradient.dtype,
        )
        weighted_gradient = gradient * delta
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
        eddy = self.config.amd.coefficient * jnp.maximum(
            production,
            0.0,
        ) / jnp.maximum(denominator, epsilon)
        return eddy + self.config.amd.molecular_viscosity

    def _amd_stress_from_gradient(self, gradient: Array) -> Array:
        strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
        viscosity = self._amd_viscosity_from_gradient(gradient)
        return 2.0 * viscosity[..., None, None] * strain

    def amd_viscosity(self, cell_velocity: Array) -> Array:
        return self._amd_viscosity_from_gradient(
            self.velocity_gradient(cell_velocity)
        )

    def skew_advection(
        self,
        cell_velocity: Array,
        *,
        gradient: Array | None = None,
    ) -> Array:
        if gradient is None:
            gradient = self.velocity_gradient(cell_velocity)
        tendency = []

        def derivative(value: Array, axis: int) -> Array:
            return _periodic_d4(
                value,
                self.dx if axis == -1 else self.dy,
                axis,
            )

        for component in range(3):
            transported = cell_velocity[..., component]
            total = jnp.zeros_like(transported)
            for direction, axis in ((0, -1), (1, -2)):
                advector = cell_velocity[..., direction]
                total += 0.5 * (
                    advector * gradient[..., component, direction]
                    + derivative(advector * transported, axis)
                )
            vertical = cell_velocity[..., 2]
            total += 0.5 * (
                vertical * gradient[..., component, 2]
                + _wall_normal_derivative(
                    vertical * transported,
                    self.dz,
                )
            )
            tendency.append(-total)
        return jnp.stack(tendency, axis=-1)

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
        return -(
            (x_flux[:, :, 1:, :] - x_flux[:, :, :-1, :]) / self.dx
            + (y_flux[:, 1:, :, :] - y_flux[:, :-1, :, :]) / self.dy
            + (z_flux[1:] - z_flux[:-1]) / self.dz
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
            velocity.z[1:-1, ..., None]
            * 0.5
            * (cells[:-1] + cells[1:])
        )

    def amd_tendency(self, cell_velocity: Array) -> Array:
        stress = self.amd_stress(cell_velocity)
        result = []
        for component in range(3):
            divergence = (
                _periodic_d4(stress[..., component, 0], self.dx, -1)
                + _periodic_d4(stress[..., component, 1], self.dy, -2)
                + _wall_normal_derivative(
                    stress[..., component, 2],
                    self.dz,
                )
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
            wall_stress=self.wall_stress(
                cell_velocity,
                wall_velocity=wall_velocity,
            ),
        )

    def vertical_sgs_stress_flux(
        self,
        cell_velocity: Array,
        lasd_coefficient: Array | None = None,
        *,
        gradient: Array | None = None,
        wall_velocity: Array | None = None,
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
        faces = self._vertical_stress_faces_from_cell_stress(
            stress[..., :, 2]
        )
        return faces.at[0].add(
            self.wall_stress(
                cell_velocity,
                wall_velocity=wall_velocity,
            )
        )

    def mp5_dissipation(
        self,
        velocity: MACVelocity,
        cell_velocity: Array | None = None,
    ) -> Array:
        """Return local MP5/Rusanov face dissipation for all momenta."""
        cells = _cell_velocity(velocity) if cell_velocity is None else cell_velocity
        face_speeds = (
            velocity.x[..., 1:],
            velocity.y[:, 1:, :],
            velocity.z[1:, ...],
        )
        directions = (
            (-1, self.dx, True),
            (-2, self.dy, True),
            (-3, self.dz, False),
        )
        result = jnp.zeros_like(cells)
        strength = self.config.mp5_dissipation_strength
        for face_speed, (axis, spacing, periodic) in zip(
            face_speeds,
            directions,
        ):
            left, right = _mp5_interface_states(
                cells,
                axis=axis - 1,
                periodic=periodic,
            )
            dissipative_flux = (
                -0.5
                * strength
                * jnp.abs(face_speed)[..., None]
                * (right - left)
            )
            if periodic:
                previous_flux = jnp.roll(
                    dissipative_flux,
                    1,
                    axis=axis - 1,
                )
            else:
                previous_flux = jnp.concatenate(
                    (
                        jnp.zeros_like(dissipative_flux[:1]),
                        dissipative_flux[:-1],
                    ),
                    axis=0,
                )
            result -= (dissipative_flux - previous_flux) / spacing
        return result

    @property
    def wall_matching_height(self) -> float:
        return (self.config.wall_matching_level + 0.5) * self.dz

    def instantaneous_wall_velocity(self, cell_velocity: Array) -> Array:
        """Sample and spatially filter the wall-model matching velocity."""
        horizontal = cell_velocity[
            self.config.wall_matching_level,
            ...,
            :2,
        ]
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
            self.wall_matching_height,
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
            self.wall_matching_height,
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
            tendency = tendency.at[..., 1].add(
                -vertical * (u - geostrophic_u)
            )
            tendency = tendency.at[..., 2].add(horizontal * u)
        return tendency

    def cell_tendency(
        self,
        velocity: MACVelocity,
        lasd_coefficient: Array | None = None,
        *,
        wall_velocity: Array | None = None,
    ) -> Array:
        cells = _cell_velocity(velocity)
        gradient = self.velocity_gradient(cells)
        tendency = (
            self.conservative_advection(velocity, cells)
            + self.sgs_tendency(
                cells,
                lasd_coefficient,
                gradient=gradient,
                wall_velocity=wall_velocity,
            )
            + self.forcing_tendency(cells)
        )
        if self.config.mp5_dissipation_strength > 0.0:
            tendency += self.mp5_dissipation(velocity, cells)
        return tendency

    def tendency(self, velocity: MACVelocity, _time: float) -> MACVelocity:
        return self._compiled_tendency(
            velocity,
            self._active_lasd_coefficient(velocity),
            self.active_wall_velocity(velocity),
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
        self._lasd_state = self.lasd_closure.initialize(
            _cell_velocity(velocity)
        )
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
        instantaneous = self.instantaneous_wall_velocity(
            _cell_velocity(velocity)
        )
        filtered = (
            (1.0 - epsilon) * self._wall_model_state.filtered_velocity
            + epsilon * instantaneous
        )
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
        expected = self.grid.shape
        if any(field.shape != expected for field in state):
            raise ValueError("LASD checkpoint field shape does not match the grid")
        self._lasd_state = state
        self._lasd_step = accepted_step
        self._lasd_interval_time = interval_time

    @property
    def fpj2_state(self) -> FPJ2State | None:
        """Return accepted FPJ-2 pressure history, if available."""
        return self._fpj2_state

    def reset_fpj2(self) -> None:
        """Discard pressure history so that two exact startup steps are used."""
        self._fpj2_state = None

    def restore_fpj2(self, state: FPJ2State) -> None:
        """Restore gauge-fixed pressure history from a checkpoint."""
        expected = self.grid.shape
        if (
            tuple(state.current_pressure.shape) != expected
            or tuple(state.previous_pressure.shape) != expected
        ):
            raise ValueError("FPJ-2 checkpoint pressure shape does not match grid")
        if min(state.current_timestep, state.previous_timestep) <= 0.0:
            raise ValueError("FPJ-2 checkpoint timesteps must be positive")
        if state.history_count not in {1, 2}:
            raise ValueError("FPJ-2 history count must be one or two")
        self._fpj2_state = state

    def _accept_fpj2_pressure(
        self,
        pressure: Array,
        timestep: float,
    ) -> None:
        pressure = self.pressure_solver.operator.project_nullspace(pressure)
        if self._fpj2_state is None:
            self._fpj2_state = FPJ2State(
                pressure,
                pressure,
                timestep,
                timestep,
                1,
            )
            return
        old = self._fpj2_state
        self._fpj2_state = FPJ2State(
            pressure,
            old.current_pressure,
            timestep,
            old.current_timestep,
            2,
        )

    def _fpj2_history_is_usable(self, timestep: float) -> bool:
        state = self._fpj2_state
        if state is None or state.history_count < 2:
            return False
        limit = self.config.fpj2_timestep_ratio_limit
        ratio = timestep / state.current_timestep
        return 1.0 / limit <= ratio <= limit

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

    def _imex_ark3_step(
        self,
        velocity: MACVelocity,
        *,
        timestep: float,
        time: float,
        lasd_coefficient: Array,
        wall_velocity: Array,
    ) -> VelocityPressureProjection:
        """Advance ARS(2,3,3) with frozen vertical SGS diffusion implicit."""
        initial_explicit, initial_implicit, frozen_viscosity = (
            self._compiled_imex_initial_tendencies(
                velocity,
                lasd_coefficient,
                wall_velocity,
            )
        )
        explicit_tendencies: list[MACVelocity] = [initial_explicit]
        implicit_tendencies: list[MACVelocity] = [initial_implicit]

        def evaluate(stage_velocity: MACVelocity) -> None:
            explicit, implicit = self._compiled_imex_tendencies(
                stage_velocity,
                frozen_viscosity,
                lasd_coefficient,
                wall_velocity,
            )
            explicit_tendencies.append(explicit)
            implicit_tendencies.append(implicit)

        history = self._fpj2_state
        use_fast_projection = (
            self.config.projection_method == "fpj2"
            and self._fpj2_history_is_usable(timestep)
        )
        predicted_pressures = (
            tuple(
                fpj2_pressure_prediction(
                    history.current_pressure,
                    history.previous_pressure,
                    current_timestep=history.current_timestep,
                    previous_timestep=history.previous_timestep,
                    next_timestep=timestep,
                    stage_abscissa=_ARK3_C[index],
                )
                for index in range(1, len(_ARK3_C))
            )
            if use_fast_projection and history is not None
            else ()
        )
        pressure_guess = (
            None if history is None else history.current_pressure
        )

        for stage_index in range(1, len(_ARK3_C)):
            terms: list[tuple[float, MACVelocity]] = [(1.0, velocity)]
            for previous in range(stage_index):
                explicit_weight = (
                    timestep
                    * _ARK3_EXPLICIT_A[stage_index][previous]
                )
                implicit_weight = (
                    timestep
                    * _ARK3_IMPLICIT_A[stage_index][previous]
                )
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
            if use_fast_projection:
                gradient = mac_pressure_gradient(
                    predicted_pressures[stage_index - 1],
                    self.grid,
                    self.pressure_solver.operator.boundaries,
                )
                stage = self.enforce_boundaries(
                    _velocity_sum(
                        (1.0, stage),
                        (-projection_timestep, gradient),
                    )
                )
            else:
                projected = self.projector.project_velocity_and_pressure(
                    stage,
                    timestep=projection_timestep,
                    initial_pressure=pressure_guess,
                )
                stage = projected.velocity
                pressure_guess = projected.pressure
            evaluate(stage)

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
        initial_pressure = (
            predicted_pressures[-1]
            if use_fast_projection
            else pressure_guess
        )
        return self.projector.project_velocity_and_pressure(
            _velocity_sum(*final_terms),
            timestep=timestep,
            initial_pressure=initial_pressure,
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
    def enforce_boundaries(velocity: MACVelocity) -> MACVelocity:
        x_boundary = 0.5 * (velocity.x[..., 0] + velocity.x[..., -1])
        y_boundary = 0.5 * (velocity.y[:, 0, :] + velocity.y[:, -1, :])
        return MACVelocity(
            velocity.x.at[..., 0].set(x_boundary).at[..., -1].set(
                x_boundary
            ),
            velocity.y.at[:, 0, :].set(y_boundary).at[:, -1, :].set(
                y_boundary
            ),
            velocity.z.at[0].set(0.0).at[-1].set(0.0),
        )

    def initial_log_profile(
        self,
        *,
        perturbation_amplitude: float = 0.05,
    ) -> MACVelocity:
        nz, ny, nx = self.grid.shape
        z = (
            jnp.arange(nz, dtype=self.pressure_solver.operator.dtype) + 0.5
        ) * self.dz
        mean_u = (
            self.config.friction_velocity
            / self.config.von_karman
            * jnp.log(z / self.config.roughness_length)
        )
        xx = 2.0 * jnp.pi * (
            jnp.arange(nx, dtype=z.dtype) + 0.5
        ) / nx
        yy = 2.0 * jnp.pi * (
            jnp.arange(ny, dtype=z.dtype) + 0.5
        ) / ny
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
            perturbation
            * jnp.cos(yy)[None, :, None]
            * jnp.sin(2.0 * xx)[None, None, :]
        )
        velocity = self.enforce_boundaries(_cells_to_faces(cells))
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
    ) -> MACVelocity:
        """Build and project a horizontally homogeneous tabulated profile."""
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
                raise ValueError(
                    "perturbation TKE must contain one value per z cell"
                )
            if bool(jnp.any(target_tke < 0.0)):
                raise ValueError("perturbation TKE must be nonnegative")
            random = jax.random.uniform(
                jax.random.PRNGKey(seed),
                cells.shape,
                dtype=dtype,
                minval=-0.5,
                maxval=0.5,
            )
            random -= jnp.mean(random, axis=(1, 2), keepdims=True)
            current_tke = 0.5 * jnp.mean(
                jnp.sum(random * random, axis=-1),
                axis=(1, 2),
            )
            scale = jnp.sqrt(
                target_tke / jnp.maximum(current_tke, jnp.finfo(dtype).eps)
            )
            random *= scale[:, None, None, None]
            cells += random
        velocity = self.enforce_boundaries(_cells_to_faces(cells))
        if self.pressure_solver.krylov.execution == "jax":
            return self.projector.project_velocity(velocity, timestep=1.0)
        return self.projector.project(velocity, timestep=1.0).velocity

    @staticmethod
    def cell_centered_velocity(velocity: MACVelocity) -> Array:
        return _cell_velocity(velocity)

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
    ) -> MACVelocity:
        if self.lasd_closure is not None and self._lasd_state is None:
            self.reset_lasd(velocity)
        if (
            self.config.wall_temporal_filter_timescale is not None
            and self._wall_model_state is None
        ):
            self.reset_wall_model(velocity)
        coefficient = self._active_lasd_coefficient(velocity)
        wall_velocity = self.active_wall_velocity(velocity)

        if self.config.sgs_time_integration == "imex_ark3":
            projected = self._imex_ark3_step(
                velocity,
                timestep=timestep,
                time=time,
                lasd_coefficient=coefficient,
                wall_velocity=wall_velocity,
            )
            advanced = self.enforce_boundaries(projected.velocity)
            if self.config.projection_method == "fpj2":
                self._accept_fpj2_pressure(projected.pressure, timestep)
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

        if self.config.projection_method == "fpj2":
            history = self._fpj2_state
            if self._fpj2_history_is_usable(timestep):
                projected = fpj2_ssprk3_velocity_step(
                    velocity,
                    tendency=stage_tendency,
                    projector=self.projector,
                    timestep=timestep,
                    current_pressure=history.current_pressure,
                    previous_pressure=history.previous_pressure,
                    current_timestep=history.current_timestep,
                    previous_timestep=history.previous_timestep,
                    time=time,
                )
            else:
                initial_pressure = (
                    None
                    if history is None
                    else history.current_pressure
                )
                projected = projected_ssprk3_velocity_pressure_step(
                    velocity,
                    tendency=stage_tendency,
                    projector=self.projector,
                    timestep=timestep,
                    time=time,
                    initial_pressure=initial_pressure,
                )
            advanced = projected.velocity
            self._accept_fpj2_pressure(projected.pressure, timestep)
        elif self.pressure_solver.krylov.execution == "jax":
            advanced = projected_ssprk3_velocity_step(
                velocity,
                tendency=stage_tendency,
                projector=self.projector,
                timestep=timestep,
                time=time,
            )
        else:
            advanced = projected_ssprk3_step(
                velocity,
                tendency=stage_tendency,
                projector=self.projector,
                timestep=timestep,
                time=time,
            ).velocity
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
    ) -> NeutralABLDiagnostic:
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
        return NeutralABLDiagnostic(
            time,
            *(float(value) for value in values),
        )

    def cfl_rate(self, velocity: MACVelocity) -> Array:
        """Return the advective CFL accumulated per unit time."""
        return (
            jnp.max(jnp.abs(velocity.x)) / self.dx
            + jnp.max(jnp.abs(velocity.y)) / self.dy
            + jnp.max(jnp.abs(velocity.z)) / self.dz
        )

    def timestep_for_cfl(
        self,
        velocity: MACVelocity,
        target_cfl: float,
        target_diffusive_cfl: float = 0.5,
    ) -> float:
        """Choose a step satisfying active explicit stability limits."""
        if not math.isfinite(target_cfl) or target_cfl <= 0.0:
            raise ValueError("target CFL must be positive and finite")
        if (
            not math.isfinite(target_diffusive_cfl)
            or target_diffusive_cfl <= 0.0
        ):
            raise ValueError("target diffusive CFL must be positive and finite")
        if self.config.sgs_time_integration == "imex_ark3":
            advective_rate, diffusive_rate = (
                self._compiled_imex_timestep_rates(
                    velocity,
                    self._active_lasd_coefficient(velocity),
                )
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

    @staticmethod
    def plane_mean_profile(velocity: MACVelocity) -> Array:
        return jnp.mean(_cell_velocity(velocity)[..., 0], axis=(1, 2))

    @staticmethod
    def plane_statistics(
        velocity: MACVelocity,
    ) -> tuple[Array, Array, Array]:
        """Return mean velocity, resolved TKE and minus-uw profiles."""
        cells = _cell_velocity(velocity)
        mean = jnp.mean(cells, axis=(1, 2))
        fluctuations = cells - mean[:, None, None, :]
        resolved_tke = 0.5 * jnp.mean(
            jnp.sum(fluctuations * fluctuations, axis=-1),
            axis=(1, 2),
        )
        minus_uw = -jnp.mean(
            fluctuations[..., 0] * fluctuations[..., 2],
            axis=(1, 2),
        )
        return mean, resolved_tke, minus_uw


__all__ = [
    "AMDModel",
    "FPJ2State",
    "NeutralABLConfig",
    "NeutralABLDiagnostic",
    "NeutralABLMomentum",
    "WallModelState",
]
