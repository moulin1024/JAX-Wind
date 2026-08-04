"""Four-rank y-slab AMD+Boussinesq integration for GABLS1."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp

from jaxwind.momentum import (
    AMDModel,
    AMDPassiveScalar,
    AMDPassiveScalarModel,
    MoninObukhovWallLaw,
    NeutralABLConfig,
    NeutralABLMomentum,
    SurfaceLayerFluxes,
)
from jaxwind.momentum.neutral_abl import _cell_velocity, _cells_to_faces
from jaxwind.pressure import (
    BoundaryCondition,
    MatrixFreePoissonSolver,
    PCGConfig,
    PoissonBoundaryConditions,
    RectilinearGrid,
    YSlabMACProjector,
    YSlabMACVelocity,
    YSlabMatrixFreePoissonSolver,
)


Array = jax.Array


class YSlabAMDBoussinesqState(NamedTuple):
    """One process's local-device batch of accepted y slabs."""

    velocity: YSlabMACVelocity
    potential_temperature: Array
    pressure: Array
    time: float
    step: int


class YSlabAMDBoussinesq:
    """Projected AMD+Boussinesq step over a global multi-process y mesh.

    The pressure backend owns the global y-slab topology. Each process holds
    only its local ``ny / process_count`` cells. Momentum and scalar kernels
    exchange the rows required by the selected nonlinear correction through
    the same global pmap axis; the complete vertical column remains local on
    every rank.
    """

    halo_width = 3

    def __init__(
        self,
        grid: RectilinearGrid,
        pressure_solver: YSlabMatrixFreePoissonSolver,
        *,
        geostrophic_wind: tuple[float, float],
        coriolis: float,
        roughness_length: float,
        gravity: float,
        reference_potential_temperature: float,
        surface_potential_temperature: float,
        surface_temperature_tendency: float,
        amd_coefficient: float,
        scalar_amd_coefficient: float,
        mp5_strength: float,
        advection_limiter: str = "mp5",
        coupling_integrator: str = "strang",
    ) -> None:
        if pressure_solver.operator.grid != grid:
            raise ValueError("distributed pressure and physical grids must match")
        self.grid = grid
        self.pressure_solver = pressure_solver
        self.projector = YSlabMACProjector(pressure_solver)
        self.device_count = pressure_solver.device_count
        self.local_device_count = pressure_solver.local_device_count
        self.axis_name = pressure_solver.distribution.axis_name
        if advection_limiter not in ("mp5", "muscl-mc"):
            raise ValueError(
                "advection limiter must be 'mp5' or 'muscl-mc'"
            )
        self.advection_limiter = advection_limiter
        self.halo_width = 3 if advection_limiter == "mp5" else 2
        self.nx = grid.shape[2]
        self.ny = grid.shape[1]
        self.nz = grid.shape[0]
        self.local_y = self.ny // self.device_count
        if self.local_y < self.halo_width:
            raise ValueError(
                "each y slab must contain at least as many cells as the "
                "selected advection halo"
            )
        self.dx = (grid.x_faces[-1] - grid.x_faces[0]) / self.nx
        self.dy = (grid.y_faces[-1] - grid.y_faces[0]) / self.ny
        self.dz = (grid.z_faces[-1] - grid.z_faces[0]) / self.nz
        self.gravity = gravity
        self.reference_potential_temperature = reference_potential_temperature
        self.surface_initial_temperature = surface_potential_temperature
        self.surface_temperature_tendency = surface_temperature_tendency
        if coupling_integrator not in ("strang", "coupled-ssprk3"):
            raise ValueError(
                "coupling integrator must be 'strang' or 'coupled-ssprk3'"
            )
        self.coupling_integrator = coupling_integrator
        self._last_surface_heat_flux_quadrature: Array | None = None

        padded_y = self.local_y + 2 * self.halo_width
        padded_grid = RectilinearGrid.uniform(
            self.nx,
            padded_y,
            self.nz,
            lx=self.nx * self.dx,
            ly=padded_y * self.dy,
            lz=self.nz * self.dz,
        )
        periodic = BoundaryCondition("periodic")
        neumann = BoundaryCondition("neumann")
        dummy_pressure = MatrixFreePoissonSolver(
            padded_grid,
            PoissonBoundaryConditions(
                periodic,
                periodic,
                periodic,
                periodic,
                neumann,
                neumann,
            ),
            dtype=pressure_solver.operator.dtype,
            krylov=PCGConfig(max_iterations=1, execution="jax"),
        )
        self.momentum_kernel = NeutralABLMomentum(
            padded_grid,
            dummy_pressure,
            NeutralABLConfig(
                friction_velocity=0.3,
                roughness_length=roughness_length,
                pressure_acceleration=0.0,
                geostrophic_wind=geostrophic_wind,
                coriolis_vertical=coriolis,
                coriolis_horizontal=0.0,
                mp5_dissipation_strength=mp5_strength,
                advection_limiter=advection_limiter,
                amd=AMDModel(coefficient=amd_coefficient),
                sgs_time_integration="explicit",
            ),
        )
        self.scalar_kernel = AMDPassiveScalar(
            padded_grid,
            AMDPassiveScalarModel(
                coefficient=scalar_amd_coefficient,
                lower_surface_flux=0.0,
                upper_surface_flux=0.0,
                mp5_dissipation_strength=mp5_strength,
                advection_limiter=advection_limiter,
            ),
        )
        self.surface_law = MoninObukhovWallLaw(
            momentum_roughness_length=roughness_length,
            thermal_roughness_length=roughness_length,
            reference_potential_temperature=reference_potential_temperature,
            von_karman=self.momentum_kernel.config.von_karman,
            gravity=gravity,
        )

        mapped = pressure_solver.pmap_options
        self._mapped_momentum_tendency = jax.pmap(
            self._momentum_tendency_local,
            in_axes=(0, 0, None),
            **mapped,
        )
        self._mapped_scalar_tendency = jax.pmap(
            self._scalar_tendency_local,
            in_axes=(0, 0, None),
            **mapped,
        )
        self._mapped_coupled_tendency = jax.pmap(
            self._coupled_tendency_local,
            in_axes=(0, 0, None),
            **mapped,
        )
        self._mapped_enforce = jax.pmap(
            self._enforce_local,
            **mapped,
        )
        self._mapped_rates = jax.pmap(
            self._rates_local,
            in_axes=(0, 0),
            **mapped,
        )
        self._mapped_surface_fluxes = jax.pmap(
            self._surface_fluxes_unpadded_local,
            in_axes=(0, 0, None),
            **mapped,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape

    def surface_temperature(self, time: Array | float) -> Array:
        return (
            jnp.asarray(time) * self.surface_temperature_tendency
            + self.surface_initial_temperature
        )

    def _send_right(self) -> list[tuple[int, int]]:
        return [
            (source, (source + 1) % self.device_count)
            for source in range(self.device_count)
        ]

    def _send_left(self) -> list[tuple[int, int]]:
        return [
            (source, (source - 1) % self.device_count)
            for source in range(self.device_count)
        ]

    def _pad_cell_rows(self, field: Array) -> Array:
        width = self.halo_width
        left = lax.ppermute(
            field[:, -width:],
            self.axis_name,
            self._send_right(),
        )
        right = lax.ppermute(
            field[:, :width],
            self.axis_name,
            self._send_left(),
        )
        return jnp.concatenate((left, field, right), axis=1)

    def _pad_y_faces(self, field: Array) -> Array:
        width = self.halo_width
        left = lax.ppermute(
            field[:, -width - 1 : -1],
            self.axis_name,
            self._send_right(),
        )
        right = lax.ppermute(
            field[:, 1 : width + 1],
            self.axis_name,
            self._send_left(),
        )
        return jnp.concatenate((left, field, right), axis=1)

    def _pad_velocity(self, velocity: YSlabMACVelocity):
        from jaxwind.pressure import MACVelocity

        return MACVelocity(
            self._pad_cell_rows(velocity.x),
            self._pad_y_faces(velocity.y),
            self._pad_cell_rows(velocity.z),
        )

    def _crop_cell_rows(self, field: Array) -> Array:
        start = self.halo_width
        return field[:, start : start + self.local_y]

    def _crop_velocity(self, velocity) -> YSlabMACVelocity:
        start = self.halo_width
        return YSlabMACVelocity(
            velocity.x[:, start : start + self.local_y],
            velocity.y[:, start : start + self.local_y + 1],
            velocity.z[:, start : start + self.local_y],
        )

    def _surface_fluxes_padded_local(
        self,
        velocity,
        theta: Array,
        time: Array,
    ) -> SurfaceLayerFluxes:
        cells = _cell_velocity(velocity)
        level = self.momentum_kernel.wall_matching_level
        return self.surface_law.surface_fluxes(
            cells[level, ..., :2],
            theta[level],
            self.surface_temperature(time),
            self.momentum_kernel.wall_matching_height,
        )

    def _surface_fluxes_unpadded_local(
        self,
        velocity: YSlabMACVelocity,
        theta: Array,
        time: Array,
    ) -> SurfaceLayerFluxes:
        padded_velocity = self._pad_velocity(velocity)
        padded_theta = self._pad_cell_rows(theta)
        fluxes = self._surface_fluxes_padded_local(
            padded_velocity,
            padded_theta,
            time,
        )
        return SurfaceLayerFluxes(
            self._crop_cell_rows(fluxes.momentum_stress[None])[0],
            self._crop_cell_rows(fluxes.heat_flux[None])[0],
            self._crop_cell_rows(fluxes.friction_velocity[None])[0],
            self._crop_cell_rows(fluxes.temperature_scale[None])[0],
            self._crop_cell_rows(fluxes.obukhov_length[None])[0],
        )

    def _momentum_tendency_local(
        self,
        velocity: YSlabMACVelocity,
        theta: Array,
        time: Array,
    ) -> YSlabMACVelocity:
        padded_velocity = self._pad_velocity(velocity)
        padded_theta = self._pad_cell_rows(theta)
        surface = self._surface_fluxes_padded_local(
            padded_velocity,
            padded_theta,
            time,
        )
        wall_stress = jnp.concatenate(
            (
                surface.momentum_stress,
                jnp.zeros_like(surface.momentum_stress[..., :1]),
            ),
            axis=-1,
        )
        tendency = self.momentum_kernel.cell_tendency(
            padded_velocity,
            wall_stress=wall_stress,
        )
        plane_sum = lax.psum(
            jnp.sum(theta, axis=(1, 2)),
            self.axis_name,
        )
        plane_mean = plane_sum / (self.nx * self.ny)
        tendency = tendency.at[..., 2].add(
            self.gravity
            / self.reference_potential_temperature
            * (padded_theta - plane_mean[:, None, None])
        )
        return self._crop_velocity(_cells_to_faces(tendency))

    def _scalar_tendency_local(
        self,
        theta: Array,
        velocity: YSlabMACVelocity,
        time: Array,
    ) -> Array:
        padded_velocity = self._pad_velocity(velocity)
        padded_theta = self._pad_cell_rows(theta)
        surface = self._surface_fluxes_padded_local(
            padded_velocity,
            padded_theta,
            time,
        )
        tendency = self.scalar_kernel.tendency(
            padded_theta,
            padded_velocity,
            lower_surface_flux=surface.heat_flux,
        )
        return self._crop_cell_rows(tendency)

    def _coupled_tendency_local(
        self,
        velocity: YSlabMACVelocity,
        theta: Array,
        time: Array,
    ) -> tuple[YSlabMACVelocity, Array, Array]:
        padded_velocity = self._pad_velocity(velocity)
        padded_theta = self._pad_cell_rows(theta)
        surface = self._surface_fluxes_padded_local(
            padded_velocity,
            padded_theta,
            time,
        )
        cells = _cell_velocity(padded_velocity)
        velocity_gradient = self.momentum_kernel.velocity_gradient(cells)
        wall_stress = jnp.concatenate(
            (
                surface.momentum_stress,
                jnp.zeros_like(surface.momentum_stress[..., :1]),
            ),
            axis=-1,
        )
        momentum = self.momentum_kernel.cell_tendency(
            padded_velocity,
            cell_velocity=cells,
            gradient=velocity_gradient,
            wall_stress=wall_stress,
        )
        plane_sum = lax.psum(
            jnp.sum(theta, axis=(1, 2)),
            self.axis_name,
        )
        plane_mean = plane_sum / (self.nx * self.ny)
        momentum = momentum.at[..., 2].add(
            self.gravity
            / self.reference_potential_temperature
            * (padded_theta - plane_mean[:, None, None])
        )
        scalar_advection = self.scalar_kernel.advective_tendency(
            padded_theta,
            padded_velocity,
        )
        scalar_advection = self._crop_cell_rows(scalar_advection)
        # Keep scalar transport constant-preserving on pressure-predicted
        # stages while retaining global conservation across every y slab.
        divergence_correction = (
            theta * self.projector._divergence_local(velocity)
        )
        correction_mean = lax.pmean(
            jnp.mean(divergence_correction),
            self.axis_name,
        )
        scalar_advection += divergence_correction - correction_mean
        scalar_sgs = self.scalar_kernel.sgs_tendency(
            padded_theta,
            velocity_gradient,
            lower_surface_flux=surface.heat_flux,
        )
        local_heat = self._crop_cell_rows(surface.heat_flux[None])[0]
        heat_mean = lax.pmean(jnp.mean(local_heat), self.axis_name)
        return (
            self._crop_velocity(_cells_to_faces(momentum)),
            scalar_advection + self._crop_cell_rows(scalar_sgs),
            heat_mean,
        )

    def _enforce_local(
        self,
        velocity: YSlabMACVelocity,
    ) -> YSlabMACVelocity:
        x_boundary = 0.5 * (velocity.x[..., 0] + velocity.x[..., -1])
        x = velocity.x.at[..., 0].set(x_boundary)
        x = x.at[..., -1].set(x_boundary)
        left_upper = lax.ppermute(
            velocity.y[:, -1],
            self.axis_name,
            self._send_right(),
        )
        right_lower = lax.ppermute(
            velocity.y[:, 0],
            self.axis_name,
            self._send_left(),
        )
        y = velocity.y.at[:, 0].set(
            0.5 * (velocity.y[:, 0] + left_upper)
        )
        y = y.at[:, -1].set(
            0.5 * (velocity.y[:, -1] + right_lower)
        )
        z = velocity.z.at[0].set(0.0).at[-1].set(0.0)
        return YSlabMACVelocity(x, y, z)

    def enforce_boundaries(
        self,
        velocity: YSlabMACVelocity,
    ) -> YSlabMACVelocity:
        return self._mapped_enforce(velocity)

    @staticmethod
    def _velocity_sum(
        *terms: tuple[float, YSlabMACVelocity],
    ) -> YSlabMACVelocity:
        return YSlabMACVelocity(
            sum(weight * value.x for weight, value in terms),
            sum(weight * value.y for weight, value in terms),
            sum(weight * value.z for weight, value in terms),
        )

    def _project(
        self,
        velocity: YSlabMACVelocity,
        timestep: float,
        pressure: Array | None,
    ):
        if self.pressure_solver.krylov.execution == "jax":
            result = self.projector.project_velocity_and_pressure(
                self.enforce_boundaries(velocity),
                timestep=timestep,
                initial_pressure=pressure,
            )
            return result._replace(
                velocity=self.enforce_boundaries(result.velocity),
            )
        result = self.projector.project(
            self.enforce_boundaries(velocity),
            timestep=timestep,
            initial_pressure=pressure,
        )
        if not result.linear_result.converged:
            raise FloatingPointError(
                "distributed pressure solve did not converge: "
                f"residual={result.linear_result.relative_residual:.3e}"
            )
        return replace(
            result,
            velocity=self.enforce_boundaries(result.velocity),
        )

    def initial_state(
        self,
        velocity: YSlabMACVelocity,
        potential_temperature: Array,
        *,
        time: float = 0.0,
        step: int = 0,
        pressure: Array | None = None,
        project: bool = True,
    ) -> YSlabAMDBoussinesqState:
        expected_scalar = (
            self.local_device_count,
            self.nz,
            self.local_y,
            self.nx,
        )
        if tuple(potential_temperature.shape) != expected_scalar:
            raise ValueError(
                f"expected local scalar shape {expected_scalar}, "
                f"got {potential_temperature.shape}"
            )
        if project:
            projected = self._project(velocity, 1.0, pressure)
            velocity = projected.velocity
            pressure = projected.pressure_correction
        else:
            velocity = self.enforce_boundaries(velocity)
            if pressure is None:
                pressure = jnp.zeros(expected_scalar, dtype=potential_temperature.dtype)
        return YSlabAMDBoussinesqState(
            velocity,
            potential_temperature,
            pressure,
            float(time),
            int(step),
        )

    def _surface_scalar_step(
        self,
        scalar: Array,
        velocity: YSlabMACVelocity,
        timestep: float,
        time: float,
    ) -> Array:
        first_tendency = self._mapped_scalar_tendency(
            scalar,
            velocity,
            time,
        )
        first = scalar + timestep * first_tendency
        second_tendency = self._mapped_scalar_tendency(
            first,
            velocity,
            time + timestep,
        )
        second = scalar + 0.25 * timestep * (
            first_tendency + second_tendency
        )
        third_tendency = self._mapped_scalar_tendency(
            second,
            velocity,
            time + 0.5 * timestep,
        )
        return scalar + timestep * (
            first_tendency / 6.0
            + second_tendency / 6.0
            + (2.0 / 3.0) * third_tendency
        )

    def step(
        self,
        state: YSlabAMDBoussinesqState,
        *,
        timestep: float,
    ) -> YSlabAMDBoussinesqState:
        if not math.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("timestep must be positive and finite")
        self._last_surface_heat_flux_quadrature = None
        if self.coupling_integrator == "coupled-ssprk3":
            first_momentum, first_scalar, first_heat = (
                self._mapped_coupled_tendency(
                    state.velocity,
                    state.potential_temperature,
                    state.time,
                )
            )
            second = self._project(
                self._velocity_sum(
                    (1.0, state.velocity),
                    (timestep, first_momentum),
                ),
                timestep,
                state.pressure,
            )
            second_theta = (
                state.potential_temperature + timestep * first_scalar
            )
            second_momentum, second_scalar, second_heat = (
                self._mapped_coupled_tendency(
                    second.velocity,
                    second_theta,
                    state.time + timestep,
                )
            )
            third = self._project(
                self._velocity_sum(
                    (1.0, state.velocity),
                    (0.25 * timestep, first_momentum),
                    (0.25 * timestep, second_momentum),
                ),
                0.5 * timestep,
                second.pressure_correction,
            )
            third_theta = state.potential_temperature + 0.25 * timestep * (
                first_scalar + second_scalar
            )
            third_momentum, third_scalar, third_heat = (
                self._mapped_coupled_tendency(
                    third.velocity,
                    third_theta,
                    state.time + 0.5 * timestep,
                )
            )
            final = self._project(
                self._velocity_sum(
                    (1.0, state.velocity),
                    (timestep / 6.0, first_momentum),
                    (timestep / 6.0, second_momentum),
                    (2.0 * timestep / 3.0, third_momentum),
                ),
                timestep,
                third.pressure_correction,
            )
            theta = state.potential_temperature + timestep * (
                first_scalar / 6.0
                + second_scalar / 6.0
                + (2.0 / 3.0) * third_scalar
            )
            self._last_surface_heat_flux_quadrature = (
                first_heat / 6.0
                + second_heat / 6.0
                + (2.0 / 3.0) * third_heat
            )
            return YSlabAMDBoussinesqState(
                final.velocity,
                theta,
                final.pressure_correction,
                state.time + timestep,
                state.step + 1,
            )
        half = 0.5 * timestep
        midpoint_theta = self._surface_scalar_step(
            state.potential_temperature,
            state.velocity,
            half,
            state.time,
        )
        first_tendency = self._mapped_momentum_tendency(
            state.velocity,
            midpoint_theta,
            state.time,
        )
        second = self._project(
            self._velocity_sum(
                (1.0, state.velocity),
                (timestep, first_tendency),
            ),
            timestep,
            state.pressure,
        )
        second_tendency = self._mapped_momentum_tendency(
            second.velocity,
            midpoint_theta,
            state.time + timestep,
        )
        third = self._project(
            self._velocity_sum(
                (1.0, state.velocity),
                (0.25 * timestep, first_tendency),
                (0.25 * timestep, second_tendency),
            ),
            0.5 * timestep,
            second.pressure_correction,
        )
        third_tendency = self._mapped_momentum_tendency(
            third.velocity,
            midpoint_theta,
            state.time + 0.5 * timestep,
        )
        final = self._project(
            self._velocity_sum(
                (1.0, state.velocity),
                (timestep / 6.0, first_tendency),
                (timestep / 6.0, second_tendency),
                (2.0 * timestep / 3.0, third_tendency),
            ),
            timestep,
            third.pressure_correction,
        )
        theta = self._surface_scalar_step(
            midpoint_theta,
            final.velocity,
            half,
            state.time + half,
        )
        return YSlabAMDBoussinesqState(
            final.velocity,
            theta,
            final.pressure_correction,
            state.time + timestep,
            state.step + 1,
        )

    @property
    def last_surface_heat_flux_quadrature(self) -> Array | None:
        """Return the accepted step's RK-weighted global lower heat flux."""
        return self._last_surface_heat_flux_quadrature

    def _rates_local(
        self,
        velocity: YSlabMACVelocity,
        theta: Array,
    ) -> tuple[Array, Array, Array]:
        padded_velocity = self._pad_velocity(velocity)
        padded_theta = self._pad_cell_rows(theta)
        cells = _cell_velocity(padded_velocity)
        gradient = self.momentum_kernel.velocity_gradient(cells)
        viscosity = self.momentum_kernel.sgs_viscosity(
            cells,
            gradient=gradient,
        )
        scalar_diffusivity = self.scalar_kernel.amd_diffusivity(
            padded_theta,
            gradient,
        )
        viscosity = self._crop_cell_rows(viscosity)
        scalar_diffusivity = self._crop_cell_rows(scalar_diffusivity)
        local_advective = (
            jnp.maximum(
                jnp.abs(velocity.x[..., :-1]),
                jnp.abs(velocity.x[..., 1:]),
            )
            / self.dx
            + jnp.maximum(
                jnp.abs(velocity.y[:, :-1, :]),
                jnp.abs(velocity.y[:, 1:, :]),
            )
            / self.dy
            + jnp.maximum(
                jnp.abs(velocity.z[:-1, ...]),
                jnp.abs(velocity.z[1:, ...]),
            )
            / self.dz
        )
        advective = jnp.max(local_advective)
        inverse_spacing_squared = (
            1.0 / self.dx**2
            + 1.0 / self.dy**2
            + 1.0 / self.dz**2
        )
        momentum_diffusive = (
            2.0 * jnp.max(viscosity) * inverse_spacing_squared
        )
        scalar_diffusive = (
            2.0 * jnp.max(scalar_diffusivity) * inverse_spacing_squared
        )
        return tuple(
            lax.pmax(value, self.axis_name)
            for value in (advective, momentum_diffusive, scalar_diffusive)
        )

    def rates(
        self,
        state: YSlabAMDBoussinesqState,
    ) -> tuple[float, float, float]:
        values = self._mapped_rates(
            state.velocity,
            state.potential_temperature,
        )
        return tuple(float(jax.device_get(value)[0]) for value in values)

    def timestep_for_cfl(
        self,
        state: YSlabAMDBoussinesqState,
        target_cfl: float,
        target_diffusive_cfl: float,
    ) -> float:
        advective, momentum_diffusive, scalar_diffusive = self.rates(state)
        candidates = [target_cfl / advective]
        for rate in (momentum_diffusive, scalar_diffusive):
            if rate > 0.0:
                candidates.append(target_diffusive_cfl / rate)
        return min(candidates)

    def surface_layer_fluxes(
        self,
        state: YSlabAMDBoussinesqState,
    ) -> SurfaceLayerFluxes:
        return self._mapped_surface_fluxes(
            state.velocity,
            state.potential_temperature,
            state.time,
        )

    def divergence_norm(self, velocity: YSlabMACVelocity) -> float:
        divergence = self.projector.divergence(velocity)
        value = self.pressure_solver.inner(divergence, divergence)
        return math.sqrt(max(float(jax.device_get(value)), 0.0))


__all__ = [
    "YSlabAMDBoussinesq",
    "YSlabAMDBoussinesqState",
]
