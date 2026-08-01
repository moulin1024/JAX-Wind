"""Actively coupled dry-convective ABL on the non-spectral MAC solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxwind.pressure import (
    MACVelocity,
    projected_ssprk3_velocity_pressure_step,
)

from .neutral_abl import (
    AMDPassiveScalar,
    NeutralABLMomentum,
    _cell_velocity,
    _cells_to_faces,
    _velocity_sum,
)
from .surface_layer import MoninObukhovWallLaw, SurfaceLayerFluxes


Array = jax.Array


@dataclass(frozen=True, slots=True)
class AMDBoussinesqConfig:
    """Physical constants for an actively transported potential temperature."""

    gravity: float = 9.81
    reference_potential_temperature: float = 300.0
    sgs_dissipation_coefficient: float = 0.93
    scalar_variance_coefficient: float = 2.02
    surface_potential_temperature: float | None = None
    surface_temperature_tendency: float = 0.0
    thermal_roughness_length: float | None = None

    def __post_init__(self) -> None:
        positive = (
            self.gravity,
            self.reference_potential_temperature,
            self.sgs_dissipation_coefficient,
            self.scalar_variance_coefficient,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Boussinesq and diagnostic constants must be positive")
        if self.surface_potential_temperature is not None and not math.isfinite(
            self.surface_potential_temperature
        ):
            raise ValueError("surface potential temperature must be finite")
        if not math.isfinite(self.surface_temperature_tendency):
            raise ValueError("surface temperature tendency must be finite")
        if (
            self.surface_potential_temperature is None
            and self.surface_temperature_tendency != 0.0
        ):
            raise ValueError(
                "surface temperature tendency requires an initial temperature"
            )
        if self.thermal_roughness_length is not None and (
            not math.isfinite(self.thermal_roughness_length)
            or self.thermal_roughness_length <= 0.0
        ):
            raise ValueError("thermal roughness length must be positive and finite")

    @property
    def buoyancy_coefficient(self) -> float:
        return self.gravity / self.reference_potential_temperature


class AMDBoussinesqState(NamedTuple):
    """Accepted non-spectral velocity, potential temperature, and pressure."""

    velocity: MACVelocity
    potential_temperature: Array
    pressure: Array
    time: float
    step: int


class AMDBoussinesqDiagnosticFields(NamedTuple):
    """Local closure and numerical-dissipation observations."""

    momentum_diffusivity: Array
    scalar_diffusivity: Array
    scalar_flux_x: Array
    scalar_flux_y: Array
    scalar_flux_z: Array
    sgs_tke: Array
    scalar_variance_numerator: Array
    scalar_variance: Array
    amd_energy_dissipation: Array
    mp5_energy_dissipation: Array
    amd_scalar_dissipation: Array
    mp5_scalar_dissipation: Array
    surface_momentum_stress: Array
    surface_heat_flux: Array
    surface_friction_velocity: Array
    surface_temperature_scale: Array
    surface_obukhov_length: Array


class AMDBoussinesq:
    """Strang-coupled active scalar around the non-spectral AMD solver.

    Potential temperature is advanced for half a step with the accepted
    velocity.  Its midpoint value supplies a frozen buoyancy force to all
    three projected momentum stages, after which the scalar completes the
    second half step with the accepted new velocity.
    """

    def __init__(
        self,
        momentum: NeutralABLMomentum,
        scalar: AMDPassiveScalar,
        config: AMDBoussinesqConfig = AMDBoussinesqConfig(),
    ) -> None:
        if momentum.grid != scalar.grid:
            raise ValueError("momentum and scalar grids must match")
        if momentum.lasd_closure is not None:
            raise ValueError("AMDBoussinesq requires the AMD momentum closure")
        if momentum.config.sgs_time_integration != "explicit":
            raise ValueError("the first coupled reference path requires explicit SGS")
        if momentum.config.projection_method != "full":
            raise ValueError("the first coupled reference path requires full projection")
        if momentum.config.wall_temporal_filter_timescale is not None:
            raise ValueError("coupled reference path requires a memoryless wall input")
        self.momentum = momentum
        self.scalar = scalar
        self.config = config
        self.grid = momentum.grid
        self.surface_law = (
            None
            if config.surface_potential_temperature is None
            else MoninObukhovWallLaw(
                momentum_roughness_length=momentum.config.roughness_length,
                thermal_roughness_length=(
                    momentum.config.roughness_length
                    if config.thermal_roughness_length is None
                    else config.thermal_roughness_length
                ),
                reference_potential_temperature=(
                    config.reference_potential_temperature
                ),
                von_karman=momentum.config.von_karman,
                gravity=config.gravity,
            )
        )
        if self.surface_law is not None and scalar.model.lower_surface_flux != 0.0:
            raise ValueError(
                "prescribed-temperature coupling requires zero fixed lower scalar flux"
            )
        self._compiled_surface_scalar_step = jax.jit(
            self._surface_scalar_step
        )
        self._compiled_surface_momentum_tendency = jax.jit(
            self._surface_momentum_tendency
        )

    def surface_potential_temperature(self, time: Array | float) -> Array:
        """Return the linearly evolving prescribed surface temperature."""
        if self.config.surface_potential_temperature is None:
            raise RuntimeError("the coupled solver has no prescribed surface temperature")
        return jnp.asarray(
            self.config.surface_potential_temperature,
        ) + jnp.asarray(time) * self.config.surface_temperature_tendency

    def _surface_layer_fluxes(
        self,
        velocity: MACVelocity,
        potential_temperature: Array,
        time: Array | float,
    ) -> SurfaceLayerFluxes:
        if self.surface_law is None:
            cells = _cell_velocity(velocity)
            neutral = self.momentum.wall_fluxes(cells)
            heat_flux = jnp.full_like(
                neutral.heat_flux,
                self.scalar.model.lower_surface_flux,
            )
            return SurfaceLayerFluxes(
                neutral.momentum_stress,
                heat_flux,
                neutral.friction_velocity,
                neutral.temperature_scale,
                neutral.obukhov_length,
            )
        cells = _cell_velocity(velocity)
        horizontal_velocity = self.momentum.wall_velocity(cells)
        matching_temperature = potential_temperature[
            self.momentum.config.wall_matching_level
        ]
        return self.surface_law.surface_fluxes(
            horizontal_velocity,
            matching_temperature,
            self.surface_potential_temperature(time),
            self.momentum.wall_matching_height,
        )

    def surface_layer_fluxes(
        self,
        state: AMDBoussinesqState,
    ) -> SurfaceLayerFluxes:
        """Diagnose lower-boundary momentum and heat fluxes for a state."""
        return self._surface_layer_fluxes(
            state.velocity,
            state.potential_temperature,
            state.time,
        )

    @staticmethod
    def _momentum_wall_stress(fluxes: SurfaceLayerFluxes) -> Array:
        tangential = fluxes.momentum_stress
        return jnp.concatenate(
            (tangential, jnp.zeros_like(tangential[..., :1])),
            axis=-1,
        )

    def _surface_scalar_tendency(
        self,
        scalar: Array,
        velocity: MACVelocity,
        time: Array,
    ) -> Array:
        fluxes = self._surface_layer_fluxes(velocity, scalar, time)
        return self.scalar.tendency(
            scalar,
            velocity,
            lower_surface_flux=fluxes.heat_flux,
        )

    def _surface_scalar_step(
        self,
        scalar: Array,
        velocity: MACVelocity,
        timestep: Array,
        time: Array,
    ) -> Array:
        """Advance scalar SSPRK3 while reevaluating MOST at every stage."""
        first_tendency = self._surface_scalar_tendency(
            scalar,
            velocity,
            time,
        )
        first = scalar + timestep * first_tendency
        second_tendency = self._surface_scalar_tendency(
            first,
            velocity,
            time + timestep,
        )
        second = scalar + 0.25 * timestep * (
            first_tendency + second_tendency
        )
        third_tendency = self._surface_scalar_tendency(
            second,
            velocity,
            time + 0.5 * timestep,
        )
        return scalar + timestep * (
            first_tendency / 6.0
            + second_tendency / 6.0
            + (2.0 / 3.0) * third_tendency
        )

    def _surface_momentum_tendency(
        self,
        velocity: MACVelocity,
        potential_temperature: Array,
        time: Array,
    ) -> MACVelocity:
        fluxes = self._surface_layer_fluxes(
            velocity,
            potential_temperature,
            time,
        )
        momentum = self.momentum.tendency_with_wall_stress(
            velocity,
            self._momentum_wall_stress(fluxes),
        )
        return _velocity_sum(
            (1.0, momentum),
            (1.0, self.buoyancy_tendency(potential_temperature)),
        )

    def initial_state(
        self,
        velocity: MACVelocity,
        potential_temperature: Array,
        *,
        time: float = 0.0,
        step: int = 0,
        pressure: Array | None = None,
    ) -> AMDBoussinesqState:
        self.scalar._validate_scalar(potential_temperature)
        if not math.isfinite(time) or time < 0.0:
            raise ValueError("initial time must be finite and nonnegative")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("initial step must be a nonnegative integer")
        dtype = potential_temperature.dtype
        initial_pressure = (
            jnp.zeros(self.grid.shape, dtype=dtype)
            if pressure is None
            else jnp.asarray(pressure, dtype=dtype)
        )
        if initial_pressure.shape != self.grid.shape:
            raise ValueError("initial pressure shape does not match the grid")
        return AMDBoussinesqState(
            self.momentum.enforce_boundaries(velocity),
            potential_temperature,
            initial_pressure,
            float(time),
            step,
        )

    def buoyancy_tendency(self, potential_temperature: Array) -> MACVelocity:
        """Return face acceleration after removing the hydrostatic plane mean."""
        self.scalar._validate_scalar(potential_temperature)
        anomaly = potential_temperature - jnp.mean(
            potential_temperature,
            axis=(1, 2),
            keepdims=True,
        )
        cells = jnp.zeros(
            potential_temperature.shape + (3,),
            dtype=potential_temperature.dtype,
        )
        cells = cells.at[..., 2].set(
            jnp.asarray(
                self.config.buoyancy_coefficient,
                dtype=potential_temperature.dtype,
            )
            * anomaly
        )
        return _cells_to_faces(cells)

    def step(
        self,
        state: AMDBoussinesqState,
        *,
        timestep: float,
    ) -> AMDBoussinesqState:
        """Advance one accepted active-scalar step with full stage projection."""
        if not math.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("timestep must be positive and finite")
        if self.surface_law is None:
            midpoint_temperature = self.scalar.step(
                state.potential_temperature,
                state.velocity,
                timestep=0.5 * timestep,
            )
        else:
            midpoint_temperature = self._compiled_surface_scalar_step(
                state.potential_temperature,
                state.velocity,
                jnp.asarray(0.5 * timestep, dtype=state.potential_temperature.dtype),
                jnp.asarray(state.time, dtype=state.potential_temperature.dtype),
            )
        buoyancy = self.buoyancy_tendency(midpoint_temperature)

        def stage_tendency(stage_velocity: MACVelocity, stage_time: float):
            if self.surface_law is not None:
                return self._compiled_surface_momentum_tendency(
                    stage_velocity,
                    midpoint_temperature,
                    jnp.asarray(stage_time, dtype=midpoint_temperature.dtype),
                )
            return _velocity_sum(
                (1.0, self.momentum.tendency(stage_velocity, stage_time)),
                (1.0, buoyancy),
            )

        projected = projected_ssprk3_velocity_pressure_step(
            state.velocity,
            tendency=stage_tendency,
            projector=self.momentum.projector,
            timestep=timestep,
            time=state.time,
            initial_pressure=state.pressure,
        )
        velocity = self.momentum.enforce_boundaries(projected.velocity)
        if self.surface_law is None:
            potential_temperature = self.scalar.step(
                midpoint_temperature,
                velocity,
                timestep=0.5 * timestep,
            )
        else:
            potential_temperature = self._compiled_surface_scalar_step(
                midpoint_temperature,
                velocity,
                jnp.asarray(0.5 * timestep, dtype=midpoint_temperature.dtype),
                jnp.asarray(
                    state.time + 0.5 * timestep,
                    dtype=midpoint_temperature.dtype,
                ),
            )
        return AMDBoussinesqState(
            velocity,
            potential_temperature,
            projected.pressure,
            state.time + timestep,
            state.step + 1,
        )

    def timestep_for_cfl(
        self,
        state: AMDBoussinesqState,
        target_cfl: float,
        target_diffusive_cfl: float = 0.5,
    ) -> float:
        """Return the joint momentum/scalar explicit stability limit."""
        momentum_step = self.momentum.timestep_for_cfl(
            state.velocity,
            target_cfl,
            target_diffusive_cfl,
        )
        scalar_step = self.scalar.timestep_for_diffusive_cfl(
            state.potential_temperature,
            state.velocity,
            target_diffusive_cfl,
        )
        return min(momentum_step, scalar_step)

    def diagnostic_fields(
        self,
        state: AMDBoussinesqState,
    ) -> AMDBoussinesqDiagnosticFields:
        """Diagnose AMD, buoyancy, scalar variance, and MP5 contributions."""
        velocity = state.velocity
        theta = state.potential_temperature
        cells = _cell_velocity(velocity)
        velocity_gradient = self.momentum.velocity_gradient(cells)
        surface_fluxes = self._surface_layer_fluxes(
            velocity,
            theta,
            state.time,
        )
        diagnostic_gradient = self.momentum.diagnostic_wall_consistent_gradient(
            cells,
            gradient=velocity_gradient,
        )
        strain = 0.5 * (
            diagnostic_gradient + jnp.swapaxes(diagnostic_gradient, -1, -2)
        )
        strain_magnitude_squared = 2.0 * jnp.einsum(
            "...ij,...ij->...",
            strain,
            strain,
        )
        strain_magnitude = jnp.sqrt(strain_magnitude_squared)
        momentum_diffusivity = jnp.maximum(
            self.momentum.sgs_viscosity(
                cells,
                gradient=velocity_gradient,
            )
            - self.momentum.config.amd.molecular_viscosity,
            0.0,
        )
        (
            scalar_diffusivity,
            scalar_gradient,
            scalar_flux_x,
            scalar_flux_y,
            scalar_flux_z,
        ) = self.scalar.sgs_fluxes(
            theta,
            velocity_gradient,
            lower_surface_flux=surface_fluxes.heat_flux,
        )
        scalar_diffusivity = jnp.maximum(
            scalar_diffusivity - self.scalar.model.molecular_diffusivity,
            0.0,
        )
        scalar_flux_z_at_cells = 0.5 * (
            scalar_flux_z[:-1] + scalar_flux_z[1:]
        )
        delta = (self.momentum.dx * self.momentum.dy * self.momentum.dz) ** (
            1.0 / 3.0
        )
        production = (
            momentum_diffusivity * strain_magnitude_squared
            + self.config.buoyancy_coefficient * scalar_flux_z_at_cells
        )
        sgs_tke = jnp.maximum(
            production * delta / self.config.sgs_dissipation_coefficient,
            0.0,
        ) ** (2.0 / 3.0)

        interior_gradient = (theta[1:] - theta[:-1]) / self.scalar.dz
        wall_diffusivity = jnp.mean(scalar_diffusivity[0])
        lower_surface_flux = jnp.mean(surface_fluxes.heat_flux)
        lower_gradient_plane = jnp.where(
            wall_diffusivity > 0.0,
            -lower_surface_flux
            / jnp.maximum(wall_diffusivity, jnp.finfo(theta.dtype).tiny),
            0.0,
        )
        lower_gradient = jnp.concatenate(
            (
                jnp.full_like(theta[:1], lower_gradient_plane),
                interior_gradient,
            ),
            axis=0,
        )
        upper_gradient = jnp.concatenate(
            (interior_gradient, jnp.zeros_like(theta[-1:])),
            axis=0,
        )
        diagnostic_gradient_z = 0.5 * (lower_gradient + upper_gradient)
        amd_scalar_dissipation = -(
            scalar_flux_x * scalar_gradient[..., 0]
            + scalar_flux_y * scalar_gradient[..., 1]
            + scalar_flux_z_at_cells * diagnostic_gradient_z
        )
        effective_scalar_coefficient = scalar_diffusivity / jnp.maximum(
            delta**2 * strain_magnitude,
            jnp.finfo(theta.dtype).tiny,
        )
        scalar_length = delta * jnp.sqrt(
            jnp.maximum(effective_scalar_coefficient, 0.0)
        )
        scalar_variance_numerator = (
            2.0
            * scalar_length
            * amd_scalar_dissipation
            / self.config.scalar_variance_coefficient
        )
        sqrt_tke = jnp.sqrt(jnp.maximum(sgs_tke, 0.0))
        scalar_variance = jnp.maximum(
            jnp.where(
                sqrt_tke > jnp.finfo(theta.dtype).tiny,
                scalar_variance_numerator
                / jnp.maximum(sqrt_tke, jnp.finfo(theta.dtype).tiny),
                0.0,
            ),
            0.0,
        )

        velocity_mean = jnp.mean(cells, axis=(1, 2), keepdims=True)
        velocity_fluctuation = cells - velocity_mean
        momentum_mp5 = self.momentum.mp5_dissipation(velocity, cells)
        theta_fluctuation = theta - jnp.mean(
            theta,
            axis=(1, 2),
            keepdims=True,
        )
        scalar_mp5 = self.scalar.mp5_dissipation(theta, velocity)
        amd_energy_dissipation = -self.momentum.resolved_tke_sgs_dissipation(
            cells,
            gradient=velocity_gradient,
        )
        mp5_energy_dissipation = -jnp.einsum(
            "...i,...i->...",
            velocity_fluctuation,
            momentum_mp5,
        )
        mp5_scalar_dissipation = -theta_fluctuation * scalar_mp5
        return AMDBoussinesqDiagnosticFields(
            momentum_diffusivity,
            scalar_diffusivity,
            scalar_flux_x,
            scalar_flux_y,
            scalar_flux_z,
            sgs_tke,
            scalar_variance_numerator,
            scalar_variance,
            amd_energy_dissipation,
            mp5_energy_dissipation,
            amd_scalar_dissipation,
            mp5_scalar_dissipation,
            surface_fluxes.momentum_stress,
            surface_fluxes.heat_flux,
            surface_fluxes.friction_velocity,
            surface_fluxes.temperature_scale,
            surface_fluxes.obukhov_length,
        )


__all__ = [
    "AMDBoussinesq",
    "AMDBoussinesqConfig",
    "AMDBoussinesqDiagnosticFields",
    "AMDBoussinesqState",
]
