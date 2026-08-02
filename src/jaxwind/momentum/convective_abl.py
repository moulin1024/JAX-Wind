"""Actively coupled dry-convective ABL on the non-spectral MAC solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxwind.pressure import (
    fpj2_pressure_prediction,
    fpj2_ssprk3_velocity_step,
    MACVelocity,
    mac_divergence,
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
    coupling_integrator: str = "strang"

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
        if self.coupling_integrator not in ("strang", "coupled-ssprk3"):
            raise ValueError("coupling integrator must be 'strang' or 'coupled-ssprk3'")

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
    ko6_energy_dissipation: Array
    mp5_energy_dissipation: Array
    amd_scalar_dissipation: Array
    mp5_scalar_dissipation: Array
    surface_momentum_stress: Array
    surface_heat_flux: Array
    surface_friction_velocity: Array
    surface_temperature_scale: Array
    surface_obukhov_length: Array


class _CoupledSSPRK3Result(NamedTuple):
    velocity: MACVelocity
    potential_temperature: Array
    pressure: Array
    surface_heat_flux_quadrature: Array


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
        self._compiled_surface_scalar_tendency = jax.jit(self._surface_scalar_tendency)
        self._compiled_surface_scalar_step = self._dispatched_surface_scalar_step
        self._compiled_surface_momentum_tendency = jax.jit(
            self._surface_momentum_tendency
        )
        self._compiled_coupled_surface_tendency = jax.jit(
            self._coupled_surface_tendency
        )
        self._compiled_stability_rates = jax.jit(self._stability_rates)
        self._last_surface_heat_flux_quadrature: Array | None = None

        def pre_step_metrics(
            velocity: MACVelocity,
            potential_temperature: Array,
            time: Array,
        ) -> tuple[Array, Array, Array, Array, Array]:
            advective, momentum_diffusive, scalar_diffusive = self._stability_rates(
                velocity, potential_temperature
            )
            fluxes = self._surface_layer_fluxes(
                velocity,
                potential_temperature,
                time,
            )
            return (
                advective,
                momentum_diffusive,
                scalar_diffusive,
                jnp.mean(
                    potential_temperature - self.config.reference_potential_temperature
                ),
                jnp.mean(fluxes.heat_flux),
            )

        def accepted_state_metrics(
            velocity: MACVelocity,
            potential_temperature: Array,
            time: Array,
        ) -> tuple[Array, Array, Array]:
            fluxes = self._surface_layer_fluxes(
                velocity,
                potential_temperature,
                time,
            )
            divergence = mac_divergence(velocity, self.grid)
            return (
                jnp.mean(
                    potential_temperature - self.config.reference_potential_temperature
                ),
                jnp.mean(fluxes.heat_flux),
                self.momentum.pressure_solver.operator.norm(divergence),
            )

        self._compiled_pre_step_metrics = jax.jit(pre_step_metrics)
        self._compiled_accepted_state_metrics = jax.jit(accepted_state_metrics)

    def surface_potential_temperature(self, time: Array | float) -> Array:
        """Return the linearly evolving prescribed surface temperature."""
        if self.config.surface_potential_temperature is None:
            raise RuntimeError(
                "the coupled solver has no prescribed surface temperature"
            )
        return (
            jnp.asarray(
                self.config.surface_potential_temperature,
            )
            + jnp.asarray(time) * self.config.surface_temperature_tendency
        )

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
        second = scalar + 0.25 * timestep * (first_tendency + second_tendency)
        third_tendency = self._surface_scalar_tendency(
            second,
            velocity,
            time + 0.5 * timestep,
        )
        return scalar + timestep * (
            first_tendency / 6.0 + second_tendency / 6.0 + (2.0 / 3.0) * third_tendency
        )

    def _dispatched_surface_scalar_step(
        self,
        scalar: Array,
        velocity: MACVelocity,
        timestep: Array,
        time: Array,
    ) -> Array:
        """SSPRK3 scalar step using bounded-size compiled stage kernels."""
        first_tendency = self._compiled_surface_scalar_tendency(
            scalar,
            velocity,
            time,
        )
        first = scalar + timestep * first_tendency
        second_tendency = self._compiled_surface_scalar_tendency(
            first,
            velocity,
            time + timestep,
        )
        second = scalar + 0.25 * timestep * (first_tendency + second_tendency)
        third_tendency = self._compiled_surface_scalar_tendency(
            second,
            velocity,
            time + 0.5 * timestep,
        )
        return scalar + timestep * (
            first_tendency / 6.0 + second_tendency / 6.0 + (2.0 / 3.0) * third_tendency
        )

    def _surface_momentum_tendency(
        self,
        velocity: MACVelocity,
        potential_temperature: Array,
        time: Array,
        buoyancy: MACVelocity,
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
            (1.0, buoyancy),
        )

    def _coupled_surface_tendency(
        self,
        velocity: MACVelocity,
        potential_temperature: Array,
        time: Array,
    ) -> tuple[MACVelocity, Array, Array]:
        """Return stage-consistent momentum, scalar, and wall-heat RHS data."""
        cells = _cell_velocity(velocity)
        velocity_gradient = self.momentum.velocity_gradient(cells)
        fluxes = self._surface_layer_fluxes(
            velocity,
            potential_temperature,
            time,
        )
        wall_stress = self._momentum_wall_stress(fluxes)
        momentum_cells = self.momentum.cell_tendency(
            velocity,
            self.momentum._active_lasd_coefficient(velocity),
            cell_velocity=cells,
            gradient=velocity_gradient,
            wall_stress=wall_stress,
        )
        momentum = _velocity_sum(
            (1.0, _cells_to_faces(momentum_cells)),
            (1.0, self.buoyancy_tendency(potential_temperature)),
        )
        scalar_advection = self.scalar.advective_tendency(
            potential_temperature,
            velocity,
        )
        # FPJ2 pressure-predicted stage velocities are not exactly solenoidal.
        # Convert conservative transport to its constant-preserving advective
        # equivalent at those stages, then remove the correction's mean so the
        # accepted scalar remains globally conservative.
        divergence_correction = potential_temperature * mac_divergence(
            velocity, self.grid
        )
        scalar_advection += divergence_correction - jnp.mean(divergence_correction)
        scalar = scalar_advection + self.scalar.sgs_tendency(
            potential_temperature,
            velocity_gradient,
            lower_surface_flux=fluxes.heat_flux,
        )
        return momentum, scalar, jnp.mean(fluxes.heat_flux)

    def _full_coupled_ssprk3_step(
        self,
        state: AMDBoussinesqState,
        timestep: float,
        initial_pressure: Array,
    ) -> _CoupledSSPRK3Result:
        """Advance coupled SSPRK3 with a pressure solve at every stage."""
        theta = state.potential_temperature
        dtype = theta.dtype
        first_momentum, first_scalar, first_heat = (
            self._compiled_coupled_surface_tendency(
                state.velocity,
                theta,
                jnp.asarray(state.time, dtype=dtype),
            )
        )
        second = self.momentum.projector.project_velocity_and_pressure(
            _velocity_sum(
                (1.0, state.velocity),
                (timestep, first_momentum),
            ),
            timestep=timestep,
            initial_pressure=initial_pressure,
        )
        second_theta = theta + timestep * first_scalar
        second_momentum, second_scalar, second_heat = (
            self._compiled_coupled_surface_tendency(
                second.velocity,
                second_theta,
                jnp.asarray(state.time + timestep, dtype=dtype),
            )
        )
        third = self.momentum.projector.project_velocity_and_pressure(
            _velocity_sum(
                (1.0, state.velocity),
                (0.25 * timestep, first_momentum),
                (0.25 * timestep, second_momentum),
            ),
            timestep=0.5 * timestep,
            initial_pressure=second.pressure,
        )
        third_theta = theta + 0.25 * timestep * (first_scalar + second_scalar)
        third_momentum, third_scalar, third_heat = (
            self._compiled_coupled_surface_tendency(
                third.velocity,
                third_theta,
                jnp.asarray(state.time + 0.5 * timestep, dtype=dtype),
            )
        )
        final = self.momentum.projector.project_velocity_and_pressure(
            _velocity_sum(
                (1.0, state.velocity),
                (timestep / 6.0, first_momentum),
                (timestep / 6.0, second_momentum),
                (2.0 * timestep / 3.0, third_momentum),
            ),
            timestep=timestep,
            initial_pressure=third.pressure,
        )
        final_theta = theta + timestep * (
            first_scalar / 6.0 + second_scalar / 6.0 + (2.0 / 3.0) * third_scalar
        )
        heat_quadrature = (
            first_heat / 6.0 + second_heat / 6.0 + (2.0 / 3.0) * third_heat
        )
        return _CoupledSSPRK3Result(
            final.velocity,
            final_theta,
            final.pressure,
            heat_quadrature,
        )

    def _fpj2_coupled_ssprk3_step(
        self,
        state: AMDBoussinesqState,
        timestep: float,
    ) -> _CoupledSSPRK3Result:
        """Advance coupled SSPRK3 with FPJ2 stage-pressure predictions."""
        history = self.momentum.fpj2_state
        if history is None:
            raise RuntimeError("FPJ2 pressure history is unavailable")
        theta = state.potential_temperature
        dtype = theta.dtype
        second_pressure = fpj2_pressure_prediction(
            history.current_pressure,
            history.previous_pressure,
            current_timestep=history.current_timestep,
            previous_timestep=history.previous_timestep,
            next_timestep=timestep,
            stage_abscissa=1.0,
        )
        third_pressure = fpj2_pressure_prediction(
            history.current_pressure,
            history.previous_pressure,
            current_timestep=history.current_timestep,
            previous_timestep=history.previous_timestep,
            next_timestep=timestep,
            stage_abscissa=0.5,
        )
        first_momentum, first_scalar, first_heat = (
            self._compiled_coupled_surface_tendency(
                state.velocity,
                theta,
                jnp.asarray(state.time, dtype=dtype),
            )
        )
        second_gradient = self.momentum.projector.pressure_gradient(second_pressure)
        second_velocity = _velocity_sum(
            (1.0, state.velocity),
            (timestep, first_momentum),
            (-timestep, second_gradient),
        )
        second_theta = theta + timestep * first_scalar
        second_momentum, second_scalar, second_heat = (
            self._compiled_coupled_surface_tendency(
                second_velocity,
                second_theta,
                jnp.asarray(state.time + timestep, dtype=dtype),
            )
        )
        third_gradient = self.momentum.projector.pressure_gradient(third_pressure)
        third_velocity = _velocity_sum(
            (1.0, state.velocity),
            (0.25 * timestep, first_momentum),
            (0.25 * timestep, second_momentum),
            (-0.5 * timestep, third_gradient),
        )
        third_theta = theta + 0.25 * timestep * (first_scalar + second_scalar)
        third_momentum, third_scalar, third_heat = (
            self._compiled_coupled_surface_tendency(
                third_velocity,
                third_theta,
                jnp.asarray(state.time + 0.5 * timestep, dtype=dtype),
            )
        )
        final = self.momentum.projector.project_velocity_and_pressure(
            _velocity_sum(
                (1.0, state.velocity),
                (timestep / 6.0, first_momentum),
                (timestep / 6.0, second_momentum),
                (2.0 * timestep / 3.0, third_momentum),
            ),
            timestep=timestep,
            initial_pressure=second_pressure,
        )
        final_theta = theta + timestep * (
            first_scalar / 6.0 + second_scalar / 6.0 + (2.0 / 3.0) * third_scalar
        )
        heat_quadrature = (
            first_heat / 6.0 + second_heat / 6.0 + (2.0 / 3.0) * third_heat
        )
        return _CoupledSSPRK3Result(
            final.velocity,
            final_theta,
            final.pressure,
            heat_quadrature,
        )

    def _coupled_ssprk3_step(
        self,
        state: AMDBoussinesqState,
        timestep: float,
    ) -> _CoupledSSPRK3Result:
        history = self.momentum.fpj2_state
        if (
            self.momentum.config.projection_method == "fpj2"
            and self.momentum._fpj2_history_is_usable(timestep)
        ):
            return self._fpj2_coupled_ssprk3_step(state, timestep)
        initial_pressure = (
            state.pressure if history is None else history.current_pressure
        )
        return self._full_coupled_ssprk3_step(
            state,
            timestep,
            initial_pressure,
        )

    @property
    def last_surface_heat_flux_quadrature(self) -> Array | None:
        """Return the accepted step's RK-weighted mean lower heat flux."""
        return self._last_surface_heat_flux_quadrature

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
        self._last_surface_heat_flux_quadrature = None
        if self.config.coupling_integrator == "coupled-ssprk3":
            result = self._coupled_ssprk3_step(state, timestep)
            if self.momentum.config.projection_method == "fpj2":
                self.momentum._accept_fpj2_pressure(
                    result.pressure,
                    timestep,
                )
            self._last_surface_heat_flux_quadrature = (
                result.surface_heat_flux_quadrature
            )
            return AMDBoussinesqState(
                self.momentum.enforce_boundaries(result.velocity),
                result.potential_temperature,
                result.pressure,
                state.time + timestep,
                state.step + 1,
            )
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
                    buoyancy,
                )
            return _velocity_sum(
                (1.0, self.momentum.tendency(stage_velocity, stage_time)),
                (1.0, buoyancy),
            )

        history = self.momentum.fpj2_state
        if (
            self.momentum.config.projection_method == "fpj2"
            and self.momentum._fpj2_history_is_usable(timestep)
        ):
            projected = fpj2_ssprk3_velocity_step(
                state.velocity,
                tendency=stage_tendency,
                projector=self.momentum.projector,
                timestep=timestep,
                current_pressure=history.current_pressure,
                previous_pressure=history.previous_pressure,
                current_timestep=history.current_timestep,
                previous_timestep=history.previous_timestep,
                time=state.time,
            )
        else:
            initial_pressure = (
                state.pressure if history is None else history.current_pressure
            )
            projected = projected_ssprk3_velocity_pressure_step(
                state.velocity,
                tendency=stage_tendency,
                projector=self.momentum.projector,
                timestep=timestep,
                time=state.time,
                initial_pressure=initial_pressure,
            )
        if self.momentum.config.projection_method == "fpj2":
            self.momentum._accept_fpj2_pressure(
                projected.pressure,
                timestep,
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
        if not math.isfinite(target_cfl) or target_cfl <= 0.0:
            raise ValueError("target CFL must be positive and finite")
        if not math.isfinite(target_diffusive_cfl) or target_diffusive_cfl <= 0.0:
            raise ValueError("target diffusive CFL must be positive and finite")
        advective, momentum_diffusive, scalar_diffusive = (
            float(value) for value in self.stability_rates(state)
        )
        if advective <= 0.0:
            raise ValueError("cannot choose a CFL step for zero velocity")
        candidates = [target_cfl / advective]
        for rate in (momentum_diffusive, scalar_diffusive):
            if rate > 0.0:
                candidates.append(target_diffusive_cfl / rate)
        return min(candidates)

    def _stability_rates(
        self,
        velocity: MACVelocity,
        potential_temperature: Array,
    ) -> tuple[Array, Array, Array]:
        cells = _cell_velocity(velocity)
        gradient = self.momentum.velocity_gradient(cells)
        momentum_diffusivity = self.momentum.sgs_viscosity(
            cells,
            gradient=gradient,
        )
        scalar_diffusivity = self.scalar.amd_diffusivity(
            potential_temperature,
            gradient,
        )
        inverse_spacing_squared = (
            1.0 / self.momentum.dx**2
            + 1.0 / self.momentum.dy**2
            + 1.0 / self.momentum.dz**2
        )
        return (
            self.momentum.cfl_rate(velocity),
            2.0 * jnp.max(momentum_diffusivity) * inverse_spacing_squared,
            2.0 * jnp.max(scalar_diffusivity) * inverse_spacing_squared,
        )

    def stability_rates(
        self,
        state: AMDBoussinesqState,
    ) -> tuple[Array, Array, Array]:
        """Return shared-gradient advective and AMD stability rates."""
        return self._compiled_stability_rates(
            state.velocity,
            state.potential_temperature,
        )

    def pre_step_metrics(
        self,
        state: AMDBoussinesqState,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Return rates, scalar mean, and heat flux in one device launch."""
        return self._compiled_pre_step_metrics(
            state.velocity,
            state.potential_temperature,
            jnp.asarray(state.time, dtype=state.potential_temperature.dtype),
        )

    def accepted_state_metrics(
        self,
        state: AMDBoussinesqState,
    ) -> tuple[Array, Array, Array]:
        """Return scalar mean, heat flux, and divergence after a step."""
        return self._compiled_accepted_state_metrics(
            state.velocity,
            state.potential_temperature,
            jnp.asarray(state.time, dtype=state.potential_temperature.dtype),
        )

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
        strain = 0.5 * (diagnostic_gradient + jnp.swapaxes(diagnostic_gradient, -1, -2))
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
        scalar_flux_z_at_cells = 0.5 * (scalar_flux_z[:-1] + scalar_flux_z[1:])
        delta = (self.momentum.dx * self.momentum.dy * self.momentum.dz) ** (1.0 / 3.0)
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
        scalar_length = delta * jnp.sqrt(jnp.maximum(effective_scalar_coefficient, 0.0))
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
        momentum_regularization = self.momentum.regularization_tendency(
            velocity,
            cells,
        )
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
        numerical_energy_dissipation = -jnp.einsum(
            "...i,...i->...",
            velocity_fluctuation,
            momentum_regularization,
        )
        ko6_energy_dissipation = (
            numerical_energy_dissipation
            if self.momentum.config.regularization_scheme == "ko6"
            else jnp.zeros_like(numerical_energy_dissipation)
        )
        mp5_energy_dissipation = (
            numerical_energy_dissipation
            if self.momentum.config.regularization_scheme == "mp5"
            else jnp.zeros_like(numerical_energy_dissipation)
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
            ko6_energy_dissipation,
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
