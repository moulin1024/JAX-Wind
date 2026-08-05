"""Unified neutral and thermally stratified ABL solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxwind.pressure import (
    MACVelocity,
    mac_divergence,
    projected_ssprk3_velocity_pressure_step,
)

from .operators import (
    MomentumOperators,
    PreparedIMEXStep,
    ScalarOperators,
    _cell_length_scales,
    _cell_velocity,
    _cells_to_faces,
    _velocity_sum,
)
from .surface_layer import MoninObukhovWallLaw, SurfaceLayerFluxes


Array = jax.Array


@dataclass(frozen=True, slots=True)
class ThermodynamicsConfig:
    """Physical constants for an actively transported potential temperature."""

    gravity: float = 9.81
    reference_potential_temperature: float = 300.0
    sgs_dissipation_coefficient: float = 0.93
    scalar_variance_coefficient: float = 2.02
    surface_potential_temperature: float | None = None
    surface_temperature_tendency: float = 0.0
    thermal_roughness_length: float | None = None
    rayleigh_sponge_start_height: float | None = None
    rayleigh_sponge_maximum_rate: float = 0.0
    rayleigh_reference_temperature_at_zero: float | None = None
    rayleigh_reference_temperature_gradient: float = 0.0

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
        sponge_values = (
            self.rayleigh_sponge_maximum_rate,
            self.rayleigh_reference_temperature_gradient,
        )
        if not all(math.isfinite(value) for value in sponge_values):
            raise ValueError("Rayleigh sponge controls must be finite")
        if self.rayleigh_sponge_maximum_rate < 0.0:
            raise ValueError("Rayleigh sponge maximum rate must be nonnegative")
        if self.rayleigh_sponge_start_height is None:
            if self.rayleigh_sponge_maximum_rate != 0.0:
                raise ValueError("Rayleigh sponge rate requires a start height")
            if self.rayleigh_reference_temperature_at_zero is not None:
                raise ValueError("Rayleigh reference requires an active sponge")
        else:
            if (
                not math.isfinite(self.rayleigh_sponge_start_height)
                or self.rayleigh_sponge_start_height < 0.0
            ):
                raise ValueError("Rayleigh sponge start height must be nonnegative")
            if self.rayleigh_sponge_maximum_rate <= 0.0:
                raise ValueError("active Rayleigh sponge rate must be positive")
            if self.rayleigh_reference_temperature_at_zero is None or not math.isfinite(
                self.rayleigh_reference_temperature_at_zero
            ):
                raise ValueError("active Rayleigh sponge requires a temperature target")

    @property
    def buoyancy_coefficient(self) -> float:
        return self.gravity / self.reference_potential_temperature


class ABLState(NamedTuple):
    """Accepted velocity, optional potential temperature, and pressure."""

    velocity: MACVelocity
    potential_temperature: Array | None
    pressure: Array
    time: float
    step: int


class PreparedABLStep(NamedTuple):
    """Device work prepared for one accepted ABL state."""

    rates: Array
    momentum: PreparedIMEXStep | None


class ABLDiagnosticFields(NamedTuple):
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
    momentum_numerical_flux_z: Array
    scalar_numerical_flux_z: Array
    surface_momentum_stress: Array
    surface_heat_flux: Array
    surface_friction_velocity: Array
    surface_temperature_scale: Array
    surface_obukhov_length: Array


class ABLSolver:
    """Single solver for neutral and thermally stratified ABL flows.

    Potential temperature is advanced for half a step with the accepted
    velocity.  Its midpoint value supplies a frozen buoyancy force to all
    three projected momentum stages, after which the scalar completes the
    second half step with the accepted new velocity.
    """

    def __init__(
        self,
        momentum: MomentumOperators,
        scalar: ScalarOperators | None = None,
        config: ThermodynamicsConfig | None = None,
    ) -> None:
        if (scalar is None) != (config is None):
            raise ValueError(
                "scalar operators and thermodynamics must be enabled together"
            )
        if scalar is not None and momentum.grid != scalar.grid:
            raise ValueError("momentum and scalar grids must match")
        if config is not None and momentum.lasd_closure is not None:
            raise ValueError("thermodynamic coupling currently requires AMD momentum")
        if (
            config is not None
            and momentum.config.wall_temporal_filter_timescale is not None
        ):
            raise ValueError("thermodynamics requires a memoryless wall input")
        self.momentum = momentum
        self.scalar = scalar
        self.config = config
        self.grid = momentum.grid
        self._rayleigh_rate: Array | None = None
        self._rayleigh_temperature_target: Array | None = None
        if config is None:
            self.surface_law = None
            self._compiled_runtime_rates = jax.jit(self._neutral_runtime_rates)
            self._compiled_diagnostic_metrics = jax.jit(
                self._neutral_diagnostic_metrics
            )
            if momentum.config.sgs_time_integration == "imex_ark3":
                self._compiled_neutral_imex_prepare = jax.jit(
                    self._neutral_imex_prepare
                )
            self._compiled_profile = jax.jit(self._neutral_profile)
            return
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
        if config.rayleigh_sponge_start_height is not None:
            if self.surface_law is None:
                raise ValueError("Rayleigh scalar sponge requires surface coupling")
            if momentum.config.geostrophic_wind is None:
                raise ValueError("Rayleigh momentum sponge requires geostrophic wind")
            top = float(self.grid.z_faces[-1])
            start = config.rayleigh_sponge_start_height
            if start >= top:
                raise ValueError(
                    "Rayleigh sponge start height must be below the domain top"
                )
            z = jnp.asarray(self.grid.z_centers)
            ramp = jnp.clip((z - start) / (top - start), 0.0, 1.0)
            self._rayleigh_rate = config.rayleigh_sponge_maximum_rate * ramp**2
            self._rayleigh_temperature_target = (
                config.rayleigh_reference_temperature_at_zero
                + config.rayleigh_reference_temperature_gradient * z
            )
        self._compiled_surface_scalar_tendency = jax.jit(self._surface_scalar_tendency)
        self._compiled_surface_scalar_step = self._dispatched_surface_scalar_step
        self._compiled_surface_momentum_tendency = jax.jit(
            self._surface_momentum_tendency
        )
        self._compiled_surface_momentum_wall_stress = jax.jit(
            self._surface_momentum_wall_stress
        )
        self._compiled_rayleigh_momentum_tendency = jax.jit(
            self._rayleigh_momentum_tendency
        )
        self._compiled_stability_rates = jax.jit(self._stability_rates)
        self._compiled_runtime_rates = jax.jit(self._thermal_runtime_rates)
        self._compiled_diagnostic_metrics = jax.jit(
            self._thermal_diagnostic_metrics
        )
        self._compiled_profile = jax.jit(self._thermal_profile)

    def surface_potential_temperature(self, time: Array | float) -> Array:
        """Return the linearly evolving prescribed surface temperature."""
        if self.config.surface_potential_temperature is None:
            raise RuntimeError("the surface has no prescribed temperature")
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
        first_cell_temperature = potential_temperature[0]
        return self.surface_law.surface_fluxes(
            horizontal_velocity,
            first_cell_temperature,
            self.surface_potential_temperature(time),
            self.momentum.wall_cell_height,
        )

    def surface_layer_fluxes(
        self,
        state: ABLState,
    ) -> SurfaceLayerFluxes:
        """Diagnose lower-boundary momentum and heat fluxes for a state."""
        if state.potential_temperature is None:
            return self.momentum.wall_fluxes(
                self.momentum.cell_centered_velocity(state.velocity)
            )
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

    def _surface_momentum_wall_stress(
        self,
        velocity: MACVelocity,
        potential_temperature: Array,
        time: Array,
    ) -> Array:
        return self._momentum_wall_stress(
            self._surface_layer_fluxes(
                velocity,
                potential_temperature,
                time,
            )
        )

    def _surface_scalar_tendency(
        self,
        scalar: Array,
        velocity: MACVelocity,
        time: Array,
    ) -> Array:
        fluxes = self._surface_layer_fluxes(velocity, scalar, time)
        tendency = self.scalar.tendency(
            scalar,
            velocity,
            lower_surface_flux=fluxes.heat_flux,
        )
        return tendency + self._rayleigh_scalar_tendency(scalar)

    def _rayleigh_scalar_tendency(self, scalar: Array) -> Array:
        if self._rayleigh_rate is None or self._rayleigh_temperature_target is None:
            return jnp.zeros_like(scalar)
        return self._rayleigh_rate[:, None, None] * (
            self._rayleigh_temperature_target[:, None, None] - scalar
        )

    def rayleigh_scalar_volume_rate(self, scalar: Array) -> Array:
        """Return the domain-mean scalar source supplied by the top sponge."""
        return self.scalar.volume_mean(self._rayleigh_scalar_tendency(scalar))

    def _rayleigh_momentum_tendency(
        self,
        velocity: MACVelocity,
        _time: float | Array = 0.0,
    ) -> MACVelocity:
        if self._rayleigh_rate is None:
            return MACVelocity(
                jnp.zeros_like(velocity.x),
                jnp.zeros_like(velocity.y),
                jnp.zeros_like(velocity.z),
            )
        cells = _cell_velocity(velocity)
        geostrophic_u, geostrophic_v = self.momentum.config.geostrophic_wind
        target = jnp.zeros_like(cells)
        target = target.at[..., 0].set(geostrophic_u)
        target = target.at[..., 1].set(geostrophic_v)
        tendency = self._rayleigh_rate[:, None, None, None] * (target - cells)
        return _cells_to_faces(tendency)

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
            (1.0, self._rayleigh_momentum_tendency(velocity, time)),
        )

    def initial_state(
        self,
        velocity: MACVelocity,
        potential_temperature: Array | None,
        *,
        time: float = 0.0,
        step: int = 0,
        pressure: Array | None = None,
    ) -> ABLState:
        if self.config is None:
            if potential_temperature is not None:
                raise ValueError(
                    "thermodynamics-disabled state cannot contain temperature"
                )
        else:
            if potential_temperature is None:
                raise ValueError("thermodynamics requires potential temperature")
            self.scalar._validate_scalar(potential_temperature)
        if not math.isfinite(time) or time < 0.0:
            raise ValueError("initial time must be finite and nonnegative")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("initial step must be a nonnegative integer")
        dtype = velocity.x.dtype
        initial_pressure = (
            jnp.zeros(self.grid.shape, dtype=dtype)
            if pressure is None
            else jnp.asarray(pressure, dtype=dtype)
        )
        if initial_pressure.shape != self.grid.shape:
            raise ValueError("initial pressure shape does not match the grid")
        self.momentum.restore_pressure(initial_pressure)
        return ABLState(
            self.momentum.enforce_boundaries(velocity),
            potential_temperature,
            initial_pressure,
            float(time),
            step,
        )

    def buoyancy_tendency(self, potential_temperature: Array) -> MACVelocity:
        """Return face acceleration after removing the hydrostatic plane mean."""
        self.scalar._validate_scalar(potential_temperature)
        anomaly = potential_temperature - self.momentum.horizontal_mean(
            potential_temperature,
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
        state: ABLState,
        *,
        timestep: float,
        prepared: PreparedABLStep | None = None,
    ) -> ABLState:
        """Advance one accepted neutral or thermally coupled ABL step."""
        if not math.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("timestep must be positive and finite")
        if state.potential_temperature is None:
            velocity = self.momentum.step(
                state.velocity,
                timestep=timestep,
                time=state.time,
                prepared=None if prepared is None else prepared.momentum,
            )
            return ABLState(
                velocity,
                None,
                self.momentum.pressure,
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

        if self.momentum.config.sgs_time_integration == "imex_ark3":
            wall_stress_provider = (
                None
                if self.surface_law is None
                else lambda stage_velocity, stage_time: (
                    self._compiled_surface_momentum_wall_stress(
                        stage_velocity,
                        midpoint_temperature,
                        jnp.asarray(stage_time, dtype=midpoint_temperature.dtype),
                    )
                )
            )
            projected = self.momentum._imex_ark3_step(
                state.velocity,
                timestep=timestep,
                time=state.time,
                lasd_coefficient=self.momentum._active_lasd_coefficient(state.velocity),
                wall_velocity=self.momentum.active_wall_velocity(state.velocity),
                initial_pressure=state.pressure,
                explicit_forcing=buoyancy,
                explicit_forcing_provider=(
                    self._compiled_rayleigh_momentum_tendency
                    if self._rayleigh_rate is not None
                    else None
                ),
                wall_stress_provider=wall_stress_provider,
            )
        else:
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
        return ABLState(
            velocity,
            potential_temperature,
            projected.pressure,
            state.time + timestep,
            state.step + 1,
        )

    def timestep_for_cfl(
        self,
        state: ABLState,
        target_cfl: float,
        target_diffusive_cfl: float = 0.5,
    ) -> float:
        """Return the active joint momentum/temperature stability limit."""
        if not math.isfinite(target_cfl) or target_cfl <= 0.0:
            raise ValueError("target CFL must be positive and finite")
        if not math.isfinite(target_diffusive_cfl) or target_diffusive_cfl <= 0.0:
            raise ValueError("target diffusive CFL must be positive and finite")
        advective, momentum_diffusive, scalar_diffusive = (
            float(value) for value in jax.device_get(self.runtime_rates(state))
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
        time: Array,
    ) -> tuple[Array, Array, Array]:
        fluxes = self._surface_layer_fluxes(
            velocity,
            potential_temperature,
            time,
        )
        return self._stability_rates_from_fluxes(
            velocity,
            potential_temperature,
            time,
            fluxes,
        )

    def _stability_rates_from_fluxes(
        self,
        velocity: MACVelocity,
        potential_temperature: Array,
        time: Array,
        fluxes: SurfaceLayerFluxes,
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
        wall_velocity = self.momentum.wall_velocity(cells)
        wall_rate = self.momentum.surface_momentum_stability_rate(
            wall_velocity,
            fluxes.momentum_stress,
        )
        thermal_rate = jnp.asarray(0.0, dtype=potential_temperature.dtype)
        if self.surface_law is not None:
            first_cell_temperature = potential_temperature[0]
            temperature_difference = (
                first_cell_temperature - self.surface_potential_temperature(time)
            )
            epsilon = jnp.finfo(potential_temperature.dtype).tiny
            thermal_rate = jnp.max(
                jnp.where(
                    jnp.abs(temperature_difference) > epsilon,
                    jnp.abs(fluxes.heat_flux)
                    / (
                        jnp.maximum(jnp.abs(temperature_difference), epsilon)
                        * self.scalar.dz_cell[0]
                    ),
                    0.0,
                )
            )
        return (
            jnp.maximum(
                self.momentum.cfl_rate(velocity),
                jnp.maximum(
                    jnp.maximum(wall_rate, thermal_rate),
                    jnp.asarray(
                        self.config.rayleigh_sponge_maximum_rate,
                        dtype=potential_temperature.dtype,
                    ),
                ),
            ),
            self.momentum.explicit_sgs_diffusion_rate(
                momentum_diffusivity,
                include_vertical=(
                    self.momentum.config.sgs_time_integration != "imex_ark3"
                ),
            ),
            self.scalar.explicit_diffusion_rate(scalar_diffusivity),
        )

    def stability_rates(
        self,
        state: ABLState,
    ) -> tuple[Array, Array, Array]:
        """Return shared-gradient advective and AMD stability rates."""
        rates = self.runtime_rates(state)
        return rates[0], rates[1], rates[2]

    def _neutral_runtime_rates(
        self,
        velocity: MACVelocity,
        lasd_coefficient: Array,
    ) -> Array:
        cells = _cell_velocity(velocity)
        gradient = self.momentum.velocity_gradient(cells)
        viscosity = self.momentum.sgs_viscosity(
            cells,
            lasd_coefficient,
            gradient=gradient,
        )
        advective_rate = jnp.maximum(
            self.momentum.cfl_rate(velocity),
            self.momentum.wall_stability_rate(cells),
        )
        momentum_diffusive_rate = self.momentum.explicit_sgs_diffusion_rate(
            viscosity,
            include_vertical=(
                self.momentum.config.sgs_time_integration != "imex_ark3"
            ),
        )
        return jnp.stack(
            (
                advective_rate,
                momentum_diffusive_rate,
                jnp.asarray(0.0, dtype=cells.dtype),
            )
        )

    def _neutral_imex_prepare(
        self,
        velocity: MACVelocity,
        lasd_coefficient: Array,
        wall_velocity: Array,
    ) -> tuple[MACVelocity, MACVelocity, Array, Array]:
        initial_explicit, initial_implicit, frozen_viscosity, rates = (
            self.momentum._compiled_imex_prepare(
                velocity,
                lasd_coefficient,
                wall_velocity,
            )
        )
        return (
            initial_explicit,
            initial_implicit,
            frozen_viscosity,
            jnp.concatenate(
                (rates, jnp.zeros((1,), dtype=velocity.x.dtype)),
            ),
        )

    def _thermal_runtime_rates(
        self,
        velocity: MACVelocity,
        potential_temperature: Array,
        time: Array,
    ) -> Array:
        return jnp.stack(
            self._stability_rates(
                velocity,
                potential_temperature,
                time,
            )
        )

    def _neutral_diagnostic_metrics(self, velocity: MACVelocity) -> Array:
        cells = _cell_velocity(velocity)
        divergence = mac_divergence(velocity, self.grid)
        return jnp.stack(
            (
                self.momentum.cfl_rate(velocity),
                self.momentum.pressure_solver.operator.norm(divergence),
                0.5 * self.momentum.volume_mean(jnp.sum(cells * cells, axis=-1)),
                jnp.asarray(jnp.nan, dtype=cells.dtype),
            )
        )

    def _thermal_diagnostic_metrics(
        self,
        velocity: MACVelocity,
        potential_temperature: Array,
    ) -> Array:
        cells = _cell_velocity(velocity)
        divergence = mac_divergence(velocity, self.grid)
        return jnp.stack(
            (
                self.momentum.cfl_rate(velocity),
                self.momentum.pressure_solver.operator.norm(divergence),
                0.5 * self.momentum.volume_mean(jnp.sum(cells * cells, axis=-1)),
                self.scalar.volume_mean(potential_temperature),
            )
        )

    def runtime_rates(self, state: ABLState) -> Array:
        """Return the three joint stability rates in one device vector."""
        if state.potential_temperature is None:
            return self._compiled_runtime_rates(
                state.velocity,
                self.momentum._active_lasd_coefficient(state.velocity),
            )
        return self._compiled_runtime_rates(
            state.velocity,
            state.potential_temperature,
            jnp.asarray(state.time, dtype=state.potential_temperature.dtype),
        )

    def diagnostic_metrics(self, state: ABLState) -> Array:
        """Return raw CFL, divergence, kinetic energy, and scalar mean."""
        if state.potential_temperature is None:
            return self._compiled_diagnostic_metrics(state.velocity)
        return self._compiled_diagnostic_metrics(
            state.velocity,
            state.potential_temperature,
        )

    def prepare_step(self, state: ABLState) -> PreparedABLStep:
        """Launch rates and reusable next-step work for an accepted state."""
        if (
            state.potential_temperature is None
            and self.momentum.config.sgs_time_integration == "imex_ark3"
        ):
            coefficient = self.momentum._active_lasd_coefficient(state.velocity)
            wall_velocity = self.momentum.active_wall_velocity(state.velocity)
            initial_explicit, initial_implicit, frozen_viscosity, rates = (
                self._compiled_neutral_imex_prepare(
                    state.velocity,
                    coefficient,
                    wall_velocity,
                )
            )
            return PreparedABLStep(
                rates,
                PreparedIMEXStep(
                    initial_explicit,
                    initial_implicit,
                    frozen_viscosity,
                    coefficient,
                    wall_velocity,
                ),
            )
        return PreparedABLStep(self.runtime_rates(state), None)

    def runtime_metrics(self, state: ABLState) -> Array:
        """Backward-compatible alias for the compact stability-rate vector."""
        return self.runtime_rates(state)

    def _profile_statistics(
        self,
        cells: Array,
        viscosity: Array,
        scalar: Array | None,
    ) -> Array:
        mean = self.momentum.horizontal_mean(cells)
        fluctuation = cells - mean[:, None, None, :]
        variance = self.momentum.horizontal_mean(fluctuation * fluctuation)
        if scalar is None:
            scalar_mean = jnp.full(mean.shape[0], jnp.nan, dtype=cells.dtype)
            scalar_variance = jnp.full_like(scalar_mean, jnp.nan)
            resolved_wscalar = jnp.full_like(scalar_mean, jnp.nan)
        else:
            scalar_mean = self.momentum.horizontal_mean(scalar)
            scalar_fluctuation = scalar - scalar_mean[:, None, None]
            scalar_variance = self.momentum.horizontal_mean(
                scalar_fluctuation * scalar_fluctuation
            )
            resolved_wscalar = self.momentum.horizontal_mean(
                fluctuation[..., 2] * scalar_fluctuation
            )
        return jnp.column_stack(
            (
                jnp.asarray(self.grid.z_centers, dtype=cells.dtype),
                mean,
                variance,
                scalar_mean,
                scalar_variance,
                self.momentum.horizontal_mean(
                    fluctuation[..., 0] * fluctuation[..., 2]
                ),
                self.momentum.horizontal_mean(
                    fluctuation[..., 1] * fluctuation[..., 2]
                ),
                resolved_wscalar,
                self.momentum.horizontal_mean(viscosity),
            )
        )

    def _neutral_profile(
        self,
        velocity: MACVelocity,
        lasd_coefficient: Array,
    ) -> Array:
        cells = _cell_velocity(velocity)
        gradient = self.momentum.velocity_gradient(cells)
        viscosity = self.momentum.sgs_viscosity(
            cells,
            lasd_coefficient,
            gradient=gradient,
        )
        return self._profile_statistics(cells, viscosity, None)

    def _thermal_profile(
        self,
        velocity: MACVelocity,
        potential_temperature: Array,
    ) -> Array:
        cells = _cell_velocity(velocity)
        gradient = self.momentum.velocity_gradient(cells)
        viscosity = self.momentum.sgs_viscosity(cells, gradient=gradient)
        return self._profile_statistics(cells, viscosity, potential_temperature)

    def profile(self, state: ABLState) -> Array:
        """Return the compact horizontally reduced profile on the device."""
        if state.potential_temperature is None:
            return self._compiled_profile(
                state.velocity,
                self.momentum._active_lasd_coefficient(state.velocity),
            )
        return self._compiled_profile(state.velocity, state.potential_temperature)

    def diagnostic_fields(
        self,
        state: ABLState,
    ) -> ABLDiagnosticFields:
        """Diagnose AMD, buoyancy, scalar variance, and MP5 contributions."""
        if state.potential_temperature is None:
            raise ValueError("thermodynamic diagnostic fields require temperature")
        velocity = state.velocity
        theta = state.potential_temperature
        cells = _cell_velocity(velocity)
        velocity_gradient = self.momentum.velocity_gradient(cells)
        surface_fluxes = self._surface_layer_fluxes(
            velocity,
            theta,
            state.time,
        )
        diagnostic_gradient = velocity_gradient
        wall_velocity = self.momentum.wall_velocity(cells)
        wall_speed = jnp.linalg.norm(wall_velocity, axis=-1)
        epsilon = jnp.finfo(theta.dtype).tiny
        direction = wall_velocity / jnp.maximum(wall_speed[..., None], epsilon)
        first_height = self.momentum.grid.z_centers[0] - self.momentum.grid.z_faces[0]
        inverse_obukhov = jnp.where(
            jnp.isfinite(surface_fluxes.obukhov_length),
            1.0 / surface_fluxes.obukhov_length,
            0.0,
        )
        momentum_similarity = jnp.ones_like(wall_speed)
        if self.surface_law is not None:
            zeta = jnp.clip(
                first_height * inverse_obukhov,
                -self.surface_law.maximum_abs_zeta,
                self.surface_law.maximum_abs_zeta,
            )
            momentum_similarity = jnp.where(
                zeta >= 0.0,
                1.0 + self.surface_law.stable_momentum_beta * zeta,
                jnp.maximum(
                    1.0 - self.surface_law.unstable_momentum_gamma * zeta,
                    1.0,
                )
                ** (-0.25),
            )
        wall_shear = (
            surface_fluxes.friction_velocity
            * momentum_similarity
            / (self.momentum.config.von_karman * first_height)
        )
        diagnostic_gradient = diagnostic_gradient.at[0, ..., 0, 2].set(
            wall_shear * direction[..., 0]
        )
        diagnostic_gradient = diagnostic_gradient.at[0, ..., 1, 2].set(
            wall_shear * direction[..., 1]
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
        scalar_flux_z_at_cells = (self.momentum.z_faces[1:] - self.momentum.z_centers)[
            :, None, None
        ] / self.scalar.dz_cell[:, None, None] * scalar_flux_z[:-1] + (
            self.momentum.z_centers - self.momentum.z_faces[:-1]
        )[:, None, None] / self.scalar.dz_cell[:, None, None] * scalar_flux_z[1:]
        delta = jnp.prod(
            _cell_length_scales(self.momentum.metrics, theta.dtype),
            axis=-1,
        ) ** (1.0 / 3.0)
        production = (
            momentum_diffusivity * strain_magnitude_squared
            + self.config.buoyancy_coefficient * scalar_flux_z_at_cells
        )
        sgs_tke = jnp.maximum(
            production * delta / self.config.sgs_dissipation_coefficient,
            0.0,
        ) ** (2.0 / 3.0)

        interior_gradient = (theta[1:] - theta[:-1]) / self.scalar.dz_center[
            :, None, None
        ]
        wall_diffusivity = self.momentum.surface_mean(scalar_diffusivity[0])
        lower_surface_flux = self.momentum.surface_mean(surface_fluxes.heat_flux)
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

        velocity_mean = self.momentum.horizontal_mean(cells, keepdims=True)
        velocity_fluctuation = cells - velocity_mean
        momentum_numerical = self.momentum.advection_dissipation(
            velocity,
            cells,
        )
        theta_fluctuation = theta - self.momentum.horizontal_mean(
            theta,
            keepdims=True,
        )
        scalar_numerical = self.scalar.advection_dissipation(theta, velocity)
        momentum_numerical_flux_z = self.momentum.vertical_advection_dissipation_flux(
            velocity, cells
        )
        scalar_numerical_flux_z = self.scalar.vertical_advection_dissipation_flux(
            theta,
            velocity,
        )
        amd_energy_dissipation = -self.momentum.resolved_tke_sgs_dissipation(
            cells,
            gradient=velocity_gradient,
        )
        # The tuple field names remain MP5-compatible for existing checkpoints
        # and diagnostics; the values represent whichever limiter is selected.
        numerical_energy_dissipation = -jnp.einsum(
            "...i,...i->...",
            velocity_fluctuation,
            momentum_numerical,
        )
        numerical_scalar_dissipation = -theta_fluctuation * scalar_numerical
        return ABLDiagnosticFields(
            momentum_diffusivity,
            scalar_diffusivity,
            scalar_flux_x,
            scalar_flux_y,
            scalar_flux_z,
            sgs_tke,
            scalar_variance_numerator,
            scalar_variance,
            amd_energy_dissipation,
            numerical_energy_dissipation,
            amd_scalar_dissipation,
            numerical_scalar_dissipation,
            momentum_numerical_flux_z,
            scalar_numerical_flux_z,
            surface_fluxes.momentum_stress,
            surface_fluxes.heat_flux,
            surface_fluxes.friction_velocity,
            surface_fluxes.temperature_scale,
            surface_fluxes.obukhov_length,
        )


__all__ = [
    "ABLDiagnosticFields",
    "ABLSolver",
    "ABLState",
    "PreparedABLStep",
    "ThermodynamicsConfig",
]
