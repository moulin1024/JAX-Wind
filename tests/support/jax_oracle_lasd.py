"""LASD and scalar-transport methods for the independent test oracle."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import jax.numpy as jnp

from jaxwind.interpreters._jax_fringe import plateau_fringe_mask

from jaxwind.domain import (
    Accepted,
    Cell,
    Evaluated,
    Field,
    LasdTrajectoryXVelocity,
    LasdTrajectoryYVelocity,
    LasdTrajectoryZVelocity,
    MomentumLasdCoefficient,
    MomentumLasdLm,
    MomentumLasdMm,
    MomentumLasdNn,
    MomentumLasdQn,
    PassiveScalarTendency,
    PotentialTemperaturePerturbation,
    PotentialTemperatureTendency,
    ScalarLasdCoefficient,
    ScalarLasdLm,
    ScalarLasdMm,
    ScalarLasdNn,
    ScalarLasdQn,
)
from jaxwind.operators import VelocityVector
from jaxwind.physics.dry_flow import (
    FilteredNeutralLogWall,
    NeutralLogWall,
    StaticSmagorinsky,
)
from jaxwind.physics.boussinesq import (
    BoussinesqFields,
    ConservativeScalarAdvection,
    LinearBoussinesqBuoyancy,
    NoBuoyancy,
    NoRayleighDamping,
    RayleighGeostrophicDamping,
    ScalarFluxBoundary,
    StaticSmagorinskyScalarFlux,
)
from jaxwind.physics.lasd import (
    DiagnosticLasdConstants,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LasdClosureEventDiagnostic,
    LasdClosureMemory,
    LasdDiagnosticFields,
    MomentumLasdMemory,
    ScalarLasdMemory,
)
from jaxwind.physics.wind_tunnel import ConcurrentPrecursorFringe

from .jax_oracle_core import (
    OracleBoussinesqContext,
    OracleDryFlowContext,
    _cell_to_full_faces,
    _history_boundary,
    _lagrangian_average,
    _lasd_beta,
    _momentum_lasd_contractions,
    _oracle_tendency,
    _require_tiny_global,
    _safe_divide,
    _scalar_cell_gradient,
    _scalar_lasd_contractions,
    _strain_magnitude,
    _truncated_horizontal_derivative,
    _two_thirds_filter,
)


class OracleLasdMixin:
    """Reference closure and scalar-transport interpretation."""

    __slots__ = ()

    @staticmethod
    def _oracle_closure_field(
        template: Field, quantity: type, payload: Any
    ) -> Field:
        return Field(
            quantity,
            Cell,
            template.ownership,
            Accepted,
            payload.astype(template.payload.dtype),
        )

    def initialize_lasd_closure(
        self, fields: BoussinesqFields, model: Any
    ) -> BoussinesqFields:
        momentum_config = model.momentum.sgs
        scalar_config = model.scalar_sgs
        if not isinstance(
            momentum_config, LagrangianScaleDependentDynamic
        ) or not isinstance(
            scalar_config,
            LagrangianScaleDependentScalarFlux,
        ):
            raise TypeError("LASD initialization requires momentum and scalar LASD")
        scalar = fields.potential_temperature
        _require_tiny_global(scalar, Cell)
        zero = jnp.zeros_like(scalar.payload)
        momentum_coefficient = jnp.full_like(
            scalar.payload,
            momentum_config.initial_coefficient,
        )
        scalar_coefficient = jnp.full_like(
            scalar.payload,
            scalar_config.initial_coefficient,
        )
        field = lambda quantity, payload: self._oracle_closure_field(  # noqa: E731
            scalar,
            quantity,
            payload,
        )
        closure = LasdClosureMemory(
            MomentumLasdMemory(
                field(MomentumLasdCoefficient, momentum_coefficient),
                field(MomentumLasdLm, zero),
                field(MomentumLasdMm, zero),
                field(MomentumLasdQn, zero),
                field(MomentumLasdNn, zero),
                field(LasdTrajectoryXVelocity, zero),
                field(LasdTrajectoryYVelocity, zero),
                field(LasdTrajectoryZVelocity, zero),
            ),
            ScalarLasdMemory(
                field(ScalarLasdCoefficient, scalar_coefficient),
                field(ScalarLasdLm, zero),
                field(ScalarLasdMm, zero),
                field(ScalarLasdQn, zero),
                field(ScalarLasdNn, zero),
            ),
            momentum_config.fingerprint + "|" + scalar_config.fingerprint,
        )
        return replace(fields, closure=closure)

    def relax_lasd_closure(
        self,
        fields: BoussinesqFields,
        target: LasdClosureMemory,
        fringe: ConcurrentPrecursorFringe,
        dt: float,
    ) -> BoussinesqFields:
        """Exponentially nudge all main LASD memory toward the precursor."""

        closure = fields.closure
        if not isinstance(closure, LasdClosureMemory) or not isinstance(
            target, LasdClosureMemory
        ):
            raise TypeError("fringe relaxation requires LASD closure memory")
        if closure.configuration_fingerprint != target.configuration_fingerprint:
            raise ValueError("main and precursor LASD fingerprints do not match")
        grid = fields.potential_temperature.ownership.grid
        rise_width, fall_width = fringe.resolved_widths(grid.lx)
        dtype = fields.potential_temperature.payload.dtype
        x = (jnp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
        mask = plateau_fringe_mask(
            x,
            start_x=fringe.start_x,
            end_x=grid.lx,
            rise_width=rise_width,
            fall_width=fall_width,
        )
        blend = -jnp.expm1(
            -jnp.asarray(dt, dtype)
            * mask
            / jnp.asarray(fringe.relaxation_time, dtype)
        )[None, None, :]

        def field(current: Field, target_field: Field) -> Field:
            if (
                current.quantity is not target_field.quantity
                or current.ownership != target_field.ownership
            ):
                raise ValueError("main and precursor LASD fields do not align")
            payload = current.payload + blend * (
                target_field.payload - current.payload
            )
            return self._oracle_closure_field(
                current,
                current.quantity,
                payload,
            )

        current_m = closure.momentum
        target_m = target.momentum
        current_s = closure.scalar
        target_s = target.scalar
        relaxed = LasdClosureMemory(
            MomentumLasdMemory(
                *(field(left, right) for left, right in zip(
                    current_m.fields(),
                    target_m.fields(),
                    strict=True,
                ))
            ),
            ScalarLasdMemory(
                *(field(left, right) for left, right in zip(
                    current_s.fields(),
                    target_s.fields(),
                    strict=True,
                ))
            ),
            closure.configuration_fingerprint,
        )
        return replace(fields, closure=relaxed)

    def prepare_lasd_closure(
        self,
        fields: BoussinesqFields,
        model: Any,
        clock: Any,
        dt: float,
    ) -> tuple[BoussinesqFields, LasdClosureEventDiagnostic]:
        momentum_config = model.momentum.sgs
        scalar_config = model.scalar_sgs
        if not isinstance(
            momentum_config, LagrangianScaleDependentDynamic
        ) or not isinstance(
            scalar_config,
            LagrangianScaleDependentScalarFlux,
        ):
            raise TypeError("LASD event requires momentum and scalar LASD")
        closure = fields.closure
        if not isinstance(closure, LasdClosureMemory):
            raise TypeError("LASD event requires initialized closure memory")
        fingerprint = momentum_config.fingerprint + "|" + scalar_config.fingerprint
        if closure.configuration_fingerprint != fingerprint:
            raise ValueError("LASD memory fingerprint does not match the model")
        context = self.boussinesq_context(fields)
        momentum = context.momentum
        old_m = closure.momentum
        old_s = closure.scalar
        interval = momentum_config.update_interval
        trajectory_x = (
            old_m.trajectory_x.payload + momentum.velocity.x.payload / interval
        )
        trajectory_y = (
            old_m.trajectory_y.payload + momentum.velocity.y.payload / interval
        )
        trajectory_z = old_m.trajectory_z.payload + momentum.w_at_cells / interval
        should_update = (clock.step + 1) % interval == 0
        field = lambda template, payload: self._oracle_closure_field(  # noqa: E731
            template,
            template.quantity,
            payload,
        )
        if should_update:
            ratio = momentum_config.test_filter_ratio
            lm, mm = _momentum_lasd_contractions(
                context.momentum, momentum_config, ratio
            )
            qn, nn = _momentum_lasd_contractions(
                context.momentum,
                momentum_config,
                ratio**2,
            )
            first_update = clock.step == interval - 1
            old_lm = jnp.where(
                first_update, momentum_config.initial_coefficient * mm, old_m.lm.payload
            )
            old_mm = jnp.where(first_update, mm, old_m.mm.payload)
            old_qn = jnp.where(
                first_update, momentum_config.initial_coefficient * nn, old_m.qn.payload
            )
            old_nn = jnp.where(first_update, nn, old_m.nn.payload)
            old_lm, old_mm, old_qn, old_nn = (
                _history_boundary(value) for value in (old_lm, old_mm, old_qn, old_nn)
            )
            interval_dt = dt * interval
            lm_avg, mm_avg = _lagrangian_average(
                lm,
                mm,
                old_lm,
                old_mm,
                trajectory_x,
                trajectory_y,
                trajectory_z,
                grid=momentum.velocity.x.ownership.grid,
                interval_dt=interval_dt,
                timescale_coefficient=momentum_config.timescale_coefficient,
            )
            qn_avg, nn_avg = _lagrangian_average(
                qn,
                nn,
                old_qn,
                old_nn,
                trajectory_x,
                trajectory_y,
                trajectory_z,
                grid=momentum.velocity.x.ownership.grid,
                interval_dt=interval_dt,
                timescale_coefficient=momentum_config.timescale_coefficient,
            )
            coefficient_2d = jnp.maximum(_safe_divide(lm_avg, mm_avg), 0.0)
            coefficient_4d = jnp.maximum(_safe_divide(qn_avg, nn_avg), 0.0)
            momentum_coefficient = jnp.clip(
                _safe_divide(
                    coefficient_2d,
                    _lasd_beta(coefficient_2d, coefficient_4d, momentum_config),
                ),
                momentum_config.minimum_coefficient,
                momentum_config.maximum_coefficient,
            )

            scalar_lm, scalar_mm = _scalar_lasd_contractions(
                context,
                momentum_config,
                ratio,
            )
            scalar_qn, scalar_nn = _scalar_lasd_contractions(
                context,
                momentum_config,
                ratio**2,
            )
            old_scalar_lm = jnp.where(
                first_update,
                scalar_config.initial_coefficient * scalar_mm,
                old_s.lm.payload,
            )
            old_scalar_mm = jnp.where(first_update, scalar_mm, old_s.mm.payload)
            old_scalar_qn = jnp.where(
                first_update,
                scalar_config.initial_coefficient * scalar_nn,
                old_s.qn.payload,
            )
            old_scalar_nn = jnp.where(first_update, scalar_nn, old_s.nn.payload)
            old_scalar_lm, old_scalar_mm, old_scalar_qn, old_scalar_nn = (
                _history_boundary(value)
                for value in (
                    old_scalar_lm,
                    old_scalar_mm,
                    old_scalar_qn,
                    old_scalar_nn,
                )
            )
            scalar_lm_avg, scalar_mm_avg = _lagrangian_average(
                scalar_lm,
                scalar_mm,
                old_scalar_lm,
                old_scalar_mm,
                trajectory_x,
                trajectory_y,
                trajectory_z,
                grid=momentum.velocity.x.ownership.grid,
                interval_dt=interval_dt,
                timescale_coefficient=momentum_config.timescale_coefficient,
                timescale_a=lm_avg,
                timescale_b=mm_avg,
            )
            scalar_qn_avg, scalar_nn_avg = _lagrangian_average(
                scalar_qn,
                scalar_nn,
                old_scalar_qn,
                old_scalar_nn,
                trajectory_x,
                trajectory_y,
                trajectory_z,
                grid=momentum.velocity.x.ownership.grid,
                interval_dt=interval_dt,
                timescale_coefficient=momentum_config.timescale_coefficient,
                timescale_a=qn_avg,
                timescale_b=nn_avg,
            )
            scalar_lm_avg = jnp.where(scalar_lm_avg > 0.0, scalar_lm_avg, 1.0e-32)
            scalar_qn_avg = jnp.where(scalar_qn_avg > 0.0, scalar_qn_avg, 1.0e-32)
            scalar_2d = jnp.maximum(_safe_divide(scalar_lm_avg, scalar_mm_avg), 0.0)
            scalar_4d = jnp.maximum(_safe_divide(scalar_qn_avg, scalar_nn_avg), 0.0)
            scalar_coefficient = jnp.clip(
                _safe_divide(
                    scalar_2d,
                    _lasd_beta(
                        scalar_2d,
                        scalar_4d,
                        momentum_config,
                        scale_dependent=scalar_config.scale_dependent,
                    ),
                ),
                scalar_config.minimum_coefficient,
                scalar_config.maximum_coefficient,
            )
            zero = jnp.zeros_like(trajectory_x)
            new_momentum = MomentumLasdMemory(
                field(old_m.coefficient, momentum_coefficient),
                field(old_m.lm, lm_avg),
                field(old_m.mm, mm_avg),
                field(old_m.qn, qn_avg),
                field(old_m.nn, nn_avg),
                field(old_m.trajectory_x, zero),
                field(old_m.trajectory_y, zero),
                field(old_m.trajectory_z, zero),
            )
            new_scalar = ScalarLasdMemory(
                field(old_s.coefficient, scalar_coefficient),
                field(old_s.lm, scalar_lm_avg),
                field(old_s.mm, scalar_mm_avg),
                field(old_s.qn, scalar_qn_avg),
                field(old_s.nn, scalar_nn_avg),
            )
        else:
            new_momentum = MomentumLasdMemory(
                old_m.coefficient,
                old_m.lm,
                old_m.mm,
                old_m.qn,
                old_m.nn,
                field(old_m.trajectory_x, trajectory_x),
                field(old_m.trajectory_y, trajectory_y),
                field(old_m.trajectory_z, trajectory_z),
            )
            new_scalar = old_s
        prepared = replace(
            fields,
            closure=LasdClosureMemory(new_momentum, new_scalar, fingerprint),
        )
        return prepared, LasdClosureEventDiagnostic(should_update, clock.step, interval)

    def momentum_context(
        self,
        context: OracleBoussinesqContext,
    ) -> OracleDryFlowContext:
        return context.momentum

    def buoyancy_tendency(
        self,
        context: OracleBoussinesqContext,
        config: LinearBoussinesqBuoyancy | NoBuoyancy,
    ) -> VelocityVector:
        momentum = context.momentum
        if isinstance(config, NoBuoyancy):
            return _oracle_tendency(
                momentum,
                jnp.zeros_like(momentum.velocity.x.payload),
                jnp.zeros_like(momentum.velocity.y.payload),
                jnp.zeros_like(momentum.velocity.z.payload),
            )
        if not isinstance(config, LinearBoussinesqBuoyancy):
            raise TypeError("unsupported Boussinesq buoyancy choice")
        hydrostatic_free_theta = context.theta_on_faces - jnp.mean(
            context.theta_on_faces,
            axis=(-2, -1),
            keepdims=True,
        )
        z = config.acceleration_per_temperature * hydrostatic_free_theta
        z = z.at[0].set(0.0).at[-1].set(0.0)
        return _oracle_tendency(
            momentum,
            jnp.zeros_like(momentum.velocity.x.payload),
            jnp.zeros_like(momentum.velocity.y.payload),
            z,
        )

    def rayleigh_damping_tendency(
        self,
        context: OracleBoussinesqContext,
        config: NoRayleighDamping | RayleighGeostrophicDamping,
    ) -> VelocityVector:
        momentum = context.momentum
        velocity = momentum.velocity
        if isinstance(config, NoRayleighDamping):
            return _oracle_tendency(
                momentum,
                jnp.zeros_like(velocity.x.payload),
                jnp.zeros_like(velocity.y.payload),
                jnp.zeros_like(velocity.z.payload),
            )
        if not isinstance(config, RayleighGeostrophicDamping):
            raise TypeError("unsupported Rayleigh damping choice")
        grid = velocity.x.ownership.grid
        if config.start_height >= grid.lz:
            raise ValueError("Rayleigh damping must start below the domain top")
        dtype = velocity.x.payload.dtype
        depth = grid.lz - config.start_height
        cell_height = (jnp.arange(grid.nz, dtype=dtype) + 0.5) * grid.dz
        face_height = jnp.arange(grid.nz + 1, dtype=dtype) * grid.dz
        cell_eta = jnp.clip((cell_height - config.start_height) / depth, 0.0, 1.0)
        face_eta = jnp.clip((face_height - config.start_height) / depth, 0.0, 1.0)
        cell_rate = jnp.asarray(config.maximum_rate, dtype=dtype) * cell_eta**2
        face_rate = (
            jnp.asarray(
                config.maximum_rate,
                dtype=velocity.z.payload.dtype,
            )
            * face_eta.astype(velocity.z.payload.dtype) ** 2
        )
        return _oracle_tendency(
            momentum,
            -cell_rate[:, None, None]
            * (velocity.x.payload - config.geostrophic_x_velocity),
            -cell_rate[:, None, None]
            * (velocity.y.payload - config.geostrophic_y_velocity),
            -face_rate[:, None, None] * velocity.z.payload,
        )

    def _oracle_scalar_tendency(
        self,
        context: OracleBoussinesqContext,
        payload: Any,
    ) -> Field:
        scalar = context.potential_temperature
        quantity = (
            PotentialTemperatureTendency
            if scalar.quantity is PotentialTemperaturePerturbation
            else PassiveScalarTendency
        )
        return Field(
            quantity,
            Cell,
            scalar.ownership,
            Evaluated,
            payload.astype(scalar.payload.dtype),
        )

    def scalar_advection_tendency(
        self,
        context: OracleBoussinesqContext,
        config: ConservativeScalarAdvection,
    ) -> Field:
        if not isinstance(config, ConservativeScalarAdvection):
            raise TypeError("unsupported conservative scalar advection choice")
        momentum = context.momentum
        grid = context.potential_temperature.ownership.grid
        theta = context.potential_temperature.payload
        vertical_flux = _two_thirds_filter(
            momentum.velocity.z.payload * context.theta_on_faces,
            grid=grid,
        )
        tendency = -(
            _truncated_horizontal_derivative(
                momentum.velocity.x.payload * theta,
                grid=grid,
                axis="x",
            )
            + _truncated_horizontal_derivative(
                momentum.velocity.y.payload * theta,
                grid=grid,
                axis="y",
            )
            + (vertical_flux[1:] - vertical_flux[:-1]) / grid.dz
        )
        return self._oracle_scalar_tendency(context, tendency)

    def scalar_sgs_tendency(
        self,
        context: OracleBoussinesqContext,
        momentum_config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
        config: StaticSmagorinskyScalarFlux | LagrangianScaleDependentScalarFlux,
        boundary: ScalarFluxBoundary = ScalarFluxBoundary(),
    ) -> Field:
        static = isinstance(momentum_config, StaticSmagorinsky) and isinstance(
            config,
            StaticSmagorinskyScalarFlux,
        )
        dynamic = isinstance(
            momentum_config,
            LagrangianScaleDependentDynamic,
        ) and isinstance(config, LagrangianScaleDependentScalarFlux)
        if not (static or dynamic):
            raise TypeError("unsupported or inconsistent scalar SGS choice")
        momentum = context.momentum
        grid = context.potential_temperature.ownership.grid
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        cell_magnitude = _strain_magnitude(
            momentum.dudx,
            momentum.dudy,
            momentum.dudz_at_cells,
            momentum.dvdx,
            momentum.dvdy,
            momentum.dvdz_at_cells,
            momentum.dwdx_at_cells,
            momentum.dwdy_at_cells,
            momentum.dwdz,
        )
        face_magnitude = _strain_magnitude(
            _cell_to_full_faces(momentum.dudx),
            _cell_to_full_faces(momentum.dudy),
            momentum.dudz_on_faces,
            _cell_to_full_faces(momentum.dvdx),
            _cell_to_full_faces(momentum.dvdy),
            momentum.dvdz_on_faces,
            momentum.dwdx_on_faces,
            momentum.dwdy_on_faces,
            _cell_to_full_faces(momentum.dwdz),
        )
        if static:
            scalar_coefficient = jnp.full_like(
                cell_magnitude,
                momentum_config.coefficient**2 / config.turbulent_prandtl,
            )
        else:
            closure = momentum.closure
            if not isinstance(closure, LasdClosureMemory):
                raise TypeError("scalar LASD requires initialized closure memory")
            scalar_coefficient = closure.scalar.coefficient.payload
        stability = jnp.ones_like(cell_magnitude)
        if dynamic and config.stability_buoyancy_coefficient > 0.0:
            n2 = jnp.maximum(
                config.stability_buoyancy_coefficient
                * _scalar_cell_gradient(context)[..., 2],
                0.0,
            )
            richardson = n2 / jnp.maximum(cell_magnitude**2, 1.0e-24)
            stability = (1.0 + config.stability_beta * richardson) ** (
                -config.stability_power
            )
        effective_scalar_coefficient = scalar_coefficient * stability
        cell_diffusivity = effective_scalar_coefficient * delta**2 * cell_magnitude
        face_diffusivity = (
            _cell_to_full_faces(effective_scalar_coefficient)
            * delta**2
            * face_magnitude
        )
        qx = -cell_diffusivity * context.dtheta_dx
        qy = -cell_diffusivity * context.dtheta_dy
        qz = -face_diffusivity * context.dtheta_dz_on_faces
        qz = qz.at[0].set(boundary.lower_flux).at[-1].set(boundary.upper_flux)
        qz = _two_thirds_filter(qz, grid=grid)
        tendency = -(
            _truncated_horizontal_derivative(qx, grid=grid, axis="x")
            + _truncated_horizontal_derivative(qy, grid=grid, axis="y")
            + (qz[1:] - qz[:-1]) / grid.dz
        )
        return self._oracle_scalar_tendency(context, tendency)

    def lasd_diagnostic_fields(
        self,
        context: OracleBoussinesqContext,
        momentum_config: LagrangianScaleDependentDynamic,
        scalar_config: LagrangianScaleDependentScalarFlux,
        boundary: ScalarFluxBoundary = ScalarFluxBoundary(),
        constants: DiagnosticLasdConstants = DiagnosticLasdConstants(),
        wall: NeutralLogWall | FilteredNeutralLogWall | None = None,
    ) -> LasdDiagnosticFields:
        """Diagnose LASD energy/variance without adding prognostic state."""
        if not isinstance(
            momentum_config, LagrangianScaleDependentDynamic
        ) or not isinstance(
            scalar_config,
            LagrangianScaleDependentScalarFlux,
        ):
            raise TypeError("LASD diagnostics require momentum and scalar LASD")
        if not isinstance(constants, DiagnosticLasdConstants):
            raise TypeError("unsupported LASD diagnostic constants")
        closure = context.momentum.closure
        if not isinstance(closure, LasdClosureMemory):
            raise TypeError("LASD diagnostics require initialized closure memory")
        momentum = context.momentum
        grid = context.potential_temperature.ownership.grid
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        magnitude = _strain_magnitude(
            momentum.dudx,
            momentum.dudy,
            momentum.dudz_at_cells,
            momentum.dvdx,
            momentum.dvdy,
            momentum.dvdz_at_cells,
            momentum.dwdx_at_cells,
            momentum.dwdy_at_cells,
            momentum.dwdz,
        )
        diagnostic_magnitude = magnitude
        wall_gradient_factor = None
        if wall is not None:
            if not isinstance(
                wall,
                (NeutralLogWall, FilteredNeutralLogWall),
            ):
                raise TypeError("LASD diagnostic wall must be NeutralLogWall")
            reference_height = 0.5 * grid.dz
            if wall.roughness_length >= reference_height:
                raise ValueError("wall roughness must be below the first cell centre")
            wall_gradient_factor = 1.0 / (
                math.log(reference_height / wall.roughness_length) * reference_height
            )
            diagnostic_dudz = momentum.dudz_at_cells.at[0].set(
                momentum.velocity.x.payload[0] * wall_gradient_factor
            )
            diagnostic_dvdz = momentum.dvdz_at_cells.at[0].set(
                momentum.velocity.y.payload[0] * wall_gradient_factor
            )
            diagnostic_magnitude = _strain_magnitude(
                momentum.dudx,
                momentum.dudy,
                diagnostic_dudz,
                momentum.dvdx,
                momentum.dvdy,
                diagnostic_dvdz,
                momentum.dwdx_at_cells,
                momentum.dwdy_at_cells,
                momentum.dwdz,
            )
        face_magnitude = _strain_magnitude(
            _cell_to_full_faces(momentum.dudx),
            _cell_to_full_faces(momentum.dudy),
            momentum.dudz_on_faces,
            _cell_to_full_faces(momentum.dvdx),
            _cell_to_full_faces(momentum.dvdy),
            momentum.dvdz_on_faces,
            momentum.dwdx_on_faces,
            momentum.dwdy_on_faces,
            _cell_to_full_faces(momentum.dwdz),
        )
        momentum_diffusivity = (
            closure.momentum.coefficient.payload * delta**2 * magnitude
        )
        scalar_coefficient = closure.scalar.coefficient.payload
        stability = jnp.ones_like(magnitude)
        if scalar_config.stability_buoyancy_coefficient > 0.0:
            n2 = jnp.maximum(
                scalar_config.stability_buoyancy_coefficient
                * _scalar_cell_gradient(context)[..., 2],
                0.0,
            )
            richardson = n2 / jnp.maximum(magnitude**2, 1.0e-24)
            stability = (1.0 + scalar_config.stability_beta * richardson) ** (
                -scalar_config.stability_power
            )
        effective_scalar_coefficient = scalar_coefficient * stability
        scalar_diffusivity = effective_scalar_coefficient * delta**2 * magnitude
        face_diffusivity = (
            _cell_to_full_faces(effective_scalar_coefficient)
            * delta**2
            * face_magnitude
        )
        diagnostic_face_diffusivity = face_diffusivity
        if wall_gradient_factor is not None:
            zero_wall_cross_gradient = jnp.zeros_like(momentum.dwdx_on_faces[:1])
            wall_face_magnitude = _strain_magnitude(
                momentum.dudx[:1],
                momentum.dudy[:1],
                momentum.velocity.x.payload[:1] * wall_gradient_factor,
                momentum.dvdx[:1],
                momentum.dvdy[:1],
                momentum.velocity.y.payload[:1] * wall_gradient_factor,
                zero_wall_cross_gradient,
                zero_wall_cross_gradient,
                momentum.dwdz[:1],
            )
            wall_scalar_diffusivity = (
                effective_scalar_coefficient[:1]
                * delta**2
                * wall_face_magnitude
            )
            if constants.horizontal_homogeneous_wall:
                wall_scalar_diffusivity = jnp.full_like(
                    wall_scalar_diffusivity,
                    jnp.mean(wall_scalar_diffusivity),
                )
            diagnostic_face_diffusivity = face_diffusivity.at[0].set(
                wall_scalar_diffusivity[0]
            )
        flux_x = -scalar_diffusivity * context.dtheta_dx
        flux_y = -scalar_diffusivity * context.dtheta_dy
        flux_z = -face_diffusivity * context.dtheta_dz_on_faces
        flux_z = flux_z.at[0].set(boundary.lower_flux).at[-1].set(boundary.upper_flux)
        flux_z = _two_thirds_filter(flux_z, grid=grid)

        shear_production = momentum_diffusivity * diagnostic_magnitude**2
        buoyancy_destruction = (
            scalar_diffusivity
            * scalar_config.stability_buoyancy_coefficient
            * _scalar_cell_gradient(context)[..., 2]
        )
        sgs_tke = jnp.maximum(
            (shear_production - buoyancy_destruction)
            * delta
            / constants.sgs_dissipation_coefficient,
            0.0,
        ) ** (2.0 / 3.0)
        diagnostic_gradient_faces = (
            context.dtheta_dz_on_faces.at[0]
            .set(
                jnp.where(
                    diagnostic_face_diffusivity[0] > 0.0,
                    -flux_z[0] / diagnostic_face_diffusivity[0],
                    0.0,
                )
            )
            .at[-1]
            .set(
                jnp.where(
                    face_diffusivity[-1] > 0.0,
                    -flux_z[-1] / face_diffusivity[-1],
                    0.0,
                )
            )
        )
        gradient_z = 0.5 * (
            diagnostic_gradient_faces[:-1] + diagnostic_gradient_faces[1:]
        )
        flux_z_at_cells = 0.5 * (flux_z[:-1] + flux_z[1:])
        scalar_dissipation = -(
            flux_x * context.dtheta_dx
            + flux_y * context.dtheta_dy
            + flux_z_at_cells * gradient_z
        )
        scalar_length = delta * jnp.sqrt(
            jnp.maximum(effective_scalar_coefficient, 0.0)
        )
        sqrt_tke = jnp.sqrt(jnp.maximum(sgs_tke, 0.0))
        valid = sqrt_tke > jnp.finfo(sqrt_tke.dtype).tiny
        scalar_variance_numerator = (
            2.0
            * scalar_length
            * scalar_dissipation
            / constants.scalar_variance_coefficient
        )
        scalar_variance = jnp.where(
            valid,
            scalar_variance_numerator / jnp.where(valid, sqrt_tke, 1.0),
            0.0,
        )
        scalar_variance = jnp.maximum(scalar_variance, 0.0)
        return LasdDiagnosticFields(
            momentum_diffusivity,
            scalar_diffusivity,
            flux_x,
            flux_y,
            flux_z,
            sgs_tke,
            scalar_variance_numerator,
            scalar_variance,
        )

    def combine_scalar_tendencies(self, tendencies: tuple[Field, ...]) -> Field:
        if not tendencies:
            raise ValueError("at least one scalar tendency is required")
        first = tendencies[0]
        for tendency in tendencies:
            _require_tiny_global(tendency, Cell)
            if (
                tendency.quantity
                not in (PotentialTemperatureTendency, PassiveScalarTendency)
                or tendency.phase is not Evaluated
            ):
                raise TypeError("only evaluated scalar tendencies may be combined")
            if tendency.ownership != first.ownership:
                raise ValueError("combined scalar tendencies must share ownership")
            if tendency.payload.dtype != first.payload.dtype:
                raise TypeError("combined scalar tendencies must share one dtype")
        return Field(
            first.quantity,
            Cell,
            first.ownership,
            Evaluated,
            sum(
                (term.payload for term in tendencies),
                jnp.zeros_like(first.payload),
            ),
        )
