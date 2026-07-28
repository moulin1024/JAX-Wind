"""LASD and scalar-transport methods for the z-slab interpreter."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import jax.numpy as jnp

from jaxwind.interpreters.jax_fringe import plateau_fringe_mask

from jaxwind.domain import (
    Accepted,
    AddressableField,
    Cell,
    Evaluated,
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
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
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


class ZSlabLasdMixin:
    """Distributed closure and scalar-transport interpretation."""

    __slots__ = ()

    def _addressable_closure_field(
        self,
        template: AddressableField,
        quantity: type,
        payload: Any,
    ) -> AddressableField:
        return AddressableField(
            quantity,
            Cell,
            self._expected_regions(Cell),
            Accepted,
            payload.astype(template.payload.dtype),
        )

    def initialize_lasd_closure(
        self,
        fields: BoussinesqFields,
        model: Any,
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
        self._validate_field(scalar, Cell)
        zero = jnp.zeros_like(scalar.payload)
        momentum_coefficient = jnp.full_like(
            scalar.payload,
            momentum_config.initial_coefficient,
        )
        scalar_coefficient = jnp.full_like(
            scalar.payload,
            scalar_config.initial_coefficient,
        )
        field = lambda quantity, payload: self._addressable_closure_field(  # noqa: E731
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
        grid = self.decomposition.grid
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
        )[None, None, None, :]

        def field(
            current: AddressableField,
            target_field: AddressableField,
        ) -> AddressableField:
            if (
                current.quantity is not target_field.quantity
                or current.regions != target_field.regions
            ):
                raise ValueError("main and precursor LASD fields do not align")
            payload = self._relax_lasd_field(
                current.payload,
                target_field.payload,
                blend,
            )
            return self._addressable_closure_field(
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
        old_m = closure.momentum
        old_s = closure.scalar
        interval = momentum_config.update_interval
        should_update = (clock.step + 1) % interval == 0
        if should_update:
            context = self.boussinesq_context(fields)
            trajectory_x, trajectory_y, trajectory_z = self._lasd_accumulate(
                context.momentum.arrays.u,
                context.momentum.arrays.v,
                context.momentum.arrays.w_at_cells,
                old_m.trajectory_x.payload,
                old_m.trajectory_y.payload,
                old_m.trajectory_z.payload,
                interval,
            )
        else:
            velocity = fields.velocity
            self._validate_velocity_cell(velocity.x, XVelocity)
            self._validate_velocity_cell(velocity.y, YVelocity)
            self._validate_field(velocity.z.owned, ZFace)
            if velocity.z.owned.quantity is not VerticalVelocity:
                raise TypeError("LASD trajectory requires vertical velocity")
            trajectory_x, trajectory_y, trajectory_z = (
                self._lasd_accumulate_velocity(
                    velocity.x.payload,
                    velocity.y.payload,
                    velocity.z.owned.payload,
                    velocity.z.lower_boundary,
                    old_m.trajectory_x.payload,
                    old_m.trajectory_y.payload,
                    old_m.trajectory_z.payload,
                    interval,
                )
            )
        field = lambda template, payload: self._addressable_closure_field(  # noqa: E731
            template,
            template.quantity,
            payload,
        )
        if should_update:
            results = self._lasd_update(
                context.momentum.arrays,
                context.arrays,
                old_m.lm.payload,
                old_m.mm.payload,
                old_m.qn.payload,
                old_m.nn.payload,
                old_s.lm.payload,
                old_s.mm.payload,
                old_s.qn.payload,
                old_s.nn.payload,
                trajectory_x,
                trajectory_y,
                trajectory_z,
                clock.step == interval - 1,
                dt * interval,
                momentum_config.filter_grid_ratio,
                momentum_config.test_filter_ratio,
                momentum_config.timescale_coefficient,
                momentum_config.initial_coefficient,
                momentum_config.minimum_coefficient,
                momentum_config.maximum_coefficient,
                momentum_config.scale_dependent,
                scalar_config.initial_coefficient,
                scalar_config.minimum_coefficient,
                scalar_config.maximum_coefficient,
                scalar_config.scale_dependent,
            )
            (
                momentum_coefficient,
                lm,
                mm,
                qn,
                nn,
                scalar_coefficient,
                scalar_lm,
                scalar_mm,
                scalar_qn,
                scalar_nn,
            ) = results
            zero = jnp.zeros_like(trajectory_x)
            new_momentum = MomentumLasdMemory(
                field(old_m.coefficient, momentum_coefficient),
                field(old_m.lm, lm),
                field(old_m.mm, mm),
                field(old_m.qn, qn),
                field(old_m.nn, nn),
                field(old_m.trajectory_x, zero),
                field(old_m.trajectory_y, zero),
                field(old_m.trajectory_z, zero),
            )
            new_scalar = ScalarLasdMemory(
                field(old_s.coefficient, scalar_coefficient),
                field(old_s.lm, scalar_lm),
                field(old_s.mm, scalar_mm),
                field(old_s.qn, scalar_qn),
                field(old_s.nn, scalar_nn),
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
        context: ZSlabBoussinesqContext,
    ) -> ZSlabDryFlowContext:
        return context.momentum

    def _scalar_tendency(
        self,
        context: ZSlabBoussinesqContext,
        payload: Any,
    ) -> AddressableField:
        quantity = (
            PotentialTemperatureTendency
            if context.potential_temperature.quantity
            is PotentialTemperaturePerturbation
            else PassiveScalarTendency
        )
        return AddressableField(
            quantity,
            Cell,
            self._expected_regions(Cell),
            Evaluated,
            payload.astype(context.potential_temperature.payload.dtype),
        )

    def buoyancy_tendency(
        self,
        context: ZSlabBoussinesqContext,
        config: LinearBoussinesqBuoyancy | NoBuoyancy,
    ) -> VelocityVector:
        velocity = context.momentum.velocity
        if isinstance(config, NoBuoyancy):
            return self._dry_tendency(
                jnp.zeros_like(velocity.x.payload),
                jnp.zeros_like(velocity.y.payload),
                jnp.zeros_like(velocity.z.owned.payload),
            )
        if not isinstance(config, LinearBoussinesqBuoyancy):
            raise TypeError("unsupported Boussinesq buoyancy choice")
        z = self._buoyancy(
            context.arrays,
            config.acceleration_per_temperature,
        )
        return self._dry_tendency(
            jnp.zeros_like(velocity.x.payload),
            jnp.zeros_like(velocity.y.payload),
            z,
        )

    def rayleigh_damping_tendency(
        self,
        context: ZSlabBoussinesqContext,
        config: NoRayleighDamping | RayleighGeostrophicDamping,
    ) -> VelocityVector:
        velocity = context.momentum.velocity
        if isinstance(config, NoRayleighDamping):
            return self._dry_tendency(
                jnp.zeros_like(velocity.x.payload),
                jnp.zeros_like(velocity.y.payload),
                jnp.zeros_like(velocity.z.owned.payload),
            )
        if not isinstance(config, RayleighGeostrophicDamping):
            raise TypeError("unsupported Rayleigh damping choice")
        if config.start_height >= self.decomposition.grid.lz:
            raise ValueError("Rayleigh damping must start below the domain top")
        x, y, z = self._rayleigh_damping(
            velocity.x.payload,
            velocity.y.payload,
            velocity.z.owned.payload,
            config.start_height,
            config.maximum_rate,
            config.geostrophic_x_velocity,
            config.geostrophic_y_velocity,
        )
        return self._dry_tendency(x, y, z)

    def scalar_advection_tendency(
        self,
        context: ZSlabBoussinesqContext,
        config: ConservativeScalarAdvection,
    ) -> AddressableField:
        if not isinstance(config, ConservativeScalarAdvection):
            raise TypeError("unsupported conservative scalar advection choice")
        return self._scalar_tendency(
            context,
            self._scalar_advection(context.arrays, context.momentum.arrays),
        )

    def scalar_sgs_tendency(
        self,
        context: ZSlabBoussinesqContext,
        momentum_config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
        config: StaticSmagorinskyScalarFlux | LagrangianScaleDependentScalarFlux,
        boundary: ScalarFluxBoundary = ScalarFluxBoundary(),
    ) -> AddressableField:
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
        if static:
            coefficient = jnp.full_like(
                context.arrays.theta,
                momentum_config.coefficient**2 / config.turbulent_prandtl,
            )
        else:
            closure = context.momentum.closure
            if not isinstance(closure, LasdClosureMemory):
                raise TypeError("scalar LASD requires initialized closure memory")
            coefficient = closure.scalar.coefficient.payload
        return self._scalar_tendency(
            context,
            self._scalar_sgs(
                context.arrays,
                context.momentum.arrays,
                coefficient,
                boundary.lower_flux,
                boundary.upper_flux,
                config.stability_buoyancy_coefficient if dynamic else 0.0,
                config.stability_beta if dynamic else 0.0,
                config.stability_power if dynamic else 1.0,
            ),
        )

    def lasd_diagnostic_fields(
        self,
        context: ZSlabBoussinesqContext,
        momentum_config: LagrangianScaleDependentDynamic,
        scalar_config: LagrangianScaleDependentScalarFlux,
        boundary: ScalarFluxBoundary = ScalarFluxBoundary(),
        constants: DiagnosticLasdConstants = DiagnosticLasdConstants(),
        wall: NeutralLogWall | FilteredNeutralLogWall | None = None,
    ) -> LasdDiagnosticFields:
        """Return owned cell diagnostics and owned upper-face scalar flux."""
        if not isinstance(
            momentum_config, LagrangianScaleDependentDynamic
        ) or not isinstance(
            scalar_config,
            LagrangianScaleDependentScalarFlux,
        ):
            raise TypeError("LASD diagnostics require momentum and scalar LASD")
        if not isinstance(constants, DiagnosticLasdConstants):
            raise TypeError("unsupported LASD diagnostic constants")
        wall_gradient_factor = 0.0
        if wall is not None:
            if not isinstance(
                wall,
                (NeutralLogWall, FilteredNeutralLogWall),
            ):
                raise TypeError("LASD diagnostic wall must be NeutralLogWall")
            reference_height = 0.5 * self.decomposition.grid.dz
            if wall.roughness_length >= reference_height:
                raise ValueError("wall roughness must be below the first cell centre")
            wall_gradient_factor = 1.0 / (
                math.log(reference_height / wall.roughness_length) * reference_height
            )
        closure = context.momentum.closure
        if not isinstance(closure, LasdClosureMemory):
            raise TypeError("LASD diagnostics require initialized closure memory")
        values = self._lasd_diagnostics(
            context.arrays,
            context.momentum.arrays,
            closure.momentum.coefficient.payload,
            closure.scalar.coefficient.payload,
            boundary.lower_flux,
            boundary.upper_flux,
            constants.sgs_dissipation_coefficient,
            constants.scalar_variance_coefficient,
            wall_gradient_factor,
            constants.horizontal_homogeneous_wall,
            scalar_config.stability_buoyancy_coefficient,
            scalar_config.stability_beta,
            scalar_config.stability_power,
        )
        return LasdDiagnosticFields(*values)

    def combine_scalar_tendencies(
        self,
        tendencies: tuple[AddressableField, ...],
    ) -> AddressableField:
        if not tendencies:
            raise ValueError("at least one scalar tendency is required")
        first = tendencies[0]
        for tendency in tendencies:
            self._validate_field(tendency, Cell)
            if (
                tendency.quantity
                not in (PotentialTemperatureTendency, PassiveScalarTendency)
                or tendency.phase is not Evaluated
            ):
                raise TypeError("only evaluated scalar tendencies may be combined")
            if tendency.payload.dtype != first.payload.dtype:
                raise TypeError("combined scalar tendencies must preserve dtype")
        return AddressableField(
            first.quantity,
            Cell,
            self._expected_regions(Cell),
            Evaluated,
            self._combine_payloads(
                tuple(term.payload for term in tendencies)
            ),
        )
