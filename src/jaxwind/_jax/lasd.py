"""LASD and scalar-transport methods for the private JAX discretization."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING, Any

from jax import lax
import jax.numpy as jnp

from jaxwind._jax.fringe import plateau_fringe_mask

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
    PassiveScalarConcentration,
    PassiveScalarTendency,
    PotentialTemperaturePerturbation,
    PotentialTemperatureTendency,
    Projected,
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
    CoriolisGeostrophic,
    FilteredNeutralLogWall,
    KinematicPressureGradient,
    NeutralLogWall,
    NoRotation,
    RotationalAdvection,
    StaticSmagorinsky,
)
from jaxwind.physics.boussinesq import (
    BoussinesqFields,
    BoussinesqModel,
    BoussinesqTendency,
    ConservativeScalarAdvection,
    LinearBoussinesqBuoyancy,
    NoBuoyancy,
    NoRayleighDamping,
    RayleighGeostrophicDamping,
    ScalarFluxBoundary,
    StaticSmagorinskyScalarFlux,
)
from jaxwind.physics.surface_transfer import (
    MoninObukhovSurfaceTransfer,
    NoSurfaceTransfer,
    SurfaceTransferResult,
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

from .scalar_interpreter import ZSlabScalarMixin


if TYPE_CHECKING:
    from .discretization import ZSlabBoussinesqContext, ZSlabDryFlowContext


class ZSlabLasdMixin(ZSlabScalarMixin):
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
            -jnp.asarray(dt, dtype) * mask / jnp.asarray(fringe.relaxation_time, dtype)
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
            payload = self.lasd.relax_field(
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
                *(
                    field(left, right)
                    for left, right in zip(
                        current_m.fields(),
                        target_m.fields(),
                        strict=True,
                    )
                )
            ),
            ScalarLasdMemory(
                *(
                    field(left, right)
                    for left, right in zip(
                        current_s.fields(),
                        target_s.fields(),
                        strict=True,
                    )
                )
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
        prepared, diagnostic, _context = self._prepare_lasd_closure(
            fields,
            model,
            clock,
            dt,
            reuse_momentum_context=False,
            retain_transport_context=False,
        )
        return prepared, diagnostic

    def prepare_lasd_closure_with_context(
        self,
        fields: BoussinesqFields,
        model: Any,
        clock: Any,
        dt: float,
    ) -> tuple[
        BoussinesqFields,
        LasdClosureEventDiagnostic,
        ZSlabDryFlowContext,
    ]:
        prepared, diagnostic, context = self._prepare_lasd_closure(
            fields,
            model,
            clock,
            dt,
            reuse_momentum_context=True,
            retain_transport_context=True,
        )
        if context is None:  # pragma: no cover - protected by the flag above
            raise RuntimeError("LASD momentum context was not retained")
        return prepared, diagnostic, context

    def prepare_lasd_closure_reusing_update_context(
        self,
        fields: BoussinesqFields,
        model: Any,
        clock: Any,
        dt: float,
    ) -> tuple[
        BoussinesqFields,
        LasdClosureEventDiagnostic,
        ZSlabDryFlowContext | None,
    ]:
        """Retain the derivative context only when this step updates LASD."""

        return self._prepare_lasd_closure(
            fields,
            model,
            clock,
            dt,
            reuse_momentum_context=True,
            retain_transport_context=False,
        )

    def _prepare_lasd_closure(
        self,
        fields: BoussinesqFields,
        model: Any,
        clock: Any,
        dt: float,
        *,
        reuse_momentum_context: bool,
        retain_transport_context: bool,
    ) -> tuple[
        BoussinesqFields,
        LasdClosureEventDiagnostic,
        ZSlabDryFlowContext | None,
    ]:
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
        retain_context = reuse_momentum_context and (
            retain_transport_context
            or not isinstance(should_update, bool)
            or should_update
        )
        retained_momentum = (
            self.dry_flow_context(fields.velocity) if retain_context else None
        )

        def momentum_context():
            return (
                retained_momentum
                if retained_momentum is not None
                else self.dry_flow_context(fields.velocity)
            )

        def accumulate_for_update(_operand):
            context = momentum_context()
            return self.lasd.accumulate(
                context.arrays.u,
                context.arrays.v,
                context.arrays.w_at_cells,
                old_m.trajectory_x.payload,
                old_m.trajectory_y.payload,
                old_m.trajectory_z.payload,
                interval,
            )

        def accumulate_for_transport(_operand):
            if retained_momentum is not None:
                return self.lasd.accumulate(
                    retained_momentum.arrays.u,
                    retained_momentum.arrays.v,
                    retained_momentum.arrays.w_at_cells,
                    old_m.trajectory_x.payload,
                    old_m.trajectory_y.payload,
                    old_m.trajectory_z.payload,
                    interval,
                )
            velocity = fields.velocity
            self._validate_velocity_cell(velocity.x, XVelocity)
            self._validate_velocity_cell(velocity.y, YVelocity)
            self._validate_field(velocity.z.owned, ZFace)
            if velocity.z.owned.quantity is not VerticalVelocity:
                raise TypeError("LASD trajectory requires vertical velocity")
            return self.lasd.accumulate_velocity(
                velocity.x.payload,
                velocity.y.payload,
                velocity.z.owned.payload,
                velocity.z.lower_boundary,
                old_m.trajectory_x.payload,
                old_m.trajectory_y.payload,
                old_m.trajectory_z.payload,
                interval,
            )

        if isinstance(should_update, bool):
            trajectories = (
                accumulate_for_update(None)
                if should_update
                else accumulate_for_transport(None)
            )
        else:
            trajectories = lax.cond(
                should_update,
                accumulate_for_update,
                accumulate_for_transport,
                operand=None,
            )
        trajectory_x, trajectory_y, trajectory_z = trajectories

        field = lambda template, payload: self._addressable_closure_field(  # noqa: E731
            template,
            template.quantity,
            payload,
        )

        def update_payloads(trajectories):
            trajectory_x, trajectory_y, trajectory_z = trajectories
            momentum = momentum_context()
            common_arguments = (
                old_m.lm.payload,
                old_m.mm.payload,
                old_m.qn.payload,
                old_m.nn.payload,
            )
            update_arguments = (
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
            )
            if scalar_config.dynamic_updates_enabled:
                context = self.boussinesq_context_from_momentum(fields, momentum)
                results = self.lasd.update(
                    context.momentum.arrays,
                    context.arrays,
                    *common_arguments,
                    old_s.lm.payload,
                    old_s.mm.payload,
                    old_s.qn.payload,
                    old_s.nn.payload,
                    *update_arguments,
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
            else:
                momentum_coefficient, lm, mm, qn, nn = (
                    self.lasd.update_momentum(
                        momentum.arrays,
                        *common_arguments,
                        *update_arguments,
                    )
                )
                scalar_coefficient = old_s.coefficient.payload
                scalar_lm = old_s.lm.payload
                scalar_mm = old_s.mm.payload
                scalar_qn = old_s.qn.payload
                scalar_nn = old_s.nn.payload
            zero = jnp.zeros_like(trajectory_x)
            return (
                momentum_coefficient,
                lm,
                mm,
                qn,
                nn,
                zero,
                zero,
                zero,
                scalar_coefficient,
                scalar_lm,
                scalar_mm,
                scalar_qn,
                scalar_nn,
            )

        def carry_payloads(trajectories):
            trajectory_x, trajectory_y, trajectory_z = trajectories
            return (
                old_m.coefficient.payload,
                old_m.lm.payload,
                old_m.mm.payload,
                old_m.qn.payload,
                old_m.nn.payload,
                trajectory_x,
                trajectory_y,
                trajectory_z,
                old_s.coefficient.payload,
                old_s.lm.payload,
                old_s.mm.payload,
                old_s.qn.payload,
                old_s.nn.payload,
            )

        if isinstance(should_update, bool):
            payloads = (
                update_payloads(trajectories)
                if should_update
                else carry_payloads(trajectories)
            )
        else:
            payloads = lax.cond(
                should_update,
                update_payloads,
                carry_payloads,
                trajectories,
            )
        (
            momentum_coefficient,
            lm,
            mm,
            qn,
            nn,
            trajectory_x,
            trajectory_y,
            trajectory_z,
            scalar_coefficient,
            scalar_lm,
            scalar_mm,
            scalar_qn,
            scalar_nn,
        ) = payloads
        new_momentum = MomentumLasdMemory(
            field(old_m.coefficient, momentum_coefficient),
            field(old_m.lm, lm),
            field(old_m.mm, mm),
            field(old_m.qn, qn),
            field(old_m.nn, nn),
            field(old_m.trajectory_x, trajectory_x),
            field(old_m.trajectory_y, trajectory_y),
            field(old_m.trajectory_z, trajectory_z),
        )
        new_scalar = ScalarLasdMemory(
            field(old_s.coefficient, scalar_coefficient),
            field(old_s.lm, scalar_lm),
            field(old_s.mm, scalar_mm),
            field(old_s.qn, scalar_qn),
            field(old_s.nn, scalar_nn),
        )
        prepared = replace(
            fields,
            closure=LasdClosureMemory(new_momentum, new_scalar, fingerprint),
        )
        return (
            prepared,
            LasdClosureEventDiagnostic(should_update, clock.step, interval),
            retained_momentum,
        )

    def momentum_context(
        self,
        context: ZSlabBoussinesqContext,
    ) -> ZSlabDryFlowContext:
        return context.momentum

    def surface_transfer(
        self,
        fields: BoussinesqFields,
        model: BoussinesqModel,
        clock: Any,
    ) -> SurfaceTransferResult | None:
        """Evaluate a coupled lower-boundary exchange law from plane means."""

        config = model.surface_transfer
        if isinstance(config, NoSurfaceTransfer):
            return None
        if not isinstance(config, MoninObukhovSurfaceTransfer):
            raise TypeError("unsupported Boussinesq surface-transfer choice")
        wall = model.momentum.wall
        if not isinstance(wall, (NeutralLogWall, FilteredNeutralLogWall)):
            raise TypeError("Monin-Obukhov transfer requires a logarithmic wall")
        grid = self.decomposition.grid
        measurement_height = 0.5 * grid.dz
        if measurement_height <= max(
            wall.roughness_length,
            config.scalar_roughness_length,
        ):
            raise ValueError(
                "surface-transfer height must exceed both roughness lengths"
            )

        buoyancy_coefficient = (
            model.buoyancy.acceleration_per_temperature
            if isinstance(model.buoyancy, LinearBoussinesqBuoyancy)
            else 0.0
        )
        transfer = self.scalar.surface_transfer(
            fields.velocity.x.payload,
            fields.velocity.y.payload,
            fields.potential_temperature.payload,
            clock.time,
            grid.dz,
            wall.roughness_length,
            config.scalar_roughness_length,
            config.surface_scalar_initial,
            config.surface_scalar_rate,
            config.x_velocity_offset,
            config.y_velocity_offset,
            buoyancy_coefficient,
            wall.von_karman,
            config.positive_zeta_momentum_slope,
            config.positive_zeta_scalar_slope,
            config.negative_zeta_momentum_coefficient,
            config.negative_zeta_scalar_coefficient,
            config.relaxation,
            config.maximum_abs_zeta,
            config.iterations,
        )
        # The mapped law is replicated on every local device.  Collapse that
        # implementation axis before returning to the semantic physics layer.
        return SurfaceTransferResult(*(value[0] for value in transfer))

    def fused_boussinesq_tendency(
        self,
        fields: BoussinesqFields,
        model: BoussinesqModel,
        *,
        wall_acceleration: tuple[Any, Any] | None = None,
        scalar_surface_source: Any | None = None,
        execution_time: Any = 0.0,
        momentum_context: ZSlabDryFlowContext | None = None,
    ) -> BoussinesqTendency:
        """Evaluate the supported Boussinesq model through the mandatory fused RHS."""
        if (wall_acceleration is None) != (scalar_surface_source is None):
            raise ValueError(
                "imposed wall acceleration and scalar source must be supplied together"
            )
        use_imposed_sources = wall_acceleration is not None
        surface_config = model.surface_transfer
        coupled_surface = isinstance(
            surface_config,
            MoninObukhovSurfaceTransfer,
        )
        if coupled_surface and use_imposed_sources:
            raise ValueError(
                "explicit surface sources cannot override coupled surface transfer"
            )
        source_mode = 2 if coupled_surface else int(use_imposed_sources)
        if wall_acceleration is not None and len(wall_acceleration) != 2:
            raise ValueError("wall acceleration must contain x and y components")
        momentum_model = model.momentum
        wall = momentum_model.wall
        common = (
            isinstance(momentum_model.advection, RotationalAdvection)
            and isinstance(momentum_model.pressure_gradient, KinematicPressureGradient)
            and isinstance(wall, (NeutralLogWall, FilteredNeutralLogWall))
            and isinstance(
                momentum_model.rotation,
                (NoRotation, CoriolisGeostrophic),
            )
            and isinstance(model.scalar_advection, ConservativeScalarAdvection)
            and isinstance(
                model.buoyancy,
                (NoBuoyancy, LinearBoussinesqBuoyancy),
            )
            and isinstance(model.rayleigh_damping, NoRayleighDamping)
            and isinstance(model.scalar_boundary, ScalarFluxBoundary)
        )
        sgs = momentum_model.sgs
        lasd = isinstance(sgs, LagrangianScaleDependentDynamic) and isinstance(
            model.scalar_sgs, LagrangianScaleDependentScalarFlux
        )
        static = isinstance(sgs, StaticSmagorinsky) and isinstance(
            model.scalar_sgs, StaticSmagorinskyScalarFlux
        )
        frozen_model = self.frozen_zero_scalar and (lasd or static)
        if not common or not (lasd or static):
            raise TypeError(
                "Boussinesq solve requires the fused legacy SGS model with "
                "a neutral log wall, supported rotation, scalar flux boundary, "
                "and no standalone Rayleigh damping"
            )
        if source_mode and frozen_model:
            raise ValueError("coupled surface sources require an active scalar")
        if frozen_model and isinstance(model.buoyancy, LinearBoussinesqBuoyancy):
            raise ValueError("Boussinesq buoyancy requires an active scalar")

        velocity = fields.velocity
        self._validate_velocity_cell(velocity.x, XVelocity)
        self._validate_velocity_cell(velocity.y, YVelocity)
        self._validate_field(velocity.z.owned, ZFace)
        if velocity.z.owned.quantity is not VerticalVelocity:
            raise TypeError("dry-flow vertical velocity requires VerticalVelocity")
        if not (
            velocity.x.phase is Projected
            and velocity.y.phase is Projected
            and velocity.z.owned.phase is Projected
        ):
            raise TypeError("dry-flow context requires projected velocity")
        scalar = fields.potential_temperature
        self._validate_field(scalar, Cell)
        if scalar.quantity not in (
            PotentialTemperaturePerturbation,
            PassiveScalarConcentration,
        ):
            raise TypeError("Boussinesq context requires a supported scalar quantity")
        if scalar.phase is not Accepted:
            raise TypeError("Boussinesq context requires accepted scalar state")
        if frozen_model and (
            scalar.quantity is not PassiveScalarConcentration
            or model.scalar_boundary.lower_flux != 0.0
            or model.scalar_boundary.upper_flux != 0.0
        ):
            raise ValueError(
                "frozen zero scalar fusion requires a passive scalar and zero fluxes"
            )

        reference_height = 0.5 * self.decomposition.grid.dz
        if wall.roughness_length >= reference_height:
            raise ValueError("wall roughness must be below the first cell centre")
        drag = (
            wall.von_karman / math.log(reference_height / wall.roughness_length)
        ) ** 2
        filtered = isinstance(wall, FilteredNeutralLogWall)
        wall_filter_width = (
            wall.filter_grid_ratio * wall.test_filter_ratio if filtered else 1.0
        )
        wall_gradient_factor = self._diagnostic_wall_gradient_factor(wall)
        rotation = momentum_model.rotation
        rotation_arguments = (
            (
                rotation.coriolis_parameter,
                rotation.geostrophic_x_velocity,
                rotation.geostrophic_y_velocity,
                rotation.horizontal_coriolis_parameter,
            )
            if isinstance(rotation, CoriolisGeostrophic)
            else (0.0, 0.0, 0.0, 0.0)
        )
        if momentum_context is None:
            fused_kernel = self.flow.fused_boussinesq
            common_arguments = (
                velocity.x.payload,
                velocity.y.payload,
                velocity.z.owned.payload,
                velocity.z.lower_boundary,
                scalar.payload,
            )
        else:
            if momentum_context.velocity is not velocity:
                raise ValueError(
                    "reused momentum context does not belong to the evaluated velocity"
                )
            fused_kernel = self.flow.fused_boussinesq_from_context
            common_arguments = (momentum_context.arrays, scalar.payload)
        forcing_arguments = (
            momentum_model.pressure_gradient.x_acceleration,
            momentum_model.pressure_gradient.y_acceleration,
            *rotation_arguments,
            drag,
            filtered,
            wall_filter_width,
            wall_gradient_factor,
        )
        closure = fields.closure
        if lasd:
            if not isinstance(closure, LasdClosureMemory):
                raise TypeError("LASD fusion requires initialized closure memory")
            momentum_coefficient = closure.momentum.coefficient.payload
            scalar_coefficient = closure.scalar.coefficient.payload
            momentum_bounds = (sgs.minimum_coefficient, sgs.maximum_coefficient)
            scalar_bounds = (
                model.scalar_sgs.minimum_coefficient,
                model.scalar_sgs.maximum_coefficient,
            )
            stability = (
                model.scalar_sgs.stability_buoyancy_coefficient,
                model.scalar_sgs.stability_beta,
                model.scalar_sgs.stability_power,
            )
        else:
            momentum_coefficient = jnp.full_like(
                scalar.payload,
                sgs.coefficient**2,
            )
            scalar_coefficient = jnp.full_like(
                scalar.payload,
                sgs.coefficient**2 / model.scalar_sgs.turbulent_prandtl,
            )
            momentum_bounds = (0.0, math.inf)
            scalar_bounds = (0.0, math.inf)
            stability = (0.0, 0.0, 1.0)
        zero_surface = jnp.zeros(
            (len(self.addressable_partitions),),
            dtype=velocity.x.payload.dtype,
        )

        def device_values(value):
            values = jnp.asarray(value, dtype=velocity.x.payload.dtype)
            if values.ndim == 0:
                return jnp.broadcast_to(values, zero_surface.shape)
            if values.shape != zero_surface.shape:
                raise ValueError("surface source must be scalar or device-replicated")
            return values

        imposed_wall = tuple(
            device_values(value)
            for value in (wall_acceleration or (zero_surface, zero_surface))
        )
        imposed_scalar = device_values(
            scalar_surface_source
            if scalar_surface_source is not None
            else zero_surface
        )
        surface_arguments = (
            (
                surface_config.scalar_roughness_length,
                surface_config.surface_scalar_initial,
                surface_config.surface_scalar_rate,
                surface_config.x_velocity_offset,
                surface_config.y_velocity_offset,
                surface_config.positive_zeta_momentum_slope,
                surface_config.positive_zeta_scalar_slope,
                surface_config.negative_zeta_momentum_coefficient,
                surface_config.negative_zeta_scalar_coefficient,
                surface_config.relaxation,
                surface_config.maximum_abs_zeta,
                surface_config.iterations,
            )
            if coupled_surface
            else (
                wall.roughness_length,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1,
            )
        )
        x, y, z, scalar_payload = fused_kernel(
            *common_arguments,
            momentum_coefficient,
            scalar_coefficient,
            *forcing_arguments,
            *momentum_bounds,
            *scalar_bounds,
            model.scalar_boundary.lower_flux,
            model.scalar_boundary.upper_flux,
            *stability,
            *imposed_wall,
            imposed_scalar,
            (
                model.buoyancy.acceleration_per_temperature
                if isinstance(model.buoyancy, LinearBoussinesqBuoyancy)
                else 0.0
            ),
            execution_time,
            wall.roughness_length,
            *surface_arguments[:5],
            wall.von_karman,
            *surface_arguments[5:],
            source_mode,
        )
        scalar_quantity = (
            PotentialTemperatureTendency
            if scalar.quantity is PotentialTemperaturePerturbation
            else PassiveScalarTendency
        )
        return BoussinesqTendency(
            self._dry_tendency(x, y, z),
            AddressableField(
                scalar_quantity,
                Cell,
                self._expected_regions(Cell),
                Evaluated,
                scalar_payload.astype(scalar.payload.dtype),
            ),
        )
