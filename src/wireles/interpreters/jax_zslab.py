"""JAX-native equal-z-slab interpretation with transient ppermute halos."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import math
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from wireles.interpreters.jax_actuator_disk import (
    filtered_disk_velocity_correction,
    gaussian_convolved_annulus,
)

from wireles.domain import (
    Accepted,
    AddressableField,
    Cell,
    Candidate,
    Divergence,
    EqualZSlab,
    Evaluated,
    LasdTrajectoryXVelocity,
    LasdTrajectoryYVelocity,
    LasdTrajectoryZVelocity,
    MomentumLasdCoefficient,
    MomentumLasdLm,
    MomentumLasdMm,
    MomentumLasdNn,
    MomentumLasdQn,
    PressureCorrection,
    PressureRhs,
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
    VerticalBoundary,
    VerticalPressureGradient,
    VerticalVelocity,
    VerticalVelocityTendency,
    XPressureGradient,
    XVelocity,
    XVelocityTendency,
    YPressureGradient,
    YVelocity,
    YVelocityTendency,
    ZFace,
)
from wireles.operators import PressureGradient, VelocityVector
from wireles.physics.dry_flow import (
    ConservativeAdvection,
    CoriolisGeostrophic,
    FilteredNeutralLogWall,
    KinematicPressureGradient,
    NeutralLogWall,
    NoRotation,
    StaticSmagorinsky,
)
from wireles.physics.boussinesq import (
    BoussinesqFields,
    ConservativeScalarAdvection,
    LinearBoussinesqBuoyancy,
    NoBuoyancy,
    NoRayleighDamping,
    RayleighGeostrophicDamping,
    ScalarFluxBoundary,
    StaticSmagorinskyScalarFlux,
)
from wireles.physics.lasd import (
    DiagnosticLasdConstants,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LasdClosureEventDiagnostic,
    LasdClosureMemory,
    LasdDiagnosticFields,
    MomentumLasdMemory,
    ScalarLasdMemory,
)
from wireles.physics.wind_tunnel import (
    ConcurrentPrecursorEnvironment,
    ConcurrentPrecursorFringe,
    NoActuatorDisk,
    NoFringe,
    PureThrustActuatorDisk,
    WindTunnelModel,
)


class PackedHaloArrays(NamedTuple):
    """Transient packed neighbor planes returned inside the SPMD program."""

    lower: Any
    upper: Any
    lower_is_physical: Any
    upper_is_physical: Any


@dataclass(frozen=True, slots=True)
class ZFaceFieldContext:
    """Owned upper faces plus the separately constructed lower boundary face."""

    owned: AddressableField
    lower_boundary: Any

    def extract_owned(self) -> AddressableField:
        return self.owned


class ZSlabDryFlowArrays(NamedTuple):
    """Per-shard velocity, interpolation, and shared gradient arrays."""

    u: Any
    v: Any
    w_upper: Any
    w_lower: Any
    u_upper: Any
    v_upper: Any
    u_lower: Any
    v_lower: Any
    w_at_cells: Any
    w_next_cell: Any
    dudx: Any
    dudy: Any
    dudz_at_cells: Any
    dvdx: Any
    dvdy: Any
    dvdz_at_cells: Any
    dwdx_at_cells: Any
    dwdy_at_cells: Any
    dwdz: Any
    dudz_upper: Any
    dvdz_upper: Any
    dwdx_upper: Any
    dwdy_upper: Any
    dudx_upper: Any
    dudy_upper: Any
    dvdx_upper: Any
    dvdy_upper: Any
    dwdz_upper: Any
    upper_is_physical: Any


@dataclass(frozen=True, slots=True)
class ZSlabDryFlowContext:
    """Typed outer context whose arrays remain addressable-only."""

    velocity: VelocityVector
    arrays: ZSlabDryFlowArrays
    closure: Any = None


class ZSlabScalarArrays(NamedTuple):
    theta: Any
    theta_upper: Any
    theta_lower: Any
    dtheta_dx: Any
    dtheta_dy: Any
    dtheta_dz_at_cells: Any
    dtheta_dz_upper: Any
    upper_is_physical: Any


@dataclass(frozen=True, slots=True)
class ZSlabBoussinesqContext:
    momentum: ZSlabDryFlowContext
    potential_temperature: AddressableField
    arrays: ZSlabScalarArrays


@dataclass(frozen=True, slots=True)
class JaxZSlabInterpreter:
    """Higher-order JAX interpretation of the first equal z-slab topology."""

    decomposition: EqualZSlab
    addressable_shards: tuple[int, ...]
    exchange_packed: Callable
    _pressure_gradient: Callable
    _divergence: Callable
    _enforce_upper_boundary: Callable
    _horizontal_divergence: Callable
    _horizontal_gradient: Callable
    _filter_horizontal: Callable
    _filter_boundary: Callable
    _correct: Callable
    _ab2_update: Callable
    _wind_tunnel: Callable
    _dry_flow_context: Callable
    _dry_advection: Callable
    _dry_wall: Callable
    _dry_sgs: Callable
    _dry_sgs_vertical_flux: Callable
    _lasd_accumulate: Callable
    _lasd_update: Callable
    _lasd_diagnostics: Callable
    _scalar_context: Callable
    _scalar_advection: Callable
    _scalar_sgs: Callable
    _buoyancy: Callable
    _rayleigh_damping: Callable

    def halo_context_elements_per_shard(self, component_count: int) -> int:
        """Stored lower and upper context planes for one addressable shard."""
        if component_count <= 0:
            raise ValueError("packed halo component count must be positive")
        return (
            2
            * component_count
            * self.decomposition.grid.ny
            * self.decomposition.grid.nx
        )

    def halo_communicated_elements(
        self,
        component_count: int,
        shard_index: int,
    ) -> int:
        """Network payload derived from the actual non-physical neighbors."""
        if component_count <= 0:
            raise ValueError("packed halo component count must be positive")
        if not 0 <= shard_index < self.decomposition.shard_count:
            raise ValueError("shard index is outside the global z mesh")
        neighbors = int(shard_index > 0) + int(
            shard_index < self.decomposition.shard_count - 1
        )
        plane = self.decomposition.grid.ny * self.decomposition.grid.nx
        return neighbors * component_count * plane

    def _expected_regions(self, location: type) -> tuple:
        all_regions = self.decomposition.regions(location)
        return tuple(all_regions[index] for index in self.addressable_shards)

    def _validate_field(self, field: AddressableField, location: type) -> None:
        if field.location is not location:
            raise TypeError(f"z-slab operator requires {location.__name__} input")
        if field.regions != self._expected_regions(location):
            raise ValueError("addressable regions do not match interpreter ownership")

    def pressure_gradient_z(
        self,
        pressure: AddressableField,
        boundary_gradient: VerticalBoundary[Any],
    ) -> ZFaceFieldContext:
        """Apply the stored-upper-face interpretation of ``G_z``."""
        self._validate_field(pressure, Cell)
        if pressure.quantity is not PressureCorrection:
            raise TypeError("pressure_gradient_z requires PressureCorrection")
        payload = self._pressure_gradient(pressure.payload, boundary_gradient.upper)
        owned = AddressableField(
            VerticalPressureGradient,
            ZFace,
            self._expected_regions(ZFace),
            Evaluated,
            payload,
        )
        return ZFaceFieldContext(owned, boundary_gradient.lower)

    def divergence_z(self, vertical_faces: ZFaceFieldContext) -> AddressableField:
        """Apply ``D_z`` using the owned upper faces and lower boundary face."""
        self._validate_field(vertical_faces.owned, ZFace)
        if vertical_faces.owned.quantity not in (
            VerticalVelocity,
            VerticalPressureGradient,
        ):
            raise TypeError("divergence_z requires a vertical face-normal quantity")
        payload = self._divergence(
            vertical_faces.owned.payload,
            vertical_faces.lower_boundary,
        )
        return AddressableField(
            Divergence,
            Cell,
            self._expected_regions(Cell),
            Evaluated,
            payload,
        )

    def _validate_velocity_cell(
        self,
        field: AddressableField,
        quantity: type,
    ) -> None:
        self._validate_field(field, Cell)
        if field.quantity is not quantity:
            raise TypeError(f"velocity component requires {quantity.__name__}")
        if field.phase not in (Candidate, Projected):
            raise TypeError("velocity component must be Candidate or Projected")

    def enforce_normal_boundary(
        self,
        velocity: VelocityVector,
        boundary: VerticalBoundary[Any],
    ) -> VelocityVector:
        """Reconstruct physical normal faces before forming the Poisson RHS."""
        self._validate_velocity_cell(velocity.x, XVelocity)
        self._validate_velocity_cell(velocity.y, YVelocity)
        self._validate_field(velocity.z.owned, ZFace)
        if velocity.z.owned.quantity is not VerticalVelocity:
            raise TypeError("vertical velocity requires VerticalVelocity")
        if velocity.z.owned.phase not in (Candidate, Projected):
            raise TypeError("vertical velocity must be Candidate or Projected")
        x_payload = self._filter_horizontal(velocity.x.payload)
        y_payload = self._filter_horizontal(velocity.y.payload)
        z_payload = self._filter_horizontal(velocity.z.owned.payload)
        dtype_probe = velocity.z.owned.payload[0, 0, 0, 0]
        upper_payload = self._enforce_upper_boundary(
            z_payload,
            self._filter_boundary(boundary.upper, dtype_probe),
        )
        owned = AddressableField(
            VerticalVelocity,
            ZFace,
            self._expected_regions(ZFace),
            velocity.z.owned.phase,
            upper_payload,
        )
        return VelocityVector(
            AddressableField(
                XVelocity,
                Cell,
                self._expected_regions(Cell),
                velocity.x.phase,
                x_payload,
            ),
            AddressableField(
                YVelocity,
                Cell,
                self._expected_regions(Cell),
                velocity.y.phase,
                y_payload,
            ),
            ZFaceFieldContext(
                owned,
                self._filter_boundary(boundary.lower, dtype_probe),
            ),
        )

    def _dry_tendency(
        self,
        x,
        y,
        z,
        lower=None,
    ) -> VelocityVector:
        dtype = z.dtype
        if lower is None:
            lower = jnp.zeros(
                (self.decomposition.grid.ny, self.decomposition.grid.nx),
                dtype=dtype,
            )
        else:
            lower = jnp.asarray(lower, dtype=dtype)
        return VelocityVector(
            AddressableField(
                XVelocityTendency,
                Cell,
                self._expected_regions(Cell),
                Evaluated,
                x,
            ),
            AddressableField(
                YVelocityTendency,
                Cell,
                self._expected_regions(Cell),
                Evaluated,
                y,
            ),
            ZFaceFieldContext(
                AddressableField(
                    VerticalVelocityTendency,
                    ZFace,
                    self._expected_regions(ZFace),
                    Evaluated,
                    z.astype(dtype),
                ),
                lower,
            ),
        )

    def dry_flow_context(self, velocity: VelocityVector) -> ZSlabDryFlowContext:
        """Perform the sole packed velocity halo exchange for all dry terms."""
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
        arrays = self._dry_flow_context(
            velocity.x.payload,
            velocity.y.payload,
            velocity.z.owned.payload,
            velocity.z.lower_boundary,
        )
        return ZSlabDryFlowContext(velocity, arrays)

    def boussinesq_context(
        self,
        fields: BoussinesqFields,
    ) -> ZSlabBoussinesqContext:
        momentum = self.dry_flow_context(fields.velocity)
        scalar = fields.potential_temperature
        self._validate_field(scalar, Cell)
        if scalar.quantity not in (
            PotentialTemperaturePerturbation,
            PassiveScalarConcentration,
        ):
            raise TypeError("Boussinesq context requires a supported scalar quantity")
        if scalar.phase is not Accepted:
            raise TypeError("Boussinesq context requires accepted scalar state")
        momentum = ZSlabDryFlowContext(
            momentum.velocity, momentum.arrays, fields.closure
        )
        return ZSlabBoussinesqContext(
            momentum,
            scalar,
            self._scalar_context(scalar.payload),
        )

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
        old_m = closure.momentum
        old_s = closure.scalar
        interval = momentum_config.update_interval
        trajectory_x, trajectory_y, trajectory_z = self._lasd_accumulate(
            context.momentum.arrays.u,
            context.momentum.arrays.v,
            context.momentum.arrays.w_at_cells,
            old_m.trajectory_x.payload,
            old_m.trajectory_y.payload,
            old_m.trajectory_z.payload,
            interval,
        )
        should_update = (clock.step + 1) % interval == 0
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
            sum(
                (term.payload for term in tendencies),
                jnp.zeros_like(first.payload),
            ),
        )

    def advection_tendency(
        self,
        context: ZSlabDryFlowContext,
        config: ConservativeAdvection,
    ) -> VelocityVector:
        if not isinstance(config, ConservativeAdvection):
            raise TypeError("unsupported z-slab advection choice")
        x, y, z = self._dry_advection(context.arrays)
        return self._dry_tendency(x, y, z)

    def pressure_gradient_tendency(
        self,
        context: ZSlabDryFlowContext,
        config: KinematicPressureGradient,
    ) -> VelocityVector:
        if not isinstance(config, KinematicPressureGradient):
            raise TypeError("unsupported pressure-gradient forcing choice")
        velocity = context.velocity
        x = jnp.full_like(velocity.x.payload, config.x_acceleration)
        y = jnp.full_like(velocity.y.payload, config.y_acceleration)
        z = jnp.zeros_like(velocity.z.owned.payload)
        return self._dry_tendency(x, y, z)

    def wall_stress_tendency(
        self,
        context: ZSlabDryFlowContext,
        config: NeutralLogWall | FilteredNeutralLogWall,
    ) -> VelocityVector:
        if not isinstance(
            config,
            (NeutralLogWall, FilteredNeutralLogWall),
        ):
            raise TypeError("unsupported wall-stress choice")
        reference_height = 0.5 * self.decomposition.grid.dz
        if config.roughness_length >= reference_height:
            raise ValueError("wall roughness must be below the first cell centre")
        drag = (
            config.von_karman / math.log(reference_height / config.roughness_length)
        ) ** 2
        filtered = isinstance(config, FilteredNeutralLogWall)
        filter_width = (
            config.filter_grid_ratio * config.test_filter_ratio
            if filtered
            else 1.0
        )
        x, y, z = self._dry_wall(
            context.arrays,
            drag,
            filtered,
            filter_width,
        )
        return self._dry_tendency(x, y, z)

    def sgs_tendency(
        self,
        context: ZSlabDryFlowContext,
        config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
    ) -> VelocityVector:
        if not isinstance(config, (StaticSmagorinsky, LagrangianScaleDependentDynamic)):
            raise TypeError("unsupported SGS choice")
        coefficient = self._momentum_sgs_coefficient(context, config)
        x, y, z = self._dry_sgs(context.arrays, coefficient)
        return self._dry_tendency(x, y, z)

    def sgs_vertical_flux(
        self,
        context: ZSlabDryFlowContext,
        config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
    ) -> tuple[Any, Any]:
        """Return filtered addressable SGS xz and yz upper-face stresses."""
        if not isinstance(config, (StaticSmagorinsky, LagrangianScaleDependentDynamic)):
            raise TypeError("unsupported SGS choice")
        return self._dry_sgs_vertical_flux(
            context.arrays,
            self._momentum_sgs_coefficient(context, config),
        )

    @staticmethod
    def _momentum_sgs_coefficient(
        context: ZSlabDryFlowContext,
        config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
    ):
        if isinstance(config, StaticSmagorinsky):
            return jnp.full_like(context.arrays.dudx, config.coefficient**2)
        closure = context.closure
        if not isinstance(closure, LasdClosureMemory):
            raise TypeError("momentum LASD requires initialized closure memory")
        return closure.momentum.coefficient.payload

    def coriolis_geostrophic_tendency(
        self,
        context: ZSlabDryFlowContext,
        config: NoRotation | CoriolisGeostrophic,
    ) -> VelocityVector:
        velocity = context.velocity
        if isinstance(config, NoRotation):
            x = jnp.zeros_like(velocity.x.payload)
            y = jnp.zeros_like(velocity.y.payload)
        elif isinstance(config, CoriolisGeostrophic):
            local_f = jnp.asarray(
                config.coriolis_parameter,
                dtype=velocity.x.payload.dtype,
            )
            horizontal_f = jnp.asarray(
                config.horizontal_coriolis_parameter,
                dtype=velocity.x.payload.dtype,
            )
            x = (
                local_f * (velocity.y.payload - config.geostrophic_y_velocity)
                - horizontal_f * context.arrays.w_at_cells
            )
            y = -local_f * (velocity.x.payload - config.geostrophic_x_velocity)
            z = horizontal_f.astype(velocity.z.owned.payload.dtype) * (
                context.arrays.u_upper - config.geostrophic_x_velocity
            )
            z = z.at[:, -1].set(
                jnp.where(
                    context.arrays.upper_is_physical[:, None, None],
                    0.0,
                    z[:, -1],
                )
            )
        else:
            raise TypeError("unsupported rotation choice")
        if isinstance(config, NoRotation):
            z = jnp.zeros_like(velocity.z.owned.payload)
        return self._dry_tendency(
            x,
            y,
            z,
        )

    def wind_tunnel_tendency(
        self,
        velocity: VelocityVector,
        model: WindTunnelModel,
        environment: Any,
    ) -> VelocityVector:
        """Evaluate distributed pure-thrust ADM and precursor fringe forcing."""
        if not isinstance(model, WindTunnelModel):
            raise TypeError("unsupported wind-tunnel model")
        self._validate_velocity_cell(velocity.x, XVelocity)
        self._validate_velocity_cell(velocity.y, YVelocity)
        self._validate_field(velocity.z.owned, ZFace)
        if velocity.z.owned.quantity is not VerticalVelocity:
            raise TypeError("wind-tunnel vertical velocity requires VerticalVelocity")

        disk = model.actuator_disk
        if isinstance(disk, PureThrustActuatorDisk):
            disk_parameters = (
                True,
                disk.x,
                disk.y,
                disk.z,
                disk.diameter,
                disk.hub_diameter,
                disk.thrust_coefficient_prime,
                disk.normal_smoothing_width,
                disk.transverse_smoothing_width,
                disk.yaw_degrees,
                disk.filtered_velocity_correction,
            )
        elif isinstance(disk, NoActuatorDisk):
            disk_parameters = (
                False,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                False,
            )
        else:
            raise TypeError("unsupported actuator-disk choice")

        fringe = model.fringe
        if isinstance(fringe, ConcurrentPrecursorFringe):
            if not isinstance(environment, ConcurrentPrecursorEnvironment):
                raise TypeError(
                    "concurrent fringe requires ConcurrentPrecursorEnvironment"
                )
            target = environment.velocity
            self._validate_velocity_cell(target.x, XVelocity)
            self._validate_velocity_cell(target.y, YVelocity)
            self._validate_field(target.z.owned, ZFace)
            if target.z.owned.quantity is not VerticalVelocity:
                raise TypeError("precursor vertical target requires VerticalVelocity")
            if fringe.start_x >= self.decomposition.grid.lx:
                raise ValueError("fringe start must lie before the periodic seam")
            fringe_parameters = (True, fringe.start_x, fringe.relaxation_time)
        elif isinstance(fringe, NoFringe):
            target = velocity
            fringe_parameters = (False, 0.0, 1.0)
        else:
            raise TypeError("unsupported wind-tunnel fringe choice")

        x, y, z = self._wind_tunnel(
            velocity.x.payload,
            velocity.y.payload,
            velocity.z.owned.payload,
            target.x.payload,
            target.y.payload,
            target.z.owned.payload,
            *disk_parameters,
            *fringe_parameters,
        )
        return self._dry_tendency(x, y, z)

    def combine_tendencies(
        self,
        tendencies: tuple[VelocityVector, ...],
    ) -> VelocityVector:
        if not tendencies:
            raise ValueError("at least one evaluated tendency is required")
        first = tendencies[0]
        for tendency in tendencies:
            self._validate_tendency_cell(tendency.x, XVelocityTendency)
            self._validate_tendency_cell(tendency.y, YVelocityTendency)
            self._validate_field(tendency.z.owned, ZFace)
            if (
                tendency.z.owned.quantity is not VerticalVelocityTendency
                or tendency.z.owned.phase is not Evaluated
            ):
                raise TypeError("only evaluated velocity tendencies may be combined")
            if (
                tendency.x.payload.dtype != first.x.payload.dtype
                or tendency.y.payload.dtype != first.y.payload.dtype
                or tendency.z.owned.payload.dtype != first.z.owned.payload.dtype
            ):
                raise TypeError("combined tendencies must preserve component dtypes")
        return self._dry_tendency(
            sum(
                (term.x.payload for term in tendencies), jnp.zeros_like(first.x.payload)
            ),
            sum(
                (term.y.payload for term in tendencies), jnp.zeros_like(first.y.payload)
            ),
            sum(
                (term.z.owned.payload for term in tendencies),
                jnp.zeros_like(first.z.owned.payload),
            ),
            sum(
                (term.z.lower_boundary for term in tendencies),
                jnp.zeros_like(first.z.lower_boundary),
            ),
        )

    def velocity_divergence(self, velocity: VelocityVector) -> AddressableField:
        """Apply the compatible hybrid ``D`` to one owned velocity product."""
        self._validate_velocity_cell(velocity.x, XVelocity)
        self._validate_velocity_cell(velocity.y, YVelocity)
        vertical = self.divergence_z(velocity.z)
        horizontal = self._horizontal_divergence(
            velocity.x.payload,
            velocity.y.payload,
        )
        return AddressableField(
            Divergence,
            Cell,
            self._expected_regions(Cell),
            Evaluated,
            horizontal + vertical.payload,
        )

    def pressure_rhs(
        self,
        divergence: AddressableField,
        inverse_dt: float,
    ) -> AddressableField:
        self._validate_field(divergence, Cell)
        if divergence.quantity is not Divergence:
            raise TypeError("pressure RHS requires Divergence")
        return AddressableField(
            PressureRhs,
            Cell,
            self._expected_regions(Cell),
            Evaluated,
            divergence.payload * inverse_dt,
        )

    def pressure_gradient(self, pressure: AddressableField) -> PressureGradient:
        """Apply compatible horizontal spectral and vertical face gradients."""
        self._validate_field(pressure, Cell)
        if pressure.quantity is not PressureCorrection:
            raise TypeError("pressure gradient requires PressureCorrection")
        gradient_x, gradient_y = self._horizontal_gradient(pressure.payload)
        return PressureGradient(
            AddressableField(
                XPressureGradient,
                Cell,
                self._expected_regions(Cell),
                Evaluated,
                gradient_x,
            ),
            AddressableField(
                YPressureGradient,
                Cell,
                self._expected_regions(Cell),
                Evaluated,
                gradient_y,
            ),
            self.pressure_gradient_z(pressure, VerticalBoundary(0.0, 0.0)),
        )

    def correct_velocity(
        self,
        velocity: VelocityVector,
        gradient: PressureGradient,
        dt: float,
    ) -> VelocityVector:
        """Return the projected velocity without changing semantic ownership."""
        x_payload = self._correct(velocity.x.payload, gradient.x.payload, dt)
        y_payload = self._correct(velocity.y.payload, gradient.y.payload, dt)
        z_payload = self._correct(
            velocity.z.owned.payload,
            gradient.z.owned.payload,
            dt,
        )
        boundary_dt = jnp.asarray(dt, dtype=z_payload.dtype)
        lower_boundary = jnp.asarray(
            velocity.z.lower_boundary,
            dtype=z_payload.dtype,
        ) - boundary_dt * jnp.asarray(
            gradient.z.lower_boundary,
            dtype=z_payload.dtype,
        )
        return VelocityVector(
            AddressableField(
                XVelocity,
                Cell,
                self._expected_regions(Cell),
                Projected,
                x_payload,
            ),
            AddressableField(
                YVelocity,
                Cell,
                self._expected_regions(Cell),
                Projected,
                y_payload,
            ),
            ZFaceFieldContext(
                AddressableField(
                    VerticalVelocity,
                    ZFace,
                    self._expected_regions(ZFace),
                    Projected,
                    z_payload,
                ),
                lower_boundary,
            ),
        )

    def _validate_tendency_cell(
        self,
        field: AddressableField,
        quantity: type,
    ) -> None:
        self._validate_field(field, Cell)
        if field.quantity is not quantity or field.phase is not Evaluated:
            raise TypeError("AB2 requires an evaluated velocity tendency")

    def ab2_candidate_velocity(
        self,
        velocity: VelocityVector,
        current_tendency: VelocityVector,
        previous_tendency: VelocityVector,
        *,
        dt: float,
        current_weight: float,
        previous_weight: float,
    ) -> VelocityVector:
        """Form a locally owned Euler/AB2 candidate."""
        self._validate_velocity_cell(velocity.x, XVelocity)
        self._validate_velocity_cell(velocity.y, YVelocity)
        if velocity.x.phase is not Projected or velocity.y.phase is not Projected:
            raise TypeError("AB2 requires projected velocity state components")
        self._validate_field(velocity.z.owned, ZFace)
        if (
            velocity.z.owned.quantity is not VerticalVelocity
            or velocity.z.owned.phase is not Projected
        ):
            raise TypeError("AB2 requires projected vertical velocity")
        self._validate_tendency_cell(current_tendency.x, XVelocityTendency)
        self._validate_tendency_cell(previous_tendency.x, XVelocityTendency)
        self._validate_tendency_cell(current_tendency.y, YVelocityTendency)
        self._validate_tendency_cell(previous_tendency.y, YVelocityTendency)
        for tendency in (current_tendency.z.owned, previous_tendency.z.owned):
            self._validate_field(tendency, ZFace)
            if (
                tendency.quantity is not VerticalVelocityTendency
                or tendency.phase is not Evaluated
            ):
                raise TypeError("AB2 requires an evaluated vertical tendency")

        def update(state, current, previous):
            return self._ab2_update(
                state,
                current,
                previous,
                dt,
                current_weight,
                previous_weight,
            )

        x_payload = update(
            velocity.x.payload,
            current_tendency.x.payload,
            previous_tendency.x.payload,
        )
        y_payload = update(
            velocity.y.payload,
            current_tendency.y.payload,
            previous_tendency.y.payload,
        )
        z_payload = update(
            velocity.z.owned.payload,
            current_tendency.z.owned.payload,
            previous_tendency.z.owned.payload,
        )
        boundary_dtype = z_payload.dtype
        boundary_dt = jnp.asarray(dt, dtype=boundary_dtype)
        current_coefficient = jnp.asarray(current_weight, dtype=boundary_dtype)
        previous_coefficient = jnp.asarray(previous_weight, dtype=boundary_dtype)
        lower_boundary = jnp.asarray(
            velocity.z.lower_boundary,
            dtype=boundary_dtype,
        ) + boundary_dt * (
            current_coefficient
            * jnp.asarray(current_tendency.z.lower_boundary, dtype=boundary_dtype)
            + previous_coefficient
            * jnp.asarray(previous_tendency.z.lower_boundary, dtype=boundary_dtype)
        )
        return VelocityVector(
            AddressableField(
                XVelocity,
                Cell,
                self._expected_regions(Cell),
                Candidate,
                x_payload,
            ),
            AddressableField(
                YVelocity,
                Cell,
                self._expected_regions(Cell),
                Candidate,
                y_payload,
            ),
            ZFaceFieldContext(
                AddressableField(
                    VerticalVelocity,
                    ZFace,
                    self._expected_regions(ZFace),
                    Candidate,
                    z_payload,
                ),
                lower_boundary,
            ),
        )

    def ab2_candidate_scalar(
        self,
        scalar: AddressableField,
        current_tendency: AddressableField,
        previous_tendency: AddressableField,
        *,
        dt: float,
        current_weight: float,
        previous_weight: float,
    ) -> AddressableField:
        self._validate_field(scalar, Cell)
        tendency_quantity = (
            PotentialTemperatureTendency
            if scalar.quantity is PotentialTemperaturePerturbation
            else PassiveScalarTendency
        )
        if (
            scalar.quantity
            not in (
                PotentialTemperaturePerturbation,
                PassiveScalarConcentration,
            )
            or scalar.phase is not Accepted
        ):
            raise TypeError("AB2 requires an accepted supported scalar")
        for tendency in (current_tendency, previous_tendency):
            self._validate_field(tendency, Cell)
            if (
                tendency.quantity is not tendency_quantity
                or tendency.phase is not Evaluated
            ):
                raise TypeError("AB2 requires evaluated scalar tendency")
        return AddressableField(
            scalar.quantity,
            Cell,
            self._expected_regions(Cell),
            Candidate,
            self._ab2_update(
                scalar.payload,
                current_tendency.payload,
                previous_tendency.payload,
                dt,
                current_weight,
                previous_weight,
            ),
        )

    def accept_scalar(self, scalar: AddressableField) -> AddressableField:
        self._validate_field(scalar, Cell)
        if (
            scalar.quantity
            not in (
                PotentialTemperaturePerturbation,
                PassiveScalarConcentration,
            )
            or scalar.phase is not Candidate
        ):
            raise TypeError("only a candidate supported scalar may be accepted")
        return AddressableField(
            scalar.quantity,
            Cell,
            self._expected_regions(Cell),
            Accepted,
            scalar.payload,
        )


def build_zslab_interpreter(
    decomposition: EqualZSlab,
    *,
    addressable_shards: tuple[int, ...],
    axis_name: str = "wireles_z",
) -> JaxZSlabInterpreter:
    """Build mapped kernels without capturing any field-sized constants."""
    shard_count = decomposition.shard_count
    if not addressable_shards:
        raise ValueError("at least one addressable shard is required")
    if len(set(addressable_shards)) != len(addressable_shards):
        raise ValueError("addressable shard indices must be unique")
    if any(not 0 <= index < shard_count for index in addressable_shards):
        raise ValueError("addressable shard index is outside the global z mesh")

    previous_permutation = tuple(
        (source, source + 1) for source in range(shard_count - 1)
    )
    next_permutation = tuple((source, source - 1) for source in range(1, shard_count))

    def exchange_local(packed):
        if packed.ndim != 4:
            raise ValueError("packed local fields must have shape (field, z, y, x)")
        index = lax.axis_index(axis_name)
        if shard_count == 1:
            lower = jnp.zeros_like(packed[:, -1])
            upper = jnp.zeros_like(packed[:, 0])
        else:
            lower = lax.ppermute(
                packed[:, -1],
                axis_name,
                previous_permutation,
            )
            upper = lax.ppermute(
                packed[:, 0],
                axis_name,
                next_permutation,
            )
        lower_physical = index == 0
        upper_physical = index == shard_count - 1
        lower = jnp.where(lower_physical, jnp.zeros_like(lower), lower)
        upper = jnp.where(upper_physical, jnp.zeros_like(upper), upper)
        return PackedHaloArrays(
            lower,
            upper,
            lower_physical,
            upper_physical,
        )

    def pressure_gradient_local(pressure, upper_boundary_gradient):
        halo = exchange_local(pressure[None, ...])
        next_cell = halo.upper[0]
        interface_gradient = (next_cell - pressure[-1]) / decomposition.grid.dz
        boundary_gradient = jnp.broadcast_to(
            jnp.asarray(upper_boundary_gradient, pressure.dtype),
            pressure.shape[1:],
        )
        last = jnp.where(halo.upper_is_physical, boundary_gradient, interface_gradient)
        interior = (pressure[1:] - pressure[:-1]) / decomposition.grid.dz
        return jnp.concatenate((interior, last[None, ...]), axis=0)

    def divergence_local(upper_faces, lower_boundary_face):
        halo = exchange_local(upper_faces[None, ...])
        boundary_face = jnp.broadcast_to(
            jnp.asarray(lower_boundary_face, upper_faces.dtype),
            upper_faces.shape[1:],
        )
        lower_first = jnp.where(
            halo.lower_is_physical,
            boundary_face,
            halo.lower[0],
        )
        lower_faces = jnp.concatenate(
            (lower_first[None, ...], upper_faces[:-1]),
            axis=0,
        )
        return (upper_faces - lower_faces) / decomposition.grid.dz

    def enforce_upper_boundary_local(upper_faces, upper_boundary_face):
        index = lax.axis_index(axis_name)
        boundary_face = jnp.broadcast_to(
            jnp.asarray(upper_boundary_face, upper_faces.dtype),
            upper_faces.shape[1:],
        )
        last = jnp.where(
            index == shard_count - 1,
            boundary_face,
            upper_faces[-1],
        )
        return upper_faces.at[-1].set(last)

    grid = decomposition.grid
    kx = 2.0 * jnp.pi * jnp.fft.rfftfreq(grid.nx, d=grid.lx / grid.nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(grid.ny, d=grid.ly / grid.ny)
    keep = jnp.ones((grid.ny, grid.nx // 2 + 1))
    if grid.nx % 2 == 0:
        kx = kx.at[-1].set(0.0)
        keep = keep.at[:, -1].set(0.0)
    if grid.ny % 2 == 0:
        ky = ky.at[grid.ny // 2].set(0.0)
        keep = keep.at[grid.ny // 2, :].set(0.0)

    x_mode = jnp.arange(grid.nx // 2 + 1)
    y_mode = jnp.fft.fftfreq(grid.ny) * grid.ny
    two_thirds = (jnp.abs(y_mode)[:, None] <= grid.ny // 3) & (
        x_mode[None, :] <= grid.nx // 3
    )

    def horizontal_derivative_local(values, axis):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        local_kx = kx.astype(values.real.dtype)
        local_ky = ky.astype(values.real.dtype)
        if axis == 0:
            multiplier = 1j * local_kx
        else:
            multiplier = 1j * local_ky[:, None]
        return jnp.fft.irfftn(
            spectrum * multiplier * keep.astype(values.real.dtype),
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def two_thirds_filter_local(values):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        return jnp.fft.irfftn(
            spectrum * two_thirds,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def wall_filter_local(values, filter_width):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        cutoff_x = jnp.floor(grid.nx / (2.0 * filter_width))
        cutoff_y = jnp.floor(grid.ny / (2.0 * filter_width))
        wall_keep = (jnp.abs(y_mode)[:, None] < cutoff_y) & (
            x_mode[None, :] < cutoff_x
        )
        return jnp.fft.irfftn(
            spectrum * wall_keep,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def truncated_derivative_local(values, axis):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        local_kx = kx.astype(values.real.dtype)
        local_ky = ky.astype(values.real.dtype)
        if axis == 0:
            multiplier = 1j * local_kx
        else:
            multiplier = 1j * local_ky[:, None]
        return jnp.fft.irfftn(
            spectrum * multiplier * two_thirds,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def strain_magnitude_local(
        dudx,
        dudy,
        dudz,
        dvdx,
        dvdy,
        dvdz,
        dwdx,
        dwdy,
        dwdz,
    ):
        sxy = 0.5 * (dudy + dvdx)
        sxz = 0.5 * (dudz + dwdx)
        syz = 0.5 * (dvdz + dwdy)
        symmetric_dot = (
            dudx * dudx
            + dvdy * dvdy
            + dwdz * dwdz
            + 2.0 * (sxy * sxy + sxz * sxz + syz * syz)
        )
        return jnp.sqrt(jnp.maximum(2.0 * symmetric_dot, 0.0))

    def dry_flow_context_local(u, v, w_upper, lower_boundary):
        halo = exchange_local(jnp.stack((u, v, w_upper), axis=0))
        lower_boundary_plane = jnp.broadcast_to(
            jnp.asarray(lower_boundary, dtype=w_upper.dtype),
            w_upper.shape[1:],
        )
        previous_u = jnp.where(halo.lower_is_physical, u[0], halo.lower[0])
        previous_v = jnp.where(halo.lower_is_physical, v[0], halo.lower[1])
        next_u_plane = jnp.where(halo.upper_is_physical, u[-1], halo.upper[0])
        next_v_plane = jnp.where(halo.upper_is_physical, v[-1], halo.upper[1])
        next_w_upper = jnp.where(
            halo.upper_is_physical,
            w_upper[-1],
            halo.upper[2],
        )
        w_lower_plane = jnp.where(
            halo.lower_is_physical,
            lower_boundary_plane,
            halo.lower[2],
        )
        lower_faces = jnp.concatenate((w_lower_plane[None], w_upper[:-1]), axis=0)
        w_at_cells = 0.5 * (lower_faces + w_upper)
        next_u = jnp.concatenate((u[1:], next_u_plane[None]), axis=0)
        next_v = jnp.concatenate((v[1:], next_v_plane[None]), axis=0)
        u_upper = 0.5 * (u + next_u)
        v_upper = 0.5 * (v + next_v)
        u_lower = 0.5 * (previous_u + u[0])
        v_lower = 0.5 * (previous_v + v[0])

        dudz_upper = (next_u - u) / grid.dz
        dvdz_upper = (next_v - v) / grid.dz
        dudz_upper = dudz_upper.at[-1].set(
            jnp.where(halo.upper_is_physical, 0.0, dudz_upper[-1])
        )
        dvdz_upper = dvdz_upper.at[-1].set(
            jnp.where(halo.upper_is_physical, 0.0, dvdz_upper[-1])
        )
        index = lax.axis_index(axis_name)
        wall_correction = 1.0 / math.log(3.0)
        dudz_upper = dudz_upper.at[0].set(
            jnp.where(
                index == 0,
                wall_correction * dudz_upper[0],
                dudz_upper[0],
            )
        )
        dvdz_upper = dvdz_upper.at[0].set(
            jnp.where(
                index == 0,
                wall_correction * dvdz_upper[0],
                dvdz_upper[0],
            )
        )
        lower_dudz = jnp.where(
            halo.lower_is_physical,
            jnp.zeros_like(previous_u),
            (u[0] - previous_u) / grid.dz,
        )
        lower_dvdz = jnp.where(
            halo.lower_is_physical,
            jnp.zeros_like(previous_v),
            (v[0] - previous_v) / grid.dz,
        )
        dudz_lower = jnp.concatenate((lower_dudz[None], dudz_upper[:-1]), axis=0)
        dvdz_lower = jnp.concatenate((lower_dvdz[None], dvdz_upper[:-1]), axis=0)

        dudx = horizontal_derivative_local(u, 0)
        dudy = horizontal_derivative_local(u, 1)
        dvdx = horizontal_derivative_local(v, 0)
        dvdy = horizontal_derivative_local(v, 1)
        dwdx_at_cells = horizontal_derivative_local(w_at_cells, 0)
        dwdy_at_cells = horizontal_derivative_local(w_at_cells, 1)
        dwdz = (w_upper - lower_faces) / grid.dz
        dwdx_upper = horizontal_derivative_local(w_upper, 0)
        dwdy_upper = horizontal_derivative_local(w_upper, 1)

        next_dudx = jnp.concatenate(
            (dudx[1:], horizontal_derivative_local(next_u_plane, 0)[None]),
            axis=0,
        )
        next_dudy = jnp.concatenate(
            (dudy[1:], horizontal_derivative_local(next_u_plane, 1)[None]),
            axis=0,
        )
        next_dvdx = jnp.concatenate(
            (dvdx[1:], horizontal_derivative_local(next_v_plane, 0)[None]),
            axis=0,
        )
        next_dvdy = jnp.concatenate(
            (dvdy[1:], horizontal_derivative_local(next_v_plane, 1)[None]),
            axis=0,
        )
        next_dwdz_plane = (next_w_upper - w_upper[-1]) / grid.dz
        next_dwdz = jnp.concatenate((dwdz[1:], next_dwdz_plane[None]), axis=0)
        next_w_cell_plane = 0.5 * (w_upper[-1] + next_w_upper)
        next_w_cell = jnp.concatenate(
            (w_at_cells[1:], next_w_cell_plane[None]),
            axis=0,
        )
        return ZSlabDryFlowArrays(
            u,
            v,
            w_upper,
            w_lower_plane,
            u_upper,
            v_upper,
            u_lower,
            v_lower,
            w_at_cells,
            next_w_cell,
            dudx,
            dudy,
            0.5 * (dudz_lower + dudz_upper),
            dvdx,
            dvdy,
            0.5 * (dvdz_lower + dvdz_upper),
            dwdx_at_cells,
            dwdy_at_cells,
            dwdz,
            dudz_upper,
            dvdz_upper,
            dwdx_upper,
            dwdy_upper,
            0.5 * (dudx + next_dudx),
            0.5 * (dudy + next_dudy),
            0.5 * (dvdx + next_dvdx),
            0.5 * (dvdy + next_dvdy),
            0.5 * (dwdz + next_dwdz),
            halo.upper_is_physical,
        )

    def scalar_context_local(theta):
        halo = exchange_local(theta[None, ...])
        previous_plane = jnp.where(
            halo.lower_is_physical,
            theta[0],
            halo.lower[0],
        )
        next_plane = jnp.where(
            halo.upper_is_physical,
            theta[-1],
            halo.upper[0],
        )
        next_theta = jnp.concatenate((theta[1:], next_plane[None]), axis=0)
        theta_upper = 0.5 * (theta + next_theta)
        theta_lower = jnp.concatenate(
            ((0.5 * (previous_plane + theta[0]))[None], theta_upper[:-1]),
            axis=0,
        )
        dtheta_dz_upper = (next_theta - theta) / grid.dz
        dtheta_dz_upper = dtheta_dz_upper.at[-1].set(
            jnp.where(halo.upper_is_physical, 0.0, dtheta_dz_upper[-1])
        )
        previous_theta = jnp.concatenate((previous_plane[None], theta[:-1]), axis=0)
        centered_dtheta_dz = (next_theta - previous_theta) / (2.0 * grid.dz)
        centered_dtheta_dz = centered_dtheta_dz.at[0].set(
            jnp.where(
                halo.lower_is_physical,
                (next_theta[0] - theta[0]) / grid.dz,
                centered_dtheta_dz[0],
            )
        )
        centered_dtheta_dz = centered_dtheta_dz.at[-1].set(
            jnp.where(
                halo.upper_is_physical,
                (theta[-1] - previous_theta[-1]) / grid.dz,
                centered_dtheta_dz[-1],
            )
        )
        return ZSlabScalarArrays(
            theta,
            theta_upper,
            theta_lower,
            horizontal_derivative_local(theta, 0),
            horizontal_derivative_local(theta, 1),
            centered_dtheta_dz,
            dtheta_dz_upper,
            halo.upper_is_physical,
        )

    def buoyancy_local(scalar, coefficient):
        local_coefficient = jnp.asarray(coefficient, dtype=scalar.theta.dtype)
        hydrostatic_free_theta = scalar.theta_upper - jnp.mean(
            scalar.theta_upper,
            axis=(-2, -1),
            keepdims=True,
        )
        z = local_coefficient * hydrostatic_free_theta
        return z.at[-1].set(jnp.where(scalar.upper_is_physical, 0.0, z[-1]))

    def rayleigh_damping_local(
        u,
        v,
        w_upper,
        start_height,
        maximum_rate,
        target_u,
        target_v,
    ):
        index = lax.axis_index(axis_name)
        local_nz = u.shape[0]
        global_cell = index * local_nz + jnp.arange(local_nz, dtype=u.dtype)
        cell_height = (global_cell + 0.5) * grid.dz
        upper_face_height = (global_cell + 1.0) * grid.dz
        depth = grid.lz - jnp.asarray(start_height, dtype=u.dtype)
        cell_eta = jnp.clip((cell_height - start_height) / depth, 0.0, 1.0)
        face_eta = jnp.clip(
            (upper_face_height - start_height) / depth,
            0.0,
            1.0,
        )
        cell_rate = jnp.asarray(maximum_rate, dtype=u.dtype) * cell_eta**2
        face_rate = jnp.asarray(maximum_rate, dtype=w_upper.dtype) * (
            face_eta.astype(w_upper.dtype) ** 2
        )
        return (
            -cell_rate[:, None, None] * (u - target_u),
            -cell_rate[:, None, None] * (v - target_v),
            -face_rate[:, None, None] * w_upper,
        )

    def scalar_advection_local(scalar, momentum):
        w_lower = jnp.concatenate(
            (momentum.w_lower[None], momentum.w_upper[:-1]),
            axis=0,
        )
        upper_flux = two_thirds_filter_local(momentum.w_upper * scalar.theta_upper)
        lower_flux = two_thirds_filter_local(w_lower * scalar.theta_lower)
        return -(
            truncated_derivative_local(momentum.u * scalar.theta, 0)
            + truncated_derivative_local(momentum.v * scalar.theta, 1)
            + (upper_flux - lower_flux) / grid.dz
        )

    def scalar_sgs_local(
        scalar,
        momentum,
        coefficient,
        lower_boundary_flux,
        upper_boundary_flux,
        stability_buoyancy_coefficient,
        stability_beta,
        stability_power,
    ):
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        cell_magnitude = strain_magnitude_local(
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
        face_magnitude = strain_magnitude_local(
            momentum.dudx_upper,
            momentum.dudy_upper,
            momentum.dudz_upper,
            momentum.dvdx_upper,
            momentum.dvdy_upper,
            momentum.dvdz_upper,
            momentum.dwdx_upper,
            momentum.dwdy_upper,
            momentum.dwdz_upper,
        )
        local_coefficient = coefficient.astype(scalar.theta.dtype)
        n2 = jnp.maximum(
            jnp.asarray(stability_buoyancy_coefficient, dtype=scalar.theta.dtype)
            * scalar.dtheta_dz_at_cells,
            0.0,
        )
        richardson = n2 / jnp.maximum(cell_magnitude**2, 1.0e-24)
        stability = (
            1.0
            + jnp.asarray(stability_beta, dtype=scalar.theta.dtype) * richardson
        ) ** (-jnp.asarray(stability_power, dtype=scalar.theta.dtype))
        effective_coefficient = local_coefficient * stability
        coefficient_halo = exchange_local(effective_coefficient[None, ...])
        next_coefficient_plane = jnp.where(
            coefficient_halo.upper_is_physical,
            effective_coefficient[-1],
            coefficient_halo.upper[0],
        )
        next_coefficient = jnp.concatenate(
            (effective_coefficient[1:], next_coefficient_plane[None]),
            axis=0,
        )
        face_coefficient = 0.5 * (effective_coefficient + next_coefficient)
        cell_diffusivity = effective_coefficient * delta**2 * cell_magnitude
        face_diffusivity = face_coefficient * delta**2 * face_magnitude
        qx = -cell_diffusivity * scalar.dtheta_dx
        qy = -cell_diffusivity * scalar.dtheta_dy
        qz = -face_diffusivity * scalar.dtheta_dz_upper
        qz = qz.at[-1].set(
            jnp.where(scalar.upper_is_physical, upper_boundary_flux, qz[-1])
        )
        qz = two_thirds_filter_local(qz)
        flux_halo = exchange_local(qz[None, ...])
        lower_plane = jnp.where(
            flux_halo.lower_is_physical,
            jnp.full_like(qz[0], lower_boundary_flux),
            flux_halo.lower[0],
        )
        lower_qz = jnp.concatenate((lower_plane[None], qz[:-1]), axis=0)
        return -(
            truncated_derivative_local(qx, 0)
            + truncated_derivative_local(qy, 1)
            + (qz - lower_qz) / grid.dz
        )

    def lasd_diagnostics_local(
        scalar,
        momentum,
        momentum_coefficient,
        scalar_coefficient,
        lower_boundary_flux,
        upper_boundary_flux,
        dissipation_coefficient,
        scalar_variance_coefficient,
        wall_gradient_factor,
        horizontal_homogeneous_wall,
        stability_buoyancy_coefficient,
        stability_beta,
        stability_power,
    ):
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        cell_magnitude = strain_magnitude_local(
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
        lower_is_physical = lax.axis_index(axis_name) == 0
        diagnostic_dudz = momentum.dudz_at_cells.at[0].set(
            jnp.where(
                lower_is_physical & (wall_gradient_factor > 0.0),
                momentum.u[0] * wall_gradient_factor,
                momentum.dudz_at_cells[0],
            )
        )
        diagnostic_dvdz = momentum.dvdz_at_cells.at[0].set(
            jnp.where(
                lower_is_physical & (wall_gradient_factor > 0.0),
                momentum.v[0] * wall_gradient_factor,
                momentum.dvdz_at_cells[0],
            )
        )
        diagnostic_magnitude = strain_magnitude_local(
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
        face_magnitude = strain_magnitude_local(
            momentum.dudx_upper,
            momentum.dudy_upper,
            momentum.dudz_upper,
            momentum.dvdx_upper,
            momentum.dvdy_upper,
            momentum.dvdz_upper,
            momentum.dwdx_upper,
            momentum.dwdy_upper,
            momentum.dwdz_upper,
        )
        momentum_diffusivity = momentum_coefficient * delta**2 * cell_magnitude
        n2 = jnp.maximum(
            jnp.asarray(stability_buoyancy_coefficient, dtype=scalar.theta.dtype)
            * scalar.dtheta_dz_at_cells,
            0.0,
        )
        richardson = n2 / jnp.maximum(cell_magnitude**2, 1.0e-24)
        stability = (
            1.0
            + jnp.asarray(stability_beta, dtype=scalar.theta.dtype) * richardson
        ) ** (-jnp.asarray(stability_power, dtype=scalar.theta.dtype))
        effective_scalar_coefficient = scalar_coefficient * stability
        scalar_diffusivity = (
            effective_scalar_coefficient * delta**2 * cell_magnitude
        )
        coefficient_halo = exchange_local(effective_scalar_coefficient[None, ...])
        next_coefficient_plane = jnp.where(
            coefficient_halo.upper_is_physical,
            effective_scalar_coefficient[-1],
            coefficient_halo.upper[0],
        )
        next_coefficient = jnp.concatenate(
            (effective_scalar_coefficient[1:], next_coefficient_plane[None]),
            axis=0,
        )
        face_diffusivity = (
            0.5 * (effective_scalar_coefficient + next_coefficient)
            * delta**2
            * face_magnitude
        )
        flux_x = -scalar_diffusivity * scalar.dtheta_dx
        flux_y = -scalar_diffusivity * scalar.dtheta_dy
        flux_z = -face_diffusivity * scalar.dtheta_dz_upper
        flux_z = flux_z.at[-1].set(
            jnp.where(scalar.upper_is_physical, upper_boundary_flux, flux_z[-1])
        )
        flux_z = two_thirds_filter_local(flux_z)
        flux_halo = exchange_local(flux_z[None, ...])
        lower_flux_plane = jnp.where(
            flux_halo.lower_is_physical,
            jnp.full_like(flux_z[0], lower_boundary_flux),
            flux_halo.lower[0],
        )
        lower_flux = jnp.concatenate((lower_flux_plane[None], flux_z[:-1]), axis=0)

        gradient_halo = exchange_local(scalar.dtheta_dz_upper[None, ...])
        zero_wall_cross_gradient = jnp.zeros_like(momentum.dwdx_upper[0])
        wall_face_magnitude = strain_magnitude_local(
            momentum.dudx[0],
            momentum.dudy[0],
            momentum.u[0] * wall_gradient_factor,
            momentum.dvdx[0],
            momentum.dvdy[0],
            momentum.v[0] * wall_gradient_factor,
            zero_wall_cross_gradient,
            zero_wall_cross_gradient,
            momentum.dwdz[0],
        )
        wall_scalar_diffusivity = (
            effective_scalar_coefficient[0] * delta**2 * wall_face_magnitude
        )
        wall_scalar_diffusivity = jnp.where(
            horizontal_homogeneous_wall,
            jnp.full_like(wall_scalar_diffusivity, jnp.mean(wall_scalar_diffusivity)),
            wall_scalar_diffusivity,
        )
        diagnostic_lower_face_diffusivity = jnp.where(
            lower_is_physical & (wall_gradient_factor > 0.0),
            wall_scalar_diffusivity,
            face_diffusivity[0],
        )
        boundary_gradient = jnp.where(
            diagnostic_lower_face_diffusivity > 0.0,
            -lower_flux_plane / diagnostic_lower_face_diffusivity,
            0.0,
        )
        lower_gradient_plane = jnp.where(
            gradient_halo.lower_is_physical,
            boundary_gradient,
            gradient_halo.lower[0],
        )
        lower_gradient = jnp.concatenate(
            (lower_gradient_plane[None], scalar.dtheta_dz_upper[:-1]),
            axis=0,
        )
        upper_gradient = scalar.dtheta_dz_upper.at[-1].set(
            jnp.where(
                scalar.upper_is_physical & (face_diffusivity[-1] > 0.0),
                -flux_z[-1]
                / jnp.where(
                    face_diffusivity[-1] > 0.0,
                    face_diffusivity[-1],
                    1.0,
                ),
                scalar.dtheta_dz_upper[-1],
            )
        )
        gradient_z = 0.5 * (lower_gradient + upper_gradient)
        flux_z_at_cells = 0.5 * (lower_flux + flux_z)
        shear_production = momentum_diffusivity * diagnostic_magnitude**2
        buoyancy_destruction = (
            scalar_diffusivity
            * stability_buoyancy_coefficient
            * scalar.dtheta_dz_at_cells
        )
        sgs_tke = jnp.maximum(
            (shear_production - buoyancy_destruction)
            * delta
            / dissipation_coefficient,
            0.0,
        ) ** (2.0 / 3.0)
        scalar_dissipation = -(
            flux_x * scalar.dtheta_dx
            + flux_y * scalar.dtheta_dy
            + flux_z_at_cells * gradient_z
        )
        scalar_length = delta * jnp.sqrt(
            jnp.maximum(effective_scalar_coefficient, 0.0)
        )
        sqrt_tke = jnp.sqrt(jnp.maximum(sgs_tke, 0.0))
        valid = sqrt_tke > jnp.finfo(sqrt_tke.dtype).tiny
        scalar_variance_numerator = (
            2.0 * scalar_length * scalar_dissipation / scalar_variance_coefficient
        )
        scalar_variance = jnp.maximum(
            jnp.where(
                valid,
                scalar_variance_numerator / jnp.where(valid, sqrt_tke, 1.0),
                0.0,
            ),
            0.0,
        )
        return (
            momentum_diffusivity,
            scalar_diffusivity,
            flux_x,
            flux_y,
            flux_z,
            sgs_tke,
            scalar_variance_numerator,
            scalar_variance,
        )

    def dry_advection_local(context):
        upper_u_flux = two_thirds_filter_local(context.w_upper * context.u_upper)
        upper_v_flux = two_thirds_filter_local(context.w_upper * context.v_upper)
        lower_u_flux_plane = two_thirds_filter_local(context.w_lower * context.u_lower)
        lower_v_flux_plane = two_thirds_filter_local(context.w_lower * context.v_lower)
        lower_u_flux = jnp.concatenate(
            (lower_u_flux_plane[None], upper_u_flux[:-1]),
            axis=0,
        )
        lower_v_flux = jnp.concatenate(
            (lower_v_flux_plane[None], upper_v_flux[:-1]),
            axis=0,
        )
        x = -(
            truncated_derivative_local(context.u * context.u, 0)
            + truncated_derivative_local(context.v * context.u, 1)
            + (upper_u_flux - lower_u_flux) / grid.dz
        )
        y = -(
            truncated_derivative_local(context.u * context.v, 0)
            + truncated_derivative_local(context.v * context.v, 1)
            + (upper_v_flux - lower_v_flux) / grid.dz
        )
        vertical_flux = two_thirds_filter_local(context.w_at_cells * context.w_at_cells)
        next_vertical_flux = two_thirds_filter_local(
            context.w_next_cell * context.w_next_cell
        )
        z = -(
            truncated_derivative_local(context.u_upper * context.w_upper, 0)
            + truncated_derivative_local(context.v_upper * context.w_upper, 1)
            + (next_vertical_flux - vertical_flux) / grid.dz
        )
        z = z.at[-1].set(jnp.where(context.upper_is_physical, 0.0, z[-1]))
        return x, y, z

    def dry_wall_local(context, drag, filtered, filter_width):
        index = lax.axis_index(axis_name)
        wall_velocity = wall_filter_local(
            jnp.stack((context.u[0], context.v[0])),
            filter_width,
        )
        wall_u = jnp.where(filtered, wall_velocity[0], context.u[0])
        wall_v = jnp.where(filtered, wall_velocity[1], context.v[0])
        speed = jnp.sqrt(wall_u * wall_u + wall_v * wall_v)
        wall_x = -drag * speed * wall_u / grid.dz
        wall_y = -drag * speed * wall_v / grid.dz
        x = jnp.zeros_like(context.u).at[0].set(jnp.where(index == 0, wall_x, 0.0))
        y = jnp.zeros_like(context.v).at[0].set(jnp.where(index == 0, wall_y, 0.0))
        return x, y, jnp.zeros_like(context.w_upper)

    def dry_sgs_vertical_flux_local(context, coefficient):
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        face_magnitude = strain_magnitude_local(
            context.dudx_upper,
            context.dudy_upper,
            context.dudz_upper,
            context.dvdx_upper,
            context.dvdy_upper,
            context.dvdz_upper,
            context.dwdx_upper,
            context.dwdy_upper,
            context.dwdz_upper,
        )
        coefficient_halo = exchange_local(coefficient[None, ...])
        next_coefficient_plane = jnp.where(
            coefficient_halo.upper_is_physical,
            coefficient[-1],
            coefficient_halo.upper[0],
        )
        next_coefficient = jnp.concatenate(
            (coefficient[1:], next_coefficient_plane[None]),
            axis=0,
        )
        face_viscosity = (
            0.5 * (coefficient + next_coefficient) * delta**2 * face_magnitude
        )
        txz = -face_viscosity * (context.dudz_upper + context.dwdx_upper)
        tyz = -face_viscosity * (context.dvdz_upper + context.dwdy_upper)
        txz = txz.at[-1].set(jnp.where(context.upper_is_physical, 0.0, txz[-1]))
        tyz = tyz.at[-1].set(jnp.where(context.upper_is_physical, 0.0, tyz[-1]))
        txz = two_thirds_filter_local(txz)
        tyz = two_thirds_filter_local(tyz)
        return txz, tyz

    def dry_sgs_local(context, coefficient):
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        cell_magnitude = strain_magnitude_local(
            context.dudx,
            context.dudy,
            context.dudz_at_cells,
            context.dvdx,
            context.dvdy,
            context.dvdz_at_cells,
            context.dwdx_at_cells,
            context.dwdy_at_cells,
            context.dwdz,
        )
        cell_viscosity = coefficient * delta**2 * cell_magnitude
        txx = -2.0 * cell_viscosity * context.dudx
        txy = -cell_viscosity * (context.dudy + context.dvdx)
        tyy = -2.0 * cell_viscosity * context.dvdy
        tzz = -2.0 * cell_viscosity * context.dwdz
        txz, tyz = dry_sgs_vertical_flux_local(context, coefficient)
        tzz = two_thirds_filter_local(tzz)

        stress_halo = exchange_local(jnp.stack((txz, tyz, tzz), axis=0))
        lower_txz_plane = jnp.where(
            stress_halo.lower_is_physical,
            jnp.zeros_like(txz[0]),
            stress_halo.lower[0],
        )
        lower_tyz_plane = jnp.where(
            stress_halo.lower_is_physical,
            jnp.zeros_like(tyz[0]),
            stress_halo.lower[1],
        )
        lower_txz = jnp.concatenate((lower_txz_plane[None], txz[:-1]), axis=0)
        lower_tyz = jnp.concatenate((lower_tyz_plane[None], tyz[:-1]), axis=0)
        next_tzz_plane = jnp.where(
            stress_halo.upper_is_physical,
            tzz[-1],
            stress_halo.upper[2],
        )
        next_tzz = jnp.concatenate((tzz[1:], next_tzz_plane[None]), axis=0)
        x = -(
            truncated_derivative_local(txx, 0)
            + truncated_derivative_local(txy, 1)
            + (txz - lower_txz) / grid.dz
        )
        y = -(
            truncated_derivative_local(txy, 0)
            + truncated_derivative_local(tyy, 1)
            + (tyz - lower_tyz) / grid.dz
        )
        z = -(
            truncated_derivative_local(txz, 0)
            + truncated_derivative_local(tyz, 1)
            + (next_tzz - tzz) / grid.dz
        )
        z = z.at[-1].set(jnp.where(stress_halo.upper_is_physical, 0.0, z[-1]))
        return x, y, z

    def lasd_accumulate_local(
        u,
        v,
        w_at_cells,
        trajectory_x,
        trajectory_y,
        trajectory_z,
        update_interval,
    ):
        interval = jnp.asarray(update_interval, dtype=u.dtype)
        return (
            trajectory_x + u / interval,
            trajectory_y + v / interval,
            trajectory_z + w_at_cells / interval,
        )

    def lasd_filter_local(values, filter_width):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        x_mode = jnp.arange(grid.nx // 2 + 1)
        y_mode = jnp.abs(jnp.fft.fftfreq(grid.ny) * grid.ny)
        cutoff_x = jnp.floor(grid.nx / (2.0 * filter_width) + 0.5)
        cutoff_y = jnp.floor(grid.ny / (2.0 * filter_width) + 0.5)
        mask = (y_mode[:, None] < cutoff_y) & (x_mode[None, :] < cutoff_x)
        return jnp.fft.irfftn(
            spectrum * mask[None, ...],
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def lasd_filter_components_local(values, filter_width):
        return jnp.moveaxis(
            jax.vmap(lambda value: lasd_filter_local(value, filter_width))(
                jnp.moveaxis(values, -1, 0)
            ),
            0,
            -1,
        )

    def strain_tensor_local(momentum):
        return jnp.stack(
            (
                momentum.dudx,
                0.5 * (momentum.dudy + momentum.dvdx),
                0.5 * (momentum.dudz_at_cells + momentum.dwdx_at_cells),
                momentum.dvdy,
                0.5 * (momentum.dvdz_at_cells + momentum.dwdy_at_cells),
                momentum.dwdz,
            ),
            axis=-1,
        )

    def symmetric_dot_local(left, right):
        return (
            left[..., 0] * right[..., 0]
            + 2.0 * left[..., 1] * right[..., 1]
            + 2.0 * left[..., 2] * right[..., 2]
            + left[..., 3] * right[..., 3]
            + 2.0 * left[..., 4] * right[..., 4]
            + left[..., 5] * right[..., 5]
        )

    def tensor_magnitude_local(tensor):
        return jnp.sqrt(jnp.maximum(2.0 * symmetric_dot_local(tensor, tensor), 0.0))

    def momentum_contractions_local(momentum, filter_grid_ratio, ratio):
        tensor = strain_tensor_local(momentum)
        magnitude = tensor_magnitude_local(tensor)
        velocity = jnp.stack((momentum.u, momentum.v, momentum.w_at_cells), axis=-1)
        products = jnp.stack(
            (
                velocity[..., 0] ** 2,
                velocity[..., 0] * velocity[..., 1],
                velocity[..., 0] * velocity[..., 2],
                velocity[..., 1] ** 2,
                velocity[..., 1] * velocity[..., 2],
                velocity[..., 2] ** 2,
            ),
            axis=-1,
        )
        width = filter_grid_ratio * ratio
        velocity_hat = lasd_filter_components_local(velocity, width)
        products_hat = lasd_filter_components_local(products, width)
        tensor_hat = lasd_filter_components_local(tensor, width)
        magnitude_tensor_hat = lasd_filter_components_local(
            magnitude[..., None] * tensor,
            width,
        )
        resolved = jnp.stack(
            (
                products_hat[..., 0] - velocity_hat[..., 0] ** 2,
                products_hat[..., 1] - velocity_hat[..., 0] * velocity_hat[..., 1],
                products_hat[..., 2] - velocity_hat[..., 0] * velocity_hat[..., 2],
                products_hat[..., 3] - velocity_hat[..., 1] ** 2,
                products_hat[..., 4] - velocity_hat[..., 1] * velocity_hat[..., 2],
                products_hat[..., 5] - velocity_hat[..., 2] ** 2,
            ),
            axis=-1,
        )
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        model = (
            2.0
            * delta**2
            * (
                magnitude_tensor_hat
                - ratio**2 * tensor_magnitude_local(tensor_hat)[..., None] * tensor_hat
            )
        )
        return symmetric_dot_local(resolved, model), symmetric_dot_local(model, model)

    def scalar_contractions_local(
        momentum,
        scalar,
        filter_grid_ratio,
        ratio,
    ):
        tensor = strain_tensor_local(momentum)
        magnitude = tensor_magnitude_local(tensor)
        velocity = jnp.stack((momentum.u, momentum.v, momentum.w_at_cells), axis=-1)
        gradient = jnp.stack(
            (scalar.dtheta_dx, scalar.dtheta_dy, scalar.dtheta_dz_at_cells),
            axis=-1,
        )
        scalar_anomaly = scalar.theta - jnp.mean(
            scalar.theta,
            axis=(-2, -1),
            keepdims=True,
        )
        velocity_scalar = velocity * scalar_anomaly[..., None]
        width = filter_grid_ratio * ratio
        velocity_hat = lasd_filter_components_local(velocity, width)
        scalar_hat = lasd_filter_local(scalar_anomaly, width)
        velocity_scalar_hat = lasd_filter_components_local(velocity_scalar, width)
        gradient_hat = lasd_filter_components_local(gradient, width)
        strain_gradient_hat = lasd_filter_components_local(
            magnitude[..., None] * gradient,
            width,
        )
        tensor_hat = lasd_filter_components_local(tensor, width)
        resolved = velocity_scalar_hat - velocity_hat * scalar_hat[..., None]
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        model = delta**2 * (
            strain_gradient_hat
            - ratio**2 * tensor_magnitude_local(tensor_hat)[..., None] * gradient_hat
        )
        return jnp.sum(resolved * model, axis=-1), jnp.sum(model * model, axis=-1)

    def safe_divide_local(numerator, denominator):
        valid = jnp.abs(denominator) > 1.0e-30
        return jnp.where(
            valid,
            numerator / jnp.where(valid, denominator, 1.0),
            0.0,
        )

    def beta_local(coefficient_2d, coefficient_4d, test_ratio, scale_dependent):
        exponent = jnp.log(test_ratio) / (jnp.log(test_ratio**2) - jnp.log(test_ratio))
        raw = (
            jnp.maximum(safe_divide_local(coefficient_4d, coefficient_2d), 0.0)
            ** exponent
        )
        beta = jnp.maximum(raw, 1.0 / test_ratio**3)
        return jnp.where(scale_dependent, beta, jnp.ones_like(beta))

    def history_boundary_local(values):
        if values.shape[0] < 2:
            return values
        index = lax.axis_index(axis_name)
        values = values.at[0].set(jnp.where(index == 0, values[1], values[0]))
        return values.at[-1].set(
            jnp.where(index == shard_count - 1, values[-2], values[-1])
        )

    def departure_interpolate_local(
        values,
        lower_plane,
        upper_plane,
        trajectory_x,
        trajectory_y,
        trajectory_z,
        interval_dt,
    ):
        extended = jnp.concatenate(
            (lower_plane[None], values, upper_plane[None]),
            axis=0,
        )
        local_nz = values.shape[0]
        z_index = jnp.arange(local_nz, dtype=trajectory_x.dtype)[:, None, None]
        y_index = jnp.arange(grid.ny, dtype=trajectory_x.dtype)[None, :, None]
        x_index = jnp.arange(grid.nx, dtype=trajectory_x.dtype)[None, None, :]
        xi = jnp.mod(x_index - trajectory_x * interval_dt / grid.dx, grid.nx)
        eta = jnp.mod(y_index - trajectory_y * interval_dt / grid.dy, grid.ny)
        zeta = jnp.clip(
            z_index - trajectory_z * interval_dt / grid.dz,
            -1.0,
            float(local_nz),
        )
        i0 = jnp.floor(xi).astype(jnp.int32)
        j0 = jnp.floor(eta).astype(jnp.int32)
        k0 = jnp.floor(zeta).astype(jnp.int32) + 1
        i1 = (i0 + 1) % grid.nx
        j1 = (j0 + 1) % grid.ny
        k1 = jnp.minimum(k0 + 1, local_nz + 1)
        fx = xi - jnp.floor(xi)
        fy = eta - jnp.floor(eta)
        fz = zeta - jnp.floor(zeta)
        q000 = extended[k0, j0, i0]
        q100 = extended[k0, j0, i1]
        q010 = extended[k0, j1, i0]
        q110 = extended[k0, j1, i1]
        q001 = extended[k1, j0, i0]
        q101 = extended[k1, j0, i1]
        q011 = extended[k1, j1, i0]
        q111 = extended[k1, j1, i1]
        q00 = (1.0 - fx) * q000 + fx * q100
        q10 = (1.0 - fx) * q010 + fx * q110
        q01 = (1.0 - fx) * q001 + fx * q101
        q11 = (1.0 - fx) * q011 + fx * q111
        q0 = (1.0 - fy) * q00 + fy * q10
        q1 = (1.0 - fy) * q01 + fy * q11
        return (1.0 - fz) * q0 + fz * q1

    def lagrangian_average_local(
        current_a,
        current_b,
        old_a,
        old_b,
        lower_a,
        upper_a,
        lower_b,
        upper_b,
        trajectory_x,
        trajectory_y,
        trajectory_z,
        interval_dt,
        timescale_coefficient,
        timescale_a=None,
        timescale_b=None,
    ):
        scale_a = old_a if timescale_a is None else timescale_a
        scale_b = old_b if timescale_b is None else timescale_b
        product = scale_a * scale_b
        valid = (scale_a > 0.0) & (scale_b >= 0.0) & (product > 0.0)
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        timescale = (
            timescale_coefficient
            * delta
            * jnp.where(
                valid,
                product ** (-0.125),
                1.0,
            )
        )
        weight = jnp.where(
            valid,
            (interval_dt / timescale) / (1.0 + interval_dt / timescale),
            0.0,
        )
        departure_a = departure_interpolate_local(
            old_a,
            lower_a,
            upper_a,
            trajectory_x,
            trajectory_y,
            trajectory_z,
            interval_dt,
        )
        departure_b = departure_interpolate_local(
            old_b,
            lower_b,
            upper_b,
            trajectory_x,
            trajectory_y,
            trajectory_z,
            interval_dt,
        )
        return (
            weight * current_a + (1.0 - weight) * departure_a,
            jnp.maximum(weight * current_b + (1.0 - weight) * departure_b, 0.0),
        )

    def lasd_update_local(
        momentum,
        scalar,
        lm_old,
        mm_old,
        qn_old,
        nn_old,
        scalar_lm_old,
        scalar_mm_old,
        scalar_qn_old,
        scalar_nn_old,
        trajectory_x,
        trajectory_y,
        trajectory_z,
        first_update,
        interval_dt,
        filter_grid_ratio,
        test_ratio,
        timescale_coefficient,
        momentum_initial,
        momentum_minimum,
        momentum_maximum,
        momentum_scale_dependent,
        scalar_initial,
        scalar_minimum,
        scalar_maximum,
        scalar_scale_dependent,
    ):
        lm, mm = momentum_contractions_local(momentum, filter_grid_ratio, test_ratio)
        qn, nn = momentum_contractions_local(
            momentum,
            filter_grid_ratio,
            test_ratio**2,
        )
        scalar_lm, scalar_mm = scalar_contractions_local(
            momentum,
            scalar,
            filter_grid_ratio,
            test_ratio,
        )
        scalar_qn, scalar_nn = scalar_contractions_local(
            momentum,
            scalar,
            filter_grid_ratio,
            test_ratio**2,
        )
        histories = (
            jnp.where(first_update, momentum_initial * mm, lm_old),
            jnp.where(first_update, mm, mm_old),
            jnp.where(first_update, momentum_initial * nn, qn_old),
            jnp.where(first_update, nn, nn_old),
            jnp.where(first_update, scalar_initial * scalar_mm, scalar_lm_old),
            jnp.where(first_update, scalar_mm, scalar_mm_old),
            jnp.where(first_update, scalar_initial * scalar_nn, scalar_qn_old),
            jnp.where(first_update, scalar_nn, scalar_nn_old),
        )
        histories = tuple(history_boundary_local(value) for value in histories)
        history_halo = exchange_local(jnp.stack(histories, axis=0))
        lower = jnp.where(
            history_halo.lower_is_physical,
            jnp.stack([value[0] for value in histories]),
            history_halo.lower,
        )
        upper = jnp.where(
            history_halo.upper_is_physical,
            jnp.stack([value[-1] for value in histories]),
            history_halo.upper,
        )

        def average_pair(
            current_a, current_b, index, timescale_a=None, timescale_b=None
        ):
            return lagrangian_average_local(
                current_a,
                current_b,
                histories[index],
                histories[index + 1],
                lower[index],
                upper[index],
                lower[index + 1],
                upper[index + 1],
                trajectory_x,
                trajectory_y,
                trajectory_z,
                interval_dt,
                timescale_coefficient,
                timescale_a,
                timescale_b,
            )

        lm_avg, mm_avg = average_pair(lm, mm, 0)
        qn_avg, nn_avg = average_pair(qn, nn, 2)
        coefficient_2d = jnp.maximum(safe_divide_local(lm_avg, mm_avg), 0.0)
        coefficient_4d = jnp.maximum(safe_divide_local(qn_avg, nn_avg), 0.0)
        momentum_coefficient = jnp.clip(
            safe_divide_local(
                coefficient_2d,
                beta_local(
                    coefficient_2d,
                    coefficient_4d,
                    test_ratio,
                    momentum_scale_dependent,
                ),
            ),
            momentum_minimum,
            momentum_maximum,
        )
        scalar_lm_avg, scalar_mm_avg = average_pair(
            scalar_lm,
            scalar_mm,
            4,
            lm_avg,
            mm_avg,
        )
        scalar_qn_avg, scalar_nn_avg = average_pair(
            scalar_qn,
            scalar_nn,
            6,
            qn_avg,
            nn_avg,
        )
        scalar_lm_avg = jnp.where(scalar_lm_avg > 0.0, scalar_lm_avg, 1.0e-32)
        scalar_qn_avg = jnp.where(scalar_qn_avg > 0.0, scalar_qn_avg, 1.0e-32)
        scalar_2d = jnp.maximum(
            safe_divide_local(scalar_lm_avg, scalar_mm_avg),
            0.0,
        )
        scalar_4d = jnp.maximum(
            safe_divide_local(scalar_qn_avg, scalar_nn_avg),
            0.0,
        )
        scalar_coefficient = jnp.clip(
            safe_divide_local(
                scalar_2d,
                beta_local(
                    scalar_2d,
                    scalar_4d,
                    test_ratio,
                    scalar_scale_dependent,
                ),
            ),
            scalar_minimum,
            scalar_maximum,
        )
        return (
            momentum_coefficient,
            lm_avg,
            mm_avg,
            qn_avg,
            nn_avg,
            scalar_coefficient,
            scalar_lm_avg,
            scalar_mm_avg,
            scalar_qn_avg,
            scalar_nn_avg,
        )

    def horizontal_divergence_local(x_velocity, y_velocity):
        x_spectrum = jnp.fft.rfftn(x_velocity, axes=(-2, -1))
        y_spectrum = jnp.fft.rfftn(y_velocity, axes=(-2, -1))
        spectrum = (
            1j * kx[None, None, :] * x_spectrum + 1j * ky[None, :, None] * y_spectrum
        ) * keep[None, ...]
        return jnp.fft.irfftn(
            spectrum,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(x_velocity.dtype)

    def horizontal_gradient_local(pressure):
        spectrum = jnp.fft.rfftn(pressure, axes=(-2, -1)) * keep[None, ...]
        gradient_x = jnp.fft.irfftn(
            spectrum * (1j * kx[None, None, :]),
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(pressure.dtype)
        gradient_y = jnp.fft.irfftn(
            spectrum * (1j * ky[None, :, None]),
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(pressure.dtype)
        return gradient_x, gradient_y

    def filter_horizontal_local(values):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        return jnp.fft.irfftn(
            spectrum * keep[None, ...],
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def filter_boundary(boundary, dtype_probe):
        plane = jnp.broadcast_to(
            jnp.asarray(boundary, dtype=dtype_probe.dtype),
            (grid.ny, grid.nx),
        )
        spectrum = jnp.fft.rfftn(plane, axes=(-2, -1))
        return jnp.fft.irfftn(
            spectrum * keep,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(plane.dtype)

    def correct_local(candidate, gradient, dt):
        local_dt = jnp.asarray(dt, dtype=candidate.dtype)
        return candidate - local_dt * gradient

    def ab2_update_local(
        state,
        current_tendency,
        previous_tendency,
        dt,
        current_weight,
        previous_weight,
    ):
        local_dt = jnp.asarray(dt, dtype=state.dtype)
        current_coefficient = jnp.asarray(current_weight, dtype=state.dtype)
        previous_coefficient = jnp.asarray(previous_weight, dtype=state.dtype)
        return state + local_dt * (
            current_coefficient * current_tendency
            + previous_coefficient * previous_tendency
        )

    def wind_tunnel_local(
        u,
        v,
        w_upper,
        target_u,
        target_v,
        target_w_upper,
        disk_enabled,
        disk_x,
        disk_y,
        disk_z,
        disk_diameter,
        hub_diameter,
        thrust_coefficient_prime,
        normal_smoothing_width,
        transverse_smoothing_width,
        yaw_degrees,
        filtered_velocity_correction_enabled,
        fringe_enabled,
        fringe_start_x,
        fringe_relaxation_time,
    ):
        dtype = u.dtype
        local_nz = u.shape[0]
        shard_index = lax.axis_index(axis_name)
        x = (jnp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
        y = (jnp.arange(grid.ny, dtype=dtype) + 0.5) * grid.dy
        z_index = shard_index * local_nz + jnp.arange(local_nz, dtype=dtype)
        z = (z_index + 0.5) * grid.dz

        periodic_x = (
            jnp.mod(x - jnp.asarray(disk_x, dtype) + 0.5 * grid.lx, grid.lx)
            - 0.5 * grid.lx
        )
        periodic_y = (
            jnp.mod(y - jnp.asarray(disk_y, dtype) + 0.5 * grid.ly, grid.ly)
            - 0.5 * grid.ly
        )
        yaw = jnp.deg2rad(jnp.asarray(yaw_degrees, dtype))
        normal_x = jnp.cos(yaw)
        normal_y = jnp.sin(yaw)
        normal_distance = (
            periodic_x[None, None, :] * normal_x
            + periodic_y[None, :, None] * normal_y
        )
        in_plane = (
            -periodic_x[None, None, :] * normal_y
            + periodic_y[None, :, None] * normal_x
        )
        radius = jnp.sqrt(
            in_plane**2 + (z[:, None, None] - jnp.asarray(disk_z, dtype)) ** 2
        )
        streamwise = jnp.exp(
            -(
                normal_distance
                / jnp.asarray(normal_smoothing_width, dtype)
            )
            ** 2
        )
        radial = gaussian_convolved_annulus(
            radius,
            outer_radius=0.5 * jnp.asarray(disk_diameter, dtype),
            inner_radius=0.5 * jnp.asarray(hub_diameter, dtype),
            smoothing_width=jnp.asarray(transverse_smoothing_width, dtype),
        )
        disk_kernel = radial * streamwise
        disk_area = 0.25 * jnp.pi * (
            jnp.asarray(disk_diameter, dtype) ** 2
            - jnp.asarray(hub_diameter, dtype) ** 2
        )
        kernel_integral = (
            lax.psum(jnp.sum(disk_kernel), axis_name)
            * grid.dx
            * grid.dy
            * grid.dz
        )
        disk_kernel = disk_kernel * disk_area / jnp.maximum(
            kernel_integral,
            jnp.finfo(dtype).tiny,
        )
        normal_velocity = u * normal_x + v * normal_y
        numerator = lax.psum(jnp.sum(normal_velocity * disk_kernel), axis_name)
        denominator = lax.psum(jnp.sum(disk_kernel), axis_name)
        disk_velocity = numerator / jnp.maximum(denominator, jnp.finfo(dtype).tiny)
        velocity_correction = jnp.where(
            jnp.asarray(filtered_velocity_correction_enabled),
            filtered_disk_velocity_correction(
                thrust_coefficient_prime,
                outer_radius=0.5 * jnp.asarray(disk_diameter, dtype),
                inner_radius=0.5 * jnp.asarray(hub_diameter, dtype),
                smoothing_width=jnp.asarray(transverse_smoothing_width, dtype),
                dtype=dtype,
            ),
            1.0,
        )
        disk_velocity = velocity_correction * disk_velocity
        disk_acceleration = (
            -0.5
            * jnp.asarray(thrust_coefficient_prime, dtype)
            * disk_velocity
            * jnp.abs(disk_velocity)
            * disk_kernel
            * jnp.asarray(disk_enabled, dtype)
        )

        half_width = 0.5 * (grid.lx - jnp.asarray(fringe_start_x, dtype))

        def cinf_step(coordinate):
            epsilon = jnp.finfo(dtype).eps
            safe = jnp.clip(coordinate, epsilon, 1.0 - epsilon)
            interior = jax.nn.sigmoid(1.0 / (1.0 - safe) - 1.0 / safe)
            return jnp.where(
                coordinate <= 0.0,
                0.0,
                jnp.where(coordinate >= 1.0, 1.0, interior),
            )

        mask = cinf_step((x - fringe_start_x) / half_width) * cinf_step(
            (grid.lx - x) / half_width
        )
        rate = (
            mask
            / jnp.asarray(fringe_relaxation_time, dtype)
            * jnp.asarray(fringe_enabled, dtype)
        )
        source_u = disk_acceleration * normal_x + rate[None, None, :] * (
            target_u - u
        )
        source_v = disk_acceleration * normal_y + rate[None, None, :] * (
            target_v - v
        )
        source_w = rate[None, None, :] * (target_w_upper - w_upper)
        return source_u, source_v, source_w

    mapped = partial(jax.pmap, axis_name=axis_name, axis_size=shard_count)
    exchange_packed = mapped(exchange_local)
    pressure_gradient = mapped(pressure_gradient_local, in_axes=(0, None))
    divergence = mapped(divergence_local, in_axes=(0, None))
    enforce_upper_boundary = mapped(
        enforce_upper_boundary_local,
        in_axes=(0, None),
    )
    horizontal_divergence = mapped(
        horizontal_divergence_local,
        in_axes=(0, 0),
    )
    horizontal_gradient = mapped(horizontal_gradient_local)
    filter_horizontal = mapped(filter_horizontal_local)
    correct = mapped(correct_local, in_axes=(0, 0, None))
    ab2_update = mapped(
        ab2_update_local,
        in_axes=(0, 0, 0, None, None, None),
    )
    wind_tunnel = mapped(
        wind_tunnel_local,
        in_axes=(0, 0, 0, 0, 0, 0) + (None,) * 14,
    )
    dry_flow_context = mapped(
        dry_flow_context_local,
        in_axes=(0, 0, 0, None),
    )
    dry_advection = mapped(dry_advection_local)
    dry_wall = mapped(dry_wall_local, in_axes=(0, None, None, None))
    dry_sgs = mapped(dry_sgs_local, in_axes=(0, 0))
    dry_sgs_vertical_flux = mapped(
        dry_sgs_vertical_flux_local,
        in_axes=(0, 0),
    )
    lasd_accumulate = mapped(
        lasd_accumulate_local,
        in_axes=(0, 0, 0, 0, 0, 0, None),
    )
    lasd_update = mapped(
        lasd_update_local,
        in_axes=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    lasd_diagnostics = mapped(
        lasd_diagnostics_local,
        in_axes=(
            0,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    scalar_context = mapped(scalar_context_local)
    scalar_advection = mapped(scalar_advection_local, in_axes=(0, 0))
    scalar_sgs = mapped(
        scalar_sgs_local,
        in_axes=(0, 0, 0, None, None, None, None, None),
    )
    buoyancy = mapped(buoyancy_local, in_axes=(0, None))
    rayleigh_damping = mapped(
        rayleigh_damping_local,
        in_axes=(0, 0, 0, None, None, None, None),
    )
    return JaxZSlabInterpreter(
        decomposition,
        addressable_shards,
        exchange_packed,
        pressure_gradient,
        divergence,
        enforce_upper_boundary,
        horizontal_divergence,
        horizontal_gradient,
        filter_horizontal,
        jax.jit(filter_boundary),
        correct,
        ab2_update,
        wind_tunnel,
        dry_flow_context,
        dry_advection,
        dry_wall,
        dry_sgs,
        dry_sgs_vertical_flux,
        lasd_accumulate,
        lasd_update,
        lasd_diagnostics,
        scalar_context,
        scalar_advection,
        scalar_sgs,
        buoyancy,
        rayleigh_damping,
    )
