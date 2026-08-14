"""Private JAX spatial discretization with transient vertical halos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax.numpy as jnp

from jaxwind.domain import (
    Accepted,
    AddressableField,
    Cell,
    Candidate,
    Divergence,
    EqualVerticalPartition,
    Evaluated,
    PressureCorrection,
    PressureRhs,
    PassiveScalarConcentration,
    PassiveScalarTendency,
    PotentialTemperaturePerturbation,
    PotentialTemperatureTendency,
    Projected,
    VerticalBoundary,
    VerticalFaceField,
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
from jaxwind.operators import PressureGradient, VelocityVector
from jaxwind.physics.boussinesq import BoussinesqFields
from jaxwind.physics.wind_tunnel import BladeElementActuatorLine

from .flow import ZSlabFlowMixin
from .lasd import ZSlabLasdMixin


class PackedHaloArrays(NamedTuple):
    """Transient packed neighbor planes returned inside the SPMD program."""

    lower: Any
    upper: Any
    lower_is_physical: Any
    upper_is_physical: Any


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
class ZSlabProjectionPreparation:
    """Filtered horizontal spectra retained until pressure correction."""

    x_spectrum: Any
    y_spectrum: Any
    z_payload: Any
    lower_boundary: Any


class ActuatorLineDiagnostic(NamedTuple):
    """Replicated per-element aerodynamic data from a distributed evaluation."""

    force_on_fluid_per_density: Any
    positions: Any
    tangents: Any
    normals: Any
    span_directions: Any
    blade_velocity: Any
    sampled_velocity: Any
    alpha_degrees: Any
    lift_coefficients: Any
    drag_coefficients: Any
    loss_factors: Any


@dataclass(frozen=True, slots=True)
class _ProjectionKernels:
    exchange_packed: Callable
    pressure_gradient: Callable
    divergence: Callable
    enforce_upper_boundary: Callable
    prepare: Callable
    finish: Callable
    horizontal_divergence: Callable
    horizontal_gradient: Callable
    filter_horizontal: Callable
    filter_boundary: Callable
    correct: Callable


@dataclass(frozen=True, slots=True)
class _FlowKernels:
    ab2_update: Callable
    ab2_update_velocity: Callable
    combine_payloads: Callable
    context: Callable
    advection: Callable
    rotational_advection: Callable
    wall: Callable
    sgs: Callable
    sgs_vertical_flux: Callable
    sgs_tke_transfer: Callable
    fused_boussinesq: Callable
    fused_boussinesq_from_context: Callable
    fused_rotational_boussinesq: Callable
    fused_rotational_boussinesq_from_context: Callable


@dataclass(frozen=True, slots=True)
class _LasdKernels:
    relax_field: Callable
    accumulate: Callable
    accumulate_velocity: Callable
    update: Callable
    update_momentum: Callable
    diagnostics: Callable


@dataclass(frozen=True, slots=True)
class _ScalarKernels:
    context: Callable
    advection: Callable
    sgs: Callable
    buoyancy: Callable
    rayleigh_damping: Callable
    surface_transfer: Callable


@dataclass(frozen=True, slots=True)
class _WindKernels:
    tendency: Callable
    actuator_line: Callable


@dataclass(frozen=True, slots=True)
class _JaxDiscretization(ZSlabLasdMixin, ZSlabFlowMixin):
    """JAX lowering over the currently selected private domain partition."""

    decomposition: EqualVerticalPartition
    addressable_partitions: tuple[int, ...]
    frozen_zero_scalar: bool
    projection: _ProjectionKernels
    flow: _FlowKernels
    lasd: _LasdKernels
    scalar: _ScalarKernels
    wind: _WindKernels

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

    def actuator_line_diagnostic(
        self,
        velocity: VelocityVector,
        line: BladeElementActuatorLine,
        *,
        time: float,
    ) -> ActuatorLineDiagnostic:
        """Evaluate moving-line loads and return one replicated shard copy."""

        if not isinstance(line, BladeElementActuatorLine):
            raise TypeError("actuator-line diagnostics require a blade line")
        dtype = velocity.x.payload.dtype
        point_count = line.blade_count * len(line.element_radii)

        def deformation(values):
            return (
                jnp.asarray(values, dtype=dtype)
                if values
                else jnp.zeros((point_count,), dtype=dtype)
            )

        values = self.wind.actuator_line(
            velocity.x.payload,
            velocity.y.payload,
            velocity.z.owned.payload,
            velocity.z.lower_boundary,
            time,
            line.x,
            line.y,
            line.z,
            line.blade_count,
            line.hub_radius,
            line.tip_radius,
            line.angular_velocity,
            line.smoothing_width,
            jnp.asarray(line.element_radii, dtype=dtype),
            jnp.asarray(line.element_widths, dtype=dtype),
            jnp.asarray(line.element_chords, dtype=dtype),
            jnp.asarray(line.element_twist_degrees, dtype=dtype),
            jnp.asarray(line.element_airfoil_ids, dtype=jnp.int32),
            jnp.asarray(line.polar_alpha_degrees, dtype=dtype),
            jnp.asarray(line.polar_lift_coefficients, dtype=dtype),
            jnp.asarray(line.polar_drag_coefficients, dtype=dtype),
            line.pitch_degrees,
            line.yaw_degrees,
            line.tilt_degrees,
            line.precone_degrees,
            line.initial_azimuth_degrees,
            line.tip_loss,
            line.root_loss,
            deformation(line.element_flap_displacements),
            deformation(line.element_edge_displacements),
            deformation(line.element_flap_slopes),
            deformation(line.element_edge_slopes),
            deformation(line.element_flap_velocities),
            deformation(line.element_edge_velocities),
        )
        replicated = tuple(value[0] for value in values[3:])
        return ActuatorLineDiagnostic(*replicated)

    def halo_communicated_elements(
        self,
        component_count: int,
        partition_index: int,
    ) -> int:
        """Network payload derived from the actual non-physical neighbors."""
        if component_count <= 0:
            raise ValueError("packed halo component count must be positive")
        if not 0 <= partition_index < self.decomposition.partition_count:
            raise ValueError("shard index is outside the global z mesh")
        neighbors = int(partition_index > 0) + int(
            partition_index < self.decomposition.partition_count - 1
        )
        plane = self.decomposition.grid.ny * self.decomposition.grid.nx
        return neighbors * component_count * plane

    def _expected_regions(self, location: type) -> tuple:
        all_regions = self.decomposition.regions(location)
        return tuple(all_regions[index] for index in self.addressable_partitions)

    def _validate_field(self, field: AddressableField, location: type) -> None:
        if field.location is not location:
            raise TypeError(
                f"distributed operator requires {location.__name__} input"
            )
        if field.regions != self._expected_regions(location):
            raise ValueError("addressable regions do not match solver ownership")

    def pressure_gradient_z(
        self,
        pressure: AddressableField,
        boundary_gradient: VerticalBoundary[Any],
    ) -> VerticalFaceField:
        """Apply the stored-upper-face interpretation of ``G_z``."""
        self._validate_field(pressure, Cell)
        if pressure.quantity is not PressureCorrection:
            raise TypeError("pressure_gradient_z requires PressureCorrection")
        payload = self.projection.pressure_gradient(
            pressure.payload,
            boundary_gradient.upper,
        )
        owned = AddressableField(
            VerticalPressureGradient,
            ZFace,
            self._expected_regions(ZFace),
            Evaluated,
            payload,
        )
        return VerticalFaceField(owned, boundary_gradient.lower)

    def divergence_z(self, vertical_faces: VerticalFaceField) -> AddressableField:
        """Apply ``D_z`` using the owned upper faces and lower boundary face."""
        self._validate_field(vertical_faces.owned, ZFace)
        if vertical_faces.owned.quantity not in (
            VerticalVelocity,
            VerticalPressureGradient,
        ):
            raise TypeError("divergence_z requires a vertical face-normal quantity")
        payload = self.projection.divergence(
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
        x_payload, y_payload, z_payload = self.projection.filter_horizontal(
            velocity.x.payload,
            velocity.y.payload,
            velocity.z.owned.payload,
        )
        boundary_dtype = velocity.z.owned.payload.dtype
        boundary_shape = (
            self.decomposition.grid.ny,
            self.decomposition.grid.nx,
        )

        def filtered_boundary(value):
            array = jnp.asarray(value, dtype=boundary_dtype)
            if array.ndim == 0:
                return jnp.broadcast_to(array, boundary_shape)
            return self.projection.filter_boundary(array)

        lower_boundary = filtered_boundary(boundary.lower)
        upper_payload = self.projection.enforce_upper_boundary(
            z_payload,
            filtered_boundary(boundary.upper),
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
            VerticalFaceField(
                owned,
                lower_boundary,
            ),
        )

    def prepare_projection(
        self,
        velocity: VelocityVector,
        boundary: VerticalBoundary[Any],
    ) -> tuple[ZSlabProjectionPreparation, AddressableField]:
        """Filter the candidate, retaining spectra through pressure solve."""
        self._validate_velocity_cell(velocity.x, XVelocity)
        self._validate_velocity_cell(velocity.y, YVelocity)
        self._validate_field(velocity.z.owned, ZFace)
        if velocity.z.owned.quantity is not VerticalVelocity:
            raise TypeError("vertical velocity requires VerticalVelocity")
        if velocity.z.owned.phase not in (Candidate, Projected):
            raise TypeError("vertical velocity must be Candidate or Projected")

        boundary_dtype = velocity.z.owned.payload.dtype
        boundary_shape = (
            self.decomposition.grid.ny,
            self.decomposition.grid.nx,
        )

        def filtered_boundary(value):
            array = jnp.asarray(value, dtype=boundary_dtype)
            if array.ndim == 0:
                return jnp.broadcast_to(array, boundary_shape)
            return self.projection.filter_boundary(array)

        lower_boundary = filtered_boundary(boundary.lower)
        x_spectrum, y_spectrum, z_payload, divergence_payload = (
            self.projection.prepare(
                velocity.x.payload,
                velocity.y.payload,
                velocity.z.owned.payload,
                lower_boundary,
                filtered_boundary(boundary.upper),
            )
        )
        prepared = ZSlabProjectionPreparation(
            x_spectrum,
            y_spectrum,
            z_payload,
            lower_boundary,
        )
        divergence = AddressableField(
            Divergence,
            Cell,
            self._expected_regions(Cell),
            Evaluated,
            divergence_payload,
        )
        return prepared, divergence

    def finish_projection(
        self,
        prepared: ZSlabProjectionPreparation,
        pressure: AddressableField,
        dt: float,
    ) -> VelocityVector:
        """Correct retained candidate spectra and materialize the velocity."""
        if not isinstance(prepared, ZSlabProjectionPreparation):
            raise TypeError("finish_projection requires a prepared JAX candidate")
        self._validate_field(pressure, Cell)
        if pressure.quantity is not PressureCorrection:
            raise TypeError("finish_projection requires PressureCorrection")
        x_payload, y_payload, z_payload = self.projection.finish(
            prepared.x_spectrum,
            prepared.y_spectrum,
            prepared.z_payload,
            pressure.payload,
            dt,
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
            VerticalFaceField(
                AddressableField(
                    VerticalVelocity,
                    ZFace,
                    self._expected_regions(ZFace),
                    Projected,
                    z_payload,
                ),
                prepared.lower_boundary,
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
            VerticalFaceField(
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
        arrays = self.flow.context(
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
            self.scalar.context(scalar.payload),
        )

    def boussinesq_context_from_momentum(
        self,
        fields: BoussinesqFields,
        momentum: ZSlabDryFlowContext,
    ) -> ZSlabBoussinesqContext:
        """Attach the scalar context without rebuilding momentum derivatives."""

        scalar = fields.potential_temperature
        self._validate_field(scalar, Cell)
        if scalar.quantity not in (
            PotentialTemperaturePerturbation,
            PassiveScalarConcentration,
        ):
            raise TypeError("Boussinesq context requires a supported scalar quantity")
        if scalar.phase is not Accepted:
            raise TypeError("Boussinesq context requires accepted scalar state")
        return ZSlabBoussinesqContext(
            ZSlabDryFlowContext(
                momentum.velocity,
                momentum.arrays,
                fields.closure,
            ),
            scalar,
            self.scalar.context(scalar.payload),
        )

    def velocity_divergence(self, velocity: VelocityVector) -> AddressableField:
        """Apply the compatible hybrid ``D`` to one owned velocity product."""
        self._validate_velocity_cell(velocity.x, XVelocity)
        self._validate_velocity_cell(velocity.y, YVelocity)
        vertical = self.divergence_z(velocity.z)
        horizontal = self.projection.horizontal_divergence(
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
        gradient_x, gradient_y = self.projection.horizontal_gradient(pressure.payload)
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
        x_payload, y_payload, z_payload, lower_boundaries = self.projection.correct(
            velocity.x.payload,
            velocity.y.payload,
            velocity.z.owned.payload,
            gradient.x.payload,
            gradient.y.payload,
            gradient.z.owned.payload,
            velocity.z.lower_boundary,
            gradient.z.lower_boundary,
            dt,
        )
        lower_boundary = lower_boundaries[0]
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
            VerticalFaceField(
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

        x_payload, y_payload, z_payload = self.flow.ab2_update_velocity(
            velocity.x.payload,
            velocity.y.payload,
            velocity.z.owned.payload,
            current_tendency.x.payload,
            current_tendency.y.payload,
            current_tendency.z.owned.payload,
            previous_tendency.x.payload,
            previous_tendency.y.payload,
            previous_tendency.z.owned.payload,
            dt,
            current_weight,
            previous_weight,
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
            VerticalFaceField(
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
            self.flow.ab2_update(
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


def build_discretization(
    decomposition: EqualVerticalPartition,
    *,
    addressable_partitions: tuple[int, ...] | None = None,
    axis_name: str = "jaxwind_z",
    porte_agel_wall_correction: bool = True,
    nonlinear_padding_ratio: float = 1.5,
    nonlinear_dealiasing: str = "three_halves",
    frozen_zero_scalar: bool = False,
    lasd_filter_backend: str = "jax",
) -> _JaxDiscretization:
    """Build the private production spatial discretization.

    A one-partition decomposition is the ordinary single-process case and
    defaults to its only addressable partition. Larger decompositions use the
    same lowering and require the caller's addressable global partition ids.
    ``frozen_zero_scalar`` is reserved for runs whose passive scalar and scalar
    boundary fluxes remain identically zero for the full integration.
    """

    from .factory import (
        build_discretization as _build_discretization,
    )

    return _build_discretization(
        decomposition,
        addressable_partitions=addressable_partitions,
        axis_name=axis_name,
        porte_agel_wall_correction=porte_agel_wall_correction,
        nonlinear_padding_ratio=nonlinear_padding_ratio,
        nonlinear_dealiasing=nonlinear_dealiasing,
        frozen_zero_scalar=frozen_zero_scalar,
        lasd_filter_backend=lasd_filter_backend,
    )
