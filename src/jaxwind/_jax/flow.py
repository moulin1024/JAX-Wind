"""Momentum and wind-tunnel methods for the private JAX discretization."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from jaxwind.domain import (
    Evaluated,
    VerticalVelocity,
    VerticalVelocityTendency,
    XVelocity,
    XVelocityTendency,
    YVelocity,
    YVelocityTendency,
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
from jaxwind.physics.lasd import (
    LagrangianScaleDependentDynamic,
    LasdClosureMemory,
)
from jaxwind.physics.wind_tunnel import (
    BladeElementActuatorDisk,
    BladeElementActuatorLine,
    ConcurrentPrecursorEnvironment,
    ConcurrentPrecursorFringe,
    NacelleTowerDrag,
    NoActuatorDisk,
    NoActuatorLine,
    NoFringe,
    PureThrustActuatorDisk,
    WindTunnelModel,
)

if TYPE_CHECKING:
    from .discretization import ZSlabDryFlowContext


class ZSlabFlowMixin:
    """Distributed dry-flow and turbine-forcing interpretation."""

    __slots__ = ()

    def advection_tendency(
        self,
        context: ZSlabDryFlowContext,
        config: RotationalAdvection,
        wall: NeutralLogWall | FilteredNeutralLogWall | None = None,
    ) -> VelocityVector:
        if not isinstance(config, RotationalAdvection):
            raise TypeError("only legacy rotational advection is supported")
        x, y, z = self.flow.rotational_advection(
            context.arrays,
            *self._wall_gradient_parameters(wall),
        )
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
            config.filter_grid_ratio * config.test_filter_ratio if filtered else 1.0
        )
        x, y, z = self.flow.wall(
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
        wall: NeutralLogWall | FilteredNeutralLogWall | None = None,
    ) -> VelocityVector:
        if not isinstance(
            config,
            (StaticSmagorinsky, LagrangianScaleDependentDynamic),
        ):
            raise TypeError("unsupported SGS choice")
        coefficient = self._momentum_sgs_coefficient(context, config)
        bounds = self._momentum_sgs_coefficient_bounds(config)
        x, y, z = self.flow.sgs(context.arrays, coefficient, *bounds)
        return self._dry_tendency(x, y, z)

    def sgs_vertical_flux(
        self,
        context: ZSlabDryFlowContext,
        config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
    ) -> tuple[Any, Any]:
        """Return filtered addressable SGS xz and yz upper-face stresses."""
        if not isinstance(
            config,
            (StaticSmagorinsky, LagrangianScaleDependentDynamic),
        ):
            raise TypeError("unsupported SGS choice")
        return self.flow.sgs_vertical_flux(
            context.arrays,
            self._momentum_sgs_coefficient(context, config),
            *self._momentum_sgs_coefficient_bounds(config),
        )

    def momentum_sgs_tke_transfer(
        self,
        context: ZSlabDryFlowContext,
        config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
        *,
        wall: NeutralLogWall | FilteredNeutralLogWall | None = None,
    ):
        """Return signed SGS transfer from resolved TKE at owned cell centres.

        Forward transfer to unresolved scales is negative. Nonlinear products
        use the legacy base grid and the first cell uses the configured log wall.
        """
        wall_gradient_factor = self._diagnostic_wall_gradient_factor(wall)
        if isinstance(config, (StaticSmagorinsky, LagrangianScaleDependentDynamic)):
            return self.flow.sgs_tke_transfer(
                context.arrays,
                self._momentum_sgs_coefficient(context, config),
                *self._momentum_sgs_coefficient_bounds(config),
                wall_gradient_factor,
            )
        raise TypeError("unsupported SGS transfer diagnostic")

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

    @staticmethod
    def _momentum_sgs_coefficient_bounds(
        config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
    ) -> tuple[float, float]:
        if isinstance(config, StaticSmagorinsky):
            return 0.0, math.inf
        return config.minimum_coefficient, config.maximum_coefficient

    def _wall_gradient_parameters(
        self,
        wall: NeutralLogWall | FilteredNeutralLogWall | None,
    ) -> tuple[bool, float, float, bool, float]:
        if wall is None:
            return False, 0.25 * self.decomposition.grid.dz, 0.4, False, 1.0
        if not isinstance(wall, (NeutralLogWall, FilteredNeutralLogWall)):
            raise TypeError("wall-gradient choice is unsupported")
        filtered = isinstance(wall, FilteredNeutralLogWall)
        width = wall.filter_grid_ratio * wall.test_filter_ratio if filtered else 1.0
        return True, wall.roughness_length, wall.von_karman, filtered, width

    def _diagnostic_wall_gradient_factor(
        self,
        wall: NeutralLogWall | FilteredNeutralLogWall | None,
    ) -> float:
        if wall is None:
            return 0.0
        if not isinstance(wall, (NeutralLogWall, FilteredNeutralLogWall)):
            raise TypeError("diagnostic wall-gradient choice is unsupported")
        reference_height = 0.5 * self.decomposition.grid.dz
        if wall.roughness_length >= reference_height:
            raise ValueError("wall roughness must be below the first cell centre")
        return 1.0 / (
            math.log(reference_height / wall.roughness_length) * reference_height
        )

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
        evaluation_time: Any | None = None,
    ) -> VelocityVector:
        """Evaluate distributed turbine and precursor-fringe forcing."""
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
                disk.prescribed_inflow_velocity,
                disk.prescribed_thrust_coefficient,
            )
        elif isinstance(disk, (NoActuatorDisk, BladeElementActuatorDisk)):
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
                0.0,
                0.0,
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
            rise_width, fall_width = fringe.resolved_widths(self.decomposition.grid.lx)
            fringe_parameters = (
                True,
                fringe.start_x,
                fringe.relaxation_time,
                rise_width,
                fall_width,
            )
        elif isinstance(fringe, NoFringe):
            target = velocity
            fringe_parameters = (False, 0.0, 1.0, 1.0, 1.0)
        else:
            raise TypeError("unsupported wind-tunnel fringe choice")

        x, y, z = self.wind.tendency(
            velocity.x.payload,
            velocity.y.payload,
            velocity.z.owned.payload,
            target.x.payload,
            target.y.payload,
            target.z.owned.payload,
            *disk_parameters,
            *fringe_parameters,
        )
        if isinstance(disk, BladeElementActuatorDisk):
            dtype = velocity.x.payload.dtype
            disk_values = self.wind.actuator_disk_bem(
                velocity.x.payload,
                velocity.y.payload,
                velocity.z.owned.payload,
                disk.x,
                disk.y,
                disk.z,
                disk.blade_count,
                disk.hub_radius,
                disk.tip_radius,
                disk.angular_velocity,
                jnp.asarray(disk.element_smoothing_widths, dtype=dtype),
                jnp.asarray(disk.element_radii, dtype=dtype),
                jnp.asarray(disk.element_widths, dtype=dtype),
                jnp.asarray(disk.element_chords, dtype=dtype),
                jnp.asarray(disk.element_twist_degrees, dtype=dtype),
                jnp.asarray(disk.element_airfoil_ids, dtype=jnp.int32),
                jnp.asarray(disk.polar_alpha_degrees, dtype=dtype),
                jnp.asarray(disk.polar_lift_coefficients, dtype=dtype),
                jnp.asarray(disk.polar_drag_coefficients, dtype=dtype),
                disk.pitch_degrees,
                disk.tip_loss,
                disk.root_loss,
            )
            x = x + disk_values[0]
            y = y + disk_values[1]
            z = z + disk_values[2]
        body = model.turbine_body
        if isinstance(body, NacelleTowerDrag):
            body_values = self.wind.nacelle_tower(
                velocity.x.payload,
                velocity.y.payload,
                body.x,
                body.y,
                body.hub_height,
                body.nacelle_length,
                body.nacelle_diameter,
                body.nacelle_drag_coefficient,
                body.tower_base_diameter,
                body.tower_top_diameter,
                body.tower_drag_coefficient,
                body.smoothing_width,
            )
            x = x + body_values[0]
            y = y + body_values[1]
            z = z + body_values[2]
        line = model.actuator_line
        if isinstance(line, BladeElementActuatorLine):
            dtype = velocity.x.payload.dtype
            time = 0.0 if evaluation_time is None else evaluation_time.time
            point_count = line.blade_count * len(line.element_radii)

            def deformation(values):
                return (
                    jnp.asarray(values, dtype=dtype)
                    if values
                    else jnp.zeros((point_count,), dtype=dtype)
                )

            line_values = self.wind.actuator_line(
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
            line_x, line_y, line_z = line_values[:3]
            x = x + line_x
            y = y + line_y
            z = z + line_z
        elif not isinstance(line, NoActuatorLine):
            raise TypeError("unsupported actuator-line choice")
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
            self.flow.combine_payloads(tuple(term.x.payload for term in tendencies)),
            self.flow.combine_payloads(tuple(term.y.payload for term in tendencies)),
            self.flow.combine_payloads(
                tuple(term.z.owned.payload for term in tendencies)
            ),
            self.flow.combine_payloads(
                tuple(term.z.lower_boundary for term in tendencies)
            ),
        )
