"""Momentum and wind-tunnel methods for the independent test oracle."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp

from jaxwind._jax.actuator_disk import (
    filtered_disk_velocity_correction,
    gaussian_convolved_annulus,
)
from jaxwind._jax.actuator_line import (
    actuator_line_deformed_kinematics,
    blade_element_kinematic_forces,
    gaussian_weights,
)
from jaxwind._jax.fringe import plateau_fringe_mask

from jaxwind.domain import (
    Cell,
    Evaluated,
    Field,
    Projected,
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
    BladeElementActuatorLine,
    ConcurrentPrecursorEnvironment,
    ConcurrentPrecursorFringe,
    NoActuatorDisk,
    NoActuatorLine,
    NoFringe,
    PureThrustActuatorDisk,
    WindTunnelModel,
)

from .jax_oracle_core import (
    OracleDryFlowContext,
    _cell_gradient_on_full_faces,
    _cell_to_full_faces,
    _horizontal_derivative,
    _oracle_tendency,
    _oracle_tendency_from_velocity,
    _require_tiny_global,
    _require_velocity_component,
    _strain_magnitude,
    _wall_filter,
)


class OracleFlowMixin:
    """Reference dry-flow and turbine-forcing interpretation."""

    __slots__ = ()

    def advection_tendency(
        self,
        context: OracleDryFlowContext,
        config: RotationalAdvection,
        wall: NeutralLogWall | FilteredNeutralLogWall | None = None,
    ) -> VelocityVector:
        if not isinstance(config, RotationalAdvection):
            raise TypeError("unsupported reference advection choice")
        velocity = context.velocity
        dudz = context.dudz_on_faces
        dvdz = context.dvdz_on_faces
        if wall is not None:
            wall_gradient = self._wall_gradient(context, wall)
            dudz = dudz.at[0].set(wall_gradient[0])
            dvdz = dvdz.at[0].set(wall_gradient[1])
        convection_x = context.velocity.y.payload * (context.dudy - context.dvdx)
        convection_x += 0.5 * (
            velocity.z.payload[1:] * (dudz[1:] - context.dwdx_on_faces[1:])
            + velocity.z.payload[:-1] * (dudz[:-1] - context.dwdx_on_faces[:-1])
        )
        convection_y = context.velocity.x.payload * (context.dvdx - context.dudy)
        convection_y += 0.5 * (
            velocity.z.payload[1:] * (dvdz[1:] - context.dwdy_on_faces[1:])
            + velocity.z.payload[:-1] * (dvdz[:-1] - context.dwdy_on_faces[:-1])
        )
        convection_z = context.u_on_faces * (
            context.dwdx_on_faces - dudz
        ) + context.v_on_faces * (context.dwdy_on_faces - dvdz)
        convection_z = convection_z.at[0].set(0.0).at[-1].set(0.0)
        return _oracle_tendency(
            context,
            -convection_x,
            -convection_y,
            -convection_z,
        )

    def pressure_gradient_tendency(
        self,
        context: OracleDryFlowContext,
        config: KinematicPressureGradient,
    ) -> VelocityVector:
        if not isinstance(config, KinematicPressureGradient):
            raise TypeError("unsupported pressure-gradient forcing choice")
        velocity = context.velocity
        x = jnp.full_like(velocity.x.payload, config.x_acceleration)
        y = jnp.full_like(velocity.y.payload, config.y_acceleration)
        z = jnp.zeros_like(velocity.z.payload)
        return _oracle_tendency(context, x, y, z)

    @staticmethod
    def _wall_gradient(
        context: OracleDryFlowContext,
        wall: NeutralLogWall | FilteredNeutralLogWall,
    ):
        velocity = context.velocity
        grid = velocity.x.ownership.grid
        wall_u = velocity.x.payload[0]
        wall_v = velocity.y.payload[0]
        if isinstance(wall, FilteredNeutralLogWall):
            filtered = _wall_filter(
                jnp.stack((wall_u, wall_v)),
                grid=grid,
                filter_width=wall.filter_grid_ratio * wall.test_filter_ratio,
            )
            wall_u, wall_v = filtered[0], filtered[1]
        elif not isinstance(wall, NeutralLogWall):
            raise TypeError("unsupported reference wall-gradient choice")
        speed = jnp.hypot(wall_u, wall_v)
        safe_speed = jnp.maximum(speed, jnp.finfo(speed.dtype).tiny)
        friction_velocity = (
            speed * wall.von_karman / math.log(0.5 * grid.dz / wall.roughness_length)
        )
        gradient = (
            jnp.stack((wall_u, wall_v))
            * friction_velocity
            / (safe_speed * wall.von_karman * 0.5 * grid.dz)
        )
        return jnp.where(
            (speed > jnp.finfo(speed.dtype).tiny)[None],
            gradient,
            0.0,
        )

    def wall_stress_tendency(
        self,
        context: OracleDryFlowContext,
        config: NeutralLogWall | FilteredNeutralLogWall,
    ) -> VelocityVector:
        if not isinstance(
            config,
            (NeutralLogWall, FilteredNeutralLogWall),
        ):
            raise TypeError("unsupported wall-stress choice")
        velocity = context.velocity
        grid = velocity.x.ownership.grid
        reference_height = 0.5 * grid.dz
        if config.roughness_length >= reference_height:
            raise ValueError("wall roughness must be below the first cell centre")
        drag = (
            config.von_karman / math.log(reference_height / config.roughness_length)
        ) ** 2
        u0 = velocity.x.payload[0]
        v0 = velocity.y.payload[0]
        if isinstance(config, FilteredNeutralLogWall):
            width = config.filter_grid_ratio * config.test_filter_ratio
            filtered = _wall_filter(
                jnp.stack((u0, v0)),
                grid=grid,
                filter_width=width,
            )
            u0, v0 = filtered[0], filtered[1]
        speed = jnp.hypot(u0, v0)
        wall_x = -drag * speed * u0 / grid.dz
        wall_y = -drag * speed * v0 / grid.dz
        x = jnp.zeros_like(velocity.x.payload).at[0].set(wall_x)
        y = jnp.zeros_like(velocity.y.payload).at[0].set(wall_y)
        z = jnp.zeros_like(velocity.z.payload)
        return _oracle_tendency(context, x, y, z)

    def sgs_tendency(
        self,
        context: OracleDryFlowContext,
        config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
        wall: NeutralLogWall | FilteredNeutralLogWall | None = None,
    ) -> VelocityVector:
        if not isinstance(
            config,
            (
                StaticSmagorinsky,
                LagrangianScaleDependentDynamic,
            ),
        ):
            raise TypeError("unsupported SGS choice")
        velocity = context.velocity
        grid = velocity.x.ownership.grid
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        magnitude = _strain_magnitude(
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
        coefficient = self._momentum_sgs_coefficient(context, config)
        if isinstance(config, LagrangianScaleDependentDynamic):
            coefficient = jnp.clip(
                coefficient,
                config.minimum_coefficient,
                config.maximum_coefficient,
            )
        eddy_viscosity = coefficient * delta**2 * magnitude
        txx = -2.0 * eddy_viscosity * context.dudx
        txy = -eddy_viscosity * (context.dudy + context.dvdx)
        tyy = -2.0 * eddy_viscosity * context.dvdy
        tzz = -2.0 * eddy_viscosity * context.dwdz
        txz, tyz = self.sgs_vertical_flux(context, config)
        x = -(
            _horizontal_derivative(txx, grid=grid, axis="x")
            + _horizontal_derivative(txy, grid=grid, axis="y")
            + (txz[1:] - txz[:-1]) / grid.dz
        )
        y = -(
            _horizontal_derivative(txy, grid=grid, axis="x")
            + _horizontal_derivative(tyy, grid=grid, axis="y")
            + (tyz[1:] - tyz[:-1]) / grid.dz
        )
        z = -(
            _horizontal_derivative(txz, grid=grid, axis="x")
            + _horizontal_derivative(tyz, grid=grid, axis="y")
            + _cell_gradient_on_full_faces(tzz, grid.dz)
        )
        z = z.at[0].set(0.0).at[-1].set(0.0)
        return _oracle_tendency(context, x, y, z)

    def sgs_vertical_flux(
        self,
        context: OracleDryFlowContext,
        config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
    ) -> tuple[Any, Any]:
        """Return filtered SGS xz and yz stresses on full vertical faces."""
        if not isinstance(
            config,
            (
                StaticSmagorinsky,
                LagrangianScaleDependentDynamic,
            ),
        ):
            raise TypeError("unsupported SGS choice")
        grid = context.velocity.x.ownership.grid
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        face_magnitude = _strain_magnitude(
            _cell_to_full_faces(context.dudx),
            _cell_to_full_faces(context.dudy),
            context.dudz_on_faces,
            _cell_to_full_faces(context.dvdx),
            _cell_to_full_faces(context.dvdy),
            context.dvdz_on_faces,
            context.dwdx_on_faces,
            context.dwdy_on_faces,
            _cell_to_full_faces(context.dwdz),
        )
        coefficient = self._momentum_sgs_coefficient(context, config)
        if isinstance(config, LagrangianScaleDependentDynamic):
            coefficient = jnp.clip(
                coefficient,
                config.minimum_coefficient,
                config.maximum_coefficient,
            )
        viscosity_on_faces = (
            _cell_to_full_faces(coefficient) * delta**2 * face_magnitude
        )
        txz = -viscosity_on_faces * (
            context.dudz_on_faces + context.dwdx_on_faces
        )
        tyz = -viscosity_on_faces * (
            context.dvdz_on_faces + context.dwdy_on_faces
        )
        txz = txz.at[0].set(0.0).at[-1].set(0.0)
        tyz = tyz.at[0].set(0.0).at[-1].set(0.0)
        return txz, tyz

    @staticmethod
    def _momentum_sgs_coefficient(
        context: OracleDryFlowContext,
        config: StaticSmagorinsky | LagrangianScaleDependentDynamic,
    ):
        if isinstance(config, StaticSmagorinsky):
            return jnp.full_like(context.dudx, config.coefficient**2)
        closure = context.closure
        if not isinstance(closure, LasdClosureMemory):
            raise TypeError("momentum LASD requires initialized closure memory")
        return closure.momentum.coefficient.payload

    def coriolis_geostrophic_tendency(
        self,
        context: OracleDryFlowContext,
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
                - horizontal_f * context.w_at_cells
            )
            y = -local_f * (velocity.x.payload - config.geostrophic_x_velocity)
            z = horizontal_f.astype(velocity.z.payload.dtype) * (
                context.u_on_faces - config.geostrophic_x_velocity
            )
            z = z.at[0].set(0.0).at[-1].set(0.0)
        else:
            raise TypeError("unsupported rotation choice")
        if isinstance(config, NoRotation):
            z = jnp.zeros_like(velocity.z.payload)
        return _oracle_tendency(
            context,
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
        """Evaluate turbine and concurrent-fringe forcing on a tiny grid."""
        if not isinstance(model, WindTunnelModel):
            raise TypeError("unsupported wind-tunnel model")
        x_ownership = _require_velocity_component(velocity.x, XVelocity)
        y_ownership = _require_velocity_component(velocity.y, YVelocity)
        z_ownership = _require_tiny_global(velocity.z, ZFace)
        if not (x_ownership.grid == y_ownership.grid == z_ownership.grid):
            raise ValueError("wind-tunnel velocity components must share ownership")
        grid = x_ownership.grid
        source_x = jnp.zeros_like(velocity.x.payload)
        source_y = jnp.zeros_like(velocity.y.payload)
        source_z = jnp.zeros_like(velocity.z.payload)

        disk = model.actuator_disk
        if isinstance(disk, PureThrustActuatorDisk):
            dtype = velocity.x.payload.dtype
            x = (jnp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
            y = (jnp.arange(grid.ny, dtype=dtype) + 0.5) * grid.dy
            z = (jnp.arange(grid.nz, dtype=dtype) + 0.5) * grid.dz
            dx = jnp.mod(x - disk.x + 0.5 * grid.lx, grid.lx) - 0.5 * grid.lx
            dy = jnp.mod(y - disk.y + 0.5 * grid.ly, grid.ly) - 0.5 * grid.ly
            yaw = jnp.deg2rad(jnp.asarray(disk.yaw_degrees, dtype=dtype))
            normal_x = jnp.cos(yaw)
            normal_y = jnp.sin(yaw)
            normal_distance = (
                dx[None, None, :] * normal_x + dy[None, :, None] * normal_y
            )
            in_plane = -dx[None, None, :] * normal_y + dy[None, :, None] * normal_x
            radius = jnp.sqrt(in_plane**2 + (z[:, None, None] - disk.z) ** 2)
            streamwise = jnp.exp(
                -((normal_distance / disk.normal_smoothing_width) ** 2)
            )
            radial = gaussian_convolved_annulus(
                radius,
                outer_radius=0.5 * disk.diameter,
                inner_radius=0.5 * disk.hub_diameter,
                smoothing_width=disk.transverse_smoothing_width,
            )
            kernel = radial * streamwise
            disk_area = 0.25 * jnp.pi * (disk.diameter**2 - disk.hub_diameter**2)
            kernel_integral = jnp.sum(kernel) * grid.dx * grid.dy * grid.dz
            kernel = (
                kernel
                * disk_area
                / jnp.maximum(
                    kernel_integral,
                    jnp.finfo(dtype).tiny,
                )
            )
            normal_velocity = (
                velocity.x.payload * normal_x + velocity.y.payload * normal_y
            )
            disk_velocity = jnp.sum(normal_velocity * kernel) / jnp.maximum(
                jnp.sum(kernel), jnp.finfo(dtype).tiny
            )
            correction = jnp.where(
                disk.filtered_velocity_correction,
                filtered_disk_velocity_correction(
                    disk.thrust_coefficient_prime,
                    outer_radius=0.5 * disk.diameter,
                    inner_radius=0.5 * disk.hub_diameter,
                    smoothing_width=disk.transverse_smoothing_width,
                    dtype=dtype,
                ),
                1.0,
            )
            disk_velocity = correction * disk_velocity
            prescribed = disk.prescribed_inflow_velocity > 0.0
            loading_velocity = (
                disk.prescribed_inflow_velocity if prescribed else disk_velocity
            )
            loading_coefficient = (
                disk.prescribed_thrust_coefficient
                if prescribed
                else disk.thrust_coefficient_prime
            )
            acceleration = (
                -0.5
                * loading_coefficient
                * loading_velocity
                * jnp.abs(loading_velocity)
                * kernel
            )
            source_x = source_x + acceleration * normal_x
            source_y = source_y + acceleration * normal_y
        elif not isinstance(disk, NoActuatorDisk):
            raise TypeError("unsupported actuator-disk choice")

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

            positions, tangents, blade_velocity, normal, _ = (
                actuator_line_deformed_kinematics(
                    x=line.x,
                    y=line.y,
                    z=line.z,
                    blade_count=line.blade_count,
                    element_radii=line.element_radii,
                    angular_velocity=line.angular_velocity,
                    time=time,
                    yaw_degrees=line.yaw_degrees,
                    tilt_degrees=line.tilt_degrees,
                    precone_degrees=line.precone_degrees,
                    initial_azimuth_degrees=line.initial_azimuth_degrees,
                    flap_displacements=deformation(line.element_flap_displacements),
                    edge_displacements=deformation(line.element_edge_displacements),
                    flap_slopes=deformation(line.element_flap_slopes),
                    edge_slopes=deformation(line.element_edge_slopes),
                    flap_velocities=deformation(line.element_flap_velocities),
                    edge_velocities=deformation(line.element_edge_velocities),
                    dtype=dtype,
                )
            )
            x = (jnp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
            y = (jnp.arange(grid.ny, dtype=dtype) + 0.5) * grid.dy
            z_cells = (jnp.arange(grid.nz, dtype=dtype) + 0.5) * grid.dz
            z_faces = jnp.arange(grid.nz + 1, dtype=dtype) * grid.dz
            weights_x = gaussian_weights(
                positions[:, 0],
                x,
                smoothing_width=line.smoothing_width,
                period=grid.lx,
            )
            weights_y = gaussian_weights(
                positions[:, 1],
                y,
                smoothing_width=line.smoothing_width,
                period=grid.ly,
            )
            weights_z_cells = gaussian_weights(
                positions[:, 2],
                z_cells,
                smoothing_width=line.smoothing_width,
            )
            weights_z_faces = gaussian_weights(
                positions[:, 2],
                z_faces,
                smoothing_width=line.smoothing_width,
            )
            sampled = jnp.stack(
                (
                    jnp.einsum(
                        "pz,py,px,zyx->p",
                        weights_z_cells,
                        weights_y,
                        weights_x,
                        velocity.x.payload,
                        optimize="optimal",
                    ),
                    jnp.einsum(
                        "pz,py,px,zyx->p",
                        weights_z_cells,
                        weights_y,
                        weights_x,
                        velocity.y.payload,
                        optimize="optimal",
                    ),
                    jnp.einsum(
                        "pz,py,px,zyx->p",
                        weights_z_faces,
                        weights_y,
                        weights_x,
                        velocity.z.payload,
                        optimize="optimal",
                    ),
                ),
                axis=1,
            )
            repeat = line.blade_count
            forces, _, _, _, _ = blade_element_kinematic_forces(
                sampled,
                tangents,
                jnp.zeros((point_count,), dtype=dtype),
                normal,
                element_radii=jnp.tile(
                    jnp.asarray(line.element_radii, dtype=dtype),
                    repeat,
                ),
                element_widths=jnp.tile(
                    jnp.asarray(line.element_widths, dtype=dtype),
                    repeat,
                ),
                element_chords=jnp.tile(
                    jnp.asarray(line.element_chords, dtype=dtype),
                    repeat,
                ),
                element_twist_degrees=jnp.tile(
                    jnp.asarray(line.element_twist_degrees, dtype=dtype),
                    repeat,
                ),
                element_airfoil_ids=jnp.tile(
                    jnp.asarray(line.element_airfoil_ids, dtype=jnp.int32),
                    repeat,
                ),
                blade_count=line.blade_count,
                hub_radius=line.hub_radius,
                tip_radius=line.tip_radius,
                pitch_degrees=line.pitch_degrees,
                polar_alpha_degrees=line.polar_alpha_degrees,
                polar_lift_coefficients=line.polar_lift_coefficients,
                polar_drag_coefficients=line.polar_drag_coefficients,
                tip_loss=line.tip_loss,
                root_loss=line.root_loss,
                blade_velocity=blade_velocity,
            )
            inverse_cell_volume = 1.0 / (grid.dx * grid.dy * grid.dz)
            source_x = source_x + inverse_cell_volume * jnp.einsum(
                "p,pz,py,px->zyx",
                forces[:, 0],
                weights_z_cells,
                weights_y,
                weights_x,
                optimize="optimal",
            )
            source_y = source_y + inverse_cell_volume * jnp.einsum(
                "p,pz,py,px->zyx",
                forces[:, 1],
                weights_z_cells,
                weights_y,
                weights_x,
                optimize="optimal",
            )
            source_z = source_z + inverse_cell_volume * jnp.einsum(
                "p,pz,py,px->zyx",
                forces[:, 2],
                weights_z_faces,
                weights_y,
                weights_x,
                optimize="optimal",
            )
            source_z = source_z.at[0].set(0.0).at[-1].set(0.0)
        elif not isinstance(line, NoActuatorLine):
            raise TypeError("unsupported actuator-line choice")

        fringe = model.fringe
        if isinstance(fringe, ConcurrentPrecursorFringe):
            if not isinstance(environment, ConcurrentPrecursorEnvironment):
                raise TypeError(
                    "concurrent fringe requires ConcurrentPrecursorEnvironment"
                )
            target = environment.velocity
            target_x = _require_velocity_component(target.x, XVelocity)
            target_y = _require_velocity_component(target.y, YVelocity)
            target_z = _require_tiny_global(target.z, ZFace)
            if not (
                target_x.grid == target_y.grid == target_z.grid == x_ownership.grid
            ):
                raise ValueError("precursor target must share main-domain ownership")
            dtype = velocity.x.payload.dtype
            x = (jnp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
            rise_width, fall_width = fringe.resolved_widths(grid.lx)
            mask = plateau_fringe_mask(
                x,
                start_x=fringe.start_x,
                end_x=grid.lx,
                rise_width=rise_width,
                fall_width=fall_width,
            )
            rate = mask / fringe.relaxation_time
            source_x = source_x + rate[None, None, :] * (
                target.x.payload - velocity.x.payload
            )
            source_y = source_y + rate[None, None, :] * (
                target.y.payload - velocity.y.payload
            )
            source_z = source_z + rate[None, None, :] * (
                target.z.payload - velocity.z.payload
            )
        elif not isinstance(fringe, NoFringe):
            raise TypeError("unsupported wind-tunnel fringe choice")

        return _oracle_tendency_from_velocity(
            velocity,
            source_x,
            source_y,
            source_z,
        )

    def combine_tendencies(
        self,
        tendencies: tuple[VelocityVector, ...],
    ) -> VelocityVector:
        if not tendencies:
            raise ValueError("at least one evaluated tendency is required")
        first = tendencies[0]
        expected_components = (
            (first.x, XVelocityTendency, Cell),
            (first.y, YVelocityTendency, Cell),
            (first.z, VerticalVelocityTendency, ZFace),
        )
        for tendency in tendencies:
            for component, expected in zip(
                (tendency.x, tendency.y, tendency.z),
                expected_components,
                strict=True,
            ):
                first_component, quantity, location = expected
                _require_tiny_global(component, location)
                if (
                    component.quantity is not quantity
                    or component.phase is not Evaluated
                ):
                    raise TypeError(
                        "only evaluated velocity tendencies may be combined"
                    )
                if component.ownership != first_component.ownership:
                    raise ValueError("combined tendencies must share one ownership")
                if component.payload.dtype != first_component.payload.dtype:
                    raise TypeError("combined tendencies must share one dtype")
        velocity = VelocityVector(
            Field(
                XVelocity,
                Cell,
                first.x.ownership,
                Projected,
                jnp.zeros_like(first.x.payload),
            ),
            Field(
                YVelocity,
                Cell,
                first.y.ownership,
                Projected,
                jnp.zeros_like(first.y.payload),
            ),
            Field(
                VerticalVelocity,
                ZFace,
                first.z.ownership,
                Projected,
                jnp.zeros_like(first.z.payload),
            ),
        )
        return _oracle_tendency_from_velocity(
            velocity,
            sum(
                (term.x.payload for term in tendencies), jnp.zeros_like(first.x.payload)
            ),
            sum(
                (term.y.payload for term in tendencies), jnp.zeros_like(first.y.payload)
            ),
            sum(
                (term.z.payload for term in tendencies), jnp.zeros_like(first.z.payload)
            ),
        )
