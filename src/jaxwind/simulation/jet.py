"""Cryogenic inlet, transport, and integrator assembly."""
from __future__ import annotations
import math
import numpy as np
from jaxwind.config.jet import JetCase, SMAGORINSKY_COEFFICIENT
from jaxwind.formulations.jet import (CryogenicState, DifferentiableInletControl,
    _add_velocity, _cell_vector_to_faces, _enforce_scalar, _conservative_nonnegative)

def build_simulation(case: JetCase, *, differentiable_inlet: bool = False):
    import jax
    import jax.numpy as jnp
    from jaxwind.simulation.jet_adaptive import AdaptiveCryogenicState, build_adaptive_advance

    if case.cfl is not None and differentiable_inlet:
        raise ValueError("adaptive step rejection is not a differentiable inlet mode")

    from jaxwind.domain import (
        AnalyticalGrid,
        SinhMapping,
        TanhMapping,
        UniformGrid,
    )
    from jaxwind import (
        AnisotropicMinimumDissipation,
        StaticSmagorinsky,
        FREE_SLIP,
        IdealGasMixture,
        OPEN,
        Boundaries,
        FlowModel,
        InflowPlane,
        MoninObukhovWall,
        PassiveScalar,
        StaggeredVelocity,
        Wall,
        build_pressure_poisson,
        build_tendency,
        cell_velocity,
        conservative_specific_tendency,
        courant_number,
        eddy_viscosity,
        enforce_open_velocity,
        face_density,
        continuity_residual,
        dilatation_correction,
        divergence,
        pressure_gradient,
        project,
        project_low_mach,
        scalar_tendency,
        zeros,
    )
    from jaxwind.cryogenic import (
        LN2InletControl,
        LN2Jet,
        ParcelExchange,
        advance_ln2_parcels,
        initial_ln2_parcels,
        vapor_nozzle_source,
    )
    from jaxwind.physics.cryogenic import (
        CryogenicMicrophysicsConfig,
        advance_fog_microphysics,
    )
    from jaxwind.open_boundary import (
        enforce_two_outlet_scalar, enforce_two_outlet_velocity,
    )

    two_outlets = case.streamwise_boundaries == "outflow-outflow"

    def enforce_velocity(velocity, plane, grid):
        if two_outlets:
            return enforce_two_outlet_velocity(velocity)
        return enforce_open_velocity(velocity, plane, grid)

    def axis_mapping(kind, focus, strength, length):
        normalized_focus = focus / length
        if kind == "sinh":
            return SinhMapping(normalized_focus, strength)
        # TanhMapping(0) is also the identity used for an unstretched axis.
        return TanhMapping(strength, normalized_focus)

    mappings = tuple(
        axis_mapping(kind, focus, strength, length)
        for kind, focus, strength, length in zip(
            case.mapping_types,
            case.mapping_focus,
            case.mapping_strength,
            case.lengths,
        )
    )
    grid = (
        UniformGrid(*case.cells, *case.lengths)
        if all(
            kind == "uniform" or strength == 0.0
            for kind, strength in zip(
                case.mapping_types, case.mapping_strength
            )
        )
        else AnalyticalGrid(*case.cells, *case.lengths, *mappings)
    )
    boundaries = Boundaries(
        Wall(FREE_SLIP),
        Wall(FREE_SLIP),
        streamwise=OPEN,
        spanwise=FREE_SLIP,
    )
    wall = MoninObukhovWall(case.roughness)
    closure = (
        AnisotropicMinimumDissipation()
        if case.momentum_closure == "amd"
        else StaticSmagorinsky(SMAGORINSKY_COEFFICIENT)
    )
    flow_model = FlowModel(
        viscosity=case.kinematic_viscosity,
        subfilter=closure,
        surface=wall,
        sidewalls=wall,
    )
    momentum_rhs = build_tendency(grid, boundaries, flow_model)
    scalar_model = PassiveScalar(
        diffusivity=case.scalar_diffusivity,
        turbulent_prandtl=0.74,
        advection_scheme=case.scalar_advection_scheme,
    )
    poisson = build_pressure_poisson(
        grid,
        backend="gmg",
        periodic_x=False,
        periodic_y=False,
        open_x_low=two_outlets,
        dtype="float32",
        config={
            "tolerance": case.gmg_tolerance,
            "presweeps": case.gmg_presweeps,
            "postsweeps": case.gmg_postsweeps,
            "max_iterations": 200,
        },
    )
    jet = LN2Jet(
        *case.nozzle,
        case.radius,
        case.speed,
        case.mass_flow_rate,
        case.vapor_quality,
        initial_temperature=case.jet_temperature,
        liquid_density=case.liquid_density,
        ramp_time=case.ramp_time,
        initial_diameter=case.initial_diameter,
        minimum_diameter=case.minimum_diameter,
        maximum_diameter=case.maximum_diameter,
        rosin_rammler_spread=case.rosin_rammler_spread,
        parcels_per_step=case.parcels_per_step,
        maximum_parcels=case.maximum_parcels,
        substeps=case.parcel_substeps,
        cone_half_angle_degrees=case.cone_half_angle_degrees,
        edge_speed_ratio=case.edge_speed_ratio,
        edge_diameter_ratio=case.edge_diameter_ratio,
        profile_radius_m=case.profile_radius_m,
        profile_axial_velocity_m_s=case.profile_axial_velocity_m_s,
        profile_radial_velocity_m_s=case.profile_radial_velocity_m_s,
        profile_d10_m=case.profile_d10_m,
        gravity_x=case.gravity[0],
        gravity_y=case.gravity[1],
        gravity_z=case.gravity[2],
    )
    microphysics = CryogenicMicrophysicsConfig(
        pressure=case.pressure,
        dry_air_density=case.dry_air_density,
        dry_air_heat_capacity=case.ambient_heat_capacity,
        dry_air_gas_constant=case.ambient_gas_constant,
        liquid_nitrogen_density=case.liquid_density,
        liquid_nitrogen_latent_heat=case.liquid_latent_heat,
        nitrogen_boiling_temperature=case.jet_temperature,
        outlet_start_x=0.875 * grid.lx,
        outlet_end_x=grid.lx,
    )
    equation_of_state = IdealGasMixture(
        pressure=case.pressure,
        background_gas_constant=case.ambient_gas_constant,
        species_gas_constants=(
            microphysics.nitrogen_gas_constant,
            microphysics.water_vapor_gas_constant,
        ),
    )
    dry_water = case.ambient_water_vapor == 0.0
    volume_source = case.source_mode == "volume"
    incompressible = case.flow_formulation == "incompressible"
    direct_vapor_mass_flow_rate = (
        jet.mass_flow_rate
        if case.fully_vaporized_within_source_cell
        else jet.vapor_mass_flow_rate
    )
    flash_mass_flow_rate = (
        jet.liquid_mass_flow_rate
        if case.fully_vaporized_within_source_cell
        else 0.0
    )

    def thermodynamic_density(temperature, water_vapor, nitrogen_density):
        if dry_water:
            remainder_gas_constant = jnp.asarray(
                case.ambient_gas_constant, temperature.dtype
            )
        else:
            water_ratio = jnp.maximum(water_vapor, 0.0)
            remainder_gas_constant = (
                case.ambient_gas_constant
                + water_ratio * microphysics.water_vapor_gas_constant
            ) / (1.0 + water_ratio)
        pressure_over_temperature = equation_of_state.pressure_field(
            temperature
        ) / jnp.maximum(temperature, equation_of_state.temperature_floor)
        density = (
            pressure_over_temperature
            - nitrogen_density
            * (
                microphysics.nitrogen_gas_constant
                - remainder_gas_constant
            )
        ) / remainder_gas_constant
        return jnp.maximum(
            density,
            nitrogen_density + equation_of_state.density_floor,
        )

    def local_width(faces, coordinate):
        index = int(
            np.clip(np.searchsorted(faces, coordinate) - 1, 0, len(faces) - 2)
        )
        return float(faces[index + 1] - faces[index])

    local_dx = local_width(grid.x_faces, jet.x)
    local_dy = local_width(grid.y_faces, jet.y)
    local_dz = local_width(grid.z_faces, jet.z)
    cell_volume = jnp.asarray(grid.cell_volumes, jnp.float32)
    inlet_cell_area = jnp.asarray(
        grid.z_widths[:, None] * grid.y_widths[None, :], jnp.float32
    )
    y_face_widths = np.concatenate(
        (
            (0.5 * grid.y_widths[0],),
            0.5 * (grid.y_widths[:-1] + grid.y_widths[1:]),
            (0.5 * grid.y_widths[-1],),
        )
    )
    z_face_widths = np.concatenate(
        (
            (0.5 * grid.z_widths[0],),
            0.5 * (grid.z_widths[:-1] + grid.z_widths[1:]),
            (0.5 * grid.z_widths[-1],),
        )
    )
    inlet_y_face_area = jnp.asarray(
        grid.z_widths[:, None] * y_face_widths[None, :], jnp.float32
    )
    inlet_z_face_area = jnp.asarray(
        z_face_widths[:, None] * grid.y_widths[None, :], jnp.float32
    )
    shape = (grid.nz, grid.ny, grid.nx)
    inlet_shape = (grid.nz, grid.ny)

    base_control = DifferentiableInletControl(
        jnp.asarray(case.gas_inlet_radius, jnp.float32),
        jnp.asarray(case.gas_inlet_temperature, jnp.float32),
        jnp.asarray(case.initial_diameter, jnp.float32),
        jnp.asarray(1.0, jnp.float32),
        jnp.asarray(case.edge_speed_ratio, jnp.float32),
        jnp.asarray(case.edge_diameter_ratio, jnp.float32),
    )
    inlet_water_vapor = jnp.full(
        inlet_shape, case.ambient_water_vapor, jnp.float32
    )
    x_cells = jnp.asarray(grid.x_centers, jnp.float32)
    y_cells = jnp.asarray(grid.y_centers, jnp.float32)
    z_cells = jnp.asarray(grid.z_centers, jnp.float32)
    y_faces = jnp.asarray(grid.y_faces, jnp.float32)
    z_faces = jnp.asarray(grid.z_faces, jnp.float32)
    if volume_source:
        nozzle_kernel = vapor_nozzle_source(grid, jet)
        inlet_radius = jnp.zeros(inlet_shape, jnp.float32)
    else:
        inlet_radius = jnp.sqrt(
            (y_cells[None, :] - jet.y) ** 2
            + (z_cells[:, None] - jet.z) ** 2
        )
        nozzle_kernel = jnp.zeros(shape, jnp.float32)
    grid_spacing = max(local_dy, local_dz)
    default_transition_width = 1.5 * grid_spacing
    zero_cells = jnp.zeros(shape, jnp.float32)
    zero_acceleration = _cell_vector_to_faces(
        zero_cells, zero_cells, zero_cells,
        grid,
    )

    def normalise_mode(mode, weight):
        tiny = jnp.finfo(mode.dtype).tiny
        total = jnp.maximum(jnp.sum(weight), tiny)
        centered = mode - jnp.sum(weight * mode) / total
        rms = jnp.sqrt(jnp.sum(weight * centered**2) / total)
        return centered / jnp.maximum(rms, tiny)

    def inlet_fields(control, time, dt=case.dt):
        momentum_correction = zero_acceleration
        midpoint = time + 0.5 * dt
        inlet_ramp = (
            jnp.asarray(1.0, jnp.float32)
            if case.ramp_time == 0.0
            else 0.5
            * (
                1.0
                - jnp.cos(
                    jnp.pi
                    * jnp.clip(midpoint / case.ramp_time, 0.0, 1.0)
                )
            )
        )
        if volume_source:
            inlet_temperature = jnp.full(
                inlet_shape, case.ambient_temperature, jnp.float32
            )
            inlet_nitrogen_density = jnp.zeros(
                inlet_shape, jnp.float32
            )
            inlet_x_velocity = jnp.full(inlet_shape, case.ambient_streamwise_velocity, jnp.float32)
            inlet_y_velocity = jnp.zeros(
                (grid.nz, grid.ny + 1), jnp.float32
            )
            inlet_z_velocity = jnp.zeros(
                (grid.nz + 1, grid.ny), jnp.float32
            )
        else:
            support_radius = (
                jnp.asarray(
                    case.subgrid_support_radius_cells * grid_spacing,
                    jnp.float32,
                )
                if case.subgrid_jet_enabled
                else control.gas_radius
            )
            transition_width = (
                case.subgrid_transition_width_cells * grid_spacing
                if case.subgrid_jet_enabled
                else default_transition_width
            )
            inlet_weight = 0.5 * (
                1.0
                - jnp.tanh(
                    (inlet_radius - support_radius) / transition_width
                )
            )
            pure_nitrogen_density = (
                case.ambient_density
                if incompressible
                else case.pressure
                / (
                    microphysics.nitrogen_gas_constant
                    * control.gas_temperature
                )
            )
            inlet_nitrogen_density = (
                pure_nitrogen_density * inlet_weight
            ).astype(jnp.float32)
            inlet_temperature = (
                case.ambient_temperature
                + inlet_weight
                * (control.gas_temperature - case.ambient_temperature)
            ).astype(jnp.float32)
            inlet_total_density = (
                jnp.full_like(inlet_temperature, case.ambient_density)
                if incompressible
                else thermodynamic_density(
                    inlet_temperature,
                    inlet_water_vapor,
                    inlet_nitrogen_density,
                )
            )
            inlet_speed = jet.vapor_mass_flow_rate / (
                jnp.sum(
                    inlet_nitrogen_density
                    * inlet_weight
                    * inlet_cell_area
                )
            )
            inlet_x_velocity = inlet_ramp * inlet_speed * inlet_weight
            inlet_y_velocity = jnp.zeros(
                (grid.nz, grid.ny + 1), jnp.float32
            )
            inlet_z_velocity = jnp.zeros(
                (grid.nz + 1, grid.ny), jnp.float32
            )

            if (
                case.subgrid_jet_enabled
                and case.subgrid_turbulence_intensity > 0.0
            ):
                length_scale = (
                    case.subgrid_integral_scale_cells * grid_spacing
                )
                wavenumber = 2.0 * jnp.pi / length_scale
                phase = time / case.subgrid_correlation_time
                relative_y = y_cells[None, :] - jet.y
                relative_z = z_cells[:, None] - jet.z
                axial_mode = (
                    jnp.sin(wavenumber * relative_y + phase)
                    * jnp.cos(wavenumber * relative_z - 0.7 * phase)
                    + 0.5
                    * jnp.cos(2.0 * wavenumber * relative_y - 1.3 * phase)
                    * jnp.sin(wavenumber * relative_z + 0.4 * phase)
                )
                axial_mode = normalise_mode(
                    axial_mode,
                    inlet_total_density * inlet_weight * inlet_cell_area
                )
                amplitude = (
                    inlet_ramp
                    * case.subgrid_turbulence_intensity
                    * inlet_speed
                )
                inlet_x_velocity = inlet_x_velocity + (
                    amplitude * inlet_weight * axial_mode
                )

                radius_y = jnp.sqrt(
                    (y_faces[None, :] - jet.y) ** 2
                    + (z_cells[:, None] - jet.z) ** 2
                )
                weight_y = 0.5 * (
                    1.0
                    - jnp.tanh(
                        (radius_y - support_radius) / transition_width
                    )
                )
                transverse_y = (
                    jnp.sin(
                        wavenumber * (z_cells[:, None] - jet.z) + phase
                    )
                    * jnp.cos(
                        wavenumber * (y_faces[None, :] - jet.y)
                        - 0.6 * phase
                    )
                )
                transverse_y = normalise_mode(
                    transverse_y, weight_y * inlet_y_face_area
                )
                inlet_y_velocity = (
                    case.subgrid_transverse_ratio
                    * amplitude
                    * weight_y
                    * transverse_y
                )

                radius_z = jnp.sqrt(
                    (y_cells[None, :] - jet.y) ** 2
                    + (z_faces[:, None] - jet.z) ** 2
                )
                weight_z = 0.5 * (
                    1.0
                    - jnp.tanh(
                        (radius_z - support_radius) / transition_width
                    )
                )
                transverse_z = (
                    -jnp.cos(
                        wavenumber * (z_faces[:, None] - jet.z) + phase
                    )
                    * jnp.sin(
                        wavenumber * (y_cells[None, :] - jet.y)
                        - 0.6 * phase
                    )
                )
                transverse_z = normalise_mode(
                    transverse_z, weight_z * inlet_z_face_area
                )
                inlet_z_velocity = (
                    case.subgrid_transverse_ratio
                    * amplitude
                    * weight_z
                    * transverse_z
                )

            if case.subgrid_jet_enabled:
                resolved_momentum = (
                    jnp.sum(
                        inlet_total_density
                        * inlet_x_velocity**2
                        * inlet_cell_area
                    )
                )
                target_momentum = (
                    jet.vapor_mass_flow_rate
                    * jet.speed
                    * inlet_ramp**2
                )
                momentum_deficit = jnp.maximum(
                    target_momentum - resolved_momentum, 0.0
                )
                momentum_length = (
                    case.subgrid_momentum_length_cells * local_dx
                )
                streamwise_weight = jnp.exp(
                    -0.5 * (x_cells / momentum_length) ** 2
                )
                momentum_kernel = (
                    inlet_weight[..., None]
                    * streamwise_weight[None, None, :]
                )
                momentum_kernel = momentum_kernel / (
                    jnp.sum(momentum_kernel * cell_volume)
                )
                axial_acceleration = (
                    momentum_deficit
                    / case.ambient_density
                    * momentum_kernel
                )
                momentum_correction = _cell_vector_to_faces(
                    axial_acceleration, zero_cells, zero_cells,
                    grid,
                )

            inlet_x_velocity = inlet_x_velocity.astype(jnp.float32)
            inlet_y_velocity = inlet_y_velocity.astype(jnp.float32)
            inlet_z_velocity = inlet_z_velocity.astype(jnp.float32)
        plane = InflowPlane(
            inlet_x_velocity,
            inlet_y_velocity,
            inlet_z_velocity,
            inlet_temperature,
        )
        return (
            plane,
            inlet_temperature,
            inlet_nitrogen_density,
            momentum_correction,
        )

    def make_initial(control):
        (
            inflow_plane,
            inlet_temperature,
            inlet_nitrogen_density,
            _momentum_correction,
        ) = inlet_fields(control, jnp.asarray(0.0, jnp.float32))
        velocity = zeros(grid, "float32", boundaries)
        if case.ambient_streamwise_velocity:
            velocity = velocity._replace(x=jnp.full_like(velocity.x, case.ambient_streamwise_velocity))
        velocity = enforce_velocity(velocity, inflow_plane, grid)
        zero_velocity = StaggeredVelocity(
            jnp.zeros_like(velocity.x),
            jnp.zeros_like(velocity.y),
            jnp.zeros_like(velocity.z),
        )
        initial_temperature = jnp.full(
            shape, case.ambient_temperature, jnp.float32
        ).at[..., 0].set(inlet_temperature)
        initial_water_vapor = jnp.full(
            shape, case.ambient_water_vapor, jnp.float32
        ).at[..., 0].set(inlet_water_vapor)
        initial_nitrogen_density = jnp.zeros(
            shape, jnp.float32
        ).at[..., 0].set(inlet_nitrogen_density)
        thermodynamic_initial_density = thermodynamic_density(
            initial_temperature,
            initial_water_vapor,
            initial_nitrogen_density,
        )
        initial_density = (
            jnp.full(shape, case.ambient_density, jnp.float32)
            if incompressible
            else thermodynamic_initial_density
        )
        initial_nitrogen = jnp.clip(
            initial_nitrogen_density / initial_density, 0.0, 1.0
        )
        initial_state = CryogenicState(
            velocity,
            jnp.zeros(shape, jnp.float32),
            initial_density,
            zero_velocity,
            initial_temperature,
            jnp.zeros(shape, jnp.float32),
            initial_water_vapor,
            jnp.zeros(shape, jnp.float32),
            initial_nitrogen,
            initial_nitrogen_density,
            jnp.zeros(shape, jnp.float32),
            jnp.zeros(shape, jnp.float32),
            jnp.zeros(shape, jnp.float32),
            jnp.zeros(shape, jnp.float32),
            jnp.zeros(shape, jnp.float32),
            initial_ln2_parcels(jet),
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0, jnp.int32),
        )
        if case.cfl is not None:
            return AdaptiveCryogenicState(
                *initial_state, jnp.asarray(case.dt, jnp.float32),
                jnp.asarray(0.0, jnp.float32), jnp.asarray(0, jnp.int32),
            )
        return initial_state

    initial = make_initial(base_control)

    def rk3_step(state, control, dt=case.dt):
        """Wray three-stage RK3 with selectable pressure projection cadence."""
        (
            inflow_plane,
            inlet_temperature,
            inlet_nitrogen_density,
            subgrid_momentum,
        ) = inlet_fields(control, state.time, dt)
        reference_temperature = jnp.full(
            shape, case.ambient_temperature, state.temperature.dtype
        ).at[..., 0].set(inlet_temperature)
        reference_nitrogen_density = jnp.zeros_like(
            state.nitrogen_density
        ).at[..., 0].set(inlet_nitrogen_density)
        reference_density = thermodynamic_density(
            reference_temperature,
            jnp.full_like(state.water_vapor, case.ambient_water_vapor),
            reference_nitrogen_density,
        )
        parcel_control = LN2InletControl(
            control.initial_diameter,
            control.speed_scale,
            control.edge_speed_ratio,
            control.edge_diameter_ratio,
        )
        current_velocity = enforce_velocity(
            state.velocity, inflow_plane, grid
        )
        parcel_exchange = (
            ParcelExchange(
                state.parcels,
                zero_acceleration.x,
                zero_acceleration.y,
                zero_acceleration.z,
                zero_cells,
                zero_cells,
                zero_cells,
                zero_cells,
                jnp.asarray(0.0, state.temperature.dtype),
            )
            if case.fully_vaporized_within_source_cell
            else advance_ln2_parcels(
                state.parcels,
                current_velocity,
                state.temperature,
                grid,
                state.step,
                dt,
                jet,
                microphysics,
                state.density,
                parcel_control,
            )
        )
        dtype = state.temperature.dtype
        midpoint = (
            state.time + 0.5 * dt if case.cfl is not None
            else (state.step.astype(dtype) + 0.5) * dt
        )
        ramp = (
            jnp.asarray(1.0, dtype)
            if jet.ramp_time == 0.0
            else 0.5
            * (
                1.0
                - jnp.cos(
                    jnp.pi
                    * jnp.clip(midpoint / jet.ramp_time, 0.0, 1.0)
                )
            )
        )
        vapor_mass_rate = (
            direct_vapor_mass_flow_rate
            * ramp
            * nozzle_kernel
            / cell_volume
            if volume_source
            else jnp.zeros(shape, dtype)
        )
        flash_power_density = (
            flash_mass_flow_rate
            * case.liquid_latent_heat
            * ramp
            * nozzle_kernel
            / cell_volume
            if volume_source
            else jnp.zeros(shape, dtype)
        )
        vapor_mixing_source = vapor_mass_rate / state.density
        gas_mass_source = parcel_exchange.gas_mass_source + vapor_mass_rate
        parcel_momentum = StaggeredVelocity(
            parcel_exchange.acceleration_x,
            parcel_exchange.acceleration_y,
            parcel_exchange.acceleration_z,
        )

        def tendencies(
            velocity,
            density,
            temperature,
            water_vapor,
            nitrogen,
            nitrogen_density,
            liquid,
            ice,
            execution_time,
        ):
            u_cell, _, _ = cell_velocity(velocity)
            vapor_momentum = _cell_vector_to_faces(
                vapor_mixing_source * (jet.speed - u_cell),
                jnp.zeros(shape, dtype),
                jnp.zeros(shape, dtype),
                grid,
            )
            momentum = _add_velocity(
                momentum_rhs(velocity, execution_time), parcel_momentum
            )
            momentum = _add_velocity(momentum, subgrid_momentum)
            if not incompressible:
                momentum = _add_velocity(
                    momentum, dilatation_correction(velocity, grid)
                )
            momentum = _add_velocity(momentum, vapor_momentum)
            buoyancy_density = (
                thermodynamic_density(
                    temperature, water_vapor, nitrogen_density
                )
                if incompressible
                else density
            )
            density_anomaly = (
                buoyancy_density - reference_density
            ) / jnp.maximum(buoyancy_density, 1.0e-6)
            momentum = _add_velocity(
                momentum,
                _cell_vector_to_faces(
                    case.gravity[0] * density_anomaly,
                    case.gravity[1] * density_anomaly,
                    case.gravity[2] * density_anomaly,
                    grid,
                ),
            )
            viscosity = eddy_viscosity(
                velocity, grid, boundaries, closure
            )
            flow_dilatation = divergence(velocity, grid)

            def transported(field):
                return (
                    scalar_tendency(
                        field,
                        velocity,
                        grid,
                        scalar_model,
                        eddy_viscosity=viscosity,
                    )
                    + field * flow_dilatation
                )

            temperature_rhs = (
                transported(temperature)
                + parcel_exchange.temperature_source
                + vapor_mixing_source
                * (jet.initial_temperature - temperature)
                - flash_power_density
                / (density * case.ambient_heat_capacity)
            )
            nitrogen_density_rhs = conservative_specific_tendency(
                nitrogen,
                density,
                velocity,
                grid,
                scalar_model,
                eddy_diffusivity=(
                    viscosity / scalar_model.turbulent_prandtl
                ),
            ) + gas_mass_source
            if dry_water:
                water_rhs = jnp.zeros_like(water_vapor)
                liquid_rhs = jnp.zeros_like(liquid)
                ice_rhs = jnp.zeros_like(ice)
            else:
                water_rhs = transported(water_vapor)
                liquid_rhs = transported(liquid)
                ice_rhs = transported(ice)
            return (
                momentum,
                temperature_rhs,
                water_rhs,
                nitrogen_density_rhs,
                liquid_rhs,
                ice_rhs,
            )

        weights = (
            (8.0 / 15.0, 0.0),
            (5.0 / 12.0, -17.0 / 60.0),
            (3.0 / 4.0, -5.0 / 12.0),
        )
        velocity = current_velocity
        temperature = state.temperature
        water_vapor = state.water_vapor
        nitrogen = state.nitrogen
        nitrogen_density = state.nitrogen_density
        liquid = state.liquid_water
        ice = state.ice_water
        execution_time = state.time
        pressure = state.pressure
        density = state.density
        previous = (
            state.momentum_tendency,
            state.temperature_tendency,
            state.water_vapor_tendency,
            state.nitrogen_density_tendency,
            state.liquid_water_tendency,
            state.ice_water_tendency,
        )
        current = previous
        lagged_pressure_gradient = (
            pressure_gradient(
                pressure,
                grid,
                periodic_x=poisson.periodic_x,
                periodic_y=poisson.periodic_y,
                open_x_low=poisson.open_x_low,
            )
            if case.time_integration == "fast-rk3"
            else None
        )
        last_stage = len(weights) - 1
        continuity_error = state.continuity_error
        peak_cfl = courant_number(velocity, grid, dt) if case.cfl is not None else 0.0
        for stage, (current_weight, previous_weight) in enumerate(weights):
            previous_stage_density = density
            current = tendencies(
                velocity,
                density,
                temperature,
                water_vapor,
                nitrogen,
                nitrogen_density,
                liquid,
                ice,
                execution_time,
            )
            current_momentum = current[0]
            previous_momentum = previous[0]
            candidate = StaggeredVelocity(
                velocity.x
                + dt
                * (
                    current_weight * current_momentum.x
                    + previous_weight * previous_momentum.x
                ),
                velocity.y
                + dt
                * (
                    current_weight * current_momentum.y
                    + previous_weight * previous_momentum.y
                ),
                velocity.z
                + dt
                * (
                    current_weight * current_momentum.z
                    + previous_weight * previous_momentum.z
                ),
            )
            substep = dt * (current_weight + previous_weight)
            if lagged_pressure_gradient is not None:
                if incompressible:
                    candidate = StaggeredVelocity(
                        candidate.x - substep * lagged_pressure_gradient.x,
                        candidate.y - substep * lagged_pressure_gradient.y,
                        candidate.z - substep * lagged_pressure_gradient.z,
                    )
                else:
                    pressure_density = face_density(density, candidate, grid)
                    candidate = StaggeredVelocity(
                        candidate.x
                        - substep
                        * lagged_pressure_gradient.x
                        / pressure_density.x,
                        candidate.y
                        - substep
                        * lagged_pressure_gradient.y
                        / pressure_density.y,
                        candidate.z
                        - substep
                        * lagged_pressure_gradient.z
                        / pressure_density.z,
                    )
            candidate = enforce_velocity(candidate, inflow_plane, grid)

            def stage_scalar(
                field, tendency, old_tendency, ambient, floor=None, ceiling=None
            ):
                updated = field + dt * (
                    current_weight * tendency
                    + previous_weight * old_tendency
                )
                if floor is not None:
                    updated = jnp.maximum(updated, floor)
                if ceiling is not None:
                    updated = jnp.minimum(updated, ceiling)
                if two_outlets:
                    return enforce_two_outlet_scalar(updated, candidate, ambient)
                return _enforce_scalar(updated, ambient)

            temperature = stage_scalar(
                temperature,
                current[1],
                previous[1],
                inlet_temperature,
                jet.initial_temperature,
            )
            nitrogen_density = stage_scalar(
                nitrogen_density,
                current[3],
                previous[3],
                inlet_nitrogen_density,
            )
            nitrogen_density = _conservative_nonnegative(
                nitrogen_density, cell_volume
            )
            if not dry_water:
                water_vapor = stage_scalar(
                    water_vapor,
                    current[2],
                    previous[2],
                    inlet_water_vapor,
                    0.0,
                )
                liquid = stage_scalar(
                    liquid, current[4], previous[4], 0.0, 0.0
                )
                ice = stage_scalar(
                    ice, current[5], previous[5], 0.0, 0.0
                )
                fog = advance_fog_microphysics(
                    temperature,
                    water_vapor,
                    liquid,
                    ice,
                    substep,
                    microphysics,
                )
                temperature = jnp.maximum(
                    fog.temperature, jet.initial_temperature
                )
                water_vapor, liquid, ice = fog.qv, fog.ql, fog.qi
            updated_thermodynamic_density = thermodynamic_density(
                temperature, water_vapor, nitrogen_density
            )
            density = (
                jnp.full_like(updated_thermodynamic_density, case.ambient_density)
                if incompressible
                else updated_thermodynamic_density
            )
            nitrogen = jnp.clip(nitrogen_density / density, 0.0, 1.0)
            if lagged_pressure_gradient is None:
                if incompressible:
                    velocity, pressure = project(
                        candidate,
                        poisson,
                        substep,
                        initial_pressure=pressure,
                    )
                else:
                    velocity, pressure = project_low_mach(
                        candidate,
                        previous_stage_density,
                        density,
                        poisson,
                        substep,
                        mass_source=gas_mass_source,
                        initial_pressure=pressure,
                    )
            elif stage == last_stage:
                if incompressible:
                    velocity, correction = project(candidate, poisson, substep)
                else:
                    velocity, correction = project_low_mach(
                        candidate,
                        state.density,
                        density,
                        poisson,
                        substep,
                        mass_source=gas_mass_source,
                        continuity_dt=dt,
                    )
                pressure = pressure + correction * (substep / dt)
            else:
                velocity = candidate
            if case.cfl is not None:
                peak_cfl = jnp.maximum(peak_cfl, courant_number(velocity, grid, dt))
            if stage == last_stage:
                if incompressible:
                    continuity_error = case.ambient_density * jnp.max(
                        jnp.abs(divergence(velocity, grid))
                    )
                else:
                    mass_previous = (
                        state.density
                        if lagged_pressure_gradient is not None
                        else previous_stage_density
                    )
                    mass_interval = (
                        dt
                        if lagged_pressure_gradient is not None
                        else substep
                    )
                    continuity_error = jnp.max(
                        jnp.abs(
                            continuity_residual(
                                velocity,
                                mass_previous,
                                density,
                                grid,
                                mass_interval,
                                gas_mass_source,
                            )
                        )
                    )
            previous = current
            execution_time = execution_time + substep

        nitrogen = jnp.clip(nitrogen_density / density, 0.0, 1.0)

        result = CryogenicState(
            velocity,
            pressure,
            density,
            current[0],
            temperature,
            current[1],
            water_vapor,
            current[2],
            nitrogen,
            nitrogen_density,
            current[3],
            liquid,
            current[4],
            ice,
            current[5],
            parcel_exchange.parcels,
            continuity_error,
            state.time + dt,
            state.step + 1,
        )
        if case.cfl is not None:
            return AdaptiveCryogenicState(*result, dt, peak_cfl, state.rejected_steps)
        return result

    def advance_controlled(state, control, count):
        return jax.lax.fori_loop(
            0, count, lambda _, carry: rk3_step(carry, control), state
        )

    if case.cfl is not None:
        def diffusivity(velocity):
            eddy = eddy_viscosity(velocity, grid, boundaries, closure)
            return jnp.maximum(
                case.kinematic_viscosity + eddy,
                case.scalar_diffusivity + eddy / scalar_model.turbulent_prandtl,
            )

        advance = build_adaptive_advance(
            lambda state, dt: rk3_step(state, base_control, dt),
            case, grid, diffusivity,
        )
        return grid, jet, microphysics, initial, advance, courant_number

    controlled = jax.jit(advance_controlled, static_argnums=2)
    if differentiable_inlet:
        return (
            grid,
            jet,
            microphysics,
            jax.jit(make_initial),
            controlled,
            courant_number,
            base_control,
        )

    def advance_block(state, count):
        return controlled(state, base_control, count)

    return (
        grid,
        jet,
        microphysics,
        initial,
        jax.jit(advance_block, static_argnums=1),
        courant_number,
    )
