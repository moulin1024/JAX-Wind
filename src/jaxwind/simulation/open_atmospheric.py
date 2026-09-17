"""Open-boundary atmospheric assembly; recorded inflow is supplied explicitly."""
from __future__ import annotations
from dataclasses import replace
from jaxwind.simulation.abl import build_models
from jaxwind.simulation.turbines import build_turbine_forcing, combine_forcings


def build_open_components(workflow, warm, first, *, return_step=False):
    from jaxwind import (build_open_atmospheric_run, build_open_atmospheric_step,
        build_pressure_poisson, enforce_open_scalar, enforce_open_velocity,
        initial_atmospheric_solution, periodic_to_open_velocity)
    case = workflow.case.physical
    grid = case.physical_grid
    velocity = periodic_to_open_velocity(warm.velocity, grid)
    velocity = enforce_open_velocity(velocity, first, grid)
    scalar_field = enforce_open_scalar(warm.scalar, first, grid)
    solution = initial_atmospheric_solution(
        grid,
        velocity,
        scalar_field,
        dtype=case.dtype,
    )
    forcing = build_turbine_forcing(workflow)
    boundaries, momentum, scalar, buoyancy, surface = build_models(
        workflow.case,
        periodic_x=False,
        forcing=forcing,
        pressure_force_enabled=workflow.options.main_pressure_force,
        evolve_scalar=workflow.options.evolve_scalar,
    )
    gmg_config = {
        "presweeps": workflow.case.options.gmg_presweeps,
        "postsweeps": workflow.case.options.gmg_postsweeps,
        "anisotropy_aware": (
            workflow.case.options.gmg_anisotropy_aware
        ),
        **(
            {}
            if workflow.case.options.gmg_tolerance is None
            else {"tolerance": workflow.case.options.gmg_tolerance}
        ),
    }
    poisson = build_pressure_poisson(
        grid,
        backend="gmg",
        periodic_x=False,
        dtype=case.dtype,
        config=gmg_config,
    )
    turbine = workflow.turbine
    scalar_source = None
    if workflow.cooling is not None:
        from jaxwind import (
            SubgridCooling,
            SubgridSpray,
            build_subgrid_cooling_source,
            build_subgrid_spray_sources,
        )

        if turbine is None:
            raise ValueError("finite-volume cooling requires a turbine")
        cooling = workflow.cooling
        source_center = (
            turbine.x_m + cooling.streamwise_offset_m,
            turbine.y_m,
            turbine.hub_height_m,
        )
        if cooling.has_momentum_jet:
            spray = SubgridSpray(
                cooling_power_w=cooling.cooling_power_w,
                mass_flow_rate_kg_s=cooling.mass_flow_rate_kg_s,
                injection_speed_m_s=cooling.injection_speed_m_s,
                air_density_kg_m3=cooling.air_density_kg_m3,
                air_heat_capacity_j_kg_k=cooling.air_heat_capacity_j_kg_k,
                nozzle_m=source_center,
                nozzle_diameter_m=cooling.nozzle_diameter_m,
                cone_half_angle_degrees=cooling.cone_half_angle_degrees,
                axial_standard_deviation_m=(
                    cooling.standard_deviation_m[0]
                ),
                minimum_radial_standard_deviation_m=(
                    cooling.standard_deviation_m[1],
                    cooling.standard_deviation_m[2],
                ),
                ramp_time_s=cooling.ramp_time_s,
            )
            scalar_source, spray_forcing = build_subgrid_spray_sources(
                grid, spray, dtype=case.dtype
            )
            momentum = replace(
                momentum,
                forcing=combine_forcings(momentum.forcing, spray_forcing),
            )
        else:
            scalar_source = build_subgrid_cooling_source(
                grid,
                SubgridCooling(
                    cooling_power_w=cooling.cooling_power_w,
                    air_density_kg_m3=cooling.air_density_kg_m3,
                    air_heat_capacity_j_kg_k=cooling.air_heat_capacity_j_kg_k,
                    center_m=source_center,
                    standard_deviation_m=cooling.standard_deviation_m,
                    ramp_time_s=cooling.ramp_time_s,
                ),
                dtype=case.dtype,
            )
    if workflow.moisture is not None:
        from jaxwind import LinearBoussinesqBuoyancy
        buoyancy = LinearBoussinesqBuoyancy(
            9.81 / workflow.moisture.reference_temperature_k
        )
    dpm_enabled = workflow.water_spray is not None and workflow.water_spray.model == "fluent-dpm"
    if dpm_enabled:
        from jaxwind.sgs import FluentSmagorinsky
        if case.dtype != "float64":
            raise ValueError("Fluent DPM currently requires numerics.dtype=float64")
        if surface is not None:
            raise ValueError("Fluent DPM currently requires an uncoupled thermal surface")
        options = workflow.water_spray.dpm
        if options.les_model == "fluent-smagorinsky":
            momentum = replace(momentum,subfilter=FluentSmagorinsky(options.smagorinsky_constant,options.von_karman))
        # amd-inferred retains the carrier AMD model and declares a separate
        # particle length assumption in the DPM configuration.
    step = build_open_atmospheric_step(
        grid,
        boundaries,
        poisson,
        momentum,
        scalar,
        buoyancy,
        surface,
        scalar_source=scalar_source,
        scalar_boundary="flux" if dpm_enabled else "cell",
        transport_scalar=not dpm_enabled,
        scheme=workflow.case.options.time_integration,
    )
    if dpm_enabled:
        from jaxwind.fluent_dpm_atmosphere import initialize_dpm, build_dpm_atmospheric_step
        from jaxwind.numerics.poisson import project
        solution = initialize_dpm(solution,workflow.moisture,workflow.water_spray)
        center = (turbine.x_m+workflow.water_spray.streamwise_offset_m,turbine.y_m,turbine.hub_height_m)
        project_feedback = lambda velocity,h,inflow: project(enforce_open_velocity(velocity,inflow,grid),poisson,h)[0]
        step = build_dpm_atmospheric_step(step,grid,boundaries,momentum,scalar,
            workflow.moisture,workflow.water_spray,center,solution.moisture.vapor[...,0],project_feedback)
    elif workflow.moisture is not None:
        from jaxwind.moist_abl import build_moist_atmospheric_step, initialize_moisture
        from jaxwind.water_spray import build_water_injection
        source = None
        if workflow.water_spray is not None:
            spray = workflow.water_spray
            source = build_water_injection(
                grid, (turbine.x_m + spray.streamwise_offset_m,
                       turbine.y_m, turbine.hub_height_m),
                spray.standard_deviation_m, spray.mass_flow_rate_kg_s,
                spray.droplet_diameter_m, spray.ramp_time_s,
                workflow.moisture.thermodynamics, dtype=case.dtype,
            )
        moist = workflow.moisture
        solution, ambient_vapor = initialize_moisture(
            solution, moist.temperature_offset_k,
            moist.ambient_relative_humidity, moist.thermodynamics,
        )
        step = build_moist_atmospheric_step(
            step, grid, boundaries, momentum, scalar,
            moist.thermodynamics, moist.temperature_offset_k,
            moist.reference_temperature_k, ambient_vapor, source,
        )
    advance = step if return_step else build_open_atmospheric_run(step)

    return solution, advance
