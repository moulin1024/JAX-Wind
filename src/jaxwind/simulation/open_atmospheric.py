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
    step = build_open_atmospheric_step(
        grid,
        boundaries,
        poisson,
        momentum,
        scalar,
        buoyancy,
        surface,
        scalar_source=scalar_source,
        scheme=workflow.case.options.time_integration,
    )
    advance = step if return_step else build_open_atmospheric_run(step)

    return solution, advance
