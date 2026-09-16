"""Atmospheric simulation assembly, independent of output execution."""
from __future__ import annotations
from types import SimpleNamespace
from jaxwind.config.abl_resolved import resolved
from jaxwind.io.initialization import initial_fields


def build_components(configured, *, forcing=None, farm=None):
    case = configured.physical
    options = configured.options
    configuration = resolved(configured)
    import jax

    jax.config.update("jax_enable_x64", case.dtype == "float64")
    import jax.numpy as jnp
    from jaxwind import (
        AnisotropicMinimumDissipation,
        CELL_AVERAGE,
        LOCAL,
        PLANAR,
        CoriolisGeostrophic,
        FlowModel,
        LinearBoussinesqBuoyancy,
        MoninObukhovSurface,
        MoninObukhovWall,
        PassiveScalar,
        StaggeredVelocity,
        build_adaptive_atmospheric_run,
        build_atmospheric_run,
        build_atmospheric_step,
        build_pressure_poisson,
        courant_number,
        divergence,
        initial_atmospheric_solution,
        monin_obukhov_boundaries,
        project,
    )

    grid = case.physical_grid
    u, v, w, scalar_field = initial_fields(case, jax, jnp)
    offset_u, offset_v = case.advection_frame_velocity_m_s
    velocity = StaggeredVelocity(u - offset_u, v - offset_v, w)
    solver_config = (
        {
            "presweeps": options.gmg_presweeps,
            "postsweeps": options.gmg_postsweeps,
            "anisotropy_aware": options.gmg_anisotropy_aware,
            **(
                {}
                if options.gmg_tolerance is None
                else {"tolerance": options.gmg_tolerance}
            ),
        }
        if options.pressure_backend == "gmg"
        else {
            "method": options.fft_method,
            "thomas_chunk": options.fft_thomas_chunk,
            "spike_block_size": options.fft_spike_block_size,
        }
    )
    poisson = build_pressure_poisson(
        grid,
        backend=options.pressure_backend,
        dtype=case.dtype,
        config=solver_config,
    )
    velocity, _ = project(velocity, poisson, 1.0)
    boundaries = monin_obukhov_boundaries()
    subfilter = AnisotropicMinimumDissipation()

    vertical_f = configuration["coriolis_vertical_s"]
    rotation = None
    if vertical_f != 0.0:
        evolved = configuration["evolved_geostrophic_velocity_m_s"]
        rotation = CoriolisGeostrophic(
            vertical_f,
            evolved[0],
            evolved[1],
            configuration["coriolis_horizontal_s"],
        )
    pressure_force = configuration["pressure_acceleration_m_s2"]
    wall = None
    coupled_surface = case.surface_scalar
    if coupled_surface is None:
        wall = MoninObukhovWall(
            configuration["roughness_length_m"],
            von_karman=case.von_karman,
            sampling=CELL_AVERAGE,
            averaging=PLANAR if options.wall_averaging == "planar" else LOCAL,
            gradient_correction=options.wall_gradient_correction,
        )
    momentum = FlowModel(
        momentum_advection_scheme=options.momentum_advection_scheme,
        body_force=(pressure_force[0], pressure_force[1], 0.0),
        forcing=forcing,
        subfilter=subfilter,
        surface=wall,
        rotation=rotation,
    )
    scalar = PassiveScalar(
        lower_flux=configuration["scalar_surface_flux"],
        turbulent_prandtl=options.turbulent_prandtl,
    )
    coefficient = configuration["buoyancy_acceleration_per_scalar"]
    buoyancy = (
        LinearBoussinesqBuoyancy(coefficient)
        if coefficient != 0.0
        else None
    )
    surface = None
    if coupled_surface is not None:
        surface = MoninObukhovSurface(
            momentum_roughness=configuration["momentum_roughness_m"],
            scalar_roughness=configuration["scalar_roughness_m"],
            surface_scalar_initial=configuration["surface_scalar_initial"],
            surface_scalar_rate=configuration[
                "surface_scalar_rate_per_second"
            ],
            x_velocity_offset=offset_u,
            y_velocity_offset=offset_v,
            buoyancy_coefficient=coefficient,
            von_karman=case.von_karman,
            positive_zeta_momentum_slope=(
                coupled_surface.positive_zeta_momentum_slope
            ),
            positive_zeta_scalar_slope=(
                coupled_surface.positive_zeta_scalar_slope
            ),
            negative_zeta_momentum_coefficient=(
                coupled_surface.negative_zeta_momentum_coefficient
            ),
            negative_zeta_scalar_coefficient=(
                coupled_surface.negative_zeta_scalar_coefficient
            ),
            iterations=coupled_surface.iterations,
            relaxation=coupled_surface.relaxation,
            maximum_abs_zeta=coupled_surface.maximum_abs_zeta,
            gradient_correction=options.wall_gradient_correction,
        )
    def step_factory(source):
        from dataclasses import replace
        return build_atmospheric_step(
            grid, boundaries, poisson, replace(momentum, forcing=source),
            scalar, buoyancy, surface, scheme=options.time_integration,
        )
    step = step_factory(forcing) if farm is None else farm.couple_step(step_factory)
    adaptive = options.cfl_ceiling is not None
    if adaptive:
        # dt_seconds is the upper bound; the CFL ceiling sets the actual step.
        advance = build_adaptive_atmospheric_run(
            step,
            grid,
            cfl_ceiling=options.cfl_ceiling,
            maximum_dt=case.dt_seconds,
        )
    else:
        advance = build_atmospheric_run(step)
    solution = initial_atmospheric_solution(
        grid,
        velocity,
        scalar_field,
        dtype=case.dtype,
    )
    if farm is not None:
        solution = farm.initialize(solution)

    diagnostics = build_diagnostics(
        configured, grid, boundaries, wall, scalar, subfilter, surface,
    )
    return SimpleNamespace(
        **vars(diagnostics), initial=solution, advance=advance,
        step=step, adaptive=adaptive,
    )


def build_diagnostics(configured, grid, boundaries, wall, scalar, subfilter, surface):
    """Attach the same profile observations to direct and workflow runs."""
    import jax
    from jaxwind import (
        atmospheric_history_diagnostics,
        atmospheric_profile_diagnostics,
        coupled_surface_exchange,
        friction_velocity,
    )

    case = configured.physical
    configuration = resolved(configured)
    offset_u, offset_v = case.advection_frame_velocity_m_s
    vertical_f = configuration["coriolis_vertical_s"]
    if surface is None:
        def diagnostic(velocity, pressure, scalar_field, execution_time):
            del execution_time
            fields, profiles = atmospheric_profile_diagnostics(
                velocity,
                pressure,
                scalar_field,
                grid,
                boundaries,
                wall,
                scalar,
                subfilter,
            )
            ustar = friction_velocity(velocity, grid, wall)
            return fields, profiles, ustar, None

        history_diagnostic = jax.jit(
            lambda current: atmospheric_history_diagnostics(
                current,
                grid,
                wall,
                coriolis=vertical_f,
                geostrophic_u=configuration[
                    "geostrophic_velocity_m_s"
                ][0],
                geostrophic_v=configuration[
                    "geostrophic_velocity_m_s"
                ][1],
            )
        )
        exchange_diagnostic = None
    else:
        def diagnostic(velocity, pressure, scalar_field, execution_time):
            exchange = coupled_surface_exchange(
                velocity,
                scalar_field,
                execution_time,
                grid,
                surface,
            )
            fields, profiles = atmospheric_profile_diagnostics(
                velocity,
                pressure,
                scalar_field,
                grid,
                boundaries,
                None,
                scalar,
                subfilter,
                x_velocity_offset=offset_u,
                y_velocity_offset=offset_v,
                lower_stress_x=exchange.stress_x,
                lower_stress_y=exchange.stress_y,
                lower_scalar_flux=exchange.scalar_flux,
            )
            return fields, profiles, exchange.friction_velocity, exchange

        history_diagnostic = None
        exchange_diagnostic = jax.jit(
            lambda current, scalar_field, execution_time: (
                coupled_surface_exchange(
                    current,
                    scalar_field,
                    execution_time,
                    grid,
                    surface,
                )
            )
        )
    profile_diagnostic = jax.jit(diagnostic)

    return SimpleNamespace(
        configured=configured, grid=grid, profile_diagnostic=profile_diagnostic,
        history_diagnostic=history_diagnostic, exchange_diagnostic=exchange_diagnostic,
        wall=wall, scalar=scalar, subfilter=subfilter, surface=surface,
    )
