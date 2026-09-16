"""Public atmospheric model construction and initialization."""
from __future__ import annotations
from typing import Any
from jaxwind.config.abl import FiniteVolumeCase
from jaxwind.config.abl_resolved import resolved
from jaxwind.io.initialization import initial_fields

def build_models(
    configured: FiniteVolumeCase,
    *,
    periodic_x: bool,
    forcing=None,
    pressure_force_enabled: bool = True,
    evolve_scalar: bool = True,
):
    """Compose identical physical closures for each workflow stage."""
    from jaxwind import (
        AnisotropicMinimumDissipation,
        CELL_AVERAGE,
        LOCAL,
        PLANAR,
        OPEN,
        Boundaries,
        CoriolisGeostrophic,
        FlowModel,
        LinearBoussinesqBuoyancy,
        MoninObukhovSurface,
        MoninObukhovWall,
        PassiveScalar,
        monin_obukhov_boundaries,
    )

    case = configured.physical
    configuration = resolved(configured)
    offset_u, offset_v = case.advection_frame_velocity_m_s
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
    coupled = case.surface_scalar
    wall = None
    surface = None
    if coupled is None:
        wall = MoninObukhovWall(
            configuration["roughness_length_m"],
            von_karman=case.von_karman,
            sampling=CELL_AVERAGE,
            averaging=PLANAR if configured.options.wall_averaging == "planar" else LOCAL,
            gradient_correction=configured.options.wall_gradient_correction,
        )
    else:
        coefficient = configuration["buoyancy_acceleration_per_scalar"]
        surface = MoninObukhovSurface(
            momentum_roughness=configuration["momentum_roughness_m"],
            scalar_roughness=configuration["scalar_roughness_m"],
            surface_scalar_initial=configuration["surface_scalar_initial"],
            surface_scalar_rate=configuration["surface_scalar_rate_per_second"],
            x_velocity_offset=offset_u,
            y_velocity_offset=offset_v,
            buoyancy_coefficient=coefficient,
            von_karman=case.von_karman,
            positive_zeta_momentum_slope=coupled.positive_zeta_momentum_slope,
            positive_zeta_scalar_slope=coupled.positive_zeta_scalar_slope,
            negative_zeta_momentum_coefficient=(
                coupled.negative_zeta_momentum_coefficient
            ),
            negative_zeta_scalar_coefficient=(
                coupled.negative_zeta_scalar_coefficient
            ),
            iterations=coupled.iterations,
            relaxation=coupled.relaxation,
            maximum_abs_zeta=coupled.maximum_abs_zeta,
            gradient_correction=configured.options.wall_gradient_correction,
        )
    pressure_force = configuration["pressure_acceleration_m_s2"]
    if not pressure_force_enabled:
        pressure_force = (0.0, 0.0)
    momentum = FlowModel(
        momentum_advection_scheme=configured.options.momentum_advection_scheme,
        outlet_backflow=configured.options.outlet_backflow,
        outlet_sponge_start_fraction=(configured.options.outlet_sponge_start_fraction if not periodic_x else None),
        outlet_sponge_timescale_seconds=configured.options.outlet_sponge_timescale_seconds,
        upstream_mode_sponge_end_fraction=(configured.options.upstream_mode_sponge_end_fraction if not periodic_x else None),
        upstream_mode_sponge_timescale_seconds=configured.options.upstream_mode_sponge_timescale_seconds,
        body_force=(pressure_force[0], pressure_force[1], 0.0),
        forcing=forcing,
        subfilter=AnisotropicMinimumDissipation(),
        surface=wall,
        rotation=rotation,
    )
    scalar = (
        PassiveScalar(
            lower_flux=configuration["scalar_surface_flux"],
            turbulent_prandtl=configured.options.turbulent_prandtl,
            advection_scheme=configured.options.scalar_advection_scheme,
        )
        if evolve_scalar
        else None
    )
    coefficient = configuration["buoyancy_acceleration_per_scalar"]
    buoyancy = (
        LinearBoussinesqBuoyancy(coefficient) if coefficient != 0.0 else None
    )
    boundaries = monin_obukhov_boundaries()
    if not periodic_x:
        boundaries = Boundaries(
            boundaries.lower,
            boundaries.upper,
            streamwise=OPEN,
        )
    return boundaries, momentum, scalar, buoyancy, surface


def fft_solver_config(configured: FiniteVolumeCase) -> dict[str, Any]:
    options = configured.options
    return {
        "method": options.fft_method,
        "thomas_chunk": options.fft_thomas_chunk,
        "spike_block_size": options.fft_spike_block_size,
    }


def initialize_periodic(
    configured: FiniteVolumeCase,
    jax,
    jnp,
    *,
    fft_config: dict[str, Any] | None = None,
):
    from jaxwind import (
        StaggeredVelocity,
        build_pressure_poisson,
        initial_atmospheric_solution,
        project,
    )

    case = configured.physical
    grid = case.physical_grid
    u, v, w, scalar = initial_fields(case, jax, jnp)
    offset_u, offset_v = case.advection_frame_velocity_m_s
    velocity = StaggeredVelocity(u - offset_u, v - offset_v, w)
    poisson = build_pressure_poisson(
        grid,
        backend="fft",
        dtype=case.dtype,
        config=(
            fft_solver_config(configured)
            if fft_config is None
            else fft_config
        ),
    )
    velocity, _ = project(velocity, poisson, 1.0)
    return initial_atmospheric_solution(
        grid,
        velocity,
        scalar,
        dtype=case.dtype,
    )


def build_periodic_advance(
    configured: FiniteVolumeCase,
    *,
    fft_config: dict[str, Any] | None = None,
):
    from jaxwind import (
        build_atmospheric_run,
        build_atmospheric_step,
        build_pressure_poisson,
    )

    case = configured.physical
    grid = case.physical_grid
    boundaries, momentum, scalar, buoyancy, surface = build_models(
        configured, periodic_x=True
    )
    poisson = build_pressure_poisson(
        grid,
        backend="fft",
        dtype=case.dtype,
        config=(
            fft_solver_config(configured)
            if fft_config is None
            else fft_config
        ),
    )
    step = build_atmospheric_step(
        grid,
        boundaries,
        poisson,
        momentum,
        scalar,
        buoyancy,
        surface,
        scheme=configured.options.time_integration,
    )
    return step, build_atmospheric_run(step)
