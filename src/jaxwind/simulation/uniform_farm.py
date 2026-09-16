"""Direct uniform-inflow AD-BEM farm with x/y pressure outlets and GMG."""
from dataclasses import replace
import math

import jax
import jax.numpy as jnp

from jaxwind.config.abl import load_fv_abl
from jaxwind import (
    StaggeredVelocity, initial_atmospheric_solution, courant_number,
    build_pressure_poisson, build_open_atmospheric_step, OPEN,
)
from jaxwind.open_boundary import InflowPlane
from .api import Simulation
from .abl import build_models
from .wind_farm import ControlledFarm


def build_simulation(case):
    configured = load_fv_abl(case)
    grid = configured.physical.physical_grid
    if not grid.is_uniform:
        raise ValueError("uniform-inflow farm currently requires a uniform mesh")
    dtype = configured.physical.dtype
    speed = case.document["physics"]["inflow"]["speed_m_s"]
    scalar_value = case.document["physics"]["scalar"]["reference_value"]
    shape = (grid.nz, grid.ny, grid.nx)
    velocity = StaggeredVelocity(
        jnp.full((grid.nz, grid.ny, grid.nx + 1), speed, dtype),
        jnp.zeros((grid.nz, grid.ny + 1, grid.nx), dtype),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype),
    )
    inflow = InflowPlane(
        jnp.full((grid.nz, grid.ny), speed, dtype),
        jnp.zeros((grid.nz, grid.ny + 1), dtype),
        jnp.zeros((grid.nz + 1, grid.ny), dtype),
        jnp.full((grid.nz, grid.ny), scalar_value, dtype),
    )
    flow = initial_atmospheric_solution(grid, velocity, jnp.full(shape, scalar_value, dtype), dtype=dtype)
    farm = ControlledFarm(case)
    boundaries, momentum, _, _, _ = build_models(
        configured, periodic_x=False, pressure_force_enabled=False, evolve_scalar=False,
    )
    boundaries = replace(boundaries, spanwise=OPEN)
    options = configured.options
    poisson = build_pressure_poisson(
        grid, backend="gmg", periodic_x=False, periodic_y=False, open_y=True,
        dtype=dtype, config={
            "tolerance": options.gmg_tolerance or 1.e-5,
            "presweeps": options.gmg_presweeps,
            "postsweeps": options.gmg_postsweeps,
            "anisotropy_aware": options.gmg_anisotropy_aware,
        },
    )
    def factory(source):
        step = build_open_atmospheric_step(
            grid, boundaries, poisson, replace(momentum, forcing=source), None,
            scheme=options.time_integration,
        )
        return lambda state, dt: step(state, dt, inflow)
    step = farm.couple_step(factory)
    initial = farm.initialize(flow)
    dt = case.document["time"]["dt_seconds"]

    @jax.jit(static_argnums=1)
    def advance(state, count):
        return jax.lax.scan(lambda current, _: (step(current, dt), None), state, None, length=count)[0]

    courant = jax.jit(lambda state: courant_number(state.velocity, grid, dt))
    def checked_advance(state, controls):
        result = advance(state, controls.count)
        value = float(courant(result))
        if not math.isfinite(value) or value > 1.0:
            raise RuntimeError(f"uniform farm stopped: CFL={value}; reduce fixed dt before rerunning")
        return result

    hub = int(jnp.argmin(jnp.abs(jnp.asarray(grid.z_centers) - case.document["physics"]["turbine"]["hub_height_m"])))
    first_rotor = min(row["x_m"] for row in case.document["physics"]["wind_farm"]["layout"])
    upstream_stop = max(3, int((first_rotor - 160.) / grid.dx))
    upstream_stop = min(upstream_stop, grid.nx)
    @jax.jit
    def state_diagnostics(state):
        u, v, w = state.velocity
        area_x, area_y = grid.dy * grid.dz, grid.dx * grid.dz
        normal_x, normal_low, normal_high = u[..., -1], -v[:, 0], v[:, -1]
        profile = u[hub, :, :upstream_stop]
        return {
            "outlet_x_minimum_u_m_s": jnp.min(normal_x),
            "outlet_x_backflow_area_fraction": jnp.mean(normal_x < 0.),
            "outlet_x_backflow_area_fraction_below_001": jnp.mean(normal_x < -0.01),
            "outlet_x_reverse_volume_flux_m3_s": -jnp.sum(jnp.minimum(normal_x, 0.)) * area_x,
            "outlet_y_low_minimum_normal_m_s": jnp.min(normal_low),
            "outlet_y_high_minimum_normal_m_s": jnp.min(normal_high),
            "outlet_y_low_backflow_area_fraction": jnp.mean(normal_low < 0.),
            "outlet_y_high_backflow_area_fraction": jnp.mean(normal_high < 0.),
            "outlet_y_low_backflow_area_fraction_below_001": jnp.mean(normal_low < -0.01),
            "outlet_y_high_backflow_area_fraction_below_001": jnp.mean(normal_high < -0.01),
            "outlet_y_reverse_volume_flux_m3_s": -jnp.sum(jnp.minimum(normal_low, 0.) + jnp.minimum(normal_high, 0.)) * area_y,
            "boundary_net_volume_flux_m3_s": jnp.sum(normal_x - u[..., 0])*area_x + jnp.sum(normal_low + normal_high)*area_y,
            "domain_minimum_u_m_s": jnp.min(u),
            "domain_negative_u_fraction": jnp.mean(u < 0.),
            "inlet_maximum_error_m_s": jnp.max(jnp.abs(u[..., 0] - speed)),
            "upstream_hub_minimum_u_m_s": jnp.min(profile),
            "upstream_hub_maximum_u_m_s": jnp.max(profile),
            "upstream_hub_second_difference_rms_m_s": jnp.sqrt(jnp.mean(jnp.diff(profile, n=2, axis=1)**2)),
        }

    return Simulation(
        case, grid, initial, checked_advance,
        courant,
        turbine_diagnostics=jax.jit(farm.diagnostics),
        state_diagnostics=state_diagnostics,
    )
