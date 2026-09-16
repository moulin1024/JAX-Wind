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
            scheme="fast-rk3",
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

    return Simulation(
        case, grid, initial, checked_advance,
        courant,
        turbine_diagnostics=jax.jit(farm.diagnostics),
    )
