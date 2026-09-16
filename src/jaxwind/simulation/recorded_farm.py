"""Replay a turbine-free periodic precursor into a nonperiodic AD-BEM farm."""
from dataclasses import replace
import jax
import jax.numpy as jnp

from jaxwind import (OPEN, StaggeredVelocity, build_pressure_poisson,
                     build_open_atmospheric_step, initial_atmospheric_solution, project)
from jaxwind.open_boundary import enforce_open_velocity
from jaxwind.config.abl import load_fv_abl
from .abl import build_models
from .wind_farm import ControlledFarm


def open_plane(plane, velocity_scale=1.):
    # The recorded v plane contains ny periodic faces. Give the receiving
    # plane its distinct high-y face. This is an input-format conversion, not
    # periodic coupling of the main solution. Supports single planes/batches.
    return plane._replace(x_velocity=plane.x_velocity * velocity_scale,
        y_velocity=jnp.concatenate((plane.y_velocity, plane.y_velocity[..., :1]), axis=-1) * velocity_scale,
        z_velocity=plane.z_velocity * velocity_scale)


def build_recorded_farm(case, warm, first, *, velocity_scale=1.):
    configured = load_fv_abl(case)
    grid = configured.physical.physical_grid
    physics = case.document["physics"]
    if "inflow" in physics or "surface_scalar" in physics or "cooling" in physics:
        raise ValueError("recorded farms require neutral precursor replay, without other inflow/cooling")
    if any(physics["flow"]["pressure_acceleration_m_s2"]) or any(physics["flow"]["coriolis_s"]):
        raise ValueError("recorded farm main requires zero body-pressure forcing and Coriolis")
    if physics["scalar"]["buoyancy_acceleration_per_unit"] != 0.:
        raise ValueError("recorded farm currently supports neutral conditions only")
    if configured.options.time_integration != "fast-rk3" or configured.options.pressure_backend != "gmg":
        raise ValueError("recorded farm requires fast-rk3 and GMG")
    if physics["wind_farm"]["controller"]["model"] != "wind-speed-lookup":
        raise ValueError("recorded farm requires wind-speed-lookup control")
    if warm.velocity.x.shape[-1] != grid.nx or warm.velocity.y.shape[1] != grid.ny:
        raise ValueError("recorded farm initialization must be horizontally periodic")
    first = open_plane(first, velocity_scale)
    velocity = StaggeredVelocity(
        jnp.concatenate((warm.velocity.x, warm.velocity.x[..., :1]), axis=2),
        jnp.concatenate((warm.velocity.y, warm.velocity.y[:, :1]), axis=1),
        warm.velocity.z,
    )
    velocity = StaggeredVelocity(*(component * velocity_scale for component in velocity))
    velocity = enforce_open_velocity(velocity, first, grid, open_y=True)
    options = configured.options
    poisson = build_pressure_poisson(grid, backend="gmg", periodic_x=False,
        periodic_y=False, open_y=True, dtype=configured.physical.dtype,
        config={"tolerance": options.gmg_tolerance or 1.e-5,
                "presweeps": options.gmg_presweeps, "postsweeps": options.gmg_postsweeps,
                "anisotropy_aware": options.gmg_anisotropy_aware})
    velocity, _ = project(velocity, poisson, 1.)
    initial = initial_atmospheric_solution(grid, velocity, warm.scalar,
                                          dtype=configured.physical.dtype)
    # The generic initializer zeros nonperiodic y-normal faces as walls.
    # Restore the already projected open-y field explicitly.
    initial = initial._replace(velocity=velocity)
    farm = ControlledFarm(case, open_domain=True)
    boundaries, momentum, _, _, _ = build_models(configured, periodic_x=False,
        pressure_force_enabled=False, evolve_scalar=False)
    boundaries = replace(boundaries, spanwise=OPEN)

    def step(state, dt, plane):
        def factory(source):
            integrate = build_open_atmospheric_step(grid, boundaries, poisson,
                replace(momentum, forcing=source), None, scheme="fast-rk3")
            return lambda current, active_dt: integrate(current, active_dt, plane)
        return farm.couple_step(factory)(state, dt)

    @jax.jit
    def advance(state, dt, planes):
        return jax.lax.scan(lambda current, plane: (step(current, dt, plane), None),
                            state, open_plane(planes, velocity_scale))[0]

    return farm.initialize(initial), advance, jax.jit(farm.diagnostics)
