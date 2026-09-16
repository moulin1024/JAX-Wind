"""Direct turbine simulation driven by a measured-profile Mann box."""

from __future__ import annotations

import numpy as np

from jaxwind.config.abl import load_fv_abl
from jaxwind.config.document import native_document
from jaxwind.config.stages import FiniteVolumeWorkflow, WorkflowOptions, _load_turbine
from jaxwind.config.synthetic_inflow import reference_profile
from jaxwind.inflow import build_mann_inflow, generate_mann_box

from .api import Simulation
from .open_atmospheric import build_open_components


def build_simulation(case):
    import jax
    import jax.numpy as jnp

    from jaxwind import StaggeredVelocity, courant_number, initial_atmospheric_solution

    configured = load_fv_abl(case)
    grid = configured.physical.physical_grid
    table = case.document["physics"]["inflow"]
    heights, speeds, intensities = reference_profile(table["reference_profile"])
    mean = np.interp(grid.z_centers, heights, speeds)
    ti = np.interp(grid.z_centers, heights, intensities)
    advection = float(np.interp(table["advection_height_m"], heights, speeds))
    duration = case.document["time"]["steps"] * case.document["time"]["dt_seconds"]
    if table["box_lengths_m"][0] / advection < duration:
        raise ValueError(
            "Mann box repeats before the requested run ends; increase box_lengths_m[0]"
        )
    print(
        f"Generating Mann box {table['box_cells']}; advection={advection:.6g} m/s, period={table['box_lengths_m'][0] / advection:.3f} s",
        flush=True,
    )
    box = generate_mann_box(
        shape=table["box_cells"],
        lengths=table["box_lengths_m"],
        length_scale=table["length_scale_m"],
        gamma=table["gamma"],
        seed=table["seed"],
    )
    inflow = build_mann_inflow(
        box,
        grid,
        mean_speed=advection,
        mean_profile=mean,
        sigma_u_profile=mean * ti,
        scalar=case.document["physics"]["scalar"]["reference_value"],
    )
    del box
    print(
        "Measured mean/TI profiles calibrated; constructing open AD-BEM simulation",
        flush=True,
    )
    dtype = configured.physical.dtype
    shape = (grid.nz, grid.ny, grid.nx)
    velocity = StaggeredVelocity(
        jnp.broadcast_to(jnp.asarray(mean, dtype)[:, None, None], shape),
        jnp.zeros(shape, dtype),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype),
    )
    warm = initial_atmospheric_solution(
        grid,
        velocity,
        jnp.full(shape, case.document["physics"]["scalar"]["reference_value"], dtype),
        dtype=dtype,
    )
    settings = case.document["time"]
    workflow = FiniteVolumeWorkflow(
        configured,
        WorkflowOptions(
            0,
            0,
            settings["steps"],
            0,
            settings.get("chunk_steps", 50),
            case.output,
            main_pressure_force=False,
            evolve_scalar=False,
        ),
        turbine=_load_turbine(native_document(case)),
    )
    initial, step = build_open_components(workflow, warm, inflow(0.0), return_step=True)
    dt = settings["dt_seconds"]

    @jax.jit(static_argnums=1)
    def advance(state, count):
        def one(current, unused):
            return step(current, dt, inflow(current.time)), None

        return jax.lax.scan(one, state, None, length=count)[0]

    courant = jax.jit(lambda state: courant_number(state.velocity, grid, dt))
    return Simulation(
        case,
        grid,
        initial,
        lambda state, controls: advance(state, controls.count),
        courant,
    )
