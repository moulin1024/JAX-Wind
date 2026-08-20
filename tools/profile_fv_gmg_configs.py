#!/usr/bin/env python3
"""Benchmark GMG configurations on a turbine-generated pressure RHS."""

from __future__ import annotations

import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from applications.fv_abl.workflow import (
        _build_turbine_forcing,
        _load_inflow_block,
        _load_solution,
        _models,
        load_workflow,
    )
    from jaxwind.fv import (
        build_gmg_solver,
        build_tendency,
        divergence,
        enforce_open_velocity,
    )
    from jaxwind.fv.poisson import _apply_laplacian

    workflow = load_workflow(arguments.config)
    case = workflow.case.physical
    grid = case.physical_grid
    output = workflow.options.output_directory
    solution = _load_solution(output / "main_final.npz", jnp)
    step_index = int(solution.step)
    inflow = _load_inflow_block(output / "precursor_inflow", step_index, step_index + 1, jnp)
    inflow = type(inflow)(*(component[0] for component in inflow))
    forcing = _build_turbine_forcing(workflow)
    boundaries, momentum, _, _, _ = _models(
        workflow.case,
        periodic_x=False,
        forcing=forcing,
        pressure_force_enabled=workflow.options.main_pressure_force,
        evolve_scalar=False,
    )
    tendency = build_tendency(grid, boundaries, momentum)
    velocity = enforce_open_velocity(solution.velocity, inflow, grid)
    current = tendency(velocity, solution.time)
    dt = jnp.asarray(case.dt_seconds, velocity.x.dtype)
    candidate = type(velocity)(
        velocity.x + dt * (1.5 * current.x - 0.5 * solution.momentum_tendency.x),
        velocity.y + dt * (1.5 * current.y - 0.5 * solution.momentum_tendency.y),
        velocity.z + dt * (1.5 * current.z - 0.5 * solution.momentum_tendency.z),
    )
    candidate = enforce_open_velocity(candidate, inflow, grid)
    right_hand_side = -divergence(candidate, grid).reshape(-1) / dt
    rhs_norm = float(jnp.linalg.norm(right_hand_side))
    initial = solution.pressure.reshape(-1)
    configurations = (
        ("2+2_tol1e-6", dict(presweeps=2, postsweeps=2, tolerance=1.0e-6)),
        ("2+1_tol1e-6", dict(presweeps=2, postsweeps=1, tolerance=1.0e-6)),
        ("1+2_tol1e-6", dict(presweeps=1, postsweeps=2, tolerance=1.0e-6)),
        ("2+2_tol1e-5", dict(presweeps=2, postsweeps=2, tolerance=1.0e-5)),
        ("2+1_tol1e-5", dict(presweeps=2, postsweeps=1, tolerance=1.0e-5)),
        ("1+1_tol1e-5", dict(presweeps=1, postsweeps=1, tolerance=1.0e-5)),
    )
    for name, settings in configurations:
        solve = jax.jit(build_gmg_solver(
            grid,
            dtype=case.pressure.dtype,
            periodic_x=False,
            **settings,
        ))
        pressure = solve(right_hand_side, initial)
        jax.block_until_ready(pressure)
        started = time.perf_counter()
        for _ in range(arguments.repeats):
            pressure = solve(right_hand_side, initial)
        jax.block_until_ready(pressure)
        elapsed = time.perf_counter() - started
        residual = _apply_laplacian(
            pressure.reshape((grid.nz, grid.ny, grid.nx)), grid, periodic_x=False
        ).reshape(-1) - right_hand_side
        print(
            f"{name}: ms/solve={1e3 * elapsed / arguments.repeats:.3f} "
            f"relative_residual={float(jnp.linalg.norm(residual)) / rhs_norm:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
