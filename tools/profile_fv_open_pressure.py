#!/usr/bin/env python3
"""Benchmark the open-boundary FV pressure solve on a workflow checkpoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path



def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("config", type=Path)
    result.add_argument("--repeats", type=int, default=10)
    result.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=(1.0e-6, 1.0e-5, 1.0e-4),
    )
    return result


def main() -> int:
    arguments = parser().parse_args()

    import jax

    from applications.fv_abl.workflow import (
        _load_inflow_block,
        _load_solution,
        load_workflow,
    )
    from jaxwind.fv import (
        build_gmg_solver,
        divergence,
        enforce_open_velocity,
        periodic_to_open_velocity,
    )

    workflow = load_workflow(arguments.config)
    case = workflow.case.physical
    grid = case.physical_grid
    output = workflow.options.output_directory
    warm = _load_solution(output / "warmup_checkpoint.npz", jax.numpy)
    inflow = _load_inflow_block(output / "precursor_inflow", 0, 1, jax.numpy)
    inflow = type(inflow)(*(component[0] for component in inflow))
    velocity = periodic_to_open_velocity(warm.velocity, grid)
    velocity = enforce_open_velocity(velocity, inflow, grid)
    right_hand_side = -divergence(velocity, grid).reshape(-1) / case.dt_seconds
    rhs_norm = float(jax.numpy.linalg.norm(right_hand_side))

    print(
        f"grid={grid.nx}x{grid.ny}x{grid.nz} "
        f"spacing={grid.dx:g}x{grid.dy:g}x{grid.dz:g} m "
        f"rhs_l2={rhs_norm:.6e}"
    )
    for tolerance in arguments.tolerances:
        solve = build_gmg_solver(
            grid,
            dtype=case.pressure.dtype,
            periodic_x=False,
            tolerance=tolerance,
        )
        compiled = jax.jit(solve)
        pressure = compiled(right_hand_side)
        jax.block_until_ready(pressure)

        started = time.perf_counter()
        for _ in range(arguments.repeats):
            pressure = compiled(right_hand_side)
        jax.block_until_ready(pressure)
        elapsed = time.perf_counter() - started

        applied = -divergence(
            __import__("jaxwind.fv", fromlist=["pressure_gradient"]).pressure_gradient(
                pressure.reshape((grid.nz, grid.ny, grid.nx)),
                grid,
                periodic_x=False,
            ),
            grid,
        ).reshape(-1)
        relative = float(jax.numpy.linalg.norm(applied - right_hand_side)) / rhs_norm
        print(
            f"tol={tolerance:.1e} ms/solve={1.0e3 * elapsed / arguments.repeats:.3f} "
            f"relative_residual={relative:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
