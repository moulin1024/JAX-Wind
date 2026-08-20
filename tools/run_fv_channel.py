"""Drive the staggered finite-volume solver on a wall-bounded channel.

The default case is a forced plane channel with no-slip walls and the AMD
subfilter model, which exercises every part of the solver: conservative
transport, the wall closure, the subfilter stress and the pressure solve.
Use ``--backend amg`` for the JAX-AMG pressure solve on a GPU, ``--backend
gmg`` for a
matrix-free geometric multigrid V-cycle built straight from the mesh, or
``--backend fft`` for a direct solve that is exact but only valid because the
horizontal boundary is periodic (which it always is here).

    python tools/run_fv_channel.py --cells 64 64 48 --steps 200 --backend amg

The AMG backend needs jaxamg built against AmgX, and two environment settings,
because AmgX allocates on the device outside the JAX memory pool:

    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export LD_LIBRARY_PATH=$AMGX_BUILD:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

Neither ``gmg`` nor ``fft`` needs any of that -- both run on plain JAX.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    AnisotropicMinimumDissipation,
    Boundaries,
    FlowModel,
    StaggeredVelocity,
    build_pressure_poisson,
    build_run,
    build_step,
    courant_number,
    divergence,
    eddy_viscosity,
    initial_solution,
    stable_timestep,
)


def perturbed_channel(grid: UniformGrid, seed: int) -> StaggeredVelocity:
    """A laminar profile with divergence-free noise to trip the transition."""
    height = (jnp.arange(grid.nz) + 0.5) * grid.dz
    profile = 6.0 * height * (grid.lz - height) / grid.lz**2
    cells = (grid.nz, grid.ny, grid.nx)
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    amplitude = 0.1
    return StaggeredVelocity(
        jnp.broadcast_to(profile[:, None, None], cells)
        + amplitude * jax.random.normal(keys[0], cells),
        amplitude * jax.random.normal(keys[1], cells),
        (amplitude * jax.random.normal(keys[2], (grid.nz + 1, grid.ny, grid.nx)))
        .at[0]
        .set(0.0)
        .at[-1]
        .set(0.0),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs=3, default=(32, 32, 32))
    parser.add_argument("--lengths", type=float, nargs=3, default=(2.0, 1.0, 1.0))
    parser.add_argument("--viscosity", type=float, default=1.0e-3)
    parser.add_argument("--forcing", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="fixed step size; defaults to the stability limit of the start field",
    )
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument("--backend", choices=("amg", "gmg", "fft"), default="amg")
    parser.add_argument("--scheme", choices=("rk3", "fast-rk3", "ab2"), default="rk3")
    parser.add_argument("--direct", action="store_true", help="disable the AMD model")
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    # Must precede the first array: single precision means leaving x64 off, so
    # every field, the assembled matrix and the solve share one dtype.
    jax.config.update("jax_enable_x64", arguments.precision == "float64")
    grid = UniformGrid(*arguments.cells, *arguments.lengths)
    boundaries = Boundaries()
    subfilter = None if arguments.direct else AnisotropicMinimumDissipation()
    model = FlowModel(
        viscosity=arguments.viscosity,
        body_force=(arguments.forcing, 0.0, 0.0),
        subfilter=subfilter,
    )
    poisson = build_pressure_poisson(
        grid,
        backend=arguments.backend,
        dtype=arguments.precision,
    )
    run = build_run(build_step(grid, boundaries, poisson, model, scheme=arguments.scheme))

    solution = initial_solution(
        grid,
        perturbed_channel(grid, arguments.seed),
        dtype=arguments.precision,
    )
    turbulent = (
        0.0
        if subfilter is None
        else float(jnp.max(eddy_viscosity(solution.velocity, grid, boundaries, subfilter)))
    )
    dt = (
        arguments.dt
        if arguments.dt is not None
        else float(
            stable_timestep(solution.velocity, grid, arguments.viscosity + turbulent)
        )
    )
    print(
        f"mesh {grid.nx}x{grid.ny}x{grid.nz}  backend {arguments.backend}  "
        f"scheme {arguments.scheme}  {arguments.precision}  dt {dt:.3e}  "
        f"unknowns {grid.cell_count:,}"
    )

    remaining = arguments.steps
    while remaining > 0:
        block = min(arguments.report_every, remaining)
        start = time.perf_counter()
        solution = run(solution, dt, block)
        jax.block_until_ready(solution.velocity.x)
        elapsed = time.perf_counter() - start
        remaining -= block
        residual = float(jnp.max(jnp.abs(divergence(solution.velocity, grid))))
        turbulent = (
            0.0
            if subfilter is None
            else float(
                jnp.max(eddy_viscosity(solution.velocity, grid, boundaries, subfilter))
            )
        )
        print(
            f"step {int(solution.step):6d}  t {float(solution.time):8.4f}  "
            f"bulk {float(jnp.mean(solution.velocity.x)):8.5f}  "
            f"CFL {float(courant_number(solution.velocity, grid, dt)):6.3f}  "
            f"max div {residual:8.2e}  max nu_t {turbulent:9.3e}  "
            f"{1000 * elapsed / block:7.2f} ms/step"
        )


if __name__ == "__main__":
    main()
