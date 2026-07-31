#!/usr/bin/env python3
"""Run the single-host multi-device y-slab GMG pressure solver."""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark y-slab halo GMG with coarse-grid replication."
    )
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--z-stretch-exponent", type=float, default=1.8)
    parser.add_argument("--coarse-cells-per-device", type=int, default=4)
    parser.add_argument("--restart", type=int, default=30)
    parser.add_argument(
        "--linear-solver",
        choices=("pcg", "gmres"),
        default="pcg",
    )
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--single", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.nx, args.ny, args.nz) <= 0:
        raise SystemExit("grid dimensions must be positive")
    if args.z_stretch_exponent <= 0.0:
        raise SystemExit("--z-stretch-exponent must be positive")

    from jax import config as jax_config

    if not args.single:
        jax_config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaxwind.pressure import (
        BoundaryCondition,
        FGMRESConfig,
        PCGConfig,
        RectilinearGrid,
        PoissonBoundaryConditions,
        YSlabConfig,
        YSlabMACProjector,
        YSlabMatrixFreePoissonSolver,
        gather_y_field,
        shard_y_field,
    )

    available = tuple(jax.local_devices())
    count = len(available) if args.devices is None else args.devices
    if count <= 0 or count > len(available):
        raise SystemExit(
            f"requested {count} devices, but {len(available)} are visible"
        )
    devices = available[:count]
    if args.ny % count:
        raise SystemExit("ny must be divisible by the selected device count")
    dtype = jnp.float32 if args.single else jnp.float64

    def uniform(cell_count: int) -> tuple[float, ...]:
        return tuple(np.linspace(0.0, 1.0, cell_count + 1))

    z_unit = np.linspace(0.0, 1.0, args.nz + 1)
    grid = RectilinearGrid(
        uniform(args.nx),
        uniform(args.ny),
        tuple(z_unit**args.z_stretch_exponent),
    )
    periodic = BoundaryCondition("periodic")
    dirichlet = BoundaryCondition("dirichlet")
    boundaries = PoissonBoundaryConditions(
        periodic,
        periodic,
        periodic,
        periodic,
        dirichlet,
        dirichlet,
    )
    krylov = (
        PCGConfig(
            max_iterations=args.max_iterations,
            relative_tolerance=args.rtol,
        )
        if args.linear_solver == "pcg"
        else FGMRESConfig(
            restart=args.restart,
            max_iterations=args.max_iterations,
            relative_tolerance=args.rtol,
        )
    )
    solver = YSlabMatrixFreePoissonSolver(
        grid,
        boundaries,
        devices=devices,
        dtype=dtype,
        krylov=krylov,
        distribution=YSlabConfig(
            coarse_cells_per_device=args.coarse_cells_per_device
        ),
    )

    def centers(faces: tuple[float, ...]) -> jax.Array:
        values = jnp.asarray(faces, dtype=dtype)
        return 0.5 * (values[1:] + values[:-1])

    x = centers(grid.x_faces)
    y = centers(grid.y_faces)
    z = centers(grid.z_faces)
    exact = (
        jnp.sin(jnp.pi * z)[:, None, None]
        * jnp.cos(2.0 * jnp.pi * y)[None, :, None]
        * jnp.cos(2.0 * jnp.pi * x)[None, None, :]
    )
    rhs = solver.operator.apply(exact)
    sharded_rhs = shard_y_field(rhs, count)
    start = time.perf_counter()
    result = solver.solve(sharded_rhs)
    jax.block_until_ready(result.solution)
    elapsed = time.perf_counter() - start
    solution = gather_y_field(result.solution)
    relative_error = float(
        solver.operator.norm(solution - exact) / solver.operator.norm(exact)
    )

    projector = YSlabMACProjector(solver)
    gradient_velocity = projector.gradient(shard_y_field(exact, count))
    stage = projector.project(gradient_velocity, timestep=1.0)
    divergence = gather_y_field(stage.divergence_after)
    divergence_norm = float(solver.operator.norm(divergence))

    print(
        f"backend={jax.default_backend()} devices={count} dtype={dtype} "
        f"shape_zyx={grid.shape}"
    )
    print(
        f"levels={solver.level_shapes} replication_level="
        f"{solver.replication_level} replicated_shape={solver.replicated_shape}"
    )
    print(
        f"smoothers={solver.serial_gmg.level_smoothers} "
        f"coarsening={solver.serial_gmg.coarsening_factors}"
    )
    print(
        f"converged={result.converged} iterations={result.iterations} "
        f"relative_residual={result.relative_residual:.6e} "
        f"relative_error={relative_error:.6e}"
    )
    print(
        f"mac_stage_divergence_norm={divergence_norm:.6e} "
        f"elapsed_seconds={elapsed:.6f}"
    )
    if not result.converged or not stage.linear_result.converged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
