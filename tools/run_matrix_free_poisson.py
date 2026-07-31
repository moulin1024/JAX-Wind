#!/usr/bin/env python3
"""Run the standalone matrix-free symmetric-GMG Poisson solver."""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a discrete manufactured FV Poisson problem with a "
            "matrix-free symmetric-GMG-preconditioned Krylov solver."
        )
    )
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--lx", type=float, default=1.0)
    parser.add_argument("--ly", type=float, default=1.0)
    parser.add_argument("--lz", type=float, default=1.0)
    parser.add_argument(
        "--boundary",
        choices=("periodic", "dirichlet", "neumann"),
        default="dirichlet",
    )
    parser.add_argument(
        "--z-stretch-exponent",
        type=float,
        default=1.0,
        help="power-law mapping z/L=(index/nz)^exponent",
    )
    parser.add_argument("--restart", type=int, default=30)
    parser.add_argument(
        "--linear-solver",
        choices=("pcg", "gmres"),
        default="pcg",
    )
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument(
        "--smoother",
        choices=("auto", "jacobi", "z_line"),
        default="auto",
    )
    parser.add_argument(
        "--coarsening",
        choices=("auto", "full", "z_semi"),
        default="auto",
    )
    parser.add_argument("--anisotropy-threshold", type=float, default=4.0)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--no-jit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.nx, args.ny, args.nz) <= 0:
        raise SystemExit("grid dimensions must be positive")
    if min(args.lx, args.ly, args.lz) <= 0.0:
        raise SystemExit("domain lengths must be positive")
    if args.z_stretch_exponent <= 0.0:
        raise SystemExit("--z-stretch-exponent must be positive")

    from jax import config as jax_config

    if not args.single:
        jax_config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaxwind.pressure import (
        FGMRESConfig,
        GMGConfig,
        MatrixFreePoissonSolver,
        PCGConfig,
        PoissonBoundaryConditions,
        RectilinearGrid,
    )

    dtype = jnp.float32 if args.single else jnp.float64
    x_faces = tuple(np.linspace(0.0, args.lx, args.nx + 1))
    y_faces = tuple(np.linspace(0.0, args.ly, args.ny + 1))
    z_unit = np.linspace(0.0, 1.0, args.nz + 1)
    z_faces = tuple(args.lz * z_unit**args.z_stretch_exponent)
    grid = RectilinearGrid(x_faces, y_faces, z_faces)
    boundary_factory = {
        "periodic": PoissonBoundaryConditions.periodic,
        "dirichlet": PoissonBoundaryConditions.homogeneous_dirichlet,
        "neumann": PoissonBoundaryConditions.homogeneous_neumann,
    }[args.boundary]
    krylov = (
        PCGConfig(
            max_iterations=args.max_iterations,
            relative_tolerance=args.rtol,
            jit_kernels=not args.no_jit,
        )
        if args.linear_solver == "pcg"
        else FGMRESConfig(
            restart=args.restart,
            max_iterations=args.max_iterations,
            relative_tolerance=args.rtol,
            jit_kernels=not args.no_jit,
        )
    )
    solver = MatrixFreePoissonSolver(
        grid,
        boundary_factory(),
        dtype=dtype,
        gmg=GMGConfig(
            smoother=args.smoother,
            coarsening=args.coarsening,
            anisotropy_threshold=args.anisotropy_threshold,
        ),
        krylov=krylov,
    )

    def centers(faces: tuple[float, ...]) -> jax.Array:
        values = jnp.asarray(faces, dtype=dtype)
        return 0.5 * (values[1:] + values[:-1])

    x = centers(grid.x_faces) / args.lx
    y = centers(grid.y_faces) / args.ly
    z = centers(grid.z_faces) / args.lz
    if args.boundary == "dirichlet":
        exact = (
            jnp.sin(jnp.pi * z)[:, None, None]
            * jnp.sin(jnp.pi * y)[None, :, None]
            * jnp.sin(jnp.pi * x)[None, None, :]
        )
    elif args.boundary == "neumann":
        exact = (
            jnp.cos(jnp.pi * z)[:, None, None]
            * jnp.cos(jnp.pi * y)[None, :, None]
            * jnp.cos(jnp.pi * x)[None, None, :]
        )
    else:
        exact = (
            jnp.cos(2.0 * jnp.pi * x)[None, None, :]
            + 0.4 * jnp.cos(4.0 * jnp.pi * y)[None, :, None]
            + 0.2 * jnp.cos(2.0 * jnp.pi * z)[:, None, None]
        )
    exact = solver.operator.project_nullspace(exact)
    rhs = solver.operator.apply(exact)

    start = time.perf_counter()
    result = solver.solve(rhs)
    jax.block_until_ready(result.solution)
    elapsed = time.perf_counter() - start
    error = solver.operator.project_nullspace(result.solution - exact)
    relative_error = float(
        solver.operator.norm(error) / solver.operator.norm(exact)
    )

    print(f"backend={jax.default_backend()} dtype={dtype}")
    print(f"shape_zyx={grid.shape} levels={solver.preconditioner.level_shapes}")
    print(
        f"smoothers={solver.preconditioner.level_smoothers} "
        f"coarsening={solver.preconditioner.coarsening_factors}"
    )
    print(
        f"converged={result.converged} iterations={result.iterations} "
        f"relative_residual={result.relative_residual:.6e}"
    )
    print(
        f"relative_solution_error={relative_error:.6e} "
        f"elapsed_seconds={elapsed:.6f}"
    )
    if not result.converged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
