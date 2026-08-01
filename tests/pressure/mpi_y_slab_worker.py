from __future__ import annotations

import json
import os

from mpi4py import MPI
import numpy as np


def main() -> None:
    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    size = communicator.Get_size()
    if size != 4:
        raise RuntimeError("MPI y-slab worker requires exactly four ranks")

    import jax
    import jax.numpy as jnp

    address = os.environ["JAXWIND_COORDINATOR_ADDRESS"]
    jax.distributed.initialize(
        coordinator_address=address,
        num_processes=size,
        process_id=rank,
        local_device_ids=[0],
    )

    from jaxwind.pressure import (
        BoundaryCondition,
        PCGConfig,
        PoissonBoundaryConditions,
        RectilinearGrid,
        YSlabConfig,
        YSlabMatrixFreePoissonSolver,
    )

    grid = RectilinearGrid.uniform(8, 8, 16, lx=1.0, ly=1.0, lz=1.0)
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    solver = YSlabMatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions(
            periodic,
            periodic,
            periodic,
            periodic,
            neumann,
            neumann,
        ),
        dtype=jnp.float32,
        krylov=PCGConfig(
            max_iterations=60,
            relative_tolerance=2.0e-6,
        ),
        distribution=YSlabConfig(coarse_cells_per_device=1),
    )
    z = (jnp.arange(16, dtype=jnp.float32) + 0.5) / 16.0
    y = (jnp.arange(8, dtype=jnp.float32) + 0.5) / 8.0
    x = (jnp.arange(8, dtype=jnp.float32) + 0.5) / 8.0
    exact = (
        jnp.cos(2.0 * jnp.pi * z)[:, None, None]
        * jnp.cos(2.0 * jnp.pi * y)[None, :, None]
        * jnp.cos(2.0 * jnp.pi * x)[None, None, :]
    )
    rhs = solver.operator.apply(exact)
    local_y = 8 // size
    start = rank * local_y
    local_exact = exact[:, start : start + local_y, :][None]
    local_rhs = rhs[:, start : start + local_y, :][None]
    apply_error = jnp.max(jnp.abs(solver.apply(local_exact) - local_rhs))
    result = solver.solve(local_rhs)
    local_error_squared = jnp.sum((result.solution - local_exact) ** 2)
    local_exact_squared = jnp.sum(local_exact**2)
    values = np.asarray((local_error_squared, local_exact_squared))
    totals = np.empty_like(values)
    communicator.Allreduce(values, totals, op=MPI.SUM)
    if rank == 0:
        print(
            json.dumps(
                {
                    "processes": jax.process_count(),
                    "global_devices": jax.device_count(),
                    "local_devices": jax.local_device_count(),
                    "apply_error": float(apply_error),
                    "relative_error": float(np.sqrt(totals[0] / totals[1])),
                    "converged": result.converged,
                    "relative_residual": result.relative_residual,
                }
            ),
            flush=True,
        )
    communicator.Barrier()
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
