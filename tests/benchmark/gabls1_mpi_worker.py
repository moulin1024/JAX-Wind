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
        raise RuntimeError("GABLS1 MPI worker requires exactly four ranks")

    import jax
    import jax.numpy as jnp

    jax.distributed.initialize(
        coordinator_address=os.environ["JAXWIND_COORDINATOR_ADDRESS"],
        num_processes=size,
        process_id=rank,
        local_device_ids=[0],
    )

    from benchmark.GABLS1.distributed_solver import YSlabAMDBoussinesq
    from jaxwind.pressure import (
        BoundaryCondition,
        GMGConfig,
        PCGConfig,
        PoissonBoundaryConditions,
        RectilinearGrid,
        YSlabConfig,
        YSlabMACVelocity,
        YSlabMatrixFreePoissonSolver,
    )

    count = 16
    grid = RectilinearGrid.uniform(
        count,
        count,
        count,
        lx=400.0,
        ly=400.0,
        lz=400.0,
    )
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    pressure = YSlabMatrixFreePoissonSolver(
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
        gmg=GMGConfig(coarse_smooth=10),
        krylov=PCGConfig(
            max_iterations=40,
            relative_tolerance=1.0e-5,
        ),
        distribution=YSlabConfig(coarse_cells_per_device=1),
    )
    coupled = YSlabAMDBoussinesq(
        grid,
        pressure,
        geostrophic_wind=(8.0, 0.0),
        coriolis=1.39e-4,
        roughness_length=0.1,
        gravity=9.81,
        reference_potential_temperature=263.5,
        surface_potential_temperature=265.0,
        surface_temperature_tendency=-0.25 / 3600.0,
        amd_coefficient=0.212,
        scalar_amd_coefficient=0.212,
        mp5_strength=1.0,
    )
    local_y = count // size
    start = rank * local_y
    z = (jnp.arange(count, dtype=jnp.float32) + 0.5) * (400.0 / count)
    theta_profile = jnp.where(z <= 100.0, 265.0, 265.0 + 0.01 * (z - 100.0))
    random = jax.random.uniform(
        jax.random.PRNGKey(0),
        (count, count, count),
        minval=-0.1,
        maxval=0.1,
    )
    random -= jnp.mean(random, axis=(1, 2), keepdims=True)
    global_theta = theta_profile[:, None, None] + random * (
        z < 50.0
    )[:, None, None]
    theta = global_theta[:, start : start + local_y, :][None]
    velocity = YSlabMACVelocity(
        jnp.full((1, count, local_y, count + 1), 8.0, dtype=jnp.float32),
        jnp.zeros((1, count, local_y + 1, count), dtype=jnp.float32),
        jnp.zeros((1, count + 1, local_y, count), dtype=jnp.float32),
    )
    state = coupled.initial_state(velocity, theta)
    advanced = coupled.step(state, timestep=0.25)
    rates = coupled.rates(advanced)
    fluxes = coupled.surface_layer_fluxes(advanced)
    local_finite = int(
        all(
            np.all(np.isfinite(np.asarray(value)))
            for value in (
                advanced.velocity.x,
                advanced.velocity.y,
                advanced.velocity.z,
                advanced.potential_temperature,
                advanced.pressure,
            )
        )
    )
    finite = communicator.allreduce(local_finite, op=MPI.MIN)
    heat_flux_sum = communicator.allreduce(
        float(jnp.sum(fluxes.heat_flux)),
        op=MPI.SUM,
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "finite": bool(finite),
                    "step": advanced.step,
                    "time": advanced.time,
                    "divergence_norm": coupled.divergence_norm(
                        advanced.velocity
                    ),
                    "advective_rate": rates[0],
                    "momentum_diffusive_rate": rates[1],
                    "scalar_diffusive_rate": rates[2],
                    "surface_heat_flux": heat_flux_sum / (count * count),
                }
            ),
            flush=True,
        )
    else:
        coupled.divergence_norm(advanced.velocity)
    communicator.Barrier()
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
