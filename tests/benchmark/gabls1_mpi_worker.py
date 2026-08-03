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
    from jaxwind.momentum import morinishi_s4_advection
    from jaxwind.pressure import (
        BoundaryCondition,
        GMGConfig,
        MACVelocity,
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
    local_y = count // size
    start = rank * local_y
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
        discretization="kep4",
    )
    global_pressure = jnp.sin(0.013 * jnp.arange(count**3, dtype=jnp.float32)).reshape(
        (count, count, count)
    )
    local_pressure = global_pressure[:, start : start + local_y, :][None]
    local_action = pressure.apply(local_pressure)
    action_parts = communicator.gather(np.asarray(local_action[0]), root=0)
    if rank == 0:
        distributed_action = np.concatenate(action_parts, axis=1)
        reference_action = np.asarray(pressure.operator.apply(global_pressure))
        operator_difference = float(
            np.max(np.abs(distributed_action - reference_action))
        )
    else:
        operator_difference = None
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
        coupling_integrator="coupled-ssprk3",
        projection_method="fpj2",
    )
    velocity_keys = jax.random.split(jax.random.PRNGKey(1998), 3)
    global_u = jax.random.normal(
        velocity_keys[0],
        (count, count, count + 1),
        dtype=jnp.float32,
    )
    global_u = global_u.at[..., -1].set(global_u[..., 0])
    global_v = jax.random.normal(
        velocity_keys[1],
        (count, count + 1, count),
        dtype=jnp.float32,
    )
    global_v = global_v.at[:, -1, :].set(global_v[:, 0, :])
    global_w = jax.random.normal(
        velocity_keys[2],
        (count + 1, count, count),
        dtype=jnp.float32,
    )
    global_w = global_w.at[0].set(0.0).at[-1].set(0.0)
    global_velocity = MACVelocity(global_u, global_v, global_w)
    local_velocity = YSlabMACVelocity(
        global_u[:, start : start + local_y, :][None],
        global_v[:, start : start + local_y + 1, :][None],
        global_w[:, start : start + local_y, :][None],
    )

    def local_s4_transport(local):
        padded = coupled._pad_velocity(local)
        tendency = coupled.momentum_kernel.kep4_advection(padded)
        return coupled._crop_velocity(tendency)

    mapped_s4_transport = jax.pmap(
        local_s4_transport,
        **pressure.pmap_options,
    )
    local_s4 = mapped_s4_transport(local_velocity)
    s4_parts = communicator.gather(
        tuple(np.asarray(component[0]) for component in local_s4),
        root=0,
    )
    if rank == 0:
        distributed_s4 = MACVelocity(
            np.concatenate(tuple(part[0] for part in s4_parts), axis=1),
            np.concatenate(
                tuple(part[1][:, :-1, :] for part in s4_parts)
                + (s4_parts[-1][1][:, -1:, :],),
                axis=1,
            ),
            np.concatenate(tuple(part[2] for part in s4_parts), axis=1),
        )
        reference_s4 = morinishi_s4_advection(
            global_velocity,
            dx=coupled.dx,
            dy=coupled.dy,
            dz=coupled.dz,
        )
        momentum_operator_difference = max(
            float(np.max(np.abs(actual - np.asarray(reference))))
            for actual, reference in zip(
                distributed_s4,
                reference_s4,
                strict=True,
            )
        )
        if momentum_operator_difference > 2.0e-6:
            raise AssertionError(
                "distributed Morinishi S4 does not match the serial operator: "
                f"{momentum_operator_difference:.3e}"
            )
    else:
        momentum_operator_difference = None
    z = (jnp.arange(count, dtype=jnp.float32) + 0.5) * (400.0 / count)
    theta_profile = jnp.where(z <= 100.0, 265.0, 265.0 + 0.01 * (z - 100.0))
    random = jax.random.uniform(
        jax.random.PRNGKey(0),
        (count, count, count),
        minval=-0.1,
        maxval=0.1,
    )
    random -= jnp.mean(random, axis=(1, 2), keepdims=True)
    global_theta = theta_profile[:, None, None] + random * (z < 50.0)[:, None, None]
    theta = global_theta[:, start : start + local_y, :][None]
    velocity = YSlabMACVelocity(
        jnp.full((1, count, local_y, count + 1), 8.0, dtype=jnp.float32),
        jnp.zeros((1, count, local_y + 1, count), dtype=jnp.float32),
        jnp.zeros((1, count + 1, local_y, count), dtype=jnp.float32),
    )
    state = coupled.initial_state(velocity, theta)
    projection_calls = 0
    project = coupled._project

    def counted_project(*args, **kwargs):
        nonlocal projection_calls
        projection_calls += 1
        return project(*args, **kwargs)

    coupled._project = counted_project
    advanced = state
    for _ in range(3):
        advanced = coupled.step(advanced, timestep=0.25)
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
                    "divergence_norm": coupled.divergence_norm(advanced.velocity),
                    "advective_rate": rates[0],
                    "momentum_diffusive_rate": rates[1],
                    "scalar_diffusive_rate": rates[2],
                    "surface_heat_flux": heat_flux_sum / (count * count),
                    "projection_calls": projection_calls,
                    "fpj2_history_count": coupled.fpj2_state.history_count,
                    "operator_difference": operator_difference,
                    "momentum_operator_difference": momentum_operator_difference,
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
