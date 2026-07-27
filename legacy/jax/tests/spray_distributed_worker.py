#!/usr/bin/env python3
"""Multi-process verification worker for shard-local spray migration."""

from __future__ import annotations

import argparse
import json
import math

import jax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    args = parser.parse_args()

    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=[0],
    )

    import jax.numpy as jnp

    from wireles_jax import (
        Params,
        SprayDPMConfig,
        initialize_sharded_spray,
        make_migrate_sharded_spray,
    )
    from wireles_jax.sharding import make_distributed_mesh

    capacity = 8
    params = Params(
        nx=4,
        ny=4,
        nz=8,
        lx=1.0,
        ly=1.0,
        lz=1.0,
        z_i=100.0,
        dt=0.001,
        thermo_enabled=True,
        moisture_enabled=True,
        qv0=0.0,
        scalar_sgs_model="fixed_prandtl",
        scalar_vertical_scheme="centered",
        momentum_wall_model="free_slip",
        sgs_model="smagorinsky",
        dtype=jnp.float32,
    )
    config = SprayDPMConfig(
        max_parcels=capacity,
        initial_parcels=1,
        injection_z=1.0,
        initial_diameter=100.0e-6,
        initial_temperature=285.0,
        sky_temperature=285.0,
        parcel_weight=1.0e8,
        substeps=1,
        turbulent_dispersion_enabled=True,
    )
    mesh = make_distributed_mesh(args.num_processes)
    spray = initialize_sharded_spray(config, params, mesh, seed=19)

    local_shards = spray.x.addressable_shards
    assert len(local_shards) == 1
    assert local_shards[0].data.shape == (capacity,)
    if args.num_processes > 1:
        assert local_shards[0].data.size < spray.x.size

    initial_mass = jnp.sum(
        spray.mass * spray.weight * spray.active.astype(spray.mass.dtype)
    )
    # Exercise the worst in-domain path: shard zero to the last shard.  The
    # distributed elementwise update acts only on the one initially active
    # parcel and never gathers its global buffer on a host.
    target_z = (
        50.0
        if args.num_processes == 1
        else 100.0 * (args.num_processes - 1) / args.num_processes
    )
    spray = spray._replace(z=jnp.where(spray.active, target_z, spray.z))
    migrate = jax.jit(make_migrate_sharded_spray(config, params, mesh))
    migrated, diagnostics = jax.block_until_ready(migrate(spray))

    local_active = int(jnp.sum(migrated.active.addressable_shards[0].data))
    expected_local = 1 if args.process_id == args.num_processes - 1 else 0
    assert local_active == expected_local
    assert int(diagnostics.active_parcels) == 1
    assert int(diagnostics.exited_parcels) == 0
    assert int(diagnostics.overflow_parcels) == 0
    assert float(diagnostics.liquid_mass) == float(initial_mass)

    from wireles_jax import (
        ShardedSprayCoupledState,
        make_spray_exchange_sharded,
        make_step_spray_dpm_sharded,
    )
    from wireles_jax.timestep_sharded import (
        initial_sharded_state,
        make_diagnostics_sharded,
        make_sharded_operators,
    )

    flow = initial_sharded_state(params, mesh)
    x_phase = jnp.arange(params.nx, dtype=params.dtype)[:, None, None]
    resolved_v = 3.0 * jnp.sin(2.0 * jnp.pi * x_phase / params.nx)
    flow = flow._replace(v=flow.v + jnp.broadcast_to(resolved_v, flow.v.shape))
    exchange = jax.jit(make_spray_exchange_sharded(config, params, mesh))
    migrated, increments, exchange_diagnostics = jax.block_until_ready(
        exchange(flow, migrated)
    )
    cell_mass = (
        config.air_density
        * params.dx
        * params.z_i
        * params.dy
        * params.z_i
        * params.dz
        * params.z_i
    )
    deposited_vapor = float(jnp.sum(increments.qv) * cell_mass)
    local_vapor = float(
        jnp.sum(increments.qv.addressable_shards[0].data) * cell_mass
    )
    local_sgs_speed = float(
        jnp.max(
            jnp.sqrt(
                migrated.sgs_u.addressable_shards[0].data**2
                + migrated.sgs_v.addressable_shards[0].data**2
                + migrated.sgs_w.addressable_shards[0].data**2
            )
            * migrated.active.addressable_shards[0].data
        )
    )
    current_local_active = int(
        jnp.sum(migrated.active.addressable_shards[0].data)
    )
    evaporated_mass = float(exchange_diagnostics.evaporated_mass)
    assert evaporated_mass > 0.0
    assert math.isclose(
        deposited_vapor, evaporated_mass, rel_tol=4.0e-6
    )
    if args.num_processes > 1:
        has_boundary_source = args.process_id >= args.num_processes - 2
        assert (local_vapor > 0.0) == has_boundary_source
    assert (local_sgs_speed > 0.0) == (current_local_active > 0), (
        args.process_id,
        local_sgs_speed,
        current_local_active,
    )

    operators = make_sharded_operators(params, mesh)
    coupled_step = jax.jit(
        make_step_spray_dpm_sharded(config, params, operators, mesh)
    )
    coupled, coupled_diagnostics = jax.block_until_ready(
        coupled_step(
            ShardedSprayCoupledState(flow=flow, spray=migrated),
            operators.pressure,
            operators.pressure_spike,
        )
    )
    flow_diagnostics = make_diagnostics_sharded(
        params, operators.horizontal, mesh
    )
    carrier_diagnostics = jax.block_until_ready(
        flow_diagnostics(
            coupled.flow.u,
            coupled.flow.v,
            coupled.flow.w,
            coupled.flow.theta,
            coupled.flow.qv,
            coupled.flow.step,
        )
    )
    assert int(coupled.flow.step) == 1
    assert float(coupled_diagnostics.evaporated_mass) > 0.0
    assert bool(jnp.isfinite(carrier_diagnostics.div_max))
    assert float(carrier_diagnostics.qv_min) >= params.qv_floor

    print(
        json.dumps(
            {
                "process_id": args.process_id,
                "num_processes": args.num_processes,
                "global_shape": migrated.x.shape,
                "local_shape": local_shards[0].data.shape,
                "local_active": local_active,
                "active_parcels": int(diagnostics.active_parcels),
                "deposited_vapor": deposited_vapor,
                "local_vapor": local_vapor,
                "local_sgs_speed": local_sgs_speed,
                "coupled_step": int(coupled.flow.step),
                "div_max": float(carrier_diagnostics.div_max),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
