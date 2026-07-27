#!/usr/bin/env python3
"""Run the fully distributed z-sharded spray/moist-LES coupling path."""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--lx", type=float, default=1000.0)
    parser.add_argument("--ly", type=float, default=1000.0)
    parser.add_argument("--lz", type=float, default=500.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--coordinator-address")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument("--local-device-ids", default="0")
    parser.add_argument("--wind-speed", type=float, default=8.0)
    parser.add_argument("--theta0", type=float, default=300.0)
    parser.add_argument("--qv0", type=float, default=0.005)
    parser.add_argument("--sgs-model", choices=("smagorinsky", "lasd"), default="smagorinsky")
    parser.add_argument("--max-parcels-per-shard", type=int, default=4096)
    parser.add_argument("--parcels-per-step", type=int, default=16)
    parser.add_argument("--mass-flow-rate", type=float, default=0.1)
    parser.add_argument("--injection-x", type=float)
    parser.add_argument("--injection-y", type=float)
    parser.add_argument("--injection-z", type=float, default=100.0)
    parser.add_argument("--injection-radius", type=float, default=10.0)
    parser.add_argument("--injection-u", type=float, default=8.0)
    parser.add_argument("--diameter", type=float, default=100.0e-6)
    parser.add_argument(
        "--diameter-distribution",
        choices=("monodisperse", "rosin-rammler", "lognormal"),
        default="monodisperse",
    )
    parser.add_argument("--minimum-diameter", type=float, default=10.0e-6)
    parser.add_argument("--maximum-diameter", type=float, default=500.0e-6)
    parser.add_argument("--rosin-rammler-spread", type=float, default=3.0)
    parser.add_argument("--lognormal-geometric-stddev", type=float, default=1.5)
    parser.add_argument("--turbulent-dispersion", action="store_true")
    parser.add_argument("--drop-temperature", type=float, default=283.15)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--shortwave-flux", type=float, default=0.0)
    parser.add_argument("--shortwave-absorption", type=float, default=0.0)
    parser.add_argument("--sky-temperature", type=float, default=273.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import jax

    if args.num_processes > 1:
        if args.coordinator_address is None:
            raise ValueError("multi-process runs require --coordinator-address")
        local_device_ids = [
            int(value) for value in args.local_device_ids.split(",") if value
        ]
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
            local_device_ids=local_device_ids,
        )

    import jax.numpy as jnp

    from wireles_jax import (
        Params,
        ShardedSprayCoupledState,
        SprayDPMConfig,
        initialize_sharded_spray,
        make_step_spray_dpm_sharded,
    )
    from wireles_jax.sharding import make_distributed_mesh, make_single_node_mesh
    from wireles_jax.timestep_sharded import (
        initial_sharded_state,
        make_diagnostics_sharded,
        make_sharded_operators,
    )

    scale = args.lz
    params = Params(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx / scale,
        ly=args.ly / scale,
        lz=1.0,
        z_i=scale,
        dt=args.dt / scale,
        nsteps=args.steps,
        c_count=args.log_every,
        initial_condition="geostrophic",
        geostrophic_u=args.wind_speed,
        momentum_wall_model="free_slip",
        initial_velocity_noise=0.0,
        time_scheme="ab2",
        thermo_enabled=True,
        moisture_enabled=True,
        theta0=args.theta0,
        qv0=args.qv0,
        scalar_sgs_model="fixed_prandtl",
        scalar_vertical_scheme="centered",
        sgs_model=args.sgs_model,
        dtype=jnp.float32,
    )
    injection_x = 0.25 * args.lx if args.injection_x is None else args.injection_x
    injection_y = 0.5 * args.ly if args.injection_y is None else args.injection_y
    spray_config = SprayDPMConfig(
        max_parcels=args.max_parcels_per_shard,
        parcels_per_step=args.parcels_per_step,
        mass_flow_rate=args.mass_flow_rate,
        injection_x=injection_x,
        injection_y=injection_y,
        injection_z=args.injection_z,
        injection_radius=args.injection_radius,
        injection_u=args.injection_u,
        initial_diameter=args.diameter,
        diameter_distribution=args.diameter_distribution,
        minimum_diameter=args.minimum_diameter,
        maximum_diameter=args.maximum_diameter,
        rosin_rammler_spread=args.rosin_rammler_spread,
        lognormal_geometric_stddev=args.lognormal_geometric_stddev,
        turbulent_dispersion_enabled=args.turbulent_dispersion,
        initial_temperature=args.drop_temperature,
        substeps=args.substeps,
        shortwave_flux=args.shortwave_flux,
        shortwave_absorption_efficiency=args.shortwave_absorption,
        sky_temperature=args.sky_temperature,
    )
    mesh = (
        make_distributed_mesh(args.devices)
        if args.num_processes > 1
        else make_single_node_mesh(args.devices)
    )
    operators = make_sharded_operators(params, mesh)
    state = ShardedSprayCoupledState(
        flow=initial_sharded_state(params, mesh),
        spray=initialize_sharded_spray(spray_config, params, mesh),
    )
    step = jax.jit(
        make_step_spray_dpm_sharded(
            spray_config, params, operators, mesh
        )
    )
    carrier_diagnostics = make_diagnostics_sharded(
        params, operators.horizontal, mesh
    )
    for index in range(args.steps):
        state, spray_diagnostics = step(
            state, operators.pressure, operators.pressure_spike
        )
        if (index + 1) % args.log_every != 0:
            continue
        flow_diagnostics = carrier_diagnostics(
            state.flow.u,
            state.flow.v,
            state.flow.w,
            state.flow.theta,
            state.flow.qv,
            state.flow.step,
        )
        state, spray_diagnostics, flow_diagnostics = jax.block_until_ready(
            (state, spray_diagnostics, flow_diagnostics)
        )
        if jax.process_index() == 0:
            print(
                f"step={int(state.flow.step):6d} "
                f"active={int(spray_diagnostics.active_parcels):7d} "
                f"liquid={float(spray_diagnostics.liquid_mass):.6e} kg "
                f"evaporated={float(spray_diagnostics.evaporated_mass):.6e} kg/step "
                f"qv_min={float(flow_diagnostics.qv_min):.6e} "
                f"div={float(flow_diagnostics.div_max):.3e}",
                flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
