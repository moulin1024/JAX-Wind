#!/usr/bin/env python3
"""Run a minimal continuously injected DPM spray/ABL coupling case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--lx", type=float, default=1000.0, help="Physical x length [m].")
    parser.add_argument("--ly", type=float, default=1000.0, help="Physical y length [m].")
    parser.add_argument("--lz", type=float, default=500.0, help="Physical z length [m].")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.1, help="Physical time step [s].")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--wind-speed", type=float, default=8.0)
    parser.add_argument("--theta0", type=float, default=300.0)
    parser.add_argument("--qv0", type=float, default=0.005)
    parser.add_argument("--max-parcels", type=int, default=4096)
    parser.add_argument("--parcels-per-step", type=int, default=16)
    parser.add_argument("--mass-flow-rate", type=float, default=0.1, help="Injected water [kg/s].")
    parser.add_argument("--injection-x", type=float, default=250.0)
    parser.add_argument("--injection-y", type=float, default=500.0)
    parser.add_argument("--injection-z", type=float, default=100.0)
    parser.add_argument("--injection-radius", type=float, default=10.0)
    parser.add_argument("--injection-u", type=float, default=8.0)
    parser.add_argument("--injection-v", type=float, default=0.0)
    parser.add_argument("--injection-w", type=float, default=0.0)
    parser.add_argument("--diameter", type=float, default=100.0e-6, help="Drop diameter [m].")
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
    parser.add_argument("--drop-temperature", type=float, default=283.15, help="Drop temperature [K].")
    parser.add_argument("--shortwave-flux", type=float, default=0.0, help="Incident shortwave [W/m2].")
    parser.add_argument("--shortwave-absorption", type=float, default=0.0)
    parser.add_argument("--sky-temperature", type=float, default=273.15)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import jax
    import numpy as np

    from wireles_jax import Params, SprayDPMConfig, run_spray_dpm
    from wireles_jax.io import save_npz

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
        time_scheme="rk3",
        thermo_enabled=True,
        moisture_enabled=True,
        theta0=args.theta0,
        qv0=args.qv0,
        use_jit=not args.no_jit,
    )
    spray_config = SprayDPMConfig(
        max_parcels=args.max_parcels,
        parcels_per_step=args.parcels_per_step,
        mass_flow_rate=args.mass_flow_rate,
        injection_x=args.injection_x,
        injection_y=args.injection_y,
        injection_z=args.injection_z,
        injection_radius=args.injection_radius,
        injection_u=args.injection_u,
        injection_v=args.injection_v,
        injection_w=args.injection_w,
        initial_diameter=args.diameter,
        diameter_distribution=args.diameter_distribution,
        minimum_diameter=args.minimum_diameter,
        maximum_diameter=args.maximum_diameter,
        rosin_rammler_spread=args.rosin_rammler_spread,
        lognormal_geometric_stddev=args.lognormal_geometric_stddev,
        turbulent_dispersion_enabled=args.turbulent_dispersion,
        initial_temperature=args.drop_temperature,
        shortwave_flux=args.shortwave_flux,
        shortwave_absorption_efficiency=args.shortwave_absorption,
        sky_temperature=args.sky_temperature,
        substeps=args.substeps,
    )

    def report(flow, diag) -> None:
        print(
            f"step={int(flow.step):6d} active={int(diag.active_parcels):6d} "
            f"liquid={float(diag.liquid_mass):.6e} kg "
            f"evaporated={float(diag.evaporated_mass):.6e} kg/step "
            f"air_heat={float(diag.air_energy_loss):.6e} J/step "
            f"net_radiation={float(diag.net_radiative_energy):.6e} J/step",
            flush=True,
        )

    state, _ = run_spray_dpm(
        params,
        spray_config,
        log_every=args.log_every,
        log_callback=report,
    )
    state = jax.block_until_ready(state)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        save_npz(args.output / "flow_final.npz", state.flow)
        np.savez(
            args.output / "spray_final.npz",
            **{name: np.asarray(value) for name, value in state.spray._asdict().items()},
        )


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
