#!/usr/bin/env python3
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the P=4 SEM ABL prototype.")
    parser.add_argument("--nelx", type=int, default=2)
    parser.add_argument("--nely", type=int, default=2)
    parser.add_argument("--nelz", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--u-fric", type=float, default=0.4)
    parser.add_argument("--zo", type=float, default=5.0e-3)
    parser.add_argument("--smag-cs", type=float, default=0.16)
    parser.add_argument("--pressure-force", type=float, default=None)
    parser.add_argument("--pressure-cycles", type=int, default=10)
    parser.add_argument("--mg-pre-smooth", type=int, default=1)
    parser.add_argument("--mg-post-smooth", type=int, default=1)
    parser.add_argument("--mg-omega", type=float, default=0.3)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--no-jit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.single:
        from jax import config as jax_config

        jax_config.update("jax_enable_x64", True)

    import jax.numpy as jnp

    from sem_jax import SemParams, run

    params = SemParams(
        nelx=args.nelx,
        nely=args.nely,
        nelz=args.nelz,
        nsteps=args.steps,
        dt=args.dt,
        log_every=args.log_every,
        u_fric=args.u_fric,
        zo=args.zo,
        smagorinsky_cs=args.smag_cs,
        pressure_force=args.pressure_force,
        pressure_max_cycles=args.pressure_cycles,
        mg_pre_smooth=args.mg_pre_smooth,
        mg_post_smooth=args.mg_post_smooth,
        mg_omega=args.mg_omega,
        dtype=jnp.float32 if args.single else jnp.float64,
        use_jit=not args.no_jit,
    )
    _, logs = run(params)
    print(" step      ustar        ke_max       div_max        cfl   press_res")
    for diag in logs:
        print(
            f"{int(diag.step):5d} "
            f"{float(diag.ustar):11.4e} "
            f"{float(diag.ke_max):11.4e} "
            f"{float(diag.div_max):11.4e} "
            f"{float(diag.cfl):9.4f} "
            f"{float(diag.pressure_residual):11.4e}"
        )


if __name__ == "__main__":
    main()
