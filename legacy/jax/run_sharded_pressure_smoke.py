from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

JAX_ROOT = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(JAX_ROOT))

from wireles_jax.config import Params
from wireles_jax.pressure_sharded import (
    make_pressure_hat_solver_z_sharded,
    make_pressure_operators_reference,
    make_sharded_pressure_operators,
    pressure_and_gradients_from_hat_z_sharded,
    put_pressure_hat_z,
    put_pressure_rhs_inner_z,
    solve_pressure_hat_fortran_reference,
)
from wireles_jax.sharding import (
    irfft2_fortran_layout,
    make_single_node_mesh,
    pressure_layout_roundtrip,
    rfft2_fortran_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-node JAX sharding pressure smoke test.")
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--devices", type=int, default=None, help="Number of local JAX devices to use.")
    parser.add_argument("--single", action="store_true", help="Use float32 instead of float64.")
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = jnp.float32 if args.single else jnp.float64

    mesh = make_single_node_mesh(args.devices)
    ndev = int(mesh.shape["z"])
    if args.ny % ndev != 0 or args.nz % ndev != 0:
        raise ValueError(f"ny={args.ny} and nz={args.nz} must both be divisible by devices={ndev}")

    params = Params(nx=args.nx, ny=args.ny, nz=args.nz, dtype=dtype)
    tol = 5.0e-5 if dtype == jnp.float32 else 5.0e-11
    rtol = tol if args.rtol is None else args.rtol
    atol = tol if args.atol is None else args.atol

    key = jax.random.PRNGKey(args.seed)
    rhs_inner = jax.random.normal(key, (params.nx, params.ny, params.nz), dtype=dtype)
    rhs_inner = rhs_inner - jnp.mean(rhs_inner)

    print(f"[sharding] devices={ndev} mesh={mesh.devices.shape} dtype={jnp.dtype(dtype)}", flush=True)
    print(f"[sharding] physical interior shape=({params.nx}, {params.ny}, {params.nz})", flush=True)

    fft_start = time.perf_counter()
    rhs_hat_ref = rfft2_fortran_layout(rhs_inner)
    fft_roundtrip = irfft2_fortran_layout(rhs_hat_ref, params.nx, params.ny)
    fft_err = float(jnp.max(jnp.abs(fft_roundtrip - rhs_inner)))
    print(f"[check] fortran-layout fft roundtrip max_abs={fft_err:.4e}", flush=True)

    rhs_hat_z = put_pressure_hat_z(rhs_hat_ref, mesh)
    layout_roundtrip = pressure_layout_roundtrip(rhs_hat_z, mesh)
    layout_err = float(jnp.max(jnp.abs(layout_roundtrip - rhs_hat_ref)))
    layout_roundtrip.block_until_ready()
    print(f"[check] z/y slab all-to-all roundtrip max_abs={layout_err:.4e}", flush=True)

    ops_ref = make_pressure_operators_reference(params)
    ops_sharded = make_sharded_pressure_operators(params, mesh)
    pressure_solver = make_pressure_hat_solver_z_sharded(params, ops_sharded, mesh)

    p_hat_ref = solve_pressure_hat_fortran_reference(rhs_hat_ref / params.dt, params, ops_ref)
    p_hat_ref.block_until_ready()

    compile_start = time.perf_counter()
    p_hat_z = pressure_solver(rhs_hat_z / params.dt)
    p_hat_z.block_until_ready()
    compile_solve_s = time.perf_counter() - compile_start

    solve_start = time.perf_counter()
    p_hat_z = pressure_solver(rhs_hat_z / params.dt)
    p_hat_z.block_until_ready()
    solve_s = time.perf_counter() - solve_start

    p_hat_err = float(jnp.max(jnp.abs(p_hat_z - p_hat_ref)))
    p_inner, dpdx_inner, dpdy_inner = pressure_and_gradients_from_hat_z_sharded(p_hat_z, params, ops_sharded)
    p_inner.block_until_ready()
    grad_norm = float(jnp.sqrt(jnp.mean(dpdx_inner * dpdx_inner + dpdy_inner * dpdy_inner)))

    rhs_inner_z = put_pressure_rhs_inner_z(rhs_inner, mesh)
    rhs_hat_from_z = rfft2_fortran_layout(rhs_inner_z)
    rhs_hat_from_z.block_until_ready()
    sharded_fft_err = float(jnp.max(jnp.abs(rhs_hat_from_z - rhs_hat_ref)))

    total_s = time.perf_counter() - fft_start
    print(f"[check] pressure_hat max_abs={p_hat_err:.4e} rtol={rtol:.1e} atol={atol:.1e}", flush=True)
    print(f"[check] z-sharded horizontal fft max_abs={sharded_fft_err:.4e}", flush=True)
    print(f"[timing] first sharded solve including compile={compile_solve_s:.3f}s", flush=True)
    print(f"[timing] cached sharded solve={solve_s:.6f}s total_script={total_s:.3f}s", flush=True)
    print(f"[diagnostic] p_rms={float(jnp.sqrt(jnp.mean(p_inner * p_inner))):.4e} grad_xy_rms={grad_norm:.4e}", flush=True)

    if not np.allclose(np.asarray(p_hat_z), np.asarray(p_hat_ref), rtol=rtol, atol=atol):
        raise SystemExit("sharded pressure solve does not match reference within tolerance")


if __name__ == "__main__":
    main()
