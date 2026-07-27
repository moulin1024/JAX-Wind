#!/usr/bin/env python3
"""Distributed 3D pressure-Poisson benchmark (channel / ABL configuration).

Horizontal directions are periodic and spectrally discretized (rfft2);
the vertical direction is wall-bounded and discretized with second-order
finite differences, giving one tridiagonal system per horizontal mode.
The discretization mirrors WIRELES_GPU (wireles_jax/pressure_sharded.py):
Fortran-compatible (nx/2+1, ny, nz) spectral layout, Neumann bottom row,
rigid-lid top row, k^2 = 0 mode pinned at the bottom, optional Nyquist
mode filtering, and a precomputed Thomas (LU) factorization so the timed
solve is forward/backward substitution only.

Pipeline (z-slab base layout, nz/P levels per GPU):

  rhs (nx, ny, nz/P)   physical z slab
    -> rfft2 (local)              (nx/2+1, ny, nz/P)
    -> all-to-all                 (nx/2+1, ny/P, nz)   y slab
    -> Thomas solve per mode      (nx/2+1, ny/P, nz)
    -> all-to-all                 (nx/2+1, ny, nz/P)   z slab
    -> irfft2 (local)             (nx, ny, nz/P)

The transpose method communicates two full spectral fields per solve. The
SPIKE method exchanges only interface scalars, using either two compact
all-to-alls or one all-gather followed by a replicated interface solve.
Validation: relative residual of the discrete tridiagonal equations
(A p_col - rhs_col) per mode, evaluated on unfiltered modes.

Single host, multiple visible GPUs:
  python poisson3d_distributed.py --nx 1024 --ny 1024 --nz 128

Slurm, one process per GPU:
  srun --ntasks=8 --cpu-bind=cores python poisson3d_distributed.py --distributed ...
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import sys
import time
from functools import partial

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a slab-decomposed spectral/FD 3D pressure-Poisson solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nx", type=int, default=1024, help="periodic points in x")
    parser.add_argument("--ny", type=int, default=1024, help="periodic points in y")
    parser.add_argument("--nz", type=int, default=128, help="vertical levels (wall-bounded)")
    parser.add_argument("--lx", type=float, default=1.0, help="domain length in x")
    parser.add_argument("--ly", type=float, default=1.0, help="domain length in y")
    parser.add_argument("--lz", type=float, default=1.0, help="domain height")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--warmup", type=int, default=3, help="warm-up executions per operation")
    parser.add_argument("--samples", type=int, default=10, help="timing samples per operation")
    parser.add_argument("--iterations", type=int, default=5, help="synchronized calls per sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-nyquist-filter",
        action="store_true",
        help="keep Nyquist modes instead of zeroing them (WIRELES default filters)",
    )
    parser.add_argument(
        "--tridiag",
        choices=("pcr", "thomas"),
        default="pcr",
        help=(
            "vertical solver: pcr runs ceil(log2(nz+1)) vectorized cyclic-"
            "reduction steps (few large kernels; stores 2*steps+1 factor "
            "arrays instead of thomas's 3); thomas runs 2(nz+1) sequential "
            "scan steps (tiny kernels, launch-latency bound on GPUs)"
        ),
    )
    parser.add_argument(
        "--method",
        choices=("transpose", "spike"),
        default="transpose",
        help=(
            "transpose moves ky slabs through two full-field all-to-alls "
            "around the vertical solve; spike keeps the z-slab layout, solves "
            "each GPU's row block locally (PCR, precomputed spike vectors) "
            "and couples blocks through a prefactored (2P+1)-row interface "
            "system, exchanging only 2 scalars per mode each way"
        ),
    )
    parser.add_argument(
        "--spike-interface-collective",
        choices=("alltoall", "allgather"),
        default="alltoall",
        help=(
            "SPIKE interface exchange: alltoall shards horizontal modes and "
            "uses two compact exchanges per solve; allgather replicates the "
            "small interface solve on every GPU and uses one exchange, avoiding "
            "the NCCL all-to-all path"
        ),
    )
    parser.add_argument(
        "--mms",
        action="store_true",
        help=(
            "replace the random RHS with a manufactured solution built from "
            "discrete eigenmodes; the solver must reproduce it to roundoff"
        ),
    )
    parser.add_argument(
        "--platform",
        choices=("cuda", "rocm"),
        help="require a specific JAX backend; otherwise use JAX_PLATFORMS/default",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="call jax.distributed.initialize(); use with one process per GPU",
    )
    parser.add_argument(
        "--skip-components",
        action="store_true",
        help="measure only the complete solve",
    )
    return parser


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def human_bytes(n: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def main() -> int:
    args = build_parser().parse_args()
    if min(args.nx, args.ny, args.nz) <= 0:
        raise SystemExit("--nx/--ny/--nz must be positive")
    if args.nx % 2 or args.ny % 2:
        raise SystemExit("--nx and --ny must be even")
    if args.warmup < 0 or args.samples <= 0 or args.iterations <= 0:
        raise SystemExit("warmup must be nonnegative; samples and iterations must be positive")

    # Manufactured solution: sum of separable terms
    #   A * trig_x(2 pi mx x / lx) * trig_y(2 pi my y / ly) * cos(mz pi zeta)
    # with zeta = (j - 1/2) / (nz - 1) at pressure node j. The vertical
    # profiles are exact discrete eigenfunctions of the FD stencil (symmetric
    # about both walls, so the one-sided Neumann rows hold exactly), hence the
    # discrete solver must return the manufactured field to roundoff.
    mms_terms = (
        (1.00, 2, 3, 1, "cos", "cos"),
        (0.70, 1, 0, 2, "sin", "cos"),
        (0.50, 0, 4, 5, "cos", "sin"),
        (0.40, 1, 1, 0, "cos", "cos"),
        (0.30, 3, 2, 3, "sin", "sin"),
    )
    if args.mms:
        for _, mx, my, mz, _, _ in mms_terms:
            if mx >= args.nx // 2 or my >= args.ny // 2:
                raise SystemExit("--mms horizontal modes must stay below Nyquist")
            if mz > args.nz - 2:
                raise SystemExit("--mms requires nz >= 8")

    # These must be set before JAX initializes a backend.
    if args.platform:
        os.environ["JAX_PLATFORMS"] = args.platform
    if args.dtype == "float64":
        os.environ["JAX_ENABLE_X64"] = "true"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    from jax import lax

    if args.distributed:
        # One Slurm task per GPU, same conventions as fft2d_distributed.py:
        # per-task GRES binding shows one local device; job-level GRES shows
        # all node devices and the task claims its node-local rank.
        visible = next(
            (
                os.environ[k]
                for k in (
                    "ROCR_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                )
                if os.environ.get(k)
            ),
            None,
        )
        if visible is not None and len(visible.split(",")) == 1:
            local_device = 0
        else:
            local_device = int(os.environ.get("SLURM_LOCALID", "0"))
        jax.distributed.initialize(local_device_ids=[local_device])

    global_devices = jax.device_count()
    local_devices = jax.local_device_count()
    process_count = jax.process_count()
    process_index = jax.process_index()

    if jax.default_backend() != "gpu":
        raise SystemExit(f"GPU backend required; got {jax.default_backend()!r}")

    nx, ny, nz = args.nx, args.ny, args.nz
    p = global_devices
    if ny % p:
        raise SystemExit(f"ny={ny} must be divisible by GPU count {p}")
    if nz % p:
        raise SystemExit(f"nz={nz} must be divisible by GPU count {p}")

    if args.method == "spike" and args.tridiag == "thomas":
        raise SystemExit("--method spike uses its own local PCR; drop --tridiag thomas")
    if args.method == "spike" and nz // p < 2:
        raise SystemExit(f"--method spike needs nz/GPUs >= 2; got {nz // p}")

    nxh = nx // 2 + 1
    ny_local = ny // p
    nz_local = nz // p
    real_dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    real_itemsize = 4 if args.dtype == "float32" else 8
    dz = args.lz / nz
    dz2 = dz * dz

    physical_bytes = nx * ny * nz * real_itemsize
    spectral_bytes = nxh * ny * nz * 2 * real_itemsize
    axis_name = "poisson_devices"
    mapped = partial(jax.pmap, axis_name=axis_name)

    # Horizontal wavenumbers, Fortran spectral layout (kx halved axis), with
    # Nyquist wavenumbers zeroed exactly as in WIRELES.
    np_real = np.float32 if args.dtype == "float32" else np.float64
    kx_np = (2.0 * np.pi * np.fft.rfftfreq(nx, d=args.lx / nx)).astype(np_real)
    ky_np = (2.0 * np.pi * np.fft.fftfreq(ny, d=args.ly / ny)).astype(np_real)
    kx_np[-1] = 0.0
    ky_np[ny // 2] = 0.0
    keep_np = np.ones((nxh, ny), dtype=bool)
    if not args.no_nyquist_filter:
        keep_np[-1, :] = False
        keep_np[:, ny // 2] = False
    kx_all = jnp.asarray(kx_np)
    ky_all = jnp.asarray(ky_np)
    keep_all = jnp.asarray(keep_np.astype(np_real))
    eps128 = float(np.finfo(np_real).eps) * 128.0

    def _move_z_first(q):
        return jnp.moveaxis(q, -1, 0)

    def _move_z_last(q):
        return jnp.moveaxis(q, 0, -1)

    @mapped
    def build_operators(_token):
        """Per-device tridiagonal rows for the local ky slab.

        Rows follow wireles_jax._pressure_tridiag_fortran_layout: Neumann
        bottom (p1 - p0 = 0), rigid-lid top (p_nz - p_{nz-1} = 0), interior
        (p_{j-1} - 2 p_j + p_{j+1}) / dz^2 - k^2 p_j, and the k^2 = 0 mode
        pinned via p0 = 0.
        """
        j = lax.axis_index(axis_name)
        ky_l = lax.dynamic_slice_in_dim(ky_all, j * ny_local, ny_local)
        keep_l = lax.dynamic_slice_in_dim(keep_all, j * ny_local, ny_local, axis=1)
        k2 = kx_all[:, None] * kx_all[:, None] + ky_l[None, :] * ky_l[None, :]
        zero_k2 = jnp.abs(k2) < eps128

        shape = (nxh, ny_local, nz + 1)
        one = jnp.asarray(1.0, real_dtype)
        a = jnp.zeros(shape, real_dtype)
        a = a.at[..., 1:nz].set(1.0 / dz2)
        a = a.at[..., nz].set(-1.0)
        b = jnp.zeros(shape, real_dtype)
        b = b.at[..., 0].set(jnp.where(zero_k2, one, -one))
        b = b.at[..., 1:nz].set(-k2[..., None] - 2.0 / dz2)
        b = b.at[..., nz].set(1.0)
        c = jnp.zeros(shape, real_dtype)
        c = c.at[..., 0].set(jnp.where(zero_k2, 0.0 * one, one))
        c = c.at[..., 1:nz].set(1.0 / dz2)
        return a, b, c, keep_l[..., None]

    @mapped
    def build_thomas_factors(a, b, c):
        """Precompute the Thomas (LU) sweep factors, as in wireles_jax."""

        def factor_step(carry, rows):
            inv_bet_prev, c_prev = carry
            a_j, b_j, c_j = rows
            gam_j = c_prev * inv_bet_prev
            inv_bet_j = 1.0 / (b_j - a_j * gam_j)
            return (inv_bet_j, c_j), (inv_bet_j, gam_j)

        inv_bet0 = 1.0 / b[..., 0]
        _, (inv_bet_tail, gam_tail) = lax.scan(
            factor_step,
            (inv_bet0, c[..., 0]),
            (
                _move_z_first(a[..., 1:]),
                _move_z_first(b[..., 1:]),
                _move_z_first(c[..., 1:]),
            ),
        )
        inv_bet = jnp.concatenate(
            (inv_bet0[..., None], _move_z_last(inv_bet_tail)), axis=-1
        )
        gam = jnp.concatenate(
            (jnp.zeros_like(inv_bet0)[..., None], _move_z_last(gam_tail)), axis=-1
        )
        return inv_bet, gam

    # Parallel cyclic reduction. Each step with stride s eliminates the
    # couplings a_i, c_i against rows i -+ s; after ceil(log2(nz+1)) steps the
    # systems are diagonal. The row reduction is RHS-independent, so the
    # per-step elimination factors alpha/gamma are precomputed and the timed
    # solve is one fused update per step:
    #   d <- d + alpha_k * d[i-s] + gamma_k * d[i+s]
    # Out-of-range neighbours are identity rows: b filled with 1, the rest
    # with 0 (the PCR invariant a_i = 0 for i < s makes those terms vanish).
    pcr_steps = nz.bit_length()  # ceil(log2(nz + 1))

    def _shift_dn(v, s, fill=0.0):
        pad = jnp.full(v.shape[:-1] + (s,), fill, dtype=v.dtype)
        return jnp.concatenate((pad, v[..., :-s]), axis=-1)

    def _shift_up(v, s, fill=0.0):
        pad = jnp.full(v.shape[:-1] + (s,), fill, dtype=v.dtype)
        return jnp.concatenate((v[..., s:], pad), axis=-1)

    def pcr_factor_arrays(a, b, c, steps):
        alphas, gammas = [], []
        for k in range(steps):
            s = 1 << k
            alpha = -a / _shift_dn(b, s, fill=1.0)
            gamma = -c / _shift_up(b, s, fill=1.0)
            b = b + alpha * _shift_dn(c, s) + gamma * _shift_up(a, s)
            a = alpha * _shift_dn(a, s)
            c = gamma * _shift_up(c, s)
            alphas.append(alpha)
            gammas.append(gamma)
        return jnp.stack(alphas), jnp.stack(gammas), 1.0 / b

    def pcr_apply(alphas, gammas, inv_b, rhs, steps):
        d = rhs
        for k in range(steps):
            s = 1 << k
            d = d + alphas[k] * _shift_dn(d, s) + gammas[k] * _shift_up(d, s)
        return d * inv_b

    @mapped
    def build_pcr_factors(a, b, c):
        return pcr_factor_arrays(a, b, c, pcr_steps)

    def pcr_solve(alphas, gammas, inv_b, rhs):
        return pcr_apply(alphas, gammas, inv_b, rhs, pcr_steps)

    def tridiag_solve_op(o1, o2, o3, rhs_col):
        """Dispatch on the selected solver; (o1, o2, o3) are its factors."""
        if args.tridiag == "thomas":
            return solve_tridiag(o1, o2, o3, rhs_col)  # (a, inv_bet, gam)
        return pcr_solve(o1, o2, o3, rhs_col)  # (alphas, gammas, inv_b)

    # ------------------------------------------------------------------
    # SPIKE (substructuring) vertical solve in the z-slab layout.
    #
    # Global system rows 0..nz per mode. Row 0 (bottom Neumann / k^2 = 0 pin)
    # joins the interface system; GPU k owns rows [k m + 1, (k+1) m] with
    # m = nz/P, which aligns exactly with its physical rhs slab (row j reads
    # rhs level j - 1) and with the output p levels. Per solve:
    #   y   = A_k^-1 d           local PCR, precomputed factors
    #   exchange 2 interface scalars per mode (y[0], y[m-1]) <- all the comm
    #   u   = M^-1 rhs_u         prefactored (2P+1)-row interface system,
    #                            unknowns (x0, alpha_0, beta_0, ..., beta_{P-1})
    #   x   = y - w L - v R      precomputed spike vectors w = A_k^-1 a_1 e_0,
    #                            v = A_k^-1 c_m e_{m-1}; L = u[2k], R = u[2k+3]
    # ------------------------------------------------------------------
    m_blk = nz_local
    spike_steps = max(1, (m_blk - 1).bit_length())
    reduced_n = 2 * p + 1
    nyq = ny // p

    def scalars_to_modes(stacked):
        """(S, nxh, ny) per-block scalars -> (P, S, nxh, nyq) on mode owners."""
        s_count = stacked.shape[0]
        t = stacked.reshape(s_count, nxh, p, nyq)
        t = jnp.moveaxis(t, 2, 0)
        if p > 1:
            t = lax.all_to_all(t, axis_name, split_axis=0, concat_axis=0, tiled=True)
        return t

    def modes_to_scalars(t):
        """(P, S, nxh, nyq) per-destination values -> (S, nxh, ny) on blocks."""
        if p > 1:
            t = lax.all_to_all(t, axis_name, split_axis=0, concat_axis=0, tiled=True)
        t = jnp.moveaxis(t, 0, 2)  # (S, nxh, P, nyq), axis 2 = ky chunk
        return t.reshape(t.shape[0], nxh, ny)

    def gather_block_scalars(stacked):
        """(S, nxh, ny) per block -> (P, S, nxh, ny) replicated on every GPU."""
        if p > 1:
            return lax.all_gather(stacked, axis_name, axis=0, tiled=False)
        return stacked[None, ...]

    def block_rows_abc(dev):
        """Tridiagonal rows owned by this block, full couplings included."""
        rows = dev * m_blk + 1 + jnp.arange(m_blk)
        interior = rows <= nz - 1  # row nz is the rigid-lid top row
        a_blk = jnp.where(interior, 1.0 / dz2, -1.0).astype(real_dtype)
        c_blk = jnp.where(interior, 1.0 / dz2, 0.0).astype(real_dtype)
        k2 = kx_all[:, None] * kx_all[:, None] + ky_all[None, :] * ky_all[None, :]
        b_blk = jnp.where(
            interior[None, None, :],
            (-k2[..., None] - 2.0 / dz2).astype(real_dtype),
            jnp.asarray(1.0, real_dtype),
        )
        return a_blk, b_blk, c_blk

    @mapped
    def build_spike_factors(_token):
        dev = lax.axis_index(axis_name)
        a_blk, b_blk, c_blk = block_rows_abc(dev)
        # The couplings that leave the block move to the interface system, so
        # the local matrix zeroes them; their values feed the spike RHS.
        a_first = a_blk[0]
        c_last = c_blk[m_blk - 1]
        a_in = jnp.broadcast_to(a_blk.at[0].set(0.0), b_blk.shape)
        c_in = jnp.broadcast_to(c_blk.at[m_blk - 1].set(0.0), b_blk.shape)
        alphas, gammas, inv_b = pcr_factor_arrays(a_in, b_blk, c_in, spike_steps)

        e_first = jnp.zeros((m_blk,), real_dtype).at[0].set(1.0)
        e_last = jnp.zeros((m_blk,), real_dtype).at[m_blk - 1].set(1.0)
        w = pcr_apply(
            alphas, gammas, inv_b, jnp.broadcast_to(a_first * e_first, b_blk.shape), spike_steps
        )
        v = pcr_apply(
            alphas, gammas, inv_b, jnp.broadcast_to(c_last * e_last, b_blk.shape), spike_steps
        )

        # Assemble and invert the static interface matrix. The all-to-all path
        # shards ky modes across GPUs; the all-gather path replicates all modes
        # so the timed solve needs only one mature NCCL all-gather collective.
        spike_endpoints = jnp.stack((w[..., 0], w[..., -1], v[..., 0], v[..., -1]))
        if args.spike_interface_collective == "allgather":
            iface = gather_block_scalars(spike_endpoints)  # (P, 4, nxh, ny)
            ky_q = ky_all
            interface_ny = ny
        else:
            iface = scalars_to_modes(spike_endpoints)  # (P, 4, nxh, nyq)
            ky_q = lax.dynamic_slice_in_dim(ky_all, dev * nyq, nyq)
            interface_ny = nyq
        k2_q = kx_all[:, None] * kx_all[:, None] + ky_q[None, :] * ky_q[None, :]
        zero_k2 = jnp.abs(k2_q) < eps128
        one = jnp.asarray(1.0, real_dtype)
        matrix = jnp.zeros((nxh, interface_ny, reduced_n, reduced_n), real_dtype)
        matrix = matrix.at[..., 0, 0].set(jnp.where(zero_k2, one, -one))
        matrix = matrix.at[..., 0, 1].set(jnp.where(zero_k2, 0.0 * one, one))
        for k in range(p):
            row_a, row_b = 1 + 2 * k, 2 + 2 * k
            matrix = matrix.at[..., row_a, row_a].set(1.0)
            matrix = matrix.at[..., row_b, row_b].set(1.0)
            matrix = matrix.at[..., row_a, 2 * k].set(iface[k, 0])
            matrix = matrix.at[..., row_b, 2 * k].set(iface[k, 1])
            if k < p - 1:
                matrix = matrix.at[..., row_a, 2 * k + 3].set(iface[k, 2])
                matrix = matrix.at[..., row_b, 2 * k + 3].set(iface[k, 3])
        minv = jnp.linalg.inv(matrix)
        return alphas, gammas, inv_b, w, v, minv

    def spike_pressure_raw(rhs_hat_local, alphas, gammas, inv_b, w, v, minv):
        """Returns (x unfiltered, (L, R) neighbour values, masked rhs)."""
        dev = lax.axis_index(axis_name)
        # Row nz is a BC row: the last physical rhs level is unused.
        zmask = (jnp.arange(m_blk) < m_blk - 1) | (dev != p - 1)
        d = jnp.where(zmask, rhs_hat_local, 0)
        y = pcr_apply(alphas, gammas, inv_b, d, spike_steps)
        block_endpoints = jnp.stack((y[..., 0], y[..., -1]))
        if args.spike_interface_collective == "allgather":
            iface = gather_block_scalars(block_endpoints)  # (P, 2, nxh, ny)
            interface_ny = ny
        else:
            iface = scalars_to_modes(block_endpoints)  # (P, 2, nxh, nyq)
            interface_ny = nyq
        rhs_u = jnp.transpose(iface, (2, 3, 0, 1)).reshape(nxh, interface_ny, 2 * p)
        rhs_u = jnp.concatenate((jnp.zeros_like(rhs_u[..., :1]), rhs_u), axis=-1)
        u = jnp.einsum("...ij,...j->...i", minv, rhs_u)
        u = jnp.concatenate((u, jnp.zeros_like(u[..., :1])), axis=-1)  # R_{P-1} = 0
        if args.spike_interface_collective == "allgather":
            left = lax.dynamic_index_in_dim(u, 2 * dev, axis=-1, keepdims=False)
            right = lax.dynamic_index_in_dim(u, 2 * dev + 3, axis=-1, keepdims=False)
            left_right = jnp.stack((left, right))  # (2, nxh, ny)
        else:
            outs = jnp.stack(
                [jnp.stack((u[..., 2 * k], u[..., 2 * k + 3])) for k in range(p)]
            )  # (P, 2, nxh, nyq)
            left_right = modes_to_scalars(outs)  # (2, nxh, ny)
        x = y - w * left_right[0][..., None] - v * left_right[1][..., None]
        return x, left_right, d

    def spike_pressure(rhs_hat_local, *ops):
        x, _, _ = spike_pressure_raw(rhs_hat_local, *ops)
        return x * keep_all[..., None].astype(x.dtype)

    def solve_tridiag(a, inv_bet, gam, rhs):
        """Thomas forward/backward substitution, verbatim from wireles_jax."""
        u0 = rhs[..., 0] * inv_bet[..., 0]

        def forward(carry, values):
            u_prev = carry
            a_j, inv_bet_j, rhs_j = values
            u_j = (rhs_j - a_j * u_prev) * inv_bet_j
            return u_j, u_j

        _, u_tail_z = lax.scan(
            forward,
            u0,
            (
                _move_z_first(a[..., 1:]),
                _move_z_first(inv_bet[..., 1:]),
                _move_z_first(rhs[..., 1:]),
            ),
        )
        u_forward = jnp.concatenate((u0[..., None], _move_z_last(u_tail_z)), axis=-1)

        def backward(next_u, values):
            u_j_forward, gam_next = values
            u_j = u_j_forward - gam_next * next_u
            return u_j, u_j

        _, u_prefix_rev_z = lax.scan(
            backward,
            u_forward[..., -1],
            (
                _move_z_first(u_forward[..., :-1])[::-1],
                _move_z_first(gam[..., 1:])[::-1],
            ),
        )
        u_prefix = _move_z_last(u_prefix_rev_z[::-1])
        return jnp.concatenate((u_prefix, u_forward[..., -1:]), axis=-1)

    def z_to_y(h):
        """(nxh, ny, nz_local) z slabs -> (nxh, ny_local, nz) y slabs."""
        nxh_, ny_, nzl = h.shape
        h = h.reshape(nxh_, p, ny_ // p, nzl)
        if p > 1:
            h = lax.all_to_all(h, axis_name, split_axis=1, concat_axis=3, tiled=True)
        return h.reshape(nxh_, ny_ // p, nzl * p)

    def y_to_z(h):
        """(nxh, ny_local, nz) y slabs -> (nxh, ny, nz_local) z slabs."""
        nxh_, nyl, nz_ = h.shape
        h = h.reshape(nxh_, nyl, p, nz_ // p)
        if p > 1:
            h = lax.all_to_all(h, axis_name, split_axis=2, concat_axis=1, tiled=True)
        return h.reshape(nxh_, nyl * p, nz_ // p)

    def make_rhs_col(rhs_hat_y):
        rhs_col = jnp.zeros(rhs_hat_y.shape[:-1] + (nz + 1,), dtype=rhs_hat_y.dtype)
        return rhs_col.at[..., 1:nz].set(rhs_hat_y[..., : nz - 1])

    def tridiag_pressure(rhs_hat_y, o1, o2, o3, keep):
        p_col = tridiag_solve_op(o1, o2, o3, make_rhs_col(rhs_hat_y))
        return p_col[..., 1:] * keep.astype(p_col.dtype)

    @mapped
    def forward_fft(rhs):
        return jnp.fft.rfftn(rhs, axes=(-2, -3))

    @mapped
    def transpose_z_to_y(h):
        return z_to_y(h)

    @mapped
    def tridiag_stage(h, o1, o2, o3, keep):
        return tridiag_pressure(h, o1, o2, o3, keep)

    @mapped
    def transpose_y_to_z(h):
        return y_to_z(h)

    @mapped
    def inverse_fft(h):
        return jnp.fft.irfftn(h, s=(ny, nx), axes=(-2, -3))

    @mapped
    def spike_stage(h, *ops):
        return spike_pressure(h, *ops)

    @mapped
    def solve_full(rhs, *ops):
        h = jnp.fft.rfftn(rhs, axes=(-2, -3))
        if args.method == "spike":
            h = spike_pressure(h, *ops)
        else:
            h = z_to_y(h)
            h = tridiag_pressure(h, ops[0], ops[1], ops[2], ops[3])
            h = y_to_z(h)
        return jnp.fft.irfftn(h, s=(ny, nx), axes=(-2, -3))

    @mapped
    def solve_residual_spike(rhs, *ops):
        """Residual of the block rows, neighbour values taken from (L, R)."""
        h = jnp.fft.rfftn(rhs, axes=(-2, -3))
        x, left_right, d = spike_pressure_raw(h, *ops)
        dev = lax.axis_index(axis_name)
        a_blk, b_blk, c_blk = block_rows_abc(dev)
        cd = x.dtype
        x_dn = jnp.concatenate((left_right[0][..., None], x[..., :-1]), axis=-1)
        x_up = jnp.concatenate((x[..., 1:], left_right[1][..., None]), axis=-1)
        residual = a_blk * x_dn + b_blk.astype(cd) * x + c_blk * x_up - d
        mask = keep_all[..., None]
        num = lax.pmax(jnp.max(jnp.abs(residual) * mask), axis_name)
        den = lax.pmax(jnp.max(jnp.abs(d) * mask), axis_name)
        return num / jnp.maximum(den, jnp.finfo(real_dtype).tiny)

    @mapped
    def solve_residual(rhs, a, b, c, o1, o2, o3, keep):
        """Relative residual of A p_col = rhs_col over unfiltered modes."""
        h = jnp.fft.rfftn(rhs, axes=(-2, -3))
        rhs_col = make_rhs_col(z_to_y(h))
        p_col = tridiag_solve_op(o1, o2, o3, rhs_col)
        cd = p_col.dtype
        p_dn = jnp.pad(p_col[..., :-1], ((0, 0), (0, 0), (1, 0)))
        p_up = jnp.pad(p_col[..., 1:], ((0, 0), (0, 0), (0, 1)))
        residual = (
            a.astype(cd) * p_dn + b.astype(cd) * p_col + c.astype(cd) * p_up - rhs_col
        )
        mask = keep.astype(real_dtype)
        num = lax.pmax(jnp.max(jnp.abs(residual) * mask), axis_name)
        den = lax.pmax(jnp.max(jnp.abs(rhs_col) * mask), axis_name)
        return num / jnp.maximum(den, jnp.finfo(real_dtype).tiny)

    @mapped
    def make_rhs(key):
        return jax.random.uniform(key, (nx, ny, nz_local), dtype=real_dtype) - 0.5

    @mapped
    def make_mms_fields(_token):
        """Manufactured (rhs, p) pair on this device's z slab.

        Interior row j of the tridiagonal system reads physical rhs level
        j - 1, and output level jj holds p_col[jj + 1]; both therefore sample
        the vertical profile at zeta = (jj + 1/2) / (nz - 1). The discrete
        eigenvalue of cos(mz pi zeta) under the interior stencil is
        lambda = (2 - 2 cos(mz pi / (nz - 1))) / dz^2.
        """
        d = lax.axis_index(axis_name)
        x = jnp.arange(nx, dtype=real_dtype) * (args.lx / nx)
        y = jnp.arange(ny, dtype=real_dtype) * (args.ly / ny)
        jj = d * nz_local + jnp.arange(nz_local, dtype=real_dtype)
        zeta = (jj + 0.5) / (nz - 1)
        p_ref = jnp.zeros((nx, ny, nz_local), real_dtype)
        rhs_ref = jnp.zeros_like(p_ref)
        for amp, mx, my, mz, trig_x, trig_y in mms_terms:
            kx_v = 2.0 * math.pi * mx / args.lx
            ky_v = 2.0 * math.pi * my / args.ly
            lam = (2.0 - 2.0 * math.cos(mz * math.pi / (nz - 1))) / dz2
            fx = jnp.cos(kx_v * x) if trig_x == "cos" else jnp.sin(kx_v * x)
            fy = jnp.cos(ky_v * y) if trig_y == "cos" else jnp.sin(ky_v * y)
            fz = jnp.cos(mz * jnp.pi * zeta)
            term = amp * fx[:, None, None] * fy[None, :, None] * fz[None, None, :]
            p_ref = p_ref + term
            rhs_ref = rhs_ref - (kx_v * kx_v + ky_v * ky_v + lam) * term
        return rhs_ref, p_ref

    @mapped
    def relative_max_difference(a_arr, b_arr):
        num = lax.pmax(jnp.max(jnp.abs(a_arr - b_arr)), axis_name)
        den = lax.pmax(jnp.max(jnp.abs(b_arr)), axis_name)
        return num / jnp.maximum(den, jnp.finfo(real_dtype).tiny)

    @mapped
    def global_max_token(x):
        return lax.pmax(x, axis_name)

    p_mms = None
    if args.mms:
        mms_token = np.zeros((local_devices,), dtype=np.float32)
        rhs, p_mms = make_mms_fields(mms_token)
    else:
        # Global device IDs are process-major in multi-process pmap programs.
        first_id = process_index * local_devices
        device_ids = np.arange(first_id, first_id + local_devices, dtype=np.uint32)
        base_key = jax.random.PRNGKey(args.seed)
        keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(jnp.asarray(device_ids))
        rhs = make_rhs(keys)
    rhs.block_until_ready()

    token = np.zeros((local_devices,), dtype=np.float32)
    global_max_token(token).block_until_ready()  # compile the timing collective

    def sync_all_processes() -> None:
        global_max_token(token).block_until_ready()

    def global_max_seconds(value: float) -> float:
        values = np.full((local_devices,), value, dtype=np.float32)
        result = global_max_token(values)
        result.block_until_ready()
        return float(np.asarray(result)[0])

    def benchmark(fn, *fn_args):
        sync_all_processes()
        t0 = time.perf_counter()
        out = fn(*fn_args)
        out.block_until_ready()
        compile_and_run = global_max_seconds(time.perf_counter() - t0)

        for _ in range(args.warmup):
            out = fn(*fn_args)
            out.block_until_ready()

        times = []
        for _ in range(args.samples):
            sync_all_processes()
            t0 = time.perf_counter()
            for _ in range(args.iterations):
                out = fn(*fn_args)
                out.block_until_ready()
            elapsed = (time.perf_counter() - t0) / args.iterations
            times.append(global_max_seconds(elapsed))
        return out, compile_and_run, times

    def report(label: str, compile_s: float, times: list[float]) -> float:
        median_s = statistics.median(times)
        if process_index == 0:
            print(f"\n{label}")
            print(f"  first call (compile+run): {compile_s * 1e3:.3f} ms")
            print(f"  median                 : {median_s * 1e3:.3f} ms")
            print(f"  best                   : {min(times) * 1e3:.3f} ms")
            print(
                f"  mean / p95             : {statistics.mean(times) * 1e3:.3f} / "
                f"{percentile(times, 0.95) * 1e3:.3f} ms"
            )
        return median_s

    if process_index == 0:
        print("Distributed 3D pressure-Poisson benchmark (spectral xy + FD z)")
        print(f"Python                  : {sys.version.split()[0]}")
        print(f"JAX                     : {jax.__version__}")
        print(f"Backend                 : {jax.default_backend()}")
        print(f"Processes               : {process_count}")
        print(f"Global / local GPUs     : {global_devices} / {local_devices}")
        print(f"Grid                    : {nx} x {ny} x {nz}")
        print(f"Spectral layout         : {nxh} x {ny} x {nz}  (Fortran, kx halved)")
        print(f"Physical z slab / GPU   : {nx} x {ny} x {nz_local}")
        print(f"Spectral y slab / GPU   : {nxh} x {ny_local} x {nz}")
        print(f"Dtype                   : {args.dtype}")
        print(f"Nyquist filter          : {not args.no_nyquist_filter}")
        if args.method == "spike":
            solver_desc = (
                f"spike (blocks of {m_blk}, local pcr {spike_steps} steps, "
                f"{reduced_n}-row interface, {args.spike_interface_collective})"
            )
        elif args.tridiag == "pcr":
            solver_desc = f"pcr ({pcr_steps} steps)"
        else:
            solver_desc = "thomas (scan)"
        print(f"Method                  : {args.method}")
        print(f"Tridiagonal solver      : {solver_desc}")
        print(f"RHS                     : {'manufactured solution (MMS)' if args.mms else 'random'}")
        print(f"Physical / spectral data: {human_bytes(physical_bytes)} / {human_bytes(spectral_bytes)}")
        print(f"Warmup / timing         : {args.warmup} / {args.samples} x {args.iterations}")

    op_token = np.zeros((local_devices,), dtype=np.float32)
    if args.method == "spike":
        pipeline_ops = tuple(build_spike_factors(op_token))
    else:
        a_op, b_op, c_op, keep_op = build_operators(op_token)
        if args.tridiag == "thomas":
            inv_bet_op, gam_op = build_thomas_factors(a_op, b_op, c_op)
            solver_ops = (a_op, inv_bet_op, gam_op)
        else:
            alphas_op, gammas_op, inv_b_op = build_pcr_factors(a_op, b_op, c_op)
            solver_ops = (alphas_op, gammas_op, inv_b_op)
        pipeline_ops = (*solver_ops, keep_op)
    pipeline_ops[0].block_until_ready()

    component_times = {}
    if not args.skip_components and args.method == "spike":
        rhs_hat, compile_s, times = benchmark(forward_fft, rhs)
        component_times["rfft2"] = report("Forward rfft2 (local)", compile_s, times)

        p_hat_z, compile_s, times = benchmark(spike_stage, rhs_hat, *pipeline_ops)
        component_times["spike"] = report(
            f"SPIKE vertical solve (local PCR {spike_steps} steps + "
            f"{reduced_n}-row interface)",
            compile_s,
            times,
        )
        del rhs_hat
        gc.collect()

        _, compile_s, times = benchmark(inverse_fft, p_hat_z)
        component_times["irfft2"] = report("Inverse irfft2 (local)", compile_s, times)
        del p_hat_z
        gc.collect()
    elif not args.skip_components:
        rhs_hat, compile_s, times = benchmark(forward_fft, rhs)
        component_times["rfft2"] = report("Forward rfft2 (local)", compile_s, times)

        hat_y, compile_s, times = benchmark(transpose_z_to_y, rhs_hat)
        component_times["z_to_y"] = report(
            "z-slab -> y-slab all-to-all" if p > 1 else "z-slab -> y-slab (local reshape)",
            compile_s,
            times,
        )
        del rhs_hat
        gc.collect()

        tridiag_label = (
            f"Tridiagonal solve (PCR, {pcr_steps} steps)"
            if args.tridiag == "pcr"
            else "Tridiagonal solve (Thomas scans)"
        )
        p_hat_y, compile_s, times = benchmark(tridiag_stage, hat_y, *solver_ops, keep_op)
        component_times["tridiag"] = report(tridiag_label, compile_s, times)
        del hat_y
        gc.collect()

        p_hat_z, compile_s, times = benchmark(transpose_y_to_z, p_hat_y)
        component_times["y_to_z"] = report(
            "y-slab -> z-slab all-to-all" if p > 1 else "y-slab -> z-slab (local reshape)",
            compile_s,
            times,
        )
        del p_hat_y
        gc.collect()

        _, compile_s, times = benchmark(inverse_fft, p_hat_z)
        component_times["irfft2"] = report("Inverse irfft2 (local)", compile_s, times)
        del p_hat_z
        gc.collect()

    p_out, compile_s, times = benchmark(solve_full, rhs, *pipeline_ops)
    full_s = report("Complete pressure-Poisson solve", compile_s, times)

    if args.method == "spike":
        error = solve_residual_spike(rhs, *pipeline_ops)
    else:
        error = solve_residual(rhs, a_op, b_op, c_op, *solver_ops, keep_op)
    error.block_until_ready()
    relative_residual = float(np.asarray(error)[0])

    mms_error = None
    if args.mms:
        diff = relative_max_difference(p_out, p_mms)
        diff.block_until_ready()
        mms_error = float(np.asarray(diff)[0])

    if process_index == 0:
        remote_payload = spectral_bytes * (p - 1) / p
        print("\nSummary")
        print(f"  solves/s                     : {1.0 / full_s:,.3f}")
        print(f"  grid throughput              : {nx * ny * nz / full_s / 1e9:,.3f} Gpoints/s")
        print(f"  tridiagonal relative residual: {relative_residual:.3e}")
        if mms_error is not None:
            print(f"  MMS relative max error       : {mms_error:.3e}")
        if component_times:
            component_sum = sum(component_times.values())
            print(f"  sum of measured stages       : {component_sum * 1e3:.3f} ms")
            print(f"  full minus isolated stages   : {(full_s - component_sum) * 1e3:+.3f} ms")
            if "z_to_y" in component_times:
                exchange_s = component_times["z_to_y"] + component_times["y_to_z"]
                print(f"  exchange fraction of stages  : {exchange_s / component_sum * 100:.2f}%")
                if p > 1:
                    bw = 2.0 * remote_payload / exchange_s / 2**30
                    print(f"  remote payload per exchange  : {human_bytes(remote_payload)}")
                    print(f"  aggregate one-way bandwidth  : {bw:,.3f} GiB/s")
                else:
                    print("  remote payload per exchange  : 0 B (single GPU)")
        if args.method == "spike":
            endpoint_bytes = 2 * nxh * ny * 2 * real_itemsize
            if p == 1:
                iface_payload = 0
                iface_collectives = 0
            elif args.spike_interface_collective == "allgather":
                iface_payload = endpoint_bytes * (p - 1)
                iface_collectives = 1
            else:
                iface_payload = 2 * endpoint_bytes * (p - 1) / p
                iface_collectives = 2
            print(f"  interface collective         : {args.spike_interface_collective}")
            print(f"  interface collectives/solve  : {iface_collectives}")
            print(f"  interface payload per solve  : {human_bytes(iface_payload)}")
            print(f"  transpose payload per solve  : {human_bytes(2 * remote_payload)}")

    sync_all_processes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
