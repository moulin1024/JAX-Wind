#!/usr/bin/env python3
"""Self-contained neutral-ABL LASD demo in JAX.

The numerical path matches the current JAX-Wind baseline: cell-centred
horizontal velocity, face-staggered vertical velocity, conservative momentum
advection, horizontal three-halves padding for every nonlinear product,
Lagrangian scale-dependent dynamic Smagorinsky stress, AB2, and an in-file
spectral/finite-difference pressure projection.  It imports only JAX and NumPy;
no ``jaxwind`` or ``spectral_fd`` package is required.

Quick check::

    python neutral_abl_lasd_jax.py --backend cpu --nx 16 --ny 16 --nz 16 \
        --steps 10

Default GPU run::

    python neutral_abl_lasd_jax.py --backend gpu
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Callable, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


Array = jax.Array


@dataclass(frozen=True)
class Config:
    nx: int = 64
    ny: int = 64
    nz: int = 64
    steps: int = 57_600
    lx: float = 4000.0
    ly: float = 4000.0
    lz: float = 1000.0
    bl_height: float = 1000.0
    ustar: float = 0.38
    z0: float = 0.005
    dtr: float = 0.5
    kappa: float = 0.4
    padding_ratio: float = 1.5
    filter_grid_ratio: float = 1.5
    test_filter_ratio: float = 2.0
    lasd_update_interval: int = 5
    lasd_timescale: float = 1.5
    coefficient_initial: float = 0.03
    coefficient_minimum: float = 1.0e-6
    coefficient_maximum: float = 0.81
    statistics_start: int = 28_800
    sample_every: int = 20

    @property
    def zi(self) -> float:
        return self.lx / (2.0 * math.pi)

    @property
    def ex(self) -> float:
        return self.lx / self.zi

    @property
    def ey(self) -> float:
        return self.ly / self.zi

    @property
    def ez(self) -> float:
        return self.lz / self.zi

    @property
    def dx(self) -> float:
        return self.ex / self.nx

    @property
    def dy(self) -> float:
        return self.ey / self.ny

    @property
    def dz(self) -> float:
        return self.ez / self.nz

    @property
    def dt(self) -> float:
        return self.dtr / self.zi

    @property
    def pressure_force(self) -> float:
        return self.ustar**2 / (self.bl_height / self.zi)


class Flow(NamedTuple):
    u: Array
    v: Array
    w_upper: Array
    w_lower: Array
    u_upper: Array
    v_upper: Array
    u_lower: Array
    v_lower: Array
    w_cell: Array
    w_next: Array
    cell_gradients: tuple[Array, ...]
    face_gradients: tuple[Array, ...]


class State(NamedTuple):
    velocity: Array
    previous_rhs: Array
    coefficient: Array
    lm: Array
    mm: Array
    qn: Array
    nn: Array
    trajectory: Array
    iteration: Array


class Operators(NamedTuple):
    step: Callable
    diagnostics: Callable
    accumulate: Callable


def select_device(requested: str) -> jax.Device:
    if requested == "cpu":
        return jax.devices("cpu")[0]
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if requested == "gpu" and not devices:
        raise SystemExit("--backend gpu requested, but CUDA JAX is unavailable")
    return devices[0] if devices else jax.devices("cpu")[0]


def initial_velocity(cfg: Config, restart: Path | None) -> Array:
    shape = (3, cfg.nz, cfg.ny, cfg.nx)
    if restart is not None:
        host = np.empty(shape, dtype=np.float32)
        count = cfg.nz * cfg.ny * cfg.nx
        for component, name in enumerate(("u", "v", "w")):
            values = np.fromfile(restart / f"{name}.bin", dtype=np.float32)
            if values.size != count:
                raise SystemExit(
                    f"invalid restart {name}.bin: {values.size} != {count}"
                )
            host[component] = values.reshape(cfg.nz, cfg.ny, cfg.nx)
        host[2, -1] = 0.0
        return jnp.asarray(host)

    x = jnp.arange(cfg.nx, dtype=jnp.float32) * cfg.dx
    y = jnp.arange(cfg.ny, dtype=jnp.float32) * cfg.dy
    zc = (jnp.arange(cfg.nz, dtype=jnp.float32) + 0.5) * cfg.dz
    zf = (jnp.arange(cfg.nz, dtype=jnp.float32) + 1.0) * cfg.dz
    base = cfg.ustar / cfg.kappa * jnp.log(jnp.maximum(zc * cfg.zi / cfg.z0, 1.0001))
    envelope = jnp.sin(math.pi * zf / cfg.ez)
    lower_envelope = jnp.concatenate((jnp.zeros_like(envelope[:1]), envelope[:-1]))
    envelope_difference = (envelope - lower_envelope) / cfg.dz
    kx_seed, ky_seed = 2.0 * math.pi / cfg.ex, 2.0 * math.pi / cfg.ey
    x_amplitude, y_amplitude = 0.01, 0.007
    u = base[:, None, None] + (
        x_amplitude
        * envelope_difference[:, None, None]
        * jnp.cos(kx_seed * x)[None, None, :]
        / kx_seed
    )
    v = (
        y_amplitude
        * envelope_difference[:, None, None]
        * jnp.cos(ky_seed * y)[None, :, None]
        / ky_seed
    )
    u = jnp.broadcast_to(u, (cfg.nz, cfg.ny, cfg.nx))
    v = jnp.broadcast_to(v, (cfg.nz, cfg.ny, cfg.nx))
    w = envelope[:, None, None] * (
        x_amplitude * jnp.sin(kx_seed * x)[None, None, :]
        + y_amplitude * jnp.sin(ky_seed * y)[None, :, None]
    )
    w = jnp.broadcast_to(w, (cfg.nz, cfg.ny, cfg.nx)).at[-1].set(0.0)
    return jnp.stack((u, v, w))


def make_operators(cfg: Config) -> Operators:
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    pnx = int(math.ceil(cfg.padding_ratio * nx))
    pny = int(math.ceil(cfg.padding_ratio * ny))
    kx = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=cfg.dx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=cfg.dy)
    keep = jnp.ones((ny, nx // 2 + 1), dtype=jnp.float32)
    if nx % 2 == 0:
        kx = kx.at[-1].set(0.0)
        keep = keep.at[:, -1].set(0.0)
    if ny % 2 == 0:
        ky = ky.at[ny // 2].set(0.0)
        keep = keep.at[ny // 2].set(0.0)
    pkx = 2.0 * jnp.pi * jnp.fft.rfftfreq(pnx, d=cfg.ex / pnx)
    pky = 2.0 * jnp.pi * jnp.fft.fftfreq(pny, d=cfg.ey / pny)
    if pnx % 2 == 0:
        pkx = pkx.at[-1].set(0.0)
    if pny % 2 == 0:
        pky = pky.at[pny // 2].set(0.0)
    pad_y0 = pny // 2 - ny // 2
    pad_y1 = pny - ny - pad_y0
    pad_x1 = pnx // 2 + 1 - (nx // 2 + 1)
    y_index = jnp.arange(ny)
    y_mode = jnp.where(y_index <= (ny - 1) // 2, y_index, y_index - ny)
    y_in_pad = y_mode % pny
    y_opposite = (-y_mode) % pny

    def pad_horizontal(values: Array) -> Array:
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1)) * keep
        shifted = jnp.fft.fftshift(spectrum, axes=(-2,))
        padded = jnp.pad(
            shifted,
            ((0, 0),) * (values.ndim - 2) + ((pad_y0, pad_y1), (0, pad_x1)),
        )
        scale = (pny * pnx) / (ny * nx)
        return (
            jnp.fft.irfftn(
                jnp.fft.ifftshift(padded, axes=(-2,)),
                s=(pny, pnx),
                axes=(-2, -1),
            )
            * scale
        ).astype(values.dtype)

    def truncate_spectrum(values: Array) -> Array:
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        shifted = jnp.fft.fftshift(spectrum, axes=(-2,))
        cropped = jnp.fft.ifftshift(
            shifted[..., pad_y0 : pad_y0 + ny, : nx // 2 + 1], axes=(-2,)
        )
        if ny % 2 == 0:
            cropped = cropped.at[..., ny // 2, :].set(
                0.5
                * (
                    spectrum[..., (-ny // 2) % pny, : nx // 2 + 1]
                    + spectrum[..., ny // 2, : nx // 2 + 1]
                )
            )
        if nx % 2 == 0:
            x_nyquist = 0.5 * (
                jnp.conj(spectrum[..., y_opposite, nx // 2])
                + spectrum[..., y_in_pad, nx // 2]
            )
            cropped = cropped.at[..., -1].set(x_nyquist)
            if ny % 2 == 0:
                cropped = cropped.at[..., ny // 2, -1].set(
                    spectrum[..., ny // 2, nx // 2].real
                )
        return cropped * ((ny * nx) / (pny * pnx))

    def inverse_base(spectrum: Array, dtype) -> Array:
        return jnp.fft.irfftn(spectrum, s=(ny, nx), axes=(-2, -1)).astype(dtype)

    def truncate(values: Array) -> Array:
        return inverse_base(truncate_spectrum(values), values.dtype)

    def gradient_pair(values: Array, padded: bool = False) -> tuple[Array, Array]:
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        local_kx, local_ky = (pkx, pky) if padded else (kx, ky)
        local_keep = 1.0 if padded else keep
        shape = (pny, pnx) if padded else (ny, nx)
        gradients = jnp.fft.irfftn(
            jnp.stack(
                (
                    spectrum * (1j * local_kx) * local_keep,
                    spectrum * (1j * local_ky[:, None]) * local_keep,
                )
            ),
            s=shape,
            axes=(-2, -1),
        ).astype(values.dtype)
        return gradients[0], gradients[1]

    def flux_divergence(x_flux: Array, y_flux: Array) -> Array:
        count = x_flux.shape[0]
        spectra = truncate_spectrum(jnp.concatenate((x_flux, y_flux)))
        return inverse_base(
            (spectra[:count] * (1j * kx) + spectra[count:] * (1j * ky[:, None])) * keep,
            x_flux.dtype,
        )

    def base_flux_divergence_spectra(xs: Array, ys: Array, dtype) -> Array:
        return inverse_base((xs * (1j * kx) + ys * (1j * ky[:, None])) * keep, dtype)

    def base_derivative(values: Array, axis: int) -> Array:
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        wave = kx if axis == 0 else ky[:, None]
        return inverse_base(spectrum * (1j * wave) * keep, values.dtype)

    def test_filter(values: Array, width: float) -> Array:
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        xm = jnp.arange(nx // 2 + 1)
        ym = jnp.abs(jnp.fft.fftfreq(ny) * ny)
        mask = (xm[None] < jnp.floor(nx / (2.0 * width) + 0.5)) & (
            ym[:, None] < jnp.floor(ny / (2.0 * width) + 0.5)
        )
        return inverse_base(spectrum * mask, values.dtype)

    def wall_filter(values: Array) -> Array:
        width = cfg.filter_grid_ratio * cfg.test_filter_ratio
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        xm = jnp.arange(nx // 2 + 1)
        ym = jnp.abs(jnp.fft.fftfreq(ny) * ny)
        mask = (xm[None] < jnp.floor(nx / (2.0 * width))) & (
            ym[:, None] < jnp.floor(ny / (2.0 * width))
        )
        return inverse_base(spectrum * mask, values.dtype)

    def make_flow(velocity: Array) -> Flow:
        u, v, w_upper = velocity
        next_u = jnp.concatenate((u[1:], u[-1:]))
        next_v = jnp.concatenate((v[1:], v[-1:]))
        lower_w = jnp.concatenate((jnp.zeros_like(w_upper[:1]), w_upper[:-1]))
        w_cell = 0.5 * (lower_w + w_upper)
        w_next = jnp.concatenate((w_cell[1:], w_cell[-1:]))
        u_upper, v_upper = 0.5 * (u + next_u), 0.5 * (v + next_v)
        u_lower, v_lower = u[0], v[0]
        gx, gy = gradient_pair(jnp.stack((u, v, w_cell, w_upper)))
        dudx, dvdx, dwdx_cell, dwdx_upper = gx
        dudy, dvdy, dwdy_cell, dwdy_upper = gy
        dudz_upper = ((next_u - u) / cfg.dz).at[-1].set(0.0)
        dvdz_upper = ((next_v - v) / cfg.dz).at[-1].set(0.0)
        correction = 1.0 / math.log(3.0) - 1.0
        dudz_upper = dudz_upper.at[0].add(correction * jnp.mean(dudz_upper[0]))
        dvdz_upper = dvdz_upper.at[0].add(correction * jnp.mean(dvdz_upper[0]))
        dudz_lower = jnp.concatenate((jnp.zeros_like(u[:1]), dudz_upper[:-1]))
        dvdz_lower = jnp.concatenate((jnp.zeros_like(v[:1]), dvdz_upper[:-1]))
        dwdz = (w_upper - lower_w) / cfg.dz
        next_dudx = jnp.concatenate((dudx[1:], dudx[-1:]))
        next_dudy = jnp.concatenate((dudy[1:], dudy[-1:]))
        next_dvdx = jnp.concatenate((dvdx[1:], dvdx[-1:]))
        next_dvdy = jnp.concatenate((dvdy[1:], dvdy[-1:]))
        next_dwdz = jnp.concatenate((dwdz[1:], jnp.zeros_like(dwdz[-1:])))
        cell = (
            dudx,
            dudy,
            0.5 * (dudz_lower + dudz_upper),
            dvdx,
            dvdy,
            0.5 * (dvdz_lower + dvdz_upper),
            dwdx_cell,
            dwdy_cell,
            dwdz,
        )
        face = (
            0.5 * (dudx + next_dudx),
            0.5 * (dudy + next_dudy),
            dudz_upper,
            0.5 * (dvdx + next_dvdx),
            0.5 * (dvdy + next_dvdy),
            dvdz_upper,
            dwdx_upper,
            dwdy_upper,
            0.5 * (dwdz + next_dwdz),
        )
        return Flow(
            u,
            v,
            w_upper,
            jnp.zeros_like(w_upper[0]),
            u_upper,
            v_upper,
            u_lower,
            v_lower,
            w_cell,
            w_next,
            cell,
            face,
        )

    def padded_state_and_gradients(flow: Flow):
        padded = pad_horizontal(
            jnp.stack(
                (
                    flow.u,
                    flow.v,
                    flow.w_upper,
                    flow.u_upper,
                    flow.v_upper,
                    flow.w_cell,
                    flow.w_next,
                )
            )
        )
        lower = pad_horizontal(jnp.stack((flow.w_lower, flow.u_lower, flow.v_lower)))
        px, py = gradient_pair(
            jnp.stack(
                (padded[0], padded[1], padded[5], padded[3], padded[4], padded[2])
            ),
            padded=True,
        )
        lower_u = jnp.concatenate((lower[1][None], padded[3][:-1]))
        lower_v = jnp.concatenate((lower[2][None], padded[4][:-1]))
        dudz_cell = (padded[3] - lower_u) / cfg.dz
        dvdz_cell = (padded[4] - lower_v) / cfg.dz
        dudz_face = 2.0 * (padded[3] - padded[0]) / cfg.dz
        dvdz_face = 2.0 * (padded[4] - padded[1]) / cfg.dz
        correction = 1.0 / math.log(3.0) - 1.0
        du_corr = correction * jnp.mean(dudz_face[0])
        dv_corr = correction * jnp.mean(dvdz_face[0])
        dudz_face = dudz_face.at[0].add(du_corr)
        dvdz_face = dvdz_face.at[0].add(dv_corr)
        dudz_cell = dudz_cell.at[0].add(0.5 * du_corr)
        dvdz_cell = dvdz_cell.at[0].add(0.5 * dv_corr)
        if nz > 1:
            dudz_cell = dudz_cell.at[1].add(0.5 * du_corr)
            dvdz_cell = dvdz_cell.at[1].add(0.5 * dv_corr)
        cell = (
            px[0],
            py[0],
            dudz_cell,
            px[1],
            py[1],
            dvdz_cell,
            px[2],
            py[2],
            2.0 * (padded[2] - padded[5]) / cfg.dz,
        )
        face = (
            px[3],
            py[3],
            dudz_face,
            px[4],
            py[4],
            dvdz_face,
            px[5],
            py[5],
            (padded[6] - padded[5]) / cfg.dz,
        )
        return padded, lower, cell, face

    def conservative_advection(flow: Flow, padded: Array, lower: Array) -> Array:
        u, v, w, uu, vu, wc, wn = padded
        uv = u * v
        horizontal = flux_divergence(
            jnp.stack((u * u, uv, uu * w)),
            jnp.stack((uv, v * v, vu * w)),
        )
        upper_u, upper_v, wz, wzn = truncate(
            jnp.stack((w * uu, w * vu, wc * wc, wn * wn))
        )
        lower_u, lower_v = truncate(
            jnp.stack((lower[0] * lower[1], lower[0] * lower[2]))
        )
        lower_u = jnp.concatenate((lower_u[None], upper_u[:-1]))
        lower_v = jnp.concatenate((lower_v[None], upper_v[:-1]))
        result = jnp.stack(
            (
                -(horizontal[0] + (upper_u - lower_u) / cfg.dz),
                -(horizontal[1] + (upper_v - lower_v) / cfg.dz),
                -(horizontal[2] + (wzn - wz) / cfg.dz),
            )
        )
        return result.at[2, -1].set(0.0)

    def strain_magnitude(g: tuple[Array, ...]) -> Array:
        sxy = 0.5 * (g[1] + g[3])
        sxz = 0.5 * (g[2] + g[6])
        syz = 0.5 * (g[5] + g[7])
        dot = g[0] ** 2 + g[4] ** 2 + g[8] ** 2 + 2.0 * (sxy**2 + sxz**2 + syz**2)
        return jnp.sqrt(jnp.maximum(2.0 * dot, 0.0))

    def sgs_tendency(flow: Flow, cell, face, coefficient: Array) -> Array:
        delta = (cfg.dx * cfg.dy * cfg.dz) ** (1.0 / 3.0)
        pc = jnp.clip(
            pad_horizontal(coefficient),
            cfg.coefficient_minimum,
            cfg.coefficient_maximum,
        )
        next_pc = jnp.concatenate((pc[1:], pc[-1:]))
        nu_cell = pc * delta**2 * strain_magnitude(cell)
        nu_face = 0.5 * (pc + next_pc) * delta**2 * strain_magnitude(face)
        txx, txy, tyy, tzz = truncate_spectrum(
            jnp.stack(
                (
                    -2.0 * nu_cell * cell[0],
                    -nu_cell * (cell[1] + cell[3]),
                    -2.0 * nu_cell * cell[4],
                    -2.0 * nu_cell * cell[8],
                )
            )
        )
        txz, tyz = truncate_spectrum(
            jnp.stack((-nu_face * (face[2] + face[6]), -nu_face * (face[5] + face[7])))
        )
        txz = txz.at[-1].set(0.0)
        tyz = tyz.at[-1].set(0.0)
        txz_r, tyz_r, tzz_r = inverse_base(
            jnp.stack((txz, tyz, tzz)), coefficient.dtype
        )
        lower_txz = jnp.concatenate((jnp.zeros_like(txz_r[:1]), txz_r[:-1]))
        lower_tyz = jnp.concatenate((jnp.zeros_like(tyz_r[:1]), tyz_r[:-1]))
        next_tzz = jnp.concatenate((tzz_r[1:], tzz_r[-1:]))
        horizontal = base_flux_divergence_spectra(
            jnp.stack((txx, txy, txz)), jnp.stack((txy, tyy, tyz)), coefficient.dtype
        )
        result = jnp.stack(
            (
                -(horizontal[0] + (txz_r - lower_txz) / cfg.dz),
                -(horizontal[1] + (tyz_r - lower_tyz) / cfg.dz),
                -(horizontal[2] + (next_tzz - tzz_r) / cfg.dz),
            )
        )
        return result.at[2, -1].set(0.0)

    def wall_tendency(flow: Flow) -> Array:
        uv = wall_filter(jnp.stack((flow.u[0], flow.v[0])))
        pu, pv = pad_horizontal(uv)
        speed = jnp.hypot(pu, pv)
        drag = (cfg.kappa / math.log(0.5 * cfg.dz * cfg.zi / cfg.z0)) ** 2
        wx, wy = truncate(
            jnp.stack((-drag * speed * pu / cfg.dz, -drag * speed * pv / cfg.dz))
        )
        result = jnp.zeros((3, nz, ny, nx), dtype=flow.u.dtype)
        return result.at[0, 0].set(wx).at[1, 0].set(wy)

    def packed_strain(flow: Flow) -> Array:
        g = flow.cell_gradients
        return jnp.stack(
            (
                g[0],
                0.5 * (g[1] + g[3]),
                0.5 * (g[2] + g[6]),
                g[4],
                0.5 * (g[5] + g[7]),
                g[8],
            )
        )

    def sym_dot(a: Array, b: Array) -> Array:
        return (
            a[0] * b[0]
            + 2.0 * a[1] * b[1]
            + 2.0 * a[2] * b[2]
            + a[3] * b[3]
            + 2.0 * a[4] * b[4]
            + a[5] * b[5]
        )

    def contractions(flow: Flow, ratio: float) -> tuple[Array, Array]:
        strain = packed_strain(flow)
        magnitude = jnp.sqrt(jnp.maximum(2.0 * sym_dot(strain, strain), 0.0))
        velocity = jnp.stack((flow.u, flow.v, flow.w_cell))
        products = jnp.stack(
            (
                velocity[0] ** 2,
                velocity[0] * velocity[1],
                velocity[0] * velocity[2],
                velocity[1] ** 2,
                velocity[1] * velocity[2],
                velocity[2] ** 2,
            )
        )
        width = cfg.filter_grid_ratio * ratio
        vh, ph, sh = (
            test_filter(velocity, width),
            test_filter(products, width),
            test_filter(strain, width),
        )
        resolved = ph - jnp.stack(
            (
                vh[0] ** 2,
                vh[0] * vh[1],
                vh[0] * vh[2],
                vh[1] ** 2,
                vh[1] * vh[2],
                vh[2] ** 2,
            )
        )
        filtered_mag_strain = test_filter(magnitude[None] * strain, width)
        delta = (cfg.dx * cfg.dy * cfg.dz) ** (1.0 / 3.0)
        model = (
            2.0
            * delta**2
            * (
                filtered_mag_strain
                - ratio**2
                * jnp.sqrt(jnp.maximum(2.0 * sym_dot(sh, sh), 0.0))[None]
                * sh
            )
        )
        return sym_dot(resolved, model), sym_dot(model, model)

    def departure(values: Array, trajectory: Array) -> Array:
        extended = jnp.concatenate((values[:1], values, values[-1:]))
        zk = jnp.arange(nz, dtype=values.dtype)[:, None, None]
        yj = jnp.arange(ny, dtype=values.dtype)[None, :, None]
        xi = jnp.arange(nx, dtype=values.dtype)[None, None, :]
        interval_dt = cfg.dt * cfg.lasd_update_interval
        x = jnp.mod(xi - trajectory[0] * interval_dt / cfg.dx, nx)
        y = jnp.mod(yj - trajectory[1] * interval_dt / cfg.dy, ny)
        z = jnp.clip(zk - trajectory[2] * interval_dt / cfg.dz, -1.0, float(nz))
        i0, j0, k0 = (
            jnp.floor(x).astype(jnp.int32),
            jnp.floor(y).astype(jnp.int32),
            jnp.floor(z).astype(jnp.int32) + 1,
        )
        i1, j1, k1 = (i0 + 1) % nx, (j0 + 1) % ny, jnp.minimum(k0 + 1, nz + 1)
        fx, fy, fz = x - jnp.floor(x), y - jnp.floor(y), z - jnp.floor(z)
        q000, q100 = extended[k0, j0, i0], extended[k0, j0, i1]
        q010, q110 = extended[k0, j1, i0], extended[k0, j1, i1]
        q001, q101 = extended[k1, j0, i0], extended[k1, j0, i1]
        q011, q111 = extended[k1, j1, i0], extended[k1, j1, i1]
        q00, q10 = (1 - fx) * q000 + fx * q100, (1 - fx) * q010 + fx * q110
        q01, q11 = (1 - fx) * q001 + fx * q101, (1 - fx) * q011 + fx * q111
        return (1 - fz) * ((1 - fy) * q00 + fy * q10) + fz * ((1 - fy) * q01 + fy * q11)

    def safe_divide(a: Array, b: Array) -> Array:
        valid = jnp.abs(b) > 1.0e-30
        return jnp.where(valid, a / jnp.where(valid, b, 1.0), 0.0)

    def average(current_a, current_b, old_a, old_b, trajectory):
        product = old_a * old_b
        valid = (old_a > 0.0) & (old_b >= 0.0) & (product > 0.0)
        delta = (cfg.dx * cfg.dy * cfg.dz) ** (1.0 / 3.0)
        timescale = (
            cfg.lasd_timescale * delta * jnp.where(valid, product ** (-0.125), 1.0)
        )
        interval_dt = cfg.dt * cfg.lasd_update_interval
        weight = jnp.where(
            valid, (interval_dt / timescale) / (1.0 + interval_dt / timescale), 0.0
        )
        return weight * current_a + (1.0 - weight) * departure(
            old_a, trajectory
        ), jnp.maximum(
            weight * current_b + (1.0 - weight) * departure(old_b, trajectory), 0.0
        )

    def update_lasd(flow: Flow, state: State, trajectory: Array):
        lm, mm = contractions(flow, cfg.test_filter_ratio)
        qn, nn = contractions(flow, cfg.test_filter_ratio**2)
        first = state.iteration + 1 == cfg.lasd_update_interval
        old = (
            jnp.where(first, cfg.coefficient_initial * mm, state.lm),
            jnp.where(first, mm, state.mm),
            jnp.where(first, cfg.coefficient_initial * nn, state.qn),
            jnp.where(first, nn, state.nn),
        )
        old = tuple(x.at[0].set(x[1]).at[-1].set(x[-2]) for x in old)
        lm, mm = average(lm, mm, old[0], old[1], trajectory)
        qn, nn = average(qn, nn, old[2], old[3], trajectory)
        c2, c4 = (
            jnp.maximum(safe_divide(lm, mm), 0.0),
            jnp.maximum(safe_divide(qn, nn), 0.0),
        )
        exponent = math.log(cfg.test_filter_ratio) / (
            math.log(cfg.test_filter_ratio**2) - math.log(cfg.test_filter_ratio)
        )
        beta = jnp.maximum(
            jnp.maximum(safe_divide(c4, c2), 0.0) ** exponent,
            1.0 / cfg.test_filter_ratio**3,
        )
        coefficient = jnp.clip(
            safe_divide(c2, beta), cfg.coefficient_minimum, cfg.coefficient_maximum
        )
        return coefficient, lm, mm, qn, nn

    inv_dz2 = 1.0 / cfg.dz**2
    pressure_k2 = kx[None] ** 2 + ky[:, None] ** 2

    def project(velocity: Array) -> Array:
        u, v, w = velocity
        lower_w = jnp.concatenate((jnp.zeros_like(w[:1]), w[:-1]))
        divergence = (
            base_derivative(u, 0) + base_derivative(v, 1) + (w - lower_w) / cfg.dz
        )
        rhs = jnp.fft.rfftn(divergence / cfg.dt, axes=(-2, -1))
        zero = pressure_k2 == 0.0
        b0 = jnp.where(zero, 1.0, -pressure_k2 - inv_dz2)
        c0 = jnp.where(zero, 0.0, inv_dz2)
        cp0 = c0 / b0
        dp0 = jnp.where(zero, 0.0, rhs[0]) / b0

        def forward(carry, row):
            cp, dp = carry
            top = row == nz - 1
            b = -pressure_k2 - jnp.where(top, inv_dz2, 2.0 * inv_dz2)
            c = jnp.where(top, 0.0, inv_dz2)
            denominator = b - inv_dz2 * cp
            new_cp = c / denominator
            new_dp = (rhs[row] - inv_dz2 * dp) / denominator
            return (new_cp, new_dp), (new_cp, new_dp)

        _, (cp_tail, dp_tail) = lax.scan(forward, (cp0, dp0), jnp.arange(1, nz))
        cp, dp = (
            jnp.concatenate((cp0[None], cp_tail)),
            jnp.concatenate((dp0[None], dp_tail)),
        )
        pressure = jnp.zeros_like(dp).at[-1].set(dp[-1])

        def backward(index, values):
            row = nz - 2 - index
            return values.at[row].set(dp[row] - cp[row] * values[row + 1])

        pressure = lax.fori_loop(0, nz - 1, backward, pressure)
        dpdx = inverse_base(pressure * (1j * kx) * keep, u.dtype)
        dpdy = inverse_base(pressure * (1j * ky[:, None]) * keep, v.dtype)
        dpdz = jnp.concatenate(
            ((pressure[1:] - pressure[:-1]) / cfg.dz, jnp.zeros_like(pressure[:1]))
        )
        corrected = velocity - cfg.dt * jnp.stack(
            (dpdx, dpdy, inverse_base(dpdz, w.dtype))
        )
        return corrected.at[2, -1].set(0.0)

    def step_impl(state: State) -> State:
        flow = make_flow(state.velocity)
        trajectory = (
            state.trajectory
            + jnp.stack((flow.u, flow.v, flow.w_cell)) / cfg.lasd_update_interval
        )
        update_now = (state.iteration + 1) % cfg.lasd_update_interval == 0

        def update(_):
            coefficient, lm, mm, qn, nn = update_lasd(flow, state, trajectory)
            return coefficient, lm, mm, qn, nn, jnp.zeros_like(trajectory)

        def retain(_):
            return state.coefficient, state.lm, state.mm, state.qn, state.nn, trajectory

        coefficient, lm, mm, qn, nn, trajectory = lax.cond(
            update_now, update, retain, operand=None
        )
        padded, lower, cell, face = padded_state_and_gradients(flow)
        rhs = (
            conservative_advection(flow, padded, lower)
            + sgs_tendency(flow, cell, face, coefficient)
            + wall_tendency(flow)
        )
        forcing = ((jnp.arange(nz) + 0.5) * cfg.dz <= cfg.bl_height / cfg.zi)[
            :, None, None
        ]
        rhs = rhs.at[0].add(cfg.pressure_force * forcing)
        tendency = jnp.where(
            state.iteration == 0, rhs, 1.5 * rhs - 0.5 * state.previous_rhs
        )
        velocity = project(state.velocity + cfg.dt * tendency)
        return State(
            velocity, rhs, coefficient, lm, mm, qn, nn, trajectory, state.iteration + 1
        )

    def diagnostics_impl(state: State) -> Array:
        flow = make_flow(state.velocity)
        divergence = (
            base_derivative(flow.u, 0)
            + base_derivative(flow.v, 1)
            + (
                flow.w_upper
                - jnp.concatenate((jnp.zeros_like(flow.w_upper[:1]), flow.w_upper[:-1]))
            )
            / cfg.dz
        )
        cfl = jnp.maximum(
            jnp.max(jnp.abs(flow.u)) * cfg.dt / cfg.dx,
            jnp.maximum(
                jnp.max(jnp.abs(flow.v)) * cfg.dt / cfg.dy,
                jnp.max(jnp.abs(flow.w_upper)) * cfg.dt / cfg.dz,
            ),
        )
        return jnp.asarray(
            (
                jnp.mean(flow.u),
                jnp.max(jnp.abs(divergence)),
                jnp.mean(state.coefficient),
                jnp.max(state.coefficient),
                cfl * cfg.lasd_update_interval,
            )
        )

    def accumulate_impl(statistics: Array, state: State) -> Array:
        flow = make_flow(state.velocity)
        u, v, w = flow.u, flow.v, flow.w_cell
        return statistics + jnp.stack((u, v, w, u * u, v * v, w * w, u * v))

    return Operators(
        jax.jit(step_impl, donate_argnums=(0,)),
        jax.jit(diagnostics_impl),
        jax.jit(accumulate_impl, donate_argnums=(0,)),
    )


def write_outputs(state: State, statistics, samples: int, cfg: Config, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    velocity = np.asarray(jax.device_get(state.velocity), dtype=np.float32)
    coefficient = np.asarray(jax.device_get(state.coefficient), dtype=np.float32)
    z = (np.arange(cfg.nz) + 0.5) * cfg.dz * cfg.zi
    w_cell = 0.5 * (
        velocity[2]
        + np.concatenate((np.zeros_like(velocity[2, :1]), velocity[2, :-1]), axis=0)
    )
    profile = np.column_stack(
        (
            z,
            velocity[0].mean((1, 2)),
            velocity[1].mean((1, 2)),
            w_cell.mean((1, 2)),
            coefficient.mean((1, 2)),
        )
    )
    np.savetxt(
        output / "lasd_profile.csv",
        profile,
        delimiter=",",
        header="z_m,mean_u,mean_v,mean_w,mean_cs2",
        comments="",
    )
    coefficient.tofile(output / "lasd_coefficient.bin")
    if statistics is not None and samples:
        averaged = np.asarray(jax.device_get(statistics)) / samples
        for name, values in zip(
            ("u", "v", "w", "u2", "v2", "w2", "uv"), averaged, strict=True
        ):
            values.astype(np.float32).tofile(output / f"ta_{name}.bin")
    print(f"Wrote {output / 'lasd_profile.csv'}")


def run(
    cfg: Config,
    device: jax.Device,
    restart: Path | None,
    output: Path,
    report_every: int,
):
    print(
        f"JAX LASD conservative 3/2 ABL: {cfg.nx}x{cfg.ny}x{cfg.nz}, device={device}",
        flush=True,
    )
    with jax.default_device(device):
        velocity = initial_velocity(cfg, restart)
        shape = (cfg.nz, cfg.ny, cfg.nx)
        state = State(
            velocity,
            jnp.zeros_like(velocity),
            jnp.full(shape, cfg.coefficient_initial, jnp.float32),
            *(jnp.zeros(shape, jnp.float32) for _ in range(4)),
            jnp.zeros((3, *shape), jnp.float32),
            jnp.asarray(0, jnp.int32),
        )
        operators = make_operators(cfg)
        statistics = (
            jnp.zeros((7, *shape), jnp.float32)
            if cfg.steps >= cfg.statistics_start
            else None
        )
        samples = 0
        started = time.perf_counter()
        for step in range(1, cfg.steps + 1):
            state = operators.step(state)
            if (
                statistics is not None
                and step >= cfg.statistics_start
                and step % cfg.sample_every == 0
            ):
                statistics = operators.accumulate(statistics, state)
                samples += 1
            if step == 1 or step % report_every == 0 or step == cfg.steps:
                mean_u, divergence, mean_c, max_c, trajectory_cfl = np.asarray(
                    jax.device_get(operators.diagnostics(state))
                )
                print(
                    f"step={step:7d} t={step * cfg.dtr:9.1f}s <u>={mean_u:.5f} div={divergence:.2e} <Cs2>={mean_c:.4f} maxCs2={max_c:.4f} trajectory-CFL={trajectory_cfl:.3f}",
                    flush=True,
                )
        state.velocity.block_until_ready()
        elapsed = time.perf_counter() - started
        print(
            f"Finished in {elapsed:.2f}s: {cfg.steps / elapsed:.2f} steps/s ({1e3 * elapsed / cfg.steps:.3f} ms/step)"
        )
        write_outputs(state, statistics, samples, cfg, output)


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--nx", type=positive_int, default=Config.nx)
    parser.add_argument("--ny", type=positive_int, default=Config.ny)
    parser.add_argument("--nz", type=positive_int, default=Config.nz)
    parser.add_argument("--steps", type=positive_int, default=Config.steps)
    parser.add_argument("--dtr", type=float, default=Config.dtr)
    parser.add_argument(
        "--lasd-update-interval", type=positive_int, default=Config.lasd_update_interval
    )
    parser.add_argument(
        "--statistics-start", type=positive_int, default=Config.statistics_start
    )
    parser.add_argument(
        "--sample-every", type=positive_int, default=Config.sample_every
    )
    parser.add_argument("--report-every", type=positive_int)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("jax_lasd_output"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.nx % 2 or args.ny % 2 or min(args.nx, args.ny, args.nz) < 4:
        raise SystemExit("nx and ny must be even; every dimension must be >= 4")
    if args.dtr <= 0:
        raise SystemExit("dtr must be positive")
    cfg = Config(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        steps=args.steps,
        dtr=args.dtr,
        lasd_update_interval=args.lasd_update_interval,
        statistics_start=args.statistics_start,
        sample_every=args.sample_every,
    )
    run(
        cfg,
        select_device(args.backend),
        args.restart,
        args.output_dir,
        args.report_every or max(1, cfg.steps // 10),
    )


if __name__ == "__main__":
    main()
