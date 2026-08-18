#!/usr/bin/env python3
"""Compare the discrete JAX and legacy-Fortran prescribed-ADM kernels."""

from __future__ import annotations

import json

import jax
jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp
import numpy as np

from jaxwind._jax.actuator_disk import gaussian_convolved_annulus


NX, NY, NZ = 128, 64, 256
DX, DY, DZ = 32.0, 16.0, 4.0
XT, YT, ZT = 1000.0, 512.0, 119.0
RADIUS, EPSILON = 89.15, 32.0


def jax_kernel() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    x = (np.arange(NX) + 0.5) * DX
    y = (np.arange(NY) + 0.5) * DY
    z = (np.arange(NZ) + 0.5) * DZ
    xx = (x - XT)[None, None, :]
    rr = np.sqrt((y[None, :, None] - YT) ** 2 + (z[:, None, None] - ZT) ** 2)
    radial = np.asarray(
        gaussian_convolved_annulus(
            jnp.asarray(rr),
            outer_radius=RADIUS,
            inner_radius=0.0,
            smoothing_width=EPSILON,
        )
    )
    raw = radial * np.exp(-(xx / EPSILON) ** 2)
    kernel = raw * (np.pi * RADIUS**2) / (raw.sum() * DX * DY * DZ)
    return kernel, (x, y, z)


def fortran_kernel() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    # These are the coordinates used literally by force_projection_pre.
    x = (np.arange(NX) + 1.0) * DX
    y = (np.arange(NY) + 1.0) * DY
    z = (np.arange(NZ) + 0.5) * DZ
    kernel = np.zeros((NZ, NY, NX), dtype=np.float64)
    nphi, nr = 64, 32
    dr = RADIUS / nr
    dphi = 2.0 * np.pi / nphi
    ix = np.arange(int(XT / DX) + 1 - 8, int(XT / DX) + 17 - 8) - 1
    iy = np.arange(int(YT / DY) + 1 - 16, int(YT / DY) + 33 - 16) - 1
    xx = x[ix][None, None, :]
    yy = y[iy][None, :, None]
    zz = z[:, None, None]
    for iphi in range(nphi):
        angle = iphi * dphi
        for ir in range(nr):
            radius = (ir + 0.5) * dr
            mu_y = radius * np.cos(angle)
            mu_z = radius * np.sin(angle)
            raw = np.exp(
                -(
                    ((xx - XT) / EPSILON) ** 2
                    + ((yy - YT - mu_y) / EPSILON) ** 2
                    + ((zz - ZT - mu_z) / EPSILON) ** 2
                )
            )
            area = radius * dphi * dr
            kernel[np.ix_(np.arange(NZ), iy, ix)] += area * raw / raw.sum()
    kernel /= DX * DY * DZ
    return kernel, (x, y, z)


def metrics(values: np.ndarray, coordinates: tuple[np.ndarray, ...]) -> dict:
    x, y, z = coordinates
    measure = values * DX * DY * DZ
    total = measure.sum()
    return {
        "integral_m2": float(total),
        "centroid_m": [
            float((measure * x[None, None, :]).sum() / total),
            float((measure * y[None, :, None]).sum() / total),
            float((measure * z[:, None, None]).sum() / total),
        ],
        "standard_deviation_m": [
            float(np.sqrt((measure * (x[None, None, :] - XT) ** 2).sum() / total)),
            float(np.sqrt((measure * (y[None, :, None] - YT) ** 2).sum() / total)),
            float(np.sqrt((measure * (z[:, None, None] - ZT) ** 2).sum() / total)),
        ],
    }


def main() -> None:
    jk, jc = jax_kernel()
    fk, fc = fortran_kernel()
    # Compare transverse/streamwise marginals in their physical coordinates.
    jm = [jk.sum(axis=axes) * DX * DY * DZ for axes in ((0, 1), (0, 2), (1, 2))]
    fm = [fk.sum(axis=axes) * DX * DY * DZ for axes in ((0, 1), (0, 2), (1, 2))]
    marginal_l1 = []
    for jvalues, fvalues, jcoord, fcoord in zip(jm, fm, jc, fc, strict=True):
        interpolated = np.interp(jcoord, fcoord, fvalues, left=0.0, right=0.0)
        marginal_l1.append(float(np.sum(np.abs(jvalues - interpolated))))
    result = {
        "jax": metrics(jk, jc),
        "fortran": metrics(fk, fc),
        "marginal_l1_area_m2": marginal_l1,
        "relative_marginal_l1": [value / (np.pi * RADIUS**2) for value in marginal_l1],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
