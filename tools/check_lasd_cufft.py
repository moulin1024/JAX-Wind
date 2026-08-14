#!/usr/bin/env python3
"""Compare the optional LASD cuFFT filter with the JAX reference filter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, default=21)
    parser.add_argument("--nz", type=int, default=4)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--first-width", type=float, default=3.0)
    parser.add_argument("--second-width", type=float, default=6.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    import jax
    import jax.numpy as jnp

    from jaxwind._jax.lasd_cufft import filter_two_scales

    shape = (args.components, args.nz, args.ny, args.nx)
    values = jax.random.normal(jax.random.key(7), shape, dtype=jnp.float32)
    widths = (args.first_width, args.second_width)

    def reference(field):
        spectrum = jnp.fft.rfftn(field, axes=(-2, -1))
        x_mode = jnp.arange(args.nx // 2 + 1)
        y_mode = jnp.abs(jnp.fft.fftfreq(args.ny) * args.ny)

        def filtered(width):
            cutoff_x = jnp.floor(args.nx / (2.0 * width) + 0.5)
            cutoff_y = jnp.floor(args.ny / (2.0 * width) + 0.5)
            mask = (y_mode[:, None] < cutoff_y) & (x_mode[None, :] < cutoff_x)
            return jnp.fft.irfftn(
                spectrum * mask[None, ...],
                s=(args.ny, args.nx),
                axes=(-2, -1),
            ).astype(field.dtype)

        return jnp.concatenate(tuple(filtered(width) for width in widths))

    expected = jax.jit(reference)(values)
    actual = jax.jit(filter_two_scales)(values, *widths)
    expected, actual = jax.block_until_ready((expected, actual))
    difference = actual - expected
    print(
        {
            "shape": shape,
            "maximum_absolute_error": float(jnp.max(jnp.abs(difference))),
            "root_mean_square_error": float(jnp.sqrt(jnp.mean(difference**2))),
            "reference_maximum": float(jnp.max(jnp.abs(expected))),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
