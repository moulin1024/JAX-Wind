#!/usr/bin/env python3
"""Standalone JAX FFT performance benchmark.

Examples:
  python jax_fft_benchmark.py
  python jax_fft_benchmark.py --shape 4096 --shape 1024x1024 --dtype complex64
  python jax_fft_benchmark.py --shape 256x256x256 --op rfftn --dtype float32
  python jax_fft_benchmark.py --shape 1024x1024 --batch 8 --platform gpu

The timed region excludes input creation, host-to-device transfer, and JIT
compilation. JAX dispatch is asynchronous, so every timed sample explicitly
waits for its final result.
"""

from __future__ import annotations

import argparse
import math
import platform as py_platform
import statistics
import sys
import time
from dataclasses import dataclass

import numpy as np


DTYPES = {
    "float32": np.float32,
    "float64": np.float64,
    "complex64": np.complex64,
    "complex128": np.complex128,
}


def parse_shape(text: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(x) for x in text.lower().replace(",", "x").split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape: {text!r}") from exc
    if not shape or any(n <= 0 for n in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def shape_string(shape: tuple[int, ...]) -> str:
    return "x".join(map(str, shape))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def human_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


@dataclass
class Result:
    shape: tuple[int, ...]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_bytes: int
    output_bytes: int
    compile_s: float
    times_s: list[float]


def make_host_input(
    rng: np.random.Generator,
    op: str,
    shape: tuple[int, ...],
    batch: int,
    dtype_name: str,
) -> np.ndarray:
    dtype = DTYPES[dtype_name]
    input_shape = (batch, *shape)
    if op == "irfftn":
        input_shape = (*input_shape[:-1], shape[-1] // 2 + 1)

    if np.issubdtype(dtype, np.complexfloating):
        real_dtype = np.float32 if dtype == np.complex64 else np.float64
        real = rng.standard_normal(input_shape).astype(real_dtype)
        imag = rng.standard_normal(input_shape).astype(real_dtype)
        return (real + 1j * imag).astype(dtype)
    return rng.standard_normal(input_shape).astype(dtype)


def benchmark_one(jax, jnp, args, device, shape: tuple[int, ...]) -> Result:
    rng = np.random.default_rng(args.seed)
    host_x = make_host_input(rng, args.op, shape, args.batch, args.dtype)
    x = jax.device_put(host_x, device=device)
    axes = tuple(range(1, len(shape) + 1))  # never transform the batch axis

    if args.op == "fftn":
        fn = lambda a: jnp.fft.fftn(a, axes=axes, norm=args.norm)
    elif args.op == "ifftn":
        fn = lambda a: jnp.fft.ifftn(a, axes=axes, norm=args.norm)
    elif args.op == "rfftn":
        fn = lambda a: jnp.fft.rfftn(a, axes=axes, norm=args.norm)
    else:
        fn = lambda a: jnp.fft.irfftn(a, s=shape, axes=axes, norm=args.norm)

    fft = jax.jit(fn, device=device)

    # First call: trace + compile + execute. Report it, but do not benchmark it.
    t0 = time.perf_counter()
    y = fft(x)
    y.block_until_ready()
    compile_s = time.perf_counter() - t0

    for _ in range(args.warmup):
        y = fft(x)
    y.block_until_ready()

    samples = []
    for _ in range(args.samples):
        t0 = time.perf_counter()
        for _ in range(args.iterations):
            y = fft(x)
        y.block_until_ready()
        samples.append((time.perf_counter() - t0) / args.iterations)

    return Result(
        shape=shape,
        input_shape=tuple(x.shape),
        output_shape=tuple(y.shape),
        input_bytes=x.size * x.dtype.itemsize,
        output_bytes=y.size * y.dtype.itemsize,
        compile_s=compile_s,
        times_s=samples,
    )


def print_result(result: Result, batch: int, op: str) -> None:
    median_s = statistics.median(result.times_s)
    best_s = min(result.times_s)
    mean_s = statistics.mean(result.times_s)
    p95_s = percentile(result.times_s, 0.95)
    n = math.prod(result.shape)
    transforms_per_s = batch / median_s

    # Conventional estimate for a complex FFT. For real FFTs, report a rough
    # half-cost estimate; this is useful for comparisons, not a hardware FLOP count.
    flop_factor = 2.5 if op in ("rfftn", "irfftn") else 5.0
    estimated_flops = batch * flop_factor * n * math.log2(max(n, 2))
    estimated_gflops = estimated_flops / median_s / 1e9
    io_gib_s = (result.input_bytes + result.output_bytes) / median_s / 2**30

    print(f"\nShape                  : {shape_string(result.shape)}  (batch={batch})")
    print(f"Input -> output        : {result.input_shape} -> {result.output_shape}")
    print(
        f"Input + output bytes   : {human_bytes(result.input_bytes)} + "
        f"{human_bytes(result.output_bytes)}"
    )
    print(f"First call (JIT+run)   : {result.compile_s * 1e3:.3f} ms")
    print(f"Execution median       : {median_s * 1e3:.3f} ms per batch")
    print(f"Execution best         : {best_s * 1e3:.3f} ms per batch")
    print(f"Execution mean / p95   : {mean_s * 1e3:.3f} / {p95_s * 1e3:.3f} ms")
    print(f"Throughput             : {transforms_per_s:,.2f} transforms/s")
    print(f"Estimated FFT rate     : {estimated_gflops:,.2f} GFLOP/s")
    print(f"Approx. array traffic  : {io_gib_s:,.2f} GiB/s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark JIT-compiled JAX FFT execution on one device.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        help="transform shape; repeat for multiple cases (e.g. 4096 or 1024x1024)",
    )
    parser.add_argument("--batch", type=int, default=1, help="number of transforms per call")
    parser.add_argument("--op", choices=("fftn", "ifftn", "rfftn", "irfftn"), default="fftn")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="complex64")
    parser.add_argument("--norm", choices=("backward", "ortho", "forward"), default="backward")
    parser.add_argument("--platform", choices=("cpu", "gpu", "tpu"), help="restrict JAX platform")
    parser.add_argument("--device", type=int, default=0, help="device index within the selected platform")
    parser.add_argument("--warmup", type=int, default=5, help="untimed calls after compilation")
    parser.add_argument("--samples", type=int, default=10, help="number of timing samples")
    parser.add_argument("--iterations", type=int, default=10, help="FFT calls per timing sample")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.shape = args.shape or [(1024, 1024)]

    if args.batch <= 0 or args.warmup < 0 or args.samples <= 0 or args.iterations <= 0:
        raise SystemExit("batch, samples, and iterations must be positive; warmup must be nonnegative")
    if args.op in ("rfftn",) and args.dtype.startswith("complex"):
        raise SystemExit("rfftn requires --dtype float32 or float64")
    if args.op == "irfftn" and not args.dtype.startswith("complex"):
        raise SystemExit("irfftn requires --dtype complex64 or complex128")

    try:
        import jax
    except ImportError:
        raise SystemExit("JAX is not installed. Install the appropriate jax/jaxlib build first.")

    if args.dtype in ("float64", "complex128"):
        jax.config.update("jax_enable_x64", True)
    if args.platform:
        jax.config.update("jax_platform_name", args.platform)

    import jax.numpy as jnp

    devices = jax.devices(args.platform) if args.platform else jax.devices()
    if args.device < 0 or args.device >= len(devices):
        raise SystemExit(f"device index {args.device} out of range; available: 0..{len(devices)-1}")
    device = devices[args.device]

    print("JAX FFT benchmark")
    print(f"Python                 : {sys.version.split()[0]} ({py_platform.machine()})")
    print(f"JAX                    : {jax.__version__}")
    print(f"Backend                : {jax.default_backend()}")
    print(f"Device                 : {device}")
    print(f"Operation / dtype      : {args.op} / {args.dtype}")
    print(f"Warmup / timing        : {args.warmup} / {args.samples} x {args.iterations} calls")

    for shape in args.shape:
        result = benchmark_one(jax, jnp, args, device, shape)
        print_result(result, args.batch, args.op)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
