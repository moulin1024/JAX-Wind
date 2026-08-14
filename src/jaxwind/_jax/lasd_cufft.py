"""Optional persistent-cuFFT implementation of the LASD test filter."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from threading import Lock


_TARGET = "jaxwind_lasd_filter_two_scales_f32"
_LOCK = Lock()
_LIBRARY = None
_REGISTERED = False


def _library_path() -> Path:
    configured = os.environ.get("JAXWIND_LASD_CUFFT_LIBRARY")
    if configured:
        return Path(configured).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    return repository / "build" / "native" / "lasd_cufft" / "libjaxwind_lasd_cufft.so"


def _register() -> None:
    global _LIBRARY, _REGISTERED
    if _REGISTERED:
        return
    with _LOCK:
        if _REGISTERED:
            return
        path = _library_path()
        if not path.is_file():
            raise RuntimeError(
                f"LASD cuFFT library not found at {path}; "
                "run tools/build_lasd_cufft.sh first or set "
                "JAXWIND_LASD_CUFFT_LIBRARY"
            )
        import jax

        library = ctypes.CDLL(str(path))
        handler = library.JaxwindLasdFilterTwoScalesF32
        jax.ffi.register_ffi_target(
            _TARGET,
            jax.ffi.pycapsule(handler),
            platform="CUDA",
        )
        _LIBRARY = library
        _REGISTERED = True


def filter_two_scales(values, first_filter_width, second_filter_width):
    """Filter a component-major float32 field at two horizontal scales."""
    import jax
    import jax.numpy as jnp

    if values.ndim != 4:
        raise ValueError("LASD cuFFT input must have shape (component, z, y, x)")
    if values.dtype != jnp.float32:
        raise TypeError("LASD cuFFT currently supports float32 fields only")
    _register()
    output = jax.ShapeDtypeStruct(
        (2 * values.shape[0], *values.shape[1:]),
        values.dtype,
    )
    call = jax.ffi.ffi_call(
        _TARGET,
        output,
        vmap_method="sequential",
    )
    return call(
        values,
        jnp.asarray(first_filter_width, dtype=values.dtype),
        jnp.asarray(second_filter_width, dtype=values.dtype),
    )


__all__ = ["filter_two_scales"]
