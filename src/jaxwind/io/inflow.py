"""Recorded inflow analysis adapters."""

from pathlib import Path

from .recording import InflowReader


def load_inflow_block(directory: Path, start: int, stop: int, jnp=None):
    return InflowReader(directory).read(start, stop)


def write_synthetic_inflow(directory, grid, inflow, *, samples, dt, chunk_size=64):
    """Record a time sampler in the existing fixed-cadence inflow format.

    The sampler must return periodic-y InflowPlane values. Samples start at
    t=0. The destination must be empty; generation is bounded by chunk_size.
    """
    import jax
    import numpy as np

    from jaxwind.open_boundary import validate_inflow_plane

    from .recording import write_chunk, write_manifest

    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("inflow output directory must be empty")
    validate_inflow_plane(inflow(0.0), grid)
    sample = jax.jit(inflow)
    directory.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, samples, chunk_size):
        times = np.arange(start, min(start + chunk_size, samples), dtype=float) * dt
        planes = [sample(time) for time in times]
        outputs = {
            name: np.stack([np.asarray(getattr(plane, name)) for plane in planes])
            for name in planes[0]._fields
        }
        outputs.update(time_seconds=times, dt_seconds=np.full(len(times), dt))
        chunks.append(write_chunk(directory, len(chunks), outputs))
    write_manifest(directory, grid, chunks)
