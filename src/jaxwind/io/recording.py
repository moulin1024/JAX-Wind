"""Chunked, versioned inflow artifacts with explicit geometry and times."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SCHEMA = "jaxwind.inflow.v1"


def write_chunk(directory, index, outputs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"chunk_{index:08d}.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **outputs)
    temporary.replace(path)
    dt = np.asarray(outputs["dt_seconds"], dtype=np.float64)
    duration = float(dt.sum())
    mean_u = (np.mean(outputs["x_velocity"], axis=2, dtype=np.float64)
              * dt[:, None]).sum(axis=0) / duration
    return {"file": path.name, "samples": len(dt), "duration_seconds": duration,
            "mean_x_velocity_profile_m_s": mean_u.tolist()}


def write_manifest(directory, grid, chunks):
    value = {"schema": SCHEMA, "chunks": chunks, "samples": sum(chunk["samples"] for chunk in chunks),
             "units": "SI", "y_faces_m": np.asarray(grid.y_faces).tolist(), "z_faces_m": np.asarray(grid.z_faces).tolist()}
    if chunks and all("duration_seconds" in chunk for chunk in chunks):
        duration = sum(chunk["duration_seconds"] for chunk in chunks)
        value["duration_seconds"] = duration
        value["mean_x_velocity_profile_m_s"] = (sum(
            np.asarray(chunk["mean_x_velocity_profile_m_s"]) * chunk["duration_seconds"]
            for chunk in chunks) / duration).tolist()
    path = Path(directory) / "metadata.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class InflowReader:
    def __init__(self, directory, grid=None, *, samples=None, dt=None):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if self.metadata.get("schema") != SCHEMA:
            raise ValueError("unsupported inflow schema")
        chunks = self.metadata["chunks"]
        if not chunks or any(type(chunk["samples"]) is not int or chunk["samples"] <= 0 for chunk in chunks):
            raise ValueError("inflow recording must contain positive chunk lengths")
        if sum(chunk["samples"] for chunk in chunks) != self.metadata["samples"]:
            raise ValueError("inflow sample count does not match its chunks")
        if len({chunk["file"] for chunk in chunks}) != len(chunks):
            raise ValueError("duplicate inflow chunk")
        if samples is not None and samples > self.metadata["samples"]:
            raise ValueError("main duration exceeds recorded inflow coverage")
        if grid is None:
            from types import SimpleNamespace
            grid = SimpleNamespace(ny=len(self.metadata["y_faces_m"])-1, nz=len(self.metadata["z_faces_m"])-1,
                                   y_faces=self.metadata["y_faces_m"], z_faces=self.metadata["z_faces_m"])
        for axis in ("y", "z"):
            if not np.array_equal(self.metadata[axis + "_faces_m"], np.asarray(getattr(grid, axis + "_faces"))):
                raise ValueError("inflow mesh does not match the receiving boundary")
        next_time = None
        for chunk in chunks:
            if Path(chunk["file"]).name != chunk["file"]:
                raise ValueError("inflow chunk must be a local filename")
            with np.load(self.directory / chunk["file"], allow_pickle=False) as data:
                times, widths = data["time_seconds"], data["dt_seconds"]
                if times.shape != (chunk["samples"],) or widths.shape != times.shape:
                    raise ValueError("invalid inflow time array shape")
                if not np.isfinite(times).all() or not np.isfinite(widths).all() or not (widths > 0).all():
                    raise ValueError("inflow times and timesteps must be finite with positive timesteps")
                tolerance = 8 * np.finfo(times.dtype).eps * max(1., float(np.max(np.abs(times))))
                if not np.allclose(times[1:], times[:-1] + widths[:-1], rtol=0., atol=tolerance) or (next_time is not None and abs(float(times[0]) - next_time) > tolerance):
                    raise ValueError("inflow timestamps are not contiguous")
                next_time = float(times[-1] + widths[-1])
                if dt is not None and not np.allclose(data["dt_seconds"], dt, rtol=0., atol=1.e-9):
                    raise ValueError("fixed-step open flow requires matching fixed-cadence inflow")
                expected = {"x_velocity": (grid.nz, grid.ny), "y_velocity": (grid.nz, grid.ny),
                            "z_velocity": (grid.nz + 1, grid.ny), "scalar": (grid.nz, grid.ny)}
                for name, shape in expected.items():
                    if data[name].shape != (chunk["samples"], *shape):
                        raise ValueError(f"invalid recorded {name} shape")

    def read(self, start, stop):
        from jaxwind import InflowPlane
        import jax.numpy as jnp
        if not 0 <= start < stop <= self.metadata["samples"]:
            raise ValueError("inflow slice outside recorded coverage")
        names = ("x_velocity", "y_velocity", "z_velocity", "scalar")
        parts = {name: [] for name in names}
        offset = 0
        for chunk in self.metadata["chunks"]:
            end = offset + chunk["samples"]
            if start < end and stop > offset:
                with np.load(self.directory / chunk["file"], allow_pickle=False) as data:
                    for name in names:
                        parts[name].append(np.array(data[name][max(0, start-offset):min(chunk["samples"], stop-offset)]))
            offset = end
        return InflowPlane(*(jnp.asarray(np.concatenate(parts[name])) for name in names))
