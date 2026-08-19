"""Buffered HDF5 recording for offline precursor boundary planes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from jaxwind.domain import AddressableField, Cell, VerticalFaceField, ZFace
from jaxwind.physics import ConcurrentPrecursorEnvironment

from .precursor_config import PrecursorPlaybackConfig, PrecursorRecordingConfig
from .precursor_config import SECTION_NAMES
from .runtime import JaxRuntime


SCHEMA = "jaxwind.precursor-sections.v2"
VELOCITY_COMPONENTS = ("u", "v", "w")
_TARGET_CHUNK_BYTES = 32 * 1024 * 1024


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "precursor recording requires h5py; install jaxwind with its "
            "declared runtime dependencies"
        ) from exc
    return h5py


@dataclass(frozen=True, slots=True)
class _LocalLayout:
    grid: Any
    partition_ids: tuple[int, ...]
    z_start: int
    z_stop: int

    @property
    def local_nz(self) -> int:
        return self.z_stop - self.z_start


def _state_payloads(state: Any) -> tuple[Any, Any | None]:
    if hasattr(state, "fields"):
        fields = state.fields
        return fields.velocity, fields.potential_temperature
    if hasattr(state, "velocity"):
        return state.velocity, None
    raise TypeError("precursor state must contain velocity or Boussinesq fields")


def _field_partition_ids(field: AddressableField) -> tuple[int, ...]:
    return tuple(region.coordinate.indices[0] for region in field.regions)


def _local_layout(velocity: Any, runtime: JaxRuntime) -> _LocalLayout:
    if not isinstance(velocity.x, AddressableField):
        raise TypeError("precursor recording requires addressable distributed fields")
    if not isinstance(velocity.y, AddressableField):
        raise TypeError("precursor y velocity must be an addressable field")
    if not isinstance(velocity.z, VerticalFaceField) or not isinstance(
        velocity.z.owned, AddressableField
    ):
        raise TypeError("precursor vertical velocity must own distributed upper faces")

    fields = (velocity.x, velocity.y, velocity.z.owned)
    partition_ids = _field_partition_ids(velocity.x)
    if partition_ids != runtime.addressable_partitions:
        raise ValueError(
            "precursor field partitions do not match the process runtime"
        )
    if any(_field_partition_ids(field) != partition_ids for field in fields[1:]):
        raise ValueError("precursor velocity components have different ownership")
    if velocity.x.location is not Cell or velocity.y.location is not Cell:
        raise ValueError("horizontal precursor velocities must be cell located")
    if velocity.z.owned.location is not ZFace:
        raise ValueError("vertical precursor velocity must be z-face located")

    regions = velocity.x.regions
    grid = regions[0].grid
    expected_start = regions[0].cell_z.start
    current = expected_start
    for region in regions:
        if region.grid != grid or region.cell_z.start != current:
            raise ValueError("precursor process ownership must be contiguous in z")
        current = region.cell_z.stop
    expected_shape = (
        len(regions),
        regions[0].cell_z.size,
        grid.ny,
        grid.nx,
    )
    if any(tuple(field.payload.shape) != expected_shape for field in fields):
        raise ValueError("precursor velocity payloads do not share the owned shape")
    return _LocalLayout(grid, partition_ids, expected_start, current)


def _validate_scalar(scalar: Any, velocity: Any, layout: _LocalLayout) -> None:
    if not isinstance(scalar, AddressableField):
        raise TypeError("precursor scalar must be an addressable distributed field")
    if scalar.location is not Cell:
        raise ValueError("precursor scalar must be cell located")
    if _field_partition_ids(scalar) != layout.partition_ids:
        raise ValueError("precursor scalar ownership differs from velocity ownership")
    if tuple(scalar.payload.shape) != tuple(velocity.x.payload.shape):
        raise ValueError("precursor scalar and velocity payload shapes differ")


def _chunk_samples(
    *,
    itemsize: int,
    values_per_section: int,
    buffer_samples: int,
) -> int:
    bytes_per_sample = max(1, itemsize * values_per_section)
    return max(
        1,
        min(buffer_samples, max(1, _TARGET_CHUNK_BYTES // bytes_per_sample)),
    )


class HDF5PrecursorRecorder:
    """Append local x-normal slabs to a rank shard without global gathers."""

    def __init__(
        self,
        path: str | Path,
        *,
        runtime: JaxRuntime,
        config: PrecursorRecordingConfig = PrecursorRecordingConfig(),
    ) -> None:
        target = Path(path)
        if target.suffix.lower() not in (".h5", ".hdf5"):
            raise ValueError("precursor recording path must end in .h5 or .hdf5")
        if (
            runtime.process_count > 1
            and target.exists()
            and not config.overwrite
        ):
            raise FileExistsError(target)
        self.path = target
        self.runtime = runtime
        self.config = config
        self.local_path = runtime.checkpoint_path(target)
        self._file = None
        self._layout: _LocalLayout | None = None
        self._has_scalar: bool | None = None
        self._scalar_quantity: str | None = None
        self._velocity_buffer: list[Any] = []
        self._scalar_buffer: list[Any] = []
        self._step_buffer: list[int] = []
        self._time_buffer: list[float] = []
        self._states_seen = 0
        self._samples_written = 0
        self._last_clock: tuple[int, float] | None = None
        self._closed = False

    def __enter__(self) -> "HDF5PrecursorRecorder":
        if self._closed:
            raise RuntimeError("precursor recorder is already closed")
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        self.close(complete=exc_type is None)

    def _initialize(self, state: Any) -> tuple[Any, Any | None]:
        velocity, scalar = _state_payloads(state)
        layout = _local_layout(velocity, self.runtime)
        if scalar is not None:
            _validate_scalar(scalar, velocity, layout)
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if self.config.overwrite else "x"
        h5py = _h5py()
        handle = h5py.File(self.local_path, mode)
        self._file = handle
        self._layout = layout
        self._has_scalar = scalar is not None
        self._scalar_quantity = (
            scalar.quantity.__name__ if scalar is not None else None
        )

        grid = layout.grid
        width = self.config.section_width
        inflow_start = self.config.inflow_start_index
        if inflow_start + width > grid.nx:
            raise ValueError("precursor inflow section exceeds the x domain")
        if width > grid.nx:
            raise ValueError("precursor section width exceeds the x domain")
        section_indices = np.stack(
            (
                np.arange(inflow_start, inflow_start + width, dtype=np.int64),
                np.arange(grid.nx - width, grid.nx, dtype=np.int64),
            )
        )
        attrs = handle.attrs
        attrs["schema"] = SCHEMA
        attrs["storage"] = (
            "single-file" if self.runtime.process_count == 1 else "rank-shard"
        )
        attrs["complete"] = False
        attrs["snapshot_semantics"] = "pre-step accepted state"
        attrs["value_units"] = "solver execution units"
        attrs["layout"] = "sample,section,component,z_local,y,x_section"
        attrs["process_index"] = self.runtime.process_index
        attrs["process_count"] = self.runtime.process_count
        attrs["global_devices"] = self.runtime.global_devices
        attrs["local_devices"] = self.runtime.local_devices
        attrs["backend"] = self.runtime.backend
        attrs["sample_every"] = self.config.sample_every
        attrs["buffer_samples"] = self.config.buffer_samples
        attrs["section_width"] = width
        attrs["inflow_start_index"] = inflow_start
        attrs["compression"] = self.config.compression or "none"
        attrs["nx"] = grid.nx
        attrs["ny"] = grid.ny
        attrs["nz"] = grid.nz
        attrs["lx"] = grid.lx
        attrs["ly"] = grid.ly
        attrs["lz"] = grid.lz
        attrs["z_start"] = layout.z_start
        attrs["z_stop"] = layout.z_stop
        attrs["scalar_quantity"] = self._scalar_quantity or "none"
        attrs["integrator_fingerprint"] = state.integrator_fingerprint
        attrs["sample_count"] = 0
        handle.create_dataset(
            "partition_ids", data=np.asarray(layout.partition_ids, dtype=np.int64)
        )

        strings = h5py.string_dtype(encoding="utf-8")
        sections = handle.create_group("sections")
        sections.create_dataset(
            "name", data=np.asarray(SECTION_NAMES, dtype=strings), dtype=strings
        )
        sections.create_dataset(
            "x_index", data=section_indices
        )
        sections.create_dataset(
            "x", data=(section_indices + 0.5) * grid.dx
        )
        coordinates = handle.create_group("coordinates")
        coordinates.create_dataset(
            "component",
            data=np.asarray(VELOCITY_COMPONENTS, dtype=strings),
            dtype=strings,
        )
        coordinates.create_dataset(
            "component_location",
            data=np.asarray(("cell", "cell", "upper-z-face"), dtype=strings),
            dtype=strings,
        )
        coordinates.create_dataset(
            "y", data=(np.arange(grid.ny, dtype=np.float64) + 0.5) * grid.dy
        )
        coordinates.create_dataset(
            "z_cell",
            data=(
                np.arange(layout.z_start, layout.z_stop, dtype=np.float64) + 0.5
            )
            * grid.dz,
        )
        coordinates.create_dataset(
            "z_face",
            data=(
                np.arange(layout.z_start, layout.z_stop, dtype=np.float64) + 1.0
            )
            * grid.dz,
        )

        handle.create_dataset(
            "step",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.config.buffer_samples,),
            dtype=np.int64,
        )
        handle.create_dataset(
            "time",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.config.buffer_samples,),
            dtype=np.float64,
        )
        velocity_dtype = np.dtype(velocity.x.payload.dtype)
        velocity_chunk = _chunk_samples(
            itemsize=velocity_dtype.itemsize,
            values_per_section=3 * layout.local_nz * grid.ny * width,
            buffer_samples=self.config.buffer_samples,
        )
        dataset_options = {
            "compression": self.config.compression,
            "shuffle": self.config.compression is not None,
        }
        handle.create_dataset(
            "velocity",
            shape=(0, len(SECTION_NAMES), 3, layout.local_nz, grid.ny, width),
            maxshape=(
                None,
                len(SECTION_NAMES),
                3,
                layout.local_nz,
                grid.ny,
                width,
            ),
            chunks=(velocity_chunk, 1, 3, layout.local_nz, grid.ny, width),
            dtype=velocity_dtype,
            **dataset_options,
        )
        if scalar is not None:
            scalar_dtype = np.dtype(scalar.payload.dtype)
            scalar_chunk = _chunk_samples(
                itemsize=scalar_dtype.itemsize,
                values_per_section=layout.local_nz * grid.ny * width,
                buffer_samples=self.config.buffer_samples,
            )
            scalar_dataset = handle.create_dataset(
                "scalar",
                shape=(0, len(SECTION_NAMES), layout.local_nz, grid.ny, width),
                maxshape=(
                    None,
                    len(SECTION_NAMES),
                    layout.local_nz,
                    grid.ny,
                    width,
                ),
                chunks=(scalar_chunk, 1, layout.local_nz, grid.ny, width),
                dtype=scalar_dtype,
                **dataset_options,
            )
            scalar_dataset.attrs["quantity"] = self._scalar_quantity
            scalar_dataset.attrs["location"] = "cell"
        handle.flush()
        return velocity, scalar

    def _validate_state(self, state: Any) -> tuple[Any, Any | None]:
        velocity, scalar = _state_payloads(state)
        assert self._layout is not None
        layout = _local_layout(velocity, self.runtime)
        if layout != self._layout:
            raise ValueError("precursor state layout changed during recording")
        has_scalar = scalar is not None
        if has_scalar != self._has_scalar:
            raise ValueError("precursor scalar presence changed during recording")
        if scalar is not None:
            _validate_scalar(scalar, velocity, layout)
            if scalar.quantity.__name__ != self._scalar_quantity:
                raise ValueError("precursor scalar quantity changed during recording")
        if state.integrator_fingerprint != self._file.attrs[
            "integrator_fingerprint"
        ]:
            raise ValueError("precursor integrator changed during recording")
        return velocity, scalar

    def _device_get(self, value: Any) -> np.ndarray:
        getter = getattr(self.runtime.jax, "device_get", None)
        return np.asarray(getter(value) if getter is not None else value)

    def _extract_sections(
        self,
        velocity: Any,
        scalar: Any | None,
    ) -> tuple[Any, Any | None]:
        assert self._layout is not None
        width = self.config.section_width
        inflow_start = self.config.inflow_start_index
        indices = self.runtime.jnp.stack(
            (
                self.runtime.jnp.arange(inflow_start, inflow_start + width),
                self.runtime.jnp.arange(
                    self._layout.grid.nx - width,
                    self._layout.grid.nx,
                ),
            )
        )
        components = (
            velocity.x.payload[..., indices],
            velocity.y.payload[..., indices],
            velocity.z.owned.payload[..., indices],
        )
        device_velocity = self.runtime.jnp.stack(components, axis=0)
        # (component, device, z, y, section, x) ->
        # (section, component, z_local, y, x)
        device_velocity = self.runtime.jnp.moveaxis(device_velocity, -2, 0).reshape(
            len(SECTION_NAMES),
            len(VELOCITY_COMPONENTS),
            self._layout.local_nz,
            self._layout.grid.ny,
            width,
        )
        if scalar is None:
            return device_velocity, None
        device_scalar = scalar.payload[..., indices]
        # (device, z, y, section, x) -> (section, z_local, y, x)
        device_scalar = self.runtime.jnp.moveaxis(device_scalar, -2, 0).reshape(
            len(SECTION_NAMES),
            self._layout.local_nz,
            self._layout.grid.ny,
            width,
        )
        return device_velocity, device_scalar

    def record(
        self,
        state: Any,
        *,
        step: int | None = None,
        time: float | None = None,
    ) -> bool:
        """Record this accepted pre-step state if it lies on the sample cadence."""

        if self._closed:
            raise RuntimeError("cannot record into a closed precursor recorder")
        if step is None or time is None:
            try:
                step = int(state.clock.step)
                time = float(state.clock.time)
            except AttributeError as exc:
                raise TypeError(
                    "precursor state must expose an accepted clock"
                ) from exc
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("precursor record step must be an integer")
        if not isinstance(time, (int, float)):
            raise TypeError("precursor record time must be a real scalar")
        time = float(time)
        if step < 0 or not math.isfinite(time):
            raise ValueError("precursor state clock is invalid")
        if self._last_clock is not None:
            last_step, last_time = self._last_clock
            if step <= last_step or time <= last_time:
                raise ValueError(
                    "precursor states must have strictly increasing clocks"
                )
        self._last_clock = (step, time)
        sample = self._states_seen % self.config.sample_every == 0
        self._states_seen += 1
        if not sample:
            return False

        if self._file is None:
            velocity, scalar = self._initialize(state)
        else:
            velocity, scalar = self._validate_state(state)
        device_velocity, device_scalar = self._extract_sections(velocity, scalar)
        self._velocity_buffer.append(device_velocity)
        if device_scalar is not None:
            self._scalar_buffer.append(device_scalar)
        self._step_buffer.append(step)
        self._time_buffer.append(time)
        if len(self._step_buffer) >= self.config.buffer_samples:
            self.flush()
        return True

    def flush(self) -> None:
        """Write the current host buffer as one extend-and-append operation."""

        if not self._step_buffer:
            return
        if self._file is None:
            raise RuntimeError("precursor recorder has no initialized HDF5 file")
        velocity = self.runtime.jnp.stack(self._velocity_buffer, axis=0)
        device_scalar = (
            self.runtime.jnp.stack(self._scalar_buffer, axis=0)
            if self._has_scalar
            else None
        )
        self._append_device_batch(
            velocity,
            device_scalar,
            np.asarray(self._step_buffer, dtype=np.int64),
            np.asarray(self._time_buffer, dtype=np.float64),
        )
        self._velocity_buffer.clear()
        self._scalar_buffer.clear()
        self._step_buffer.clear()
        self._time_buffer.clear()

    def _append_device_batch(
        self,
        velocity: Any,
        scalar: Any | None,
        steps: np.ndarray,
        times: np.ndarray,
    ) -> None:
        """Append one already-batched, device-resident hyperslab."""

        if self._file is None:
            raise RuntimeError("precursor recorder has no initialized HDF5 file")
        count = len(steps)
        if count <= 0 or times.shape != (count,):
            raise ValueError("precursor batch clocks have inconsistent shapes")
        start = self._samples_written
        stop = start + count
        for name in ("step", "time", "velocity"):
            dataset = self._file[name]
            dataset.resize(stop, axis=0)
        self._file["step"][start:stop] = steps
        self._file["time"][start:stop] = times
        self._file["velocity"][start:stop] = np.ascontiguousarray(
            self._device_get(velocity)
        )
        if self._has_scalar:
            if scalar is None:
                raise ValueError("precursor scalar batch is missing")
            scalar_dataset = self._file["scalar"]
            scalar_dataset.resize(stop, axis=0)
            scalar_dataset[start:stop] = np.ascontiguousarray(
                self._device_get(scalar)
            )
        elif scalar is not None:
            raise ValueError("unexpected precursor scalar batch")
        self._samples_written = stop
        self._file.attrs["sample_count"] = stop
        self._file.flush()

    def record_batch(
        self,
        state: Any,
        velocity: Any,
        scalar: Any | None,
        *,
        steps: np.ndarray,
        times: np.ndarray,
    ) -> None:
        """Append a compiled block of consecutive pre-step section samples."""

        if self._closed:
            raise RuntimeError("cannot record into a closed precursor recorder")
        steps = np.asarray(steps, dtype=np.int64)
        times = np.asarray(times, dtype=np.float64)
        if steps.ndim != 1 or times.shape != steps.shape or len(steps) == 0:
            raise ValueError("precursor batch clocks must be nonempty vectors")
        if not np.all(np.diff(steps) == self.config.sample_every) or not np.all(
            np.diff(times) > 0.0
        ):
            raise ValueError(
                "precursor batch clocks must match the recording cadence"
            )
        if self._last_clock is not None:
            if steps[0] <= self._last_clock[0] or times[0] <= self._last_clock[1]:
                raise ValueError("precursor batch clocks are not increasing")
        if self._file is None:
            self._initialize(state)
        else:
            self._validate_state(state)
        assert self._layout is not None
        count = len(steps)
        velocity_shape = (
            count,
            len(SECTION_NAMES),
            len(VELOCITY_COMPONENTS),
            self._layout.local_nz,
            self._layout.grid.ny,
            self.config.section_width,
        )
        if tuple(velocity.shape) != velocity_shape:
            raise ValueError("compiled precursor velocity batch shape is invalid")
        scalar_shape = (
            count,
            len(SECTION_NAMES),
            self._layout.local_nz,
            self._layout.grid.ny,
            self.config.section_width,
        )
        if scalar is not None and tuple(scalar.shape) != scalar_shape:
            raise ValueError("compiled precursor scalar batch shape is invalid")
        self.flush()
        self._append_device_batch(velocity, scalar, steps, times)
        self._states_seen += count * self.config.sample_every
        self._last_clock = (int(steps[-1]), float(times[-1]))

    def close(self, *, complete: bool = True) -> None:
        """Flush and close this shard, marking whether the run completed."""

        if self._closed:
            return
        self._closed = True
        if self._file is None:
            if complete:
                raise RuntimeError(
                    "cannot complete a precursor recording with no samples"
                )
            return
        try:
            self.flush()
            self._file.attrs["complete"] = bool(complete)
            self._file.attrs["sample_count"] = self._samples_written
            self._file.flush()
        finally:
            self._file.close()
            self._file = None


def _attribute_dict(handle) -> dict[str, Any]:
    return {name: handle.attrs[name] for name in handle.attrs}


def _validate_shards(path: Path, runtime: JaxRuntime):
    h5py = _h5py()
    handles = []
    try:
        for process_index in range(runtime.process_count):
            shard = path.with_name(
                f"{path.stem}.process-{process_index:05d}{path.suffix}"
            )
            handles.append(h5py.File(shard, "r"))
        first = handles[0]
        attrs = _attribute_dict(first)
        if attrs.get("schema") != SCHEMA:
            raise ValueError("unsupported precursor shard schema")
        if attrs.get("storage") != "rank-shard":
            raise ValueError("multi-process precursor input is not a rank shard")
        if not bool(attrs.get("complete", False)):
            raise ValueError("precursor shard is incomplete")
        if int(attrs.get("process_count", -1)) != runtime.process_count or int(
            attrs.get("global_devices", -1)
        ) != runtime.global_devices:
            raise ValueError("precursor shards do not match the current runtime")
        expected_z = 0
        first_step = np.asarray(first["step"])
        first_time = np.asarray(first["time"])
        comparable = (
            "schema",
            "process_count",
            "global_devices",
            "sample_every",
            "section_width",
            "inflow_start_index",
            "nx",
            "ny",
            "nz",
            "lx",
            "ly",
            "lz",
            "scalar_quantity",
            "integrator_fingerprint",
            "sample_count",
        )
        for process_index, handle in enumerate(handles):
            current = _attribute_dict(handle)
            if any(current.get(name) != attrs.get(name) for name in comparable):
                raise ValueError("precursor shard metadata is inconsistent")
            if int(current.get("process_index", -1)) != process_index:
                raise ValueError("precursor shard process index is inconsistent")
            if not bool(current.get("complete", False)):
                raise ValueError("precursor shard is incomplete")
            if int(current["z_start"]) != expected_z:
                raise ValueError("precursor shards do not cover contiguous z slabs")
            expected_z = int(current["z_stop"])
            if not np.array_equal(np.asarray(handle["step"]), first_step):
                raise ValueError("precursor shard steps are not synchronized")
            if not np.array_equal(np.asarray(handle["time"]), first_time):
                raise ValueError("precursor shard times are not synchronized")
        if expected_z != int(attrs["nz"]):
            raise ValueError("precursor shards do not cover the global z extent")
        return handles, attrs, first_step, first_time
    except Exception:
        for handle in handles:
            handle.close()
        raise


def _create_catalog(path: Path, runtime: JaxRuntime, *, overwrite: bool) -> None:
    h5py = _h5py()
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    handles, attrs, steps, times = _validate_shards(path, runtime)
    temporary = path.with_name(f".{path.name}.catalog-tmp-{os.getpid()}")
    try:
        first = handles[0]
        sample_count = len(steps)
        section_count = len(SECTION_NAMES)
        nz = int(attrs["nz"])
        ny = int(attrs["ny"])
        width = int(attrs["section_width"])
        velocity_layout = h5py.VirtualLayout(
            shape=(sample_count, section_count, 3, nz, ny, width),
            dtype=first["velocity"].dtype,
        )
        scalar_layout = None
        if "scalar" in first:
            scalar_layout = h5py.VirtualLayout(
                shape=(sample_count, section_count, nz, ny, width),
                dtype=first["scalar"].dtype,
            )
        for handle in handles:
            z_start = int(handle.attrs["z_start"])
            z_stop = int(handle.attrs["z_stop"])
            filename = Path(handle.filename).name
            velocity_source = h5py.VirtualSource(
                filename, "velocity", shape=handle["velocity"].shape
            )
            velocity_layout[:, :, :, z_start:z_stop, :, :] = velocity_source
            if scalar_layout is not None:
                if "scalar" not in handle:
                    raise ValueError("precursor scalar datasets are inconsistent")
                scalar_source = h5py.VirtualSource(
                    filename, "scalar", shape=handle["scalar"].shape
                )
                scalar_layout[:, :, z_start:z_stop, :, :] = scalar_source

        with h5py.File(temporary, "w", libver="latest") as catalog:
            for name, value in attrs.items():
                if name not in ("process_index", "local_devices", "z_start", "z_stop"):
                    catalog.attrs[name] = value
            catalog.attrs["storage"] = "virtual-dataset-catalog"
            catalog.attrs["complete"] = True
            catalog.attrs["layout"] = (
                "sample,section,component,z,y,x_section"
            )
            catalog.create_dataset("step", data=steps)
            catalog.create_dataset("time", data=times)
            first.copy("sections", catalog)
            coordinates = catalog.create_group("coordinates")
            first.copy("coordinates/component", coordinates, name="component")
            first.copy(
                "coordinates/component_location",
                coordinates,
                name="component_location",
            )
            first.copy("coordinates/y", coordinates, name="y")
            grid_dz = float(attrs["lz"]) / nz
            coordinates.create_dataset(
                "z_cell", data=(np.arange(nz, dtype=np.float64) + 0.5) * grid_dz
            )
            coordinates.create_dataset(
                "z_face", data=(np.arange(nz, dtype=np.float64) + 1.0) * grid_dz
            )
            catalog.create_virtual_dataset("velocity", velocity_layout)
            if scalar_layout is not None:
                scalar = catalog.create_virtual_dataset("scalar", scalar_layout)
                scalar.attrs["quantity"] = attrs["scalar_quantity"]
                scalar.attrs["location"] = "cell"
            shards = catalog.create_group("shards")
            for process_index, handle in enumerate(handles):
                shards[f"process_{process_index:05d}"] = h5py.ExternalLink(
                    Path(handle.filename).name, "/"
                )
            catalog.flush()
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        for handle in handles:
            handle.close()


def _synchronization_name(path: Path, phase: str) -> str:
    identity = hashlib.sha256(str(path.absolute()).encode()).hexdigest()[:16]
    return f"jaxwind-precursor-{phase}-{identity}"


def finalize_precursor_recording(
    path: str | Path,
    *,
    runtime: JaxRuntime,
    overwrite: bool = False,
) -> Path:
    """Collectively publish a global HDF5 catalog after all shards are closed."""

    target = Path(path)
    runtime.synchronize(_synchronization_name(target, "shards-complete"))
    if runtime.process_count == 1:
        h5py = _h5py()
        with h5py.File(target, "r") as handle:
            if handle.attrs.get("schema") != SCHEMA or not bool(
                handle.attrs.get("complete", False)
            ):
                raise ValueError("single-process precursor recording is incomplete")
        return target
    if runtime.is_primary:
        _create_catalog(target, runtime, overwrite=overwrite)
    runtime.synchronize(_synchronization_name(target, "catalog-complete"))
    return target


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class HDF5PrecursorPlayback:
    """Replay buffered, rank-local planes as same-layout fringe targets."""

    def __init__(
        self,
        path: str | Path,
        *,
        runtime: JaxRuntime,
        state: Any,
        config: PrecursorPlaybackConfig = PrecursorPlaybackConfig(),
    ) -> None:
        self.path = Path(path)
        self.runtime = runtime
        self.config = config
        self.local_path = runtime.checkpoint_path(self.path)
        velocity, _scalar = _state_payloads(state)
        self._layout = _local_layout(velocity, runtime)
        self._fingerprint = state.integrator_fingerprint
        h5py = _h5py()
        self._file = h5py.File(self.local_path, "r")
        self._closed = False
        self._cache_start = 0
        self._cache_stop = 0
        self._cache = None
        try:
            self._validate_file()
        except Exception:
            self._file.close()
            self._closed = True
            raise

    def __enter__(self) -> "HDF5PrecursorPlayback":
        if self._closed:
            raise RuntimeError("precursor playback is already closed")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _validate_file(self) -> None:
        attrs = self._file.attrs
        expected_storage = (
            "single-file" if self.runtime.process_count == 1 else "rank-shard"
        )
        if attrs.get("schema") != SCHEMA:
            raise ValueError("unsupported precursor playback schema")
        if attrs.get("storage") != expected_storage:
            raise ValueError("precursor playback storage does not match the runtime")
        if not bool(attrs.get("complete", False)):
            raise ValueError("precursor playback file is incomplete")
        runtime_metadata = (
            ("process_count", self.runtime.process_count),
            ("global_devices", self.runtime.global_devices),
            ("local_devices", self.runtime.local_devices),
            ("process_index", self.runtime.process_index),
        )
        if any(int(attrs.get(name, -1)) != value for name, value in runtime_metadata):
            raise ValueError("precursor playback topology does not match the runtime")
        grid = self._layout.grid
        grid_metadata = (
            ("nx", grid.nx),
            ("ny", grid.ny),
            ("nz", grid.nz),
            ("lx", grid.lx),
            ("ly", grid.ly),
            ("lz", grid.lz),
        )
        if any(attrs.get(name) != value for name, value in grid_metadata):
            raise ValueError("precursor playback grid does not match the main state")
        if (
            int(attrs.get("z_start", -1)) != self._layout.z_start
            or int(attrs.get("z_stop", -1)) != self._layout.z_stop
        ):
            raise ValueError("precursor playback z ownership does not match main state")
        if attrs.get("integrator_fingerprint") != self._fingerprint:
            raise ValueError("precursor playback integrator does not match main state")
        self._sample_every = int(attrs.get("sample_every", -1))
        if self._sample_every <= 0:
            raise ValueError("precursor sample interval is invalid")
        self._section_width = int(attrs.get("section_width", -1))
        if self._section_width <= 0:
            raise ValueError("precursor section width is invalid")

        names = tuple(_text(value) for value in self._file["sections/name"][:])
        if names != SECTION_NAMES:
            raise ValueError("precursor playback sections are not recognized")
        self._section_index = names.index(self.config.section)
        self._steps = np.asarray(self._file["step"], dtype=np.int64)
        self._times = np.asarray(self._file["time"], dtype=np.float64)
        if self._steps.size == 0:
            raise ValueError("precursor playback contains no samples")
        if not np.all(np.diff(self._steps) == self._sample_every):
            raise ValueError("precursor playback steps do not match its cadence")
        if not np.all(np.diff(self._times) > 0.0):
            raise ValueError("precursor playback times must be strictly increasing")
        expected_shape = (
            len(self._steps),
            len(SECTION_NAMES),
            len(VELOCITY_COMPONENTS),
            self._layout.local_nz,
            grid.ny,
            self._section_width,
        )
        if self._file["velocity"].shape != expected_shape:
            raise ValueError("precursor playback velocity shape is inconsistent")
        self._first_step = int(self._steps[0])

    @property
    def sample_count(self) -> int:
        return len(self._steps)

    @property
    def first_step(self) -> int:
        return self._first_step

    @property
    def final_step(self) -> int:
        return int(self._steps[-1])

    @property
    def covered_steps(self) -> int:
        """Number of main steps covered when each sample is held by cadence."""

        return self.sample_count * self._sample_every

    @property
    def progress_interval_steps(self) -> int:
        """Physical steps represented by one playback cache buffer."""

        return self.config.buffer_samples * self._sample_every

    def _planes(self, index: int) -> Any:
        return self._plane_batch(index, 1)[0]

    def _plane_batch(self, index: int, count: int) -> Any:
        if count <= 0 or count > self.config.buffer_samples:
            raise ValueError("precursor playback batch exceeds its read buffer")
        if not (
            self._cache_start <= index
            and index + count <= self._cache_stop
        ):
            start = (index // self.config.buffer_samples) * self.config.buffer_samples
            stop = min(start + self.config.buffer_samples, self.sample_count)
            host = np.asarray(
                self._file["velocity"][start:stop, self._section_index]
            )
            self._cache = self.runtime.jnp.asarray(host)
            self._cache_start = start
            self._cache_stop = stop
        relative = index - self._cache_start
        return self._cache[relative : relative + count]

    def _validate_request(
        self,
        state: Any,
        *,
        step: int,
        time: float,
        count: int = 1,
    ) -> int:
        if self._closed:
            raise RuntimeError("cannot read from closed precursor playback")
        if state.integrator_fingerprint != self._fingerprint:
            raise ValueError("main integrator changed during precursor playback")
        velocity, _scalar = _state_payloads(state)
        if _local_layout(velocity, self.runtime) != self._layout:
            raise ValueError("main layout changed during precursor playback")
        relative_step = step - self._first_step
        index = relative_step // self._sample_every
        if (
            count <= 0
            or relative_step < 0
            or index + count > self.sample_count
            or self._steps[index] != step - relative_step % self._sample_every
        ):
            raise IndexError(f"no precursor sample exists for main step {step}")
        recorded_time = float(self._times[index])
        current_time = float(time)
        if relative_step % self._sample_every:
            sample_dt = (
                float(self._times[1] - self._times[0]) / self._sample_every
                if self.sample_count > 1
                else 0.0
            )
            recorded_time += (relative_step % self._sample_every) * sample_dt
        tolerance = 32.0 * np.finfo(np.float64).eps * max(
            1.0,
            abs(recorded_time),
            abs(current_time),
            abs(relative_step),
        )
        if not math.isclose(
            recorded_time,
            current_time,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("precursor and main clocks are not synchronized")
        return index

    def plane_batch(
        self,
        state: Any,
        *,
        step: int,
        time: float,
        count: int,
    ) -> Any:
        """Read one bounded, consecutive plane block for compiled playback."""

        index = self._validate_request(
            state,
            step=step,
            time=time,
            count=count,
        )
        return self._plane_batch(index, count)

    def environment_from_plane(
        self,
        state: Any,
        plane: Any,
    ) -> ConcurrentPrecursorEnvironment:
        """Place one local ``(component,z,y,x_section)`` slab at inlet x."""

        velocity, _scalar = _state_payloads(state)
        slab = self.runtime.jnp.roll(
            plane,
            shift=self.config.spanwise_shift_cells,
            axis=-2,
        )
        local_shape = tuple(int(extent) for extent in velocity.x.payload.shape)
        slabs = slab.reshape(
            len(VELOCITY_COMPONENTS),
            local_shape[0],
            local_shape[1],
            local_shape[2],
            self._section_width,
        )
        if self._section_width == 1:
            target = self.runtime.jnp.broadcast_to(
                slabs,
                (len(VELOCITY_COMPONENTS),) + local_shape,
            )
        else:
            target = self.runtime.jnp.concatenate(
                (
                    slabs,
                    self.runtime.jnp.zeros(
                        (len(VELOCITY_COMPONENTS),)
                        + local_shape[:-1]
                        + (local_shape[-1] - self._section_width,),
                        dtype=slabs.dtype,
                    ),
                ),
                axis=-1,
            )
        target_velocity = replace(
            velocity,
            x=replace(velocity.x, payload=target[0]),
            y=replace(velocity.y, payload=target[1]),
            z=VerticalFaceField(
                replace(velocity.z.owned, payload=target[2]),
                velocity.z.lower_boundary,
            ),
        )
        return ConcurrentPrecursorEnvironment(target_velocity)

    def environment(
        self,
        state: Any,
        *,
        step: int | None = None,
        time: float | None = None,
    ) -> ConcurrentPrecursorEnvironment:
        """Build the fringe environment matching this main pre-step clock."""

        if step is None or time is None:
            step = int(state.clock.step)
            time = float(state.clock.time)
        index = self._validate_request(state, step=step, time=time)
        return self.environment_from_plane(state, self._planes(index))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._file.close()
        self._cache = None


__all__ = [
    "HDF5PrecursorPlayback",
    "HDF5PrecursorRecorder",
    "PrecursorPlaybackConfig",
    "PrecursorRecordingConfig",
    "SCHEMA",
    "finalize_precursor_recording",
]
