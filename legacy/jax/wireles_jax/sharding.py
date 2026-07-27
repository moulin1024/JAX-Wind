from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _shard_map(
    f: Callable[..., Any],
    *,
    mesh: Mesh,
    in_specs: Any,
    out_specs: Any,
    axis_name: str,
    additional_axis_names: tuple[str, ...] = (),
) -> Callable[..., Any]:
    try:
        shard_map = jax.shard_map
    except AttributeError:  # pragma: no cover - old JAX compatibility
        from jax.experimental.shard_map import shard_map

    kwargs = {
        "mesh": mesh,
        "in_specs": in_specs,
        "out_specs": out_specs,
        "axis_names": {axis_name, *additional_axis_names},
    }
    signature = inspect.signature(shard_map)
    if "check_vma" in signature.parameters:
        kwargs["check_vma"] = False
    elif "check_rep" in signature.parameters:
        kwargs["check_rep"] = False
    return shard_map(f, **kwargs)


def make_single_node_mesh(num_devices: int | None = None, axis_name: str = "z") -> Mesh:
    devices = jax.local_devices()
    if num_devices is None:
        num_devices = len(devices)
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}")
    if num_devices > len(devices):
        raise ValueError(f"Requested {num_devices} device(s), but JAX only sees {len(devices)} local device(s).")
    return Mesh(np.asarray(devices[:num_devices]), (axis_name,))


def make_distributed_mesh(num_devices: int | None = None, axis_name: str = "z") -> Mesh:
    """Build one globally consistent mesh across all JAX processes.

    ``jax.devices()`` returns the global device set after
    ``jax.distributed.initialize``.  Every process must construct the same
    mesh, while only the addressable shards are materialized on that process.
    """

    devices = jax.devices()
    if num_devices is None:
        num_devices = len(devices)
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}")
    if num_devices > len(devices):
        raise ValueError(
            f"Requested {num_devices} global device(s), but JAX only sees {len(devices)} device(s)."
        )
    selected = devices[:num_devices]
    addressable = {
        (device.process_index, device.id) for device in jax.local_devices()
    }
    if not any((device.process_index, device.id) in addressable for device in selected):
        raise ValueError(
            "The selected global mesh contains no device addressable by this process. "
            "Use all global devices or choose a mesh that includes every process."
        )
    return Mesh(np.asarray(selected), (axis_name,))


def make_adjoint_distributed_mesh(
    num_devices: int | None = None,
    *,
    adjoint_axis_name: str = "adjoint",
    z_axis_name: str = "z",
) -> Mesh:
    """Build a `(precursor/turbine, z-slab)` mesh on one JAX cluster."""
    devices = jax.devices()
    if num_devices is None:
        num_devices = len(devices)
    if num_devices < 4 or num_devices % 2:
        raise ValueError("Adjoint mesh requires an even device count >= 4")
    if num_devices > len(devices):
        raise ValueError(
            f"Requested {num_devices} devices, but JAX sees {len(devices)}"
        )
    selected = np.asarray(devices[:num_devices]).reshape(2, num_devices // 2)
    return Mesh(selected, (adjoint_axis_name, z_axis_name))


def mesh_size(mesh: Mesh, axis_name: str = "z") -> int:
    return int(mesh.shape[axis_name])


def z_slab_spec(axis_name: str = "z") -> P:
    return P(None, None, axis_name)


def adjoint_z_slab_spec(
    adjoint_axis_name: str = "adjoint", z_axis_name: str = "z"
) -> P:
    return P(adjoint_axis_name, None, None, z_axis_name)


def y_slab_spec(axis_name: str = "z") -> P:
    return P(None, axis_name, None)


def adjoint_y_slab_spec(
    adjoint_axis_name: str = "adjoint", z_axis_name: str = "z"
) -> P:
    return P(adjoint_axis_name, None, z_axis_name, None)


def z_slab_sharding(mesh: Mesh, axis_name: str = "z") -> NamedSharding:
    return NamedSharding(mesh, z_slab_spec(axis_name))


def make_array_from_local_callback(
    shape: tuple[int, ...],
    sharding: NamedSharding,
    callback: Callable[[tuple[slice, ...]], Any],
    *,
    dtype: Any,
) -> jax.Array:
    """Create a distributed array without constructing a global host value.

    The callback is invoked only for indices addressable by the current
    process.  It must return exactly that process-local shard.
    """

    np_dtype = np.dtype(dtype)

    def local_callback(index: tuple[slice, ...]) -> np.ndarray:
        normalized = tuple(
            slice(
                0 if part.start is None else part.start,
                extent if part.stop is None else part.stop,
                1 if part.step is None else part.step,
            )
            for part, extent in zip(index, shape, strict=True)
        )
        value = np.asarray(callback(normalized), dtype=np_dtype)
        expected = tuple(
            len(range(part.start, part.stop, part.step)) for part in normalized
        )
        if value.shape != expected:
            raise ValueError(
                f"Local callback returned shape {value.shape} for index {normalized}; expected {expected}."
            )
        return value

    return jax.make_array_from_callback(shape, sharding, local_callback)


def y_slab_sharding(mesh: Mesh, axis_name: str = "z") -> NamedSharding:
    return NamedSharding(mesh, y_slab_spec(axis_name))


def put_z_slab(array: jax.Array, mesh: Mesh, axis_name: str = "z") -> jax.Array:
    validate_z_slab_shape(array.shape, mesh_size(mesh, axis_name))
    return jax.device_put(array, z_slab_sharding(mesh, axis_name))


def validate_z_slab_shape(shape: tuple[int, ...], num_devices: int) -> None:
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D spectral array, got shape {shape}")
    _, ny, nz = shape
    if ny % num_devices != 0:
        raise ValueError(f"y dimension {ny} must be divisible by num_devices={num_devices}")
    if nz % num_devices != 0:
        raise ValueError(f"z dimension {nz} must be divisible by num_devices={num_devices}")


def rfft2_fortran_layout(q: jax.Array) -> jax.Array:
    """Horizontal real FFT with Fortran-compatible `(nx/2+1, ny, nz)` layout."""

    return jnp.fft.rfftn(q, axes=(-2, -3))


def irfft2_fortran_layout(q_hat: jax.Array, nx: int, ny: int) -> jax.Array:
    """Inverse horizontal FFT for `rfft2_fortran_layout`."""

    return jnp.fft.irfftn(q_hat, s=(ny, nx), axes=(-2, -3)).real


def make_pressure_z_slab_to_y_slab(
    mesh: Mesh,
    axis_name: str = "z",
    adjoint_axis_name: str | None = None,
) -> Callable[[jax.Array], jax.Array]:
    ndev = mesh_size(mesh, axis_name)

    def local_z_to_y(h_local: jax.Array) -> jax.Array:
        nxh, ny, nz_local = h_local.shape
        ny_local = ny // ndev
        split_y = h_local.reshape(nxh, ndev, ny_local, nz_local)
        h_y = lax.all_to_all(split_y, axis_name, split_axis=1, concat_axis=3, tiled=True)
        return h_y.reshape(nxh, ny_local, nz_local * ndev)

    mapped_local = local_z_to_y
    in_spec = z_slab_spec(axis_name)
    out_spec = y_slab_spec(axis_name)
    additional = ()
    if adjoint_axis_name is not None:
        mapped_local = jax.vmap(local_z_to_y)
        in_spec = adjoint_z_slab_spec(adjoint_axis_name, axis_name)
        out_spec = adjoint_y_slab_spec(adjoint_axis_name, axis_name)
        additional = (adjoint_axis_name,)
    return _shard_map(
        mapped_local,
        mesh=mesh,
        in_specs=in_spec,
        out_specs=out_spec,
        axis_name=axis_name,
        additional_axis_names=additional,
    )


def pressure_z_slab_to_y_slab(h_z: jax.Array, mesh: Mesh, axis_name: str = "z") -> jax.Array:
    """Redistribute `(nxh, ny, nz_local)` z slabs to `(nxh, ny_local, nz)` y slabs."""

    validate_z_slab_shape(h_z.shape, mesh_size(mesh, axis_name))
    mapped = make_pressure_z_slab_to_y_slab(mesh, axis_name)
    return mapped(h_z)


def make_pressure_y_slab_to_z_slab(
    mesh: Mesh,
    axis_name: str = "z",
    adjoint_axis_name: str | None = None,
) -> Callable[[jax.Array], jax.Array]:
    ndev = mesh_size(mesh, axis_name)

    def local_y_to_z(h_local: jax.Array) -> jax.Array:
        nxh, ny_local, nz = h_local.shape
        nz_local = nz // ndev
        split_z = h_local.reshape(nxh, ny_local, ndev, nz_local)
        h_z = lax.all_to_all(split_z, axis_name, split_axis=2, concat_axis=1, tiled=True)
        return h_z.reshape(nxh, ny_local * ndev, nz_local)

    mapped_local = local_y_to_z
    in_spec = y_slab_spec(axis_name)
    out_spec = z_slab_spec(axis_name)
    additional = ()
    if adjoint_axis_name is not None:
        mapped_local = jax.vmap(local_y_to_z)
        in_spec = adjoint_y_slab_spec(adjoint_axis_name, axis_name)
        out_spec = adjoint_z_slab_spec(adjoint_axis_name, axis_name)
        additional = (adjoint_axis_name,)
    return _shard_map(
        mapped_local,
        mesh=mesh,
        in_specs=in_spec,
        out_specs=out_spec,
        axis_name=axis_name,
        additional_axis_names=additional,
    )


def pressure_y_slab_to_z_slab(h_y: jax.Array, mesh: Mesh, axis_name: str = "z") -> jax.Array:
    """Redistribute `(nxh, ny_local, nz)` y slabs back to `(nxh, ny, nz_local)` z slabs."""

    validate_z_slab_shape(h_y.shape, mesh_size(mesh, axis_name))
    mapped = make_pressure_y_slab_to_z_slab(mesh, axis_name)
    return mapped(h_y)


def pressure_layout_roundtrip(h_z: jax.Array, mesh: Mesh, axis_name: str = "z") -> jax.Array:
    return pressure_y_slab_to_z_slab(pressure_z_slab_to_y_slab(h_z, mesh, axis_name), mesh, axis_name)
