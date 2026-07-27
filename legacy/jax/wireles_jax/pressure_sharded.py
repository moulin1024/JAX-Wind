from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .config import Params
from .pressure import _solve_tridiag
from .sharding import (
    adjoint_y_slab_spec,
    irfft2_fortran_layout,
    make_array_from_local_callback,
    make_pressure_y_slab_to_z_slab,
    make_pressure_z_slab_to_y_slab,
    mesh_size,
    rfft2_fortran_layout,
    validate_z_slab_shape,
    y_slab_sharding,
    y_slab_spec,
    z_slab_sharding,
    z_slab_spec,
    _shard_map,
)


class ShardedPressureOperators(NamedTuple):
    kx: jax.Array
    ky: jax.Array
    pressure_a: jax.Array
    pressure_inv_bet: jax.Array
    pressure_gam: jax.Array
    pressure_mode_keep: jax.Array


class ShardedSpikeOperators(NamedTuple):
    local_a: jax.Array
    local_inv_bet: jax.Array
    local_gam: jax.Array
    spike_left: jax.Array
    spike_right: jax.Array
    interface_inv: jax.Array


def _numpy_dtype(dtype: jnp.dtype) -> type[np.floating]:
    return np.float64 if jnp.dtype(dtype) == jnp.float64 else np.float32


def _pressure_mode_filter_fortran_layout(params: Params) -> np.ndarray:
    keep = np.ones((params.nx // 2 + 1, params.ny), dtype=bool)
    if params.pressure_filter_nyquist:
        if params.nx % 2 == 0:
            keep[-1, :] = False
        if params.ny % 2 == 0:
            keep[:, params.ny // 2] = False
    return keep[:, :, None]


def _pressure_tridiag_fortran_layout(
    params: Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtype = _numpy_dtype(params.dtype)
    n = params.nz
    dz2 = params.dz * params.dz
    kx = (2.0 * np.pi * np.fft.rfftfreq(params.nx, d=params.dx)).astype(dtype)
    ky = (2.0 * np.pi * np.fft.fftfreq(params.ny, d=params.dy)).astype(dtype)
    if params.nx % 2 == 0:
        kx[-1] = 0.0
    if params.ny % 2 == 0:
        ky[params.ny // 2] = 0.0

    k2 = kx[:, None] * kx[:, None] + ky[None, :] * ky[None, :]
    shape = (kx.shape[0], ky.shape[0], n)
    a = np.zeros(shape, dtype=dtype)
    b = np.zeros(shape, dtype=dtype)
    c = np.zeros(shape, dtype=dtype)

    zero_k2 = np.abs(k2) < np.finfo(dtype).eps * 128.0
    b[:, :, 0] = -k2 - 1.0 / dz2
    c[:, :, 0] = 1.0 / dz2
    b[:, :, 0] = np.where(zero_k2, 1.0, b[:, :, 0])
    c[:, :, 0] = np.where(zero_k2, 0.0, c[:, :, 0])

    if n > 2:
        a[:, :, 1:-1] = 1.0 / dz2
        b[:, :, 1:-1] = -k2[:, :, None] - 2.0 / dz2
        c[:, :, 1:-1] = 1.0 / dz2
    a[:, :, -1] = 1.0 / dz2
    b[:, :, -1] = -k2 - 1.0 / dz2

    inv_bet = np.zeros(shape, dtype=dtype)
    gam = np.zeros(shape, dtype=dtype)
    bet = b[:, :, 0]
    inv_bet[:, :, 0] = 1.0 / bet
    for j in range(1, n):
        gam[:, :, j] = c[:, :, j - 1] * inv_bet[:, :, j - 1]
        bet = b[:, :, j] - a[:, :, j] * gam[:, :, j]
        inv_bet[:, :, j] = 1.0 / bet

    keep = _pressure_mode_filter_fortran_layout(params).astype(dtype)
    return kx, ky, a, b, inv_bet, gam, keep


def make_sharded_pressure_operators(
    params: Params,
    mesh: Mesh,
    axis_name: str = "z",
    *,
    with_transpose_factors: bool = True,
) -> ShardedPressureOperators:
    ndev = mesh_size(mesh, axis_name)
    if params.ny % ndev != 0:
        raise ValueError(f"ny={params.ny} must be divisible by num_devices={ndev}")
    if params.nz % ndev != 0:
        raise ValueError(f"nz={params.nz} must be divisible by num_devices={ndev}")

    y_sharding = y_slab_sharding(mesh, axis_name)
    replicated = NamedSharding(mesh, P())
    dtype = _numpy_dtype(params.dtype)
    kx = (2.0 * np.pi * np.fft.rfftfreq(params.nx, d=params.dx)).astype(dtype)
    ky = (2.0 * np.pi * np.fft.fftfreq(params.ny, d=params.dy)).astype(dtype)
    if params.nx % 2 == 0:
        kx[-1] = 0.0
    if params.ny % 2 == 0:
        ky[params.ny // 2] = 0.0

    operator_shape = (params.nx // 2 + 1, params.ny, params.nz)

    def local_pressure_factors(index: tuple[slice, ...]) -> tuple[np.ndarray, ...]:
        kx_slice, ky_slice, z_slice = index
        local_kx = kx[kx_slice]
        local_ky = ky[ky_slice]
        nz_local = z_slice.stop - z_slice.start
        if z_slice.start != 0 or z_slice.stop != params.nz:
            raise ValueError("Pressure y pencils must own the complete vertical column.")
        k2 = local_kx[:, None] ** 2 + local_ky[None, :] ** 2
        shape = (local_kx.size, local_ky.size, nz_local)
        a = np.zeros(shape, dtype=dtype)
        b = np.zeros(shape, dtype=dtype)
        c = np.zeros(shape, dtype=dtype)
        dz2 = params.dz * params.dz
        zero_k2 = np.abs(k2) < np.finfo(dtype).eps * 128.0

        b[:, :, 0] = -k2 - 1.0 / dz2
        c[:, :, 0] = 1.0 / dz2
        b[:, :, 0] = np.where(zero_k2, 1.0, b[:, :, 0])
        c[:, :, 0] = np.where(zero_k2, 0.0, c[:, :, 0])
        if params.nz > 2:
            a[:, :, 1:-1] = 1.0 / dz2
            b[:, :, 1:-1] = -k2[:, :, None] - 2.0 / dz2
            c[:, :, 1:-1] = 1.0 / dz2
        a[:, :, -1] = 1.0 / dz2
        b[:, :, -1] = -k2 - 1.0 / dz2

        inv_bet = np.zeros(shape, dtype=dtype)
        gam = np.zeros(shape, dtype=dtype)
        bet = b[:, :, 0]
        inv_bet[:, :, 0] = 1.0 / bet
        for k in range(1, params.nz):
            gam[:, :, k] = c[:, :, k - 1] * inv_bet[:, :, k - 1]
            bet = b[:, :, k] - a[:, :, k] * gam[:, :, k]
            inv_bet[:, :, k] = 1.0 / bet

        keep = np.ones(shape, dtype=dtype)
        if params.pressure_filter_nyquist:
            global_kx = np.arange(kx_slice.start, kx_slice.stop)
            global_ky = np.arange(ky_slice.start, ky_slice.stop)
            if params.nx % 2 == 0:
                keep[global_kx == params.nx // 2, :, :] = 0.0
            if params.ny % 2 == 0:
                keep[:, global_ky == params.ny // 2, :] = 0.0
        return a, inv_bet, gam, keep

    def factor_callback(component: int):
        return lambda index: local_pressure_factors(index)[component]

    if with_transpose_factors:
        pressure_a = make_array_from_local_callback(
            operator_shape, y_sharding, factor_callback(0), dtype=params.dtype
        )
        pressure_inv_bet = make_array_from_local_callback(
            operator_shape, y_sharding, factor_callback(1), dtype=params.dtype
        )
        pressure_gam = make_array_from_local_callback(
            operator_shape, y_sharding, factor_callback(2), dtype=params.dtype
        )
        pressure_mode_keep = make_array_from_local_callback(
            operator_shape, y_sharding, factor_callback(3), dtype=params.dtype
        )
    else:
        dummy = jax.device_put(
            jnp.zeros((1, 1, 1), dtype=params.dtype), replicated
        )
        pressure_a = dummy
        pressure_inv_bet = dummy
        pressure_gam = dummy
        pressure_mode_keep = dummy
    return ShardedPressureOperators(
        kx=jax.device_put(jnp.asarray(kx[:, None, None], dtype=params.dtype), replicated),
        ky=jax.device_put(jnp.asarray(ky[None, :, None], dtype=params.dtype), replicated),
        pressure_a=pressure_a,
        pressure_inv_bet=pressure_inv_bet,
        pressure_gam=pressure_gam,
        pressure_mode_keep=pressure_mode_keep,
    )


def make_pressure_operators_reference(params: Params) -> ShardedPressureOperators:
    kx, ky, a, _, inv_bet, gam, keep = _pressure_tridiag_fortran_layout(params)
    return ShardedPressureOperators(
        kx=jnp.asarray(kx[:, None, None], dtype=params.dtype),
        ky=jnp.asarray(ky[None, :, None], dtype=params.dtype),
        pressure_a=jnp.asarray(a, dtype=params.dtype),
        pressure_inv_bet=jnp.asarray(inv_bet, dtype=params.dtype),
        pressure_gam=jnp.asarray(gam, dtype=params.dtype),
        pressure_mode_keep=jnp.asarray(keep, dtype=params.dtype),
    )


def _numpy_thomas_factors(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inv_bet = np.zeros_like(b)
    gam = np.zeros_like(b)
    bet = b[..., 0]
    inv_bet[..., 0] = 1.0 / bet
    for k in range(1, b.shape[-1]):
        gam[..., k] = c[..., k - 1] * inv_bet[..., k - 1]
        bet = b[..., k] - a[..., k] * gam[..., k]
        inv_bet[..., k] = 1.0 / bet
    return inv_bet, gam


def _numpy_thomas_apply(
    a: np.ndarray,
    inv_bet: np.ndarray,
    gam: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    forward = np.empty_like(rhs)
    forward[..., 0] = rhs[..., 0] * inv_bet[..., 0]
    for k in range(1, rhs.shape[-1]):
        forward[..., k] = (
            rhs[..., k] - a[..., k] * forward[..., k - 1]
        ) * inv_bet[..., k]
    result = np.empty_like(rhs)
    result[..., -1] = forward[..., -1]
    for k in range(rhs.shape[-1] - 2, -1, -1):
        result[..., k] = forward[..., k] - gam[..., k + 1] * result[..., k + 1]
    return result


def _spike_block_factors_numpy(
    params: Params,
    block: int,
    nblocks: int,
    kx: np.ndarray,
    ky: np.ndarray,
) -> tuple[np.ndarray, ...]:
    dtype = _numpy_dtype(params.dtype)
    m = params.nz // nblocks
    rows = block * m + np.arange(m)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    shape = (kx.size, ky.size, m)
    a = np.zeros(shape, dtype=dtype)
    b = np.zeros(shape, dtype=dtype)
    c = np.zeros(shape, dtype=dtype)
    inv_dz2 = 1.0 / (params.dz * params.dz)
    a[..., 1:] = inv_dz2
    c[..., :-1] = inv_dz2
    b[...] = -k2[..., None] - 2.0 * inv_dz2

    if rows[0] == 0:
        b[..., 0] = -k2 - inv_dz2
        zero_k2 = np.abs(k2) < np.finfo(dtype).eps * 128.0
        b[..., 0] = np.where(zero_k2, 1.0, b[..., 0])
        c[..., 0] = np.where(zero_k2, 0.0, c[..., 0])
    if rows[-1] == params.nz - 1:
        b[..., -1] = -k2 - inv_dz2

    external_left = np.zeros(k2.shape, dtype=dtype)
    external_right = np.zeros(k2.shape, dtype=dtype)
    if block > 0:
        external_left[...] = inv_dz2
    if block < nblocks - 1:
        external_right[...] = inv_dz2

    # Remove inter-block couplings from the local block matrix.
    a[..., 0] = 0.0
    c[..., -1] = 0.0
    inv_bet, gam = _numpy_thomas_factors(a, b, c)
    rhs_left = np.zeros(shape, dtype=dtype)
    rhs_right = np.zeros(shape, dtype=dtype)
    rhs_left[..., 0] = external_left
    rhs_right[..., -1] = external_right
    spike_left = _numpy_thomas_apply(a, inv_bet, gam, rhs_left)
    spike_right = _numpy_thomas_apply(a, inv_bet, gam, rhs_right)
    return a, inv_bet, gam, spike_left, spike_right


def make_sharded_spike_operators(
    params: Params,
    mesh: Mesh,
    axis_name: str = "z",
) -> ShardedSpikeOperators:
    """Build local-block SPIKE factors for the existing nz-row pressure matrix."""

    ndev = mesh_size(mesh, axis_name)
    if params.nz % ndev != 0:
        raise ValueError(f"nz={params.nz} must be divisible by num_devices={ndev}")
    if params.ny % ndev != 0:
        raise ValueError(f"ny={params.ny} must be divisible by num_devices={ndev}")
    if params.nz // ndev < 2:
        raise ValueError("SPIKE pressure solve requires at least two z rows per device.")

    dtype = _numpy_dtype(params.dtype)
    kx = (2.0 * np.pi * np.fft.rfftfreq(params.nx, d=params.dx)).astype(dtype)
    ky = (2.0 * np.pi * np.fft.fftfreq(params.ny, d=params.dy)).astype(dtype)
    if params.nx % 2 == 0:
        kx[-1] = 0.0
    if params.ny % 2 == 0:
        ky[params.ny // 2] = 0.0
    shape = (params.nx // 2 + 1, params.ny, params.nz)
    z_sharding = z_slab_sharding(mesh, axis_name)

    def block_component(component: int):
        def callback(index: tuple[slice, ...]) -> np.ndarray:
            kx_slice, ky_slice, z_slice = index
            m = params.nz // ndev
            if z_slice.stop - z_slice.start != m or z_slice.start % m:
                raise ValueError("SPIKE operators require contiguous equal z blocks.")
            block = z_slice.start // m
            factors = _spike_block_factors_numpy(
                params, block, ndev, kx[kx_slice], ky[ky_slice]
            )
            return factors[component]

        return callback

    reduced_n = 2 * ndev
    interface_shape = (
        params.nx // 2 + 1,
        params.ny,
        reduced_n,
        reduced_n,
    )
    interface_sharding = NamedSharding(mesh, P(None, axis_name, None, None))

    def interface_callback(index: tuple[slice, ...]) -> np.ndarray:
        kx_slice, ky_slice, row_slice, col_slice = index
        if row_slice.start != 0 or row_slice.stop != reduced_n:
            raise ValueError("SPIKE interface rows must be replicated.")
        if col_slice.start != 0 or col_slice.stop != reduced_n:
            raise ValueError("SPIKE interface columns must be replicated.")
        local_kx = kx[kx_slice]
        local_ky = ky[ky_slice]
        matrix = np.zeros(
            (local_kx.size, local_ky.size, reduced_n, reduced_n), dtype=dtype
        )
        for block in range(ndev):
            factors = _spike_block_factors_numpy(
                params, block, ndev, local_kx, local_ky
            )
            spike_left = factors[3]
            spike_right = factors[4]
            row_first = 2 * block
            row_last = row_first + 1
            matrix[..., row_first, row_first] = 1.0
            matrix[..., row_last, row_last] = 1.0
            if block > 0:
                beta_prev = 2 * (block - 1) + 1
                matrix[..., row_first, beta_prev] = spike_left[..., 0]
                matrix[..., row_last, beta_prev] = spike_left[..., -1]
            if block < ndev - 1:
                alpha_next = 2 * (block + 1)
                matrix[..., row_first, alpha_next] = spike_right[..., 0]
                matrix[..., row_last, alpha_next] = spike_right[..., -1]
        return np.linalg.inv(matrix).astype(dtype)

    return ShardedSpikeOperators(
        local_a=make_array_from_local_callback(
            shape, z_sharding, block_component(0), dtype=params.dtype
        ),
        local_inv_bet=make_array_from_local_callback(
            shape, z_sharding, block_component(1), dtype=params.dtype
        ),
        local_gam=make_array_from_local_callback(
            shape, z_sharding, block_component(2), dtype=params.dtype
        ),
        spike_left=make_array_from_local_callback(
            shape, z_sharding, block_component(3), dtype=params.dtype
        ),
        spike_right=make_array_from_local_callback(
            shape, z_sharding, block_component(4), dtype=params.dtype
        ),
        interface_inv=make_array_from_local_callback(
            interface_shape,
            interface_sharding,
            interface_callback,
            dtype=params.dtype,
        ),
    )


def _solve_pressure_hat_local(
    rhs_hat: jax.Array,
    pressure_a: jax.Array,
    pressure_inv_bet: jax.Array,
    pressure_gam: jax.Array,
    pressure_mode_keep: jax.Array,
    params: Params,
    *,
    clear_zero_mode: bool = True,
) -> jax.Array:
    rhs_col = lax.cond(
        jnp.asarray(clear_zero_mode),
        lambda rhs: rhs.at[0, 0, 0].set(0.0),
        lambda rhs: rhs,
        rhs_hat,
    )
    p_col = _solve_tridiag(pressure_a, pressure_inv_bet, pressure_gam, rhs_col)
    return p_col * pressure_mode_keep.astype(p_col.dtype)


def solve_pressure_hat_fortran_reference(
    rhs_hat: jax.Array,
    params: Params,
    ops: ShardedPressureOperators,
) -> jax.Array:
    return _solve_pressure_hat_local(
        rhs_hat,
        ops.pressure_a,
        ops.pressure_inv_bet,
        ops.pressure_gam,
        ops.pressure_mode_keep,
        params,
    )


def solve_pressure_hat_z_sharded(
    rhs_hat_z: jax.Array,
    params: Params,
    ops: ShardedPressureOperators,
    mesh: Mesh,
    axis_name: str = "z",
) -> jax.Array:
    solver = make_pressure_hat_solver_z_sharded(params, ops, mesh, axis_name)
    return solver(rhs_hat_z, ops)


def make_pressure_hat_solver_z_sharded(
    params: Params,
    ops: ShardedPressureOperators,
    mesh: Mesh,
    axis_name: str = "z",
    adjoint_axis_name: str | None = None,
):
    z_to_y = make_pressure_z_slab_to_y_slab(
        mesh, axis_name, adjoint_axis_name
    )
    y_to_z = make_pressure_y_slab_to_z_slab(
        mesh, axis_name, adjoint_axis_name
    )

    def local_solve(
        rhs_hat: jax.Array,
        pressure_a: jax.Array,
        pressure_inv_bet: jax.Array,
        pressure_gam: jax.Array,
        pressure_mode_keep: jax.Array,
    ) -> jax.Array:
        return _solve_pressure_hat_local(
            rhs_hat,
            pressure_a,
            pressure_inv_bet,
            pressure_gam,
            pressure_mode_keep,
            params,
            clear_zero_mode=lax.axis_index(axis_name) == 0,
        )

    mapped_local = local_solve
    rhs_spec = y_slab_spec(axis_name)
    output_spec = y_slab_spec(axis_name)
    additional = ()
    if adjoint_axis_name is not None:
        mapped_local = jax.vmap(
            local_solve,
            in_axes=(0, None, None, None, None),
            out_axes=0,
        )
        rhs_spec = adjoint_y_slab_spec(adjoint_axis_name, axis_name)
        output_spec = rhs_spec
        additional = (adjoint_axis_name,)
    mapped = _shard_map(
        mapped_local,
        mesh=mesh,
        in_specs=(
            rhs_spec,
            y_slab_spec(axis_name),
            y_slab_spec(axis_name),
            y_slab_spec(axis_name),
            y_slab_spec(axis_name),
        ),
        out_specs=output_spec,
        axis_name=axis_name,
        additional_axis_names=additional,
    )

    def solve(
        rhs_hat_z: jax.Array,
        runtime_ops: ShardedPressureOperators | None = None,
    ) -> jax.Array:
        active_ops = ops if runtime_ops is None else runtime_ops
        rhs_hat_y = z_to_y(rhs_hat_z)
        p_hat_y = mapped(
            rhs_hat_y,
            active_ops.pressure_a,
            active_ops.pressure_inv_bet,
            active_ops.pressure_gam,
            active_ops.pressure_mode_keep,
        )
        return y_to_z(p_hat_y)

    return solve


def make_pressure_hat_solver_z_sharded_spike(
    params: Params,
    mesh: Mesh,
    axis_name: str = "z",
):
    """Solve the nz-row pressure system in z layout with compact SPIKE exchanges."""

    ndev = mesh_size(mesh, axis_name)
    ny_local = params.ny // ndev
    interface_spec = P(None, axis_name, None, None)

    def scalars_to_modes(stacked: jax.Array) -> jax.Array:
        # (2, nxh, ny) on z blocks -> (P, 2, nxh, ny/P) on mode owners.
        values = stacked.reshape(2, stacked.shape[1], ndev, ny_local)
        values = jnp.moveaxis(values, 2, 0)
        if ndev > 1:
            values = lax.all_to_all(
                values,
                axis_name,
                split_axis=0,
                concat_axis=0,
                tiled=True,
            )
        return values

    def modes_to_scalars(values: jax.Array) -> jax.Array:
        # (P, 2, nxh, ny/P) mode-owner outputs -> (2, nxh, ny) on z blocks.
        if ndev > 1:
            values = lax.all_to_all(
                values,
                axis_name,
                split_axis=0,
                concat_axis=0,
                tiled=True,
            )
        values = jnp.moveaxis(values, 0, 2)
        return values.reshape(2, values.shape[1], params.ny)

    def local_solve(
        rhs_hat: jax.Array,
        local_a: jax.Array,
        local_inv_bet: jax.Array,
        local_gam: jax.Array,
        spike_left: jax.Array,
        spike_right: jax.Array,
        interface_inv: jax.Array,
    ) -> jax.Array:
        rank = lax.axis_index(axis_name)
        rhs_hat = lax.cond(
            rank == 0,
            lambda rhs: rhs.at[0, 0, 0].set(0.0),
            lambda rhs: rhs,
            rhs_hat,
        )
        local_y = _solve_tridiag(
            local_a, local_inv_bet, local_gam, rhs_hat
        )
        endpoints = jnp.stack((local_y[..., 0], local_y[..., -1]))
        interface_rhs_blocks = scalars_to_modes(endpoints)
        interface_rhs = jnp.transpose(
            interface_rhs_blocks, (2, 3, 0, 1)
        ).reshape(interface_inv.shape[:-2] + (2 * ndev,))
        interface_values = jnp.einsum(
            "...ij,...j->...i", interface_inv, interface_rhs
        )

        destination_values = []
        zeros = jnp.zeros_like(interface_values[..., 0])
        for block in range(ndev):
            left = (
                interface_values[..., 2 * (block - 1) + 1]
                if block > 0
                else zeros
            )
            right = (
                interface_values[..., 2 * (block + 1)]
                if block < ndev - 1
                else zeros
            )
            destination_values.append(jnp.stack((left, right)))
        left_right = modes_to_scalars(jnp.stack(destination_values))
        result = (
            local_y
            - spike_left * left_right[0][..., None]
            - spike_right * left_right[1][..., None]
        )
        if params.pressure_filter_nyquist:
            keep = jnp.ones(result.shape[:2], dtype=result.real.dtype)
            if params.nx % 2 == 0:
                keep = keep.at[-1, :].set(0.0)
            if params.ny % 2 == 0:
                keep = keep.at[:, params.ny // 2].set(0.0)
            result = result * keep[..., None]
        return result

    z = z_slab_spec(axis_name)
    mapped = _shard_map(
        local_solve,
        mesh=mesh,
        in_specs=(z, z, z, z, z, z, interface_spec),
        out_specs=z,
        axis_name=axis_name,
    )

    def solve(
        rhs_hat_z: jax.Array,
        spike_ops: ShardedSpikeOperators,
    ) -> jax.Array:
        return mapped(
            rhs_hat_z,
            spike_ops.local_a,
            spike_ops.local_inv_bet,
            spike_ops.local_gam,
            spike_ops.spike_left,
            spike_ops.spike_right,
            spike_ops.interface_inv,
        )

    return solve


def pressure_and_gradients_from_hat_z_sharded(
    p_hat_z: jax.Array,
    params: Params,
    ops: ShardedPressureOperators,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    fields_hat = jnp.stack(
        (
            p_hat_z,
            1j * ops.kx.astype(p_hat_z.real.dtype) * p_hat_z,
            1j * ops.ky.astype(p_hat_z.real.dtype) * p_hat_z,
        ),
        axis=0,
    )
    fields_inner = irfft2_fortran_layout(fields_hat, params.nx, params.ny)
    return (
        fields_inner[0].astype(params.dtype),
        fields_inner[1].astype(params.dtype),
        fields_inner[2].astype(params.dtype),
    )


def solve_pressure_inner_z_sharded(
    rhs_inner_z: jax.Array,
    params: Params,
    ops: ShardedPressureOperators,
    mesh: Mesh,
    axis_name: str = "z",
) -> jax.Array:
    rhs_hat_z = rfft2_fortran_layout(rhs_inner_z)
    p_hat_z = solve_pressure_hat_z_sharded(rhs_hat_z / params.dt, params, ops, mesh, axis_name)
    p_inner, _, _ = pressure_and_gradients_from_hat_z_sharded(p_hat_z, params, ops)
    return p_inner


def put_pressure_rhs_inner_z(
    rhs_inner: jax.Array,
    mesh: Mesh,
    axis_name: str = "z",
) -> jax.Array:
    if len(rhs_inner.shape) != 3:
        raise ValueError(f"Expected `(nx, ny, nz)` RHS interior array, got {rhs_inner.shape}")
    validate_z_slab_shape(rhs_inner.shape, mesh_size(mesh, axis_name))
    return jax.device_put(rhs_inner, z_slab_sharding(mesh, axis_name))


def put_pressure_hat_z(
    rhs_hat: jax.Array,
    mesh: Mesh,
    axis_name: str = "z",
) -> jax.Array:
    if len(rhs_hat.shape) != 3:
        raise ValueError(f"Expected `(nx/2+1, ny, nz)` spectral array, got {rhs_hat.shape}")
    validate_z_slab_shape(rhs_hat.shape, mesh_size(mesh, axis_name))
    return jax.device_put(rhs_hat, z_slab_sharding(mesh, axis_name))
