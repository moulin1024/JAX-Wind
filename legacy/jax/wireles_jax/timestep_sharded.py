from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .config import Params
from .derivative import (
    ddx,
    ddy,
    ddxy_filter_many,
    ddz_w,
    gradxy,
    horizontal_filter_many,
)
from .diagnostics import validate_cfl, validate_lasd_cfl
from .grid import make_operators
from .pressure_sharded import (
    ShardedPressureOperators,
    ShardedSpikeOperators,
    make_pressure_hat_solver_z_sharded,
    make_pressure_hat_solver_z_sharded_spike,
    make_sharded_pressure_operators,
    make_sharded_spike_operators,
    pressure_and_gradients_from_hat_z_sharded,
)
from .rhs import add_coriolis_geostrophic_forcing_inner
from .sgs import _trilinear_departure_interp, classic_smagorinsky
from .sharding import (
    _shard_map,
    adjoint_z_slab_spec,
    make_array_from_local_callback,
    make_distributed_mesh,
    mesh_size,
    rfft2_fortran_layout,
    z_slab_sharding,
    z_slab_spec,
)
from .state import Diagnostics, Operators
from .wall import wall_stress
from .wind_tunnel import classic_fringe_window


class ShardedFlowState(NamedTuple):
    u: jax.Array
    v: jax.Array
    w: jax.Array
    p: jax.Array
    theta: jax.Array
    qv: jax.Array
    rhs_u_prev: jax.Array
    rhs_v_prev: jax.Array
    rhs_w_prev: jax.Array
    rhs_theta_prev: jax.Array
    rhs_qv_prev: jax.Array
    lm_old: jax.Array
    mm_old: jax.Array
    qn_old: jax.Array
    nn_old: jax.Array
    cs2: jax.Array
    scalar_c: jax.Array
    u_lag: jax.Array
    v_lag: jax.Array
    w_lag: jax.Array
    step: jax.Array


class ShardedOperators(NamedTuple):
    horizontal: Operators
    pressure: ShardedPressureOperators
    pressure_spike: ShardedSpikeOperators | None


class LocalGradientBundle(NamedTuple):
    u: jax.Array
    v: jax.Array
    w: jax.Array
    theta: jax.Array
    qv: jax.Array
    dudx: jax.Array
    dudy: jax.Array
    dudz: jax.Array
    dudz_face: jax.Array
    dvdx: jax.Array
    dvdy: jax.Array
    dvdz: jax.Array
    dvdz_face: jax.Array
    dwdx: jax.Array
    dwdy: jax.Array
    dwdz: jax.Array
    dtheta_dx: jax.Array
    dtheta_dy: jax.Array
    dqv_dx: jax.Array
    dqv_dy: jax.Array


def _validate_sharded_params(params: Params, mesh: Mesh, axis_name: str) -> None:
    ndev = mesh_size(mesh, axis_name)
    if params.time_scheme != "ab2":
        raise ValueError("Distributed z-sharded timestep currently supports time_scheme='ab2' only.")
    if params.sgs_model not in {"smagorinsky", "lasd"}:
        raise ValueError("Distributed z-sharded timestep supports sgs_model='smagorinsky' or 'lasd'.")
    if params.thermo_enabled and params.scalar_sgs_model != "fixed_prandtl":
        raise ValueError(
            "Distributed z-sharded thermo currently requires scalar_sgs_model='fixed_prandtl'."
        )
    if params.thermo_enabled and params.scalar_vertical_scheme != "centered":
        raise ValueError(
            "Distributed z-sharded thermo currently requires scalar_vertical_scheme='centered'."
        )
    if params.thermo_enabled and params.theta_bc != "flux":
        raise ValueError("Distributed z-sharded thermo currently supports theta_bc='flux' only.")
    if params.thermo_enabled and params.scalar_stability_correction:
        raise ValueError(
            "Distributed z-sharded thermo does not yet support scalar_stability_correction."
        )
    if params.top_boundary_condition != "rigid_lid":
        raise ValueError("Distributed z-sharded timestep currently supports rigid_lid only.")
    if params.ny % ndev != 0:
        raise ValueError(f"ny={params.ny} must be divisible by devices={ndev}.")
    if params.nz % ndev != 0:
        raise ValueError(f"nz={params.nz} must be divisible by devices={ndev}.")
    if params.nz // ndev < 2:
        raise ValueError("Each device must own at least two z planes for the wall-model stencil.")


def make_sharded_operators(params: Params, mesh: Mesh, axis_name: str = "z") -> ShardedOperators:
    _validate_sharded_params(params, mesh, axis_name)
    return ShardedOperators(
        horizontal=make_operators(params),
        pressure=make_sharded_pressure_operators(
            params,
            mesh,
            axis_name,
            with_transpose_factors=params.sharded_pressure_solver == "transpose",
        ),
        pressure_spike=(
            make_sharded_spike_operators(params, mesh, axis_name)
            if params.sharded_pressure_solver == "spike"
            else None
        ),
    )


def initial_sharded_state(params: Params, mesh: Mesh, seed: int = 0, axis_name: str = "z") -> ShardedFlowState:
    _validate_sharded_params(params, mesh, axis_name)
    shape = (params.nx, params.ny, params.nz)
    z_sharding = z_slab_sharding(mesh, axis_name)
    scalar_sharding = NamedSharding(mesh, P(None, None, axis_name, None))
    np_dtype = np.dtype(params.dtype)
    np_sgs_dtype = np.dtype(params.sgs_dtype)

    def z_coordinates(index: tuple[slice, ...]) -> tuple[np.ndarray, np.ndarray]:
        z_slice = index[2]
        k = np.arange(z_slice.start, z_slice.stop, dtype=np_dtype)
        z_solver = (k + np_dtype.type(0.5)) * np_dtype.type(params.dz)
        return k, z_solver

    def profile(index: tuple[slice, ...], component: str) -> np.ndarray:
        x_slice, y_slice, _ = index
        k, z_solver = z_coordinates(index)
        z_phys = z_solver * np_dtype.type(params.z_i)
        local_shape = (
            x_slice.stop - x_slice.start,
            y_slice.stop - y_slice.start,
            k.size,
        )
        if component in {"u", "v"}:
            if params.initial_condition == "geostrophic":
                value = params.geostrophic_u if component == "u" else params.geostrophic_v
                vertical = np.full(k.shape, value, dtype=np_dtype)
            elif component == "v":
                vertical = np.zeros(k.shape, dtype=np_dtype)
            else:
                z_log = np.maximum(z_phys, params.zo * 1.01)
                target = (params.u_fric / params.vonk) * np.log(z_log / params.zo)
                cap = (params.u_fric / params.vonk) * np.log(
                    max(params.bl_height, params.zo * 1.01) / params.zo
                )
                target = np.where(z_phys >= params.bl_height, cap, target)
                if params.initial_condition == "uniform":
                    all_z = (np.arange(params.nz, dtype=np_dtype) + 0.5) * params.dz
                    all_phys = all_z * params.z_i
                    all_log = (params.u_fric / params.vonk) * np.log(
                        np.maximum(all_phys, params.zo * 1.01) / params.zo
                    )
                    all_log = np.where(all_phys >= params.bl_height, cap, all_log)
                    forced = all_z <= params.forcing_height
                    vertical = np.full(k.shape, np.mean(all_log[forced]), dtype=np_dtype)
                else:
                    vertical = target.astype(np_dtype)
        elif component in {"w", "p", "zero"}:
            vertical = np.zeros(k.shape, dtype=np_dtype)
        elif component == "theta":
            if params.theta_bc == "dirichlet":
                height = params.lz * params.z_i
                vertical = params.theta_bottom + (params.theta_top - params.theta_bottom) * z_phys / height
            elif params.theta_profile == "deardorff_cbl":
                zi = params.cbl_mixed_layer_height
                thickness = params.cbl_inversion_thickness
                inversion_bottom = zi - 0.5 * thickness
                inversion_top = zi + 0.5 * thickness
                ramp = np.clip((z_phys - inversion_bottom) / thickness, 0.0, 1.0)
                ramp = ramp * ramp * (3.0 - 2.0 * ramp)
                vertical = (
                    params.theta0
                    + params.cbl_inversion_strength * ramp
                    + params.cbl_free_atmosphere_gradient * np.maximum(z_phys - inversion_top, 0.0)
                )
            else:
                vertical = params.theta0 + params.theta_initial_gradient * z_phys
        elif component == "qv":
            vertical = np.maximum(params.qv0 + params.qv_initial_gradient * z_phys, params.qv_floor)
        else:  # pragma: no cover - internal programming error
            raise ValueError(component)
        result = np.broadcast_to(np.asarray(vertical, dtype=np_dtype), local_shape).copy()

        stream = {"u": 1, "v": 2, "w": 3, "theta": 4}.get(component)
        if stream is not None:
            if component == "theta":
                amplitude = params.theta_perturbation_amplitude
                if params.theta_perturbation_height is None:
                    envelope = np.sin(np.pi * z_solver / params.lz)
                else:
                    envelope = np.where(
                        z_phys < params.theta_perturbation_height,
                        np.sin(np.pi * z_phys / params.theta_perturbation_height),
                        0.0,
                    )
            else:
                amplitude = params.initial_velocity_noise
                envelope = np.ones_like(z_solver)
                if params.momentum_wall_model == "abl":
                    envelope = (k < min(4, params.nz)).astype(np_dtype)
                    if component == "w" or (
                        component == "v" and params.initial_condition != "geostrophic"
                    ):
                        envelope = np.zeros_like(envelope)
            if amplitude > 0.0:
                for local_k, global_k in enumerate(k.astype(np.int64)):
                    if envelope[local_k] == 0.0:
                        continue
                    rng = np.random.default_rng(np.random.SeedSequence([seed, stream, int(global_k)]))
                    perturbation = rng.standard_normal(result.shape[:2])
                    if component != "theta":
                        # White grid-scale noise is not a valid LES initial
                        # condition for the centered rotational operator.  It
                        # excites unresolved Nyquist modes before the first
                        # nonlinear step.  Retain only smooth, resolved modes;
                        # the initial pressure projection then makes the
                        # velocity perturbation solenoidal.
                        perturbation_hat = np.fft.rfft2(perturbation)
                        mode_x = np.fft.fftfreq(result.shape[0]) * result.shape[0]
                        mode_y = np.fft.rfftfreq(result.shape[1]) * result.shape[1]
                        cutoff_x = max(1, result.shape[0] // 16)
                        cutoff_y = max(1, result.shape[1] // 16)
                        keep = (
                            np.abs(mode_x)[:, None] <= cutoff_x
                        ) & (mode_y[None, :] <= cutoff_y)
                        perturbation = np.fft.irfft2(
                            perturbation_hat * keep,
                            s=result.shape[:2],
                        ).real
                        perturbation -= np.mean(perturbation)
                        perturbation_std = np.std(perturbation)
                        if perturbation_std > 0.0:
                            perturbation /= perturbation_std
                    result[:, :, local_k] += (
                        amplitude * envelope[local_k] * perturbation.astype(np_dtype)
                    )
        return result

    def make_field(component: str, *, dtype=params.dtype) -> jax.Array:
        return make_array_from_local_callback(
            shape, z_sharding, lambda index: profile(index, component), dtype=dtype
        )

    def zeros(*, dtype=params.dtype) -> jax.Array:
        return make_array_from_local_callback(
            shape,
            z_sharding,
            lambda index: np.zeros(
                tuple(part.stop - part.start for part in index), dtype=np.dtype(dtype)
            ),
            dtype=dtype,
        )

    base_cs2 = params.smagorinsky_cs * params.smagorinsky_cs

    def constant_sgs(value: float) -> jax.Array:
        return make_array_from_local_callback(
            shape,
            z_sharding,
            lambda index: np.full(
                tuple(part.stop - part.start for part in index), value, dtype=np_sgs_dtype
            ),
            dtype=params.sgs_dtype,
        )

    scalar_shape = shape + (2,)

    def scalar_coeff(index: tuple[slice, ...]) -> np.ndarray:
        local_shape = tuple(part.stop - part.start for part in index)
        result = np.empty(local_shape, dtype=np_sgs_dtype)
        result[..., 0] = base_cs2 / params.prandtl_t
        result[..., 1] = base_cs2 / params.schmidt_t
        return result

    return ShardedFlowState(
        u=make_field("u"),
        v=make_field("v"),
        w=make_field("w"),
        p=zeros(),
        theta=make_field("theta"),
        qv=make_field("qv"),
        rhs_u_prev=zeros(),
        rhs_v_prev=zeros(),
        rhs_w_prev=zeros(),
        rhs_theta_prev=zeros(),
        rhs_qv_prev=zeros(),
        lm_old=zeros(dtype=params.sgs_dtype),
        mm_old=zeros(dtype=params.sgs_dtype),
        qn_old=zeros(dtype=params.sgs_dtype),
        nn_old=zeros(dtype=params.sgs_dtype),
        cs2=constant_sgs(base_cs2),
        scalar_c=make_array_from_local_callback(
            scalar_shape, scalar_sharding, scalar_coeff, dtype=params.sgs_dtype
        ),
        u_lag=zeros(dtype=params.sgs_dtype),
        v_lag=zeros(dtype=params.sgs_dtype),
        w_lag=zeros(dtype=params.sgs_dtype),
        step=jax.device_put(jnp.array(0, dtype=jnp.int32), NamedSharding(mesh, P())),
    )


def _plane_zeros(q: jax.Array) -> jax.Array:
    return jnp.zeros(q.shape[:2] + (1,), dtype=q.dtype)


def _ppermute_lower(last_plane: jax.Array, axis_name: str, ndev: int) -> jax.Array:
    if ndev == 1:
        return jnp.zeros_like(last_plane)
    return lax.ppermute(last_plane, axis_name, perm=[(i, i + 1) for i in range(ndev - 1)])


def _ppermute_upper(first_plane: jax.Array, axis_name: str, ndev: int) -> jax.Array:
    if ndev == 1:
        return jnp.zeros_like(first_plane)
    return lax.ppermute(first_plane, axis_name, perm=[(i, i - 1) for i in range(1, ndev)])


def _z_halo_many(
    qs: tuple[jax.Array, ...],
    lower_boundaries: tuple[jax.Array, ...],
    upper_boundaries: tuple[jax.Array, ...],
    axis_name: str,
    ndev: int,
) -> tuple[jax.Array, ...]:
    """Exchange a same-layout field bundle with one collective per direction."""

    if not qs:
        return ()
    if len(qs) != len(lower_boundaries) or len(qs) != len(upper_boundaries):
        raise ValueError("Packed halo fields and boundary tuples must have matching lengths.")
    reference_shape = qs[0].shape
    reference_dtype = qs[0].dtype
    if any(q.shape != reference_shape or q.dtype != reference_dtype for q in qs):
        raise ValueError("Packed halo fields must have identical shapes and dtypes.")
    packed = jnp.stack(qs, axis=-1)
    lower_boundary = jnp.stack(lower_boundaries, axis=-1)
    upper_boundary = jnp.stack(upper_boundaries, axis=-1)
    if ndev == 1:
        halo = jnp.concatenate((lower_boundary, packed, upper_boundary), axis=2)
    else:
        rank = lax.axis_index(axis_name)
        lower_recv = _ppermute_lower(packed[:, :, -1:, :], axis_name, ndev)
        upper_recv = _ppermute_upper(packed[:, :, :1, :], axis_name, ndev)
        lower = jnp.where(rank == 0, lower_boundary, lower_recv)
        upper = jnp.where(rank == ndev - 1, upper_boundary, upper_recv)
        halo = jnp.concatenate((lower, packed, upper), axis=2)
    return tuple(halo[..., field] for field in range(len(qs)))


def _copy_boundary_halos(
    qs: tuple[jax.Array, ...],
    axis_name: str,
    ndev: int,
) -> tuple[jax.Array, ...]:
    return _z_halo_many(
        qs,
        tuple(q[:, :, :1] for q in qs),
        tuple(q[:, :, -1:] for q in qs),
        axis_name,
        ndev,
    )


def _zero_boundary_halos(
    qs: tuple[jax.Array, ...],
    axis_name: str,
    ndev: int,
) -> tuple[jax.Array, ...]:
    return _z_halo_many(
        qs,
        tuple(_plane_zeros(q) for q in qs),
        tuple(_plane_zeros(q) for q in qs),
        axis_name,
        ndev,
    )


def _copy_boundary_halo(q: jax.Array, axis_name: str, ndev: int) -> jax.Array:
    return _copy_boundary_halos((q,), axis_name, ndev)[0]


def _zero_boundary_halo(q: jax.Array, axis_name: str, ndev: int) -> jax.Array:
    return _zero_boundary_halos((q,), axis_name, ndev)[0]


def _set_w_physical_boundaries(w: jax.Array, axis_name: str, ndev: int) -> jax.Array:
    rank = lax.axis_index(axis_name)
    w_top = w.at[:, :, -1].set(0.0)
    w = jnp.where(rank == ndev - 1, w_top, w)
    return w


def _velocity_halos(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    axis_name: str,
    ndev: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    w = _set_w_physical_boundaries(w, axis_name, ndev)
    return _z_halo_many(
        (u, v, w),
        (u[:, :, :1], v[:, :, :1], _plane_zeros(w)),
        (u[:, :, -1:], v[:, :, -1:], _plane_zeros(w)),
        axis_name,
        ndev,
    )


def _halos_from_interior_many(
    q_halos: tuple[jax.Array, ...],
    axis_name: str,
    ndev: int,
) -> tuple[jax.Array, ...]:
    return _zero_boundary_halos(
        tuple(q[:, :, 1:-1] for q in q_halos), axis_name, ndev
    )


def _set_rank0_plane(q: jax.Array, plane_index: int, value: jax.Array, axis_name: str) -> jax.Array:
    rank = lax.axis_index(axis_name)
    updated = q.at[:, :, plane_index].set(value)
    return jnp.where(rank == 0, updated, q)


def _set_top_rank_plane(
    q: jax.Array,
    plane_index: int,
    value: jax.Array | float,
    axis_name: str,
    ndev: int,
) -> jax.Array:
    rank = lax.axis_index(axis_name)
    updated = q.at[:, :, plane_index].set(jnp.asarray(value, dtype=q.dtype))
    return jnp.where(rank == ndev - 1, updated, q)


def _apply_porte_agel_wall_correction_local(
    dudz_face: jax.Array,
    dvdz_face: jax.Array,
    axis_name: str,
    horizontal_average: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Apply the single-domain correction at global upper face zero.

    Halo index one is the upper face owned by the first physical center on
    rank zero.  Using index two here applies the correction one level too
    high and was inherited from the former cell-centred-gradient layout.
    """
    fr1 = 1.0 / jnp.log(jnp.asarray(3.0, dtype=dudz_face.dtype)) - 1.0
    dudz_plane = dudz_face[:, :, 1]
    dvdz_plane = dvdz_face[:, :, 1]
    dudz_correction = jnp.mean(dudz_plane) if horizontal_average else dudz_plane
    dvdz_correction = jnp.mean(dvdz_plane) if horizontal_average else dvdz_plane
    dudz_corr = dudz_face.at[:, :, 1].add(fr1 * dudz_correction)
    dvdz_corr = dvdz_face.at[:, :, 1].add(fr1 * dvdz_correction)
    rank = lax.axis_index(axis_name)
    return (
        jnp.where(rank == 0, dudz_corr, dudz_face),
        jnp.where(rank == 0, dvdz_corr, dvdz_face),
    )


def _wall_stress_local(
    u: jax.Array,
    v: jax.Array,
    params: Params,
    axis_name: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    txz0, tyz0, dudz0, dvdz0, ustar = wall_stress(u, v, params)
    rank = lax.axis_index(axis_name)
    zero2 = jnp.zeros_like(txz0)
    return (
        jnp.where(rank == 0, txz0, zero2),
        jnp.where(rank == 0, tyz0, zero2),
        jnp.where(rank == 0, dudz0, zero2),
        jnp.where(rank == 0, dvdz0, zero2),
        jnp.where(rank == 0, ustar, zero2),
    )


def _assemble_rhs_inner_local(
    c: jax.Array,
    div_t: jax.Array,
    params: Params,
    pressure_force: bool,
    axis_name: str,
) -> jax.Array:
    rhs = (-c - div_t)[:, :, 1:-1]
    if pressure_force and params.driving_pressure_force != 0.0:
        z_local = rhs.shape[2]
        rank = lax.axis_index(axis_name)
        k_global = rank * z_local + jnp.arange(z_local, dtype=params.dtype)
        z = (k_global + 0.5) * params.dz
        mask = (z <= params.forcing_height).astype(params.dtype)
        rhs = rhs + params.driving_pressure_force * mask[None, None, :]
    return rhs.astype(params.dtype)


def _ab_update_inner(q: jax.Array, rhs: jax.Array, rhs_prev: jax.Array, step: jax.Array, params: Params) -> jax.Array:
    euler = q + params.dt * rhs
    ab2 = q + params.dt * (1.5 * rhs - 0.5 * rhs_prev)
    step_mask = step == 0
    if step_mask.ndim < q.ndim:
        step_mask = step_mask.reshape(
            step_mask.shape + (1,) * (q.ndim - step_mask.ndim)
        )
    return jnp.where(step_mask, euler, ab2)


def _to_sgs(q: jax.Array, params: Params) -> jax.Array:
    return q.astype(params.sgs_dtype)


def _physical_mask_halo(q: jax.Array) -> jax.Array:
    return jnp.zeros_like(q).at[:, :, 1:-1].set(1.0)


def _sym_dot(a: jax.Array, b: jax.Array) -> jax.Array:
    return (
        a[..., 0] * b[..., 0]
        + 2.0 * a[..., 1] * b[..., 1]
        + 2.0 * a[..., 2] * b[..., 2]
        + a[..., 3] * b[..., 3]
        + 2.0 * a[..., 4] * b[..., 4]
        + a[..., 5] * b[..., 5]
    )


def _strain_magnitude(sij: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.maximum(2.0 * _sym_dot(sij, sij), 0.0))


def _safe_divide(num: jax.Array, den: jax.Array) -> jax.Array:
    valid = jnp.abs(den) > 1.0e-30
    safe_den = jnp.where(valid, den, 1.0)
    return jnp.where(valid, num / safe_den, 0.0)


def _shift_z_minus(q: jax.Array) -> jax.Array:
    return jnp.concatenate((q[:, :, :1], q[:, :, :-1]), axis=2)


def _shift_z_plus(q: jax.Array) -> jax.Array:
    return jnp.concatenate((q[:, :, 1:], q[:, :, -1:]), axis=2)


def _avg_next_halo(q: jax.Array) -> jax.Array:
    return 0.5 * (q + _shift_z_plus(q))


def _avg_prev_halo(q: jax.Array) -> jax.Array:
    return 0.5 * (q + _shift_z_minus(q))


def _convec_halo(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    dudy: jax.Array,
    dudz_face: jax.Array,
    dvdx: jax.Array,
    dvdz_face: jax.Array,
    dwdx: jax.Array,
    dwdy: jax.Array,
    axis_name: str,
    ndev: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rotational convection with the single-domain staggered semantics."""
    vertical_x_upper = w * (dudz_face - dwdx)
    vertical_y_upper = w * (dvdz_face - dwdy)
    cx = v * (dudy - dvdx) + 0.5 * (
        _shift_z_minus(vertical_x_upper) + vertical_x_upper
    )
    cy = u * (dvdx - dudy) + 0.5 * (
        _shift_z_minus(vertical_y_upper) + vertical_y_upper
    )
    cz = _avg_next_halo(u) * (dwdx - dudz_face)
    cz = cz + _avg_next_halo(v) * (dwdy - dvdz_face)
    cz = _set_top_rank_plane(cz, -2, 0.0, axis_name, ndev)
    return cx, cy, cz


def _gradient_to_upper_halo(q: jax.Array, params: Params) -> jax.Array:
    return (_shift_z_plus(q) - q) / params.dz


def _strain_uv_halo(
    dudx: jax.Array,
    dudy: jax.Array,
    dudz: jax.Array,
    dvdx: jax.Array,
    dvdy: jax.Array,
    dvdz: jax.Array,
    dwdx: jax.Array,
    dwdy: jax.Array,
    dwdz: jax.Array,
    axis_name: str,
) -> jax.Array:
    ux = dudx
    uy = dudy
    uz = dudz
    vx = dvdx
    vy = dvdy
    vz = dvdz
    wx = _avg_prev_halo(dwdx)
    wy = _avg_prev_halo(dwdy)
    wz = dwdz
    return jnp.stack(
        (
            ux,
            0.5 * (uy + vx),
            0.5 * (uz + wx),
            vy,
            0.5 * (vz + wy),
            wz,
        ),
        axis=-1,
    )


def _strain_w_halo(
    dudx: jax.Array,
    dudy: jax.Array,
    dudz: jax.Array,
    dvdx: jax.Array,
    dvdy: jax.Array,
    dvdz: jax.Array,
    dwdx: jax.Array,
    dwdy: jax.Array,
    dwdz: jax.Array,
    *,
    dudz_face: jax.Array | None = None,
    dvdz_face: jax.Array | None = None,
) -> jax.Array:
    ux = _avg_next_halo(dudx)
    uy = _avg_next_halo(dudy)
    uz = _avg_next_halo(dudz) if dudz_face is None else dudz_face
    vx = _avg_next_halo(dvdx)
    vy = _avg_next_halo(dvdy)
    vz = _avg_next_halo(dvdz) if dvdz_face is None else dvdz_face
    wx = dwdx
    wy = dwdy
    wz = _avg_next_halo(dwdz)
    return jnp.stack(
        (
            ux,
            0.5 * (uy + vx),
            0.5 * (uz + wx),
            vy,
            0.5 * (vz + wy),
            wz,
        ),
        axis=-1,
    )


def _spectral_box_filter_concat_hat_halo(
    q_hat: jax.Array,
    template: jax.Array,
    params: Params,
    filter_width: float,
) -> jax.Array:
    x_mode = jnp.abs(jnp.fft.fftfreq(params.nx, d=1.0) * params.nx)
    y_mode = jnp.fft.rfftfreq(params.ny, d=1.0) * params.ny
    cutoff_x = jnp.rint(params.nx / (2.0 * filter_width))
    cutoff_y = jnp.rint(params.ny / (2.0 * filter_width))
    keep = (x_mode[:, None] < cutoff_x) & (y_mode[None, :] < cutoff_y)
    q_filtered = jnp.fft.irfft2(
        q_hat * keep[:, :, None, None].astype(q_hat.dtype),
        s=(params.nx, params.ny),
        axes=(0, 1),
    ).real
    return jnp.zeros_like(template).at[:, :, 1:-1, :].set(q_filtered.astype(template.dtype))


def _velocity_products_halo(u: jax.Array, v: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
    w_uv = _avg_prev_halo(w)
    vel = jnp.stack((u, v, w_uv), axis=-1)
    uu = jnp.stack(
        (
            vel[..., 0] * vel[..., 0],
            vel[..., 0] * vel[..., 1],
            vel[..., 0] * vel[..., 2],
            vel[..., 1] * vel[..., 1],
            vel[..., 1] * vel[..., 2],
            vel[..., 2] * vel[..., 2],
        ),
        axis=-1,
    )
    return vel, uu


def _lmqn_from_filtered_halo(
    filtered: jax.Array,
    params: Params,
    test_ratio: float,
) -> tuple[jax.Array, jax.Array]:
    vel_hat = filtered[..., :3]
    uu_hat = filtered[..., 3:9]
    sij_hat = filtered[..., 9:15]
    ssij_hat = filtered[..., 15:21]
    l_ij = jnp.stack(
        (
            uu_hat[..., 0] - vel_hat[..., 0] * vel_hat[..., 0],
            uu_hat[..., 1] - vel_hat[..., 0] * vel_hat[..., 1],
            uu_hat[..., 2] - vel_hat[..., 0] * vel_hat[..., 2],
            uu_hat[..., 3] - vel_hat[..., 1] * vel_hat[..., 1],
            uu_hat[..., 4] - vel_hat[..., 1] * vel_hat[..., 2],
            uu_hat[..., 5] - vel_hat[..., 2] * vel_hat[..., 2],
        ),
        axis=-1,
    )
    delta = params.sgs_delta
    m_ij = 2.0 * delta * delta * (
        ssij_hat - test_ratio * test_ratio * _strain_magnitude(sij_hat)[..., None] * sij_hat
    )
    return _sym_dot(l_ij, m_ij), _sym_dot(m_ij, m_ij)


def _lmqn_pair_halo(
    vel: jax.Array,
    uu: jax.Array,
    sij: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    ssij = _strain_magnitude(sij)[..., None] * sij
    q = jnp.concatenate((vel, uu, sij, ssij), axis=-1)
    q_hat = jnp.fft.rfft2(q[:, :, 1:-1, :], axes=(0, 1))
    filtered_2d = _spectral_box_filter_concat_hat_halo(q_hat, q, params, params.fgr * params.tfr)
    filtered_4d = _spectral_box_filter_concat_hat_halo(q_hat, q, params, params.fgr * params.tfr * params.tfr)
    lm, mm = _lmqn_from_filtered_halo(filtered_2d, params, params.tfr)
    qn, nn = _lmqn_from_filtered_halo(filtered_4d, params, params.tfr * params.tfr)
    return lm, mm, qn, nn


def _lagrangian_interp_halo(
    q: jax.Array,
    u_lag: jax.Array,
    v_lag: jax.Array,
    w_lag: jax.Array,
    params: Params,
) -> jax.Array:
    dt_lag = params.dt * params.cs_count
    return _trilinear_departure_interp(
        q,
        -u_lag * dt_lag,
        -v_lag * dt_lag,
        -w_lag * dt_lag,
        params,
    )


def _lagrangian_average_halo(
    current_a: jax.Array,
    current_b: jax.Array,
    old_a: jax.Array,
    old_b: jax.Array,
    u_lag: jax.Array,
    v_lag: jax.Array,
    w_lag: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array]:
    dt_lag = params.dt * params.cs_count
    product = old_a * old_b
    valid = (old_a > 0.0) & (old_b >= 0.0) & (product > 0.0)
    tn = 1.5 * params.sgs_delta
    tn = tn * jnp.where(valid, product ** (-0.125), 1.0)
    eps = jnp.where(valid, (dt_lag / tn) / (1.0 + dt_lag / tn), 0.0)
    a_interp = _lagrangian_interp_halo(old_a, u_lag, v_lag, w_lag, params)
    b_interp = _lagrangian_interp_halo(old_b, u_lag, v_lag, w_lag, params)
    return eps * current_a + (1.0 - eps) * a_interp, eps * current_b + (1.0 - eps) * b_interp


def _w_stress_mask_halo(q: jax.Array, params: Params, axis_name: str) -> jax.Array:
    rank = lax.axis_index(axis_name)
    z_local = q.shape[2] - 2
    k_global = rank * z_local + jnp.arange(z_local)
    mask_inner = (k_global < params.nz - 1).astype(q.dtype)
    return jnp.zeros_like(q).at[:, :, 1:-1].set(mask_inner[None, None, :])


def _stress_from_cs2_halo(
    cs2: jax.Array,
    sij_uv: jax.Array,
    sij_w: jax.Array,
    params: Params,
    axis_name: str,
) -> tuple[jax.Array, ...]:
    delta = params.sgs_delta
    factor_uv = -2.0 * cs2 * delta * delta * _strain_magnitude(sij_uv)
    txx = factor_uv * sij_uv[..., 0]
    txy = factor_uv * sij_uv[..., 1]
    tyy = factor_uv * sij_uv[..., 3]
    tzz = factor_uv * sij_uv[..., 5]

    cs2_w = _avg_next_halo(cs2)
    factor_w = -2.0 * cs2_w * delta * delta * _strain_magnitude(sij_w)
    txz = factor_w * sij_w[..., 2]
    tyz = factor_w * sij_w[..., 4]
    w_mask = _w_stress_mask_halo(cs2, params, axis_name)
    mask = _physical_mask_halo(cs2)
    return txx * mask, txy * mask, txz * w_mask, tyy * mask, tyz * w_mask, tzz * mask


def _update_lasd_coefficients_halo(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    sij_uv: jax.Array,
    cs2_i: jax.Array,
    lm_i: jax.Array,
    mm_i: jax.Array,
    qn_i: jax.Array,
    nn_i: jax.Array,
    u_lag_i: jax.Array,
    v_lag_i: jax.Array,
    w_lag_i: jax.Array,
    step: jax.Array,
    params: Params,
    axis_name: str,
    ndev: int,
) -> tuple[jax.Array, ...]:
    u = _to_sgs(u, params)
    v = _to_sgs(v, params)
    w = _to_sgs(w, params)
    sij_uv = _to_sgs(sij_uv, params)

    cs2_state = _zero_boundary_halo(_to_sgs(cs2_i, params), axis_name, ndev)
    lm_state, mm_state, qn_state, nn_state = _copy_boundary_halos(
        (
            _to_sgs(lm_i, params),
            _to_sgs(mm_i, params),
            _to_sgs(qn_i, params),
            _to_sgs(nn_i, params),
        ),
        axis_name,
        ndev,
    )

    u_lag, v_lag, w_lag = _copy_boundary_halos(
        (
            _to_sgs(u_lag_i, params),
            _to_sgs(v_lag_i, params),
            _to_sgs(w_lag_i, params),
        ),
        axis_name,
        ndev,
    )
    u_lag = u_lag + u / params.cs_count
    v_lag = v_lag + v / params.cs_count
    w_lag = w_lag + _avg_prev_halo(w) / params.cs_count

    should_update = ((step + 1) % params.cs_count) == 0

    def do_update(_: None) -> tuple[jax.Array, ...]:
        vel, uu = _velocity_products_halo(u, v, w)
        lm, mm, qn, nn = _lmqn_pair_halo(vel, uu, sij_uv, params)

        first_update = step == params.cs_count - 1
        lm_old = jnp.where(first_update, 0.03 * mm, lm_state)
        mm_old = jnp.where(first_update, mm, mm_state)
        qn_old = jnp.where(first_update, 0.03 * nn, qn_state)
        nn_old = jnp.where(first_update, nn, nn_state)
        lm_old, mm_old, qn_old, nn_old = _copy_boundary_halos(
            (
                lm_old[:, :, 1:-1],
                mm_old[:, :, 1:-1],
                qn_old[:, :, 1:-1],
                nn_old[:, :, 1:-1],
            ),
            axis_name,
            ndev,
        )

        lm_avg, mm_avg = _lagrangian_average_halo(lm, mm, lm_old, mm_old, u_lag, v_lag, w_lag, params)
        qn_avg, nn_avg = _lagrangian_average_halo(qn, nn, qn_old, nn_old, u_lag, v_lag, w_lag, params)

        cs2_2d = jnp.maximum(_safe_divide(lm_avg, mm_avg), 0.0)
        cs2_4d = jnp.maximum(_safe_divide(qn_avg, nn_avg), 0.0)
        exponent = jnp.log(jnp.asarray(params.tfr, dtype=params.sgs_dtype)) / (
            jnp.log(jnp.asarray(params.tfr * params.tfr, dtype=params.sgs_dtype))
            - jnp.log(jnp.asarray(params.tfr, dtype=params.sgs_dtype))
        )
        beta = _safe_divide(cs2_4d, cs2_2d) ** exponent
        beta = jnp.maximum(beta, 1.0 / (params.tfr * params.tfr * params.tfr))
        cs2_new = jnp.clip(_safe_divide(cs2_2d, beta), 1.0e-6, 0.81)
        cs2_new = cs2_new * _physical_mask_halo(cs2_new)
        cs2_new = _zero_boundary_halo(cs2_new[:, :, 1:-1], axis_name, ndev)
        zero = jnp.zeros_like(u_lag)
        return cs2_new, lm_avg, mm_avg, qn_avg, nn_avg, zero, zero, zero

    def skip_update(_: None) -> tuple[jax.Array, ...]:
        return cs2_state, lm_state, mm_state, qn_state, nn_state, u_lag, v_lag, w_lag

    return jax.lax.cond(should_update, do_update, skip_update, operand=None)


def _build_gradient_bundle_local(
    u_i: jax.Array,
    v_i: jax.Array,
    w_i: jax.Array,
    theta_i: jax.Array | None,
    qv_i: jax.Array | None,
    params: Params,
    ops: Operators,
    axis_name: str,
    ndev: int,
) -> tuple[LocalGradientBundle, tuple[jax.Array, ...]]:
    """Build filtered fields and velocity/scalar gradients once per RHS."""

    w_i = _set_w_physical_boundaries(w_i, axis_name, ndev)
    if theta_i is None:
        u_h, v_h, w_h = _z_halo_many(
            (u_i, v_i, w_i),
            (u_i[:, :, :1], v_i[:, :, :1], _plane_zeros(w_i)),
            (u_i[:, :, -1:], v_i[:, :, -1:], _plane_zeros(w_i)),
            axis_name,
            ndev,
        )
        filtered, dx, dy = ddxy_filter_many((u_h, v_h, w_h), params, ops)
        u_h, v_h, w_h = filtered
        dudx_h, dvdx_h, dwdx_h = dx
        dudy_h, dvdy_h, dwdy_h = dy
        theta_h = jnp.zeros_like(u_h)
        qv_h = jnp.zeros_like(u_h)
        dtheta_dx_h = jnp.zeros_like(u_h)
        dtheta_dy_h = jnp.zeros_like(u_h)
        dqv_dx_h = jnp.zeros_like(u_h)
        dqv_dy_h = jnp.zeros_like(u_h)
    elif qv_i is None:
        u_h, v_h, w_h, theta_h = _z_halo_many(
            (u_i, v_i, w_i, theta_i),
            (
                u_i[:, :, :1],
                v_i[:, :, :1],
                _plane_zeros(w_i),
                theta_i[:, :, :1],
            ),
            (
                u_i[:, :, -1:],
                v_i[:, :, -1:],
                _plane_zeros(w_i),
                theta_i[:, :, -1:],
            ),
            axis_name,
            ndev,
        )
        filtered, dx, dy = ddxy_filter_many(
            (u_h, v_h, w_h, theta_h), params, ops
        )
        u_h, v_h, w_h, theta_h = filtered
        dudx_h, dvdx_h, dwdx_h, dtheta_dx_h = dx
        dudy_h, dvdy_h, dwdy_h, dtheta_dy_h = dy
        qv_h = jnp.zeros_like(theta_h)
        dqv_dx_h = jnp.zeros_like(theta_h)
        dqv_dy_h = jnp.zeros_like(theta_h)
    else:
        u_h, v_h, w_h, theta_h, qv_h = _z_halo_many(
            (u_i, v_i, w_i, theta_i, qv_i),
            (
                u_i[:, :, :1],
                v_i[:, :, :1],
                _plane_zeros(w_i),
                theta_i[:, :, :1],
                qv_i[:, :, :1],
            ),
            (
                u_i[:, :, -1:],
                v_i[:, :, -1:],
                _plane_zeros(w_i),
                theta_i[:, :, -1:],
                qv_i[:, :, -1:],
            ),
            axis_name,
            ndev,
        )
        filtered, dx, dy = ddxy_filter_many(
            (u_h, v_h, w_h, theta_h, qv_h), params, ops
        )
        u_h, v_h, w_h, theta_h, qv_h = filtered
        dudx_h, dvdx_h, dwdx_h, dtheta_dx_h, dqv_dx_h = dx
        dudy_h, dvdy_h, dwdy_h, dtheta_dy_h, dqv_dy_h = dy

    # Keep the two staggered representations used by the proven single-domain
    # path distinct.  The upper-face gradients drive rotational convection
    # and vertical shear stress; their face average drives the centred SGS
    # strain.  Treating a centred gradient as a face quantity destabilises the
    # first non-zero vertical velocity perturbation.
    dudz_face_h = _gradient_to_upper_halo(u_h, params)
    dvdz_face_h = _gradient_to_upper_halo(v_h, params)
    dwdz_h = ddz_w(w_h, params)
    if params.momentum_wall_model == "abl":
        dudz_face_h, dvdz_face_h = _apply_porte_agel_wall_correction_local(
            dudz_face_h,
            dvdz_face_h,
            axis_name,
            horizontal_average=params.horizontal_homogeneous,
        )
    txz0, tyz0, dudz0, dvdz0, ustar = _wall_stress_local(
        u_h, v_h, params, axis_name
    )
    dudz_h = 0.5 * (_shift_z_minus(dudz_face_h) + dudz_face_h)
    dvdz_h = 0.5 * (_shift_z_minus(dvdz_face_h) + dvdz_face_h)
    dudz_h = _set_rank0_plane(dudz_h, 1, dudz0, axis_name)
    dvdz_h = _set_rank0_plane(dvdz_h, 1, dvdz0, axis_name)
    dudz_h, dvdz_h, dwdz_h = _halos_from_interior_many(
        (dudz_h, dvdz_h, dwdz_h), axis_name, ndev
    )
    return (
        LocalGradientBundle(
            u=u_h,
            v=v_h,
            w=w_h,
            theta=theta_h,
            qv=qv_h,
            dudx=dudx_h,
            dudy=dudy_h,
            dudz=dudz_h,
            dudz_face=dudz_face_h,
            dvdx=dvdx_h,
            dvdy=dvdy_h,
            dvdz=dvdz_h,
            dvdz_face=dvdz_face_h,
            dwdx=dwdx_h,
            dwdy=dwdy_h,
            dwdz=dwdz_h,
            dtheta_dx=dtheta_dx_h,
            dtheta_dy=dtheta_dy_h,
            dqv_dx=dqv_dx_h,
            dqv_dy=dqv_dy_h,
        ),
        (txz0, tyz0, ustar),
    )


def _periodic_distance_local(
    coordinate: jax.Array, centre: float, length: float
) -> jax.Array:
    distance = jnp.abs(coordinate - jnp.asarray(centre, dtype=coordinate.dtype))
    return jnp.minimum(distance, jnp.asarray(length, dtype=coordinate.dtype) - distance)


def _periodic_offset_local(
    coordinate: jax.Array,
    centre: float,
    length: float,
) -> jax.Array:
    period = jnp.asarray(length, dtype=coordinate.dtype)
    half_period = 0.5 * period
    return jnp.mod(coordinate - centre + half_period, period) - half_period


def _wind_tunnel_sources_local(
    u_i: jax.Array,
    v_i: jax.Array,
    w_i: jax.Array,
    theta_i: jax.Array,
    qv_i: jax.Array,
    params: Params,
    axis_name: str,
    *,
    include_static_fringe: bool,
    adjoint_axis_name: str | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Shard-local forcing with only scalar reductions across the z mesh."""
    rank = lax.axis_index(axis_name)
    nz_local = u_i.shape[2]
    x = (
        (jnp.arange(params.nx, dtype=params.dtype) + 0.5)
        * params.dx
        * params.z_i
    )[:, None, None]
    y = (
        (jnp.arange(params.ny, dtype=params.dtype) + 0.5)
        * params.dy
        * params.z_i
    )[None, :, None]
    global_k = rank * nz_local + jnp.arange(nz_local, dtype=jnp.int32)
    z = ((global_k.astype(params.dtype) + 0.5) * params.dz * params.z_i)[
        None, None, :
    ]
    source_u = jnp.zeros_like(u_i)
    source_v = jnp.zeros_like(v_i)
    source_w = jnp.zeros_like(w_i)
    source_theta = jnp.zeros_like(theta_i)
    source_qv = jnp.zeros_like(qv_i)
    turbine_active = (
        jnp.asarray(1.0, dtype=params.dtype)
        if adjoint_axis_name is None
        else (lax.axis_index(adjoint_axis_name) == 1).astype(params.dtype)
    )

    if params.actuator_disk_enabled:
        lx = params.lx * params.z_i
        ly = params.ly * params.z_i
        dx = _periodic_offset_local(x, params.actuator_disk_x, lx)
        dy = _periodic_offset_local(y, params.actuator_disk_y, ly)
        yaw = jnp.deg2rad(
            jnp.asarray(params.actuator_disk_yaw_degrees, dtype=params.dtype)
        )
        normal_x = jnp.cos(yaw)
        normal_y = jnp.sin(yaw)
        normal_distance = dx * normal_x + dy * normal_y
        in_plane_distance = -dx * normal_y + dy * normal_x
        radius = jnp.sqrt(
            in_plane_distance**2 + (z - params.actuator_disk_z) ** 2
        )
        sigma_x = max(
            params.actuator_disk_thickness, 1.5 * params.dx * params.z_i
        )
        streamwise = jnp.exp(-0.5 * (normal_distance / sigma_x) ** 2)
        streamwise = streamwise / jnp.maximum(
            jnp.sum(streamwise[:, 0, 0]) * params.dx * params.z_i,
            jnp.asarray(jnp.finfo(params.dtype).tiny, dtype=params.dtype),
        )
        edge_width = max(
            0.5 * (params.dy + params.dz) * params.z_i,
            0.25 * params.actuator_disk_thickness,
        )
        disk = 0.5 * (
            1.0
            - jnp.tanh(
                (radius - 0.5 * params.actuator_disk_diameter) / edge_width
            )
        )
        if params.actuator_disk_hub_diameter > 0.0:
            disk = disk * 0.5 * (
                1.0
                + jnp.tanh(
                    (radius - 0.5 * params.actuator_disk_hub_diameter)
                    / edge_width
                )
            )
        disk = streamwise * disk
        normal_velocity = u_i * normal_x + v_i * normal_y
        weighted_velocity = lax.psum(jnp.sum(normal_velocity * disk), axis_name)
        disk_weight = lax.psum(jnp.sum(disk), axis_name)
        disk_velocity = weighted_velocity / jnp.maximum(
            disk_weight, jnp.asarray(jnp.finfo(u_i.dtype).tiny, dtype=u_i.dtype)
        )
        disk_acceleration = -turbine_active * (
            0.5
            * params.z_i
            * params.actuator_disk_ct_prime
            * disk_velocity
            * jnp.abs(disk_velocity)
            * disk
        )
        source_u = source_u + disk_acceleration * normal_x
        source_v = source_v + disk_acceleration * normal_y

    if params.cold_source_enabled:
        lx = params.lx * params.z_i
        ly = params.ly * params.z_i
        dx = _periodic_distance_local(x, params.cold_source_x, lx)
        dy = _periodic_distance_local(y, params.cold_source_y, ly)
        radial2 = dy * dy + (z - params.cold_source_z) ** 2
        sigma_x = max(
            params.cold_source_sigma_x, 1.5 * params.dx * params.z_i
        )
        sigma_r = max(
            params.cold_source_sigma_r,
            1.5 * max(params.dy, params.dz) * params.z_i,
        )
        kernel = jnp.exp(
            -0.5 * (dx / sigma_x) ** 2 - 0.5 * radial2 / sigma_r**2
        )
        cell_volume = params.dx * params.dy * params.dz * params.z_i**3
        normalization = lax.psum(jnp.sum(kernel), axis_name) * cell_volume
        kernel = kernel / jnp.maximum(
            normalization,
            jnp.asarray(jnp.finfo(params.dtype).tiny, dtype=params.dtype),
        )
        source_u = source_u + turbine_active * (
            params.z_i
            * params.cold_source_momentum_flux
            / params.cold_source_density
            * kernel
        )
        if params.thermo_enabled:
            source_theta = source_theta - turbine_active * (
                params.z_i
                * params.cold_source_cooling_power
                / (
                    params.cold_source_density
                    * params.cold_source_heat_capacity
                )
                * kernel
            )

    if include_static_fringe and params.fringe_enabled:
        domain_x = params.lx * params.z_i
        mask = classic_fringe_window(
            x,
            params.fringe_start_x,
            domain_x,
        )
        rate = params.z_i / params.fringe_timescale
        source_u = source_u + turbine_active * rate * mask * (params.fringe_target_u - u_i)
        source_v = source_v + turbine_active * rate * mask * (params.fringe_target_v - v_i)
        source_w = source_w - turbine_active * rate * mask * w_i
        if params.thermo_enabled:
            target_theta = (
                params.theta0
                if params.fringe_target_theta is None
                else params.fringe_target_theta
            )
            source_theta = source_theta + turbine_active * rate * mask * (target_theta - theta_i)
            source_qv = source_qv + turbine_active * rate * mask * (params.qv0 - qv_i)
    return source_u, source_v, source_w, source_theta, source_qv


def make_concurrent_fringe_sources_sharded(
    params: Params,
    mesh: Mesh,
    axis_name: str = "z",
    adjoint_axis_name: str | None = None,
):
    if not params.fringe_enabled:
        raise ValueError("Concurrent precursor coupling requires fringe_enabled=true")
    x_centres = (
        (np.arange(params.nx, dtype=np.float64) + 0.5)
        * params.dx
        * params.z_i
    )
    fringe_start_index = int(np.searchsorted(x_centres, params.fringe_start_x))
    if fringe_start_index >= params.nx:
        raise ValueError("Concurrent fringe contains no cell centres")

    def local_sources(
        u_i,
        v_i,
        w_i,
        theta_i,
        qv_i,
        target_u_i,
        target_v_i,
        target_w_i,
        target_theta_i,
        target_qv_i,
    ):
        x = (
            (jnp.arange(fringe_start_index, params.nx, dtype=params.dtype) + 0.5)
            * params.dx
            * params.z_i
        )[:, None, None]
        domain_x = params.lx * params.z_i
        mask = classic_fringe_window(
            x,
            params.fringe_start_x,
            domain_x,
        )
        rate = params.z_i / params.fringe_timescale
        def expand(source_slice, reference):
            return jnp.zeros_like(reference).at[fringe_start_index:, :, :].set(
                source_slice
            )

        turbine_active = (
            jnp.asarray(1.0, dtype=params.dtype)
            if adjoint_axis_name is None
            else (lax.axis_index(adjoint_axis_name) == 1).astype(params.dtype)
        )
        return (
            turbine_active * expand(rate * mask * (target_u_i - u_i[fringe_start_index:]), u_i),
            turbine_active * expand(rate * mask * (target_v_i - v_i[fringe_start_index:]), v_i),
            turbine_active * expand(rate * mask * (target_w_i - w_i[fringe_start_index:]), w_i),
            turbine_active * expand(
                rate * mask * (target_theta_i - theta_i[fringe_start_index:]),
                theta_i,
            ),
            turbine_active * expand(rate * mask * (target_qv_i - qv_i[fringe_start_index:]), qv_i),
        )

    z = z_slab_spec(axis_name)
    mapped_local_sources = local_sources
    additional = ()
    if adjoint_axis_name is not None:
        z = adjoint_z_slab_spec(adjoint_axis_name, axis_name)
        mapped_local_sources = jax.vmap(local_sources)
        additional = (adjoint_axis_name,)
    return _shard_map(
        mapped_local_sources,
        mesh=mesh,
        in_specs=(z,) * 10,
        out_specs=(z,) * 5,
        axis_name=axis_name,
        additional_axis_names=additional,
    )


def make_momentum_rhs_sharded(
    params: Params,
    ops: Operators,
    mesh: Mesh,
    axis_name: str = "z",
    *,
    concurrent_fringe: bool = False,
    adjoint_axis_name: str | None = None,
):
    ndev = mesh_size(mesh, axis_name)

    def local_rhs(
        u_i: jax.Array,
        v_i: jax.Array,
        w_i: jax.Array,
        theta_i: jax.Array,
        qv_i: jax.Array,
        cs2_i: jax.Array,
        lm_i: jax.Array,
        mm_i: jax.Array,
        qn_i: jax.Array,
        nn_i: jax.Array,
        scalar_c_i: jax.Array,
        u_lag_i: jax.Array,
        v_lag_i: jax.Array,
        w_lag_i: jax.Array,
        step: jax.Array,
    ) -> tuple[jax.Array, ...]:
        gradients, wall = _build_gradient_bundle_local(
            u_i,
            v_i,
            w_i,
            theta_i if params.thermo_enabled else None,
            qv_i if params.thermo_enabled and params.moisture_enabled else None,
            params,
            ops,
            axis_name,
            ndev,
        )
        u_h, v_h, w_h = gradients.u, gradients.v, gradients.w
        dudx_h, dudy_h, dudz_h = (
            gradients.dudx,
            gradients.dudy,
            gradients.dudz,
        )
        dudz_face_h = gradients.dudz_face
        dvdx_h, dvdy_h, dvdz_h = (
            gradients.dvdx,
            gradients.dvdy,
            gradients.dvdz,
        )
        dvdz_face_h = gradients.dvdz_face
        dwdx_h, dwdy_h, dwdz_h = (
            gradients.dwdx,
            gradients.dwdy,
            gradients.dwdz,
        )
        txz0, tyz0, _ = wall

        cx_h, cy_h, cz_h = _convec_halo(
            u_h,
            v_h,
            w_h,
            dudy_h,
            dudz_face_h,
            dvdx_h,
            dvdz_face_h,
            dwdx_h,
            dwdy_h,
            axis_name,
            ndev,
        )
        if params.sgs_model == "lasd":
            sij_uv = _strain_uv_halo(
                _to_sgs(dudx_h, params),
                _to_sgs(dudy_h, params),
                _to_sgs(dudz_h, params),
                _to_sgs(dvdx_h, params),
                _to_sgs(dvdy_h, params),
                _to_sgs(dvdz_h, params),
                _to_sgs(dwdx_h, params),
                _to_sgs(dwdy_h, params),
                _to_sgs(dwdz_h, params),
                axis_name,
            )
            sij_w = _strain_w_halo(
                _to_sgs(dudx_h, params),
                _to_sgs(dudy_h, params),
                _to_sgs(dudz_h, params),
                _to_sgs(dvdx_h, params),
                _to_sgs(dvdy_h, params),
                _to_sgs(dvdz_h, params),
                _to_sgs(dwdx_h, params),
                _to_sgs(dwdy_h, params),
                _to_sgs(dwdz_h, params),
                dudz_face=_to_sgs(dudz_face_h, params),
                dvdz_face=_to_sgs(dvdz_face_h, params),
            )
            (
                cs2_h,
                lm_h,
                mm_h,
                qn_h,
                nn_h,
                u_lag_h,
                v_lag_h,
                w_lag_h,
            ) = _update_lasd_coefficients_halo(
                u_h,
                v_h,
                w_h,
                sij_uv,
                cs2_i,
                lm_i,
                mm_i,
                qn_i,
                nn_i,
                u_lag_i,
                v_lag_i,
                w_lag_i,
                step,
                params,
                axis_name,
                ndev,
            )
            txx_h, txy_h, txz_h, tyy_h, tyz_h, tzz_h = _stress_from_cs2_halo(cs2_h, sij_uv, sij_w, params, axis_name)
            sgs_state = (
                cs2_h[:, :, 1:-1],
                lm_h[:, :, 1:-1],
                mm_h[:, :, 1:-1],
                qn_h[:, :, 1:-1],
                nn_h[:, :, 1:-1],
                u_lag_h[:, :, 1:-1],
                v_lag_h[:, :, 1:-1],
                w_lag_h[:, :, 1:-1],
            )
        else:
            txx_h, txy_h, txz_h, tyy_h, tyz_h, tzz_h = classic_smagorinsky(
                dudx_h,
                dudy_h,
                dudz_h,
                dvdx_h,
                dvdy_h,
                dvdz_h,
                dwdx_h,
                dwdy_h,
                dwdz_h,
                params,
                dudz_face=dudz_face_h,
                dvdz_face=dvdz_face_h,
            )
            sgs_state = (
                _to_sgs(cs2_i, params),
                _to_sgs(lm_i, params),
                _to_sgs(mm_i, params),
                _to_sgs(qn_i, params),
                _to_sgs(nn_i, params),
                _to_sgs(u_lag_i, params),
                _to_sgs(v_lag_i, params),
                _to_sgs(w_lag_i, params),
            )
        if params.molecular_viscosity_internal > 0.0:
            nu = jnp.asarray(
                params.molecular_viscosity_internal, dtype=txx_h.dtype
            )
            txx_h = txx_h - 2.0 * nu * dudx_h
            txy_h = txy_h - nu * (dudy_h + dvdx_h)
            txz_h = txz_h - nu * (dudz_face_h + dwdx_h)
            tyy_h = tyy_h - 2.0 * nu * dvdy_h
            tyz_h = tyz_h - nu * (dvdz_face_h + dwdy_h)
            tzz_h = tzz_h - 2.0 * nu * dwdz_h
        txz_h = _set_top_rank_plane(txz_h, -2, 0.0, axis_name, ndev)
        tyz_h = _set_top_rank_plane(tyz_h, -2, 0.0, axis_name, ndev)
        txz_h, tyz_h, tzz_h = _halos_from_interior_many(
            (txz_h, tyz_h, tzz_h), axis_name, ndev
        )
        txz_h = _set_rank0_plane(txz_h, 0, txz0.astype(txz_h.dtype), axis_name)
        tyz_h = _set_rank0_plane(tyz_h, 0, tyz0.astype(tyz_h.dtype), axis_name)

        txy_dx_h, txy_dy_h = gradxy(txy_h, params, ops)
        divtx_h = ddx(txx_h, params, ops) + txy_dy_h + ddz_w(txz_h, params)
        divty_h = txy_dx_h + ddy(tyy_h, params, ops) + ddz_w(tyz_h, params)
        divtz_h = (
            ddx(txz_h, params, ops)
            + ddy(tyz_h, params, ops)
            + _gradient_to_upper_halo(tzz_h, params)
        )
        divtz_h = _set_top_rank_plane(divtz_h, -2, 0.0, axis_name, ndev)

        rhs_u_h = _assemble_rhs_inner_local(cx_h, divtx_h, params, True, axis_name)
        rhs_v_h = _assemble_rhs_inner_local(cy_h, divty_h, params, False, axis_name)
        rhs_u_h, rhs_v_h = add_coriolis_geostrophic_forcing_inner(
            rhs_u_h,
            rhs_v_h,
            u_h[:, :, 1:-1],
            v_h[:, :, 1:-1],
            params,
        )
        rhs_w = _assemble_rhs_inner_local(
            cz_h, divtz_h, params, False, axis_name
        )
        if params.thermo_enabled:
            (
                theta,
                qv,
                rhs_theta,
                rhs_qv,
                buoyancy,
                scalar_c,
            ) = _scalar_rhs_from_bundle_local(
                gradients, sgs_state[0], params, ops, axis_name, ndev
            )
            rhs_w = rhs_w + buoyancy
        else:
            theta = theta_i
            qv = qv_i
            rhs_theta = jnp.zeros_like(theta_i)
            rhs_qv = jnp.zeros_like(qv_i)
            scalar_c = scalar_c_i
        source_u, source_v, source_w, source_theta, source_qv = (
            _wind_tunnel_sources_local(
                u_h[:, :, 1:-1],
                v_h[:, :, 1:-1],
                w_h[:, :, 1:-1],
                theta,
                qv,
                params,
                axis_name,
                include_static_fringe=not concurrent_fringe,
                adjoint_axis_name=adjoint_axis_name,
            )
        )
        rhs_u_h = rhs_u_h + source_u
        rhs_v_h = rhs_v_h + source_v
        rhs_w = rhs_w + source_w
        rhs_theta = rhs_theta + source_theta
        rhs_qv = rhs_qv + source_qv
        return (
            u_h[:, :, 1:-1],
            v_h[:, :, 1:-1],
            w_h[:, :, 1:-1],
            theta,
            qv,
            rhs_u_h,
            rhs_v_h,
            rhs_w,
            rhs_theta,
            rhs_qv,
            *sgs_state,
            scalar_c,
        )

    z = z_slab_spec(axis_name)
    z_scalar = P(None, None, axis_name, None)
    step_spec = P()
    mapped_local_rhs = local_rhs
    additional = ()
    if adjoint_axis_name is not None:
        z = adjoint_z_slab_spec(adjoint_axis_name, axis_name)
        z_scalar = P(adjoint_axis_name, None, None, axis_name, None)
        step_spec = P(adjoint_axis_name)
        mapped_local_rhs = jax.vmap(local_rhs)
        additional = (adjoint_axis_name,)
    return _shard_map(
        mapped_local_rhs,
        mesh=mesh,
        in_specs=(
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z_scalar,
            z,
            z,
            z,
            step_spec,
        ),
        out_specs=(
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z_scalar,
        ),
        axis_name=axis_name,
        additional_axis_names=additional,
    )


def _scalar_rhs_from_bundle_local(
    gradients: LocalGradientBundle,
    cs2_i: jax.Array,
    params: Params,
    ops: Operators,
    axis_name: str,
    ndev: int,
) -> tuple[jax.Array, ...]:
    theta_h = gradients.theta
    qv_h = gradients.qv
    u_h = gradients.u
    v_h = gradients.v
    w_h = gradients.w
    sij = _strain_uv_halo(
        _to_sgs(gradients.dudx, params),
        _to_sgs(gradients.dudy, params),
        _to_sgs(gradients.dudz, params),
        _to_sgs(gradients.dvdx, params),
        _to_sgs(gradients.dvdy, params),
        _to_sgs(gradients.dvdz, params),
        _to_sgs(gradients.dwdx, params),
        _to_sgs(gradients.dwdy, params),
        _to_sgs(gradients.dwdz, params),
        axis_name,
    )
    strain_mag = _strain_magnitude(sij).astype(params.dtype)
    cs2_h = _copy_boundary_halo(cs2_i.astype(params.dtype), axis_name, ndev)
    rank = lax.axis_index(axis_name)

    def transport_rhs(
        phi_h: jax.Array,
        dphi_dx_h: jax.Array,
        dphi_dy_h: jax.Array,
        turbulent_number: float,
        surface_flux: float,
        top_gradient_physical: float,
    ) -> jax.Array:
        scalar_coeff_h = cs2_h / turbulent_number
        kappa_h = (
            scalar_coeff_h * params.sgs_delta * params.sgs_delta * strain_mag
            + params.molecular_diffusivity_internal
        )
        kappa_h = kappa_h * _physical_mask_halo(kappa_h)

        adv_x_h = ddx(u_h * phi_h, params, ops)
        adv_y_h = ddy(v_h * phi_h, params, ops)
        phi_upper_h = 0.5 * (phi_h + _shift_z_plus(phi_h))
        adv_z_upper_h = w_h * phi_upper_h
        adv_z_lower_h = _shift_z_minus(adv_z_upper_h)
        adv_div_h = (
            adv_x_h
            + adv_y_h
            + (adv_z_upper_h - adv_z_lower_h) / params.dz
        )

        dphi_dz_upper_h = (_shift_z_plus(phi_h) - phi_h) / params.dz
        kappa_upper_h = 0.5 * (kappa_h + _shift_z_plus(kappa_h))
        qx_h = -kappa_h * dphi_dx_h
        qy_h = -kappa_h * dphi_dy_h
        qz_upper_h = -kappa_upper_h * dphi_dz_upper_h
        qz_upper_h = jnp.where(
            rank == ndev - 1,
            qz_upper_h.at[:, :, -2].set(
                -kappa_upper_h[:, :, -2]
                * jnp.asarray(
                    top_gradient_physical * params.z_i,
                    dtype=params.dtype,
                )
            ),
            qz_upper_h,
        )
        qz_lower_h = _shift_z_minus(qz_upper_h)
        qz_lower_h = jnp.where(
            rank == 0,
            qz_lower_h.at[:, :, 1].set(
                jnp.asarray(surface_flux, dtype=params.dtype)
            ),
            qz_lower_h,
        )
        diff_div_h = (
            ddx(qx_h, params, ops)
            + ddy(qy_h, params, ops)
            + (qz_upper_h - qz_lower_h) / params.dz
        )
        return (-adv_div_h - diff_div_h)[:, :, 1:-1].astype(params.dtype)

    theta_top_gradient = (
        0.0 if params.theta_top_gradient is None else params.theta_top_gradient
    )
    rhs_theta = transport_rhs(
        theta_h,
        gradients.dtheta_dx,
        gradients.dtheta_dy,
        params.prandtl_t,
        params.surface_theta_flux,
        theta_top_gradient,
    )
    rhs_qv = jnp.zeros_like(rhs_theta)
    if params.moisture_enabled:
        rhs_qv = transport_rhs(
            qv_h,
            gradients.dqv_dx,
            gradients.dqv_dy,
            params.schmidt_t,
            params.transported_surface_qv_flux,
            0.0,
        )

    if params.buoyancy_reference == "ambient":
        qv_for_buoyancy = qv_h if params.moisture_enabled else jnp.zeros_like(qv_h)
        theta_v_anomaly_h = (
            theta_h * (1.0 + 0.61 * qv_for_buoyancy) - params.theta_v0
        )
    else:
        theta_base = jnp.asarray(params.theta0, dtype=theta_h.dtype)
        theta_anomaly_h = theta_h - theta_base
        theta_prime_h = theta_anomaly_h - jnp.mean(
            theta_anomaly_h, axis=(0, 1), keepdims=True
        )
        if params.moisture_enabled:
            qv_base = jnp.asarray(params.qv0, dtype=qv_h.dtype)
            qv_anomaly_h = qv_h - qv_base
            qv_prime_h = qv_anomaly_h - jnp.mean(
                qv_anomaly_h, axis=(0, 1), keepdims=True
            )
            qv_mean_h = qv_h - qv_prime_h
            theta_mean_h = theta_h - theta_prime_h
            covariance_h = theta_prime_h * qv_prime_h
            theta_v_anomaly_h = (
                theta_prime_h * (1.0 + 0.61 * qv_mean_h)
                + 0.61 * theta_mean_h * qv_prime_h
                + 0.61
                * (
                    covariance_h
                    - jnp.mean(covariance_h, axis=(0, 1), keepdims=True)
                )
            )
        else:
            theta_v_anomaly_h = theta_prime_h
    theta_v_face_anomaly_h = 0.5 * (
        theta_v_anomaly_h + _shift_z_plus(theta_v_anomaly_h)
    )
    buoyancy = (
        params.z_i
        * params.g
        / params.theta_v0
        * theta_v_face_anomaly_h[:, :, 1:-1]
    )
    buoyancy = jnp.where(
        rank == ndev - 1,
        buoyancy.at[:, :, -1].set(0.0),
        buoyancy,
    ).astype(params.dtype)
    scalar_c = jnp.stack(
        (cs2_i / params.prandtl_t, cs2_i / params.schmidt_t), axis=-1
    ).astype(params.sgs_dtype)
    return (
        theta_h[:, :, 1:-1],
        qv_h[:, :, 1:-1],
        rhs_theta,
        rhs_qv,
        buoyancy,
        scalar_c,
    )


def make_scalar_rhs_buoyancy_sharded(
    params: Params,
    ops: Operators,
    mesh: Mesh,
    axis_name: str = "z",
):
    """Centered conservative moist-scalar transport on distributed z slabs."""

    ndev = mesh_size(mesh, axis_name)

    def local_rhs(
        u_i: jax.Array,
        v_i: jax.Array,
        w_i: jax.Array,
        theta_i: jax.Array,
        qv_i: jax.Array,
        cs2_i: jax.Array,
    ) -> tuple[jax.Array, ...]:
        gradients, _ = _build_gradient_bundle_local(
            u_i, v_i, w_i, theta_i, qv_i, params, ops, axis_name, ndev
        )
        return _scalar_rhs_from_bundle_local(
            gradients, cs2_i, params, ops, axis_name, ndev
        )

    z = z_slab_spec(axis_name)
    z_scalar = P(None, None, axis_name, None)
    return _shard_map(
        local_rhs,
        mesh=mesh,
        in_specs=(z, z, z, z, z, z),
        out_specs=(z, z, z, z, z, z_scalar),
        axis_name=axis_name,
    )


def make_project_velocity_sharded(
    params: Params,
    pressure_ops: ShardedPressureOperators,
    mesh: Mesh,
    axis_name: str = "z",
    spike_ops: ShardedSpikeOperators | None = None,
    adjoint_axis_name: str | None = None,
):
    ndev = mesh_size(mesh, axis_name)
    pressure_solver = (
        make_pressure_hat_solver_z_sharded(
            params, pressure_ops, mesh, axis_name, adjoint_axis_name
        )
        if params.sharded_pressure_solver == "transpose"
        else None
    )
    spike_solver = (
        make_pressure_hat_solver_z_sharded_spike(params, mesh, axis_name)
        if params.sharded_pressure_solver == "spike"
        else None
    )
    if params.sharded_pressure_solver == "spike" and spike_ops is None:
        raise ValueError("SPIKE pressure projection requires ShardedSpikeOperators.")
    if adjoint_axis_name is not None and params.sharded_pressure_solver == "spike":
        raise ValueError("The adjoint dimension currently requires transpose pressure projection.")

    def local_divergence_hat(
        u_i: jax.Array,
        v_i: jax.Array,
        w_i: jax.Array,
        kx: jax.Array,
        ky: jax.Array,
    ) -> jax.Array:
        w_i = _set_w_physical_boundaries(w_i, axis_name, ndev)
        w_h = _zero_boundary_halo(w_i, axis_name, ndev)
        dwdz_i = ((w_h[:, :, 1:-1] - w_h[:, :, :-2]) / params.dz).astype(params.dtype)
        u_hat = rfft2_fortran_layout(u_i)
        v_hat = rfft2_fortran_layout(v_i)
        dwdz_hat = rfft2_fortran_layout(dwdz_i)
        return (
            1j * kx.astype(u_hat.real.dtype) * u_hat
            + 1j * ky.astype(v_hat.real.dtype) * v_hat
            + dwdz_hat
        )

    def local_correct(
        u_i: jax.Array,
        v_i: jax.Array,
        w_i: jax.Array,
        p_i: jax.Array,
        dpdx_i: jax.Array,
        dpdy_i: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        p_h = _copy_boundary_halo(p_i, axis_name, ndev)
        dpdz_i = ((p_h[:, :, 2:] - p_h[:, :, 1:-1]) / params.dz).astype(params.dtype)
        rank = lax.axis_index(axis_name)
        dpdz_i = jnp.where(rank == ndev - 1, dpdz_i.at[:, :, -1].set(0.0), dpdz_i)
        u_new = u_i - params.dt * dpdx_i
        v_new = v_i - params.dt * dpdy_i
        w_new = w_i - params.dt * dpdz_i
        w_new = _set_w_physical_boundaries(w_new, axis_name, ndev)
        return u_new.astype(params.dtype), v_new.astype(params.dtype), w_new.astype(params.dtype), p_i.astype(params.dtype)

    z = z_slab_spec(axis_name)
    mapped_divergence = local_divergence_hat
    mapped_correct = local_correct
    additional = ()
    if adjoint_axis_name is not None:
        z = adjoint_z_slab_spec(adjoint_axis_name, axis_name)
        mapped_divergence = jax.vmap(
            local_divergence_hat, in_axes=(0, 0, 0, None, None)
        )
        mapped_correct = jax.vmap(local_correct)
        additional = (adjoint_axis_name,)
    div_hat_fn = _shard_map(
        mapped_divergence,
        mesh=mesh,
        in_specs=(z, z, z, P(), P()),
        out_specs=z,
        axis_name=axis_name,
        additional_axis_names=additional,
    )
    correct_fn = _shard_map(
        mapped_correct,
        mesh=mesh,
        in_specs=(z, z, z, z, z, z),
        out_specs=(z, z, z, z),
        axis_name=axis_name,
        additional_axis_names=additional,
    )

    def project(
        u_i: jax.Array,
        v_i: jax.Array,
        w_i: jax.Array,
        runtime_pressure_ops: ShardedPressureOperators | None = None,
        runtime_spike_ops: ShardedSpikeOperators | None = None,
    ) -> tuple[jax.Array, ...]:
        active_ops = pressure_ops if runtime_pressure_ops is None else runtime_pressure_ops
        div_hat = div_hat_fn(u_i, v_i, w_i, active_ops.kx, active_ops.ky)
        if params.sharded_pressure_solver == "spike":
            active_spike = spike_ops if runtime_spike_ops is None else runtime_spike_ops
            if active_spike is None:  # pragma: no cover - guarded at construction
                raise ValueError("Missing runtime SPIKE pressure operators.")
            p_hat = spike_solver(div_hat / params.dt, active_spike)
        else:
            p_hat = pressure_solver(div_hat / params.dt, active_ops)
        p_i, dpdx_i, dpdy_i = pressure_and_gradients_from_hat_z_sharded(
            p_hat, params, active_ops
        )
        return correct_fn(u_i, v_i, w_i, p_i, dpdx_i, dpdy_i)

    return project


def make_apply_moisture_bounds_sharded(
    mesh: Mesh,
    axis_name: str = "z",
    adjoint_axis_name: str | None = None,
):
    """Return a globally conservative positivity correction for distributed qv."""

    def local_bounds(qv_i: jax.Array, floor: jax.Array) -> jax.Array:
        floor = jnp.asarray(floor, dtype=qv_i.dtype)
        shifted = qv_i - floor
        positive = jnp.maximum(shifted, 0.0)
        negative_mass = lax.psum(
            -jnp.sum(jnp.minimum(shifted, 0.0)), axis_name
        )
        positive_mass = lax.psum(jnp.sum(positive), axis_name)
        scale = jnp.where(
            positive_mass > negative_mass,
            (positive_mass - negative_mass)
            / jnp.maximum(positive_mass, jnp.asarray(1.0e-30, qv_i.dtype)),
            0.0,
        )
        return floor + positive * scale

    z = z_slab_spec(axis_name)
    mapped_bounds = local_bounds
    floor_spec = P()
    additional = ()
    if adjoint_axis_name is not None:
        z = adjoint_z_slab_spec(adjoint_axis_name, axis_name)
        mapped_bounds = jax.vmap(local_bounds, in_axes=(0, None))
        additional = (adjoint_axis_name,)
    return _shard_map(
        mapped_bounds,
        mesh=mesh,
        in_specs=(z, floor_spec),
        out_specs=z,
        axis_name=axis_name,
        additional_axis_names=additional,
    )


def make_apply_rayleigh_sponge_sharded(
    params: Params,
    mesh: Mesh,
    axis_name: str = "z",
    adjoint_axis_name: str | None = None,
):
    """Return the exact exponential top sponge on distributed z slabs."""

    def local_sponge(
        u_i: jax.Array, v_i: jax.Array, w_i: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if not params.sponge_enabled:
            return u_i, v_i, w_i
        rank = lax.axis_index(axis_name)
        nz_local = u_i.shape[2]
        global_k = rank * nz_local + jnp.arange(nz_local, dtype=params.dtype)
        center_z = (global_k + 0.5) * params.dz * params.z_i
        face_z = (global_k + 1.0) * params.dz * params.z_i
        top = params.lz * params.z_i
        depth = max(
            top - params.sponge_start_height, params.dz * params.z_i
        )

        def decay(z: jax.Array, dtype: jnp.dtype) -> jax.Array:
            eta = jnp.clip(
                (z - params.sponge_start_height) / depth, 0.0, 1.0
            )
            strength_dt = (
                params.dt_physical / params.sponge_timescale
            ) * eta**params.sponge_power
            return jnp.exp(-strength_dt).astype(dtype)[None, None, :]

        center_decay = decay(center_z, u_i.dtype)
        face_decay = decay(face_z, w_i.dtype)
        if params.sponge_target == "plane_mean":
            target_u = jnp.mean(u_i, axis=(0, 1), keepdims=True)
            target_v = jnp.mean(v_i, axis=(0, 1), keepdims=True)
        else:
            target_u = jnp.asarray(params.geostrophic_u, dtype=u_i.dtype)
            target_v = jnp.asarray(params.geostrophic_v, dtype=v_i.dtype)
        u_new = target_u + (u_i - target_u) * center_decay
        v_new = target_v + (v_i - target_v) * center_decay.astype(v_i.dtype)
        if params.sponge_target == "plane_mean":
            u_new = u_new + target_u - jnp.mean(
                u_new, axis=(0, 1), keepdims=True
            )
            v_new = v_new + target_v - jnp.mean(
                v_new, axis=(0, 1), keepdims=True
            )
        return u_new, v_new, w_i * face_decay

    z = z_slab_spec(axis_name)
    mapped_sponge = local_sponge
    additional = ()
    if adjoint_axis_name is not None:
        z = adjoint_z_slab_spec(adjoint_axis_name, axis_name)
        mapped_sponge = jax.vmap(local_sponge)
        additional = (adjoint_axis_name,)
    return _shard_map(
        mapped_sponge,
        mesh=mesh,
        in_specs=(z, z, z),
        out_specs=(z, z, z),
        axis_name=axis_name,
        additional_axis_names=additional,
    )


def make_horizontal_filter_sharded(
    params: Params,
    ops: Operators,
    mesh: Mesh,
    axis_name: str = "z",
    adjoint_axis_name: str | None = None,
):
    """Apply the configured horizontal cutoff to complete distributed fields."""

    def local_filter(*fields: jax.Array) -> tuple[jax.Array, ...]:
        return horizontal_filter_many(fields, params, ops)

    z = z_slab_spec(axis_name)
    mapped_filter = local_filter
    additional = ()
    if adjoint_axis_name is not None:
        z = adjoint_z_slab_spec(adjoint_axis_name, axis_name)
        mapped_filter = jax.vmap(local_filter)
        additional = (adjoint_axis_name,)
    return _shard_map(
        mapped_filter,
        mesh=mesh,
        in_specs=(z,) * 5,
        out_specs=(z,) * 5,
        axis_name=axis_name,
        additional_axis_names=additional,
    )


def make_mean_u_at_height_sharded(
    params: Params,
    mesh: Mesh,
    height: float,
    axis_name: str = "z",
    adjoint_axis_name: str | None = None,
):
    """Return a scalar horizontal mean, linearly interpolated in z."""

    dz_physical = params.dz * params.z_i
    coordinate = height / dz_physical - 0.5
    lower = int(np.clip(np.floor(coordinate), 0, params.nz - 1))
    upper = min(lower + 1, params.nz - 1)
    alpha = float(np.clip(coordinate - lower, 0.0, 1.0))
    if lower == upper:
        alpha = 0.0

    def local_probe(u_i: jax.Array) -> jax.Array:
        rank = lax.axis_index(axis_name)
        nz_local = u_i.shape[2]
        global_k = rank * nz_local + jnp.arange(nz_local)
        weights = (
            (1.0 - alpha) * (global_k == lower).astype(u_i.dtype)
            + alpha * (global_k == upper).astype(u_i.dtype)
        )
        local_sum = jnp.sum(u_i * weights[None, None, :])
        return lax.psum(local_sum, axis_name) / jnp.asarray(
            params.nx * params.ny, dtype=u_i.dtype
        )

    z = z_slab_spec(axis_name)
    out_spec = P()
    mapped_probe = local_probe
    additional = ()
    if adjoint_axis_name is not None:
        z = adjoint_z_slab_spec(adjoint_axis_name, axis_name)
        out_spec = P(adjoint_axis_name)
        mapped_probe = jax.vmap(local_probe)
        additional = (adjoint_axis_name,)
    return _shard_map(
        mapped_probe,
        mesh=mesh,
        in_specs=z,
        out_specs=out_spec,
        axis_name=axis_name,
        additional_axis_names=additional,
    )


def make_step_ab2_sharded(
    params: Params,
    ops: ShardedOperators,
    mesh: Mesh,
    axis_name: str = "z",
    *,
    concurrent_fringe: bool = False,
    adjoint_axis_name: str | None = None,
):
    _validate_sharded_params(params, mesh, axis_name)
    momentum_rhs = make_momentum_rhs_sharded(
        params,
        ops.horizontal,
        mesh,
        axis_name,
        concurrent_fringe=concurrent_fringe,
        adjoint_axis_name=adjoint_axis_name,
    )
    concurrent_fringe_sources = (
        make_concurrent_fringe_sources_sharded(
            params, mesh, axis_name, adjoint_axis_name
        )
        if concurrent_fringe
        else None
    )
    moisture_bounds = make_apply_moisture_bounds_sharded(
        mesh, axis_name, adjoint_axis_name
    )
    sponge = make_apply_rayleigh_sponge_sharded(
        params, mesh, axis_name, adjoint_axis_name
    )
    post_filter = make_horizontal_filter_sharded(
        params, ops.horizontal, mesh, axis_name, adjoint_axis_name
    )
    project = make_project_velocity_sharded(
        params,
        ops.pressure,
        mesh,
        axis_name,
        ops.pressure_spike,
        adjoint_axis_name,
    )

    def step(
        state: ShardedFlowState,
        runtime_pressure_ops: ShardedPressureOperators | None = None,
        runtime_spike_ops: ShardedSpikeOperators | None = None,
        fringe_target: tuple[jax.Array, ...] | None = None,
    ) -> ShardedFlowState:
        (
            u,
            v,
            w,
            theta,
            qv,
            rhs_u,
            rhs_v,
            rhs_w,
            rhs_theta,
            rhs_qv,
            cs2,
            lm_old,
            mm_old,
            qn_old,
            nn_old,
            u_lag,
            v_lag,
            w_lag,
            scalar_c,
        ) = momentum_rhs(
            state.u,
            state.v,
            state.w,
            state.theta,
            state.qv,
            state.cs2,
            state.lm_old,
            state.mm_old,
            state.qn_old,
            state.nn_old,
            state.scalar_c,
            state.u_lag,
            state.v_lag,
            state.w_lag,
            state.step,
        )
        if concurrent_fringe:
            if fringe_target is None:
                raise ValueError("Concurrent sharded step requires a precursor fringe target")
            source_u, source_v, source_w, source_theta, source_qv = (
                concurrent_fringe_sources(
                    u,
                    v,
                    w,
                    theta,
                    qv,
                    *fringe_target,
                )
            )
            rhs_u = rhs_u + source_u
            rhs_v = rhs_v + source_v
            rhs_w = rhs_w + source_w
            rhs_theta = rhs_theta + source_theta
            rhs_qv = rhs_qv + source_qv
        u_star = _ab_update_inner(u, rhs_u, state.rhs_u_prev, state.step, params)
        v_star = _ab_update_inner(v, rhs_v, state.rhs_v_prev, state.step, params)
        w_star = _ab_update_inner(w, rhs_w, state.rhs_w_prev, state.step, params)
        theta_new = _ab_update_inner(
            theta, rhs_theta, state.rhs_theta_prev, state.step, params
        )
        qv_new = _ab_update_inner(
            qv, rhs_qv, state.rhs_qv_prev, state.step, params
        )
        u_star, v_star, w_star = sponge(u_star, v_star, w_star)
        u_star, v_star, w_star, theta_new, qv_new = post_filter(
            u_star, v_star, w_star, theta_new, qv_new
        )
        if params.moisture_enabled:
            qv_new = moisture_bounds(
                qv_new, jnp.asarray(params.qv_floor, dtype=params.dtype)
            )
        u_new, v_new, w_new, p_new = project(
            u_star,
            v_star,
            w_star,
            runtime_pressure_ops,
            runtime_spike_ops,
        )
        return ShardedFlowState(
            u=u_new,
            v=v_new,
            w=w_new,
            p=p_new,
            theta=theta_new,
            qv=qv_new,
            rhs_u_prev=rhs_u,
            rhs_v_prev=rhs_v,
            rhs_w_prev=rhs_w,
            rhs_theta_prev=rhs_theta,
            rhs_qv_prev=rhs_qv,
            lm_old=lm_old,
            mm_old=mm_old,
            qn_old=qn_old,
            nn_old=nn_old,
            cs2=cs2,
            scalar_c=scalar_c,
            u_lag=u_lag,
            v_lag=v_lag,
            w_lag=w_lag,
            step=state.step + jnp.array(1, dtype=state.step.dtype),
        )

    return step


def make_diagnostics_sharded(
    params: Params,
    ops: Operators,
    mesh: Mesh,
    axis_name: str = "z",
):
    ndev = mesh_size(mesh, axis_name)

    def local_diag(
        u_i: jax.Array,
        v_i: jax.Array,
        w_i: jax.Array,
        theta_i: jax.Array,
        qv_i: jax.Array,
        step: jax.Array,
    ) -> Diagnostics:
        u_h, v_h, w_h = _velocity_halos(u_i, v_i, w_i, axis_name, ndev)
        _, _, _, _, ustar = _wall_stress_local(u_h, v_h, params, axis_name)
        ustar_sum = lax.psum(jnp.sum(ustar), axis_name)
        ustar_count = params.nx * params.ny
        z_local = u_i.shape[2]
        valid_f = jnp.ones((1, 1, z_local), dtype=params.dtype)

        w_uv = 0.5 * (w_h[:, :, :-2] + w_h[:, :, 1:-1])
        ke_local = jnp.max(jnp.where(valid_f > 0.0, u_i * u_i + v_i * v_i + w_uv * w_uv, 0.0))
        div_h = ddx(u_h, params, ops) + ddy(v_h, params, ops) + ddz_w(w_h, params)
        div_local = jnp.max(jnp.where(valid_f > 0.0, jnp.abs(div_h[:, :, 1:-1]), 0.0))
        cfl_x_local = jnp.max(jnp.where(valid_f > 0.0, jnp.abs(u_i), 0.0)) * params.dt / params.dx
        cfl_y_local = jnp.max(jnp.where(valid_f > 0.0, jnp.abs(v_i), 0.0)) * params.dt / params.dy
        cfl_z_local = jnp.max(jnp.where(valid_f > 0.0, jnp.abs(w_i), 0.0)) * params.dt / params.dz
        zero_time = jnp.asarray(0.0, dtype=params.dtype)
        theta_v = theta_i * (1.0 + 0.61 * qv_i) if params.moisture_enabled else theta_i
        theta_v_min = (
            lax.pmin(jnp.min(theta_v), axis_name) if params.thermo_enabled else zero_time
        )
        qv_min = lax.pmin(jnp.min(qv_i), axis_name) if params.moisture_enabled else zero_time
        qv_floor_hits = (
            lax.psum(jnp.sum(qv_i <= params.qv_floor), axis_name)
            if params.moisture_enabled
            else zero_time
        )
        return Diagnostics(
            step=step,
            ustar=ustar_sum / jnp.asarray(ustar_count, dtype=params.dtype),
            ke_max=lax.pmax(ke_local, axis_name),
            div_max=lax.pmax(div_local, axis_name),
            cfl_x=lax.pmax(cfl_x_local, axis_name),
            cfl_y=lax.pmax(cfl_y_local, axis_name),
            cfl_z=lax.pmax(cfl_z_local, axis_name),
            theta_v_min=theta_v_min,
            qv_min=qv_min,
            qv_floor_hits=qv_floor_hits,
            elapsed_s=zero_time,
            remaining_s=zero_time,
            total_s=zero_time,
        )

    z = z_slab_spec(axis_name)
    return _shard_map(
        local_diag,
        mesh=mesh,
        in_specs=(z, z, z, z, z, P()),
        out_specs=Diagnostics(
            P(), P(), P(), P(), P(), P(), P(),
            P(), P(), P(), P(), P(), P(),
        ),
        axis_name=axis_name,
    )


ABL_PROFILE_NAMES = (
    "mean_u",
    "mean_v",
    "mean_w",
    "var_u",
    "var_v",
    "var_w",
    "resolved_uw_face",
    "sgs_txz_face",
    "mean_cs",
)


def make_abl_profiles_sharded(
    params: Params,
    ops: Operators,
    mesh: Mesh,
    axis_name: str = "z",
):
    """Return replicated one-dimensional ABL statistics, never global fields."""
    ndev = mesh_size(mesh, axis_name)

    def local_profiles(
        u_i: jax.Array,
        v_i: jax.Array,
        w_i: jax.Array,
        cs2_i: jax.Array,
    ) -> jax.Array:
        gradients, _ = _build_gradient_bundle_local(
            u_i,
            v_i,
            w_i,
            None,
            None,
            params,
            ops,
            axis_name,
            ndev,
        )
        u_h, v_h, w_h = gradients.u, gradients.v, gradients.w
        w_center_h = _avg_prev_halo(w_h)
        u_face_h = _avg_next_halo(u_h)

        def fluctuation(q: jax.Array) -> tuple[jax.Array, jax.Array]:
            mean = jnp.mean(q, axis=(0, 1), keepdims=True)
            return mean, q - mean

        mean_u_h, u_prime_h = fluctuation(u_h)
        mean_v_h, v_prime_h = fluctuation(v_h)
        mean_w_h, w_prime_h = fluctuation(w_center_h)
        mean_u_face_h, u_face_prime_h = fluctuation(u_face_h)
        mean_w_face_h, w_face_prime_h = fluctuation(w_h)
        del mean_u_face_h, mean_w_face_h

        sij_uv = _strain_uv_halo(
            _to_sgs(gradients.dudx, params),
            _to_sgs(gradients.dudy, params),
            _to_sgs(gradients.dudz, params),
            _to_sgs(gradients.dvdx, params),
            _to_sgs(gradients.dvdy, params),
            _to_sgs(gradients.dvdz, params),
            _to_sgs(gradients.dwdx, params),
            _to_sgs(gradients.dwdy, params),
            _to_sgs(gradients.dwdz, params),
            axis_name,
        )
        sij_w = _strain_w_halo(
            _to_sgs(gradients.dudx, params),
            _to_sgs(gradients.dudy, params),
            _to_sgs(gradients.dudz, params),
            _to_sgs(gradients.dvdx, params),
            _to_sgs(gradients.dvdy, params),
            _to_sgs(gradients.dvdz, params),
            _to_sgs(gradients.dwdx, params),
            _to_sgs(gradients.dwdy, params),
            _to_sgs(gradients.dwdz, params),
            dudz_face=_to_sgs(gradients.dudz_face, params),
            dvdz_face=_to_sgs(gradients.dvdz_face, params),
        )
        if params.sgs_model == "lasd":
            cs2_h = _zero_boundary_halo(
                _to_sgs(cs2_i, params), axis_name, ndev
            )
            _, _, txz_h, _, _, _ = _stress_from_cs2_halo(
                cs2_h, sij_uv, sij_w, params, axis_name
            )
        else:
            _, _, txz_h, _, _, _ = classic_smagorinsky(
                gradients.dudx,
                gradients.dudy,
                gradients.dudz,
                gradients.dvdx,
                gradients.dvdy,
                gradients.dvdz,
                gradients.dwdx,
                gradients.dwdy,
                gradients.dwdz,
                params,
                dudz_face=gradients.dudz_face,
                dvdz_face=gradients.dvdz_face,
            )
        if params.molecular_viscosity_internal > 0.0:
            txz_h = txz_h - params.molecular_viscosity_internal * (
                gradients.dudz_face + gradients.dwdx
            )
        txz_h = _set_top_rank_plane(txz_h, -2, 0.0, axis_name, ndev)

        physical = slice(1, -1)
        local = jnp.stack(
            (
                mean_u_h[0, 0, physical],
                mean_v_h[0, 0, physical],
                mean_w_h[0, 0, physical],
                jnp.mean(u_prime_h * u_prime_h, axis=(0, 1))[physical],
                jnp.mean(v_prime_h * v_prime_h, axis=(0, 1))[physical],
                jnp.mean(w_prime_h * w_prime_h, axis=(0, 1))[physical],
                jnp.mean(
                    u_face_prime_h * w_face_prime_h, axis=(0, 1)
                )[physical],
                jnp.mean(txz_h, axis=(0, 1))[physical].astype(params.dtype),
                jnp.mean(
                    jnp.sqrt(jnp.maximum(cs2_i, 0.0)), axis=(0, 1)
                ),
            ),
            axis=0,
        ).astype(params.dtype)
        return lax.all_gather(
            local, axis_name, axis=1, tiled=True
        )

    z = z_slab_spec(axis_name)
    return _shard_map(
        local_profiles,
        mesh=mesh,
        in_specs=(z, z, z, z),
        out_specs=P(),
        axis_name=axis_name,
    )


def make_flow_slices_sharded(
    params: Params,
    mesh: Mesh,
    *,
    horizontal_height: float,
    x_index: int | None = None,
    y_index: int | None = None,
    axis_name: str = "z",
):
    """Return replicated xy, xz, and yz streamwise-velocity slices."""
    x_index = params.nx // 2 if x_index is None else x_index
    y_index = params.ny // 2 if y_index is None else y_index
    if not 0 <= x_index < params.nx or not 0 <= y_index < params.ny:
        raise ValueError("Flow-slice x/y indices must lie inside the domain")
    z_float = horizontal_height / (params.dz * params.z_i) - 0.5
    z_lower = int(np.clip(np.floor(z_float), 0, params.nz - 1))
    z_upper = min(z_lower + 1, params.nz - 1)
    upper_weight = float(np.clip(z_float - z_lower, 0.0, 1.0))
    lower_weight = 1.0 - upper_weight

    def local_slices(u_i: jax.Array) -> tuple[jax.Array, ...]:
        rank = lax.axis_index(axis_name)
        nz_local = u_i.shape[2]
        global_k = rank * nz_local + jnp.arange(nz_local)
        z_weights = (
            lower_weight * (global_k == z_lower).astype(u_i.dtype)
            + upper_weight * (global_k == z_upper).astype(u_i.dtype)
        )
        xy_local = jnp.sum(u_i * z_weights[None, None, :], axis=2)
        xy = lax.psum(xy_local, axis_name)
        xz = lax.all_gather(
            u_i[:, y_index, :], axis_name, axis=1, tiled=True
        )
        yz = lax.all_gather(
            u_i[x_index, :, :], axis_name, axis=1, tiled=True
        )
        return xy, xz, yz

    z = z_slab_spec(axis_name)
    return _shard_map(
        local_slices,
        mesh=mesh,
        in_specs=(z,),
        out_specs=(P(), P(), P()),
        axis_name=axis_name,
    )


def run_sharded(
    params: Params,
    *,
    num_devices: int | None = None,
    log_every: int | None = None,
    log_callback=None,
    status_callback=None,
    stop_callback=None,
    seed: int = 0,
    axis_name: str = "z",
) -> tuple[ShardedFlowState, ShardedOperators]:
    mesh = make_distributed_mesh(num_devices, axis_name)
    _validate_sharded_params(params, mesh, axis_name)
    ops = make_sharded_operators(params, mesh, axis_name)
    state = initial_sharded_state(params, mesh, seed=seed, axis_name=axis_name)
    initial_project = make_project_velocity_sharded(
        params, ops.pressure, mesh, axis_name, ops.pressure_spike
    )
    if params.use_jit:
        if status_callback:
            status_callback("[precompile] projecting the initial velocity field")
        initial_project = jax.jit(initial_project)
    u0, v0, w0, p0 = initial_project(
        state.u,
        state.v,
        state.w,
        ops.pressure,
        ops.pressure_spike,
    )
    state = state._replace(u=u0, v=v0, w=w0, p=p0)
    step_fn = make_step_ab2_sharded(params, ops, mesh, axis_name)
    diag_fn = make_diagnostics_sharded(params, ops.horizontal, mesh, axis_name)

    if params.use_jit:
        if status_callback:
            status_callback(
                f"[precompile] lowering sharded ab2 step for {params.nx}x{params.ny}x{params.nz} on {mesh_size(mesh, axis_name)} device(s)"
            )
        lowered = jax.jit(step_fn).lower(
            state, ops.pressure, ops.pressure_spike
        )
        if status_callback:
            status_callback("[precompile] compiling sharded step kernel")
        start_compile = time.perf_counter()
        step_fn = lowered.compile()
        if status_callback:
            status_callback(f"[precompile] done in {time.perf_counter() - start_compile:.1f}s")
        diag_fn = jax.jit(diag_fn)

    log_every = params.c_count if log_every is None else log_every
    start = time.perf_counter()

    def emit_diag(current_state: ShardedFlowState, step_index: int) -> bool:
        diag = diag_fn(
            current_state.u,
            current_state.v,
            current_state.w,
            current_state.theta,
            current_state.qv,
            current_state.step,
        )
        diag.step.block_until_ready()
        validate_cfl(diag)
        should_stop = (
            bool(stop_callback(diag)) if stop_callback is not None else False
        )
        if log_callback is None:
            return should_stop
        elapsed = time.perf_counter() - start
        if step_index == 0:
            remaining = 0.0
            total = 0.0
        else:
            total = elapsed * params.nsteps / step_index
            remaining = max(0.0, total - elapsed)
        diag = diag._replace(elapsed_s=elapsed, remaining_s=remaining, total_s=total)
        log_callback(diag)
        return should_stop

    if emit_diag(state, 0):
        return state, ops
    for step_index in range(1, params.nsteps + 1):
        if (
            params.sgs_model == "lasd"
            and (step_index % params.cs_count) == 0
        ):
            update_diag = diag_fn(
                state.u, state.v, state.w, state.theta, state.qv, state.step
            )
            update_diag = jax.block_until_ready(update_diag)
            validate_cfl(update_diag)
            validate_lasd_cfl(update_diag, params)
        state = step_fn(state, ops.pressure, ops.pressure_spike)
        if step_index % log_every == 0 or step_index == params.nsteps:
            if emit_diag(state, step_index):
                break
    return state, ops
