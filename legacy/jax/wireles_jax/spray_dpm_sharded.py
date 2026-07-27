"""Distributed, shard-local storage and z-slab migration for spray parcels."""

from __future__ import annotations

from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .config import Params
from .sharding import (
    _shard_map,
    make_array_from_local_callback,
    mesh_size,
    z_slab_spec,
)
from .spray_dpm import (
    SprayDPMConfig,
    SprayGasIncrements,
    SprayState,
    _cic_deposit,
    _cic_min_sample,
    _cic_sample,
    _hydrostatic_exner_and_pressure,
    _advance_sgs_velocity_seen,
    _advance_parcel_implicit,
    _eddy_crossing_limited_time_scale,
    _parcel_rates_from_samples,
    _proposed_phase_change,
    _sgs_velocity_statistics,
    inject_spray,
)
from .timestep_sharded import (
    ShardedFlowState,
    ShardedOperators,
    _plane_zeros,
    _set_w_physical_boundaries,
    _z_halo_many,
    make_apply_moisture_bounds_sharded,
    make_step_ab2_sharded,
)


_PACKED_FIELDS = 18


class SprayMigrationDiagnostics(NamedTuple):
    active_parcels: jax.Array
    exited_parcels: jax.Array
    overflow_parcels: jax.Array
    liquid_mass: jax.Array


class ShardedSprayDiagnostics(NamedTuple):
    active_parcels: jax.Array
    liquid_mass: jax.Array
    evaporated_mass: jax.Array
    air_energy_loss: jax.Array
    net_radiative_energy: jax.Array
    exited_parcels: jax.Array
    overflow_parcels: jax.Array


class ShardedSprayCoupledState(NamedTuple):
    flow: ShardedFlowState
    spray: SprayState


def spray_sharding(mesh: Mesh, axis_name: str = "z") -> NamedSharding:
    return NamedSharding(mesh, P(axis_name))


def _spray_specs(axis_name: str) -> SprayState:
    spec = P(axis_name)
    return SprayState(*(spec for _ in SprayState._fields))


def _sample_numpy_diameters(
    config: SprayDPMConfig,
    rng: np.random.Generator,
    count: int,
    dtype: np.dtype,
) -> np.ndarray:
    if config.diameter_distribution == "monodisperse":
        return np.full(count, config.initial_diameter, dtype=dtype)
    if config.diameter_distribution == "rosin_rammler":
        uniform = rng.random(count)
        scale = config.initial_diameter
        spread = config.rosin_rammler_spread
        fmin = 1.0 - np.exp(-((config.minimum_diameter / scale) ** spread))
        fmax = 1.0 - np.exp(-((config.maximum_diameter / scale) ** spread))
        probability = fmin + uniform * (fmax - fmin)
        return np.asarray(
            scale * (-np.log1p(-probability)) ** (1.0 / spread), dtype=dtype
        )
    if config.diameter_distribution == "lognormal":
        values = config.initial_diameter * np.exp(
            np.log(config.lognormal_geometric_stddev) * rng.standard_normal(count)
        )
        return np.asarray(
            np.clip(values, config.minimum_diameter, config.maximum_diameter),
            dtype=dtype,
        )
    fractions = np.asarray(config.tabulated_mass_fractions, dtype=np.float64)
    fractions /= fractions.sum()
    indices = rng.choice(len(fractions), size=count, p=fractions)
    return np.asarray(config.tabulated_diameters, dtype=dtype)[indices]


def initialize_sharded_spray(
    config: SprayDPMConfig,
    params: Params,
    mesh: Mesh,
    *,
    seed: int = 0,
    axis_name: str = "z",
) -> SprayState:
    """Create distributed buffers without materializing a global host array.

    ``max_parcels`` is the capacity *per z shard*. The abstract global length
    is ``num_shards * max_parcels``, while every device stores only one local
    fixed-capacity buffer.
    """
    ndev = mesh_size(mesh, axis_name)
    capacity = config.max_parcels
    global_shape = (ndev * capacity,)
    sharding = spray_sharding(mesh, axis_name)
    dtype = np.dtype(params.dtype)
    domain_z = params.lz * params.z_i
    if not 0.0 < config.injection_z < domain_z:
        raise ValueError("spray injection_z must lie inside the physical domain")
    injection_owner = min(int(config.injection_z / (domain_z / ndev)), ndev - 1)
    if config.initial_parcels > capacity:
        raise ValueError("initial_parcels exceeds the per-shard parcel capacity")

    def local_state(index: tuple[slice, ...]) -> dict[str, np.ndarray]:
        local_slice = index[0]
        rank = local_slice.start // capacity
        rng = np.random.default_rng(np.random.SeedSequence([seed, rank]))
        radial = config.injection_radius * np.sqrt(rng.random(capacity))
        angle = 2.0 * np.pi * rng.random(capacity)
        diameter = _sample_numpy_diameters(config, rng, capacity, dtype)
        mass = (np.pi / 6.0) * config.liquid_density * diameter**3
        solute_mass = config.salinity_mass_fraction * mass
        residual_volume = np.maximum(
            (np.pi / 6.0) * diameter**3
            - (mass - solute_mass) / config.water_density,
            0.0,
        )
        active = np.zeros(capacity, dtype=np.bool_)
        if rank == injection_owner:
            active[: config.initial_parcels] = True
        return {
            "x": np.asarray(config.injection_x + radial * np.cos(angle), dtype=dtype),
            "y": np.asarray(config.injection_y + radial * np.sin(angle), dtype=dtype),
            "z": np.full(capacity, config.injection_z, dtype=dtype),
            "u": np.full(capacity, config.injection_u, dtype=dtype),
            "v": np.full(capacity, config.injection_v, dtype=dtype),
            "w": np.full(capacity, config.injection_w, dtype=dtype),
            "mass": np.asarray(mass, dtype=dtype),
            "solute_mass": np.asarray(solute_mass, dtype=dtype),
            "residual_volume": np.asarray(residual_volume, dtype=dtype),
            "diameter": np.asarray(diameter, dtype=dtype),
            "temperature": np.full(capacity, config.initial_temperature, dtype=dtype),
            "weight": np.full(capacity, config.parcel_weight, dtype=dtype),
            "sgs_u": np.zeros(capacity, dtype=dtype),
            "sgs_v": np.zeros(capacity, dtype=dtype),
            "sgs_w": np.zeros(capacity, dtype=dtype),
            "parcel_id": np.asarray(
                np.arange(capacity), dtype=np.uint32
            ),
            "active": active,
        }

    def field(name: str, field_dtype) -> jax.Array:
        return make_array_from_local_callback(
            global_shape,
            sharding,
            lambda index: local_state(index)[name],
            dtype=field_dtype,
        )

    return SprayState(
        x=field("x", params.dtype),
        y=field("y", params.dtype),
        z=field("z", params.dtype),
        u=field("u", params.dtype),
        v=field("v", params.dtype),
        w=field("w", params.dtype),
        mass=field("mass", params.dtype),
        solute_mass=field("solute_mass", params.dtype),
        residual_volume=field("residual_volume", params.dtype),
        diameter=field("diameter", params.dtype),
        temperature=field("temperature", params.dtype),
        weight=field("weight", params.dtype),
        sgs_u=field("sgs_u", params.dtype),
        sgs_v=field("sgs_v", params.dtype),
        sgs_w=field("sgs_w", params.dtype),
        parcel_id=field("parcel_id", jnp.uint32),
        active=field("active", jnp.bool_),
    )


def _pack_spray(state: SprayState, active: jax.Array | None = None) -> jax.Array:
    if active is None:
        active = state.active
    id_low = (state.parcel_id & jnp.uint32(0xFFFF)).astype(state.x.dtype)
    id_high = (state.parcel_id >> jnp.uint32(16)).astype(state.x.dtype)
    return jnp.stack(
        (
            state.x,
            state.y,
            state.z,
            state.u,
            state.v,
            state.w,
            state.mass,
            state.solute_mass,
            state.residual_volume,
            state.diameter,
            state.temperature,
            state.weight,
            state.sgs_u,
            state.sgs_v,
            state.sgs_w,
            id_low,
            id_high,
            active.astype(state.x.dtype),
        ),
        axis=-1,
    )


def _unpack_spray(packed: jax.Array) -> SprayState:
    if packed.shape[-1] != _PACKED_FIELDS:
        raise ValueError(f"packed spray state must have {_PACKED_FIELDS} columns")
    id_low = jnp.rint(packed[:, 15]).astype(jnp.uint32)
    id_high = jnp.rint(packed[:, 16]).astype(jnp.uint32)
    return SprayState(
        x=packed[:, 0],
        y=packed[:, 1],
        z=packed[:, 2],
        u=packed[:, 3],
        v=packed[:, 4],
        w=packed[:, 5],
        mass=packed[:, 6],
        solute_mass=packed[:, 7],
        residual_volume=packed[:, 8],
        diameter=packed[:, 9],
        temperature=packed[:, 10],
        weight=packed[:, 11],
        sgs_u=packed[:, 12],
        sgs_v=packed[:, 13],
        sgs_w=packed[:, 14],
        parcel_id=id_low | (id_high << jnp.uint32(16)),
        active=packed[:, 17] > 0.5,
    )


def _compact_packed(packed: jax.Array, capacity: int) -> tuple[jax.Array, jax.Array]:
    active = packed[:, -1] > 0.5
    order = jnp.argsort(~active, stable=True)
    overflow = jnp.maximum(jnp.sum(active) - capacity, 0)
    return packed[order[:capacity]], overflow


def make_inject_sharded_spray(
    config: SprayDPMConfig,
    params: Params,
    mesh: Mesh,
    axis_name: str = "z",
):
    """Return a shard-map injector that operates only on the owning z slab."""
    ndev = mesh_size(mesh, axis_name)
    domain_z = params.lz * params.z_i
    owner = min(int(config.injection_z / (domain_z / ndev)), ndev - 1)

    def local_inject(state: SprayState, step: jax.Array) -> SprayState:
        rank = lax.axis_index(axis_name)
        injected = inject_spray(state, step, params.dt_physical, config)
        is_owner = rank == owner
        return jax.tree.map(
            lambda new, old: jnp.where(is_owner, new, old), injected, state
        )

    return _shard_map(
        local_inject,
        mesh=mesh,
        in_specs=(_spray_specs(axis_name), P()),
        out_specs=_spray_specs(axis_name),
        axis_name=axis_name,
    )


def make_migrate_sharded_spray(
    config: SprayDPMConfig,
    params: Params,
    mesh: Mesh,
    axis_name: str = "z",
):
    """Return packed nearest-neighbor parcel migration across z shards."""
    ndev = mesh_size(mesh, axis_name)
    capacity = config.max_parcels
    slab_height = params.lz * params.z_i / ndev
    send_down_perm = [(rank, rank - 1) for rank in range(1, ndev)]
    send_up_perm = [(rank, rank + 1) for rank in range(ndev - 1)]

    def local_migrate(
        state: SprayState,
    ) -> tuple[SprayState, SprayMigrationDiagnostics]:
        rank = lax.axis_index(axis_name)
        exited = jnp.asarray(0, dtype=jnp.int32)
        overflow = jnp.asarray(0, dtype=jnp.int32)
        current = state
        lower = rank.astype(current.z.dtype) * slab_height
        upper = (rank.astype(current.z.dtype) + 1.0) * slab_height

        # At most ndev-1 neighbor hops are required for any in-domain parcel;
        # the additional pass compacts the final owner buffer.
        for _ in range(ndev):
            below = current.active & (current.z < lower)
            above = current.active & (current.z >= upper)
            exit_lower = below & (rank == 0)
            exit_upper = above & (rank == ndev - 1)
            exited = exited + jnp.sum(exit_lower) + jnp.sum(exit_upper)
            send_down = below & (rank > 0)
            send_up = above & (rank < ndev - 1)
            stay = current.active & ~(below | above)

            packed = _pack_spray(current)
            down_packed, _ = _compact_packed(
                packed.at[:, -1].set(send_down.astype(packed.dtype)), capacity
            )
            up_packed, _ = _compact_packed(
                packed.at[:, -1].set(send_up.astype(packed.dtype)), capacity
            )
            from_upper = lax.ppermute(
                down_packed, axis_name, perm=send_down_perm
            )
            from_lower = lax.ppermute(
                up_packed, axis_name, perm=send_up_perm
            )
            stay_packed = packed.at[:, -1].set(stay.astype(packed.dtype))
            combined = jnp.concatenate((stay_packed, from_lower, from_upper), axis=0)
            compacted, local_overflow = _compact_packed(combined, capacity)
            overflow = overflow + local_overflow.astype(jnp.int32)
            current = _unpack_spray(compacted)

        active_local = jnp.sum(current.active)
        liquid_local = jnp.sum(
            current.mass * current.weight * current.active.astype(current.mass.dtype)
        )
        diagnostics = SprayMigrationDiagnostics(
            active_parcels=lax.psum(active_local, axis_name),
            exited_parcels=lax.psum(exited, axis_name),
            overflow_parcels=lax.psum(overflow, axis_name),
            liquid_mass=lax.psum(liquid_local, axis_name),
        )
        return current, diagnostics

    replicated = SprayMigrationDiagnostics(P(), P(), P(), P())
    return _shard_map(
        local_migrate,
        mesh=mesh,
        in_specs=(_spray_specs(axis_name),),
        out_specs=(_spray_specs(axis_name), replicated),
        axis_name=axis_name,
    )


def _local_cic_coordinates(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    params: Params,
    rank: jax.Array,
    local_nz: int,
    *,
    z_offset: float,
) -> tuple[jax.Array, ...]:
    dx = params.dx * params.z_i
    dy = params.dy * params.z_i
    dz = params.dz * params.z_i
    domain_x = params.lx * params.z_i
    domain_y = params.ly * params.z_i

    x_index = jnp.mod(x, domain_x) / dx
    y_index = jnp.mod(y, domain_y) / dy
    z_index = jnp.clip(z / dz - z_offset, 0.0, params.nz - 1.0)
    ix0_raw = jnp.floor(x_index).astype(jnp.int32)
    iy0_raw = jnp.floor(y_index).astype(jnp.int32)
    iz0_global = jnp.floor(z_index).astype(jnp.int32)
    ix0 = jnp.mod(ix0_raw, params.nx)
    iy0 = jnp.mod(iy0_raw, params.ny)
    ix1 = jnp.mod(ix0 + 1, params.nx)
    iy1 = jnp.mod(iy0 + 1, params.ny)
    iz1_global = jnp.minimum(iz0_global + 1, params.nz - 1)
    local_start = rank.astype(jnp.int32) * local_nz
    iz0 = jnp.clip(iz0_global - local_start + 1, 0, local_nz + 1)
    iz1 = jnp.clip(iz1_global - local_start + 1, 0, local_nz + 1)
    fx = x_index - ix0_raw
    fy = y_index - iy0_raw
    fz = jnp.where(iz1_global > iz0_global, z_index - iz0_global, 0.0)
    return ix0, ix1, iy0, iy1, iz0, iz1, fx, fy, fz


def _fold_deposit_halos(
    fields_h: tuple[jax.Array, ...],
    axis_name: str,
    ndev: int,
) -> tuple[jax.Array, ...]:
    """Return owner-local fields after transferring CIC halo contributions."""
    packed = jnp.stack(fields_h, axis=-1)
    interior = packed[:, :, 1:-1, :]
    if ndev == 1:
        return tuple(interior[..., index] for index in range(len(fields_h)))
    recv_from_upper = lax.ppermute(
        packed[:, :, :1, :],
        axis_name,
        perm=[(rank, rank - 1) for rank in range(1, ndev)],
    )
    recv_from_lower = lax.ppermute(
        packed[:, :, -1:, :],
        axis_name,
        perm=[(rank, rank + 1) for rank in range(ndev - 1)],
    )
    interior = interior.at[:, :, :1, :].add(recv_from_lower)
    interior = interior.at[:, :, -1:, :].add(recv_from_upper)
    return tuple(interior[..., index] for index in range(len(fields_h)))


def make_spray_exchange_sharded(
    config: SprayDPMConfig,
    params: Params,
    mesh: Mesh,
    axis_name: str = "z",
):
    """Advance and couple distributed parcels without gathering either phase."""
    if not params.thermo_enabled or not params.moisture_enabled:
        raise ValueError(
            "distributed spray requires thermo_enabled and moisture_enabled"
        )
    ndev = mesh_size(mesh, axis_name)
    local_nz = params.nz // ndev
    halo_shape = (params.nx, params.ny, local_nz + 2)
    dt_sub = params.dt_physical / config.substeps
    domain_x = params.lx * params.z_i
    domain_y = params.ly * params.z_i
    cell_mass = (
        config.air_density
        * params.dx
        * params.z_i
        * params.dy
        * params.z_i
        * params.dz
        * params.z_i
    )

    def local_statistics(
        u_i: jax.Array,
        v_i: jax.Array,
        w_i: jax.Array,
    ) -> tuple[jax.Array, ...]:
        w_i = _set_w_physical_boundaries(w_i, axis_name, ndev)
        (w_faces_h,) = _z_halo_many(
            (w_i,),
            (_plane_zeros(w_i),),
            (_plane_zeros(w_i),),
            axis_name,
            ndev,
        )
        w_center_i = 0.5 * (
            w_faces_h[:, :, 1:-1] + w_faces_h[:, :, :-2]
        )
        u_h, v_h, w_center_h = _z_halo_many(
            (u_i, v_i, w_center_i),
            (
                u_i[:, :, :1],
                v_i[:, :, :1],
                w_center_i[:, :, :1],
            ),
            (
                u_i[:, :, -1:],
                v_i[:, :, -1:],
                w_center_i[:, :, -1:],
            ),
            axis_name,
            ndev,
        )
        return tuple(
            field[:, :, 1:-1]
            for field in _sgs_velocity_statistics(
                u_h, v_h, w_center_h, params
            )
        )

    def local_substep(
        u_i: jax.Array,
        v_i: jax.Array,
        w_i: jax.Array,
        theta_i: jax.Array,
        qv_i: jax.Array,
        variance_u_i: jax.Array,
        variance_v_i: jax.Array,
        variance_w_i: jax.Array,
        time_scale_i: jax.Array,
        spray: SprayState,
        counter: jax.Array,
    ) -> tuple[SprayState, SprayGasIncrements, ShardedSprayDiagnostics]:
        rank = lax.axis_index(axis_name)
        w_i = _set_w_physical_boundaries(w_i, axis_name, ndev)
        (w_faces_h,) = _z_halo_many(
            (w_i,),
            (_plane_zeros(w_i),),
            (_plane_zeros(w_i),),
            axis_name,
            ndev,
        )
        w_center_i = 0.5 * (
            w_faces_h[:, :, 1:-1] + w_faces_h[:, :, :-2]
        )
        (
            u_h,
            v_h,
            w_center_h,
            theta_h,
            qv_h,
            variance_u_h,
            variance_v_h,
            variance_w_h,
            time_scale_h,
        ) = _z_halo_many(
            (
                u_i,
                v_i,
                w_center_i,
                theta_i,
                qv_i,
                variance_u_i,
                variance_v_i,
                variance_w_i,
                time_scale_i,
            ),
            (
                u_i[:, :, :1],
                v_i[:, :, :1],
                w_center_i[:, :, :1],
                theta_i[:, :, :1],
                qv_i[:, :, :1],
                variance_u_i[:, :, :1],
                variance_v_i[:, :, :1],
                variance_w_i[:, :, :1],
                time_scale_i[:, :, :1],
            ),
            (
                u_i[:, :, -1:],
                v_i[:, :, -1:],
                w_center_i[:, :, -1:],
                theta_i[:, :, -1:],
                qv_i[:, :, -1:],
                variance_u_i[:, :, -1:],
                variance_v_i[:, :, -1:],
                variance_w_i[:, :, -1:],
                time_scale_i[:, :, -1:],
            ),
            axis_name,
            ndev,
        )
        coords = _local_cic_coordinates(
            spray.x,
            spray.y,
            spray.z,
            params,
            rank,
            local_nz,
            z_offset=0.5,
        )
        gas_u = _cic_sample(u_h, coords)
        gas_v = _cic_sample(v_h, coords)
        gas_w = _cic_sample(w_center_h, coords)
        if config.turbulent_dispersion_enabled:
            sampled_statistics = tuple(
                _cic_sample(field, coords)
                for field in (
                    variance_u_h,
                    variance_v_h,
                    variance_w_h,
                    time_scale_h,
                )
            )
            sampled_statistics = (
                *sampled_statistics[:3],
                _eddy_crossing_limited_time_scale(
                    spray,
                    gas_u,
                    gas_v,
                    gas_w,
                    sampled_statistics[3],
                    params,
                ),
            )
            spray = _advance_sgs_velocity_seen(
                spray,
                *sampled_statistics,
                dt_sub,
                counter,
                config,
            )
        rates = _parcel_rates_from_samples(
            spray,
            gas_u + spray.sgs_u,
            gas_v + spray.sgs_v,
            gas_w + spray.sgs_w,
            _cic_sample(theta_h, coords),
            jnp.maximum(_cic_sample(qv_h, coords), 0.0),
            params,
            config,
        )
        proposed_phase_change = _proposed_phase_change(
            spray, rates, dt_sub, config
        )
        active_value = spray.active.astype(params.dtype)
        proposed_weighted_change = (
            spray.weight * proposed_phase_change * active_value
        )
        (proposed_vapor_mass,) = _fold_deposit_halos(
            (
                _cic_deposit(
                    proposed_weighted_change,
                    coords,
                    halo_shape,
                    params.dtype,
                ),
            ),
            axis_name,
            ndev,
        )
        available_vapor_mass = jnp.maximum(
            qv_i - params.qv_floor, 0.0
        ) * cell_mass
        condensation_ratio = jnp.where(
            proposed_vapor_mass < 0.0,
            available_vapor_mass
            / jnp.maximum(-proposed_vapor_mass, 1.0e-30),
            1.0,
        )
        condensation_ratio = jnp.clip(condensation_ratio, 0.0, 1.0)
        (condensation_ratio_h,) = _z_halo_many(
            (condensation_ratio,),
            (condensation_ratio[:, :, :1],),
            (condensation_ratio[:, :, -1:],),
            axis_name,
            ndev,
        )
        condensation_scale = _cic_min_sample(condensation_ratio_h, coords)
        phase_change = jnp.where(
            proposed_phase_change < 0.0,
            condensation_scale * proposed_phase_change,
            proposed_phase_change,
        )
        advanced = _advance_parcel_implicit(
            spray,
            gas_u + spray.sgs_u,
            gas_v + spray.sgs_v,
            gas_w + spray.sgs_w,
            rates,
            phase_change,
            dt_sub,
            params,
            config,
        )
        old_mass = spray.mass
        new_x = jnp.mod(
            spray.x + 0.5 * dt_sub * (spray.u + advanced.u), domain_x
        )
        new_y = jnp.mod(
            spray.y + 0.5 * dt_sub * (spray.v + advanced.v), domain_y
        )
        new_z = spray.z + 0.5 * dt_sub * (spray.w + advanced.w)
        active = spray.active & (advanced.diameter >= config.min_diameter)

        weighted_evaporation = spray.weight * phase_change * active_value
        weighted_energy = spray.weight * advanced.convective_energy * active_value
        weighted_radiation = spray.weight * advanced.radiative_energy * active_value
        weighted_mass = spray.weight * old_mass * active_value
        impulse_u = -weighted_mass * advanced.drag_delta_u
        impulse_v = -weighted_mass * advanced.drag_delta_v
        impulse_w = -weighted_mass * advanced.drag_delta_w
        face_coords = _local_cic_coordinates(
            spray.x,
            spray.y,
            spray.z,
            params,
            rank,
            local_nz,
            z_offset=1.0,
        )

        updated = SprayState(
            x=new_x,
            y=new_y,
            z=new_z,
            u=jnp.where(spray.active, advanced.u, spray.u),
            v=jnp.where(spray.active, advanced.v, spray.v),
            w=jnp.where(spray.active, advanced.w, spray.w),
            mass=jnp.where(spray.active, advanced.mass, spray.mass),
            solute_mass=spray.solute_mass,
            residual_volume=spray.residual_volume,
            diameter=jnp.where(
                spray.active, advanced.diameter, spray.diameter
            ),
            temperature=jnp.where(
                spray.active, advanced.temperature, spray.temperature
            ),
            weight=spray.weight,
            sgs_u=spray.sgs_u,
            sgs_v=spray.sgs_v,
            sgs_w=spray.sgs_w,
            parcel_id=spray.parcel_id,
            active=active,
        )
        deposited = _fold_deposit_halos(
            (
                _cic_deposit(impulse_u, coords, halo_shape, params.dtype),
                _cic_deposit(impulse_v, coords, halo_shape, params.dtype),
                _cic_deposit(impulse_w, face_coords, halo_shape, params.dtype),
                -_cic_deposit(weighted_energy, coords, halo_shape, params.dtype),
                _cic_deposit(
                    weighted_evaporation, coords, halo_shape, params.dtype
                ),
            ),
            axis_name,
            ndev,
        )
        global_k = rank.astype(params.dtype) * local_nz + jnp.arange(
            local_nz, dtype=params.dtype
        )
        z_centers = (global_k + 0.5) * params.dz * params.z_i
        exner, _ = _hydrostatic_exner_and_pressure(z_centers, params, config)
        increments = SprayGasIncrements(
            u=deposited[0] / cell_mass,
            v=deposited[1] / cell_mass,
            w=deposited[2] / cell_mass,
            theta=deposited[3]
            / (cell_mass * config.air_heat_capacity * exner[None, None, :]),
            qv=deposited[4] / cell_mass,
        )
        liquid_local = jnp.sum(
            updated.weight
            * updated.mass
            * updated.active.astype(updated.mass.dtype)
        )
        zero_count = jnp.asarray(0, dtype=jnp.int32)
        diagnostics = ShardedSprayDiagnostics(
            active_parcels=lax.psum(jnp.sum(updated.active), axis_name),
            liquid_mass=lax.psum(liquid_local, axis_name),
            evaporated_mass=lax.psum(jnp.sum(weighted_evaporation), axis_name),
            air_energy_loss=lax.psum(jnp.sum(weighted_energy), axis_name),
            net_radiative_energy=lax.psum(
                jnp.sum(weighted_radiation), axis_name
            ),
            exited_parcels=zero_count,
            overflow_parcels=zero_count,
        )
        return updated, increments, diagnostics

    z = z_slab_spec(axis_name)
    replicated = ShardedSprayDiagnostics(
        P(), P(), P(), P(), P(), P(), P()
    )
    statistics = _shard_map(
        local_statistics,
        mesh=mesh,
        in_specs=(z, z, z),
        out_specs=(z, z, z, z),
        axis_name=axis_name,
    )
    substep = _shard_map(
        local_substep,
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
            _spray_specs(axis_name),
            P(),
        ),
        out_specs=(
            _spray_specs(axis_name),
            SprayGasIncrements(z, z, z, z, z),
            replicated,
        ),
        axis_name=axis_name,
    )
    migrate = make_migrate_sharded_spray(config, params, mesh, axis_name)

    def exchange(
        flow: ShardedFlowState,
        spray: SprayState,
    ) -> tuple[SprayState, SprayGasIncrements, ShardedSprayDiagnostics]:
        increments = SprayGasIncrements(
            *(jnp.zeros_like(flow.u) for _ in SprayGasIncrements._fields)
        )
        scalar_zero = jnp.asarray(0.0, dtype=params.dtype)
        count_zero = jnp.asarray(0, dtype=jnp.int32)
        evaporated = scalar_zero
        air_energy = scalar_zero
        radiation = scalar_zero
        exited = count_zero
        overflow = count_zero
        current = spray
        if config.turbulent_dispersion_enabled:
            sgs_statistics = statistics(flow.u, flow.v, flow.w)
        else:
            zeros = jnp.zeros_like(flow.u)
            sgs_statistics = (zeros, zeros, zeros, jnp.ones_like(zeros))
        migration = SprayMigrationDiagnostics(
            count_zero, count_zero, count_zero, scalar_zero
        )
        for substep_index in range(config.substeps):
            counter = (
                flow.step.astype(jnp.uint32) * config.substeps
                + jnp.asarray(substep_index, dtype=jnp.uint32)
            )
            current, subincrements, diagnostics = substep(
                flow.u,
                flow.v,
                flow.w,
                flow.theta,
                flow.qv + increments.qv,
                *sgs_statistics,
                current,
                counter,
            )
            current, migration = migrate(current)
            increments = jax.tree.map(
                lambda total, value: total + value,
                increments,
                subincrements,
            )
            evaporated = evaporated + diagnostics.evaporated_mass
            air_energy = air_energy + diagnostics.air_energy_loss
            radiation = radiation + diagnostics.net_radiative_energy
            exited = exited + migration.exited_parcels
            overflow = overflow + migration.overflow_parcels
        diagnostics = ShardedSprayDiagnostics(
            active_parcels=migration.active_parcels,
            liquid_mass=migration.liquid_mass,
            evaporated_mass=evaporated,
            air_energy_loss=air_energy,
            net_radiative_energy=radiation,
            exited_parcels=exited,
            overflow_parcels=overflow,
        )
        return current, increments, diagnostics

    return exchange


def make_step_spray_dpm_sharded(
    config: SprayDPMConfig,
    params: Params,
    operators: ShardedOperators,
    mesh: Mesh,
    axis_name: str = "z",
):
    """Return one fully distributed, two-way coupled spray/carrier step."""
    inject = make_inject_sharded_spray(config, params, mesh, axis_name)
    exchange = make_spray_exchange_sharded(config, params, mesh, axis_name)
    flow_step = make_step_ab2_sharded(params, operators, mesh, axis_name)
    moisture_bounds = make_apply_moisture_bounds_sharded(mesh, axis_name)

    def step(
        state: ShardedSprayCoupledState,
        runtime_pressure_ops=None,
        runtime_spike_ops=None,
    ) -> tuple[ShardedSprayCoupledState, ShardedSprayDiagnostics]:
        injected = inject(state.spray, state.flow.step)
        spray, increments, diagnostics = exchange(state.flow, injected)
        forced_flow = state.flow._replace(
            u=state.flow.u + increments.u,
            v=state.flow.v + increments.v,
            w=state.flow.w + increments.w,
            theta=state.flow.theta + increments.theta,
            qv=moisture_bounds(
                state.flow.qv + increments.qv,
                jnp.asarray(params.qv_floor, dtype=params.dtype),
            ),
        )
        advanced_flow = flow_step(
            forced_flow,
            runtime_pressure_ops,
            runtime_spike_ops,
        )
        return (
            ShardedSprayCoupledState(flow=advanced_flow, spray=spray),
            diagnostics,
        )

    return step
