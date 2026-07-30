"""Distributed closures for cryogenic low-Mach carrier flow."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from .config import Params
from .cryogenic_microphysics import (
    CryogenicMicrophysicsConfig,
    advance_fog_microphysics,
    mass_only_outlet_update,
    saturation_mixing_ratio,
    smooth_outlet_window,
    stokes_terminal_velocity,
)
from .sharding import (
    _shard_map,
    make_array_from_local_callback,
    mesh_size,
    z_slab_sharding,
    z_slab_spec,
)
from .spray_dpm import SprayDPMConfig, SprayState
from .spray_dpm_sharded import (
    ShardedSprayDiagnostics,
    make_inject_sharded_spray,
    make_spray_exchange_sharded,
    spray_sharding,
)
from .timestep_sharded import (
    ShardedFlowState,
    ShardedOperators,
    _ab_update_inner,
    _copy_boundary_halo,
    _shift_z_minus,
    _shift_z_plus,
    _set_w_physical_boundaries,
    make_apply_moisture_bounds_sharded,
    make_scalar_rhs_buoyancy_sharded,
    make_step_ab2_sharded,
)


class CryogenicScalarState(NamedTuple):
    """Eulerian gas and suspended condensate mixing ratios."""

    yn2: jax.Array
    ql: jax.Array
    qi: jax.Array
    enthalpy: jax.Array
    rhs_yn2_prev: jax.Array
    rhs_ql_prev: jax.Array
    rhs_qi_prev: jax.Array
    rhs_enthalpy_prev: jax.Array


class ShardedCryogenicState(NamedTuple):
    flow: ShardedFlowState
    spray: SprayState
    scalars: CryogenicScalarState


class ShardedCryogenicDiagnostics(NamedTuple):
    spray: ShardedSprayDiagnostics
    nitrogen_gas_mass: jax.Array
    liquid_fog_mass: jax.Array
    ice_fog_mass: jax.Array
    total_water_mass: jax.Array
    max_relative_humidity: jax.Array
    outlet_volume_rate: jax.Array
    nitrogen_sensible_cooling: jax.Array
    nitrogen_outlet_mass: jax.Array
    fog_condensed_mass: jax.Array
    fog_evaporated_mass: jax.Array


def initial_cryogenic_scalar_state(
    flow: ShardedFlowState,
    config: CryogenicMicrophysicsConfig | None = None,
) -> CryogenicScalarState:
    """Create zero nitrogen-anomaly and fog fields with flow-compatible sharding."""

    config = CryogenicMicrophysicsConfig() if config is None else config
    zero = jnp.zeros_like(flow.qv)
    enthalpy = (
        config.dry_air_heat_capacity * flow.theta
        + config.water_vapor_latent_heat * flow.qv
    )
    return CryogenicScalarState(
        yn2=zero,
        ql=zero,
        qi=zero,
        enthalpy=enthalpy.astype(flow.theta.dtype),
        rhs_yn2_prev=zero,
        rhs_ql_prev=zero,
        rhs_qi_prev=zero,
        rhs_enthalpy_prev=zero,
    )


def load_cryogenic_checkpoint_sidecar(
    directory: str | Path,
    flow: ShardedFlowState,
    spray_config: SprayDPMConfig,
    params: Params,
    mesh: Mesh,
    *,
    rank: int | None = None,
    axis_name: str = "z",
) -> ShardedCryogenicState:
    """Restore this process's nitrogen/fog and parcel checkpoint slab."""

    root = Path(directory)
    rank = jax.process_index() if rank is None else rank
    with (root / "cryogenic_manifest.json").open(
        encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "wireles-jax-cryogenic-zslab-v1":
        raise ValueError(
            "unsupported cryogenic checkpoint format: "
            f"{manifest.get('format')!r}"
        )
    if int(manifest["source_parts"]) != mesh_size(mesh, axis_name):
        raise ValueError("cryogenic checkpoint partition count does not match mesh")
    if tuple(manifest["global_shape"]) != (
        params.nx,
        params.ny,
        params.nz,
    ):
        raise ValueError("cryogenic checkpoint grid does not match run")
    if int(manifest["step"]) != int(jax.device_get(flow.step)):
        raise ValueError("flow and cryogenic checkpoints have different steps")
    if int(manifest["max_parcels_per_shard"]) != spray_config.max_parcels:
        raise ValueError("cryogenic parcel capacity does not match configuration")
    if tuple(manifest.get("scalar_fields", ())) != CryogenicScalarState._fields:
        raise ValueError(
            "cryogenic checkpoint scalar layout does not match current model"
        )

    archive = np.load(root / f"cryogenic_rank{rank:05d}.npz")
    try:
        z_sharding = z_slab_sharding(mesh, axis_name)
        parcel_sharding = spray_sharding(mesh, axis_name)
        shape = (params.nx, params.ny, params.nz)
        parcel_shape = (
            mesh_size(mesh, axis_name) * spray_config.max_parcels,
        )

        def restore(
            name: str,
            global_shape: tuple[int, ...],
            sharding,
        ) -> jax.Array:
            local = np.asarray(archive[name])

            def callback(index):
                expected = tuple(
                    part.stop - part.start for part in index
                )
                if local.shape != expected:
                    raise ValueError(
                        f"checkpoint field {name} has {local.shape}; "
                        f"expected local shape {expected}"
                    )
                return local

            return make_array_from_local_callback(
                global_shape,
                sharding,
                callback,
                dtype=local.dtype,
            )

        scalars = CryogenicScalarState(
            **{
                name: restore(f"scalar_{name}", shape, z_sharding)
                for name in CryogenicScalarState._fields
            }
        )
        spray = SprayState(
            **{
                name: restore(
                    f"spray_{name}",
                    parcel_shape,
                    parcel_sharding,
                )
                for name in SprayState._fields
            }
        )
    finally:
        archive.close()
    return ShardedCryogenicState(flow=flow, spray=spray, scalars=scalars)


def make_prescribed_ln2_mass_outlet_sharded(
    params: Params,
    mesh: Mesh,
    *,
    mass_flow_rate: float,
    source_x: float,
    source_y: float,
    source_z: float,
    source_sigma_x: float,
    source_sigma_r: float,
    config: CryogenicMicrophysicsConfig,
    axis_name: str = "z",
    return_scalar_sink: bool = False,
):
    """Return the mass-only outlet target divergence for a prescribed LN2 jet.

    The source assumes that the prescribed LN2 mass flow vaporizes inside the
    same Gaussian support as the equivalent cooling source.  Its added ideal-
    gas volume is removed in the outlet strip.  The returned closure changes
    only the continuity constraint; it contains no velocity, temperature, or
    humidity relaxation.
    """

    if mass_flow_rate < 0.0:
        raise ValueError("mass_flow_rate must be non-negative")
    ndev = mesh_size(mesh, axis_name)
    local_nz = params.nz // ndev
    cell_volume = params.dx * params.dy * params.dz * params.z_i**3
    domain_x = params.lx * params.z_i
    domain_y = params.ly * params.z_i
    sigma_x = max(source_sigma_x, 1.5 * params.dx * params.z_i)
    sigma_r = max(
        source_sigma_r,
        1.5 * max(params.dy, params.dz) * params.z_i,
    )

    def periodic_distance(
        coordinate: jax.Array,
        origin: float,
        period: float,
    ) -> jax.Array:
        return jnp.mod(coordinate - origin + 0.5 * period, period) - 0.5 * period

    def local_target(temperature_i: jax.Array) -> jax.Array:
        rank = lax.axis_index(axis_name)
        i = jnp.arange(params.nx, dtype=params.dtype)
        j = jnp.arange(params.ny, dtype=params.dtype)
        k = rank * local_nz + jnp.arange(local_nz, dtype=params.dtype)
        x = (i + 0.5) * params.dx * params.z_i
        y = (j + 0.5) * params.dy * params.z_i
        z = (k + 0.5) * params.dz * params.z_i
        x3 = x[:, None, None]
        y3 = y[None, :, None]
        z3 = z[None, None, :]
        dx = periodic_distance(x3, source_x, domain_x)
        dy = periodic_distance(y3, source_y, domain_y)
        kernel = jnp.exp(
            -0.5 * (dx / sigma_x) ** 2
            - 0.5 * (dy * dy + (z3 - source_z) ** 2) / sigma_r**2
        )
        kernel_integral = lax.psum(jnp.sum(kernel), axis_name) * cell_volume
        evaporation_mass_rate = (
            jnp.asarray(mass_flow_rate, dtype=params.dtype)
            * kernel
            / jnp.maximum(
                kernel_integral,
                jnp.asarray(jnp.finfo(params.dtype).tiny, dtype=params.dtype),
            )
        )
        nitrogen_density = config.pressure / (
            config.nitrogen_gas_constant
            * jnp.maximum(temperature_i, jnp.asarray(77.34, params.dtype))
        )
        expansion = evaporation_mass_rate / nitrogen_density

        outlet_weight = jnp.broadcast_to(
            smooth_outlet_window(
                x3,
                config.outlet_start_x,
                config.outlet_end_x,
            ),
            temperature_i.shape,
        )
        volume_rate = lax.psum(jnp.sum(expansion), axis_name) * cell_volume
        outlet_volume = (
            lax.psum(jnp.sum(outlet_weight), axis_name) * cell_volume
        )
        sink = volume_rate * outlet_weight / jnp.maximum(
            outlet_volume,
            jnp.asarray(jnp.finfo(params.dtype).tiny, dtype=params.dtype),
        )

        # Solver derivatives use the internal coordinate x / z_i.
        target = ((expansion - sink) * params.z_i).astype(params.dtype)
        # Remove the last floating-point reduction residual so the Neumann
        # pressure problem remains compatible even in float32.
        residual = lax.psum(jnp.sum(target), axis_name)
        target = target - residual / jnp.asarray(
            params.nx * params.ny * params.nz,
            dtype=params.dtype,
        )
        final_residual = lax.psum(jnp.sum(target), axis_name)
        target = jnp.where(
            lax.axis_index(axis_name) == 0,
            target.at[0, 0, 0].add(-final_residual),
            target,
        )
        if return_scalar_sink:
            return target, (sink * params.z_i).astype(params.dtype)
        return target

    z_spec = z_slab_spec(axis_name)
    return _shard_map(
        local_target,
        mesh=mesh,
        in_specs=z_spec,
        out_specs=(
            (z_spec, z_spec) if return_scalar_sink else z_spec
        ),
        axis_name=axis_name,
    )


def make_cryogenic_buoyancy_sharded(
    params: Params,
    config: CryogenicMicrophysicsConfig,
    mesh: Mesh,
    axis_name: str = "z",
):
    """Return nitrogen-composition buoyancy plus suspended-water loading."""

    ndev = mesh_size(mesh, axis_name)
    def local_buoyancy(
        yn2_i: jax.Array,
        qv_i: jax.Array,
        ql_i: jax.Array,
        qi_i: jax.Array,
    ) -> jax.Array:
        epsilon = (
            config.dry_air_gas_constant
            / config.water_vapor_gas_constant
        )
        baseline_factor = (
            1.0 + qv_i / epsilon
        ) / (1.0 + qv_i)
        mixture_factor = (
            1.0
            + qv_i / epsilon
            + yn2_i
            * config.nitrogen_gas_constant
            / config.dry_air_gas_constant
        ) / (1.0 + qv_i + yn2_i)
        anomaly_i = (
            mixture_factor / baseline_factor - 1.0
        ) - ql_i - qi_i
        anomaly_h = _copy_boundary_halo(anomaly_i, axis_name, ndev)
        anomaly_face = 0.5 * (
            anomaly_h + _shift_z_plus(anomaly_h)
        )
        buoyancy = (
            params.g * params.z_i * anomaly_face[:, :, 1:-1]
        ).astype(params.dtype)
        buoyancy = _set_w_physical_boundaries(
            buoyancy, axis_name, ndev
        )
        return buoyancy

    z_spec = z_slab_spec(axis_name)
    return _shard_map(
        local_buoyancy,
        mesh=mesh,
        in_specs=(z_spec, z_spec, z_spec, z_spec),
        out_specs=z_spec,
        axis_name=axis_name,
    )


def make_fog_settling_sharded(
    params: Params,
    config: CryogenicMicrophysicsConfig,
    mesh: Mesh,
    axis_name: str = "z",
):
    """Return conservative downward settling tendencies for liquid and ice fog."""

    ndev = mesh_size(mesh, axis_name)
    liquid_speed = stokes_terminal_velocity(
        config.liquid_fog_diameter,
        config.water_density,
        config,
    )
    ice_speed = stokes_terminal_velocity(
        config.ice_fog_diameter,
        config.ice_density,
        config,
    )

    def local_settling(
        ql_i: jax.Array,
        qi_i: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        rank = lax.axis_index(axis_name)

        def tendency(q_i: jax.Array, speed: float) -> jax.Array:
            q_h = _copy_boundary_halo(q_i, axis_name, ndev)
            upper_flux = -jnp.asarray(
                speed, dtype=params.dtype
            ) * _shift_z_plus(q_h)
            upper_flux = jnp.where(
                rank == ndev - 1,
                upper_flux.at[:, :, -2].set(0.0),
                upper_flux,
            )
            lower_flux = _shift_z_minus(upper_flux)
            return (
                -(upper_flux - lower_flux)[:, :, 1:-1]
                / params.dz
            ).astype(params.dtype)

        return tendency(ql_i, liquid_speed), tendency(qi_i, ice_speed)

    z_spec = z_slab_spec(axis_name)
    return _shard_map(
        local_settling,
        mesh=mesh,
        in_specs=(z_spec, z_spec),
        out_specs=(z_spec, z_spec),
        axis_name=axis_name,
    )


def make_step_cryogenic_sharded(
    spray_config: SprayDPMConfig,
    microphysics_config: CryogenicMicrophysicsConfig,
    params: Params,
    operators: ShardedOperators,
    mesh: Mesh,
    axis_name: str = "z",
):
    """Return one coupled LN2-droplet, humid-carrier, and fog step.

    The existing spray exchange is reused conservatively, but its phase-mass
    increment is routed to ``yn2`` instead of water vapour.  Water vapour,
    liquid fog, and ice fog then undergo a local saturation adjustment at
    fixed moist enthalpy.
    """

    if spray_config.material != "nitrogen":
        raise ValueError("cryogenic coupling requires spray material='nitrogen'")
    if not params.thermo_enabled or not params.moisture_enabled:
        raise ValueError(
            "cryogenic coupling requires thermo and moisture transport"
        )

    inject = make_inject_sharded_spray(
        spray_config, params, mesh, axis_name
    )
    exchange = make_spray_exchange_sharded(
        spray_config, params, mesh, axis_name
    )
    flow_step = make_step_ab2_sharded(
        params, operators, mesh, axis_name
    )
    scalar_params = replace(
        params,
        surface_theta_flux=0.0,
        surface_qv_flux=0.0,
        theta_top_gradient=0.0,
    )
    scalar_rhs = make_scalar_rhs_buoyancy_sharded(
        scalar_params,
        operators.horizontal,
        mesh,
        axis_name,
    )
    positivity = make_apply_moisture_bounds_sharded(mesh, axis_name)
    composition_buoyancy = make_cryogenic_buoyancy_sharded(
        params, microphysics_config, mesh, axis_name
    )
    fog_settling = make_fog_settling_sharded(
        params, microphysics_config, mesh, axis_name
    )
    physical_cell_volume = (
        params.dx * params.dy * params.dz * params.z_i**3
    )
    cell_air_mass = (
        microphysics_config.dry_air_density * physical_cell_volume
    )
    x_coordinates = (
        jnp.arange(params.nx, dtype=params.dtype) + 0.5
    ) * params.dx * params.z_i
    zero_floor = jnp.asarray(0.0, dtype=params.dtype)

    def step(
        state: ShardedCryogenicState,
        runtime_pressure_ops=None,
        runtime_spike_ops=None,
    ) -> tuple[ShardedCryogenicState, ShardedCryogenicDiagnostics]:
        injected = inject(state.spray, state.flow.step)
        spray, increments, spray_diagnostics = exchange(
            state.flow, injected
        )
        yn2_forced = state.scalars.yn2 + increments.qv
        carrier_heat_capacity = (
            microphysics_config.dry_air_heat_capacity
            + state.scalars.yn2
            * microphysics_config.nitrogen_gas_heat_capacity
        )
        latent_cooled_theta = state.flow.theta + (
            increments.theta
            * microphysics_config.dry_air_heat_capacity
            / carrier_heat_capacity
        )
        enthalpy_forced = (
            state.scalars.enthalpy
            + increments.theta
            * microphysics_config.dry_air_heat_capacity
            + jnp.maximum(increments.qv, 0.0)
            * microphysics_config.nitrogen_gas_heat_capacity
            * microphysics_config.nitrogen_boiling_temperature
        )
        mixed_heat_capacity = (
            microphysics_config.dry_air_heat_capacity
            + yn2_forced
            * microphysics_config.nitrogen_gas_heat_capacity
        )
        mixed_theta = (
            enthalpy_forced
            - microphysics_config.water_vapor_latent_heat
            * state.flow.qv
            + microphysics_config.water_fusion_latent_heat
            * state.scalars.qi
        ) / mixed_heat_capacity
        forced_flow = state.flow._replace(
            u=state.flow.u + increments.u,
            v=state.flow.v + increments.v,
            w=state.flow.w + increments.w,
            theta=mixed_theta.astype(params.dtype),
        )

        evaporation_mass_rate = (
            increments.qv
            * microphysics_config.dry_air_density
            / params.dt_physical
        )
        outlet = mass_only_outlet_update(
            evaporation_mass_rate,
            forced_flow.theta,
            yn2_forced,
            x_coordinates,
            physical_cell_volume,
            microphysics_config,
        )
        target_divergence = (
            outlet.target_divergence * params.z_i
        ).astype(params.dtype)
        target_divergence = target_divergence - jnp.mean(
            target_divergence
        )
        outlet_sink_internal = (
            params.z_i * outlet.volume_sink
        ).astype(params.dtype)

        (
            yn2_transport,
            ql_transport,
            rhs_yn2,
            rhs_ql,
            _,
            _,
        ) = scalar_rhs(
            forced_flow.u,
            forced_flow.v,
            forced_flow.w,
            yn2_forced,
            state.scalars.ql,
            forced_flow.cs2,
        )
        zeros = jnp.zeros_like(state.scalars.qi)
        (
            qi_transport,
            _,
            rhs_qi,
            _,
            _,
            _,
        ) = scalar_rhs(
            forced_flow.u,
            forced_flow.v,
            forced_flow.w,
            state.scalars.qi,
            zeros,
            forced_flow.cs2,
        )
        settling_ql, settling_qi = fog_settling(
            state.scalars.ql,
            state.scalars.qi,
        )
        rhs_ql = rhs_ql + settling_ql
        rhs_qi = rhs_qi + settling_qi
        (
            enthalpy_transport,
            _,
            rhs_enthalpy,
            _,
            _,
            _,
        ) = scalar_rhs(
            forced_flow.u,
            forced_flow.v,
            forced_flow.w,
            enthalpy_forced,
            zeros,
            forced_flow.cs2,
        )
        rhs_yn2 = rhs_yn2 + (
            params.z_i * outlet.nitrogen_tendency
        ) - outlet_sink_internal * yn2_forced
        rhs_enthalpy = (
            rhs_enthalpy
            - microphysics_config.water_fusion_latent_heat
            * settling_qi
            - outlet_sink_internal * enthalpy_forced
            + params.z_i
            * outlet.nitrogen_tendency
            * microphysics_config.nitrogen_gas_heat_capacity
            * forced_flow.theta
        )
        yn2_new = _ab_update_inner(
            yn2_transport,
            rhs_yn2,
            state.scalars.rhs_yn2_prev,
            state.flow.step,
            params,
        )
        ql_new = _ab_update_inner(
            ql_transport,
            rhs_ql,
            state.scalars.rhs_ql_prev,
            state.flow.step,
            params,
        )
        qi_new = _ab_update_inner(
            qi_transport,
            rhs_qi,
            state.scalars.rhs_qi_prev,
            state.flow.step,
            params,
        )
        enthalpy_new = _ab_update_inner(
            enthalpy_transport,
            rhs_enthalpy,
            state.scalars.rhs_enthalpy_prev,
            state.flow.step,
            params,
        )
        yn2_new = positivity(yn2_new, zero_floor)
        ql_new = positivity(ql_new, zero_floor)
        qi_new = positivity(qi_new, zero_floor)

        extra_rhs_w = composition_buoyancy(
            yn2_forced,
            forced_flow.qv,
            state.scalars.ql,
            state.scalars.qi,
        )
        advanced_flow = flow_step(
            forced_flow,
            runtime_pressure_ops,
            runtime_spike_ops,
            extra_rhs_w=extra_rhs_w,
            extra_rhs_theta=(
                -outlet_sink_internal * forced_flow.theta
            ),
            extra_rhs_qv=(
                -outlet_sink_internal * forced_flow.qv
            ),
            target_divergence=target_divergence,
        )
        mixture_heat_capacity = (
            microphysics_config.dry_air_heat_capacity
            + yn2_new
            * microphysics_config.nitrogen_gas_heat_capacity
        )
        enthalpy_temperature = (
            enthalpy_new
            - microphysics_config.water_vapor_latent_heat
            * advanced_flow.qv
            + microphysics_config.water_fusion_latent_heat * qi_new
        ) / mixture_heat_capacity
        fog_update = advance_fog_microphysics(
            enthalpy_temperature,
            advanced_flow.qv,
            ql_new,
            qi_new,
            params.dt_physical,
            microphysics_config,
            heat_capacity=(
                microphysics_config.dry_air_heat_capacity
                + yn2_new
                * microphysics_config.nitrogen_gas_heat_capacity
            ),
        )
        advanced_flow = advanced_flow._replace(
            theta=fog_update.temperature.astype(params.dtype),
            qv=fog_update.qv.astype(params.dtype),
        )
        scalars = CryogenicScalarState(
            yn2=yn2_new.astype(params.dtype),
            ql=fog_update.ql.astype(params.dtype),
            qi=fog_update.qi.astype(params.dtype),
            enthalpy=enthalpy_new.astype(params.dtype),
            rhs_yn2_prev=rhs_yn2.astype(params.dtype),
            rhs_ql_prev=rhs_ql.astype(params.dtype),
            rhs_qi_prev=rhs_qi.astype(params.dtype),
            rhs_enthalpy_prev=rhs_enthalpy.astype(params.dtype),
        )
        qsat = saturation_mixing_ratio(
            advanced_flow.theta,
            microphysics_config.pressure,
            microphysics_config,
        )
        diagnostics = ShardedCryogenicDiagnostics(
            spray=spray_diagnostics,
            nitrogen_gas_mass=jnp.sum(scalars.yn2) * cell_air_mass,
            liquid_fog_mass=jnp.sum(scalars.ql) * cell_air_mass,
            ice_fog_mass=jnp.sum(scalars.qi) * cell_air_mass,
            total_water_mass=jnp.sum(
                advanced_flow.qv + scalars.ql + scalars.qi
            )
            * cell_air_mass,
            max_relative_humidity=jnp.max(
                advanced_flow.qv
                / jnp.maximum(
                    qsat,
                    jnp.asarray(1.0e-12, dtype=params.dtype),
                )
            ),
            outlet_volume_rate=jnp.sum(outlet.volume_sink)
            * physical_cell_volume,
            nitrogen_sensible_cooling=jnp.sum(
                (latent_cooled_theta - mixed_theta)
                * carrier_heat_capacity
            )
            * cell_air_mass,
            nitrogen_outlet_mass=jnp.sum(
                -outlet.nitrogen_tendency
                + outlet.volume_sink * yn2_forced
            )
            * microphysics_config.dry_air_density
            * physical_cell_volume
            * params.dt_physical,
            fog_condensed_mass=jnp.sum(
                fog_update.condensed_or_deposited
            )
            * cell_air_mass,
            fog_evaporated_mass=jnp.sum(
                fog_update.evaporated_or_sublimated
            )
            * cell_air_mass,
        )
        return (
            ShardedCryogenicState(
                flow=advanced_flow,
                spray=spray,
                scalars=scalars,
            ),
            diagnostics,
        )

    return step
