from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


JAX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JAX_ROOT))


def _case(*, initial_parcels: int = 2, parcels_per_step: int = 0):
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params, SprayDPMConfig
    from wireles_jax.sharding import make_single_node_mesh

    params = Params(
        nx=4,
        ny=4,
        nz=4,
        lx=1.0,
        ly=1.0,
        lz=1.0,
        z_i=100.0,
        dt=0.01,
        dtype=jnp.float32,
    )
    config = SprayDPMConfig(
        max_parcels=8,
        initial_parcels=initial_parcels,
        parcels_per_step=parcels_per_step,
        mass_flow_rate=0.2,
        injection_x=25.0,
        injection_y=25.0,
        injection_z=25.0,
        diameter_distribution="tabulated",
        tabulated_diameters=(50.0e-6, 150.0e-6),
        tabulated_mass_fractions=(0.25, 0.75),
    )
    return params, config, make_single_node_mesh(1)


def test_sharded_initialization_builds_only_local_fixed_capacity_buffer() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import initialize_sharded_spray

    params, config, mesh = _case()
    spray = initialize_sharded_spray(config, params, mesh, seed=7)

    assert spray.x.shape == (config.max_parcels,)
    assert spray.x.addressable_shards[0].data.shape == (config.max_parcels,)
    assert int(jnp.sum(spray.active)) == config.initial_parcels
    assert bool(jnp.all(spray.diameter > 0.0))


def test_packed_migration_state_preserves_large_ids_and_sgs_velocity() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import initialize_sharded_spray
    from wireles_jax.spray_dpm_sharded import _pack_spray, _unpack_spray

    params, config, mesh = _case()
    spray = initialize_sharded_spray(config, params, mesh, seed=7)
    parcel_ids = jnp.asarray(
        [0, 16_777_217, 4_000_000_000, 65_537, 9, 10, 11, 12],
        dtype=jnp.uint32,
    )
    spray = spray._replace(
        parcel_id=parcel_ids,
        sgs_u=jnp.linspace(-1.0, 1.0, config.max_parcels),
        sgs_v=jnp.linspace(2.0, 3.0, config.max_parcels),
        sgs_w=jnp.linspace(-4.0, -2.0, config.max_parcels),
    )
    unpacked = _unpack_spray(_pack_spray(spray))

    np.testing.assert_array_equal(
        np.asarray(unpacked.parcel_id), np.asarray(parcel_ids)
    )
    np.testing.assert_array_equal(np.asarray(unpacked.sgs_u), np.asarray(spray.sgs_u))
    np.testing.assert_array_equal(np.asarray(unpacked.sgs_v), np.asarray(spray.sgs_v))
    np.testing.assert_array_equal(np.asarray(unpacked.sgs_w), np.asarray(spray.sgs_w))
    np.testing.assert_array_equal(
        np.asarray(unpacked.solute_mass), np.asarray(spray.solute_mass)
    )
    np.testing.assert_array_equal(
        np.asarray(unpacked.residual_volume), np.asarray(spray.residual_volume)
    )


def test_sharded_migration_removes_physical_boundary_exits() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import initialize_sharded_spray, make_migrate_sharded_spray

    params, config, mesh = _case()
    spray = initialize_sharded_spray(config, params, mesh, seed=8)
    initial_mass = jnp.sum(
        spray.mass * spray.weight * spray.active.astype(spray.mass.dtype)
    )
    spray = spray._replace(z=jnp.where(spray.active, -1.0, spray.z))

    migrate = jax.jit(make_migrate_sharded_spray(config, params, mesh))
    migrated, diagnostics = jax.block_until_ready(migrate(spray))

    assert int(jnp.sum(migrated.active)) == 0
    assert int(diagnostics.active_parcels) == 0
    assert int(diagnostics.exited_parcels) == config.initial_parcels
    assert int(diagnostics.overflow_parcels) == 0
    assert float(diagnostics.liquid_mass) == pytest.approx(0.0)
    assert float(initial_mass) > 0.0


def test_sharded_injection_preserves_requested_mass_flow() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import initialize_sharded_spray, make_inject_sharded_spray

    params, config, mesh = _case(initial_parcels=0, parcels_per_step=4)
    spray = initialize_sharded_spray(config, params, mesh, seed=9)
    inject = jax.jit(make_inject_sharded_spray(config, params, mesh))
    injected = jax.block_until_ready(inject(spray, jnp.asarray(0, jnp.int32)))
    liquid_mass = jnp.sum(
        injected.mass
        * injected.weight
        * injected.active.astype(injected.mass.dtype)
    )

    assert int(jnp.sum(injected.active)) == config.parcels_per_step
    assert float(liquid_mass) == pytest.approx(
        config.mass_flow_rate * params.dt_physical,
        rel=2.0e-6,
    )


def test_two_device_transverse_nozzle_and_cic_halo_are_conservative() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    if jax.device_count() < 2:
        pytest.skip("requires two local JAX devices")

    from wireles_jax import (
        Params,
        SprayDPMConfig,
        initialize_sharded_spray,
        make_spray_exchange_sharded,
    )
    from wireles_jax.sharding import make_single_node_mesh
    from wireles_jax.timestep_sharded import initial_sharded_state

    params = Params(
        nx=8,
        ny=8,
        nz=8,
        lx=1.0,
        ly=1.0,
        lz=2.0,
        z_i=1.0,
        dt=0.00025,
        thermo_enabled=True,
        moisture_enabled=True,
        qv0=0.0,
        scalar_sgs_model="fixed_prandtl",
        scalar_vertical_scheme="centered",
        momentum_wall_model="free_slip",
        sgs_model="smagorinsky",
        dtype=jnp.float32,
    )
    config = SprayDPMConfig(
        material="nitrogen",
        max_parcels=32,
        initial_parcels=16,
        parcel_weight=1.0e7,
        injection_x=0.25,
        injection_y=0.5,
        injection_z=1.0,
        injection_radius=0.15,
        injection_streamwise_thickness=2.0 * params.dx * params.z_i,
        injection_u=8.0,
        initial_diameter=150.0e-6,
        initial_temperature=77.34,
        boiling_temperature=77.34,
        liquid_density=808.0,
        water_density=808.0,
        liquid_heat_capacity=2040.0,
        latent_heat=199000.0,
        substeps=1,
    )
    mesh = make_single_node_mesh(2)
    spray = initialize_sharded_spray(config, params, mesh, seed=31)
    flow = initial_sharded_state(params, mesh)
    exchange = jax.jit(
        make_spray_exchange_sharded(config, params, mesh)
    )

    migrated, increments, diagnostics = jax.block_until_ready(
        exchange(flow, spray)
    )
    local_counts = [
        int(jnp.sum(shard.data))
        for shard in migrated.active.addressable_shards
    ]
    cell_mass = (
        config.air_density
        * params.dx
        * params.z_i
        * params.dy
        * params.z_i
        * params.dz
        * params.z_i
    )
    deposited_vapor = float(jnp.sum(increments.qv) * cell_mass)

    assert all(count > 0 for count in local_counts)
    assert deposited_vapor == pytest.approx(
        float(diagnostics.evaporated_mass),
        rel=5.0e-6,
    )


def test_sharded_evaporation_conserves_liquid_and_vapor_mass() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import (
        Params,
        SprayDPMConfig,
        initialize_sharded_spray,
        make_spray_exchange_sharded,
    )
    from wireles_jax.sharding import make_single_node_mesh
    from wireles_jax.timestep_sharded import initial_sharded_state

    params = Params(
        nx=4,
        ny=4,
        nz=4,
        lx=1.0,
        ly=1.0,
        lz=1.0,
        z_i=100.0,
        dt=0.001,
        thermo_enabled=True,
        moisture_enabled=True,
        qv0=0.0,
        scalar_sgs_model="fixed_prandtl",
        scalar_vertical_scheme="centered",
        momentum_wall_model="free_slip",
        sgs_model="smagorinsky",
        dtype=jnp.float32,
    )
    config = SprayDPMConfig(
        max_parcels=4,
        initial_parcels=1,
        parcel_weight=1.0e8,
        injection_x=25.0,
        injection_y=25.0,
        injection_z=50.0,
        initial_diameter=100.0e-6,
        initial_temperature=285.0,
        sky_temperature=285.0,
        substeps=2,
    )
    mesh = make_single_node_mesh(1)
    flow = initial_sharded_state(params, mesh)
    spray = initialize_sharded_spray(config, params, mesh)
    initial_liquid = jnp.sum(spray.mass * spray.weight * spray.active)
    exchange = jax.jit(make_spray_exchange_sharded(config, params, mesh))
    _, increments, diagnostics = jax.block_until_ready(exchange(flow, spray))

    cell_mass = (
        config.air_density
        * params.dx
        * params.z_i
        * params.dy
        * params.z_i
        * params.dz
        * params.z_i
    )
    deposited_vapor = jnp.sum(increments.qv) * cell_mass
    assert float(diagnostics.evaporated_mass) > 0.0
    assert float(deposited_vapor) == pytest.approx(
        float(diagnostics.evaporated_mass), rel=3.0e-6
    )
    assert float(initial_liquid) == pytest.approx(
        float(diagnostics.liquid_mass + diagnostics.evaporated_mass),
        rel=3.0e-6,
    )


def test_sharded_condensation_conserves_liquid_and_vapor_mass() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import (
        Params,
        SprayDPMConfig,
        initialize_sharded_spray,
        make_spray_exchange_sharded,
    )
    from wireles_jax.sharding import make_single_node_mesh
    from wireles_jax.timestep_sharded import initial_sharded_state

    params = Params(
        nx=4,
        ny=4,
        nz=4,
        lx=1.0,
        ly=1.0,
        lz=1.0,
        z_i=100.0,
        dt=0.0001,
        thermo_enabled=True,
        moisture_enabled=True,
        qv0=0.01,
        scalar_sgs_model="fixed_prandtl",
        scalar_vertical_scheme="centered",
        momentum_wall_model="free_slip",
        sgs_model="smagorinsky",
        dtype=jnp.float32,
    )
    config = SprayDPMConfig(
        max_parcels=4,
        initial_parcels=1,
        parcel_weight=1.0e8,
        injection_x=25.0,
        injection_y=25.0,
        injection_z=50.0,
        initial_diameter=100.0e-6,
        initial_temperature=250.0,
        liquid_emissivity=0.0,
        substeps=1,
    )
    mesh = make_single_node_mesh(1)
    flow = initial_sharded_state(params, mesh)
    spray = initialize_sharded_spray(config, params, mesh)
    initial_liquid = jnp.sum(spray.mass * spray.weight * spray.active)
    exchange = jax.jit(make_spray_exchange_sharded(config, params, mesh))
    updated, increments, diagnostics = jax.block_until_ready(
        exchange(flow, spray)
    )
    final_liquid = jnp.sum(
        updated.mass * updated.weight * updated.active
    )
    cell_mass = (
        config.air_density
        * params.dx
        * params.z_i
        * params.dy
        * params.z_i
        * params.dz
        * params.z_i
    )
    vapor_change = jnp.sum(increments.qv) * cell_mass

    assert float(diagnostics.evaporated_mass) < 0.0
    assert float(final_liquid) > float(initial_liquid)
    assert float(final_liquid - initial_liquid) == pytest.approx(
        -float(vapor_change), rel=4.0e-5
    )


def test_fully_distributed_spray_step_advects_moisture() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import (
        Params,
        ShardedSprayCoupledState,
        SprayDPMConfig,
        initialize_sharded_spray,
        make_step_spray_dpm_sharded,
    )
    from wireles_jax.sharding import make_single_node_mesh
    from wireles_jax.timestep_sharded import (
        initial_sharded_state,
        make_sharded_operators,
    )

    params = Params(
        nx=4,
        ny=4,
        nz=4,
        lx=1.0,
        ly=1.0,
        lz=1.0,
        z_i=100.0,
        dt=1.0e-4,
        thermo_enabled=True,
        moisture_enabled=True,
        qv0=0.001,
        scalar_sgs_model="fixed_prandtl",
        scalar_vertical_scheme="centered",
        momentum_wall_model="free_slip",
        sgs_model="smagorinsky",
        dtype=jnp.float32,
    )
    config = SprayDPMConfig(
        max_parcels=4,
        initial_parcels=1,
        parcel_weight=1.0e7,
        injection_x=25.0,
        injection_y=25.0,
        injection_z=50.0,
        initial_diameter=100.0e-6,
        initial_temperature=285.0,
        sky_temperature=285.0,
        substeps=1,
    )
    mesh = make_single_node_mesh(1)
    operators = make_sharded_operators(params, mesh)
    state = ShardedSprayCoupledState(
        flow=initial_sharded_state(params, mesh),
        spray=initialize_sharded_spray(config, params, mesh),
    )
    step = jax.jit(
        make_step_spray_dpm_sharded(config, params, operators, mesh)
    )
    updated, diagnostics = jax.block_until_ready(step(state))

    assert int(updated.flow.step) == 1
    assert float(diagnostics.evaporated_mass) > 0.0
    assert float(jnp.min(updated.flow.qv)) >= params.qv_floor
    assert float(jnp.sum(updated.flow.qv - state.flow.qv)) > 0.0
