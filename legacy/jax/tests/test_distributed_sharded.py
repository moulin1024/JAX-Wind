from __future__ import annotations

import sys
from pathlib import Path

import pytest


JAX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JAX_ROOT))


def test_sharded_pressure_operator_path_never_builds_reference_global_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax import pressure_sharded
    from wireles_jax.sharding import make_single_node_mesh

    def fail_global_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("global pressure operator builder was called")

    monkeypatch.setattr(
        pressure_sharded,
        "_pressure_tridiag_fortran_layout",
        fail_global_builder,
    )
    params = Params(nx=8, ny=8, nz=8, dtype=jnp.float32)
    operators = pressure_sharded.make_sharded_pressure_operators(
        params, make_single_node_mesh(1)
    )
    assert operators.pressure_a.shape == (5, 8, 8)


def test_distributed_initial_state_contains_thermo_without_global_initializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax import init
    from wireles_jax.sharding import make_single_node_mesh
    from wireles_jax.timestep_sharded import initial_sharded_state

    def fail_global_initializer(*args, **kwargs):
        del args, kwargs
        raise AssertionError("global initial_state was called")

    monkeypatch.setattr(init, "initial_state", fail_global_initializer)
    params = Params(
        nx=8,
        ny=8,
        nz=8,
        thermo_enabled=True,
        scalar_sgs_model="fixed_prandtl",
        scalar_vertical_scheme="centered",
        surface_theta_flux=0.01,
        dtype=jnp.float32,
    )
    state = initial_sharded_state(params, make_single_node_mesh(1), seed=4)
    assert state.theta.shape == (8, 8, 8)
    assert state.scalar_c.shape == (8, 8, 8, 2)
    assert float(jnp.min(state.theta)) > 0.0


def test_sharded_momentum_rhs_matches_single_domain() -> None:
    """Guard staggered face/center and inter-slab halo semantics."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import NamedSharding, PartitionSpec as P

    from wireles_jax import Params
    from wireles_jax.grid import make_operators
    from wireles_jax.init import initial_state
    from wireles_jax.sharding import make_single_node_mesh, put_z_slab
    from wireles_jax.timestep import _momentum_rhs
    from wireles_jax.timestep_sharded import make_momentum_rhs_sharded

    params = Params(
        nx=8,
        ny=8,
        nz=8,
        lx=2.0,
        ly=2.0,
        lz=1.0,
        dt=1.0e-3,
        time_scheme="ab2",
        momentum_wall_model="abl",
        wall_stress_model="dynamic",
        initial_condition="log_law",
        initial_velocity_noise=0.0,
        sgs_model="lasd",
        cs_count=5,
        horizontal_dealias=True,
        dtype=jnp.float32,
    )
    ops = make_operators(params)
    state = initial_state(params, seed=0)
    key = jax.random.PRNGKey(7)
    u = state.u + 0.02 * jax.random.normal(key, state.u.shape)
    v = state.v + 0.02 * jax.random.normal(
        jax.random.fold_in(key, 1), state.v.shape
    )
    w = 0.01 * jax.random.normal(
        jax.random.fold_in(key, 2), state.w.shape
    )
    w = w.at[:, :, -1].set(0.0)
    state = state._replace(u=u, v=v, w=w)
    expected = _momentum_rhs(state, params, ops, update_lasd=False)

    # Exercise an actual slab interface whenever the test environment exposes
    # at least two devices, while remaining useful on ordinary one-CPU CI.
    mesh = make_single_node_mesh(min(2, jax.device_count()))

    def slab(q):
        return put_z_slab(q, mesh)

    scalar_c = jax.device_put(
        state.scalar_c,
        NamedSharding(mesh, P(None, None, "z", None)),
    )
    sharded_rhs = make_momentum_rhs_sharded(params, ops, mesh)
    actual = jax.block_until_ready(
        sharded_rhs(
            slab(u),
            slab(v),
            slab(w),
            slab(state.theta),
            slab(state.qv),
            slab(state.cs2),
            slab(state.lm_old),
            slab(state.mm_old),
            slab(state.qn_old),
            slab(state.nn_old),
            scalar_c,
            slab(state.u_lag),
            slab(state.v_lag),
            slab(state.w_lag),
            state.step,
        )
    )

    for index in (0, 1, 2, 5, 6, 7):
        assert float(jnp.max(jnp.abs(actual[index] - expected[index]))) == pytest.approx(
            0.0, abs=2.0e-6
        )


def test_sharded_scalar_surface_flux_is_conservative() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.grid import make_operators
    from wireles_jax.sharding import make_single_node_mesh, put_z_slab
    from wireles_jax.timestep_sharded import make_scalar_rhs_buoyancy_sharded

    params = Params(
        nx=8,
        ny=8,
        nz=8,
        lx=2.0,
        ly=2.0,
        lz=1.0,
        thermo_enabled=True,
        surface_theta_flux=0.06,
        scalar_sgs_model="fixed_prandtl",
        scalar_vertical_scheme="centered",
        momentum_wall_model="free_slip",
        dtype=jnp.float32,
    )
    mesh = make_single_node_mesh(1)
    shape = (params.nx, params.ny, params.nz)
    zeros = put_z_slab(jnp.zeros(shape, dtype=params.dtype), mesh)
    theta = put_z_slab(jnp.full(shape, params.theta0, dtype=params.dtype), mesh)
    qv = put_z_slab(jnp.zeros(shape, dtype=params.dtype), mesh)
    cs2 = put_z_slab(jnp.zeros(shape, dtype=params.sgs_dtype), mesh)
    scalar_rhs = make_scalar_rhs_buoyancy_sharded(
        params, make_operators(params), mesh
    )
    _, _, rhs_theta, rhs_qv, buoyancy, _ = jax.block_until_ready(
        scalar_rhs(zeros, zeros, zeros, theta, qv, cs2)
    )

    assert float(jnp.mean(rhs_theta)) == pytest.approx(
        params.surface_theta_flux / params.lz, rel=2.0e-6, abs=2.0e-7
    )
    assert float(jnp.max(jnp.abs(buoyancy))) == pytest.approx(0.0, abs=1.0e-7)
    assert float(jnp.max(jnp.abs(rhs_qv))) == pytest.approx(0.0, abs=1.0e-7)


def test_sharded_moisture_surface_flux_is_conservative() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.grid import make_operators
    from wireles_jax.sharding import make_single_node_mesh, put_z_slab
    from wireles_jax.timestep_sharded import make_scalar_rhs_buoyancy_sharded

    params = Params(
        nx=8,
        ny=8,
        nz=8,
        lx=2.0,
        ly=2.0,
        lz=1.0,
        thermo_enabled=True,
        moisture_enabled=True,
        qv0=0.004,
        surface_qv_flux=2.0e-5,
        scalar_sgs_model="fixed_prandtl",
        scalar_vertical_scheme="centered",
        momentum_wall_model="free_slip",
        dtype=jnp.float32,
    )
    mesh = make_single_node_mesh(1)
    shape = (params.nx, params.ny, params.nz)
    zeros = put_z_slab(jnp.zeros(shape, dtype=params.dtype), mesh)
    theta = put_z_slab(jnp.full(shape, params.theta0, dtype=params.dtype), mesh)
    qv = put_z_slab(jnp.full(shape, params.qv0, dtype=params.dtype), mesh)
    cs2 = put_z_slab(jnp.zeros(shape, dtype=params.sgs_dtype), mesh)
    scalar_rhs = make_scalar_rhs_buoyancy_sharded(
        params, make_operators(params), mesh
    )
    _, _, _, rhs_qv, buoyancy, _ = jax.block_until_ready(
        scalar_rhs(zeros, zeros, zeros, theta, qv, cs2)
    )

    assert float(jnp.mean(rhs_qv)) == pytest.approx(
        params.surface_qv_flux / params.lz,
        rel=2.0e-6,
        abs=2.0e-8,
    )
    assert float(jnp.max(jnp.abs(buoyancy))) == pytest.approx(0.0, abs=1.0e-7)


def test_spike_pressure_matches_transpose_pressure() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.pressure_sharded import (
        make_pressure_hat_solver_z_sharded,
        make_pressure_hat_solver_z_sharded_spike,
        make_sharded_pressure_operators,
        make_sharded_spike_operators,
    )
    from wireles_jax.sharding import (
        make_single_node_mesh,
        put_z_slab,
        rfft2_fortran_layout,
    )

    params = Params(
        nx=8,
        ny=8,
        nz=8,
        lx=2.0,
        ly=2.0,
        lz=1.0,
        pressure_filter_nyquist=True,
        momentum_wall_model="free_slip",
        dtype=jnp.float32,
    )
    mesh = make_single_node_mesh(1)
    transpose_ops = make_sharded_pressure_operators(params, mesh)
    spike_ops = make_sharded_spike_operators(params, mesh)
    rhs = put_z_slab(
        jax.random.normal(
            jax.random.PRNGKey(91),
            (params.nx, params.ny, params.nz),
            dtype=params.dtype,
        ),
        mesh,
    )
    rhs_hat = rfft2_fortran_layout(rhs)
    transpose = make_pressure_hat_solver_z_sharded(
        params, transpose_ops, mesh
    )
    spike = make_pressure_hat_solver_z_sharded_spike(params, mesh)
    expected, actual = jax.block_until_ready(
        (transpose(rhs_hat, transpose_ops), spike(rhs_hat, spike_ops))
    )
    assert float(jnp.max(jnp.abs(actual - expected))) == pytest.approx(
        0.0, abs=2.0e-5
    )


def test_concurrent_fringe_accepts_only_the_transmitted_x_tail() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.sharding import make_single_node_mesh
    from wireles_jax.timestep_sharded import make_concurrent_fringe_sources_sharded

    params = Params(
        nx=8,
        ny=4,
        nz=4,
        lx=8.0,
        ly=4.0,
        lz=4.0,
        z_i=1.0,
        dt=1.0e-3,
        time_scheme="ab2",
        momentum_wall_model="free_slip",
        thermo_enabled=True,
        scalar_sgs_model="fixed_prandtl",
        fringe_enabled=True,
        fringe_start_x=6.0,
        fringe_timescale=1.0,
        dtype=jnp.float32,
    )
    mesh = make_single_node_mesh(1)
    source = make_concurrent_fringe_sources_sharded(params, mesh)
    shape = (params.nx, params.ny, params.nz)
    current = jnp.zeros(shape, dtype=params.dtype)
    target_tail = jnp.ones((2, params.ny, params.nz), dtype=params.dtype)
    zeros_tail = jnp.zeros_like(target_tail)
    source_u, source_v, source_w, source_theta, source_qv = source(
        current,
        current,
        current,
        current,
        current,
        target_tail,
        zeros_tail,
        zeros_tail,
        zeros_tail,
        zeros_tail,
    )
    assert float(jnp.max(jnp.abs(source_u[:6]))) == 0.0
    assert float(jnp.max(source_u[6:])) > 0.0
    assert bool(jnp.allclose(source_u[6], source_u[7], rtol=1.0e-6))
    assert float(jnp.max(jnp.abs(source_v))) == 0.0
    assert float(jnp.max(jnp.abs(source_w))) == 0.0
    assert float(jnp.max(jnp.abs(source_theta))) == 0.0
    assert float(jnp.max(jnp.abs(source_qv))) == 0.0


def test_sharded_rayleigh_sponge_matches_single_domain_semantics() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.sharding import make_single_node_mesh, put_z_slab
    from wireles_jax.sponge import apply_rayleigh_sponge
    from wireles_jax.timestep_sharded import make_apply_rayleigh_sponge_sharded

    params = Params(
        nx=8,
        ny=4,
        nz=8,
        lx=4.0,
        ly=2.0,
        lz=2.0,
        z_i=1.0,
        dt=0.01,
        sponge_enabled=True,
        sponge_start_height=1.0,
        sponge_timescale=0.5,
        sponge_power=2.0,
        sponge_target="geostrophic",
        geostrophic_u=5.0,
        geostrophic_v=0.25,
        dtype=jnp.float32,
    )
    key = jax.random.PRNGKey(31)
    u = 4.0 + jax.random.normal(key, (8, 4, 8), dtype=params.dtype)
    v = 0.1 * u
    w = 0.2 * u
    expected = apply_rayleigh_sponge(u, v, w, params)
    mesh = make_single_node_mesh(1)
    sponge = make_apply_rayleigh_sponge_sharded(params, mesh)
    actual = sponge(
        put_z_slab(u, mesh), put_z_slab(v, mesh), put_z_slab(w, mesh)
    )
    for expected_field, actual_field in zip(expected, actual, strict=True):
        assert float(jnp.max(jnp.abs(expected_field - actual_field))) == pytest.approx(
            0.0, abs=2.0e-7
        )
