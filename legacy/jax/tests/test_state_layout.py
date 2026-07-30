from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


JAX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JAX_ROOT))


def test_persistent_state_has_one_uniform_z_slab_extent() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params, initial_state

    params = Params(
        nx=4,
        ny=6,
        nz=8,
        momentum_wall_model="free_slip",
        thermo_enabled=True,
        moisture_enabled=True,
        sgs_model="lasd",
        scalar_sgs_model="lasd",
        dtype=jnp.float32,
    )
    state = initial_state(params)

    for name, value in state._asdict().items():
        if name == "step":
            continue
        assert value.shape[:3] == (params.nx, params.ny, params.nz), name
    assert state.scalar_c.shape == (params.nx, params.ny, params.nz, 2)
    assert state.w.shape == state.u.shape
    np.testing.assert_allclose(np.asarray(state.w[:, :, -1]), 0.0)


def test_owned_upper_faces_form_a_conservative_vertical_divergence() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax.grid import divergence_upper_faces

    upper_flux = jnp.asarray([[[1.0, 3.0, -2.0, 4.0]]], dtype=jnp.float32)
    bottom_flux = jnp.asarray([[0.25]], dtype=jnp.float32)
    divergence = divergence_upper_faces(upper_flux, 0.5, bottom_flux)

    volume_integral = np.asarray(divergence).sum(axis=2) * 0.5
    np.testing.assert_allclose(
        volume_integral,
        np.asarray(upper_flux[:, :, -1] - bottom_flux),
        atol=2.0e-7,
    )


def test_center_and_face_vertical_gradients_keep_distinct_locations() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.derivative import ddz_uv_face
    from wireles_jax.grid import face_gradient_to_center

    params = Params(
        nx=1,
        ny=1,
        nz=4,
        lz=4.0,
        momentum_wall_model="free_slip",
        dtype=jnp.float32,
    )
    q = jnp.asarray([[[0.0, 1.0, 4.0, 9.0]]], dtype=params.dtype)
    face = ddz_uv_face(q, params)
    np.testing.assert_allclose(
        np.asarray(face)[0, 0],
        [1.0, 3.0, 5.0, 0.0],
    )

    wall_gradient = jnp.asarray([[7.0]], dtype=params.dtype)
    center = face_gradient_to_center(face, wall_gradient)
    np.testing.assert_allclose(
        np.asarray(center)[0, 0],
        [7.0, 2.0, 4.0, 2.5],
    )


def test_rotational_convection_uses_direct_face_shear() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax.convection import convec

    shape = (1, 1, 4)
    zero = jnp.zeros(shape, dtype=jnp.float32)
    w = jnp.asarray([[[2.0, 4.0, 6.0, 0.0]]], dtype=jnp.float32)
    dudz_face = jnp.asarray([[[3.0, 5.0, 7.0, 0.0]]], dtype=jnp.float32)
    cx, cy, cz = convec(
        zero,
        zero,
        w,
        zero,
        dudz_face,
        zero,
        zero,
        zero,
        zero,
    )
    np.testing.assert_allclose(
        np.asarray(cx)[0, 0],
        [3.0, 13.0, 31.0, 21.0],
    )
    np.testing.assert_allclose(np.asarray(cy), 0.0)
    np.testing.assert_allclose(np.asarray(cz), 0.0)


@pytest.mark.parametrize("top_boundary", ["rigid_lid", "klemp_durran"])
def test_projection_matches_the_no_ghost_divergence_operator(top_boundary: str) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.grid import make_operators
    from wireles_jax.pressure import _divergence_hat, project_velocity

    kwargs = {}
    if top_boundary == "klemp_durran":
        kwargs["theta_top_gradient"] = 0.003
    params = Params(
        nx=8,
        ny=6,
        nz=7,
        lx=4.0,
        ly=3.0,
        lz=1.5,
        z_i=1600.0,
        dt=1.25 / 1600.0,
        momentum_wall_model="free_slip",
        top_boundary_condition=top_boundary,
        theta0=300.0,
        dtype=jnp.float32,
        **kwargs,
    )
    ops = make_operators(params)
    keys = jax.random.split(jax.random.PRNGKey(17), 3)
    shape = (params.nx, params.ny, params.nz)
    u, v, w = (0.1 * jax.random.normal(key, shape) for key in keys)
    if top_boundary == "rigid_lid":
        w = w.at[:, :, -1].set(0.0)

    u, v, w, _ = jax.block_until_ready(project_velocity(u, v, w, params, ops))
    divergence = jnp.fft.irfft2(
        _divergence_hat(u, v, w, params, ops),
        s=(params.nx, params.ny),
        axes=(0, 1),
    ).real
    np.testing.assert_allclose(np.asarray(divergence), 0.0, atol=4.0e-6)


def test_one_device_slab_projection_matches_single_device_projection() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.grid import make_operators
    from wireles_jax.pressure import project_velocity
    from wireles_jax.pressure_sharded import make_sharded_pressure_operators
    from wireles_jax.sharding import make_single_node_mesh, put_z_slab
    from wireles_jax.timestep_sharded import make_project_velocity_sharded

    params = Params(
        nx=4,
        ny=4,
        nz=4,
        lx=2.0,
        ly=2.0,
        lz=1.0,
        dt=1.0e-3,
        momentum_wall_model="free_slip",
        pressure_filter_nyquist=True,
        dtype=jnp.float32,
    )
    keys = jax.random.split(jax.random.PRNGKey(29), 3)
    shape = (params.nx, params.ny, params.nz)
    u, v, w = (jax.random.normal(key, shape) for key in keys)
    w = w.at[:, :, -1].set(0.0)
    expected = jax.block_until_ready(
        project_velocity(u, v, w, params, make_operators(params))
    )

    mesh = make_single_node_mesh(1)
    pressure_ops = make_sharded_pressure_operators(params, mesh)
    project_sharded = make_project_velocity_sharded(params, pressure_ops, mesh)
    actual = jax.block_until_ready(
        project_sharded(
            put_z_slab(u, mesh),
            put_z_slab(v, mesh),
            put_z_slab(w, mesh),
        )
    )

    for single_field, slab_field in zip(expected, actual):
        np.testing.assert_allclose(
            np.asarray(slab_field),
            np.asarray(single_field),
            rtol=3.0e-5,
            atol=3.0e-5,
        )


def test_sharded_projection_accepts_zero_mean_volume_source() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.derivative import ddx, ddy
    from wireles_jax.grid import make_operators
    from wireles_jax.pressure_sharded import make_sharded_pressure_operators
    from wireles_jax.sharding import make_single_node_mesh, put_z_slab
    from wireles_jax.timestep_sharded import make_project_velocity_sharded

    params = Params(
        nx=8,
        ny=4,
        nz=4,
        lx=2.0,
        ly=1.0,
        lz=1.0,
        dt=1.0e-3,
        momentum_wall_model="free_slip",
        pressure_filter_nyquist=True,
        dtype=jnp.float32,
    )
    shape = (params.nx, params.ny, params.nz)
    x = (jnp.arange(params.nx) + 0.5) * params.dx
    target = 0.02 * jnp.sin(2.0 * jnp.pi * x / params.lx)
    target = jnp.broadcast_to(target[:, None, None], shape)
    zeros = jnp.zeros(shape, dtype=jnp.float32)
    mesh = make_single_node_mesh(1)
    pressure_ops = make_sharded_pressure_operators(params, mesh)
    project = make_project_velocity_sharded(params, pressure_ops, mesh)
    u, v, w, _ = jax.block_until_ready(
        project(
            put_z_slab(zeros, mesh),
            put_z_slab(zeros, mesh),
            put_z_slab(zeros, mesh),
            target_divergence=put_z_slab(target, mesh),
        )
    )

    ops = make_operators(params)
    dwdz = jnp.concatenate(
        (w[:, :, :1], w[:, :, 1:] - w[:, :, :-1]), axis=2
    ) / params.dz
    divergence = ddx(u, params, ops) + ddy(v, params, ops) + dwdz
    np.testing.assert_allclose(
        np.asarray(divergence),
        np.asarray(target),
        rtol=2.0e-4,
        atol=2.0e-5,
    )
