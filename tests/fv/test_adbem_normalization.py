"""Annular AD-BEM normalization must survive coarse grids and z partitioning."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind._jax.wind import build_blade_element_disk_kernel
from jaxwind.domain import AnalyticalGrid, TanhMapping, UniformGrid


def _run(grid, widths, *, partitions=1, uniform=False, disk_z=7.0):
    kernel = build_blade_element_disk_kernel(
        grid=grid, axis_name="slabs", partition_count=partitions,
        periodic_x=False, periodic_y=False,
    )
    radii = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    shape = (grid.nz, grid.ny, grid.nx)
    x = np.asarray(grid.x_centers)[None, None, :]
    z = np.asarray(grid.z_centers)[:, None, None]
    u = np.broadcast_to(10.0 if uniform else 8.0 + 0.03*x + 0.07*z, shape)
    u = jnp.asarray(u, dtype=jnp.float32)
    zero = jnp.zeros_like(u)
    f32 = lambda a: jnp.asarray(a, dtype=jnp.float32)

    def local(uu, vv, ww):
        return kernel(
            uu, vv, ww, f32(16.), f32(12.), f32(disk_z),
            3, f32(.5), f32(6.), f32(2.), f32(widths), f32(radii),
            f32([1., 1., 1.]), f32([.5, .5, .5]), f32([0., 0., 0.]),
            jnp.zeros(3, dtype=jnp.int32),
            f32([-180., -90., 0., 90., 180.]),
            f32([[0., -1., 0., 1., 0.]]),
            f32([[.1, .1, .1, .1, .1]]), f32(0.), False, False,
        )

    # Named vmap exercises the same psum/pmax collectives without requiring
    # multiple physical devices in CI.
    slab_shape = (partitions, grid.nz // partitions, grid.ny, grid.nx)
    with jax.default_matmul_precision("highest"):
        values = jax.jit(jax.vmap(local, axis_name="slabs"))(
            u.reshape(slab_shape), zero.reshape(slab_shape), zero.reshape(slab_shape)
        )
    sources = [np.asarray(a).reshape(shape).astype(np.float64) for a in values[:3]]
    forces, sampled = [np.asarray(a[0], dtype=np.float64) for a in values[3:5]]
    return sources, forces, sampled, np.asarray(u, dtype=np.float64), radii


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("width", [.15, 2.0])
def test_sampling_and_force_match_float64_gaussian(mapped, width):
    grid = (
        AnalyticalGrid(8, 8, 8, 32., 24., 24.,
                       z_mapping=TanhMapping(.8, focus=.2))
        if mapped else UniformGrid(8, 8, 8, 32., 24., 24.)
    )
    widths = np.full(3, width, dtype=np.float32)
    sources, forces, sampled, u, radii = _run(grid, widths)
    # Independent direct float64 evaluation: these raw weights are representable
    # in float64, including the case where float32 loses every x weight.
    raw_x = np.exp(-((np.asarray(grid.x_centers)[None, :] - 16.) / widths[:, None])**2)
    if width == .15:
        assert np.max(raw_x) < np.finfo(np.float32).tiny
    xweights = raw_x * np.asarray(grid.x_widths)[None, :]
    xweights /= xweights.sum(axis=1, keepdims=True)
    radius = np.hypot(np.asarray(grid.z_centers)[:, None] - 7.,
                      np.asarray(grid.y_centers)[None, :] - 12.)
    radial = np.exp(-((radius[None] - radii[:, None, None]) / widths[:, None, None])**2)
    radial *= np.asarray(grid.z_widths)[None, :, None] * np.asarray(grid.y_widths)[None, None, :]
    radial /= radial.sum(axis=(1, 2), keepdims=True)
    expected = np.einsum("rzy,rx,zyx->r", radial, xweights, u)
    np.testing.assert_allclose(sampled[:, 0], expected, rtol=2e-6)
    volumes = (np.asarray(grid.z_widths)[:, None, None]
               * np.asarray(grid.y_widths)[None, :, None]
               * np.asarray(grid.x_widths)[None, None, :])
    # Check the spatial force distribution too: normalization must not broaden it.
    expected_source = np.einsum("r,rzy,rx->zyx", forces[:, 0], radial, xweights) / volumes
    np.testing.assert_allclose(sources[0], expected_source, rtol=5e-5, atol=2e-6)
    np.testing.assert_allclose(np.sum(sources[0]*volumes), forces[:, 0].sum(), rtol=2e-6)
    np.testing.assert_allclose(np.sum(u*sources[0]*volumes), np.sum(forces*sampled), rtol=2e-6)
    assert all(np.isfinite(s).all() for s in sources)


@pytest.mark.parametrize("disk_z", [7.0, 23.8])
def test_narrow_rings_preserve_uniform_inflow_across_slabs(disk_z):
    grid = UniformGrid(8, 8, 8, 32., 24., 24.)
    widths = np.full(3, .15, dtype=np.float32)
    single = _run(grid, widths, uniform=True, disk_z=disk_z)
    split = _run(grid, widths, partitions=2, uniform=True, disk_z=disk_z)
    for result in (single, split):
        np.testing.assert_allclose(result[2][:, 0], 10., rtol=2e-6)
        assert np.all(result[1][:, 0] < 0.)
        assert all(np.isfinite(s).all() for s in result[0])
        assert np.all(result[0][2][-1] == 0.)
    for a, b in zip(single[0] + list(single[1:3]), split[0] + list(split[1:3])):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)

    # Independent upper-face quadrature excludes the impermeable top face.
    # Near the top, letting that zero-area face set the log shift can still
    # underflow every usable weight of a narrow ring.
    yy = np.asarray(grid.y_centers)[None, :] - 12.
    zz = np.asarray(grid.z_faces[1:])[:, None] - disk_z
    radius = np.hypot(yy, zz)
    radial = np.exp(-((radius[None] - single[4][:, None, None]) / widths[:, None, None])**2)
    face_widths = np.full(grid.nz, grid.dz)
    face_widths[-1] = 0.
    radial *= face_widths[None, :, None] * grid.dy
    radial /= radial.sum(axis=(1, 2), keepdims=True)
    xweights = np.exp(-((np.asarray(grid.x_centers)[None, :] - 16.) / widths[:, None])**2)
    xweights /= xweights.sum(axis=1, keepdims=True)
    expected_z = np.einsum("r,rzy,zy,rx->zyx", single[1][:, 1], radial, -yy/radius, xweights)
    expected_z /= np.where(face_widths > 0., face_widths, 1.)[:, None, None] * grid.dx * grid.dy
    np.testing.assert_allclose(single[0][2], expected_z, rtol=5e-5, atol=2e-6)
