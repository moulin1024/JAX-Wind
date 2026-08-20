from __future__ import annotations

from types import SimpleNamespace
import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    InflowPlane,
    StaggeredVelocity,
    assemble_pressure_matrix,
    build_pressure_poisson,
    divergence,
    enforce_open_scalar,
    enforce_open_velocity,
    extract_inflow_plane,
    matrix_vector_product,
    periodic_to_open_velocity,
    pressure_gradient,
    project,
)


class OpenBoundaryTest(unittest.TestCase):
    grid = UniformGrid(8, 6, 4, 2.0, 1.5, 1.0)

    def periodic_state(self):
        keys = jax.random.split(jax.random.PRNGKey(91), 4)
        velocity = StaggeredVelocity(
            jax.random.normal(keys[0], (4, 6, 8)),
            jax.random.normal(keys[1], (4, 6, 8)),
            jax.random.normal(keys[2], (5, 6, 8)).at[0].set(0.0).at[-1].set(0.0),
        )
        return SimpleNamespace(
            velocity=velocity,
            scalar=jax.random.normal(keys[3], (4, 6, 8)),
        )

    def test_recording_extracts_exactly_one_yz_layer(self) -> None:
        state = self.periodic_state()
        plane = extract_inflow_plane(state, self.grid, plane=3)

        self.assertEqual(plane.x_velocity.shape, (4, 6))
        self.assertEqual(plane.y_velocity.shape, (4, 6))
        self.assertEqual(plane.z_velocity.shape, (5, 6))
        self.assertEqual(plane.scalar.shape, (4, 6))
        self.assertTrue(bool(jnp.all(plane.scalar == state.scalar[..., 3])))

    def test_inflow_and_second_order_outflow_are_enforced(self) -> None:
        state = self.periodic_state()
        plane = extract_inflow_plane(state, self.grid)
        velocity = enforce_open_velocity(
            periodic_to_open_velocity(state.velocity, self.grid),
            plane,
            self.grid,
        )
        scalar = enforce_open_scalar(state.scalar, plane, self.grid)

        self.assertEqual(velocity.x.shape, (4, 6, 9))
        self.assertTrue(bool(jnp.all(velocity.x[..., 0] == plane.x_velocity)))
        self.assertTrue(bool(jnp.all(velocity.y[..., 0] == plane.y_velocity)))
        self.assertTrue(bool(jnp.all(velocity.z[..., 0] == plane.z_velocity)))
        self.assertTrue(bool(jnp.all(scalar[..., 0] == plane.scalar)))
        for field in (velocity.x, velocity.y, velocity.z, scalar):
            expected = (4.0 * field[..., -2] - field[..., -3]) / 3.0
            self.assertLess(float(jnp.max(jnp.abs(field[..., -1] - expected))), 1e-14)


class OpenPressureTest(unittest.TestCase):
    grid = UniformGrid(8, 6, 4, 2.0, 1.5, 1.0)

    def random_open_velocity(self):
        keys = jax.random.split(jax.random.PRNGKey(92), 3)
        return StaggeredVelocity(
            jax.random.normal(keys[0], (4, 6, 9)),
            jax.random.normal(keys[1], (4, 6, 8)),
            jax.random.normal(keys[2], (5, 6, 8)).at[0].set(0.0).at[-1].set(0.0),
        )

    def test_sparse_matrix_matches_the_constrained_gradient(self) -> None:
        pressure = jax.random.normal(jax.random.PRNGKey(93), (4, 6, 8))
        matrix = assemble_pressure_matrix(
            self.grid,
            periodic_x=False,
            reference_cell=None,
        )
        applied = matrix_vector_product(matrix, pressure.reshape(-1)).reshape(
            pressure.shape
        )
        expected = -divergence(
            pressure_gradient(pressure, self.grid, periodic_x=False),
            self.grid,
        )
        self.assertLess(float(jnp.max(jnp.abs(applied - expected))), 1e-12)

    def test_open_gmg_projection_preserves_boundaries_and_divergence(self) -> None:
        velocity = self.random_open_velocity()
        poisson = build_pressure_poisson(
            self.grid,
            backend="gmg",
            periodic_x=False,
        )
        projected, _ = project(velocity, poisson, 0.05)

        self.assertLess(
            float(jnp.max(jnp.abs(divergence(projected, self.grid)))),
            1e-7,
        )
        self.assertTrue(bool(jnp.all(projected.x[..., 0] == velocity.x[..., 0])))
        self.assertTrue(bool(jnp.all(projected.y[..., 0] == velocity.y[..., 0])))
        self.assertTrue(bool(jnp.all(projected.z[..., 0] == velocity.z[..., 0])))
        self.assertTrue(bool(jnp.all(projected.y[..., -1] == velocity.y[..., -1])))
        self.assertTrue(bool(jnp.all(projected.z[..., -1] == velocity.z[..., -1])))

    def test_fft_rejects_a_nonperiodic_streamwise_domain(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "FFT pressure backend requires periodic x"
        ):
            build_pressure_poisson(self.grid, backend="fft", periodic_x=False)


if __name__ == "__main__":
    unittest.main()
