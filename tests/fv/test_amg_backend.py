from __future__ import annotations

import importlib.util
import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    CLASSICAL_AMG_PCG,
    StaggeredVelocity,
    build_pressure_poisson,
    divergence,
    project,
)


HAS_JAXAMG = importlib.util.find_spec("jaxamg") is not None
# AmgX allocates outside the JAX memory pool, so a run also needs
# XLA_PYTHON_CLIENT_PREALLOCATE=false to leave it room on the device.
ON_GPU = HAS_JAXAMG and jax.default_backend() == "gpu"


class AmgConfigurationTest(unittest.TestCase):
    def test_the_default_is_classical_amg_preconditioned_conjugate_gradients(
        self,
    ) -> None:
        self.assertEqual(CLASSICAL_AMG_PCG["solver"], "PCG")
        preconditioner = CLASSICAL_AMG_PCG["preconditioner"]
        self.assertEqual(preconditioner["solver"], "AMG")
        self.assertEqual(preconditioner["algorithm"], "CLASSICAL")

    @unittest.skipUnless(HAS_JAXAMG, "jaxamg is not installed")
    def test_jaxamg_accepts_the_default_configuration(self) -> None:
        """The config must survive JAX-AMG's own validator, GPU or not."""
        from jaxamg import config as amgx_config

        prepared = amgx_config.prepare_config(dict(CLASSICAL_AMG_PCG))
        self.assertIn('"PCG"', prepared)
        self.assertIn('"CLASSICAL"', prepared)

    @unittest.skipIf(HAS_JAXAMG, "jaxamg is installed")
    def test_a_missing_jaxamg_names_the_submodule(self) -> None:
        grid = UniformGrid(4, 4, 4, 1.0, 1.0, 1.0)
        with self.assertRaises(ImportError) as raised:
            build_pressure_poisson(grid, backend="amg")
        self.assertIn("external/jax-amg", str(raised.exception))


@unittest.skipUnless(ON_GPU, "the AMG backend needs jaxamg on a GPU")
class AmgSolveTest(unittest.TestCase):
    """The one path that only a GPU with AmgX can exercise."""

    grid = UniformGrid(16, 16, 8, 2.0, 2.0, 1.0)

    def velocity(self) -> StaggeredVelocity:
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        cells = (self.grid.nz, self.grid.ny, self.grid.nx)
        return StaggeredVelocity(
            jax.random.normal(keys[0], cells),
            jax.random.normal(keys[1], cells),
            jax.random.normal(keys[2], (self.grid.nz + 1, self.grid.ny, self.grid.nx))
            .at[0]
            .set(0.0)
            .at[-1]
            .set(0.0),
        )

    def test_amgx_reproduces_the_reference_solution(self) -> None:
        right_hand_side = divergence(self.velocity(), self.grid)
        amg = build_pressure_poisson(self.grid, backend="amg").solve(right_hand_side)
        reference = build_pressure_poisson(self.grid, backend="fft").solve(
            right_hand_side
        )
        scale = float(jnp.max(jnp.abs(reference)))
        self.assertGreater(scale, 0.0)
        self.assertLess(float(jnp.max(jnp.abs(amg - reference))), 1.0e-7 * scale)

    def test_the_amg_projection_removes_divergence(self) -> None:
        velocity = self.velocity()
        poisson = build_pressure_poisson(self.grid, backend="amg")
        before = float(jnp.max(jnp.abs(divergence(velocity, self.grid))))
        projected, _ = project(velocity, poisson, 0.05)
        after = float(jnp.max(jnp.abs(divergence(projected, self.grid))))
        self.assertLess(after, 1.0e-8 * before)

    def test_the_amg_solve_runs_under_jit(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="amg")
        compiled = jax.jit(lambda field: project(field, poisson, 0.05)[0])
        projected = compiled(self.velocity())
        self.assertTrue(bool(jnp.all(jnp.isfinite(projected.x))))
        self.assertLess(
            float(jnp.max(jnp.abs(divergence(projected, self.grid)))),
            1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
