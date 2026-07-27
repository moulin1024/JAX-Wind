from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    Candidate,
    Cell,
    Field,
    GlobalTestRegion,
    UniformGrid,
    VerticalBoundary,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from jaxwind.interpreters.jax_reference import (  # noqa: E402
    JaxReferencePressureSolver,
    JaxReferenceProjection,
)
from jaxwind.operators import VelocityVector, project  # noqa: E402


class ReferenceProjectionLawTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = UniformGrid(4, 4, 4, 4.0, 4.0, 4.0)
        cells = GlobalTestRegion(self.grid, Cell)
        faces = GlobalTestRegion(self.grid, ZFace)
        z = jnp.arange(self.grid.nz, dtype=jnp.float64)[:, None, None]
        zf = jnp.arange(self.grid.nz + 1, dtype=jnp.float64)[:, None, None]
        y = jnp.arange(self.grid.ny, dtype=jnp.float64)[None, :, None]
        x = jnp.arange(self.grid.nx, dtype=jnp.float64)[None, None, :]
        self.velocity = VelocityVector(
            Field(
                XVelocity,
                Cell,
                cells,
                Candidate,
                jnp.sin(0.7 * x + 0.2 * y + 0.3 * z),
            ),
            Field(
                YVelocity,
                Cell,
                cells,
                Candidate,
                jnp.cos(0.4 * x + 0.6 * y - 0.2 * z),
            ),
            Field(
                VerticalVelocity,
                ZFace,
                faces,
                Candidate,
                jnp.sin(0.5 * zf + 0.3 * y + 0.2 * x),
            ),
        )
        self.algebra = JaxReferenceProjection()
        self.solver = JaxReferencePressureSolver()
        self.boundary = VerticalBoundary(0.0, 0.0)

    def project(self, velocity):
        return project(
            velocity,
            dt=0.2,
            normal_boundary=self.boundary,
            algebra=self.algebra,
            pressure_solver=self.solver,
        )

    def test_projection_eliminates_divergence_and_fixes_gauge(self) -> None:
        result = self.project(self.velocity)

        self.assertLess(float(jnp.max(jnp.abs(result.divergence.payload))), 3.0e-13)
        self.assertLess(float(jnp.abs(jnp.mean(result.pressure.payload))), 3.0e-13)
        self.assertEqual(float(jnp.max(jnp.abs(result.velocity.z.payload[0]))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(result.velocity.z.payload[-1]))), 0.0)

    def test_projection_is_idempotent(self) -> None:
        once = self.project(self.velocity)
        twice = self.project(once.velocity)

        for component in ("x", "y", "z"):
            first = getattr(once.velocity, component).payload
            second = getattr(twice.velocity, component).payload
            self.assertLess(float(jnp.max(jnp.abs(first - second))), 4.0e-13)

    def test_pressure_gauge_does_not_change_gradient(self) -> None:
        result = self.project(self.velocity)
        shifted = type(result.pressure)(
            result.pressure.quantity,
            result.pressure.location,
            result.pressure.ownership,
            result.pressure.phase,
            result.pressure.payload + 17.0,
        )
        original = self.algebra.pressure_gradient(result.pressure)
        translated = self.algebra.pressure_gradient(shifted)

        for component in ("x", "y", "z"):
            left = getattr(original, component).payload
            right = getattr(translated, component).payload
            self.assertLess(float(jnp.max(jnp.abs(left - right))), 4.0e-13)

    def test_projection_rejects_invalid_timestep(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            project(
                self.velocity,
                dt=0.0,
                normal_boundary=self.boundary,
                algebra=self.algebra,
                pressure_solver=self.solver,
            )


if __name__ == "__main__":
    unittest.main()
