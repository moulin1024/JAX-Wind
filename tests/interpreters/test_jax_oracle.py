from __future__ import annotations

import ast
import inspect
import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    Cell,
    Evaluated,
    Field,
    GlobalTestRegion,
    PressureCorrection,
    UniformGrid,
    VerticalBoundary,
    VerticalVelocity,
    ZFace,
)
from tests.support import jax_oracle  # noqa: E402


class JaxOracleLawTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = UniformGrid(4, 3, 6, 4.0, 3.0, 3.0)
        self.cell_region = GlobalTestRegion(self.grid, Cell)
        self.face_region = GlobalTestRegion(self.grid, ZFace)

    def pressure_field(self, values) -> Field:
        return Field(
            PressureCorrection,
            Cell,
            self.cell_region,
            Evaluated,
            jnp.asarray(values, dtype=jnp.float64),
        )

    def test_gradient_of_constant_is_zero_including_boundaries(self) -> None:
        pressure = self.pressure_field(jnp.ones(self.cell_region.storage_shape) * 7.0)

        gradient = jax_oracle.pressure_gradient_z(
            pressure,
            VerticalBoundary(0.0, 0.0),
        )

        self.assertEqual(gradient.payload.shape, self.face_region.storage_shape)
        self.assertEqual(float(jnp.max(jnp.abs(gradient.payload))), 0.0)

    def test_discrete_integration_by_parts(self) -> None:
        z = jnp.arange(self.grid.nz, dtype=jnp.float64)[:, None, None]
        y = jnp.arange(self.grid.ny, dtype=jnp.float64)[None, :, None]
        x = jnp.arange(self.grid.nx, dtype=jnp.float64)[None, None, :]
        pressure = self.pressure_field(jnp.sin(0.31 * z + 0.17 * y + 0.11 * x))
        interior = jnp.cos(
            jnp.arange(self.grid.nz - 1, dtype=jnp.float64)[:, None, None]
            + 0.13 * y
            + 0.07 * x
        )
        zeros = jnp.zeros((1, self.grid.ny, self.grid.nx), jnp.float64)
        velocity = Field(
            VerticalVelocity,
            ZFace,
            self.face_region,
            Evaluated,
            jnp.concatenate((zeros, interior, zeros), axis=0),
        )

        gradient = jax_oracle.pressure_gradient_z(
            pressure,
            VerticalBoundary(0.0, 0.0),
        )
        divergence = jax_oracle.divergence_z(velocity)
        left = jnp.sum(pressure.payload * divergence.payload) * self.grid.dz
        right = jnp.sum(gradient.payload * velocity.payload) * self.grid.dz

        self.assertLess(float(jnp.abs(left + right)), 2.0e-13)

    def test_integrated_divergence_is_net_boundary_flux(self) -> None:
        z = jnp.arange(self.grid.nz + 1, dtype=jnp.float64)[:, None, None]
        y = jnp.arange(self.grid.ny, dtype=jnp.float64)[None, :, None]
        x = jnp.arange(self.grid.nx, dtype=jnp.float64)[None, None, :]
        values = 0.2 * z + 0.03 * y - 0.01 * x
        velocity = Field(
            VerticalVelocity,
            ZFace,
            self.face_region,
            Evaluated,
            values,
        )

        divergence = jax_oracle.divergence_z(velocity)
        integrated = jnp.sum(divergence.payload) * self.grid.dz
        boundary_flux = jnp.sum(values[-1] - values[0])

        self.assertLess(float(jnp.abs(integrated - boundary_flux)), 2.0e-13)

    def test_oracle_rejects_non_global_or_oversized_fields(self) -> None:
        oversized = UniformGrid(
            1,
            1,
            jax_oracle.MAX_ORACLE_CELLS + 1,
            1.0,
            1.0,
            1.0,
        )
        field = Field(
            PressureCorrection,
            Cell,
            GlobalTestRegion(oversized, Cell),
            Evaluated,
            jnp.zeros((oversized.nz, 1, 1), jnp.float64),
        )

        with self.assertRaisesRegex(ValueError, "bounded global limit"):
            jax_oracle.pressure_gradient_z(
                field,
                VerticalBoundary(0.0, 0.0),
            )

    def test_oracle_has_no_production_or_pressure_import(self) -> None:
        tree = ast.parse(inspect.getsource(jax_oracle))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        prohibited = ("jax_zslab",)
        self.assertFalse(
            any(name.endswith(prohibited) for name in imports)
        )


if __name__ == "__main__":
    unittest.main()
