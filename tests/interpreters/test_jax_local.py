from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from wireles.domain import (  # noqa: E402
    Cell,
    Evaluated,
    Field,
    GlobalTestRegion,
    PressureCorrection,
    UniformGrid,
    VerticalBoundary,
)
from wireles.interpreters import jax_local, jax_reference  # noqa: E402


class JaxLocalCommutingTests(unittest.TestCase):
    def test_local_upper_face_storage_commutes_with_reference(self) -> None:
        grid = UniformGrid(5, 4, 7, 5.0, 4.0, 3.5)
        z = jnp.arange(grid.nz, dtype=jnp.float64)[:, None, None]
        y = jnp.arange(grid.ny, dtype=jnp.float64)[None, :, None]
        x = jnp.arange(grid.nx, dtype=jnp.float64)[None, None, :]
        values = jnp.sin(0.23 * z + 0.17 * y + 0.11 * x)
        pressure = Field(
            PressureCorrection,
            Cell,
            GlobalTestRegion(grid, Cell),
            Evaluated,
            values,
        )
        boundary = VerticalBoundary(0.375, -0.25)
        reference_gradient = jax_reference.pressure_gradient_z(pressure, boundary)
        reference_divergence = jax_reference.divergence_z(reference_gradient)

        local_gradient = jax_local.pressure_gradient_upper_faces(
            values,
            dz=grid.dz,
            last_upper_gradient=boundary.upper,
        )
        local_divergence = jax_local.divergence_from_upper_faces(
            local_gradient,
            dz=grid.dz,
            lower_face=boundary.lower,
        )

        self.assertLess(
            float(jnp.max(jnp.abs(local_gradient - reference_gradient.payload[1:]))),
            2.0e-13,
        )
        self.assertLess(
            float(jnp.max(jnp.abs(local_divergence - reference_divergence.payload))),
            2.0e-13,
        )

    def test_local_kernels_are_jittable_without_semantic_change(self) -> None:
        values = jnp.arange(4 * 3 * 2, dtype=jnp.float64).reshape(4, 3, 2)
        eager = jax_local.pressure_gradient_upper_faces(
            values,
            dz=0.5,
            last_upper_gradient=0.0,
        )
        compiled = jax.jit(
            lambda field: jax_local.pressure_gradient_upper_faces(
                field,
                dz=0.5,
                last_upper_gradient=0.0,
            )
        )(values)

        self.assertEqual(float(jnp.max(jnp.abs(eager - compiled))), 0.0)


if __name__ == "__main__":
    unittest.main()
