from __future__ import annotations

from types import SimpleNamespace
import unittest

import jax.numpy as jnp

from jaxwind.domain import (
    AddressableField,
    Cell,
    DistributionSpec,
    EqualZSlab,
    Evaluated,
    MeshAxis,
    MeshTopology,
    PressureCorrection,
    PressureRhs,
    UniformGrid,
)
from jaxwind.pressure import SpectralFDPressureAdapter


class FakeSolver:
    def __init__(self, grid: UniformGrid, *, discretization="cell-centered-compatible"):
        self.config = SimpleNamespace(
            discretization=discretization,
            data_layout="z-first",
            nx=grid.nx,
            ny=grid.ny,
            nz=grid.nz,
            lx=grid.lx,
            ly=grid.ly,
            lz=grid.lz,
        )
        self.global_devices = 2
        self.local_devices = 2
        self.process_index = 0
        self.local_input_shape = (2, grid.nz // 2, grid.ny, grid.nx)

    def solve(self, rhs):
        return rhs - jnp.mean(rhs)


class SpectralFDAdapterBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = UniformGrid(4, 4, 8, 4.0, 4.0, 8.0)
        self.decomposition = EqualZSlab(
            self.grid,
            MeshTopology((MeshAxis("z", 2),)),
            DistributionSpec.z_slab(),
        )

    def test_adapter_preserves_cell_ownership_and_hides_solver_layout(self) -> None:
        adapter = SpectralFDPressureAdapter(
            self.decomposition,
            (0, 1),
            FakeSolver(self.grid),
        )
        payload = jnp.arange(2 * 4 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4, 4)
        rhs = AddressableField(
            PressureRhs,
            Cell,
            self.decomposition.regions(Cell),
            Evaluated,
            payload,
        )

        pressure = adapter.solve(rhs)

        self.assertIs(pressure.quantity, PressureCorrection)
        self.assertEqual(pressure.regions, rhs.regions)
        self.assertEqual(pressure.payload.shape, payload.shape)
        self.assertAlmostEqual(float(jnp.mean(pressure.payload)), 0.0, places=6)

    def test_adapter_rejects_an_incompatible_external_discretization(self) -> None:
        with self.assertRaisesRegex(ValueError, "cell-centered-compatible"):
            SpectralFDPressureAdapter(
                self.decomposition,
                (0, 1),
                FakeSolver(self.grid, discretization="legacy-augmented"),
            )


if __name__ == "__main__":
    unittest.main()
