from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

from jaxwind_archiv.domain import (
    Cell,
    DistributionSpec,
    DomainAxis,
    EqualZSlab,
    Evaluated,
    Field,
    GlobalTestRegion,
    MeshAxis,
    MeshCoordinate,
    MeshTopology,
    OwnedInterval,
    Partitioned,
    PressureCorrection,
    Replicated,
    UniformGrid,
    ZFace,
)


class FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def make_decomposition(shards: int, cells_per_shard: int = 3) -> EqualZSlab:
    grid = UniformGrid(
        nx=4,
        ny=6,
        nz=shards * cells_per_shard,
        lx=4.0,
        ly=6.0,
        lz=float(shards * cells_per_shard),
    )
    topology = MeshTopology((MeshAxis("z", shards),))
    return EqualZSlab(grid, topology, DistributionSpec.z_slab())


class DomainImportTests(unittest.TestCase):
    def test_domain_import_does_not_import_jax(self) -> None:
        root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(root / "src")
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import jaxwind.domain; "
                "assert not any(n == 'jax' or n.startswith('jax.') "
                "for n in sys.modules)",
            ],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class MeshAndDistributionTests(unittest.TestCase):
    def test_mesh_and_coordinate_validation(self) -> None:
        topology = MeshTopology((MeshAxis("z", 4),))
        coordinate = MeshCoordinate((3,))

        coordinate.validate(topology)
        self.assertEqual(coordinate.index(topology, "z"), 3)
        self.assertEqual(topology.size, 4)

        with self.assertRaisesRegex(ValueError, "outside"):
            MeshCoordinate((4,)).validate(topology)
        with self.assertRaisesRegex(ValueError, "rank"):
            MeshCoordinate((0, 0)).validate(topology)

    def test_distribution_is_total_and_mesh_general(self) -> None:
        topology = MeshTopology((MeshAxis("row", 2), MeshAxis("column", 3)))
        distribution = DistributionSpec(
            Partitioned("row"),
            Replicated(),
            Partitioned("column"),
        )

        distribution.validate(topology)
        self.assertIsInstance(distribution.placement(DomainAxis.Y), Replicated)

        with self.assertRaisesRegex(ValueError, "cannot partition multiple"):
            DistributionSpec(
                Partitioned("row"),
                Replicated(),
                Partitioned("row"),
            ).validate(topology)

    def test_first_interpreter_rejects_unsupported_topologies(self) -> None:
        grid = UniformGrid(4, 4, 8, 1.0, 1.0, 1.0)
        topology = MeshTopology((MeshAxis("z", 2),))

        with self.assertRaisesRegex(ValueError, "x partitioning"):
            EqualZSlab(
                grid,
                topology,
                DistributionSpec(Partitioned("z"), Replicated(), Replicated()),
            )
        with self.assertRaisesRegex(ValueError, "divisible"):
            EqualZSlab(
                UniformGrid(4, 4, 7, 1.0, 1.0, 1.0),
                topology,
                DistributionSpec.z_slab(),
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            EqualZSlab(
                grid,
                MeshTopology((MeshAxis("z", 2), MeshAxis("x", 1))),
                DistributionSpec.z_slab(),
            )


class OwnershipLawTests(unittest.TestCase):
    def test_cell_and_stored_face_partitions_cover_exactly_once(self) -> None:
        for shards in range(1, 5):
            for cells_per_shard in range(1, 5):
                with self.subTest(shards=shards, cells=cells_per_shard):
                    decomposition = make_decomposition(shards, cells_per_shard)
                    cells = decomposition.regions(Cell)
                    faces = decomposition.regions(ZFace)

                    cell_coordinates = [
                        coordinate
                        for region in cells
                        for coordinate in region.stored_z.coordinates()
                    ]
                    face_coordinates = [
                        coordinate
                        for region in faces
                        for coordinate in region.stored_z.coordinates()
                    ]

                    self.assertEqual(
                        cell_coordinates,
                        list(range(decomposition.grid.nz)),
                    )
                    self.assertEqual(
                        [0] + face_coordinates,
                        list(range(decomposition.grid.nz + 1)),
                    )
                    self.assertEqual(len(face_coordinates), len(set(face_coordinates)))

    def test_boundary_flags_and_equal_storage_shapes(self) -> None:
        decomposition = make_decomposition(4, 2)
        cells = decomposition.regions(Cell)
        faces = decomposition.regions(ZFace)

        self.assertTrue(cells[0].lower_physical)
        self.assertFalse(cells[0].upper_physical)
        self.assertFalse(cells[1].lower_physical)
        self.assertTrue(cells[-1].upper_physical)
        self.assertEqual(
            tuple(region.storage_shape for region in cells),
            tuple(region.storage_shape for region in faces),
        )
        self.assertEqual(
            faces[1].stored_z.start,
            cells[1].cell_z.start + 1,
        )

    def test_rank_relabelling_does_not_change_logical_coverage(self) -> None:
        decomposition = make_decomposition(4, 2)
        coordinates = {
            coordinate
            for region in reversed(decomposition.regions(ZFace))
            for coordinate in region.stored_z.coordinates()
        }
        self.assertEqual(coordinates, set(range(1, decomposition.grid.nz + 1)))

    def test_owned_interval_rejects_empty_or_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            OwnedInterval(2, 2)
        with self.assertRaises(ValueError):
            OwnedInterval(-1, 2)


class FieldValidationTests(unittest.TestCase):
    def test_global_and_owned_field_shapes_are_location_specific(self) -> None:
        decomposition = make_decomposition(2, 3)
        cell_region = decomposition.regions(Cell)[0]
        global_faces = GlobalTestRegion(decomposition.grid, ZFace)

        cell_payload = FakeArray(cell_region.storage_shape)
        cell_field = Field(
            PressureCorrection,
            Cell,
            cell_region,
            Evaluated,
            cell_payload,
        )
        face_field = Field(
            PressureCorrection,
            ZFace,
            global_faces,
            Evaluated,
            FakeArray(global_faces.storage_shape),
        )

        self.assertIs(cell_field.payload, cell_payload)
        self.assertEqual(face_field.payload.shape[0], decomposition.grid.nz + 1)

    def test_field_rejects_forged_location_and_shape(self) -> None:
        decomposition = make_decomposition(2, 3)
        region = decomposition.regions(Cell)[0]

        with self.assertRaisesRegex(ValueError, "location"):
            Field(
                PressureCorrection,
                ZFace,
                region,
                Evaluated,
                FakeArray(region.storage_shape),
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            Field(
                PressureCorrection,
                Cell,
                region,
                Evaluated,
                FakeArray((99, 1, 1)),
            )


if __name__ == "__main__":
    unittest.main()
