"""Laws for solver-independent analytic mesh generation."""

from __future__ import annotations

import json

import pytest

from jaxwind_archiv.domain import RectilinearGrid as DomainRectilinearGrid
from jaxwind_archiv.meshing import (
    AxisMeshSpec,
    MeshSpec,
    axis_statistics,
    generate_axis_faces,
    generate_mesh,
    load_mesh,
    load_mesh_spec,
    write_mesh,
)
from jaxwind_archiv.meshing.cli import main
from jaxwind_archiv.pressure import RectilinearGrid as PressureRectilinearGrid


def test_zero_strength_is_the_exact_uniform_limit() -> None:
    uniform = generate_axis_faces(AxisMeshSpec(0.0, 8.0, 4))
    single = generate_axis_faces(AxisMeshSpec(0.0, 8.0, 4, "single", 0.0, 0.0))
    double = generate_axis_faces(AxisMeshSpec(0.0, 8.0, 4, "double", 3.0, 0.0))

    assert uniform == (0.0, 2.0, 4.0, 6.0, 8.0)
    assert single == uniform
    assert double == uniform


def test_single_sided_maps_are_mirrored_and_strength_clusters() -> None:
    lower = generate_axis_faces(AxisMeshSpec(0.0, 1.0, 8, "single", 0.0, 3.0))
    upper = generate_axis_faces(AxisMeshSpec(0.0, 1.0, 8, "single", 1.0, 3.0))
    lower_widths = tuple(b - a for a, b in zip(lower, lower[1:]))
    upper_widths = tuple(b - a for a, b in zip(upper, upper[1:]))

    assert lower_widths[0] < lower_widths[-1]
    assert upper_widths[-1] < upper_widths[0]
    assert lower == pytest.approx(tuple(1.0 - value for value in upper[::-1]))


def test_double_sided_map_clusters_from_both_sides_to_interior_point() -> None:
    point = 3.0
    faces = generate_axis_faces(AxisMeshSpec(0.0, 10.0, 10, "double", point, 2.5))
    point_index = faces.index(point)
    widths = tuple(b - a for a, b in zip(faces, faces[1:]))

    assert len(faces) == 11
    assert widths[point_index - 1] < widths[0]
    assert widths[point_index] < widths[-1]
    assert all(right > left for left, right in zip(faces, faces[1:]))


def test_three_axis_controls_are_independent() -> None:
    specification = MeshSpec(
        AxisMeshSpec(-2.0, 2.0, 8, "double", 0.0, 2.0),
        AxisMeshSpec(0.0, 6.0, 6),
        AxisMeshSpec(0.0, 3.0, 5, "single", 0.0, 1.5),
    )
    mesh = generate_mesh(specification)

    assert mesh.grid.shape == (5, 6, 8)
    assert 0.0 in mesh.grid.x_faces
    assert mesh.grid.y_faces == pytest.approx(tuple(float(i) for i in range(7)))
    assert axis_statistics(mesh.grid.z_faces).minimum_spacing < 3.0 / 5.0


def test_axis_spec_rejects_ambiguous_cluster_points() -> None:
    with pytest.raises(ValueError, match="must be an axis boundary"):
        AxisMeshSpec(0.0, 1.0, 8, "single", 0.5, 2.0)
    with pytest.raises(ValueError, match="strictly inside"):
        AxisMeshSpec(0.0, 1.0, 8, "double", 0.0, 2.0)
    with pytest.raises(ValueError, match="does not accept a point"):
        AxisMeshSpec(0.0, 1.0, 8, "uniform", 0.0, 0.0)


def _configuration() -> str:
    return """
[mesh.x]
lower_m = 0.0
upper_m = 20.0
cells = 8
clustering = "double"
point_m = 8.0
strength = 2.0

[mesh.y]
lower_m = -5.0
upper_m = 5.0
cells = 4

[mesh.z]
lower_m = 0.0
upper_m = 10.0
cells = 6
clustering = "single"
point_m = 0.0
strength = 1.5
"""


def test_versioned_mesh_artifact_round_trip(tmp_path) -> None:
    config = tmp_path / "mesh.toml"
    output = tmp_path / "mesh.json"
    config.write_text(_configuration(), encoding="utf-8")

    generated = generate_mesh(load_mesh_spec(config))
    write_mesh(generated, output)
    restored = load_mesh(output)
    document = json.loads(output.read_text(encoding="utf-8"))

    assert restored == generated
    assert document["schema"] == "jaxwind.rectilinear-mesh.v1"
    assert document["storage_order"] == "z-y-x"
    assert document["axes"]["z"]["statistics"]["minimum_spacing_m"] > 0.0
    with pytest.raises(FileExistsError):
        write_mesh(generated, output)


def test_meshing_cli_generates_and_inspects_from_any_directory(
    tmp_path, capsys
) -> None:
    config = tmp_path / "input.toml"
    output = tmp_path / "nested" / "mesh.json"
    config.write_text(_configuration(), encoding="utf-8")

    assert main(["generate", str(config), "--output", str(output)]) == 0
    generated_output = capsys.readouterr().out
    assert "shape (z,y,x): 6 x 4 x 8" in generated_output
    assert output.is_file()

    assert main(["inspect", str(output)]) == 0
    inspected_output = capsys.readouterr().out
    assert "adjacent=" in inspected_output


def test_rectilinear_grid_is_domain_owned_with_pressure_compatibility() -> None:
    assert PressureRectilinearGrid is DomainRectilinearGrid
