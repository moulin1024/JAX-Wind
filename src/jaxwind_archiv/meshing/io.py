"""TOML input and versioned JSON output for the meshing application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from jaxwind_archiv.domain import RectilinearGrid

from .analytic import axis_statistics
from .model import AxisMeshSpec, GeneratedMesh, MeshSpec


MESH_SCHEMA = "jaxwind.rectilinear-mesh.v1"
_AXIS_KEYS = {
    "lower_m",
    "upper_m",
    "cells",
    "clustering",
    "point_m",
    "strength",
}


def _axis_spec(name: str, table: Any) -> AxisMeshSpec:
    if not isinstance(table, dict):
        raise ValueError(f"mesh.{name} must be a TOML table")
    unknown = set(table) - _AXIS_KEYS
    if unknown:
        raise ValueError(f"unknown mesh.{name} keys: {sorted(unknown)}")
    missing = {"lower_m", "upper_m", "cells"} - set(table)
    if missing:
        raise ValueError(f"missing mesh.{name} keys: {sorted(missing)}")
    try:
        return AxisMeshSpec(
            lower=float(table["lower_m"]),
            upper=float(table["upper_m"]),
            cells=table["cells"],
            clustering=table.get("clustering", "uniform"),
            point=(None if "point_m" not in table else float(table["point_m"])),
            strength=float(table.get("strength", 0.0)),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid mesh.{name}: {error}") from error


def load_mesh_spec(path: Path | str) -> MeshSpec:
    """Load and validate one standalone meshing TOML file."""

    source = Path(path)
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    mesh = document.get("mesh")
    if not isinstance(mesh, dict):
        raise ValueError("missing [mesh] table")
    unknown = set(mesh) - {"x", "y", "z"}
    if unknown:
        raise ValueError(f"unknown mesh axes: {sorted(unknown)}")
    missing = {"x", "y", "z"} - set(mesh)
    if missing:
        raise ValueError(f"missing mesh axes: {sorted(missing)}")
    return MeshSpec(*(_axis_spec(name, mesh[name]) for name in ("x", "y", "z")))


def _axis_document(specification: AxisMeshSpec, faces: tuple[float, ...]) -> dict:
    statistics = axis_statistics(faces)
    configuration = {
        "lower_m": specification.lower,
        "upper_m": specification.upper,
        "cells": specification.cells,
        "clustering": specification.clustering,
        "point_m": specification.point,
        "strength": specification.strength,
    }
    return {
        "configuration": configuration,
        "faces_m": list(faces),
        "statistics": {
            "minimum_spacing_m": statistics.minimum_spacing,
            "maximum_spacing_m": statistics.maximum_spacing,
            "maximum_adjacent_ratio": statistics.maximum_adjacent_ratio,
        },
    }


def _mesh_document(mesh: GeneratedMesh) -> dict:
    return {
        "schema": MESH_SCHEMA,
        "storage_order": "z-y-x",
        "axes": {
            "x": _axis_document(mesh.specification.x, mesh.grid.x_faces),
            "y": _axis_document(mesh.specification.y, mesh.grid.y_faces),
            "z": _axis_document(mesh.specification.z, mesh.grid.z_faces),
        },
    }


def write_mesh(
    mesh: GeneratedMesh,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a deterministic, versioned mesh artifact."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"mesh output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_mesh_document(mesh), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _configuration_from_json(name: str, value: Any) -> AxisMeshSpec:
    if not isinstance(value, dict):
        raise ValueError(f"mesh axis {name!r} has no configuration object")
    return AxisMeshSpec(
        lower=float(value["lower_m"]),
        upper=float(value["upper_m"]),
        cells=value["cells"],
        clustering=value["clustering"],
        point=(None if value.get("point_m") is None else float(value["point_m"])),
        strength=float(value["strength"]),
    )


def load_mesh(path: Path | str) -> GeneratedMesh:
    """Load the portable mesh artifact without importing a solver backend."""

    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != MESH_SCHEMA:
        raise ValueError(f"unsupported mesh schema in {source}")
    axes = document.get("axes")
    if not isinstance(axes, dict) or set(axes) != {"x", "y", "z"}:
        raise ValueError("mesh artifact must contain exactly x, y, and z axes")

    specifications = []
    faces = []
    for name in ("x", "y", "z"):
        axis = axes[name]
        if not isinstance(axis, dict):
            raise ValueError(f"mesh axis {name!r} must be an object")
        specifications.append(_configuration_from_json(name, axis.get("configuration")))
        values = axis.get("faces_m")
        if not isinstance(values, list):
            raise ValueError(f"mesh axis {name!r} has no faces_m array")
        faces.append(tuple(float(value) for value in values))

    specification = MeshSpec(*specifications)
    grid = RectilinearGrid(*faces)
    expected_shape = (
        specification.z.cells,
        specification.y.cells,
        specification.x.cells,
    )
    if grid.shape != expected_shape:
        raise ValueError(
            f"mesh face counts imply shape {grid.shape}, expected {expected_shape}"
        )
    return GeneratedMesh(specification, grid)
