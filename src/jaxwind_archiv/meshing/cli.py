"""Command-line shell for standalone analytic mesh generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from .analytic import axis_statistics, generate_mesh
from .io import load_mesh, load_mesh_spec, write_mesh
from .model import GeneratedMesh


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jaxwind-mesh",
        description=(
            "Generate solver-independent rectilinear meshes from analytic "
            "per-axis clustering controls."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate a mesh JSON artifact")
    generate.add_argument("config", type=Path, help="standalone meshing TOML file")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--overwrite", action="store_true")
    inspect = commands.add_parser("inspect", help="validate and summarize a mesh")
    inspect.add_argument("mesh", type=Path)
    return parser


def _summary(mesh: GeneratedMesh, path: Path | None = None) -> str:
    lines = []
    if path is not None:
        lines.append(f"mesh: {path}")
    lines.append(
        "shape (z,y,x): " + " x ".join(str(value) for value in mesh.grid.shape)
    )
    for name, faces in (
        ("x", mesh.grid.x_faces),
        ("y", mesh.grid.y_faces),
        ("z", mesh.grid.z_faces),
    ):
        statistics = axis_statistics(faces)
        lines.append(
            f"{name}: [{faces[0]:.9g}, {faces[-1]:.9g}] "
            f"dmin={statistics.minimum_spacing:.9g} "
            f"dmax={statistics.maximum_spacing:.9g} "
            f"adjacent={statistics.maximum_adjacent_ratio:.6g}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the meshing application."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            mesh = generate_mesh(load_mesh_spec(args.config))
            output = write_mesh(mesh, args.output, overwrite=args.overwrite)
            print(_summary(mesh, output))
            return 0
        mesh = load_mesh(args.mesh)
        print(_summary(mesh, args.mesh))
        return 0
    except (
        FileExistsError,
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        parser.error(str(error))
    return 2  # pragma: no cover


__all__ = ["main"]
