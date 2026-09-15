"""Versioned case documents and declaration-relative inheritance."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


FORMULATIONS = {"boussinesq", "low-mach-abl", "cryogenic-incompressible", "cryogenic-low-mach"}
PATH_KEYS = {"initial_profile", "reference_results", "source_workflow", "source_case",
             "incompressible_checkpoint", "low_mach_checkpoint", "checkpoint",
             "warmup_restart_checkpoint", "input_directory", "reference_profile"}


def merge(base: dict, overrides: dict) -> dict:
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def _paths(value: dict, parent: Path) -> None:
    for key, item in value.items():
        if isinstance(item, dict):
            _paths(item, parent)
        elif key in PATH_KEYS and isinstance(item, str):
            value[key] = str((parent / item).resolve())


def _load(path: Path, chain: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in chain:
        raise ValueError("case inheritance cycle: " + " -> ".join(map(str, (*chain, path))))
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    parent = document.pop("extends", None)
    _paths(document, path.parent)
    if parent is not None:
        if not isinstance(parent, str) or not parent:
            raise ValueError("extends must be one case path")
        document = merge(_load(path.parent / parent, (*chain, path)), document)
    return document


@dataclass(frozen=True)
class CaseSpec:
    source: Path
    document: dict[str, Any]

    def __fspath__(self) -> str:
        return str(self.source)


@dataclass(frozen=True)
class ResolvedCase(CaseSpec):
    @property
    def formulation(self) -> str:
        return self.document["formulation"]

    @property
    def fingerprint(self) -> str:
        content = deepcopy(self.document)
        content.pop("output", None)
        return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @property
    def output(self) -> Path:
        return Path(self.document["output"]["directory"])


def _positive(value: Any, name: str, *, integer: bool = False) -> None:
    types = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, types) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive {'integer' if integer else 'number'}")


def validate(document: dict) -> None:
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("expected schema_version = 1; historical cases must be migrated")
    allowed = {"schema_version", "formulation", "case", "mesh", "physics", "initial_conditions", "numerics", "time", "diagnostics", "output", "workflow"}
    unknown = document.keys() - allowed
    if unknown:
        raise ValueError("unknown case sections: " + ", ".join(sorted(unknown)))
    if document.get("formulation") not in FORMULATIONS:
        raise ValueError("unsupported formulation")
    for key in ("case", "numerics", "time", "output"):
        if not isinstance(document.get(key), dict):
            raise ValueError(f"missing [{key}] table")
    if not isinstance(document["case"].get("name"), str) or not document["case"]["name"].strip():
        raise ValueError("case.name must be nonempty")
    time = document["time"]
    _positive(time.get("dt_seconds"), "time.dt_seconds")
    _positive(time.get("steps"), "time.steps", integer=True)
    if "chunk_steps" in time:
        _positive(time["chunk_steps"], "time.chunk_steps", integer=True)
    if "checkpoint_every_steps" in time:
        _positive(time["checkpoint_every_steps"], "time.checkpoint_every_steps", integer=True)
    if "frame_count" in time:
        if type(time["frame_count"]) is not int or not 0 <= time["frame_count"] <= time["steps"]:
            raise ValueError("time.frame_count must be an integer between zero and steps")
    if "cfl" in time:
        _positive(time["cfl"], "time.cfl")
        if document["formulation"] not in {"boussinesq", "cryogenic-low-mach", "cryogenic-incompressible"}:
            raise ValueError("this formulation uses fixed timesteps; adaptive CFL is unsupported")
    if "mesh" in document:
        mesh = document["mesh"]
        for key, integer in (("cells", True), ("lengths_m", False)):
            if key not in mesh and document["formulation"] == "low-mach-abl":
                continue  # Unspecified geometry is inherited from source_case.
            values = mesh.get(key)
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError(f"mesh.{key} must contain three values")
            for value in values:
                _positive(value, f"mesh.{key}", integer=integer)
    elif document["formulation"] != "low-mach-abl":
        raise ValueError("missing [mesh]")
    backend = document["numerics"].get("pressure_backend")
    if backend not in {"fft", "gmg", "amg"}:
        raise ValueError("numerics.pressure_backend must be fft, gmg, or amg")
    if document["numerics"].get("dtype", "float32") not in {"float32", "float64"}:
        raise ValueError("numerics.dtype must be float32 or float64")
    formulation = document["formulation"]
    physics = document.get("physics", {})
    physical_sections = ({"flow", "scalar", "surface_scalar", "turbine", "wind_farm", "cooling", "inflow"} if formulation == "boussinesq"
                         else {"thermodynamics"} if formulation == "low-mach-abl" else {"ambient", "walls", "jet", "source"})
    if not isinstance(physics, dict) or physics.keys() - physical_sections:
        raise ValueError("unknown physics sections for this formulation")
    if "inflow" in physics:
        from .synthetic_inflow import validate_mann_inflow
        validate_mann_inflow(document)
    if "wind_farm" in physics:
        from .wind_farm import validate_wind_farm
        validate_wind_farm(document)
    if formulation != "boussinesq":
        if document["numerics"].get("dtype", "float32") != "float32":
            raise ValueError("this formulation currently supports float32 only")
        allowed_numerics = {"dtype", "pressure_backend", "time_integration"}
        if formulation.startswith("cryogenic-"):
            if backend != "gmg":
                raise ValueError("cryogenic formulations currently require GMG pressure")
            allowed_numerics |= {"gmg_tolerance", "gmg_presweeps", "gmg_postsweeps", "momentum_closure", "scalar_advection_scheme"}
        unknown = document["numerics"].keys() - allowed_numerics
        if unknown:
            raise ValueError(f"unknown numerics settings: {', '.join(sorted(unknown))}")
    for section, keys in {
        "time": {"dt_seconds", "steps", "chunk_steps", "cfl", "frame_count", "statistics_window_seconds", "checkpoint_every_steps"},
        "output": {"directory"},
    }.items():
        unknown = document[section].keys() - keys
        if unknown:
            raise ValueError(f"unknown {section} settings: {', '.join(sorted(unknown))}")
    directory = document["output"].get("directory")
    if not isinstance(directory, str) or not directory:
        raise ValueError("output.directory must be nonempty")


def load_case(path: str | Path | ResolvedCase) -> ResolvedCase:
    if isinstance(path, ResolvedCase):
        validate(path.document)
        return path
    source = Path(path).resolve()
    document = _load(source)
    validate(document)
    # Output is intentionally relative to invocation, unlike input assets.
    output = document["output"]["directory"].replace("{pressure_backend}", document["numerics"]["pressure_backend"])
    document["output"]["directory"] = str(Path(output).resolve())
    return ResolvedCase(source, document)


def native_document(path: str | Path | ResolvedCase) -> dict:
    """Lower the shared schema into formulation-specific typed loader inputs.

    This is an internal adapter; it never writes temporary case files.
    """
    case = load_case(path)
    doc = deepcopy(case.document)
    result = {"case": doc["case"], "time": doc["time"], "numerics": doc["numerics"]}
    if "mesh" in doc:
        result["domain"] = doc["mesh"]
    result.update(doc.get("physics", {}))
    result.pop("inflow", None)  # Owned by the direct open-inflow builder.
    result.pop("wind_farm", None)  # Owned by the controlled periodic builder.
    if "diagnostics" in doc:
        result["diagnostics"] = doc["diagnostics"]
    if case.formulation == "boussinesq":
        result["finite_volume"] = result["numerics"]
        result["numerics"] = {"dtype": result["finite_volume"].pop("dtype", "float32")}
        result["finite_volume"]["output_directory"] = doc["output"]["directory"]
        result["finite_volume"]["chunk_steps"] = result["time"].pop("chunk_steps", 100)
        if "cfl" in result["time"]:
            result["finite_volume"]["cfl_ceiling"] = result["time"].pop("cfl")
        for key in ("checkpoint_every_steps", "frame_count", "statistics_window_seconds"):
            result["time"].pop(key, None)
        if "workflow" in doc:
            result["finite_volume_workflow"] = doc["workflow"]
        for name in ("turbine", "cooling"):
            if name in result:
                result["finite_volume_" + name] = result.pop(name)
    else:
        result["time"].setdefault("chunk_steps", 100)
        result["output"] = doc["output"]
        if case.formulation == "low-mach-abl":
            result["time"].setdefault("frame_count", min(100, result["time"]["steps"]))
            result["restart"] = doc.get("initial_conditions", {})
            if "checkpoint" in result["restart"]:
                result["restart"].setdefault("initial_condition", "configured")
            if "source_case" in result["restart"]:
                result["restart"]["source_workflow"] = result["restart"].pop("source_case")
        else:
            result["time"].setdefault("checkpoint_every_steps", result["time"]["chunk_steps"] * 10)
            result["numerics"]["flow_formulation"] = "low-mach" if case.formulation == "cryogenic-low-mach" else "incompressible"
    return result


def derive_case(base: str | Path, output: str | Path, *, cells=None, cfl=None) -> Path:
    import os
    source, destination = Path(base).resolve(), Path(output).resolve()
    parent = load_case(source)
    if cells is None and cfl is None:
        raise ValueError("provide --cells and/or --cfl")
    overrides = {"case": {"name": destination.stem}, "output": {"directory": f"runs/{destination.stem}"}}
    if cells is not None:
        overrides["mesh"] = {"cells": list(cells)}
        if parent.formulation in {"boussinesq", "low-mach-abl"}:
            overrides["case"]["profile_resampling"] = "linear"
    if cfl is not None:
        overrides["time"] = {"cfl": cfl}
    validate(merge(parent.document, overrides))
    from .toml import dumps
    content = dumps({"extends": os.path.relpath(source, destination.parent), **overrides})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return destination
