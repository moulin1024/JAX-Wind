"""Declarative configuration for the benchmark-independent ABL runner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tomllib
from typing import Any


CASE_SCHEMA = "jaxwind.case.v2"


def _require_table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"configuration requires [{name}]")
    return value


def _positive(table: dict[str, Any], *names: str) -> None:
    for name in names:
        value = table.get(name)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")


def _positive_integer(table: dict[str, Any], *names: str) -> None:
    for name in names:
        value = table.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class CaseConfig:
    """Validated, serializable case description loaded from TOML."""

    source: Path
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.data["name"])

    def section(self, name: str) -> dict[str, Any]:
        return _require_table(self.data, name)

    def resolved_json(self) -> str:
        return json.dumps(self.data, indent=2, sort_keys=True) + "\n"

    def with_overrides(self, overrides: list[str]) -> CaseConfig:
        data = deepcopy(self.data)
        for expression in overrides:
            if "=" not in expression:
                raise ValueError(f"override must be path=value, got {expression!r}")
            path, encoded = expression.split("=", 1)
            keys = path.split(".")
            if not all(keys):
                raise ValueError(f"invalid override path: {path!r}")
            try:
                value = tomllib.loads(f"value = {encoded}")["value"]
            except tomllib.TOMLDecodeError as error:
                raise ValueError(f"invalid TOML override value: {encoded!r}") from error
            target = data
            for key in keys[:-1]:
                nested = target.get(key)
                if not isinstance(nested, dict):
                    raise ValueError(f"override path does not name a table: {path!r}")
                target = nested
            if keys[-1] not in target:
                raise ValueError(f"override does not name an existing key: {path!r}")
            target[keys[-1]] = value
        validate_case(data)
        return CaseConfig(self.source, data)


def validate_case(data: dict[str, Any]) -> None:
    if data.get("schema") != CASE_SCHEMA:
        raise ValueError(f"unsupported case schema: {data.get('schema')!r}")
    if not isinstance(data.get("name"), str) or not data["name"]:
        raise ValueError("configuration requires a non-empty name")

    grid = _require_table(data, "grid")
    shape = grid.get("shape")
    extent = grid.get("extent")
    if not (
        isinstance(shape, list)
        and len(shape) == 3
        and all(isinstance(value, int) and value >= 4 for value in shape)
    ):
        raise ValueError("grid.shape must contain nx, ny, nz >= 4")
    if not (
        isinstance(extent, list)
        and len(extent) == 3
        and all(isinstance(value, (int, float)) and value > 0 for value in extent)
    ):
        raise ValueError("grid.extent must contain positive lx, ly, lz")

    numerics = _require_table(data, "numerics")
    if numerics.get("dtype") not in {"float32", "float64"}:
        raise ValueError("numerics.dtype must be float32 or float64")
    if numerics.get("sgs_time_integration") not in {"explicit", "imex_ark3"}:
        raise ValueError("unsupported SGS time integration")
    _positive(
        numerics,
        "target_cfl",
        "target_diffusive_cfl",
        "pressure_relative_tolerance",
    )
    _positive_integer(numerics, "pressure_max_iterations", "pressure_coarse_smooth")

    time = _require_table(data, "time")
    _positive(time, "end", "maximum_step", "sample_interval")
    _positive_integer(
        time,
        "history_interval",
        "log_interval",
        "checkpoint_interval",
    )
    sample_start = time.get("sample_start")
    if (
        not isinstance(sample_start, (int, float))
        or not 0 <= sample_start < time["end"]
    ):
        raise ValueError("time.sample_start must lie in [0, time.end)")
    if time.get("sample_basis") not in {"step", "time"}:
        raise ValueError("time.sample_basis must be step or time")
    if time["sample_basis"] == "step":
        _positive_integer(time, "sample_interval")

    momentum = _require_table(data, "momentum")
    _positive(momentum, "friction_velocity", "roughness_length")
    geostrophic = momentum.get("geostrophic_wind")
    if geostrophic is not None and not (
        isinstance(geostrophic, list)
        and len(geostrophic) == 2
        and all(isinstance(value, (int, float)) for value in geostrophic)
    ):
        raise ValueError("momentum.geostrophic_wind must be a two-vector or null")

    sgs = _require_table(data, "sgs")
    if sgs.get("model") not in {"amd", "multilevel_lasd"}:
        raise ValueError("sgs.model must be amd or multilevel_lasd")
    _positive(sgs, "coefficient")

    thermodynamics = _require_table(data, "thermodynamics")
    enabled = thermodynamics.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("thermodynamics.enabled must be boolean")
    if enabled:
        if sgs["model"] != "amd":
            raise ValueError("thermodynamic coupling currently requires AMD momentum")
        _positive(
            thermodynamics,
            "gravity",
            "reference_potential_temperature",
            "sgs_coefficient",
        )

    surface = _require_table(data, "surface")
    thermal_boundary = surface.get("thermal_boundary")
    if thermal_boundary not in {"adiabatic", "flux", "temperature"}:
        raise ValueError(
            "surface.thermal_boundary must be adiabatic, flux, or temperature"
        )
    if not enabled and thermal_boundary != "adiabatic":
        raise ValueError("disabled thermodynamics requires an adiabatic surface")
    momentum_stability = surface.get("momentum_stability", "neutral")
    if momentum_stability not in {"neutral", "most"}:
        raise ValueError("surface.momentum_stability must be neutral or most")
    if momentum_stability == "most" and thermal_boundary != "temperature":
        raise ValueError("MOST momentum coupling requires a surface temperature")
    if thermal_boundary == "temperature" and momentum_stability != "most":
        raise ValueError("surface-temperature coupling requires MOST momentum")
    if thermal_boundary == "flux":
        value = surface.get("heat_flux")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("surface.heat_flux must be finite")
    if thermal_boundary == "temperature":
        value = surface.get("potential_temperature")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("surface.potential_temperature must be finite")
        tendency = surface.get("temperature_tendency", 0.0)
        if not isinstance(tendency, (int, float)) or not math.isfinite(tendency):
            raise ValueError("surface.temperature_tendency must be finite")
        if "thermal_roughness_length" in surface:
            _positive(surface, "thermal_roughness_length")

    initial = _require_table(data, "initial")
    velocity = _require_table(initial, "velocity")
    if velocity.get("kind") not in {"constant", "table"}:
        raise ValueError("initial.velocity.kind must be constant or table")
    temperature_initial = _require_table(initial, "potential_temperature")
    if temperature_initial.get("kind") not in {
        "none",
        "inversion",
        "convective",
    }:
        raise ValueError("unsupported initial potential-temperature primitive")
    if enabled == (temperature_initial["kind"] == "none"):
        raise ValueError("thermodynamics and initial potential temperature disagree")


def load_case(path: str | Path) -> CaseConfig:
    source = Path(path).resolve()
    with source.open("rb") as stream:
        data = tomllib.load(stream)
    validate_case(data)
    return CaseConfig(source, data)


__all__ = ["CASE_SCHEMA", "CaseConfig", "load_case", "validate_case"]
