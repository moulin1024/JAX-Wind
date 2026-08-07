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


def _optional_positive(table: dict[str, Any], *names: str) -> None:
    for name in names:
        if name not in table:
            continue
        value = table[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")


def _validate_axis_mapping(name: str, table: Any) -> None:
    if not isinstance(table, dict):
        raise ValueError(f"grid.mapping.{name} must be a table")
    unknown = set(table) - {"function", "focus", "strength"}
    if unknown:
        raise ValueError(f"unknown grid.mapping.{name} keys: {sorted(unknown)}")
    function = table.get("function")
    if function not in {"uniform", "exponential"}:
        raise ValueError(
            f"grid.mapping.{name}.function must be uniform or exponential"
        )
    strength = table.get("strength", 0.0)
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(strength)
        or not 0.0 <= strength <= 50.0
    ):
        raise ValueError(f"grid.mapping.{name}.strength must lie in [0, 50]")
    focus = table.get("focus")
    if function == "uniform":
        if focus is not None or strength != 0.0:
            raise ValueError(
                f"uniform grid.mapping.{name} accepts no focus and zero strength"
            )
        return
    if (
        isinstance(focus, bool)
        or not isinstance(focus, (int, float))
        or not math.isfinite(focus)
        or not 0.0 <= focus <= 1.0
    ):
        raise ValueError(f"grid.mapping.{name}.focus must lie in [0, 1]")


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
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value > 0
            for value in extent
        )
    ):
        raise ValueError("grid.extent must contain positive lx, ly, lz")
    mapping = grid.get("mapping")
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise ValueError("grid.mapping must be a table")
        unknown_axes = set(mapping) - {"x", "y", "z"}
        if unknown_axes:
            raise ValueError(f"unknown grid.mapping axes: {sorted(unknown_axes)}")
        for name, table in mapping.items():
            _validate_axis_mapping(name, table)

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
    _optional_positive(
        momentum,
        "von_karman",
        "wall_filter_width",
        "wall_temporal_filter_gamma",
        "wall_layer_matching_filter_ratio",
    )
    face_reconstruction = momentum.get("most_consistent_face_reconstruction", True)
    if not isinstance(face_reconstruction, bool):
        raise ValueError("momentum.most_consistent_face_reconstruction must be boolean")
    mean_momentum = data.get("mean_momentum")
    if mean_momentum is not None:
        if not isinstance(mean_momentum, dict):
            raise ValueError("mean_momentum must be a table")
        enabled_constraint = mean_momentum.get("enabled", False)
        if not isinstance(enabled_constraint, bool):
            raise ValueError("mean_momentum.enabled must be boolean")
        if enabled_constraint:
            _positive(mean_momentum, "timescale")
            _optional_positive(mean_momentum, "matching_filter_ratio")
            gain = mean_momentum.get("gain", 1.0)
            if (
                isinstance(gain, bool)
                or not isinstance(gain, (int, float))
                or not math.isfinite(gain)
                or not 0.0 < gain <= 1.0
            ):
                raise ValueError("mean_momentum.gain must lie in (0, 1]")
    obsolete_wall_matching = {
        key for key in ("wall_matching_height", "wall_matching_level") if key in momentum
    }
    if obsolete_wall_matching:
        raise ValueError(
            "point wall matching is unsupported; the wall law filters the actual "
            f"first control volume: {sorted(obsolete_wall_matching)}"
        )
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
        _positive(
            thermodynamics,
            "gravity",
            "reference_potential_temperature",
            "sgs_coefficient",
        )
        scalar_sgs_model = thermodynamics.get("scalar_sgs_model", "amd")
        if scalar_sgs_model not in {"amd", "fv_dynamic"}:
            raise ValueError(
                "thermodynamics.scalar_sgs_model must be amd or fv_dynamic"
            )
        if scalar_sgs_model == "fv_dynamic" and sgs["model"] != "multilevel_lasd":
            raise ValueError(
                "FV-dynamic scalar SGS requires multilevel_lasd momentum"
            )
        for name in (
            "minimum_dynamic_coefficient",
            "maximum_dynamic_coefficient",
        ):
            if name in thermodynamics:
                value = thermodynamics[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0.0
                ):
                    raise ValueError(f"thermodynamics.{name} must be nonnegative")
        if float(thermodynamics.get("minimum_dynamic_coefficient", 0.0)) > float(
            thermodynamics.get("maximum_dynamic_coefficient", 1.0)
        ):
            raise ValueError("dynamic scalar coefficient bounds are invalid")

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
    if momentum_stability == "most" and thermal_boundary not in {
        "temperature",
        "flux",
    }:
        raise ValueError("MOST momentum coupling requires temperature or heat flux")
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
    if velocity.get("kind") not in {"constant", "log_law", "table"}:
        raise ValueError(
            "initial.velocity.kind must be constant, log_law, or table"
        )
    if velocity.get("kind") == "log_law":
        perturbation = velocity.get("perturbation_amplitude", 0.05)
        if (
            isinstance(perturbation, bool)
            or not isinstance(perturbation, (int, float))
            or not math.isfinite(perturbation)
            or perturbation < 0.0
        ):
            raise ValueError(
                "initial.velocity.perturbation_amplitude must be finite and "
                "nonnegative"
            )
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
