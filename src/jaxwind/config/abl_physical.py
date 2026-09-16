"""Uniform ABL composition from physical case data."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from jaxwind.domain import (
    AnalyticalGrid,
    IdentityMapping,
    SinhMapping,
    TanhMapping,
    UniformGrid,
)
from jaxwind.domain.grid import Grid

from .abl_types import (
    BoussinesqCase,
    DiagnosticReference,
    SurfaceScalarEvolution,
    TabulatedBoussinesqState,
)


def _table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] table")
    return value


def _keys(
    table: dict[str, Any],
    expected: set[str],
    *,
    name: str,
    optional: set[str] | None = None,
) -> None:
    optional = set() if optional is None else optional
    missing = expected - table.keys()
    unknown = table.keys() - expected - optional
    if missing:
        raise ValueError(f"[{name}] is missing: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"[{name}] has unknown keys: {', '.join(sorted(unknown))}")


def _string(table: dict[str, Any], key: str) -> str:
    value = table[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(table: dict[str, Any], key: str) -> int:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(table: dict[str, Any], key: str) -> float:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _numbers(
    table: dict[str, Any],
    key: str,
    *,
    length: int | None = None,
) -> tuple[float, ...]:
    value = table[key]
    if not isinstance(value, list) or (
        length is not None and len(value) != length
    ):
        count = "a list of" if length is None else str(length)
        raise ValueError(f"{key} must contain {count} numbers")
    temporary = {str(index): item for index, item in enumerate(value)}
    return tuple(_number(temporary, str(index)) for index in range(len(value)))


def _integers(
    table: dict[str, Any],
    key: str,
    *,
    length: int,
) -> tuple[int, ...]:
    value = table[key]
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must contain {length} integers")
    temporary = {str(index): item for index, item in enumerate(value)}
    return tuple(_integer(temporary, str(index)) for index in range(length))


def _physical_grid(
    domain: dict[str, Any],
    cells: tuple[int, int, int],
    lengths_m: tuple[float, float, float],
) -> Grid:
    """Build the physical grid from an optional analytical mapping table."""

    table = domain.get("mapping")
    if table is None:
        return UniformGrid(*cells, *lengths_m)
    if not isinstance(table, dict):
        raise ValueError("domain.mapping must be a table")
    _keys(
        table,
        {"types", "focus_m", "strength"},
        name="domain.mapping",
    )
    raw_types = table["types"]
    if (
        not isinstance(raw_types, list)
        or len(raw_types) != 3
        or any(
            not isinstance(value, str) or not value for value in raw_types
        )
    ):
        raise ValueError("domain.mapping.types must contain 3 strings")
    mapping_types = tuple(
        value.lower().replace("_", "-") for value in raw_types
    )
    focus_m = _numbers(table, "focus_m", length=3)
    strengths = _numbers(table, "strength", length=3)
    mappings = []
    nonuniform = False
    for axis, kind, focus, strength, length in zip(
        "xyz", mapping_types, focus_m, strengths, lengths_m, strict=True
    ):
        if kind not in {"uniform", "tanh", "sinh"}:
            raise ValueError(
                f"domain.mapping.types[{axis}] must be uniform, tanh, or sinh"
            )
        if not 0.0 <= focus <= length:
            raise ValueError(
                f"domain.mapping.focus_m[{axis}] must lie in [0, {length}]"
            )
        if strength < 0.0:
            raise ValueError(
                f"domain.mapping.strength[{axis}] must be nonnegative"
            )
        if kind == "uniform":
            if strength != 0.0:
                raise ValueError(
                    f"domain.mapping.strength[{axis}] must be zero for uniform"
                )
            mappings.append(IdentityMapping())
        elif kind == "tanh":
            mappings.append(TanhMapping(strength, focus / length))
            nonuniform = nonuniform or strength > 0.0
        else:
            mappings.append(SinhMapping(focus / length, strength))
            nonuniform = nonuniform or strength > 0.0
    if not nonuniform:
        return UniformGrid(*cells, *lengths_m)
    return AnalyticalGrid(*cells, *lengths_m, *mappings)


def compose_abl(
    *,
    name: str,
    citation: str,
    cells: tuple[int, int, int],
    lengths_m: tuple[float, float, float],
    scalar_reference_value: float,
    scalar_surface_flux: float,
    surface_scalar: SurfaceScalarEvolution | None,
    buoyancy_acceleration_per_scalar: float,
    pressure_acceleration_m_s2: tuple[float, float],
    geostrophic_velocity_m_s: tuple[float, float],
    advection_frame_velocity_m_s: tuple[float, float],
    coriolis_s: tuple[float, float],
    roughness_length_m: float,
    von_karman: float,
    initial_condition: TabulatedBoussinesqState,
    reference_results: str | Path,
    dt_seconds: float,
    steps: int,
    sample_start_step: int,
    sample_every_steps: int,
    diagnostic_reference: DiagnosticReference,
    dtype: str,
    physical_grid: Grid | None = None,
) -> BoussinesqCase:
    """Compose one direct finite-volume ABL case from physical inputs."""

    grid = (
        UniformGrid(*cells, *lengths_m)
        if physical_grid is None
        else physical_grid
    )
    if (grid.nx, grid.ny, grid.nz) != cells or (
        grid.lx,
        grid.ly,
        grid.lz,
    ) != lengths_m:
        raise ValueError("physical_grid must match the composed cells and lengths")
    return BoussinesqCase(
        name=name,
        citation=citation,
        physical_grid=grid,
        scalar_reference_value=scalar_reference_value,
        initial_condition=initial_condition,
        diagnostic_reference=diagnostic_reference,
        reference_results=Path(reference_results),
        pressure_acceleration_m_s2=pressure_acceleration_m_s2,
        geostrophic_velocity_m_s=geostrophic_velocity_m_s,
        coriolis_s=coriolis_s,
        roughness_length_m=roughness_length_m,
        von_karman=von_karman,
        scalar_surface_flux=scalar_surface_flux,
        buoyancy_acceleration_per_scalar=buoyancy_acceleration_per_scalar,
        surface_scalar=surface_scalar,
        dt_seconds=dt_seconds,
        dtype=dtype,
        sample_start_step=sample_start_step,
        sample_every_steps=sample_every_steps,
        steps=steps,
        advection_frame_velocity_m_s=advection_frame_velocity_m_s,
    )


def load_abl(path: str | Path) -> BoussinesqCase:
    """Load the one fixed ABL schema and compose finite-volume case components."""

    source = Path(path)
    from .document import native_document
    document = native_document(path)
    expected_tables = {
        "case",
        "domain",
        "flow",
        "scalar",
        "time",
        "numerics",
        "diagnostics",
    }
    optional_tables = {
        "finite_volume",
        "finite_volume_turbine",
        "finite_volume_workflow",
        "finite_volume_cooling",
        "moisture",
        "water_spray",
        "surface_scalar",
    }
    if not expected_tables <= document.keys() or not document.keys() <= (
        expected_tables | optional_tables
    ):
        missing = expected_tables - document.keys()
        unknown = document.keys() - expected_tables - optional_tables
        details = []
        if missing:
            details.append("missing tables: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown tables: " + ", ".join(sorted(unknown)))
        raise ValueError("invalid ABL document; " + "; ".join(details))

    case = _table(document, "case")
    domain = _table(document, "domain")
    flow = _table(document, "flow")
    scalar = _table(document, "scalar")
    time = _table(document, "time")
    numerics = _table(document, "numerics")
    diagnostics = _table(document, "diagnostics")
    surface_scalar = (
        _table(document, "surface_scalar")
        if "surface_scalar" in document
        else None
    )
    _keys(
        case,
        {"name", "citation", "initial_profile", "reference_results", "seed"},
        name="case",
        optional={"profile_resampling"},
    )
    _keys(
        domain,
        {"cells", "lengths_m"},
        name="domain",
        optional={"mapping"},
    )
    _keys(
        flow,
        {
            "pressure_acceleration_m_s2",
            "geostrophic_velocity_m_s",
            "advection_frame_velocity_m_s",
            "coriolis_s",
            "roughness_length_m",
            "von_karman",
        },
        name="flow",
    )
    _keys(
        scalar,
        {
            "reference_value",
            "surface_flux",
            "buoyancy_acceleration_per_unit",
        },
        name="scalar",
    )
    _keys(time, {"dt_seconds", "steps"}, name="time")
    _keys(numerics, {"dtype"}, name="numerics")
    _keys(
        diagnostics,
        {
            "sample_start_step",
            "sample_every_steps",
            "reference_length_m",
            "reference_velocity_m_s",
            "reference_scalar",
            "inversion_search_max_height_m",
            "spectrum_heights_m",
        },
        name="diagnostics",
    )
    if surface_scalar is not None:
        _keys(
            surface_scalar,
            {"initial_value", "rate_per_second", "roughness_length_m"},
            name="surface_scalar",
        )

    cells = _integers(domain, "cells", length=3)
    lengths = _numbers(domain, "lengths_m", length=3)
    physical_grid = _physical_grid(
        domain,
        (cells[0], cells[1], cells[2]),
        (lengths[0], lengths[1], lengths[2]),
    )
    pressure_acceleration = _numbers(
        flow, "pressure_acceleration_m_s2", length=2
    )
    geostrophic = _numbers(flow, "geostrophic_velocity_m_s", length=2)
    advection_frame = _numbers(
        flow, "advection_frame_velocity_m_s", length=2
    )
    coriolis = _numbers(flow, "coriolis_s", length=2)
    spectrum_heights = _numbers(diagnostics, "spectrum_heights_m")
    return compose_abl(
        name=_string(case, "name"),
        citation=_string(case, "citation"),
        cells=(cells[0], cells[1], cells[2]),
        lengths_m=(lengths[0], lengths[1], lengths[2]),
        scalar_reference_value=_number(scalar, "reference_value"),
        scalar_surface_flux=_number(scalar, "surface_flux"),
        surface_scalar=(
            SurfaceScalarEvolution(
                initial_value=_number(surface_scalar, "initial_value"),
                rate_per_second=_number(surface_scalar, "rate_per_second"),
                roughness_length_m=_number(
                    surface_scalar, "roughness_length_m"
                ),
            )
            if surface_scalar is not None
            else None
        ),
        buoyancy_acceleration_per_scalar=_number(
            scalar, "buoyancy_acceleration_per_unit"
        ),
        pressure_acceleration_m_s2=(
            pressure_acceleration[0],
            pressure_acceleration[1],
        ),
        geostrophic_velocity_m_s=(geostrophic[0], geostrophic[1]),
        advection_frame_velocity_m_s=(
            advection_frame[0],
            advection_frame[1],
        ),
        coriolis_s=(coriolis[0], coriolis[1]),
        roughness_length_m=_number(flow, "roughness_length_m"),
        von_karman=_number(flow, "von_karman"),
        initial_condition=TabulatedBoussinesqState(
            source.parent / _string(case, "initial_profile"),
            seed=_integer(case, "seed"),
            profile_resampling=case.get("profile_resampling", "strict"),
        ),
        reference_results=source.parent / _string(case, "reference_results"),
        dt_seconds=_number(time, "dt_seconds"),
        steps=_integer(time, "steps"),
        sample_start_step=_integer(diagnostics, "sample_start_step"),
        sample_every_steps=_integer(diagnostics, "sample_every_steps"),
        diagnostic_reference=DiagnosticReference(
            length_m=_number(diagnostics, "reference_length_m"),
            velocity_m_s=_number(diagnostics, "reference_velocity_m_s"),
            scalar=_number(diagnostics, "reference_scalar"),
            inversion_search_max_height_m=_number(
                diagnostics, "inversion_search_max_height_m"
            ),
            spectrum_heights_m=tuple(spectrum_heights),
        ),
        dtype=_string(numerics, "dtype"),
        physical_grid=physical_grid,
    )


__all__ = ["compose_abl", "load_abl"]
