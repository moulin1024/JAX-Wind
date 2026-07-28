"""TOML loading for direct JAX-native actuator-line cases."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from jaxwind.openfast import (
    OpenFASTInputError,
    load_openfast_modal_turbine,
    load_openfast_rigid_turbine,
)

from .models import (
    AeroelasticConfig,
    CaseConfig,
    ConfigError,
    DomainConfig,
    FlowConfig,
    NumericsConfig,
    OutputConfig,
    StaticSgsConfig,
    TimeConfig,
    TurbineConfig,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def _table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing [{name}] table")
    return value


def _optional_table(
    document: dict[str, Any],
    name: str,
) -> dict[str, Any] | None:
    value = document.get(name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _boolean(table: dict[str, Any], key: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _string(table: dict[str, Any], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _integer(table: dict[str, Any], key: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    return value


def _optional_integer(
    table: dict[str, Any],
    key: str,
) -> int | None:
    return None if key not in table else _integer(table, key)


def _number(table: dict[str, Any], key: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{key} must be finite")
    return result


def _optional_number(table: dict[str, Any], key: str) -> float | None:
    return None if key not in table else _number(table, key)


def _location(table: dict[str, Any]) -> tuple[float, float]:
    value = table.get("location_m")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        raise ConfigError("turbine.location_m must contain numeric [x, y]")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ConfigError("turbine.location_m must be finite")
    return result


def _resolved_path(source: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (source.parent / path).resolve()


def load_case(path: str | Path) -> CaseConfig:
    source = Path(path).resolve()
    with source.open("rb") as stream:
        document = tomllib.load(stream)

    case = _table(document, "case")
    domain = _table(document, "domain")
    flow = _table(document, "flow")
    sgs = _table(document, "sgs")
    time = _table(document, "time")
    numerics = _table(document, "numerics")
    aeroelastic_table = _optional_table(document, "aeroelastic")
    turbine = _table(document, "turbine")
    output = _table(document, "output")

    input_file = _resolved_path(
        source,
        _string(turbine, "openfast_input_file"),
    )
    aeroelastic = (
        AeroelasticConfig()
        if aeroelastic_table is None
        else AeroelasticConfig(
            enabled=_boolean(aeroelastic_table, "enabled"),
            air_density_kg_m3=_number(
                aeroelastic_table,
                "air_density_kg_m3",
            ),
            gravity_m_s2=_number(
                aeroelastic_table,
                "gravity_m_s2",
            ),
            maximum_tip_deflection_m=_number(
                aeroelastic_table,
                "maximum_tip_deflection_m",
            ),
        )
    )
    try:
        modal_openfast = (
            load_openfast_modal_turbine(input_file)
            if aeroelastic.enabled
            else None
        )
        openfast = (
            modal_openfast.rigid
            if modal_openfast is not None
            else load_openfast_rigid_turbine(input_file)
        )
    except OpenFASTInputError as error:
        raise ConfigError(str(error)) from error
    return CaseConfig(
        runner=_string(case, "runner"),
        name=_string(case, "name"),
        domain=DomainConfig(
            nx=_integer(domain, "nx"),
            ny=_integer(domain, "ny"),
            nz=_integer(domain, "nz"),
            lx_m=_number(domain, "lx_m"),
            ly_m=_number(domain, "ly_m"),
            lz_m=_number(domain, "lz_m"),
        ),
        flow=FlowConfig(
            friction_velocity_m_s=_number(
                flow,
                "friction_velocity_m_s",
            ),
            roughness_length_m=_number(flow, "roughness_length_m"),
            forcing_height_m=_number(flow, "forcing_height_m"),
            von_karman=_number(flow, "von_karman"),
        ),
        sgs=StaticSgsConfig(
            coefficient=_number(sgs, "coefficient"),
        ),
        time=TimeConfig(
            dt_seconds=_number(time, "dt_seconds"),
            steps=_integer(time, "steps"),
        ),
        numerics=NumericsConfig(
            dtype=_string(numerics, "dtype"),
            pressure_method=_string(numerics, "pressure_method"),
            cfl_abort=_number(numerics, "cfl_abort"),
        ),
        aeroelastic=aeroelastic,
        turbine=TurbineConfig(
            location_m=_location(turbine),
            openfast_input_file=input_file,
            openfast=openfast,
            modal_openfast=modal_openfast,
            smoothing_width_m=_number(
                turbine,
                "smoothing_width_m",
            ),
            hub_height_override_m=_optional_number(
                turbine,
                "hub_height_m",
            ),
            rotor_speed_override_rpm=_optional_number(
                turbine,
                "rotor_speed_rpm",
            ),
            pitch_override_degrees=_optional_number(
                turbine,
                "pitch_degrees",
            ),
            yaw_override_degrees=_optional_number(
                turbine,
                "yaw_degrees",
            ),
            initial_azimuth_override_degrees=_optional_number(
                turbine,
                "initial_azimuth_degrees",
            ),
        ),
        output=OutputConfig(
            directory=_string(output, "directory"),
            log_every_steps=_integer(output, "log_every_steps"),
            flow_slice_every_steps=_optional_integer(
                output,
                "flow_slice_every_steps",
            ),
        ),
    )
