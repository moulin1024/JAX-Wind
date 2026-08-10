"""Uniform ABL composition from physical case data."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from jaxwind.domain import ScaleSystem, UniformGrid
from jaxwind.integrators import AB2Config
from jaxwind.physics import (
    BoussinesqModel,
    ConservativeAdvection,
    ConservativeScalarAdvection,
    CoriolisGeostrophic,
    DryFlowModel,
    KinematicPressureGradient,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LinearBoussinesqBuoyancy,
    NeutralLogWall,
    NoRayleighDamping,
    NoRotation,
    ScalarFluxBoundary,
)

from applications.boussinesq import (
    BoussinesqCase,
    DiagnosticReference,
    OutputSchedule,
    PressureProjection,
    ScalarScaleSystem,
    TabulatedBoussinesqState,
)


def _table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] table")
    return value


def _keys(table: dict[str, Any], expected: set[str], *, name: str) -> None:
    missing = expected - table.keys()
    unknown = table.keys() - expected
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
    if not isinstance(value, list) or (length is not None and len(value) != length):
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


def compose_abl(
    *,
    name: str,
    citation: str,
    cells: tuple[int, int, int],
    lengths_m: tuple[float, float, float],
    length_scale_m: float,
    velocity_scale_m_s: float,
    scalar_scale: float,
    scalar_quantity: str,
    scalar_reference_value: float,
    scalar_surface_flux: float,
    buoyancy_acceleration_per_scalar: float,
    pressure_acceleration_m_s2: tuple[float, float],
    geostrophic_velocity_m_s: tuple[float, float],
    coriolis_s: tuple[float, float],
    roughness_length_m: float,
    von_karman: float,
    initial_condition: TabulatedBoussinesqState,
    reference_results: str | Path,
    dt_seconds: float,
    steps: int,
    filter_grid_ratio: float,
    test_filter_ratio: float,
    lasd_update_interval_steps: int,
    lasd_timescale_coefficient: float,
    momentum_initial_coefficient: float,
    momentum_minimum_coefficient: float,
    momentum_maximum_coefficient: float,
    scalar_initial_coefficient: float,
    scalar_minimum_coefficient: float,
    scalar_maximum_coefficient: float,
    stratification_beta: float,
    stratification_power: float,
    output_directory: str | Path,
    sample_start_step: int,
    sample_every_steps: int,
    log_every_steps: int,
    checkpoint_every_steps: int,
    diagnostic_reference: DiagnosticReference,
    dtype: str,
    pressure_method: str,
    thomas_chunk: int,
    cfl_warning: float,
    cfl_abort: float,
    trajectory_cfl_abort: float,
    nonlinear_padding_ratio: float,
) -> BoussinesqCase:
    """Lower one set of physical inputs without classifying the flow regime."""

    grid = UniformGrid(*cells, *lengths_m)
    mechanical_scales = ScaleSystem(length_scale_m, velocity_scale_m_s)
    scalar_scales = ScalarScaleSystem(
        mechanical_scales,
        scalar_scale,
        scalar_quantity,
        scalar_reference_value,
    )
    vertical_coriolis, horizontal_coriolis = coriolis_s
    if vertical_coriolis == 0.0:
        if horizontal_coriolis != 0.0:
            raise ValueError("horizontal Coriolis requires vertical Coriolis")
        rotation = NoRotation()
    else:
        rotation = CoriolisGeostrophic(
            mechanical_scales.to_execution_inverse_time(vertical_coriolis),
            mechanical_scales.to_execution_velocity(geostrophic_velocity_m_s[0]),
            mechanical_scales.to_execution_velocity(geostrophic_velocity_m_s[1]),
            mechanical_scales.to_execution_inverse_time(horizontal_coriolis),
        )
    buoyancy_coefficient = scalar_scales.to_execution_buoyancy_coefficient(
        buoyancy_acceleration_per_scalar
    )
    momentum_sgs = LagrangianScaleDependentDynamic(
        filter_grid_ratio=filter_grid_ratio,
        test_filter_ratio=test_filter_ratio,
        update_interval=lasd_update_interval_steps,
        timescale_coefficient=lasd_timescale_coefficient,
        initial_coefficient=momentum_initial_coefficient,
        minimum_coefficient=momentum_minimum_coefficient,
        maximum_coefficient=momentum_maximum_coefficient,
    )
    scalar_sgs = LagrangianScaleDependentScalarFlux(
        initial_coefficient=scalar_initial_coefficient,
        minimum_coefficient=scalar_minimum_coefficient,
        maximum_coefficient=scalar_maximum_coefficient,
        stability_buoyancy_coefficient=buoyancy_coefficient,
        stability_beta=stratification_beta,
        stability_power=stratification_power,
    )
    model = BoussinesqModel(
        momentum=DryFlowModel(
            advection=ConservativeAdvection(),
            pressure_gradient=KinematicPressureGradient(
                mechanical_scales.to_execution_acceleration(
                    pressure_acceleration_m_s2[0]
                ),
                mechanical_scales.to_execution_acceleration(
                    pressure_acceleration_m_s2[1]
                ),
            ),
            wall=NeutralLogWall(
                mechanical_scales.to_execution_length(roughness_length_m),
                von_karman=von_karman,
            ),
            sgs=momentum_sgs,
            rotation=rotation,
        ),
        scalar_advection=ConservativeScalarAdvection(),
        scalar_sgs=scalar_sgs,
        buoyancy=LinearBoussinesqBuoyancy(buoyancy_coefficient),
        rayleigh_damping=NoRayleighDamping(),
        scalar_boundary=ScalarFluxBoundary(
            scalar_scales.to_execution_flux(scalar_surface_flux),
            0.0,
        ),
    )
    return BoussinesqCase(
        name=name,
        citation=citation,
        physical_grid=grid,
        mechanical_scales=mechanical_scales,
        scalar_scales=scalar_scales,
        model=model,
        integrator=AB2Config(mechanical_scales.to_execution_time(dt_seconds)),
        initial_condition=initial_condition,
        diagnostic_reference=diagnostic_reference,
        reference_results=Path(reference_results),
        pressure=PressureProjection(dtype, pressure_method, thomas_chunk),
        output=OutputSchedule(
            directory=Path(output_directory),
            sample_start_step=sample_start_step,
            sample_every_steps=sample_every_steps,
            log_every_steps=log_every_steps,
            checkpoint_every_steps=checkpoint_every_steps,
        ),
        steps=steps,
        cfl_warning=cfl_warning,
        cfl_abort=cfl_abort,
        trajectory_cfl_abort=trajectory_cfl_abort,
        nonlinear_padding_ratio=nonlinear_padding_ratio,
    )


def load_abl(path: str | Path) -> BoussinesqCase:
    """Load the one fixed ABL schema and compose generic solver components."""

    source = Path(path)
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    expected_tables = {
        "case",
        "domain",
        "scales",
        "flow",
        "scalar",
        "time",
        "sgs",
        "numerics",
        "diagnostics",
        "output",
    }
    if set(document) != expected_tables:
        missing = expected_tables - document.keys()
        unknown = document.keys() - expected_tables
        details = []
        if missing:
            details.append("missing tables: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown tables: " + ", ".join(sorted(unknown)))
        raise ValueError("invalid ABL document; " + "; ".join(details))

    case = _table(document, "case")
    domain = _table(document, "domain")
    scales = _table(document, "scales")
    flow = _table(document, "flow")
    scalar = _table(document, "scalar")
    time = _table(document, "time")
    sgs = _table(document, "sgs")
    numerics = _table(document, "numerics")
    diagnostics = _table(document, "diagnostics")
    output = _table(document, "output")
    _keys(
        case,
        {"name", "citation", "initial_profile", "reference_results", "seed"},
        name="case",
    )
    _keys(domain, {"cells", "lengths_m"}, name="domain")
    _keys(scales, {"length_m", "velocity_m_s", "scalar"}, name="scales")
    _keys(
        flow,
        {
            "pressure_acceleration_m_s2",
            "geostrophic_velocity_m_s",
            "coriolis_s",
            "roughness_length_m",
            "von_karman",
        },
        name="flow",
    )
    _keys(
        scalar,
        {
            "quantity",
            "reference_value",
            "surface_flux",
            "buoyancy_acceleration_per_unit",
        },
        name="scalar",
    )
    _keys(time, {"dt_seconds", "steps"}, name="time")
    _keys(
        sgs,
        {
            "filter_grid_ratio",
            "test_filter_ratio",
            "update_interval_steps",
            "timescale_coefficient",
            "momentum_initial_coefficient",
            "momentum_minimum_coefficient",
            "momentum_maximum_coefficient",
            "scalar_initial_coefficient",
            "scalar_minimum_coefficient",
            "scalar_maximum_coefficient",
            "stratification_beta",
            "stratification_power",
        },
        name="sgs",
    )
    _keys(
        numerics,
        {
            "dtype",
            "pressure_method",
            "thomas_chunk",
            "cfl_warning",
            "cfl_abort",
            "trajectory_cfl_abort",
            "nonlinear_padding_ratio",
        },
        name="numerics",
    )
    _keys(
        diagnostics,
        {
            "sample_start_step",
            "sample_every_steps",
            "log_every_steps",
            "checkpoint_every_steps",
            "reference_length_m",
            "reference_velocity_m_s",
            "reference_scalar",
            "inversion_search_max_height_m",
            "spectrum_heights_m",
        },
        name="diagnostics",
    )
    _keys(output, {"directory"}, name="output")

    cells = _integers(domain, "cells", length=3)
    lengths = _numbers(domain, "lengths_m", length=3)
    pressure_acceleration = _numbers(
        flow, "pressure_acceleration_m_s2", length=2
    )
    geostrophic = _numbers(flow, "geostrophic_velocity_m_s", length=2)
    coriolis = _numbers(flow, "coriolis_s", length=2)
    spectrum_heights = _numbers(diagnostics, "spectrum_heights_m")
    return compose_abl(
        name=_string(case, "name"),
        citation=_string(case, "citation"),
        cells=(cells[0], cells[1], cells[2]),
        lengths_m=(lengths[0], lengths[1], lengths[2]),
        length_scale_m=_number(scales, "length_m"),
        velocity_scale_m_s=_number(scales, "velocity_m_s"),
        scalar_scale=_number(scales, "scalar"),
        scalar_quantity=_string(scalar, "quantity"),
        scalar_reference_value=_number(scalar, "reference_value"),
        scalar_surface_flux=_number(scalar, "surface_flux"),
        buoyancy_acceleration_per_scalar=_number(
            scalar, "buoyancy_acceleration_per_unit"
        ),
        pressure_acceleration_m_s2=(
            pressure_acceleration[0],
            pressure_acceleration[1],
        ),
        geostrophic_velocity_m_s=(geostrophic[0], geostrophic[1]),
        coriolis_s=(coriolis[0], coriolis[1]),
        roughness_length_m=_number(flow, "roughness_length_m"),
        von_karman=_number(flow, "von_karman"),
        initial_condition=TabulatedBoussinesqState(
            source.parent / _string(case, "initial_profile"),
            seed=_integer(case, "seed"),
        ),
        reference_results=source.parent / _string(case, "reference_results"),
        dt_seconds=_number(time, "dt_seconds"),
        steps=_integer(time, "steps"),
        filter_grid_ratio=_number(sgs, "filter_grid_ratio"),
        test_filter_ratio=_number(sgs, "test_filter_ratio"),
        lasd_update_interval_steps=_integer(sgs, "update_interval_steps"),
        lasd_timescale_coefficient=_number(sgs, "timescale_coefficient"),
        momentum_initial_coefficient=_number(
            sgs, "momentum_initial_coefficient"
        ),
        momentum_minimum_coefficient=_number(
            sgs, "momentum_minimum_coefficient"
        ),
        momentum_maximum_coefficient=_number(
            sgs, "momentum_maximum_coefficient"
        ),
        scalar_initial_coefficient=_number(sgs, "scalar_initial_coefficient"),
        scalar_minimum_coefficient=_number(sgs, "scalar_minimum_coefficient"),
        scalar_maximum_coefficient=_number(sgs, "scalar_maximum_coefficient"),
        stratification_beta=_number(sgs, "stratification_beta"),
        stratification_power=_number(sgs, "stratification_power"),
        output_directory=_string(output, "directory"),
        sample_start_step=_integer(diagnostics, "sample_start_step"),
        sample_every_steps=_integer(diagnostics, "sample_every_steps"),
        log_every_steps=_integer(diagnostics, "log_every_steps"),
        checkpoint_every_steps=_integer(
            diagnostics, "checkpoint_every_steps"
        ),
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
        pressure_method=_string(numerics, "pressure_method"),
        thomas_chunk=_integer(numerics, "thomas_chunk"),
        cfl_warning=_number(numerics, "cfl_warning"),
        cfl_abort=_number(numerics, "cfl_abort"),
        trajectory_cfl_abort=_number(numerics, "trajectory_cfl_abort"),
        nonlinear_padding_ratio=_number(numerics, "nonlinear_padding_ratio"),
    )


__all__ = ["compose_abl", "load_abl"]
