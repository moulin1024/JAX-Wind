"""ABL composition from physical case data."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from jaxwind.domain import PassiveScalarScaleSystem, ScaleSystem, UniformGrid
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
    NeutralLogWall,
    NoBuoyancy,
    NoRayleighDamping,
    ScalarFluxBoundary,
)

from applications.boussinesq import (
    BoussinesqCase,
    OutputSchedule,
    PressureProjection,
    TabulatedVelocityTKE,
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
    length: int,
) -> tuple[float, ...]:
    value = table[key]
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must contain {length} numbers")
    temporary = {str(index): item for index, item in enumerate(value)}
    return tuple(_number(temporary, str(index)) for index in range(length))


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


def _steps(seconds: float, dt_seconds: float, *, name: str) -> int:
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    raw = seconds / dt_seconds
    result = round(raw)
    if not math.isclose(raw, result, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{name} must contain an integer number of steps")
    return result


def compose_abl(
    *,
    name: str,
    citation: str,
    cells: tuple[int, int, int],
    lengths_m: tuple[float, float, float],
    geostrophic_velocity_m_s: tuple[float, float],
    coriolis_s: tuple[float, float],
    roughness_length_m: float,
    passive_scalar_surface_flux_kg_m2_s: float,
    initial_condition: TabulatedVelocityTKE,
    reference_results: str | Path,
    dt_seconds: float,
    duration_seconds: float,
    statistics_start_seconds: float,
    output_directory: str | Path | None = None,
    statistics_every_seconds: float = 240.0,
    log_every_seconds: float = 480.0,
    checkpoint_every_seconds: float = 4800.0,
    air_density_kg_m3: float = 1.0,
    scalar_reference_kg_m3: float = 1.0,
    von_karman: float = 0.4,
    lasd_update_interval_steps: int = 5,
    dtype: str = "float32",
    pressure_method: str = "spike",
    thomas_chunk: int = 20,
    cfl_warning: float = 0.25,
    cfl_abort: float = 1.0,
    trajectory_cfl_abort: float = 1.0,
    nonlinear_padding_ratio: float = 1.5,
) -> BoussinesqCase:
    """Compose a geostrophic ABL case from canonical SI inputs.

    This function performs no execution and never inspects ``name``. It owns
    the application-level numerical composition and lowers dimensional
    parameters to the generic solver's execution units. The scalar described
    by this schema is explicitly passive, so it cannot impose thermal
    stability on the flow.
    """

    if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("time step must be finite and positive")
    if not math.isfinite(air_density_kg_m3) or air_density_kg_m3 <= 0.0:
        raise ValueError("air density must be finite and positive")
    if not math.isfinite(passive_scalar_surface_flux_kg_m2_s):
        raise ValueError("passive-scalar surface flux must be finite")
    if not all(math.isfinite(value) for value in coriolis_s):
        raise ValueError("Coriolis parameters must be finite")
    vertical_coriolis, horizontal_coriolis = coriolis_s
    if vertical_coriolis == 0.0:
        raise ValueError("geostrophic ABL requires nonzero rotation")

    grid = UniformGrid(*cells, *lengths_m)
    geostrophic_speed = math.hypot(*geostrophic_velocity_m_s)
    if geostrophic_speed <= 0.0:
        raise ValueError("geostrophic velocity must be nonzero")
    mechanical_scales = ScaleSystem(grid.lz, geostrophic_speed)
    scalar_scales = PassiveScalarScaleSystem(
        mechanical_scales,
        scalar_reference_kg_m3,
    )
    momentum_lasd = LagrangianScaleDependentDynamic(
        filter_grid_ratio=1.5,
        test_filter_ratio=2.0,
        update_interval=lasd_update_interval_steps,
        timescale_coefficient=1.5,
        initial_coefficient=0.03,
        minimum_coefficient=1.0e-6,
        maximum_coefficient=0.81,
    )
    scalar_lasd = LagrangianScaleDependentScalarFlux()
    model = BoussinesqModel(
        momentum=DryFlowModel(
            advection=ConservativeAdvection(),
            pressure_gradient=KinematicPressureGradient(0.0, 0.0),
            wall=NeutralLogWall(
                mechanical_scales.to_execution_length(roughness_length_m),
                von_karman=von_karman,
            ),
            sgs=momentum_lasd,
            rotation=CoriolisGeostrophic(
                mechanical_scales.to_execution_inverse_time(vertical_coriolis),
                mechanical_scales.to_execution_velocity(
                    geostrophic_velocity_m_s[0]
                ),
                mechanical_scales.to_execution_velocity(
                    geostrophic_velocity_m_s[1]
                ),
                mechanical_scales.to_execution_inverse_time(horizontal_coriolis),
            ),
        ),
        scalar_advection=ConservativeScalarAdvection(),
        scalar_sgs=scalar_lasd,
        buoyancy=NoBuoyancy(),
        rayleigh_damping=NoRayleighDamping(),
        scalar_boundary=ScalarFluxBoundary(
            scalar_scales.to_execution_concentration_flux(
                passive_scalar_surface_flux_kg_m2_s / air_density_kg_m3
            ),
            0.0,
        ),
    )
    steps = _steps(duration_seconds, dt_seconds, name="duration")
    statistics_start_step = _steps(
        statistics_start_seconds,
        dt_seconds,
        name="statistics start",
    )
    output = Path(output_directory or Path("outputs") / name)
    return BoussinesqCase(
        name=name,
        citation=citation,
        physical_grid=grid,
        mechanical_scales=mechanical_scales,
        scalar_scales=scalar_scales,
        model=model,
        integrator=AB2Config(
            mechanical_scales.to_execution_time(dt_seconds)
        ),
        initial_condition=initial_condition,
        reference_results=Path(reference_results),
        pressure=PressureProjection(dtype, pressure_method, thomas_chunk),
        output=OutputSchedule(
            directory=output,
            sample_start_step=statistics_start_step,
            sample_every_steps=_steps(
                statistics_every_seconds,
                dt_seconds,
                name="statistics interval",
            ),
            log_every_steps=_steps(
                log_every_seconds,
                dt_seconds,
                name="log interval",
            ),
            checkpoint_every_steps=_steps(
                checkpoint_every_seconds,
                dt_seconds,
                name="checkpoint interval",
            ),
        ),
        steps=steps,
        cfl_warning=cfl_warning,
        cfl_abort=cfl_abort,
        trajectory_cfl_abort=trajectory_cfl_abort,
        nonlinear_padding_ratio=nonlinear_padding_ratio,
    )


def load_abl(path: str | Path) -> BoussinesqCase:
    """Load the fixed ABL schema and compose its generic components."""

    source = Path(path)
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    expected_tables = {
        "case",
        "domain",
        "flow",
        "passive_scalar",
        "time",
        "numerics",
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
    flow = _table(document, "flow")
    passive_scalar = _table(document, "passive_scalar")
    time = _table(document, "time")
    numerics = _table(document, "numerics")
    output = _table(document, "output")
    _keys(
        case,
        {"name", "citation", "initial_profile", "reference_results", "seed"},
        name="case",
    )
    _keys(domain, {"cells", "lengths_m"}, name="domain")
    _keys(
        flow,
        {
            "geostrophic_velocity_m_s",
            "coriolis_s",
            "roughness_length_m",
        },
        name="flow",
    )
    _keys(
        passive_scalar,
        {"surface_flux_kg_m2_s"},
        name="passive_scalar",
    )
    _keys(
        time,
        {
            "dt_seconds",
            "duration_seconds",
            "statistics_start_seconds",
            "statistics_every_seconds",
            "log_every_seconds",
            "checkpoint_every_seconds",
        },
        name="time",
    )
    _keys(
        numerics,
        {
            "lasd_update_interval_steps",
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
    _keys(output, {"directory"}, name="output")

    cells = _integers(domain, "cells", length=3)
    lengths = _numbers(domain, "lengths_m", length=3)
    geostrophic = _numbers(flow, "geostrophic_velocity_m_s", length=2)
    coriolis = _numbers(flow, "coriolis_s", length=2)
    return compose_abl(
        name=_string(case, "name"),
        citation=_string(case, "citation"),
        cells=(cells[0], cells[1], cells[2]),
        lengths_m=(lengths[0], lengths[1], lengths[2]),
        geostrophic_velocity_m_s=(geostrophic[0], geostrophic[1]),
        coriolis_s=(coriolis[0], coriolis[1]),
        roughness_length_m=_number(flow, "roughness_length_m"),
        passive_scalar_surface_flux_kg_m2_s=_number(
            passive_scalar,
            "surface_flux_kg_m2_s",
        ),
        initial_condition=TabulatedVelocityTKE(
            source.parent / _string(case, "initial_profile"),
            seed=_integer(case, "seed"),
        ),
        reference_results=source.parent / _string(case, "reference_results"),
        dt_seconds=_number(time, "dt_seconds"),
        duration_seconds=_number(time, "duration_seconds"),
        statistics_start_seconds=_number(time, "statistics_start_seconds"),
        output_directory=_string(output, "directory"),
        statistics_every_seconds=_number(time, "statistics_every_seconds"),
        log_every_seconds=_number(time, "log_every_seconds"),
        checkpoint_every_seconds=_number(time, "checkpoint_every_seconds"),
        lasd_update_interval_steps=_integer(
            numerics,
            "lasd_update_interval_steps",
        ),
        dtype=_string(numerics, "dtype"),
        pressure_method=_string(numerics, "pressure_method"),
        thomas_chunk=_integer(numerics, "thomas_chunk"),
        cfl_warning=_number(numerics, "cfl_warning"),
        cfl_abort=_number(numerics, "cfl_abort"),
        trajectory_cfl_abort=_number(numerics, "trajectory_cfl_abort"),
        nonlinear_padding_ratio=_number(numerics, "nonlinear_padding_ratio"),
    )


def derive_abl_stability(case: BoussinesqCase) -> str:
    """Derive the surface-forcing regime from the composed physical model.

    This value is diagnostic only. It never selects a solver or application.
    Passive scalars have no buoyancy feedback and therefore imply the neutral
    limit. Thermally active models are classified by their lower-boundary
    buoyancy flux.
    """

    from jaxwind.physics import LinearBoussinesqBuoyancy

    buoyancy = case.model.buoyancy
    if isinstance(buoyancy, NoBuoyancy):
        return "neutral"
    if not isinstance(buoyancy, LinearBoussinesqBuoyancy):
        raise TypeError("unsupported ABL buoyancy law")
    buoyancy_flux = (
        buoyancy.acceleration_per_temperature
        * case.model.scalar_boundary.lower_flux
    )
    if buoyancy_flux > 0.0:
        return "convective"
    if buoyancy_flux < 0.0:
        return "stable"
    return "stratification-controlled"


__all__ = ["compose_abl", "derive_abl_stability", "load_abl"]
