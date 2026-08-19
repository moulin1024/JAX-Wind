"""Run and validate a data-configured wind-farm benchmark workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from applications.pressure_driven_lasd.config import CaseConfig, ConfigError, load_case

from .legacy_inflow import STRICT_LEGACY_INFLOW


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "cases" / "DTU10MWPrecursor" / "benchmark_adbem.toml"


def _table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value: Any = document
    for component in name.split("."):
        if not isinstance(value, dict):
            break
        value = value.get(component)
    if not isinstance(value, dict):
        raise ConfigError(f"missing [{name}] table")
    return value


def _string(table: dict[str, Any], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _number(table: dict[str, Any], key: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{key} must be finite")
    return result


def _integer(table: dict[str, Any], key: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    return value


def _boolean(table: dict[str, Any], key: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _steps(duration_hours: float, dt_seconds: float, label: str) -> int:
    raw = duration_hours * 3600.0 / dt_seconds
    steps = int(round(raw))
    if not math.isclose(raw, steps, rel_tol=0.0, abs_tol=1.0e-10):
        raise ConfigError(f"{label} must contain an integer number of time steps")
    return steps


@dataclass(frozen=True, slots=True)
class TurbineBenchmark:
    model: str
    openfast_model_environment: str
    x_m: float
    hub_height_m: float
    rotor_diameter_m: float
    blade_count: int
    radial_stations: int
    rotor_speed_rpm: float
    blade_pitch_degrees: float
    smearing_azimuthal_elements: int
    body_smoothing_width_m: float
    nacelle_drag_coefficient: float
    tower_drag_coefficient: float


@dataclass(frozen=True, slots=True)
class WakeModelBenchmark:
    thrust_coefficient: float
    spinup_seconds: float
    fit_min_d: float
    fit_max_d: float
    maximum_deficit_rmse: float
    maximum_expansion_rate_relative_error: float


@dataclass(frozen=True, slots=True)
class WindFarmBenchmark:
    source: Path
    case: CaseConfig
    schema: str
    output_directory: Path
    precursor_steps: int
    main_steps: int
    sample_buffer: int
    read_buffer: int
    compression: str
    frame_count: int
    gif_fps: int
    turbine: TurbineBenchmark
    wake_model: WakeModelBenchmark

    @property
    def warmup_directory(self) -> Path:
        return ROOT / self.case.output.directory

    @property
    def result_directory(self) -> Path:
        return ROOT / self.output_directory

    def resolved(self, *, openfast_model: str | None) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "config": str(self.source),
            "warmup": {
                "duration_hours": self.case.time.duration_hours,
                "steps": self.case.time.steps,
                "dt_seconds": self.case.time.dt_seconds,
                "output": str(self.warmup_directory),
            },
            "precursor": {
                "steps": self.precursor_steps,
                "duration_hours": (
                    self.precursor_steps * self.case.time.dt_seconds / 3600.0
                ),
                "sample_every_steps": STRICT_LEGACY_INFLOW.update_interval_steps,
                "sample_buffer": self.sample_buffer,
                "compression": self.compression,
            },
            "main": {
                "steps": self.main_steps,
                "duration_hours": (
                    self.main_steps * self.case.time.dt_seconds / 3600.0
                ),
                "output": str(self.result_directory),
                "compatibility": "strict-cuda-fortran",
                "pressure_gradient": False,
                "fringe": False,
                "inflow_planes": [
                    STRICT_LEGACY_INFLOW.start_plane,
                    STRICT_LEGACY_INFLOW.end_plane,
                ],
            },
            "turbine": {
                **asdict(self.turbine),
                "openfast_model": openfast_model,
            },
            "wake_model": asdict(self.wake_model),
        }


def load_benchmark(path: str | Path) -> WindFarmBenchmark:
    source = Path(path).resolve()
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    case = load_case(source)
    benchmark = _table(document, "benchmark")
    compatibility = _table(document, "benchmark.compatibility")
    turbine_data = _table(document, "benchmark.turbine")
    wake_data = _table(document, "benchmark.wake_model")
    schema = _string(benchmark, "schema")
    if schema != "jaxwind.windfarm-benchmark.v1":
        raise ConfigError(f"unsupported wind-farm benchmark schema: {schema}")
    contract = STRICT_LEGACY_INFLOW
    expected_compatibility = {
        "mode": "strict-cuda-fortran",
        "inflow_start_plane": contract.start_plane,
        "inflow_end_plane": contract.end_plane,
        "inflow_update_steps": contract.update_interval_steps,
        "spanwise_cycle_updates": contract.cycle_interval_updates,
        "main_pressure_gradient": False,
        "fringe": False,
    }
    actual_compatibility = {
        "mode": _string(compatibility, "mode"),
        "inflow_start_plane": _integer(compatibility, "inflow_start_plane"),
        "inflow_end_plane": _integer(compatibility, "inflow_end_plane"),
        "inflow_update_steps": _integer(compatibility, "inflow_update_steps"),
        "spanwise_cycle_updates": _integer(
            compatibility, "spanwise_cycle_updates"
        ),
        "main_pressure_gradient": _boolean(
            compatibility, "main_pressure_gradient"
        ),
        "fringe": _boolean(compatibility, "fringe"),
    }
    if actual_compatibility != expected_compatibility:
        raise ConfigError(
            "benchmark compatibility table must match the strict CUDA-Fortran "
            "inlet contract"
        )
    precursor_steps = _steps(
        _number(benchmark, "precursor_duration_hours"),
        case.time.dt_seconds,
        "precursor duration",
    )
    main_steps = _steps(
        _number(benchmark, "main_duration_hours"),
        case.time.dt_seconds,
        "main duration",
    )
    if precursor_steps % contract.update_interval_steps:
        raise ConfigError("precursor duration must align with inlet updates")
    if main_steps > precursor_steps:
        raise ConfigError("main duration cannot exceed the precursor recording")
    compression = _string(benchmark, "compression")
    if compression not in ("none", "lzf", "gzip"):
        raise ConfigError("benchmark compression must be none, lzf, or gzip")
    turbine = TurbineBenchmark(
        model=_string(turbine_data, "model"),
        openfast_model_environment=_string(
            turbine_data, "openfast_model_environment"
        ),
        x_m=_number(turbine_data, "x_m"),
        hub_height_m=_number(turbine_data, "hub_height_m"),
        rotor_diameter_m=_number(turbine_data, "rotor_diameter_m"),
        blade_count=_integer(turbine_data, "blade_count"),
        radial_stations=_integer(turbine_data, "radial_stations"),
        rotor_speed_rpm=_number(turbine_data, "rotor_speed_rpm"),
        blade_pitch_degrees=_number(turbine_data, "blade_pitch_degrees"),
        smearing_azimuthal_elements=_integer(
            turbine_data, "smearing_azimuthal_elements"
        ),
        body_smoothing_width_m=_number(
            turbine_data, "body_smoothing_width_m"
        ),
        nacelle_drag_coefficient=_number(
            turbine_data, "nacelle_drag_coefficient"
        ),
        tower_drag_coefficient=_number(
            turbine_data, "tower_drag_coefficient"
        ),
    )
    if turbine.model != "dtu-10mw-ad-bem":
        raise ConfigError("benchmark turbine must be dtu-10mw-ad-bem")
    if not 0.0 < turbine.x_m < case.domain.lx_m:
        raise ConfigError("benchmark turbine x position must lie in the domain")
    if min(
        turbine.hub_height_m,
        turbine.rotor_diameter_m,
        turbine.rotor_speed_rpm,
    ) <= 0.0:
        raise ConfigError(
            "benchmark rotor geometry and prescribed speed must be positive"
        )
    if min(turbine.blade_count, turbine.radial_stations) <= 0:
        raise ConfigError("benchmark blade and radial-station counts must be positive")
    if turbine.smearing_azimuthal_elements <= 0:
        raise ConfigError("AD-BEM smearing count must be positive")
    wake_model = WakeModelBenchmark(
        thrust_coefficient=_number(wake_data, "thrust_coefficient"),
        spinup_seconds=_number(wake_data, "spinup_seconds"),
        fit_min_d=_number(wake_data, "fit_min_d"),
        fit_max_d=_number(wake_data, "fit_max_d"),
        maximum_deficit_rmse=_number(wake_data, "maximum_deficit_rmse"),
        maximum_expansion_rate_relative_error=_number(
            wake_data, "maximum_expansion_rate_relative_error"
        ),
    )
    if not 0.0 < wake_model.thrust_coefficient < 1.0:
        raise ConfigError("wake-model thrust coefficient must lie in (0, 1)")
    if not 0.0 <= wake_model.spinup_seconds < main_steps * case.time.dt_seconds:
        raise ConfigError("wake-model spinup must lie within the main run")
    if not 0.0 <= wake_model.fit_min_d < wake_model.fit_max_d:
        raise ConfigError("wake-model fit interval is invalid")
    if min(
        wake_model.maximum_deficit_rmse,
        wake_model.maximum_expansion_rate_relative_error,
    ) <= 0.0:
        raise ConfigError("wake-model acceptance tolerances must be positive")
    sample_buffer = _integer(benchmark, "sample_buffer")
    read_buffer = _integer(benchmark, "read_buffer")
    frame_count = _integer(benchmark, "frame_count")
    gif_fps = _integer(benchmark, "gif_fps")
    if min(sample_buffer, read_buffer, frame_count, gif_fps) <= 0:
        raise ConfigError("benchmark buffers and visualization counts must be positive")
    return WindFarmBenchmark(
        source=source,
        case=case,
        schema=schema,
        output_directory=Path(_string(benchmark, "output_directory")),
        precursor_steps=precursor_steps,
        main_steps=main_steps,
        sample_buffer=sample_buffer,
        read_buffer=read_buffer,
        compression=compression,
        frame_count=frame_count,
        gif_fps=gif_fps,
        turbine=turbine,
        wake_model=wake_model,
    )


def _warmup_command(config: WindFarmBenchmark, *, restart: Path | None) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "applications.pressure_driven_lasd",
        str(config.source),
        "--output",
        str(config.warmup_directory),
    ]
    if restart is not None:
        command.extend(("--restart", str(restart)))
    return command


def _precursor_main_command(
    config: WindFarmBenchmark,
    *,
    openfast_model: str,
    overwrite: bool,
) -> list[str]:
    turbine = config.turbine
    command = [
        sys.executable,
        "-u",
        "-m",
        "applications.windfarm_precursor",
        str(config.source),
        "--restart",
        str(config.warmup_directory / "checkpoint_final.npz"),
        "--output",
        str(config.result_directory),
        "--precursor-steps",
        str(config.precursor_steps),
        "--main-steps",
        str(config.main_steps),
        "--sample-buffer",
        str(config.sample_buffer),
        "--read-buffer",
        str(config.read_buffer),
        "--compression",
        config.compression,
        "--frames",
        str(config.frame_count),
        "--gif-fps",
        str(config.gif_fps),
        "--turbine",
        turbine.model,
        "--openfast-model",
        openfast_model,
        "--turbine-x-m",
        str(turbine.x_m),
        "--rotor-speed-rpm",
        str(turbine.rotor_speed_rpm),
        "--blade-pitch-degrees",
        str(turbine.blade_pitch_degrees),
        "--ad-bem-smearing-azimuthal-elements",
        str(turbine.smearing_azimuthal_elements),
        "--body-smoothing-width-m",
        str(turbine.body_smoothing_width_m),
        "--nacelle-drag-coefficient",
        str(turbine.nacelle_drag_coefficient),
        "--tower-drag-coefficient",
        str(turbine.tower_drag_coefficient),
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def _comparison_command(config: WindFarmBenchmark) -> list[str]:
    turbine = config.turbine
    wake = config.wake_model
    result = config.result_directory
    return [
        sys.executable,
        str(ROOT / "tools" / "compare_hub_height_gaussian_wake.py"),
        str(result / "main_xz_frames.npz"),
        "--precursor-recording",
        str(result / "precursor.h5"),
        "--precursor-dt-seconds",
        str(config.case.time.dt_seconds),
        "--output",
        str(result / "gaussian_wake_comparison"),
        "--turbine-x-m",
        str(turbine.x_m),
        "--hub-height-m",
        str(turbine.hub_height_m),
        "--rotor-diameter-m",
        str(turbine.rotor_diameter_m),
        "--ct",
        str(wake.thrust_coefficient),
        "--lx-m",
        str(config.case.domain.lx_m),
        "--ly-m",
        str(config.case.domain.ly_m),
        "--lz-m",
        str(config.case.domain.lz_m),
        "--spinup-seconds",
        str(wake.spinup_seconds),
        "--fit-min-d",
        str(wake.fit_min_d),
        "--fit-max-d",
        str(wake.fit_max_d),
    ]


def _check_openfast_geometry(
    config: WindFarmBenchmark,
    openfast_model: str,
) -> None:
    from jaxwind.windfarm import load_openfast_rigid_turbine

    rotor = load_openfast_rigid_turbine(openfast_model)
    expected = config.turbine
    mismatches = []
    if rotor.blade_count != expected.blade_count:
        mismatches.append(f"blade count {rotor.blade_count} != {expected.blade_count}")
    if len(rotor.element_radii_m) != expected.radial_stations:
        mismatches.append(
            "radial stations "
            f"{len(rotor.element_radii_m)} != {expected.radial_stations}"
        )
    if not math.isclose(
        rotor.hub_height_m,
        expected.hub_height_m,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        mismatches.append(
            f"hub height {rotor.hub_height_m} != {expected.hub_height_m} m"
        )
    diameter = 2.0 * rotor.tip_radius_m
    if not math.isclose(
        diameter,
        expected.rotor_diameter_m,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        mismatches.append(
            f"rotor diameter {diameter} != {expected.rotor_diameter_m} m"
        )
    if mismatches:
        raise ConfigError(
            "OpenFAST deck does not match the benchmark: " + "; ".join(mismatches)
        )


def _validate(config: WindFarmBenchmark) -> dict[str, Any]:
    path = (
        config.result_directory
        / "gaussian_wake_comparison"
        / "hub_height_wake_comparison.json"
    )
    result = json.loads(path.read_text())
    reference = result["reference_gaussian"]
    fitted = result["fitted_gaussian"]
    relative_error = abs(fitted["k"] - reference["k"]) / reference["k"]
    checks = {
        "deficit_rmse": {
            "value": reference["rmse_deficit"],
            "maximum": config.wake_model.maximum_deficit_rmse,
            "passed": (
                reference["rmse_deficit"]
                <= config.wake_model.maximum_deficit_rmse
            ),
        },
        "expansion_rate_relative_error": {
            "value": relative_error,
            "maximum": (
                config.wake_model.maximum_expansion_rate_relative_error
            ),
            "passed": (
                relative_error
                <= config.wake_model.maximum_expansion_rate_relative_error
            ),
        },
    }
    validation = {
        "schema": "jaxwind.windfarm-benchmark-validation.v1",
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "comparison": str(path),
    }
    target = config.result_directory / "benchmark_validation.json"
    target.write_text(json.dumps(validation, indent=2) + "\n")
    if not validation["passed"]:
        raise RuntimeError(f"wind-farm benchmark validation failed: {target}")
    return validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="restart the warmup and replace precursor/main benchmark outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_benchmark(args.config)
    environment_name = config.turbine.openfast_model_environment
    openfast_model = os.environ.get(environment_name)
    display_model = openfast_model or f"${{{environment_name}}}"
    if args.dry_run:
        print(
            json.dumps(
                {
                    **config.resolved(openfast_model=display_model),
                    "commands": {
                        "warmup": _warmup_command(config, restart=None),
                        "precursor_main": _precursor_main_command(
                            config,
                            openfast_model=display_model,
                            overwrite=args.overwrite,
                        ),
                        "comparison": _comparison_command(config),
                    },
                },
                indent=2,
            )
        )
        return 0
    if openfast_model is None or not Path(openfast_model).is_file():
        raise FileNotFoundError(
            f"set {environment_name} to the DTU-10MW OpenFAST .fst deck"
        )
    _check_openfast_geometry(config, openfast_model)
    final_checkpoint = config.warmup_directory / "checkpoint_final.npz"
    latest_checkpoint = config.warmup_directory / "checkpoint_latest.npz"
    if args.overwrite or not final_checkpoint.is_file():
        restart = (
            latest_checkpoint
            if not args.overwrite and latest_checkpoint.is_file()
            else None
        )
        warmup = _warmup_command(config, restart=restart)
        if args.overwrite:
            warmup.append("--overwrite")
        subprocess.run(warmup, cwd=ROOT, check=True)
    subprocess.run(
        _precursor_main_command(
            config,
            openfast_model=openfast_model,
            overwrite=args.overwrite,
        ),
        cwd=ROOT,
        check=True,
    )
    subprocess.run(_comparison_command(config), cwd=ROOT, check=True)
    validation = _validate(config)
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
