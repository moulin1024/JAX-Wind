from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from jaxwind.runners.concurrent_precursor_adm import ConfigError, load_case


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "runners" / "dtu10mw_concurrent_precursor_smoke"
CONFIG = CASE_DIR / "config.toml"
PRODUCTION_CONFIG = (
    ROOT / "runners" / "dtu10mw_concurrent_precursor_1h" / "config.toml"
)


def _cli_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = (
        source if not existing else os.pathsep.join((source, existing))
    )
    return environment


def test_dtu_smoke_case_reuses_the_warmup_and_derives_adm_parameters() -> None:
    case = load_case(CONFIG)

    assert case.runner == "concurrent_precursor_adm"
    assert case.base.name == "pressure_driven_warmup_128x64x256"
    assert (
        case.base.domain.nx,
        case.base.domain.ny,
        case.base.domain.nz,
    ) == (128, 64, 256)
    assert case.concurrent.steps == 2
    assert case.lasd_update_interval_steps == 4
    assert case.output.field_sample_every_steps is None
    assert case.fringe.relaxation_time_seconds == 4.0
    assert case.fringe_plateau_width_m == 256.0
    assert case.effective_fringe_damping_length_m == 384.0
    assert case.predicted_fringe_residual_fraction < 0.01
    assert case.turbine.diameter_m == 178.3
    assert case.turbine.hub_height_m == 119.0
    assert case.turbine.local_thrust_coefficient == pytest.approx(
        1.5278640450004206
    )
    assert case.rotor_cells_y == pytest.approx(
        case.turbine.diameter_m / case.base.domain.dy_m
    )
    assert case.rotor_cells_y > 8.0
    assert case.warmup.checkpoint == (
        ROOT
        / "outputs/pressure_driven_warmup_128x64x256/checkpoint_final.npz"
    )


def test_dtu_smoke_case_dry_run_does_not_import_jax() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "jaxwind", str(CONFIG), "--dry-run"],
        cwd=ROOT,
        env=_cli_environment(),
        check=True,
        capture_output=True,
        text=True,
    )

    resolved = tomllib.loads(completed.stdout)
    assert resolved["runner"] == "concurrent_precursor_adm"
    assert resolved["turbine"]["diameter_m"] == 178.3
    assert resolved["turbine"]["under_resolved_for_science"] is False
    assert math.isclose(resolved["time"]["additional_duration_seconds"], 0.2)
    assert resolved["fringe"]["plateau_width_m"] == 256.0
    assert resolved["fringe"]["relaxes_lasd_closure_memory"] is True
    assert resolved["fringe"]["predicted_residual_fraction"] < 0.01
    assert "jax" not in completed.stderr.lower()


def test_dtu_one_hour_case_samples_velocity_every_ten_seconds() -> None:
    case = load_case(PRODUCTION_CONFIG)

    assert case.concurrent.steps == 36_000
    assert case.concurrent.launch == "serial"
    assert case.base.sgs.update_interval_steps == 4
    assert case.concurrent.lasd_update_interval_steps == 8
    assert case.lasd_update_interval_steps == 8
    assert case.estimated_lasd_trajectory_cfl == pytest.approx(
        0.7068636024872894
    )
    assert case.base.time.dt_seconds == 0.1
    assert case.output.field_sample_every_steps == 100
    assert (
        case.concurrent.steps * case.base.time.dt_seconds
    ) == pytest.approx(3600.0)
    assert (
        case.concurrent.steps // case.output.field_sample_every_steps
    ) == 360
    resolved = case.resolved()
    assert resolved["sgs"]["update_interval_steps"] == 8
    assert resolved["concurrent"]["warmup_lasd_update_interval_steps"] == 4


def test_case_rejects_a_fringe_that_cannot_meet_its_attenuation_target() -> None:
    case = load_case(CONFIG)

    with pytest.raises(ConfigError, match="attenuation is too weak"):
        replace(
            case,
            fringe=replace(
                case.fringe,
                relaxation_time_seconds=50.0,
            ),
        )
