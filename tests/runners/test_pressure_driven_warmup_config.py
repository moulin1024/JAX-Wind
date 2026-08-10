from __future__ import annotations

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


from jaxwind.runners.pressure_driven_warmup import ConfigError, load_case


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "runners" / "pressure_driven_warmup"
CONFIG = CASE_DIR / "config.toml"


def _cli_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = (
        source if not existing else os.pathsep.join((source, existing))
    )
    return environment


def test_canonical_case_resolves_physical_and_numerical_choices() -> None:
    case = load_case(CONFIG)
    assert case.runner == "pressure_driven_warmup"
    assert (case.domain.nx, case.domain.ny, case.domain.nz) == (64, 64, 64)
    assert case.domain.lx_m == pytest.approx(2_000.0 * math.pi)
    assert case.domain.ly_m == pytest.approx(2_000.0 * math.pi)
    assert case.domain.lz_m == 1_000.0
    assert (
        case.domain.dx_m,
        case.domain.dy_m,
        case.domain.dz_m,
    ) == pytest.approx((31.25 * math.pi, 31.25 * math.pi, 15.625))
    assert case.flow.friction_velocity_m_s == 0.4
    assert case.flow.roughness_length_m == 0.001
    assert case.flow.pressure_acceleration_m_s2 == pytest.approx(1.6e-4)
    assert case.wall.porte_agel_correction
    assert case.sgs.model == "lasd"
    assert case.time.integrator == "ab2"
    assert case.time.dt_seconds == 0.1
    assert case.time.steps == 360_000
    assert case.sample_start_step == 288_000
    assert case.output.log_every_steps == 1_000
    assert case.output.sample_every_steps == 100
    assert case.output.checkpoint_every_steps == 36_000
    assert case.estimated_startup_cfl < case.numerics.cfl_abort
    assert case.estimated_lasd_trajectory_cfl < case.numerics.lasd_trajectory_cfl_abort


def test_dry_run_prints_the_resolved_plan_without_loading_jax() -> None:
    assert not (CASE_DIR / "run.py").exists()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jaxwind",
            str(CASE_DIR),
            "--dry-run",
        ],
        cwd=ROOT,
        env=_cli_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = tomllib.loads(completed.stdout)
    assert resolved["runner"] == "pressure_driven_warmup"
    assert resolved["time"]["steps"] == 360_000
    assert resolved["flow"]["pressure_acceleration_m_s2"] == pytest.approx(1.6e-4)
    assert "jax" not in completed.stderr.lower()


def test_legacy_case_without_explicit_porte_agel_flag_keeps_correction(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy_config.toml"
    legacy.write_text(
        CONFIG.read_text().replace("porte_agel_correction = true\n", "")
    )
    assert load_case(legacy).wall.porte_agel_correction


@pytest.mark.parametrize("model", ("mgm", "amd"))
def test_config_rejects_removed_sgs_models(tmp_path: Path, model: str) -> None:
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(CONFIG.read_text().replace('model = "lasd"', f'model = "{model}"'))
    with pytest.raises(ConfigError, match="must be 'lasd'"):
        load_case(invalid)


def test_cli_runs_a_declarative_case_directory_without_run_py(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "copied_case"
    case_dir.mkdir()
    (case_dir / "config.toml").write_text(CONFIG.read_text())

    completed = subprocess.run(
        [sys.executable, "-m", "jaxwind", str(case_dir), "--dry-run"],
        cwd=ROOT,
        env=_cli_environment(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert not (case_dir / "run.py").exists()
    assert tomllib.loads(completed.stdout)["case"] == "pressure_driven_warmup_64x64x64"


def test_config_rejects_a_lasd_trajectory_that_crosses_the_abort_limit(
    tmp_path: Path,
) -> None:
    text = CONFIG.read_text().replace(
        "update_interval_steps = 4",
        "update_interval_steps = 100",
    )
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(text)
    with pytest.raises(ConfigError, match="trajectory CFL"):
        load_case(invalid)
