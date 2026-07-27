from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from jaxwind.runners.concurrent_precursor_adm import load_case


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "runners" / "dtu10mw_concurrent_precursor_smoke"
CONFIG = CASE_DIR / "config.toml"


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
        [sys.executable, "-m", "jaxwind", str(CASE_DIR), "--dry-run"],
        cwd=ROOT,
        env=_cli_environment(),
        check=True,
        capture_output=True,
        text=True,
    )

    resolved = json.loads(completed.stdout)
    assert resolved["runner"] == "concurrent_precursor_adm"
    assert resolved["turbine"]["diameter_m"] == 178.3
    assert resolved["turbine"]["under_resolved_for_science"] is False
    assert math.isclose(resolved["time"]["additional_duration_seconds"], 0.2)
    assert "jax" not in completed.stderr.lower()
