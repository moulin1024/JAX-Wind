from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from jaxwind.runners.direct_rigid_alm import load_case


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "runners" / "nrel5mw_direct_alm_smoke"
CONFIG = CASE_DIR / "config.toml"


def _cli_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = (
        source if not existing else os.pathsep.join((source, existing))
    )
    return environment


def test_direct_alm_case_has_requested_grid_and_centered_turbine() -> None:
    case = load_case(CONFIG)

    assert (
        case.domain.nx,
        case.domain.ny,
        case.domain.nz,
    ) == (128, 128, 512)
    assert (
        case.domain.lx_m,
        case.domain.ly_m,
        case.domain.lz_m,
    ) == (512.0, 512.0, 512.0)
    assert (
        case.domain.dx_m,
        case.domain.dy_m,
        case.domain.dz_m,
    ) == (4.0, 4.0, 1.0)
    assert case.cell_count == 8_388_608
    assert case.turbine.location_m == (256.0, 256.0)
    assert case.turbine.hub_height_m == pytest.approx(90.0, abs=1.0e-5)
    assert case.turbine.openfast.blade_count == 3
    assert len(case.turbine.openfast.element_radii_m) == 19
    assert case.aeroelastic.enabled
    assert case.turbine.modal_openfast is not None
    assert case.turbine.modal_openfast.enabled_modes == (
        True,
        True,
        True,
    )
    assert case.turbine.smoothing_width_m == 4.0
    assert case.time.dt_seconds == 0.05
    assert case.time.steps == 60
    assert case.output.flow_slice_every_steps == 2


def test_direct_alm_dry_run_validates_without_importing_jax() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "jaxwind", str(CASE_DIR), "--dry-run"],
        cwd=ROOT,
        env=_cli_environment(),
        check=True,
        capture_output=True,
        text=True,
    )

    resolved = tomllib.loads(completed.stdout)
    assert resolved["runner"] == "direct_aeroelastic_alm"
    assert resolved["domain"]["cells"] == [128, 128, 512]
    assert resolved["domain"]["cell_count"] == 8_388_608
    assert resolved["turbine"]["location_m"] == [256.0, 256.0]
    assert resolved["turbine"]["model"] == (
        "openfast_modal_aeroelastic_actuator_line"
    )
    assert resolved["aeroelastic"]["enabled"]
    assert resolved["aeroelastic"]["enabled_blade_modes"] == [
        "flap1",
        "flap2",
        "edge1",
    ]
    assert resolved["output"]["writes_full_field_checkpoint"] is False
    assert resolved["output"]["flow_slice_every_steps"] == 2
    assert "jax" not in completed.stderr.lower()
