from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.LinPorteAgel2019.case import (
    PAPER_CASE,
    THRUST_COEFFICIENT_BY_YAW,
    local_thrust_coefficient,
    paper_settings,
)
from benchmark.LinPorteAgel2019 import run_precursor


def test_case_3_matches_paper_geometry_and_resolution() -> None:
    assert PAPER_CASE.domain == (6.4, 0.8, 0.4)
    assert PAPER_CASE.grid == (128, 64, 32)
    assert PAPER_CASE.rotor_diameter == 0.15
    assert PAPER_CASE.hub_height == 0.125
    _, ny, nz = PAPER_CASE.grid
    _, ly, lz = PAPER_CASE.domain
    assert PAPER_CASE.rotor_diameter / (ly / ny) == pytest.approx(12.0)
    assert PAPER_CASE.rotor_diameter / (lz / nz) == pytest.approx(12.0)


@pytest.mark.parametrize("yaw", PAPER_CASE.yaw_degrees)
def test_paper_yaw_uses_measured_thrust_and_rotor_normal_loading(yaw: float) -> None:
    settings = paper_settings(yaw)
    ct = THRUST_COEFFICIENT_BY_YAW[yaw]
    induction = 0.5 * (1.0 - math.sqrt(1.0 - ct))

    assert settings["actuator_disk_yaw_degrees"] == yaw
    assert settings["actuator_disk_ct_prime"] == pytest.approx(
        ct / (1.0 - induction) ** 2
    )
    assert settings["actuator_disk_enabled"] is True


def test_local_thrust_coefficient_rejects_non_momentum_range() -> None:
    with pytest.raises(ValueError):
        local_thrust_coefficient(1.0)


def test_quick_case_only_reduces_cost() -> None:
    settings = paper_settings(20.0, quick=True)
    assert (settings["lx"], settings["ly"], settings["lz"]) == PAPER_CASE.domain
    assert settings["steps"] == 8
    assert settings["use_jit"] is False


def test_precursor_defaults_match_paper_model_and_exclude_spinup() -> None:
    args = run_precursor.parse_args([])

    assert args.sgs == "lasd"
    assert args.wall_model == "filtered"
    assert args.lasd_update_interval == 10
    assert args.sample_start == 12_000
