from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.yang2024.case import (  # noqa: E402
    INFLOW_FIT,
    OPERATING_POINTS,
    PAPER_CASE,
    local_thrust_coefficient,
    paper_settings,
)


def test_tunnel_geometry_reproduces_reported_blockage() -> None:
    assert PAPER_CASE.test_section_m == (24.0, 6.0, 3.6)
    assert PAPER_CASE.test_section_area_m2 == pytest.approx(21.6)
    assert PAPER_CASE.rotor_diameter_m == pytest.approx(1.26)
    assert PAPER_CASE.geometric_blockage_ratio == pytest.approx(
        PAPER_CASE.reported_blockage_ratio,
        abs=3.0e-4,
    )


def test_reference_contains_all_nine_paper_operating_points() -> None:
    assert tuple(OPERATING_POINTS) == tuple(f"R{i}" for i in range(1, 10))
    assert OPERATING_POINTS["R1"].wind_speed_m_s == pytest.approx(1.2)
    assert OPERATING_POINTS["R1"].mean_thrust_n == pytest.approx(2.29)
    rated = OPERATING_POINTS["R9"]
    assert rated.wind_speed_m_s == pytest.approx(4.4)
    assert rated.test_rotor_speed_rpm == pytest.approx(480.0)
    assert rated.measured_ct == pytest.approx(0.810)
    assert rated.measured_cp == pytest.approx(0.459)
    assert rated.mean_thrust_n == pytest.approx(12.21)
    assert rated.mean_torque_n_m == pytest.approx(0.61)


def test_measured_inflow_log_fit_is_recovered_from_versioned_data() -> None:
    assert INFLOW_FIT.friction_velocity_m_s == pytest.approx(
        0.1229413268,
        rel=1.0e-8,
    )
    assert 1.0e3 * INFLOW_FIT.roughness_length_m == pytest.approx(
        0.0161003204,
        rel=1.0e-8,
    )
    assert INFLOW_FIT.maximum_reconstruction_error_m_s < 1.0e-8
    assert INFLOW_FIT.minimum_height_m == pytest.approx(0.1)
    assert INFLOW_FIT.maximum_height_m == pytest.approx(2.0)


def test_rated_paper_uniform_settings_use_tunnel_and_measured_ct() -> None:
    settings = paper_settings()
    assert (settings["lx"], settings["ly"], settings["lz"]) == (
        24.0,
        6.0,
        3.6,
    )
    assert (settings["nx"], settings["ny"], settings["nz"]) == PAPER_CASE.grid
    assert settings["initial_condition"] == "uniform_flow"
    assert settings["uniform_u"] == pytest.approx(4.4)
    assert settings["momentum_wall_model"] == "free_slip"
    assert settings["fringe_enabled"] is True
    assert settings["actuator_disk_ct_prime"] == pytest.approx(
        local_thrust_coefficient(0.810)
    )


def test_measured_log_override_is_normalized_at_rated_hub_speed() -> None:
    settings = paper_settings(inflow_mode="measured_log")
    assert settings["initial_condition"] == "log_law"
    assert settings["momentum_wall_model"] == "abl"
    assert settings["wall_stress_model"] == "dynamic_neutral"
    assert settings["fringe_enabled"] is False
    assert settings["bl_height"] == pytest.approx(3.6)
    assert settings["pressure_force_height"] == pytest.approx(3.6)
    assert INFLOW_FIT.velocity_m_s(
        PAPER_CASE.hub_height_m,
        friction_velocity_m_s=float(settings["u_fric"]),
    ) == pytest.approx(4.4)


def test_nonrated_adm_is_rejected_without_measured_ct() -> None:
    with pytest.raises(ValueError, match="limited to R9"):
        paper_settings(condition="R8")


def test_quick_case_only_reduces_numerical_cost() -> None:
    settings = paper_settings(quick=True)
    assert (settings["lx"], settings["ly"], settings["lz"]) == (
        24.0,
        6.0,
        3.6,
    )
    assert (settings["nx"], settings["ny"], settings["nz"]) == (96, 24, 18)
    assert settings["steps"] == 4
    assert settings["use_jit"] is False


def test_dry_run_resolves_without_importing_jax() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmark" / "yang2024" / "run.py"),
            "--dry-run",
            "--quick",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["scope"] == "R9 rated pure-thrust actuator-disk milestone"
    assert payload["test_section_m"] == [24.0, 6.0, 3.6]
    assert payload["solver_settings"]["steps"] == 4
    assert "jax" not in completed.stderr.lower()


def test_local_thrust_coefficient_matches_momentum_theory() -> None:
    thrust_coefficient = 0.810
    induction = 0.5 * (1.0 - math.sqrt(1.0 - thrust_coefficient))
    assert local_thrust_coefficient(thrust_coefficient) == pytest.approx(
        thrust_coefficient / (1.0 - induction) ** 2
    )
    with pytest.raises(ValueError):
        local_thrust_coefficient(1.0)


def test_distributed_warmup_config_matches_requested_smoke_case() -> None:
    config_path = (
        ROOT
        / "benchmark"
        / "yang2024"
        / "configs"
        / "warmup_128x32x64.toml"
    )
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    assert (
        config["grid"]["nx"],
        config["grid"]["ny"],
        config["grid"]["nz"],
    ) == (128, 32, 64)
    assert (
        config["grid"]["lx"],
        config["grid"]["ly"],
        config["grid"]["lz"],
    ) == PAPER_CASE.test_section_m
    assert config["physics"]["u_fric"] == pytest.approx(
        INFLOW_FIT.friction_velocity_for_hub_speed(
            OPERATING_POINTS["R9"].wind_speed_m_s,
            PAPER_CASE.hub_height_m,
        )
    )
    assert config["physics"]["zo"] == pytest.approx(
        INFLOW_FIT.roughness_length_m
    )
    assert config["physics"]["wall_stress_model"] == "dynamic_neutral"
    assert config["physics"]["bl_height"] == pytest.approx(3.6)
    assert config["physics"]["pressure_force_height"] == pytest.approx(3.6)
    assert config["actuator_disk"]["enabled"] is False
    assert config["runtime"]["precision"] == "float32"
    assert config["runtime"]["sgs_precision"] == "float32"


def test_refined_distributed_warmup_config_matches_output_run() -> None:
    config_path = (
        ROOT
        / "benchmark"
        / "yang2024"
        / "configs"
        / "warmup_256x64x128.toml"
    )
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    assert (
        config["grid"]["nx"],
        config["grid"]["ny"],
        config["grid"]["nz"],
    ) == (256, 64, 128)
    assert config["time"]["steps"] == 10_000
    assert config["time"]["dt"] == pytest.approx(1.0e-3)
    assert config["time"]["log_every"] == 100
    assert config["physics"]["wall_stress_model"] == "dynamic_neutral"
    assert config["physics"]["bl_height"] == pytest.approx(3.6)
    assert config["physics"]["pressure_force_height"] == pytest.approx(3.6)
    assert config["sgs"]["model"] == "lasd"
