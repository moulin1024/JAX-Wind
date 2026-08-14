from __future__ import annotations

import json
from pathlib import Path

import pytest

from applications.abl.__main__ import main
from applications.abl.config import load_abl
from applications.initial_conditions import REQUIRED_COLUMNS, load_initial_profile
from jaxwind.physics import (
    CoriolisGeostrophic,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LinearBoussinesqBuoyancy,
    MoninObukhovSurfaceTransfer,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "cases" / "GABLS1" / "config.toml"
CASE = load_abl(CONFIG)
CONFIG_64 = ROOT / "cases" / "GABLS1" / "config_64.toml"
CASE_64 = load_abl(CONFIG_64)


def test_case_is_physical_data_for_the_uniform_abl_solver() -> None:
    lower = CONFIG.read_text().lower()

    for selector in ("runner", "workflow", "model", "integrator", "stability"):
        assert selector not in lower
    assert "[surface_scalar]" in lower


def test_case_composes_generic_coupled_surface_exchange() -> None:
    assert isinstance(CASE.model.momentum.rotation, CoriolisGeostrophic)
    assert isinstance(CASE.model.momentum.sgs, LagrangianScaleDependentDynamic)
    assert isinstance(CASE.model.scalar_sgs, LagrangianScaleDependentScalarFlux)
    assert isinstance(CASE.model.buoyancy, LinearBoussinesqBuoyancy)
    assert isinstance(CASE.model.surface_transfer, MoninObukhovSurfaceTransfer)
    assert CASE.model.scalar_boundary.lower_flux == 0.0
    assert CASE.model.surface_transfer.surface_scalar_rate < 0.0


def test_grid_timing_and_physical_inputs_match_gabls1() -> None:
    grid = CASE.physical_grid

    assert (grid.nx, grid.ny, grid.nz) == (32, 32, 32)
    assert (grid.lx, grid.ly, grid.lz) == (400.0, 400.0, 400.0)
    assert grid.dz == pytest.approx(12.5)
    assert CASE.dt_seconds == pytest.approx(0.25)
    assert CASE.steps == 129600
    assert CASE.output.sample_start_step == 115200
    assert CASE.output.sample_every_steps == 240
    assert CASE.model.momentum.sgs.update_interval == 5
    assert CASE.advection_frame_velocity_m_s == (8.0, 0.0)
    assert CASE.model.momentum.rotation.geostrophic_x_velocity == 0.0
    assert CASE.model.surface_transfer.x_velocity_offset == pytest.approx(1.0)
    assert CASE.model.buoyancy.acceleration_per_temperature == pytest.approx(
        (9.81 / 263.5) * 400.0 / 8.0**2
    )


def test_initial_state_uses_the_shared_profile_format() -> None:
    table = load_initial_profile(CASE)

    assert table.dtype.names == REQUIRED_COLUMNS
    assert table.shape == (32,)
    assert table["u_m_s"][0] == pytest.approx(8.0)
    assert table["scalar"][7] == pytest.approx(265.0)
    assert table["scalar"][8] == pytest.approx(265.0625)
    assert table["scalar_rms"][0] == pytest.approx(0.1 / 3.0**0.5)
    assert table["scalar_rms"][4] == 0.0


def test_refined_case_preserves_physical_duration_and_sampling_window() -> None:
    grid = CASE_64.physical_grid
    table = load_initial_profile(CASE_64)

    assert (grid.nx, grid.ny, grid.nz) == (64, 64, 64)
    assert grid.dz == pytest.approx(6.25)
    assert CASE_64.dt_seconds == pytest.approx(1.0 / 12.0)
    assert CASE_64.steps == 388800
    assert CASE_64.output.sample_start_step == 345600
    assert CASE_64.output.sample_every_steps == 720
    assert CASE_64.model.momentum.sgs.update_interval == 8
    assert table.shape == (64,)
    assert table["z_m"][0] == pytest.approx(3.125)
    assert table["z_m"][-1] == pytest.approx(396.875)
    assert table["scalar"][16] == pytest.approx(265.03125)


def test_official_participant_data_are_active_reference_assets() -> None:
    reference = CASE.reference_results.parent / "official_12p5m"
    source = json.loads((reference / "SOURCE.json").read_text())

    assert source["files"] == 63
    assert len(tuple(reference.glob("*/*_A9_32.dat"))) == 7
    assert len(tuple(reference.glob("*/*_C9_32.dat"))) == 7

    refined_reference = CASE.reference_results.parent / "official_6p25m"
    refined_source = json.loads(
        (refined_reference / "SOURCE.json").read_text()
    )
    assert refined_source["resolution_m"] == pytest.approx(6.25)
    assert refined_source["files"] == 90
    assert len(tuple(refined_reference.glob("*/*_A9_64.dat"))) == 10
    assert len(tuple(refined_reference.glob("*/*_C9_64.dat"))) == 10
    assert len(tuple(refined_reference.glob("*/*_E_64.dat"))) == 10


def test_dry_run_resolves_the_same_application_entry_point(capsys) -> None:
    assert main([str(CONFIG), "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["case"] == "gabls1_lasd_32x32x32"
    assert result["physics"]["surface_transfer"] == (
        "MoninObukhovSurfaceTransfer"
    )
    assert result["physics"]["surface_scalar_rate_per_second"] == pytest.approx(
        -0.25 / 3600.0
    )
