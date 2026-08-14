from __future__ import annotations

import json
from pathlib import Path

from applications.windfarm_precursor.__main__ import main
from applications.pressure_driven_lasd.config import load_case


ROOT = Path(__file__).resolve().parents[2]


def test_offline_precursor_dry_run_resolves_the_real_checkpoint(capsys) -> None:
    assert main(["--dry-run", "--precursor-steps", "2"]) == 0

    resolved = json.loads(capsys.readouterr().out)
    assert resolved["case"] == "pressure_driven_lasd_64x64x64"
    assert resolved["precursor_steps"] == 2
    assert resolved["main_steps"] == 2
    assert resolved["section"] == "inflow"
    assert resolved["recording"] is None
    assert resolved["turbine"] == "none"
    assert resolved["restart"].endswith(
        "outputs/pressure_driven_lasd_64x64x64_gpu/checkpoint_final.npz"
    )


def test_replay_dry_run_resolves_dtu_10mw_adm(capsys) -> None:
    assert main(
        [
            "--dry-run",
            "--recording",
            "outputs/precursor.h5",
            "--main-steps",
            "36000",
            "--turbine",
            "dtu-10mw-adm",
        ]
    ) == 0

    resolved = json.loads(capsys.readouterr().out)
    assert resolved["recording"] == "outputs/precursor.h5"
    assert resolved["main_steps"] == 36000
    assert resolved["turbine"] == "dtu-10mw-adm"
    assert resolved["thrust_coefficient_prime"] == 4.0 / 3.0


def test_dtu_10mw_case_has_requested_domain_and_resolution() -> None:
    case = load_case(ROOT / "cases" / "DTU10MWPrecursor" / "config.toml")

    assert (case.domain.lx_m, case.domain.ly_m, case.domain.lz_m) == (
        4096.0,
        1024.0,
        1024.0,
    )
    assert (case.domain.nx, case.domain.ny, case.domain.nz) == (128, 64, 256)
    assert (case.domain.dx_m, case.domain.dy_m, case.domain.dz_m) == (
        32.0,
        16.0,
        4.0,
    )
    assert case.time.duration_hours == 10.0
    assert case.time.dt_seconds == 0.2
    assert case.time.steps == 180_000
    assert case.numerics.pressure_tridiag == "pcr"
    assert case.output.checkpoint_every_steps == 18_000
    assert case.sgs.closure == "static-smagorinsky"


def test_dtu_10mw_lasd_benchmark_matches_the_production_grid() -> None:
    case = load_case(
        ROOT
        / "cases"
        / "DTU10MWPrecursor"
        / "config_lasd_benchmark.toml"
    )

    assert (case.domain.nx, case.domain.ny, case.domain.nz) == (128, 64, 256)
    assert case.time.dt_seconds == 0.2
    assert case.numerics.dtype == "float32"
    assert case.numerics.pressure_tridiag == "pcr"
    assert case.sgs.closure == "lasd"
    assert case.sgs.update_interval_steps == 4
    assert not case.sgs.scalar_lasd_enabled
    assert case.sgs.reuse_rhs_momentum_context
