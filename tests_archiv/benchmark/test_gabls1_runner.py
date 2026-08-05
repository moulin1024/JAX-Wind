from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from benchmark.GABLS1 import run
from benchmark.GABLS1.reference import (
    SET_COLUMNS,
    ensemble_on_grid,
    load_period_sets,
    load_time_series,
)


REFERENCE = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "GABLS1"
    / "reference"
    / "official_12p5m"
)
REFERENCE_6P25 = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "GABLS1"
    / "reference"
    / "official_6p25m"
)


def test_gabls1_defaults_are_the_official_coarse_case() -> None:
    args = run.parse_args([])

    assert (args.nx, args.ny, args.nz) == (32, 32, 32)
    assert args.end_hours == 9.0
    assert args.sample_start_hours == 8.0
    assert args.amd_coefficient == 0.212
    assert args.scalar_amd_coefficient == 0.212
    assert args.target_cfl == 0.9
    assert not hasattr(args, "projection_method")
    assert args.coupling_integrator == "strang"
    assert args.sgs_time_integration == "explicit"
    assert args.rayleigh_sponge_start_height is None
    assert args.rayleigh_sponge_maximum_rate == 0.2
    assert not hasattr(args, "advection_limiter")
    assert args.pressure_smooth == 1
    assert args.metrics_every == 300
    assert args.reference_dir == REFERENCE


def test_gabls1_short_modes_keep_bounded_end_to_end_scope() -> None:
    quick = run.parse_args(["--quick"])
    smoke = run.parse_args(["--smoke"])

    assert (quick.nx, quick.ny, quick.nz) == (8, 8, 8)
    assert quick.max_steps == 4
    assert quick.sample_start_hours == 0.0
    assert quick.metrics_every == 1
    assert (smoke.nx, smoke.ny, smoke.nz) == (16, 16, 16)
    assert smoke.end_hours == 0.02
    assert smoke.sample_start_hours == 0.0
    assert smoke.metrics_every == 20


def test_gabls1_accepts_coupled_ssprk3() -> None:
    args = run.parse_args(["--coupling-integrator", "coupled-ssprk3"])

    assert args.coupling_integrator == "coupled-ssprk3"


def test_gabls1_accepts_imex_ark3_with_full_projection() -> None:
    args = run.parse_args(
        [
            "--quick",
            "--sgs-time-integration",
            "imex_ark3",
        ]
    )

    coupled, _, _ = run._build_coupled(args)

    assert coupled.momentum.config.sgs_time_integration == "imex_ark3"
    assert coupled.surface_law is not None


def test_gabls1_rejects_removed_projection_method_option() -> None:
    with pytest.raises(SystemExit):
        run.parse_args(["--projection-method", "fpj2"])


def test_gabls1_builds_optional_rayleigh_sponge() -> None:
    args = run.parse_args(
        [
            "--quick",
            "--rayleigh-sponge-start-height",
            "300",
            "--rayleigh-sponge-maximum-rate",
            "0.2",
        ]
    )

    coupled, _, _ = run._build_coupled(args)

    assert coupled.config.rayleigh_sponge_start_height == 300.0
    assert coupled.config.rayleigh_sponge_maximum_rate == 0.2


def test_gabls1_rejects_imex_with_coupled_ssprk3() -> None:
    with pytest.raises(SystemExit):
        run.parse_args(
            [
                "--sgs-time-integration",
                "imex_ark3",
                "--coupling-integrator",
                "coupled-ssprk3",
            ]
        )


def test_gabls1_exposes_only_mp5_advection() -> None:
    with pytest.raises(SystemExit):
        run.parse_args(["--advection-limiter", "muscl-mc"])

    args = run.parse_args(["--advection-dissipation-strength", "0.75"])
    assert args.mp5_strength == 0.75
    assert run.parse_args(["--mp5-strength", "0.5"]).mp5_strength == 0.5


def test_gabls1_rejects_removed_mesh_option() -> None:
    with pytest.raises(SystemExit):
        run.parse_args(["--mesh", "m.json"])


def test_eta_log_fields_uses_measured_simulation_throughput() -> None:
    status = run._eta_log_fields(
        start_wall=10.0,
        current_wall=110.0,
        start_simulation_time=1_000.0,
        simulation_time=1_050.0,
        final_simulation_time=1_150.0,
        now=datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
    )

    assert status == (
        "wall=00:01:40 speed=0.50x remain=00:03:20 ETA=2026-08-02T12:03:20+00:00"
    )


def test_eta_log_fields_handles_startup_without_progress() -> None:
    status = run._eta_log_fields(
        start_wall=10.0,
        current_wall=10.0,
        start_simulation_time=1_000.0,
        simulation_time=1_000.0,
        final_simulation_time=1_150.0,
    )

    assert status == (
        "wall=00:00:00 speed=calculating remain=calculating ETA=calculating"
    )


def test_official_12p5m_archive_parses_all_sets_and_participants() -> None:
    for set_name in "ABCD":
        datasets = load_period_sets(REFERENCE, set_name, period=9)
        assert len(datasets) == 7
        assert tuple(datasets[0].values) == SET_COLUMNS[set_name]
    assert len(load_time_series(REFERENCE)) == 7


def test_official_6p25m_archive_parses_all_sets_and_participants() -> None:
    for set_name in "ABCD":
        datasets = load_period_sets(REFERENCE_6P25, set_name, period=9)
        assert len(datasets) == 10
        assert tuple(datasets[0].values) == SET_COLUMNS[set_name]
    assert len(load_time_series(REFERENCE_6P25)) == 10
    target = np.asarray((100.0, 200.0, 300.0))
    variance = ensemble_on_grid(
        load_period_sets(REFERENCE_6P25, "B", period=9),
        "z",
        target,
    )["w_var_resolved"]
    assert np.all(variance["minimum"] >= 0.0)


def test_reference_ensemble_interpolates_without_extrapolation() -> None:
    datasets = load_period_sets(REFERENCE, "A", period=9)
    target = np.asarray((6.25, 100.0, 393.75))

    ensemble = ensemble_on_grid(datasets, "z", target)

    assert np.all(ensemble["u_mean"]["count"] >= 1)
    assert np.all(np.isfinite(ensemble["theta_mean"]["mean"]))
