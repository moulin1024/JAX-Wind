from __future__ import annotations

from pathlib import Path

import numpy as np

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


def test_gabls1_defaults_are_the_official_coarse_case() -> None:
    args = run.parse_args([])

    assert (args.nx, args.ny, args.nz) == (32, 32, 32)
    assert args.end_hours == 9.0
    assert args.sample_start_hours == 8.0
    assert args.amd_coefficient == 0.212
    assert args.scalar_amd_coefficient == 0.212
    assert args.reference_dir == REFERENCE


def test_gabls1_short_modes_keep_bounded_end_to_end_scope() -> None:
    quick = run.parse_args(["--quick"])
    smoke = run.parse_args(["--smoke"])

    assert (quick.nx, quick.ny, quick.nz) == (8, 8, 8)
    assert quick.max_steps == 4
    assert quick.sample_start_hours == 0.0
    assert (smoke.nx, smoke.ny, smoke.nz) == (16, 16, 16)
    assert smoke.end_hours == 0.02
    assert smoke.sample_start_hours == 0.0


def test_official_12p5m_archive_parses_all_sets_and_participants() -> None:
    for set_name in "ABCD":
        datasets = load_period_sets(REFERENCE, set_name, period=9)
        assert len(datasets) == 7
        assert tuple(datasets[0].values) == SET_COLUMNS[set_name]
    assert len(load_time_series(REFERENCE)) == 7


def test_reference_ensemble_interpolates_without_extrapolation() -> None:
    datasets = load_period_sets(REFERENCE, "A", period=9)
    target = np.asarray((6.25, 100.0, 393.75))

    ensemble = ensemble_on_grid(datasets, "z", target)

    assert np.all(ensemble["u_mean"]["count"] >= 1)
    assert np.all(np.isfinite(ensemble["theta_mean"]["mean"]))

