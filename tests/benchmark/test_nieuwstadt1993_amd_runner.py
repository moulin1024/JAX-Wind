from __future__ import annotations

from benchmark.Nieuwstadt1993 import run, run_amd


def test_nieuwstadt_orchestrator_defaults_to_nonspectral_amd() -> None:
    args = run.parse_args([])

    assert args.solver == "amd-nonspectral"
    assert args.amd_coefficient == 0.212
    assert args.dt == 1.25
    assert args.nx == 40
    assert args.ny == 40
    assert args.nz == 48


def test_nonspectral_amd_runner_uses_canonical_case_and_averaging_window() -> None:
    args = run_amd.parse_args([])

    assert args.end_tstar == 11.0
    assert args.sample_start_tstar == 10.0
    assert args.amd_coefficient == 0.212
    assert args.scalar_amd_coefficient == 0.212
    assert args.dt_max == 1.25
    assert (args.nx, args.ny, args.nz) == (40, 40, 48)


def test_quick_nonspectral_runner_has_bounded_end_to_end_scope() -> None:
    args = run_amd.parse_args(["--quick"])

    assert (args.nx, args.ny, args.nz) == (8, 8, 8)
    assert args.max_steps == 4
    assert args.sample_start_tstar == 0.0
