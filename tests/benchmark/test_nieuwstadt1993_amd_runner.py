from __future__ import annotations

import pytest

from benchmark.Nieuwstadt1993 import run, run_amd


def test_nieuwstadt_orchestrator_defaults_to_nonspectral_amd() -> None:
    args = run.parse_args([])

    assert not hasattr(args, "solver")
    assert args.amd_coefficient == 0.212
    assert args.dt == 1.25
    assert args.nx == 40
    assert args.ny == 40
    assert args.nz == 48


def test_nieuwstadt_rejects_removed_solver_selection() -> None:
    with pytest.raises(SystemExit):
        run.parse_args(["--solver", "lasd-semantic"])


def test_nonspectral_amd_runner_uses_canonical_case_and_averaging_window() -> None:
    args = run_amd.parse_args([])

    assert args.end_tstar == 11.0
    assert args.sample_start_tstar == 10.0
    assert args.amd_coefficient == 0.212
    assert args.scalar_amd_coefficient == 0.212
    assert args.dt_max == 1.25
    assert args.sgs_time_integration == "explicit"
    assert not hasattr(args, "projection_method")
    assert (args.nx, args.ny, args.nz) == (40, 40, 48)


def test_nieuwstadt_runners_accept_imex_full_projection_setup() -> None:
    flags = ["--sgs-time-integration", "imex_ark3"]

    assert run.parse_args(flags).sgs_time_integration == "imex_ark3"
    assert run_amd.parse_args(flags).sgs_time_integration == "imex_ark3"


def test_nieuwstadt_runners_reject_removed_projection_method_option() -> None:
    for parser in (run.parse_args, run_amd.parse_args):
        with pytest.raises(SystemExit):
            parser(["--projection-method", "fpj2"])


def test_quick_nonspectral_runner_has_bounded_end_to_end_scope() -> None:
    args = run_amd.parse_args(["--quick"])

    assert (args.nx, args.ny, args.nz) == (8, 8, 8)
    assert args.max_steps == 4
    assert args.sample_start_tstar == 0.0
