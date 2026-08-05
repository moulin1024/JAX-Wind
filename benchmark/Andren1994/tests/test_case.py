from __future__ import annotations

import numpy as np
import pytest

from benchmark.Andren1994 import fig13_budget, run
from benchmark.Andren1994.overlay_paper_figures import (
    FIGURES,
    FIGURE13_AXES,
    FIGURE14_AXES,
    FIGURE15_AXES,
    FIGURE7_AXIS,
    Axis,
    _pixels,
)


def test_canonical_configuration_matches_paper_time_and_grid() -> None:
    args = run.parse_args([])
    assert args.end_ft == 0.1
    assert args.sample_start_ft == 0.05
    assert args.sgs == "amd"
    assert args.amd_coefficient == 0.212
    assert args.sgs_time_integration == "imex_ark3"
    assert not hasattr(args, "advection_limiter")
    assert not hasattr(args, "projection_method")


def test_runner_rejects_removed_advection_limiter_option() -> None:
    with pytest.raises(SystemExit):
        run.parse_args(["--advection-limiter", "muscl-mc"])


def test_table_a1_initial_profiles_are_complete() -> None:
    assert len(run.INITIAL_U) == len(run.INITIAL_V) == len(run.INITIAL_TKE) == 40
    assert run.INITIAL_U[12] == 10.71
    assert run.INITIAL_V[3] == 2.84
    assert run.INITIAL_TKE[0] == 0.365
    assert run.INITIAL_TKE[21] == 0.0


def test_runner_rejects_removed_legacy_quick_mode() -> None:
    with pytest.raises(SystemExit):
        run.parse_args(["--quick"])


def test_fig13_budget_normalization_and_tendency_are_explicit() -> None:
    first = {
        name: np.full(2, value)
        for name, value in zip(fig13_budget.TERMS, range(1, 6), strict=True)
    }
    first["resolved_flux"] = np.asarray([2.0, 4.0])
    second = {name: np.array(values, copy=True) for name, values in first.items()}
    second["resolved_flux"] = np.asarray([5.0, 10.0])
    budget = fig13_budget.averaged_budget(
        [10.0, 13.0], [first, second], ustar=0.4, dz=10.0
    )
    scale = run.F_CORIOLIS * 1.0e-3
    assert np.allclose(budget["tendency"], np.asarray([1.0, 2.0]) / scale)
    assert np.allclose(budget["production"], 1.0 / scale)
    assert np.allclose(
        budget["closure_residual"],
        budget["tendency"]
        - sum((budget[name] for name in fig13_budget.TERMS), np.zeros(2)),
    )


def test_paper_overlay_axis_registration_maps_exact_corners() -> None:
    axis = Axis(10, 20, 110, 220, -1.0, 1.0, 0.0, 0.5)
    points = _pixels(axis, np.asarray([-1.0, 1.0]), np.asarray([0.0, 0.5]))
    assert points == [(10, 220), (110, 20)]
    log_axis = Axis(10, 20, 110, 220, 1.0, 1000.0, 0.01, 10.0, True, True)
    log_points = _pixels(
        log_axis,
        np.asarray([1.0, 1000.0]),
        np.asarray([0.01, 10.0]),
    )
    assert log_points == [(10, 220), (110, 20)]


def test_figure7_registration_uses_the_printed_frame_and_range() -> None:
    assert FIGURE7_AXIS == Axis(324, 832, 736, 1243, 0.0, 8.0, 0.0, 0.35)
    points = _pixels(
        FIGURE7_AXIS,
        np.asarray([0.0, 8.0]),
        np.asarray([0.0, 0.35]),
    )
    assert points == [(324, 1243), (736, 832)]


def test_figure14_registration_uses_each_printed_panel() -> None:
    assert FIGURE14_AXES == (
        Axis(332, 138, 740, 551, 0.0, 10.0, 0.0, 0.35),
        Axis(327, 806, 735, 1217, 0.0, 15.0, 0.0, 0.35),
    )
    assert _pixels(
        FIGURE14_AXES[0],
        np.asarray([0.0, 10.0]),
        np.asarray([0.0, 0.35]),
    ) == [(332, 551), (740, 138)]
    assert _pixels(
        FIGURE14_AXES[1],
        np.asarray([0.0, 15.0]),
        np.asarray([0.0, 0.35]),
    ) == [(327, 1217), (735, 806)]


def test_figure13_registration_uses_all_four_budget_panels() -> None:
    assert FIGURE13_AXES == (
        Axis(190, 138, 483, 436, -40.0, 40.0, 0.0, 0.35),
        Axis(568, 138, 861, 436, -40.0, 40.0, 0.0, 0.35),
        Axis(190, 520, 483, 817, -40.0, 40.0, 0.0, 0.35),
        Axis(568, 520, 861, 817, -40.0, 40.0, 0.0, 0.35),
    )
    for axis in FIGURE13_AXES:
        assert _pixels(
            axis,
            np.asarray([-40.0, 40.0]),
            np.asarray([0.0, 0.35]),
        ) == [(axis.left, axis.bottom), (axis.right, axis.top)]


def test_figure15_registration_uses_all_four_log_frames() -> None:
    assert FIGURE15_AXES == (
        Axis(196, 521, 489, 818, 1.0, 1000.0, 0.01, 10.0, True, True),
        Axis(573, 522, 869, 818, 1.0, 1000.0, 0.01, 10.0, True, True),
        Axis(195, 902, 488, 1198, 1.0, 1000.0, 0.01, 10.0, True, True),
        Axis(572, 902, 867, 1199, 1.0, 1000.0, 0.01, 10.0, True, True),
    )
    expected_corners = (
        ((196, 818), (489, 521)),
        ((573, 818), (869, 522)),
        ((195, 1198), (488, 902)),
        ((572, 1199), (867, 902)),
    )
    for axis, corners in zip(FIGURE15_AXES, expected_corners, strict=True):
        assert _pixels(
            axis,
            np.asarray([1.0, 1000.0]),
            np.asarray([0.01, 10.0]),
        ) == list(corners)


def test_paper_overlay_sheet_registers_every_numbered_figure() -> None:
    assert tuple(spec.number for spec in FIGURES) == tuple(range(1, 20))
    assert {spec.number for spec in FIGURES if spec.comparison} == {
        2,
        4,
        5,
        6,
        7,
        8,
        13,
        14,
        15,
    }
    for spec in FIGURES:
        left, top, right, bottom = spec.crop
        assert 0 <= left < right <= 1008
        assert 0 <= top < bottom <= 1440
