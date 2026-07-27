from __future__ import annotations

import json
import math

import numpy as np

from benchmark.Andren1994 import fig13_budget, run, run_lasd
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
    solver = run.solver_namespace(args)
    assert (solver.nx, solver.ny, solver.nz) == (40, 40, 40)
    assert (solver.lx, solver.ly, solver.lz) == (4000.0, 2000.0, 1500.0)
    assert math.isclose(solver.hours * 3600.0 * solver.coriolis, 10.0)
    assert math.isclose(solver.average_start_hours * 3600.0 * solver.coriolis, 7.0)
    assert math.isclose(solver.average_window_hours * 3600.0 * solver.coriolis, 3.0)
    assert solver.horizontal_coriolis == solver.coriolis
    assert solver.roughness == 0.1
    assert solver.smagorinsky == 0.17


def test_table_a1_and_reference_envelope_are_complete() -> None:
    table = run.paper_initial_profiles()
    assert len(table) == 40
    assert table["z_m"][0] == 18.75
    assert table["z_m"][-1] == 1481.25
    assert table["u_m_s"][12] == 10.71
    assert table["v_m_s"][3] == 2.84
    assert table["tke_m2_s2"][0] == 0.365
    assert table["tke_m2_s2"][21] == 0.0
    reference = json.loads(run.REFERENCE_RESULTS.read_text())
    ratios = tuple(reference["ustar_over_ug"].values())
    assert min(ratios) == 0.0402
    assert max(ratios) == 0.0448


def test_quick_mode_is_explicitly_noncanonical() -> None:
    args = run.parse_args(["--quick"])
    solver = run.solver_namespace(args)
    assert (solver.nx, solver.ny, solver.nz) == (8, 8, 8)
    assert solver.hours * 3600.0 / solver.dt == 8.0
    assert solver.average_start_hours == 0.0


def test_lasd_fifth_model_uses_safe_trajectory_cadence() -> None:
    args = run_lasd.parse_args([])
    assert (args.nx, args.ny, args.nz) == (40, 40, 40)
    assert args.dt == 0.8
    assert args.lasd_update_interval == 5
    assert math.isclose(args.hours * 3600.0 * run.F_CORIOLIS, 10.0)


def test_profile_variance_excludes_temporal_plane_mean_drift() -> None:
    samples = [
        {
            "scalar": np.asarray([mean]),
            "scalar2": np.asarray([mean**2 + 2.0]),
        }
        for mean in (0.0, 10.0, 20.0)
    ]
    averaged = run_lasd._average_profile_samples(samples)
    assert np.allclose(averaged["resolved_scalar_variance"], 2.0)
    contaminated = averaged["scalar2"] - averaged["scalar"] ** 2
    assert contaminated[0] > 60.0


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
