from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from benchmark.Andren1994.overlay_paper_figures import (
    FIGURES,
    FIGURE13_AXES,
    FIGURE14_AXES,
    FIGURE15_AXES,
    FIGURE11_AXIS,
    FIGURE3_AXES,
    FIGURE7_AXIS,
    Axis,
    _momentum_stationarity,
    _pixels,
)
from jaxwind.runners import load_case
from jaxwind.cli import main
from jaxwind.runners.abl import andren1994
from jaxwind.runners.abl import andren1994_budget as fig13_budget
from jaxwind.runners.abl import andren1994_initial


ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = ROOT / "benchmark" / "Andren1994"
CONFIG = CASE_DIR / "config.toml"


def test_canonical_configuration_matches_paper_time_and_grid() -> None:
    configured = load_case(CONFIG)
    case = configured.configuration
    assert configured.runner_name == "abl"
    assert case.workflow == "warmup"
    assert (case.domain.nx, case.domain.ny, case.domain.nz) == (40, 40, 40)
    assert (case.domain.lx_m, case.domain.ly_m, case.domain.lz_m) == (
        4000.0,
        2000.0,
        1500.0,
    )
    assert math.isclose(
        case.time.duration_hours * 3600.0 * case.flow.coriolis_s,
        10.0,
    )
    assert math.isclose(
        case.time.sample_start_hours * 3600.0 * case.flow.coriolis_s,
        7.0,
    )
    assert case.benchmark.horizontal_coriolis_s == case.flow.coriolis_s
    assert case.flow.roughness_length_m == 0.1
    assert case.sgs.model == "lasd"


def test_table_a1_and_reference_envelope_are_complete() -> None:
    case = load_case(CONFIG).configuration
    table = andren1994_initial.load_initial_profiles(
        case.benchmark.initial_profiles
    )
    assert len(table) == 40
    assert table["z_m"][0] == 18.75
    assert table["z_m"][-1] == 1481.25
    assert table["u_m_s"][12] == 10.71
    assert table["v_m_s"][3] == 2.84
    assert table["tke_m2_s2"][0] == 0.365
    assert table["tke_m2_s2"][21] == 0.0
    reference = json.loads(case.benchmark.reference_results.read_text())
    ratios = tuple(reference["ustar_over_ug"].values())
    assert min(ratios) == 0.0402
    assert max(ratios) == 0.0448


def test_benchmark_case_is_pure_configuration() -> None:
    assert CONFIG.is_file()
    assert not (CASE_DIR / "run.py").exists()
    assert not (CASE_DIR / "run_lasd.py").exists()
    assert not (CASE_DIR / "fig13_budget.py").exists()


def test_uniform_cli_dry_run_resolves_andren_configuration(capsys) -> None:
    assert main([str(CONFIG), "--dry-run"]) == 0
    resolved = capsys.readouterr().out
    assert 'runner = "abl"' in resolved
    assert 'workflow = "warmup"' in resolved
    assert 'name = "andren1994"' in resolved
    assert 'model = "lasd"' in resolved


def test_lasd_model_uses_safe_trajectory_cadence() -> None:
    case = load_case(CONFIG).configuration
    assert case.time.dt_seconds == 0.8
    assert case.sgs.update_interval_steps == 5
    assert case.benchmark.thomas_chunk == 20
    assert case.estimated_lasd_trajectory_cfl == 0.8


def test_lasd_uses_three_halves_padding() -> None:
    assert andren1994.NONLINEAR_PADDING_RATIO == 1.5


def test_profile_variance_excludes_temporal_plane_mean_drift() -> None:
    samples = [
        {
            "scalar": np.asarray([mean]),
            "scalar2": np.asarray([mean**2 + 2.0]),
        }
        for mean in (0.0, 10.0, 20.0)
    ]
    averaged = andren1994._average_profile_samples(samples)
    assert np.allclose(averaged["resolved_scalar_variance"], 2.0)
    contaminated = averaged["scalar2"] - averaged["scalar"] ** 2
    assert contaminated[0] > 60.0


def test_statistics_restart_accepts_new_fig11_profile(tmp_path) -> None:
    path = tmp_path / "statistics_samples.npz"
    samples = [
        {"base": np.asarray([1.0, 2.0])},
        {
            "base": np.asarray([3.0, 4.0]),
            "resolved_tke_sgs_transfer": np.asarray([-5.0, -2.0]),
        },
    ]
    andren1994._write_statistics_state(path, [1.0, 2.0], samples)
    _, loaded = andren1994._load_statistics_state(path)
    assert np.all(np.isnan(loaded[0]["resolved_tke_sgs_transfer"]))
    averaged = andren1994._average_profile_samples(loaded)
    assert np.allclose(averaged["resolved_tke_sgs_transfer"], [-5.0, -2.0])


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
    scale = andren1994_initial.F_CORIOLIS * 1.0e-3
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


def test_figure3_stationarity_is_one_for_balanced_component_momentum() -> None:
    surface_uw = np.asarray([-0.1])
    surface_vw = np.asarray([-0.05])
    u = np.full((1, 2), 10.0 - 250.0)
    v = np.full((1, 2), 500.0)
    cu, cv = _momentum_stationarity(u, v, surface_uw, surface_vw, dz=1.0)
    assert np.allclose(cu, 1.0)
    assert np.allclose(cv, 1.0)
    assert FIGURE3_AXES == (
        Axis(332, 138, 741, 551, 0.0, 14.0, 0.0, 2.0),
        Axis(333, 795, 742, 1208, 0.0, 14.0, 0.0, 3.0),
    )


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


def test_figure11_registration_uses_signed_dissipation_range() -> None:
    assert FIGURE11_AXIS == Axis(350, 136, 757, 549, -150.0, 0.0, 0.0, 0.35)
    assert _pixels(
        FIGURE11_AXIS,
        np.asarray([-150.0, 0.0]),
        np.asarray([0.0, 0.35]),
    ) == [(350, 549), (757, 136)]


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
            11,
            13,
        14,
        15,
    }
    for spec in FIGURES:
        left, top, right, bottom = spec.crop
        assert 0 <= left < right <= 1008
        assert 0 <= top < bottom <= 1440
