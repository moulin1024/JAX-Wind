from __future__ import annotations

import numpy as np

from benchmark.Andren1994 import run
from benchmark.Andren1994.overlay_paper_figures import (
    _history_data,
    _profile_data,
)


def test_andren_runner_defaults_to_filter_free_amd() -> None:
    args = run.parse_args([])

    assert args.sgs == "amd"
    assert args.amd_coefficient == 0.212
    assert args.pressure_discretization == "kep4"
    assert args.end_ft == 0.1
    assert args.sample_start_ft == 0.05


def test_andren_runner_accepts_lasd_and_canonical_amd_controls() -> None:
    lasd = run.parse_args(["--sgs", "lasd"])
    amd = run.parse_args(
        [
            "--sgs",
            "amd",
            "--amd-coefficient",
            "0.3",
            "--end-ft",
            "10",
            "--sample-start-ft",
            "7",
        ]
    )

    assert lasd.sgs == "lasd"
    assert amd.sgs == "amd"
    assert amd.amd_coefficient == 0.3
    assert amd.end_ft == 10.0
    assert amd.sample_start_ft == 7.0


def test_paper_overlay_reads_current_amd_profile_schema(tmp_path) -> None:
    header = (
        "z_m,zf_over_ustar,mean_u_m_s,mean_v_m_s,"
        "var_u_m2_s2,var_v_m2_s2,var_w_m2_s2,"
        "resolved_uw_m2_s2,resolved_vw_m2_s2,"
        "sgs_uw_m2_s2,sgs_vw_m2_s2,"
        "total_uw_m2_s2,total_vw_m2_s2,phi_m\n"
    )
    rows = (
        "10,0.1,1,0,4,8,12,-1,-2,-0.5,-0.25,-1.5,-2.25,0.9\n"
        "20,0.2,2,0,8,12,16,-2,-3,-0.5,-0.25,-2.5,-3.25,1.1\n"
    )
    (tmp_path / "andren1994_profiles.csv").write_text(header + rows)

    profile = _profile_data(tmp_path, statistics_ustar=2.0)

    assert np.allclose(profile["height"], (0.1, 0.2))
    assert np.allclose(profile["u_variance"], (1.0, 2.0))
    assert np.allclose(profile["total_uw"], (-0.375, -0.625))
    assert np.allclose(profile["phi_m"], (0.9, 1.1))
    assert profile["phi_c"] is None
    assert _history_data(tmp_path, statistics_ustar=2.0) is None
