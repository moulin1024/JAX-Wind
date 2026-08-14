from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.gabls1_reference import ensemble_on_grid, load_period_sets
from tools.overlay_gabls1 import (
    _flux_ensemble,
    _select_reference_dir,
    overlay_results,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "cases" / "GABLS1" / "reference" / "official_12p5m"
REFERENCE_64 = ROOT / "cases" / "GABLS1" / "reference" / "official_6p25m"


def test_overlay_uses_uniform_profiles_and_raw_official_records(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "results"
    output_dir = tmp_path / "overlays"
    result_dir.mkdir()
    z = (np.arange(32, dtype=float) + 0.5) * 12.5
    profiles = ensemble_on_grid(load_period_sets(REFERENCE, "A", 9), z)
    fluxes = _flux_ensemble(REFERENCE, z)
    columns = {
        "z_m": z,
        "mean_u_m_s": profiles["u_mean"]["mean"],
        "mean_v_m_s": profiles["v_mean"]["mean"],
        "mean_scalar": profiles["theta_mean"]["mean"],
        "total_uw_m2_s2": fluxes["uw_total"]["mean"],
        "total_vw_m2_s2": fluxes["vw_total"]["mean"],
        "total_scalar_flux": fluxes["wtheta_total"]["mean"],
    }
    np.savetxt(
        result_dir / "profiles.csv",
        np.column_stack(tuple(columns.values())),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )

    written = overlay_results(result_dir, REFERENCE, output_dir)

    assert set(written) == {
        "figure",
        "complete_figure",
        "set_a_figure",
        "set_b_figure",
        "set_c_figure",
        "set_d_figure",
        "set_e_figure",
        "comparison",
        "complete_profile_comparison",
        "complete_time_comparison",
        "complete_manifest",
        "checkout",
    }
    assert all(path.is_file() for path in written.values())
    with Image.open(written["figure"]) as figure:
        assert figure.width >= 2000
        assert figure.height >= 1200
    checkout = json.loads(written["checkout"].read_text())
    assert checkout["participant_count"] == 7
    assert checkout["reference_resolution_m"] == 12.5
    assert checkout["theta_rmse_below_200m_k"] < 1.0e-12
    comparison = np.genfromtxt(written["comparison"], delimiter=",", names=True)
    assert comparison.shape == (32,)
    complete_profiles = np.genfromtxt(
        written["complete_profile_comparison"],
        delimiter=",",
        names=True,
    )
    assert complete_profiles.shape == (32,)
    complete_times = np.genfromtxt(
        written["complete_time_comparison"],
        delimiter=",",
        names=True,
    )
    assert complete_times.shape == (109,)
    manifest = json.loads(written["complete_manifest"].read_text())
    assert manifest["panels_total"] == 30
    assert manifest["participants_by_set"]["E"] == [
        "CSU",
        "LLNL",
        "MO",
        "NERSC",
        "UIB",
        "WU",
        "WVU",
    ]
    assert "boundary_layer_height" in manifest["reference_only"]
    with Image.open(written["complete_figure"]) as complete:
        assert complete.height > complete.width


def test_refined_results_select_official_6p25m_data(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    z = (np.arange(64, dtype=float) + 0.5) * 6.25
    np.savetxt(
        result_dir / "profiles.csv",
        np.column_stack((z, np.zeros_like(z))),
        delimiter=",",
        header="z_m,mean_u_m_s",
        comments="",
    )

    assert _select_reference_dir(result_dir) == REFERENCE_64
    assert len(load_period_sets(REFERENCE_64, "A", 9)) == 10
