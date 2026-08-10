from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.overlay_andren1994 import overlay_results


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "cases" / "Andren1994" / "reference" / "figure_panels"


def test_all_published_figure_panels_are_reference_data() -> None:
    manifest = json.loads((REFERENCE / "manifest.json").read_text())

    assert tuple(manifest["panels"]) == tuple(
        f"{number:02d}" for number in range(1, 20)
    )
    for specification in manifest["panels"].values():
        with Image.open(REFERENCE / specification["file"]) as panel:
            left, top, right, bottom = specification["crop"]
            assert panel.size == (right - left, bottom - top)
            panel.verify()


def test_overlay_uses_active_profile_and_summary_outputs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    output = tmp_path / "overlays"
    results.mkdir()
    z = np.linspace(18.75, 318.75, 9)
    height = np.linspace(0.01, 0.09, z.size)
    ustar = 0.4
    scalar_flux = 1.0e-3
    columns = {
        "z_m": z,
        "z_f_over_ustar": height,
        "mean_u_m_s": np.log(z),
        "mean_v_m_s": 0.2 * np.log(z),
        "mean_scalar_kg_m3": -0.01 * np.log(z),
        "resolved_u_variance_m2_s2": ustar**2 * np.linspace(4.0, 0.2, z.size),
        "resolved_v_variance_m2_s2": ustar**2 * np.linspace(2.5, 0.1, z.size),
        "resolved_w_variance_m2_s2": ustar**2 * np.linspace(1.5, 0.1, z.size),
        "resolved_scalar_variance_kg2_m6": (scalar_flux / ustar) ** 2
        * np.linspace(4.0, 0.2, z.size),
        "total_uw_m2_s2": ustar**2 * np.linspace(-0.9, -0.1, z.size),
        "total_vw_m2_s2": ustar**2 * np.linspace(-0.5, 0.1, z.size),
        "total_wc_kg_m2_s": scalar_flux * np.linspace(0.9, 0.1, z.size),
        "sgs_wc_kg_m2_s": scalar_flux * np.linspace(0.5, 0.05, z.size),
        "resolved_tke_sgs_transfer_m2_s3": (
            1.0e-4 * ustar**2 * np.linspace(-100.0, -5.0, z.size)
        ),
        "momentum_diffusivity_m2_s": np.linspace(1.0, 8.0, z.size),
        "scalar_diffusivity_m2_s": np.linspace(2.0, 12.0, z.size),
    }
    np.savetxt(
        results / "profiles.csv",
        np.column_stack(tuple(columns.values())),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )
    (results / "summary.json").write_text(
        json.dumps(
            {
                "physics": {
                    "geostrophic_velocity_m_s": [10.0, 0.0],
                    "passive_scalar_surface_flux_kg_m2_s": scalar_flux,
                    "coriolis_vertical_s": 1.0e-4,
                },
                "runtime": {"ustar_m_s": ustar},
                "comparison": {"ustar_over_ug": ustar / 10.0},
            }
        )
    )
    nondimensional_time = np.linspace(0.1, 10.0, 9)
    history = {
        "time_hours": nondimensional_time / (1.0e-4 * 3600.0),
        "integrated_total_tke_m3_s2": (
            ustar**3 / 1.0e-4 * np.linspace(0.5, 1.0, 9)
        ),
        "momentum_stationarity_cu": np.linspace(0.5, 1.0, 9),
        "momentum_stationarity_cv": np.linspace(0.8, 1.2, 9),
    }
    np.savetxt(
        results / "history.csv",
        np.column_stack(tuple(history.values())),
        delimiter=",",
        header=",".join(history),
        comments="",
    )
    spectrum = {
        "k_ustar_over_f": np.geomspace(2.0, 200.0, 9),
        "kEu_over_ustar2": np.geomspace(2.0, 0.02, 9),
        "kEv_over_ustar2": np.geomspace(1.5, 0.02, 9),
        "kEw_over_ustar2": np.geomspace(1.0, 0.02, 9),
        "kEc_over_cstar2": np.geomspace(2.5, 0.02, 9),
    }
    np.savetxt(
        results / "spectra.csv",
        np.column_stack(tuple(spectrum.values())),
        delimiter=",",
        header=",".join(spectrum),
        comments="",
    )

    generated = overlay_results(results, REFERENCE, output)

    assert {path.name for path in generated} == {
        "figure_02_overlay.png",
        "figure_03_overlay.png",
        "figure_04_overlay.png",
        "figure_05_overlay.png",
        "figure_06_overlay.png",
        "figure_07_overlay.png",
        "figure_08_overlay.png",
        "figure_11_overlay.png",
        "figure_14_overlay.png",
        "figure_15_overlay.png",
        "andren1994_profile_overlays.png",
        "andren1994_complete_overlay.png",
        "overlay_manifest.json",
    }
    with Image.open(output / "figure_05_overlay.png") as overlay:
        pixels = np.asarray(overlay.convert("RGB"))
        assert np.any(np.all(pixels == (210, 24, 72), axis=-1))
