from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.overlay_nieuwstadt1993 import FIGURES, overlay_results


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "cases" / "Nieuwstadt1993" / "reference" / "figures"


def test_registered_paper_figures_are_active_reference_data() -> None:
    assert tuple(sorted(int(path.stem[3:]) for path in REFERENCE.glob("fig*.png"))) == (
        tuple(range(1, 18))
    )
    for number in FIGURES:
        with Image.open(REFERENCE / f"fig{number}.png") as image:
            assert image.width >= 1000
            assert image.height >= 700
            image.verify()


def test_selected_overlays_use_uniform_diagnostic_outputs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    output = tmp_path / "overlays"
    results.mkdir()
    z = (np.arange(48, dtype=np.float64) + 0.5) * 50.0
    profile_columns = {
        "z_m": z,
        "total_scalar_flux": 0.06 * (1.0 - z / 1800.0),
        "sgs_scalar_flux": 0.01 * np.exp(-z / 400.0),
        "resolved_w_variance_m2_s2": 0.4 * np.sin(np.pi * z / 2400.0) ** 2,
        "sgs_tke_m2_s2": 0.04 * np.exp(-z / 1200.0),
        "pressure_variance_m4_s4": 0.1 * np.ones_like(z),
        "w_third_moment_m3_s3": 0.02 * np.sin(np.pi * z / 1800.0),
        "pressure_vertical_transport_m3_s3": 0.1 * np.sin(np.pi * z / 1800.0),
        "updraft_scalar_excess": 0.05 * (1.0 - z / 2400.0),
    }
    np.savetxt(
        results / "profiles.csv",
        np.column_stack(tuple(profile_columns.values())),
        delimiter=",",
        header=",".join(profile_columns),
        comments="",
    )
    radial_rows = []
    for height in (325.0, 975.0, 1575.0):
        for wavenumber in np.geomspace(1.1, 40.0, 12):
            radial_rows.append(
                (wavenumber, 0.1 / wavenumber, 0.05 / wavenumber, 0.01, height)
            )
    np.savetxt(
        results / "radial_spectra.csv",
        np.asarray(radial_rows),
        delimiter=",",
        header=(
            "wavenumber_reference_length,horizontal_energy,vertical_energy,"
            "scalar_energy,sample_height_m"
        ),
        comments="",
    )
    (results / "summary.json").write_text(
        json.dumps(
            {
                "case": "uniform_test_case",
                "physics": {
                    "scalar_surface_flux": 0.06,
                    "buoyancy_acceleration_per_scalar": 9.81 / 300.0,
                },
                "diagnostic_reference": {"length_m": 1600.0},
                "diagnostic_metrics": {
                    "boundary_layer_height_m": 1675.0,
                    "boundary_layer_height_ratio": 1675.0 / 1600.0,
                    "buoyancy_velocity_ratio": (1675.0 / 1600.0) ** (1.0 / 3.0),
                },
            }
        )
    )

    written = overlay_results(results, REFERENCE, output)

    assert {path.name for path in written} == {
        *(f"figure_{number:02d}_overlay.png" for number in FIGURES),
        "nieuwstadt1993_selected_overlays.png",
        "overlay_manifest.json",
    }
    with Image.open(output / "nieuwstadt1993_selected_overlays.png") as montage:
        assert montage.width > 1000
        assert montage.height > 1000
