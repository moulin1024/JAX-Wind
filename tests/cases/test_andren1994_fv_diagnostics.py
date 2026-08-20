from __future__ import annotations

from pathlib import Path

import numpy as np

from applications.abl.config import load_abl
from tools.run_fv_andren1994 import (
    DEFAULT_CONFIG,
    PROFILE_NAMES,
    ProfileAccumulator,
    _write_profiles,
    _write_spectra,
)


def test_extended_fv_outputs_supply_every_registered_overlay_column(
    tmp_path: Path,
) -> None:
    case = load_abl(DEFAULT_CONFIG)
    grid = case.physical_grid
    accumulator = ProfileAccumulator(grid.nz, grid.nx)
    rng = np.random.default_rng(1994)
    fields = tuple(
        rng.normal(size=(grid.nz, grid.ny, grid.nx))
        for _ in range(4)
    )
    profiles = {
        name: np.full(grid.nz, 0.1, dtype=np.float64)
        for name in PROFILE_NAMES
    }
    profiles["u"] = np.linspace(4.0, 10.0, grid.nz)
    profiles["v"] = np.linspace(1.0, 0.0, grid.nz)
    accumulator.sample(
        fields,
        profiles,
        ustar=0.4,
        spectrum_level=10,
    )

    profile_path = tmp_path / "profiles.csv"
    spectrum_path = tmp_path / "spectra.csv"
    _write_profiles(profile_path, case, accumulator)
    _write_spectra(spectrum_path, case, accumulator)
    profile_names = set(
        np.genfromtxt(profile_path, delimiter=",", names=True).dtype.names
    )
    spectrum_names = set(
        np.genfromtxt(spectrum_path, delimiter=",", names=True).dtype.names
    )

    assert {
        "total_uw_m2_s2",
        "total_vw_m2_s2",
        "total_wc_kg_m2_s",
        "sgs_wc_kg_m2_s",
        "resolved_tke_sgs_transfer_m2_s3",
        "momentum_diffusivity_m2_s",
        "scalar_diffusivity_m2_s",
        "pressure_variance_m4_s4",
        "w_third_moment_m3_s3",
    }.issubset(profile_names)
    assert {
        "k_ustar_over_f",
        "kEu_over_ustar2",
        "kEv_over_ustar2",
        "kEw_over_ustar2",
        "kEc_over_cstar2",
        "sample_height_m",
    }.issubset(spectrum_names)
