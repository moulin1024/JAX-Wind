from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from applications.abl.config import load_abl
from tools.run_fv_andren1994 import PROFILE_NAMES
from tools.run_fv_nieuwstadt1993 import (
    DEFAULT_CONFIG,
    ConvectiveAccumulator,
    _profile_columns,
    _write_columns,
    _write_radial_spectra,
    parse_arguments,
    run,
)


def test_fv_dry_run_uses_buoyancy_and_gmg(capsys) -> None:
    result = run(parse_arguments(["--dry-run"]))
    printed = json.loads(capsys.readouterr().out)

    assert result == printed
    assert result["cells"] == [40, 40, 48]
    assert result["lengths_m"] == [6400.0, 6400.0, 2400.0]
    assert result["pressure_backend"] == "gmg"
    assert result["time_integration"] == "AB2"
    assert result["buoyancy_acceleration_per_scalar"] == 0.0327
    assert result["scalar_surface_flux"] == 0.06
    assert result["spectrum_heights_m"] == [320.0, 960.0, 1600.0]


def test_fv_diagnostic_outputs_cover_the_nieuwstadt_overlay(
    tmp_path: Path,
) -> None:
    case = load_abl(DEFAULT_CONFIG)
    grid = case.physical_grid
    accumulator = ConvectiveAccumulator(case)
    rng = np.random.default_rng(1993)
    fields = tuple(
        rng.normal(size=(grid.nz, grid.ny, grid.nx))
        for _ in range(4)
    )
    profiles = {
        name: np.full(grid.nz, 0.1, dtype=np.float64)
        for name in PROFILE_NAMES
    }
    profiles["resolved_wc"] = np.linspace(0.05, -0.01, grid.nz)
    profiles["sgs_wc"] = np.linspace(0.01, 0.0, grid.nz)
    accumulator.sample(fields, profiles, ustar=0.01, grid=grid)

    profile_path = tmp_path / "profiles.csv"
    radial_path = tmp_path / "radial_spectra.csv"
    _write_columns(profile_path, _profile_columns(case, accumulator))
    _write_radial_spectra(radial_path, case, accumulator)
    profile_names = set(
        np.genfromtxt(profile_path, delimiter=",", names=True).dtype.names
    )
    radial = np.genfromtxt(radial_path, delimiter=",", names=True)

    assert {
        "total_scalar_flux",
        "sgs_scalar_flux",
        "resolved_w_variance_m2_s2",
        "sgs_tke_m2_s2",
        "pressure_variance_m4_s4",
        "w_third_moment_m3_s3",
        "pressure_vertical_transport_m3_s3",
        "updraft_scalar_excess",
    }.issubset(profile_names)
    assert set(radial.dtype.names) == {
        "wavenumber_reference_length",
        "horizontal_energy",
        "vertical_energy",
        "scalar_energy",
        "sample_height_m",
    }
    assert set(radial["sample_height_m"]) == {325.0, 975.0, 1575.0}


def test_fv_runner_rejects_invalid_controls() -> None:
    arguments = parse_arguments(["--dry-run", "--chunk", "0"])

    try:
        run(arguments)
    except ValueError as error:
        assert "--chunk must be positive" in str(error)
    else:
        raise AssertionError("a zero-sized FV execution chunk was accepted")
