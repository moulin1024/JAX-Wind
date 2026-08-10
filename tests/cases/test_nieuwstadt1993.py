from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import numpy as np

from applications.initial_conditions import REQUIRED_COLUMNS, load_initial_profile
from applications.abl.__main__ import main
from applications.abl.config import load_abl
from applications.abl.evaluate import ProfileStatistics, _bulk_metrics, _write_spectra
from jaxwind.physics import (
    ConservativeAdvection,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LinearBoussinesqBuoyancy,
    NoRotation,
)


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "cases" / "Nieuwstadt1993"
CONFIG = CASE_DIR / "config.toml"
CASE = load_abl(CONFIG)


def test_case_uses_the_uniform_abl_schema_without_regime_selection() -> None:
    andren_tables = {
        line[1:-1]
        for line in (ROOT / "cases" / "Andren1994" / "config.toml")
        .read_text()
        .splitlines()
        if line.startswith("[") and line.endswith("]")
    }
    tables = {
        line[1:-1]
        for line in CONFIG.read_text().splitlines()
        if line.startswith("[") and line.endswith("]")
    }
    lower = CONFIG.read_text().lower()

    assert tables == andren_tables
    for selector in ("runner", "workflow", "model", "integrator", "stability"):
        assert selector not in lower


def test_case_composes_the_same_generic_solver_products() -> None:
    assert isinstance(CASE.model.momentum.advection, ConservativeAdvection)
    assert isinstance(CASE.model.momentum.rotation, NoRotation)
    assert isinstance(CASE.model.momentum.sgs, LagrangianScaleDependentDynamic)
    assert isinstance(CASE.model.scalar_sgs, LagrangianScaleDependentScalarFlux)
    assert isinstance(CASE.model.buoyancy, LinearBoussinesqBuoyancy)
    assert CASE.model.buoyancy.acceleration_per_temperature > 0.0
    assert CASE.model.scalar_boundary.lower_flux > 0.0


def test_physical_grid_scales_and_sampling_match_the_reference_case() -> None:
    grid = CASE.physical_grid

    assert (grid.nx, grid.ny, grid.nz) == (40, 40, 48)
    assert (grid.lx, grid.ly, grid.lz) == (6400.0, 6400.0, 2400.0)
    assert CASE.dt_seconds == pytest.approx(1.25)
    assert CASE.steps == 9646
    assert CASE.output.sample_start_step == 8742
    assert CASE.output.sample_every_steps == 20
    assert CASE.mechanical_scales.velocity == pytest.approx(
        (9.81 * 0.06 * 1600.0 / 300.0) ** (1.0 / 3.0)
    )
    assert CASE.scalar_scales.magnitude == pytest.approx(
        0.06 / CASE.mechanical_scales.velocity
    )


def test_initial_state_is_data_in_the_shared_profile_format() -> None:
    table = load_initial_profile(CASE)

    assert table.dtype.names == REQUIRED_COLUMNS
    assert table.shape == (48,)
    assert table["scalar"][0] == 300.0
    assert table["scalar"][-1] == pytest.approx(303.0738)
    assert table["w_upper_rms_m_s"][0] > 0.0
    assert table["w_upper_rms_m_s"][-1] == 0.0
    assert math.isclose(table["u_m_s"][0], 0.0)


def test_reference_metrics_are_data_not_evaluator_branches() -> None:
    reference = json.loads(CASE.reference_results.read_text())

    assert set(reference["metrics"]) == {
        "boundary_layer_height_ratio",
        "buoyancy_velocity_ratio",
        "entrainment_flux_ratio",
    }


def test_dry_run_uses_the_common_abl_entry_point(capsys) -> None:
    assert main([str(CONFIG), "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["case"] == "nieuwstadt1993_lasd_40x40x48"
    assert result["physics"]["rotation"] == "NoRotation"
    assert result["physics"]["scalar_quantity"] == "potential_temperature"
    assert result["physics"]["scalar_surface_flux"] == pytest.approx(0.06)
    assert result["physics"]["buoyancy_acceleration_per_scalar"] == pytest.approx(
        9.81 / 300.0
    )


def test_generic_statistics_supply_bulk_metrics_and_multiple_spectrum_heights(
    tmp_path: Path,
) -> None:
    statistics = ProfileStatistics(CASE.physical_grid.nz)
    fields = np.zeros((CASE.physical_grid.nz, 2, 2), dtype=np.float64)
    diagnostics = {
        name: np.zeros(CASE.physical_grid.nz, dtype=np.float64)
        for name in ProfileStatistics.DIAGNOSTIC_NAMES
    }
    inversion_index = 33
    diagnostics["sgs_wc"][:] = 0.06
    diagnostics["sgs_wc"][inversion_index] = -0.006
    modes = np.broadcast_to(np.arange(4, dtype=np.float64), (3, 4))
    spectra = {
        "mode": modes,
        "u": np.ones((3, 4)),
        "v": np.ones((3, 4)),
        "w": np.ones((3, 4)),
        "scalar": np.ones((3, 4)),
        "height_m": np.broadcast_to(
            np.asarray(CASE.diagnostic_reference.spectrum_heights_m)[:, None],
            (3, 4),
        ),
        "radial_wavenumber_reference": np.broadcast_to(
            np.arange(4, dtype=np.float64)[None] + 0.5,
            (3, 4),
        ),
        "radial_horizontal": np.ones((3, 4)),
        "radial_w": np.ones((3, 4)),
        "radial_scalar": np.ones((3, 4)),
    }
    statistics.sample(
        fields,
        fields,
        fields,
        fields,
        ustar=1.0,
        diagnostics=diagnostics,
        spectra=spectra,
    )

    metrics = _bulk_metrics(CASE, statistics)
    path = tmp_path / "spectra.csv"
    _write_spectra(path, CASE, statistics)
    written = np.genfromtxt(path, delimiter=",", names=True)

    boundary_height = (inversion_index + 0.5) * CASE.physical_grid.dz
    assert metrics["boundary_layer_height_ratio"] == pytest.approx(
        boundary_height / CASE.diagnostic_reference.length_m
    )
    assert metrics["buoyancy_velocity_ratio"] == pytest.approx(
        (boundary_height / CASE.diagnostic_reference.length_m) ** (1.0 / 3.0)
    )
    assert metrics["entrainment_flux_ratio"] == pytest.approx(0.1)
    assert written.shape == (9,)
    assert set(written["sample_height_m"]) == set(
        CASE.diagnostic_reference.spectrum_heights_m
    )
