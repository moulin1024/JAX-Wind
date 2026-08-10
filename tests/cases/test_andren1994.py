from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from applications.initial_conditions import load_initial_profile
from applications.abl.__main__ import main
from applications.abl.config import load_abl
from applications.abl.evaluate import ProfileStatistics
from jaxwind.physics import (
    ConservativeAdvection,
    CoriolisGeostrophic,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LinearBoussinesqBuoyancy,
    NeutralLogWall,
)


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "cases" / "Andren1994"
CONFIG = CASE_DIR / "config.toml"
CASE = load_abl(CONFIG)


def test_case_is_strict_toml_without_implementation_selectors() -> None:
    text = CONFIG.read_text(encoding="utf-8").lower()

    assert CONFIG.is_file()
    assert not (CASE_DIR / "case.py").exists()
    for selector in ("runner", "workflow", "model", "integrator", "stability"):
        assert selector not in text


def test_abl_composition_is_case_agnostic_and_performs_no_execution() -> None:
    recipe = ROOT / "applications" / "abl" / "config.py"
    text = recipe.read_text(encoding="utf-8").lower()

    assert "andren" not in text
    assert "andrén" not in text
    assert "ekman" not in text
    assert "import jax" not in text
    assert "build_solver" not in text
    assert "evaluate(" not in text


def test_toml_schema_rejects_unknown_keys(tmp_path: Path) -> None:
    invalid = tmp_path / "config.toml"
    invalid.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "[domain]",
            "[domain]\nrunner = \"abl\"",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[domain\] has unknown keys: runner"):
        load_abl(invalid)


def test_production_solver_has_no_andren_or_ekman_mode() -> None:
    source_root = ROOT / "src" / "jaxwind"
    mentions = {
        str(path.relative_to(source_root)): word
        for path in source_root.rglob("*.py")
        for word in ("andren", "andrén", "ekman")
        if word in path.read_text(encoding="utf-8").lower()
    }

    assert not mentions


def test_case_composes_generic_physics_objects() -> None:
    momentum = CASE.model.momentum

    assert isinstance(momentum.advection, ConservativeAdvection)
    assert isinstance(momentum.wall, NeutralLogWall)
    assert isinstance(momentum.sgs, LagrangianScaleDependentDynamic)
    assert isinstance(momentum.rotation, CoriolisGeostrophic)
    assert isinstance(CASE.model.scalar_sgs, LagrangianScaleDependentScalarFlux)
    assert isinstance(CASE.model.buoyancy, LinearBoussinesqBuoyancy)
    assert CASE.model.buoyancy.acceleration_per_temperature == 0.0
    assert momentum.sgs.update_interval == 5
    assert CASE.nonlinear_padding_ratio == 1.5


def test_canonical_grid_and_time_match_andren1994() -> None:
    grid = CASE.physical_grid
    rotation = CASE.model.momentum.rotation
    coriolis = CASE.mechanical_scales.from_execution_inverse_time(
        rotation.coriolis_parameter
    )

    assert (grid.nx, grid.ny, grid.nz) == (40, 40, 40)
    assert (grid.lx, grid.ly, grid.lz) == (4000.0, 2000.0, 1500.0)
    assert CASE.dt_seconds == pytest.approx(0.8)
    assert CASE.duration_seconds * coriolis == pytest.approx(10.0)
    assert (
        CASE.output.sample_start_step * CASE.dt_seconds * coriolis
        == pytest.approx(7.0)
    )
    assert CASE.mechanical_scales.from_execution_inverse_time(
        rotation.horizontal_coriolis_parameter
    ) == pytest.approx(coriolis)
    assert CASE.output.sample_every_steps == 300
    assert CASE.output.log_every_steps == 600
    assert CASE.output.checkpoint_every_steps == 6000


def test_table_a1_and_reference_envelope_are_active_case_inputs() -> None:
    table = load_initial_profile(CASE)
    reference = json.loads(
        (CASE_DIR / "reference" / "reference_results.json").read_text()
    )
    ratios = tuple(reference["ustar_over_ug"].values())

    assert table.shape == (40,)
    assert table["z_m"][0] == 18.75
    assert table["z_m"][-1] == 1481.25
    assert table["u_m_s"][12] == 10.71
    assert table["v_m_s"][3] == 2.84
    assert 1.5 * table["u_rms_m_s"][0] ** 2 == pytest.approx(0.365)
    assert min(ratios) == 0.0402
    assert max(ratios) == 0.0448


def test_dry_run_resolves_python_composition_without_jax(capsys) -> None:
    assert main(["--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["case"] == "andren1994_lasd_40x40x40"
    assert result["physics"]["rotation"] == "CoriolisGeostrophic"
    assert result["physics"]["momentum_sgs"] == (
        "LagrangianScaleDependentDynamic"
    )
    assert result["physics"]["buoyancy_acceleration_per_scalar"] == 0.0
    assert result["physics"]["coriolis_vertical_s"] == pytest.approx(1.0e-4)


def test_profile_statistics_do_not_fold_mean_drift_into_variance() -> None:
    statistics = ProfileStatistics(1)
    zeros = np.zeros((1, 2, 2), dtype=np.float64)
    for mean in (0.0, 10.0, 20.0):
        scalar = np.asarray([[[mean - 1.0, mean + 1.0], [mean - 1.0, mean + 1.0]]])
        statistics.sample(zeros, zeros, zeros, scalar, ustar=0.4)

    profiles = statistics.profiles()
    assert profiles["scalar"][0] == pytest.approx(10.0)
    assert profiles["scalar_variance"][0] == pytest.approx(1.0)
    assert math.isclose(statistics.mean_ustar, 0.4)


def test_extended_profile_diagnostics_are_restartable(tmp_path: Path) -> None:
    statistics = ProfileStatistics(2)
    fields = np.zeros((2, 2, 2), dtype=np.float64)
    diagnostics = {
        name: np.full(2, index + 1.0)
        for index, name in enumerate(ProfileStatistics.DIAGNOSTIC_NAMES)
    }
    spectra = {
        name: np.arange(3, dtype=np.float64) + index
        for index, name in enumerate(ProfileStatistics.SPECTRUM_NAMES)
    }
    statistics.sample(
        fields,
        fields,
        fields,
        fields,
        ustar=0.4,
        diagnostics=diagnostics,
        spectra=spectra,
    )
    path = tmp_path / "statistics_latest.npz"
    statistics.save(path)

    restarted = ProfileStatistics.load(path, 2)

    assert restarted.diagnostic_count == 1
    assert restarted.spectrum_count == 1
    for name, expected in diagnostics.items():
        assert np.array_equal(restarted.profiles()[name], expected)
    for name, expected in spectra.items():
        assert np.array_equal(restarted.spectra()[name], expected)
