from __future__ import annotations

from pathlib import Path

import pytest

from applications.abl.config import load_abl
from applications.fv_abl.config import load_fv_abl
from applications.fv_abl.evaluate import resolved


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = (
    ROOT / "cases" / "Andren1994" / "config.toml",
    ROOT / "cases" / "Nieuwstadt1993" / "config.toml",
    ROOT / "cases" / "GABLS1" / "config.toml",
)


@pytest.mark.parametrize("path", CONFIGS)
def test_one_toml_feeds_both_solver_cores(path: Path) -> None:
    spectral = load_abl(path)
    finite_volume = load_fv_abl(path)

    assert finite_volume.physical == spectral
    assert finite_volume.options.pressure_backend == "gmg"
    assert finite_volume.options.time_integration == "ab2"
    assert finite_volume.options.momentum_closure == "amd"
    assert resolved(finite_volume)["case"] == spectral.name


def test_fv_options_are_strict_configuration(tmp_path: Path) -> None:
    text = CONFIGS[0].read_text(encoding="utf-8")
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        text.replace(
            'pressure_backend = "gmg"',
            'pressure_backend = "spectral"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pressure_backend must be one of"):
        load_fv_abl(invalid)


def test_pressure_backend_is_a_one_option_switch(tmp_path: Path) -> None:
    text = CONFIGS[0].read_text(encoding="utf-8")
    switched = tmp_path / "fft.toml"
    switched.write_text(
        text.replace(
            'pressure_backend = "gmg"',
            'pressure_backend = "fft"',
        ),
        encoding="utf-8",
    )

    gmg = load_fv_abl(CONFIGS[0])
    fft = load_fv_abl(switched)

    assert gmg.options.pressure_backend == "gmg"
    assert gmg.options.output_directory.name == "andren1994_fv_gmg_40x40x40"
    assert fft.options.pressure_backend == "fft"
    assert fft.options.output_directory.name == "andren1994_fv_fft_40x40x40"


def test_adaptive_rk_options_are_configuration_only(tmp_path: Path) -> None:
    text = CONFIGS[0].read_text(encoding="utf-8")
    configured = tmp_path / "adaptive.toml"
    configured.write_text(
        text.replace(
            'time_integration = "ab2"',
            'time_integration = "rk3"\ncfl_ceiling = 0.8',
        ),
        encoding="utf-8",
    )
    case = load_fv_abl(configured)
    assert case.options.time_integration == "rk3"
    assert case.options.cfl_ceiling == 0.8
    assert resolved(case)["dt_interpretation"] == "maximum"


def test_fv_core_contains_no_benchmark_dispatch() -> None:
    core = ROOT / "applications" / "fv_abl"
    source = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in core.glob("*.py")
    )

    assert "andren" not in source
    assert "andrén" not in source
    assert "nieuwstadt" not in source
    assert "gabls" not in source
