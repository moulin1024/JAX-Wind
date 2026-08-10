from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmark.PressureDrivenLASD.case import main
from benchmark.PressureDrivenLASD.config import load_case
from benchmark.PressureDrivenLASD.reporting import write_log_law_svg


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "benchmark" / "PressureDrivenLASD"
CONFIG = CASE_DIR / "config.toml"


def _write_profile(path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("z_m", "mean_u_m_s"))
        writer.writerow((10.0, 9.0))
        writer.writerow((100.0, 11.0))
        writer.writerow((900.0, 13.5))


def test_benchmark_owns_a_direct_solver_case() -> None:
    assert CONFIG.is_file()
    assert (CASE_DIR / "case.py").is_file()

    case = load_case(CONFIG)
    assert case.output.sample_start_hours == pytest.approx(
        0.8 * case.time.duration_hours
    )
    assert Path(case.output.directory) == Path(
        "outputs/pressure_driven_lasd_64x64x64_gpu"
    )


def test_case_resolves_its_configuration(capsys) -> None:
    assert main(["--config", str(CONFIG), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert 'case = "pressure_driven_lasd_64x64x64"' in output
    assert 'closure = "lagrangian-scale-dependent-dynamic"' in output


def test_log_law_plot_is_a_case_owned_dependency_free_svg(tmp_path: Path) -> None:
    profile = tmp_path / "profiles.csv"
    figure = tmp_path / "loglaw.svg"
    _write_profile(profile)
    write_log_law_svg(
        profile,
        figure,
        friction_velocity_m_s=0.4,
        roughness_length_m=0.001,
        von_karman=0.4,
        statistics_label="final 20%",
    )
    text = figure.read_text()
    assert text.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "LASD mean, final 20%" in text
