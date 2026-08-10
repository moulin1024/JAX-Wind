from __future__ import annotations

import csv
from pathlib import Path

import pytest

from jaxwind.cli import main
from jaxwind.runners import load_case
from jaxwind.runners.pressure_driven_warmup.log_law import write_log_law_svg


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


def test_benchmark_is_a_pure_declarative_case() -> None:
    assert CONFIG.is_file()
    assert not tuple(CASE_DIR.glob("*.py"))

    configured = load_case(CASE_DIR)
    case = configured.configuration
    assert configured.runner_name == "pressure_driven_warmup"
    assert configured.config_path == CONFIG
    assert case.sgs.model == "lasd"
    assert case.output.sample_start_hours == pytest.approx(
        0.8 * case.time.duration_hours
    )
    assert configured.output_directory == Path(
        "outputs/pressure_driven_lasd_64x64x64_gpu"
    )


def test_uniform_cli_resolves_benchmark_configuration(capsys) -> None:
    assert main([str(CONFIG), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert 'runner = "pressure_driven_warmup"' in output
    assert 'model = "lasd"' in output
    assert 'case = "pressure_driven_lasd_64x64x64"' in output


def test_log_law_plot_is_a_runner_owned_dependency_free_svg(tmp_path: Path) -> None:
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
