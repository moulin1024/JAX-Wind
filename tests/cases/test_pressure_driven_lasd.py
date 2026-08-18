from __future__ import annotations

import csv
from pathlib import Path

import pytest

from applications.pressure_driven_lasd.config import load_case
from applications.pressure_driven_lasd.evaluate import main
from applications.pressure_driven_lasd.reporting import write_log_law_svg


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "cases" / "PressureDrivenLASD"
CONFIG = CASE_DIR / "config.toml"


def _write_profile(path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("z_m", "mean_u_m_s"))
        writer.writerow((10.0, 9.0))
        writer.writerow((100.0, 11.0))
        writer.writerow((900.0, 13.5))


def test_case_is_data_only_configuration() -> None:
    assert CONFIG.is_file()
    assert not tuple(CASE_DIR.rglob("*.py"))

    case = load_case(CONFIG)
    assert case.output.sample_start_hours == pytest.approx(
        0.8 * case.time.duration_hours
    )
    assert Path(case.output.directory) == Path(
        "outputs/pressure_driven_lasd_64x64x64_gpu"
    )
    assert case.numerics.pressure_tridiag == "pcr"
    assert case.numerics.pressure_thomas_chunk == 1
    assert case.time.dt_seconds == pytest.approx(0.6)
    assert case.sgs.update_interval_steps == 8
    assert case.sgs.lasd_filter_backend == "cufft"
    assert case.sgs.reuse_rhs_momentum_context
    assert not case.sgs.scalar_lasd_enabled


def test_case_resolves_its_configuration(capsys) -> None:
    assert main([str(CONFIG), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert 'case = "pressure_driven_lasd_64x64x64"' in output
    assert 'closure = "lagrangian-scale-dependent-dynamic"' in output
    assert 'momentum_advection = "legacy-rotational"' in output
    assert 'pressure_tridiag = "pcr"' in output
    assert 'nonlinear_scheme = "legacy-fortran-pre-rhs-filtering"' in output
    assert "dt_seconds = 0.6" in output
    assert "update_interval_steps = 8" in output


def test_case_can_select_cufft_lasd_filtering(tmp_path: Path) -> None:
    assert load_case(CONFIG).sgs.lasd_filter_backend == "cufft"


@pytest.mark.parametrize(
    ("table", "selector"),
    (
        ("[flow]", 'advection = "conservative"'),
        ("[numerics]", 'nonlinear_dealiasing = "three_halves"'),
        ("[numerics]", "nonlinear_padding_ratio = 1.5"),
    ),
)
def test_removed_nonlinear_selectors_are_rejected(
    tmp_path: Path,
    table: str,
    selector: str,
) -> None:
    selected = tmp_path / "config.toml"
    selected.write_text(
        CONFIG.read_text(encoding="utf-8").replace(table, f"{table}\n{selector}"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="removed|was removed"):
        load_case(selected)


def test_cufft_lasd_filtering_requires_float32(tmp_path: Path) -> None:
    selected = tmp_path / "config.toml"
    selected.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            'dtype = "float32"', 'dtype = "float64"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cuFFT LASD backend"):
        load_case(selected)


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
