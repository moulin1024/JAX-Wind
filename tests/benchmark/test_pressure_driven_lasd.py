from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmark.PressureDrivenLASD import run as lasd_run
from jaxwind.runners.pressure_driven_warmup import load_case


def _write_profile(path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("z_m", "mean_u_m_s"))
        writer.writerow((10.0, 9.0))
        writer.writerow((100.0, 11.0))
        writer.writerow((900.0, 13.5))


def test_default_entrypoint_uses_the_lasd_case() -> None:
    args = lasd_run._arguments([])
    case = load_case(lasd_run.CONFIG, statistics_fraction=0.2)
    assert case.sgs.model == "lasd"
    assert case.output.sample_start_hours == pytest.approx(
        0.8 * case.time.duration_hours
    )
    assert args.max_steps is None
    assert not args.allow_cpu


def test_entrypoint_finds_src_checkout_without_an_editable_install() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util; "
                "from benchmark.PressureDrivenLASD.run import ROOT, SOURCE; "
                "spec = importlib.util.find_spec('jaxwind'); "
                "assert SOURCE == ROOT / 'src'; "
                "assert spec is not None and str(spec.origin).startswith(str(SOURCE))"
            ),
        ],
        cwd=lasd_run.ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""


def test_log_law_plot_is_dependency_free_svg(tmp_path: Path) -> None:
    profile = tmp_path / "profiles.csv"
    figure = tmp_path / "loglaw.svg"
    _write_profile(profile)
    lasd_run.write_log_law_svg(
        profile,
        figure,
        friction_velocity_m_s=0.4,
        roughness_length_m=0.001,
        von_karman=0.4,
    )
    text = figure.read_text()
    assert text.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "LASD mean, final 20%" in text


def test_lasd_entrypoint_uses_the_lasd_output(monkeypatch) -> None:
    captured = {}

    def fake_run_benchmark(argv, **options):
        captured["argv"] = argv
        captured.update(options)
        return 7

    monkeypatch.setattr(lasd_run, "run_benchmark", fake_run_benchmark)
    assert lasd_run.main(["--allow-cpu"]) == 7
    assert captured["argv"] == ["--allow-cpu"]
    assert captured["default_output"] == lasd_run.DEFAULT_OUTPUT
