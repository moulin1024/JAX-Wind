from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmark.PressureDrivenMGM import run
from jaxwind.runners.pressure_driven_warmup import load_case


def _write_profile(path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("z_m", "mean_u_m_s"))
        writer.writerow((10.0, 9.0))
        writer.writerow((100.0, 11.0))
        writer.writerow((900.0, 13.5))


def test_default_entrypoint_uses_the_configured_mgm_case() -> None:
    args = run._arguments([])
    case = load_case(run.CONFIG, statistics_fraction=0.2)
    assert case.sgs.model == "mgm"
    assert case.output.sample_start_hours == pytest.approx(
        0.8 * case.time.duration_hours
    )
    assert args.max_steps is None
    assert not args.allow_cpu
    assert args.dt is None
    assert args.hours is None


def test_time_overrides_keep_statistics_in_the_final_twenty_percent() -> None:
    overridden = load_case(
        run.CONFIG,
        dt_seconds=0.2,
        duration_hours=2.0,
        statistics_fraction=0.2,
    )
    assert overridden.time.dt_seconds == 0.2
    assert overridden.time.duration_hours == 2.0
    assert overridden.time.steps == 36_000
    assert overridden.output.sample_start_hours == pytest.approx(1.6)
    assert overridden.sample_start_step == 28_800


def test_entrypoint_finds_src_checkout_without_an_editable_install() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util; "
                "from benchmark.PressureDrivenMGM.run import ROOT, SOURCE; "
                "spec = importlib.util.find_spec('jaxwind'); "
                "assert SOURCE == ROOT / 'src'; "
                "assert spec is not None and str(spec.origin).startswith(str(SOURCE))"
            ),
        ],
        cwd=run.ROOT,
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
    run.write_log_law_svg(
        profile,
        figure,
        friction_velocity_m_s=0.4,
        roughness_length_m=0.001,
        von_karman=0.4,
    )
    text = figure.read_text()
    assert text.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "MGM mean, final 20%" in text
    assert "U+ = ln(z/z0)/kappa" in text


def test_gpu_guard_rejects_an_accidental_cpu_long_run() -> None:
    class Device:
        platform = "cpu"

        def __str__(self) -> str:
            return "CpuDevice(id=0)"

    class Jax:
        @staticmethod
        def devices() -> list[Device]:
            return [Device()]

    with pytest.raises(RuntimeError, match="no JAX GPU"):
        run._require_gpu(Jax, allow_cpu=False)
    assert run._require_gpu(Jax, allow_cpu=True)


def test_pressure_solver_preflight_finds_the_submodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "external" / "bw1000_benchmark"
    package = checkout / "spectral_fd"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.delenv("JAXWIND_SPECTRAL_FD_SOURCE", raising=False)

    assert run._configure_pressure_solver() == checkout.resolve()
    assert os.environ["JAXWIND_SPECTRAL_FD_SOURCE"] == str(checkout.resolve())


def test_pressure_solver_preflight_has_an_actionable_missing_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.delenv("JAXWIND_SPECTRAL_FD_SOURCE", raising=False)
    with pytest.raises(RuntimeError, match="git submodule update"):
        run._configure_pressure_solver()
