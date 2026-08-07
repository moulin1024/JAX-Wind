from __future__ import annotations

import csv
from pathlib import Path

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


def test_default_entrypoint_is_the_complete_64_cubed_mgm_case() -> None:
    args = run._arguments([])
    case = load_case(run.CONFIG)
    assert (case.domain.nx, case.domain.ny, case.domain.nz) == (64, 64, 64)
    assert case.sgs.model == "mgm"
    assert case.time.steps == 360_000
    assert args.max_steps is None
    assert not args.allow_cpu


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
    assert "MGM mean, final 2 h" in text
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
