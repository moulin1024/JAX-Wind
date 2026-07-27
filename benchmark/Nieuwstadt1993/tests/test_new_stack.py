from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmark" / "Nieuwstadt1993"


def test_default_benchmark_runner_selects_the_new_semantic_stack() -> None:
    orchestration = (BENCHMARK / "run.py").read_text()
    driver = (BENCHMARK / "run_new.py").read_text()
    assert "run_new.py" in orchestration
    assert "legacy/jax" not in orchestration
    assert "solve.py" not in orchestration
    assert "from wireles." in driver
    assert "from spectral_fd" in driver
    assert "wireles_jax" not in driver
    assert "legacy/jax" not in driver


def test_new_benchmark_enables_lasd_stable_stratification_correction() -> None:
    driver = (BENCHMARK / "run_new.py").read_text()
    assert "stability_buoyancy_coefficient=buoyancy_coefficient" in driver
    assert "stability_beta=30.0" in driver
    assert "stability_power=2.0" in driver
