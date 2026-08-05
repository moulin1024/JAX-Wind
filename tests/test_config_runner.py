from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxwind.config import load_case
from jaxwind.runner import build_simulation, run_case


ROOT = Path(__file__).parents[1]
CASES = {
    "andren": ROOT / "benchmark" / "Andren1994" / "case.toml",
    "nieuwstadt": ROOT / "benchmark" / "Nieuwstadt1993" / "case.toml",
    "gabls1": ROOT / "benchmark" / "GABLS1" / "case.toml",
}


def _quick(path: Path):
    return load_case(path).with_overrides(
        [
            "grid.shape=[8, 8, 8]",
            'numerics.dtype="float32"',
            "time.sample_start=0.0",
            'time.sample_basis="step"',
            "time.sample_interval=1",
            "time.log_interval=1",
            "time.checkpoint_interval=2",
            "time.maximum_step=0.25",
        ]
    )


def test_benchmarks_are_declarative_configs_without_runners() -> None:
    for path in CASES.values():
        assert path.is_file()
        assert not (path.parent / "run.py").exists()
        assert not (path.parent / "run_amd.py").exists()


def test_configs_encode_the_canonical_cases() -> None:
    andren = load_case(CASES["andren"])
    nieuwstadt = load_case(CASES["nieuwstadt"])
    gabls1 = load_case(CASES["gabls1"])

    assert andren.section("grid")["shape"] == [40, 40, 40]
    assert andren.section("sgs")["model"] == "multilevel_lasd"
    assert len(andren.section("initial")["velocity"]["u"]) == 40
    assert nieuwstadt.section("grid")["shape"] == [40, 40, 48]
    assert nieuwstadt.section("surface")["heat_flux"] == 0.06
    assert nieuwstadt.section("thermodynamics")["enabled"] is True
    assert gabls1.section("grid")["shape"] == [32, 32, 32]
    assert gabls1.section("time")["sample_start"] == 8.0 * 3600.0
    assert andren.section("surface")["thermal_boundary"] == "adiabatic"
    assert nieuwstadt.section("surface")["thermal_boundary"] == "flux"
    assert gabls1.section("surface")["thermal_boundary"] == "temperature"
    for config in (andren, nieuwstadt, gabls1):
        assert "scalar" not in config.data
        assert "boussinesq" not in config.data


def test_config_overrides_are_typed_and_must_name_existing_keys() -> None:
    config = load_case(CASES["andren"]).with_overrides(
        ["grid.shape=[16, 16, 16]", "sgs.coefficient=0.3"]
    )

    assert config.section("grid")["shape"] == [16, 16, 16]
    assert config.section("sgs")["coefficient"] == 0.3
    with pytest.raises(ValueError, match="existing key"):
        config.with_overrides(['numerics.projection_method="fpj2"'])


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_each_config_builds_and_advances_with_the_generic_runner(case: Path) -> None:
    simulation = build_simulation(_quick(case))
    state = simulation.initial_state()
    advanced = simulation.step(state, min(simulation.timestep(state), 0.25))

    assert advanced.step == 1
    assert np.all(np.isfinite(np.asarray(advanced.velocity.x)))
    if simulation.thermodynamics_enabled:
        assert np.all(np.isfinite(np.asarray(advanced.potential_temperature)))


def test_all_cases_use_the_same_solver_and_state_types() -> None:
    simulations = [build_simulation(_quick(path)) for path in CASES.values()]
    states = [simulation.initial_state() for simulation in simulations]

    assert {type(simulation.solver).__name__ for simulation in simulations} == {
        "ABLSolver"
    }
    assert {type(state).__name__ for state in states} == {"ABLState"}
    assert states[0].potential_temperature is None
    assert states[1].potential_temperature is not None
    assert states[2].potential_temperature is not None


def test_generic_runner_writes_and_restores_one_checkpoint(tmp_path: Path) -> None:
    config = _quick(CASES["andren"])
    first = run_case(config, output_dir=tmp_path / "first", max_steps=1)
    resumed = run_case(
        config,
        output_dir=tmp_path / "resumed",
        restart=tmp_path / "first" / "checkpoint.npz",
        max_steps=2,
    )

    assert first["final_step"] == 1
    assert resumed["final_step"] == 2
    assert (tmp_path / "resumed" / "profiles.csv").is_file()
    assert (tmp_path / "resumed" / "resolved_config.json").is_file()
