from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxwind.config import load_case
from jaxwind.runner import (
    PROFILE_COLUMNS,
    _estimated_completion,
    build_simulation,
    run_case,
)


ROOT = Path(__file__).parents[1]
CASES = {
    "andren": ROOT / "benchmark" / "Andren1994" / "case.toml",
    "nieuwstadt": ROOT / "benchmark" / "Nieuwstadt1993" / "case.toml",
    "gabls1": ROOT / "benchmark" / "GABLS1" / "case.toml",
}
STRETCHED_GABLS1 = ROOT / "benchmark" / "GABLS1" / "case_64_stretched.toml"
STRETCHED_LOG_LAW = (
    ROOT / "benchmark" / "NeutralLogLawAMD" / "case_z_stretched.toml"
)


def _quick(path: Path):
    config = load_case(path)
    overrides = [
        "grid.shape=[8, 8, 8]",
        'numerics.dtype="float32"',
        "time.sample_start=0.0",
        'time.sample_basis="step"',
        "time.sample_interval=1",
        "time.history_interval=1",
        "time.log_interval=1",
        "time.checkpoint_interval=2",
        "time.maximum_step=0.25",
    ]
    return config.with_overrides(overrides)


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


def test_gabls1_64_case_uses_independent_analytic_stretching() -> None:
    config = load_case(STRETCHED_GABLS1)
    mapping = config.section("grid")["mapping"]

    assert config.section("grid")["shape"] == [64, 64, 64]
    assert mapping["x"] == {
        "function": "exponential",
        "focus": 0.5,
        "strength": 2.0,
    }
    assert mapping["y"] == mapping["x"]
    assert mapping["z"]["focus"] == 0.0
    assert "wall_matching_height" not in config.section("momentum")

    simulation = build_simulation(_quick(STRETCHED_GABLS1))
    assert simulation.grid.uniform_axes == (False, False, False)
    assert simulation.grid.x_faces[4] == 200.0
    assert simulation.grid.y_faces[4] == 200.0
    assert simulation.grid.z_widths[0] < simulation.grid.z_widths[-1]
    assert simulation.momentum.wall_cell_height == simulation.grid.z_widths[0]
    state = simulation.initial_state()
    advanced = simulation.step(state, min(simulation.timestep(state), 0.25))
    assert advanced.step == 1
    assert np.all(np.isfinite(np.asarray(advanced.velocity.x)))
    assert np.all(np.isfinite(np.asarray(advanced.potential_temperature)))


def test_pressure_driven_log_law_uses_ground_focused_z_mapping() -> None:
    config = load_case(STRETCHED_LOG_LAW)
    assert config.section("grid")["shape"] == [64, 64, 64]
    assert config.section("grid")["extent"] == [4000.0, 4000.0, 1000.0]

    simulation = build_simulation(_quick(STRETCHED_LOG_LAW))
    momentum = simulation.config.section("momentum")
    friction_velocity = float(momentum["friction_velocity"])
    height = float(simulation.config.section("grid")["extent"][2])

    assert simulation.grid.uniform_axes == (True, True, False)
    assert simulation.grid.z_widths[0] < simulation.grid.z_widths[-1]
    assert simulation.momentum.wall_cell_height == simulation.grid.z_widths[0]
    assert simulation.momentum.pressure_acceleration == pytest.approx(
        friction_velocity**2 / height
    )

    state = simulation.initial_state()
    cells = np.asarray(simulation.momentum.cell_centered_velocity(state.velocity))
    mean_u = np.mean(cells[..., 0], axis=(1, 2))
    expected = np.asarray(
        simulation.momentum.wall_law.cell_average_log_denominators(
            np.asarray(simulation.grid.z_faces[:-1], dtype=np.float32),
            np.asarray(simulation.grid.z_faces[1:], dtype=np.float32),
        )
    ) * (friction_velocity / 0.4)
    np.testing.assert_allclose(mean_u, expected, rtol=2.0e-6, atol=2.0e-6)

    advanced = simulation.step(state, min(simulation.timestep(state), 0.001))
    assert advanced.step == 1
    assert np.all(np.isfinite(np.asarray(advanced.velocity.x)))


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


def test_tabulated_initial_state_uses_one_projection() -> None:
    simulation = build_simulation(_quick(CASES["andren"]))
    projector = simulation.momentum.projector
    calls = {"velocity": 0, "velocity_pressure": 0}
    project_velocity = projector.project_velocity
    project_velocity_and_pressure = projector.project_velocity_and_pressure

    def counted_velocity(*args, **kwargs):
        calls["velocity"] += 1
        return project_velocity(*args, **kwargs)

    def counted_velocity_pressure(*args, **kwargs):
        calls["velocity_pressure"] += 1
        return project_velocity_and_pressure(*args, **kwargs)

    projector.project_velocity = counted_velocity
    projector.project_velocity_and_pressure = counted_velocity_pressure
    simulation.initial_state()

    assert calls == {"velocity": 0, "velocity_pressure": 1}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_runtime_observations_are_compact_device_arrays(case: Path) -> None:
    simulation = build_simulation(_quick(case))
    state = simulation.initial_state()

    prepared = simulation.prepare_step(state)
    rates = np.asarray(prepared.rates)
    diagnostics = np.asarray(simulation.diagnostic_metrics(state))
    profile = np.asarray(simulation.solver.profile(state))

    assert rates.shape == (3,)
    assert diagnostics.shape == (4,)
    assert np.all(np.isfinite(rates))
    assert np.all(np.isfinite(diagnostics[:3]))
    assert profile.shape == (simulation.grid.shape[0], len(PROFILE_COLUMNS))
    assert np.all(np.isfinite(profile[:, :7]))
    assert np.all(np.isfinite(profile[:, 9:11]))
    assert np.all(np.isfinite(profile[:, 12]))
    if state.potential_temperature is None:
        assert prepared.momentum is not None
        assert rates[2] == 0.0
        assert np.isnan(diagnostics[3])
        assert np.all(np.isnan(profile[:, 7:9]))
        assert np.all(np.isnan(profile[:, 11]))
    else:
        assert prepared.momentum is None
        assert np.isfinite(diagnostics[3])
        assert np.all(np.isfinite(profile[:, 7:]))


def test_prepared_neutral_imex_step_is_bitwise_identical() -> None:
    simulation = build_simulation(_quick(CASES["andren"]))
    state = simulation.initial_state()
    lasd = simulation.momentum.lasd_state
    assert lasd is not None
    lasd_step, interval_time = simulation.momentum.lasd_progress
    prepared = simulation.prepare_step(state)
    np.testing.assert_array_equal(
        np.asarray(prepared.rates),
        np.asarray(simulation.runtime_rates(state)),
    )
    timestep = simulation.timestep_from_metrics(np.asarray(prepared.rates))
    prepared_result = simulation.step(state, timestep, prepared)
    prepared_lasd = simulation.momentum.lasd_state
    assert prepared_lasd is not None

    simulation.momentum.restore_pressure(state.pressure)
    simulation.momentum.restore_lasd(
        lasd,
        accepted_step=lasd_step,
        interval_time=interval_time,
    )
    direct_result = simulation.step(state, timestep)
    direct_lasd = simulation.momentum.lasd_state
    assert direct_lasd is not None

    for prepared_value, direct_value in zip(
        prepared_result.velocity,
        direct_result.velocity,
        strict=True,
    ):
        np.testing.assert_array_equal(prepared_value, direct_value)
    np.testing.assert_array_equal(prepared_result.pressure, direct_result.pressure)
    for prepared_value, direct_value in zip(
        prepared_lasd,
        direct_lasd,
        strict=True,
    ):
        np.testing.assert_array_equal(prepared_value, direct_value)


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


def test_history_interval_is_independent_of_steps_and_logging(tmp_path: Path) -> None:
    config = _quick(CASES["andren"]).with_overrides(
        ["time.history_interval=2", "time.log_interval=99"]
    )
    run_case(config, output_dir=tmp_path, max_steps=3)

    history = np.loadtxt(tmp_path / "history.csv", delimiter=",", skiprows=1)
    assert history[:, 0].tolist() == [2.0, 3.0]


def test_estimated_completion_reports_clock_time_and_duration() -> None:
    estimate = _estimated_completion(
        elapsed=2.0,
        simulated=1.0,
        remaining=1800.5,
    )

    assert estimate.startswith("ETA=")
    assert estimate.endswith("remaining=1h00m01s")
    assert _estimated_completion(elapsed=1.0, simulated=1.0, remaining=0.0) == (
        "ETA=done"
    )
