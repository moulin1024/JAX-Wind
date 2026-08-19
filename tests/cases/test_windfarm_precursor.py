from __future__ import annotations

import json
import math
from pathlib import Path
import tomllib

import jax.numpy as jnp
import numpy as np
import pytest

from applications.windfarm_precursor.__main__ import main
from applications.windfarm_precursor.benchmark import (
    load_benchmark,
    main as benchmark_main,
)
from applications.windfarm_precursor.replay import _legacy_force_inflow_component
from applications.pressure_driven_lasd.config import load_case
from tools.compare_hub_height_gaussian_wake import (
    precursor_rotor_turbulence_intensity,
)


ROOT = Path(__file__).resolve().parents[2]


def test_offline_precursor_dry_run_resolves_the_real_checkpoint(capsys) -> None:
    assert main(["--dry-run", "--precursor-steps", "2"]) == 0

    resolved = json.loads(capsys.readouterr().out)
    assert resolved["case"] == "pressure_driven_lasd_64x64x64"
    assert resolved["precursor_steps"] == 2
    assert resolved["main_steps"] == 2
    assert resolved["section"] == "inflow"
    assert resolved["recording"] is None
    assert resolved["turbine"] == "none"
    assert resolved["restart"].endswith(
        "outputs/pressure_driven_lasd_64x64x64_gpu/checkpoint_final.npz"
    )


def test_replay_dry_run_resolves_dtu_10mw_adm(capsys) -> None:
    assert main(
        [
            "--dry-run",
            "--recording",
            "outputs/precursor.h5",
            "--main-steps",
            "36000",
            "--turbine",
            "dtu-10mw-adm",
            "--turbine-x-m",
            "1000",
        ]
    ) == 0

    resolved = json.loads(capsys.readouterr().out)
    assert resolved["recording"] == "outputs/precursor.h5"
    assert resolved["main_steps"] == 36000
    assert resolved["turbine"] == "dtu-10mw-adm"
    assert resolved["thrust_coefficient_prime"] == 4.0 / 3.0
    assert resolved["turbine_x_m"] == 1000.0
    assert resolved["compatibility"] == "strict-cuda-fortran"
    assert resolved["inflow_enforcement"] == "legacy-overwrite"
    assert resolved["inflow_start_plane"] == 10
    assert resolved["inflow_end_plane"] == 20
    assert resolved["inflow_update_steps"] == 10
    assert resolved["spanwise_cycle_updates"] == 4
    assert resolved["main_pressure_gradient"] == "off"


def test_removed_nonlegacy_inlet_options_are_rejected() -> None:
    import pytest

    with pytest.raises(SystemExit):
        main(["--dry-run", "--main-pressure-gradient", "on"])
    with pytest.raises(SystemExit):
        main(["--dry-run", "--fringe-start-fraction", "0.75"])


def test_legacy_inflow_blend_matches_fortran_k_plus_one_write_to_k() -> None:
    payload = np.arange(1 * 4 * 3 * 20, dtype=np.float32).reshape(1, 4, 3, 20)
    target = (1000.0 + np.arange(1 * 4 * 3 * 11, dtype=np.float32)).reshape(
        1, 4, 3, 11
    )
    blend = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    result = np.asarray(
        _legacy_force_inflow_component(
            jnp.asarray(payload),
            jnp.asarray(target),
            1,
            jnp.asarray(blend),
            jnp=jnp,
        )
    )

    source_block = np.roll(target, 1, axis=-2)
    expected = payload.copy()
    for k in range(3):
        base = payload[:, k + 1, :, 0]
        source = source_block[:, k + 1, :, 0]
        expected[:, k, :, :9] = base[..., None] + blend * (
            source - base
        )[..., None]
    expected[..., 9:20] = source_block
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.0)


def test_dtu_10mw_case_has_requested_domain_and_resolution() -> None:
    case = load_case(ROOT / "cases" / "DTU10MWPrecursor" / "config.toml")

    assert (case.domain.lx_m, case.domain.ly_m, case.domain.lz_m) == (
        4096.0,
        1024.0,
        1024.0,
    )
    assert (case.domain.nx, case.domain.ny, case.domain.nz) == (128, 64, 256)
    assert (case.domain.dx_m, case.domain.dy_m, case.domain.dz_m) == (
        32.0,
        16.0,
        4.0,
    )
    assert case.time.duration_hours == 10.0
    assert case.time.dt_seconds == 0.2
    assert case.time.steps == 180_000
    assert case.numerics.pressure_tridiag == "pcr"
    assert case.output.checkpoint_every_steps == 18_000
    assert case.sgs.closure == "static-smagorinsky"


def test_dtu_10mw_lasd_benchmark_matches_the_production_grid() -> None:
    case = load_case(
        ROOT
        / "cases"
        / "DTU10MWPrecursor"
        / "config_lasd_benchmark.toml"
    )

    assert (case.domain.nx, case.domain.ny, case.domain.nz) == (128, 64, 256)
    assert case.time.dt_seconds == 0.2
    assert case.numerics.dtype == "float32"
    assert case.numerics.pressure_tridiag == "pcr"
    assert case.sgs.closure == "lasd"
    assert case.sgs.update_interval_steps == 4
    assert not case.sgs.scalar_lasd_enabled
    assert case.sgs.reuse_rhs_momentum_context


def test_dtu_10mw_adbem_benchmark_resolves_all_three_stages() -> None:
    benchmark = load_benchmark(
        ROOT / "cases" / "DTU10MWPrecursor" / "benchmark_adbem.toml"
    )

    grid = (
        benchmark.case.domain.nx,
        benchmark.case.domain.ny,
        benchmark.case.domain.nz,
    )
    assert grid == (
        128,
        64,
        256,
    )
    assert benchmark.case.time.dt_seconds == 0.1
    assert benchmark.case.time.duration_hours == 10.0
    assert benchmark.case.time.steps == 360_000
    assert benchmark.precursor_steps == 36_000
    assert benchmark.main_steps == 36_000
    assert benchmark.turbine.model == "dtu-10mw-ad-bem"
    assert benchmark.turbine.x_m == 1000.0
    assert benchmark.turbine.blade_count == 3
    assert benchmark.turbine.radial_stations == 38
    assert benchmark.turbine.rotor_speed_rpm == 9.6
    assert benchmark.turbine.blade_pitch_degrees == 0.0
    assert benchmark.turbine.smearing_azimuthal_elements == 64
    assert benchmark.turbine.body_smoothing_width_m == 96.0
    assert benchmark.wake_model.maximum_deficit_rmse == 0.03


def test_dtu_10mw_adbem_benchmark_dry_run_is_self_contained(
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.delenv("JAXWIND_DTU10MW_FAST", raising=False)

    assert benchmark_main(["--dry-run"]) == 0

    resolved = json.loads(capsys.readouterr().out)
    assert resolved["schema"] == "jaxwind.windfarm-benchmark.v1"
    assert resolved["warmup"]["steps"] == 360_000
    assert resolved["precursor"]["steps"] == 36_000
    assert resolved["main"]["steps"] == 36_000
    assert resolved["main"]["compatibility"] == "strict-cuda-fortran"
    assert not resolved["main"]["pressure_gradient"]
    assert not resolved["main"]["fringe"]
    assert resolved["turbine"]["openfast_model"] == "${JAXWIND_DTU10MW_FAST}"
    command = resolved["commands"]["precursor_main"]
    assert command[command.index("--rotor-speed-rpm") + 1] == "9.6"
    assert command[command.index("--blade-pitch-degrees") + 1] == "0.0"


def test_precursor_ti_uses_the_configured_time_window(tmp_path: Path) -> None:
    import h5py

    recording = tmp_path / "precursor.h5"
    with h5py.File(recording, "w") as archive:
        archive.attrs["ly"] = 1.0
        archive.attrs["lz"] = 1.0
        archive.create_dataset("sections/name", data=np.asarray([b"inflow"]))
        archive.create_dataset("step", data=np.asarray([0, 10, 20]))
        archive.create_dataset("coordinates/y", data=np.asarray([0.25, 0.75]))
        archive.create_dataset(
            "coordinates/z_cell", data=np.asarray([0.25, 0.75])
        )
        velocity = np.zeros((3, 1, 3, 2, 2, 2), dtype=np.float32)
        velocity[0, 0, 0] = 100.0
        velocity[1, 0, 0] = 1.0
        velocity[2, 0, 0] = 3.0
        archive.create_dataset(
            "velocity",
            data=velocity,
            chunks=(1, 1, 3, 2, 2, 2),
        )

    intensity, details = precursor_rotor_turbulence_intensity(
        recording,
        section="inflow",
        dt_seconds=0.1,
        spinup_seconds=1.0,
        ly_m=4.0,
        lz_m=4.0,
        turbine_y_m=2.0,
        hub_height_m=2.0,
        rotor_diameter_m=5.0,
    )

    assert intensity == 0.5
    assert details["samples"] == 2
    assert details["time_window_seconds"] == [1.0, 2.0]


def test_hitsz_active_case_resolves_wind_tunnel_scale() -> None:
    path = ROOT / "cases" / "HITSZWindTunnel" / "config.toml"
    case = load_case(path)
    with path.open("rb") as stream:
        document = tomllib.load(stream)

    assert (case.domain.nx, case.domain.ny, case.domain.nz) == (256, 64, 128)
    assert (case.domain.lx_m, case.domain.ly_m, case.domain.lz_m) == (
        24.0,
        6.0,
        3.6,
    )
    assert (case.domain.dx_m, case.domain.dy_m, case.domain.dz_m) == (
        0.09375,
        0.09375,
        0.028125,
    )
    assert case.time.dt_seconds == 0.0025
    assert case.time.duration_hours == 0.025
    assert case.time.steps == 36_000
    assert case.sgs.update_interval_steps == 4
    assert case.sgs.lasd_filter_backend == "cufft"
    assert case.estimated_startup_cfl < case.numerics.cfl_abort
    experiment = document["experiment"]
    assert experiment["condition"] == "R9"
    assert experiment["rotor_diameter_m"] == 1.26
    assert experiment["hub_height_m"] == 0.876
    assert experiment["rotor_speed_rpm"] == 480.0
    assert experiment["measured_thrust_coefficient"] == 0.810
    assert case.flow.friction_velocity_m_s == 0.12294132680014663
    assert case.flow.roughness_length_m == 1.6100320416141182e-5
    assert experiment["fitted_hub_height_wind_speed_m_s"] == (
        3.3514673026030772
    )
    assert experiment["fitted_hub_height_tip_speed_ratio"] == (
        9.448773056383162
    )
    workflow = document["workflow"]
    assert workflow["coarse_warmup_seconds"] == 900.0
    assert workflow["fine_extension_seconds"] == 90.0
    assert workflow["coarse_grid"] == [128, 32, 64]
    assert workflow["fine_grid"] == [256, 64, 128]
    assert workflow["precursor_duration_seconds"] == 90.0
    assert workflow["main_duration_seconds"] == 90.0
    assert workflow["turbine_model"] == "hitsz-r9-ad-bem"
    assert not workflow["main_pressure_gradient"]
    assert not workflow["fringe"]


def test_hitsz_adbem_dry_run_retains_operating_point(capsys) -> None:
    config = ROOT / "cases" / "HITSZWindTunnel" / "config.toml"
    assert main(
        [
            str(config),
            "--dry-run",
            "--precursor-steps",
            "36000",
            "--main-steps",
            "36000",
            "--turbine",
            "hitsz-r9-ad-bem",
            "--turbine-x-m",
            "12.0",
            "--rotor-speed-rpm",
            "480.0",
            "--blade-pitch-degrees",
            "0.0",
        ]
    ) == 0

    resolved = json.loads(capsys.readouterr().out)
    assert resolved["case"] == "hitsz_r9_fine_extension_adbem_256x64x128"
    assert resolved["turbine"] == "hitsz-r9-ad-bem"
    assert resolved["rotor_speed_rpm"] == 480.0
    assert resolved["blade_pitch_degrees"] == 0.0


def test_hitsz_coarse_warmup_preserves_cfl_similarity() -> None:
    fine = load_case(ROOT / "cases" / "HITSZWindTunnel" / "config.toml")
    coarse = load_case(
        ROOT / "cases" / "HITSZWindTunnel" / "config_coarse.toml"
    )

    assert (coarse.domain.nx, coarse.domain.ny, coarse.domain.nz) == (
        128,
        32,
        64,
    )
    assert coarse.time.duration_hours == 0.25
    assert coarse.time.steps == 180_000
    assert coarse.time.dt_seconds == 2.0 * fine.time.dt_seconds
    assert coarse.estimated_startup_cfl == pytest.approx(
        fine.estimated_startup_cfl,
        rel=5.0e-4,
    )
    assert coarse.sgs.update_interval_steps == fine.sgs.update_interval_steps


def test_hitsz_inflow_fit_reproduces_versioned_log_law() -> None:
    profile = (
        ROOT / "cases" / "HITSZWindTunnel" / "reference" / "inflow_profile.csv"
    )
    import csv

    with profile.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    x = [math.log(float(row["height_mm"]) * 1.0e-3) for row in rows]
    velocity = [float(row["wind_speed_m_s"]) for row in rows]
    mean_x = sum(x) / len(x)
    mean_velocity = sum(velocity) / len(velocity)
    slope = sum(
        (xx - mean_x) * (uu - mean_velocity)
        for xx, uu in zip(x, velocity, strict=True)
    ) / sum((xx - mean_x) ** 2 for xx in x)
    intercept = mean_velocity - slope * mean_x

    assert 0.4 * slope == pytest.approx(0.12294132680014663)
    assert math.exp(-intercept / slope) == pytest.approx(
        1.6100320416141182e-5
    )
