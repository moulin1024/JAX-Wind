from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_warmup_diagnostics import (  # noqa: E402
    cold_plume_centerline,
    configured_run_params,
    liquid_nitrogen_cooling_power,
    local_device_id,
    make_diagnostic_figure,
)


ROOT = Path(__file__).resolve().parents[3]


def _args(**overrides) -> Namespace:
    values = {
        "liquid_nitrogen_nozzle": True,
        "ln2_mass_flow_kg_s": 0.020,
        "ln2_injection_speed": 8.0,
        "ln2_x": 12.15,
        "ln2_y": 3.0,
        "ln2_z": 0.876,
        "ln2_sigma_x": 0.15,
        "ln2_sigma_r": 0.15,
        "ln2_specific_cooling_j_kg": 383_675.0,
        "ln2_cooling_power_w": None,
        "ln2_carrier_density": 1.225,
        "ln2_carrier_heat_capacity": 1005.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_default_ln2_nozzle_builds_ambient_buoyant_cooling_source() -> None:
    import jax.numpy as jnp

    from wireles_jax import Params

    configured = Params(
        nx=8,
        ny=4,
        nz=4,
        lx=24.0,
        ly=6.0,
        lz=3.6,
        z_i=1.0,
        momentum_wall_model="free_slip",
        thermo_enabled=False,
        moisture_enabled=False,
        sgs_model="smagorinsky",
        scalar_sgs_model="fixed_prandtl",
        dtype=jnp.float32,
    )
    args = _args()

    params, baseline = configured_run_params(configured, args, total_steps=12)

    assert params.nsteps == 12
    assert params.thermo_enabled is True
    assert params.horizontal_homogeneous is False
    assert params.buoyancy_reference == "ambient"
    assert params.cold_source_enabled is True
    assert params.cold_source_momentum_flux == pytest.approx(0.16)
    assert params.cold_source_cooling_power == pytest.approx(7673.5)
    assert params.cold_source_x == pytest.approx(12.15)
    assert baseline.thermo_enabled is False
    assert baseline.cold_source_enabled is False
    assert baseline.horizontal_homogeneous is True


def test_explicit_ln2_cooling_power_overrides_enthalpy_estimate() -> None:
    assert liquid_nitrogen_cooling_power(
        _args(ln2_cooling_power_w=12_000.0)
    ) == pytest.approx(12_000.0)


def test_local_device_defaults_to_mpi_local_rank(monkeypatch) -> None:
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "3")

    assert local_device_id(Namespace(local_device_id=None)) == 3
    assert local_device_id(Namespace(local_device_id=1)) == 1


def test_cold_plume_centerline_uses_temperature_deficit_weights() -> None:
    from types import SimpleNamespace

    params = SimpleNamespace(theta0=300.0, nz=3, dz=1.0, z_i=1.0)
    theta = np.full((1, 3, 3), 300.0)
    theta[0, 1] = (300.0, 299.0, 297.0)
    theta[0, 2] = (299.999, 299.999, 299.998)

    centroid, peak = cold_plume_centerline(theta, params)

    assert np.isnan(centroid[0, 0])
    assert centroid[0, 1] == pytest.approx(2.25)
    assert np.isnan(centroid[0, 2])
    np.testing.assert_allclose(peak, [[0.0, 3.0, 0.002]], rtol=1.0e-5)


def test_quiescent_jet_diagnostic_has_no_zero_ustar_failure(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    names = (
        "mean_u",
        "var_u",
        "var_v",
        "var_w",
        "resolved_uw_face",
        "sgs_txz_face",
        "mean_cs",
    )
    profiles = np.zeros((2, len(names), 4))
    profiles[:, 0, :] = (0.0, 0.1, 0.2, 0.1)
    profiles[:, 1:4, :] = 0.01
    params = SimpleNamespace(
        nz=4,
        dz=0.25,
        z_i=1.0,
        lz=1.0,
        pressure_ustar=0.0,
        vonk=0.4,
        zo=0.001,
        bl_height=1.0,
        sponge_enabled=False,
    )
    output = tmp_path / "pure_jet_diagnostics.png"

    make_diagnostic_figure(
        output,
        profiles,
        names,
        np.zeros(2),
        params,
        duration_seconds=1.0,
    )

    assert output.stat().st_size > 0


def test_8x4x2_ln2_experiment_config_resolves() -> None:
    import jax.numpy as jnp

    from run_single import RUN_DEFAULTS, load_config_file, params_from_settings

    config = (
        ROOT
        / "benchmark"
        / "LiquidNitrogenHubJet"
        / "configs"
        / "warmup_8x4x2_256x128x256.toml.example"
    )
    settings = dict(RUN_DEFAULTS)
    settings.update(load_config_file(config))
    configured = params_from_settings(settings, jnp)
    args = _args(
        ln2_x=1.0,
        ln2_y=2.0,
        ln2_z=1.0,
        ln2_sigma_x=0.15,
        ln2_sigma_r=0.15,
    )

    params, _ = configured_run_params(configured, args, total_steps=2500)

    assert (params.nx, params.ny, params.nz) == (256, 128, 256)
    assert (
        params.lx * params.z_i,
        params.ly * params.z_i,
        params.lz * params.z_i,
    ) == pytest.approx((8.0, 4.0, 2.0))
    assert params.dt_physical == pytest.approx(0.0004)
    assert params.nsteps == 2500
    assert params.uniform_u == pytest.approx(0.0)
    assert params.initial_velocity_noise == pytest.approx(0.0)
    assert params.driving_pressure_force == pytest.approx(0.0)
    assert params.cold_source_x == pytest.approx(1.0)
    assert params.cold_source_z == pytest.approx(1.0)
    assert params.cold_source_sigma_x == pytest.approx(0.15)
    assert params.fringe_enabled is False
    assert params.fringe_start_x == pytest.approx(7.5)
    assert params.fringe_target_u == pytest.approx(0.0)
