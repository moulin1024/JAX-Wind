from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from applications.windfarm_precursor.__main__ import main
from applications.windfarm_precursor.replay import _legacy_force_inflow_component
from applications.pressure_driven_lasd.config import load_case


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
            "--spanwise-shift-cells",
            "31",
        ]
    ) == 0

    resolved = json.loads(capsys.readouterr().out)
    assert resolved["recording"] == "outputs/precursor.h5"
    assert resolved["main_steps"] == 36000
    assert resolved["turbine"] == "dtu-10mw-adm"
    assert resolved["thrust_coefficient_prime"] == 4.0 / 3.0
    assert resolved["turbine_x_m"] == 1000.0
    assert resolved["spanwise_shift_cells"] == 31
    assert resolved["main_pressure_gradient"] == "off"


def test_replay_dry_run_accepts_explicit_main_pressure_gradient(capsys) -> None:
    assert main(["--dry-run", "--main-pressure-gradient", "on"]) == 0

    resolved = json.loads(capsys.readouterr().out)
    assert resolved["main_pressure_gradient"] == "on"


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
