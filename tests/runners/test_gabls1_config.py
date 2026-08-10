from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxwind.cli import main
from jaxwind.runners.gabls1 import ConfigError, load_case


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "benchmark" / "GABLS1" / "case.toml"
LASD_CONFIG = ROOT / "benchmark" / "GABLS1" / "case_lasd.toml"


def test_canonical_gabls1_case_is_complete() -> None:
    case = load_case(CONFIG)
    assert (case.domain.nx, case.domain.ny, case.domain.nz) == (32, 32, 32)
    assert case.time.steps == 129_600
    assert case.time.sample_start_step == 115_200
    assert case.flow.geostrophic_u_m_s == 8.0
    assert case.flow.coriolis_s == 1.39e-4
    assert case.thermal.surface_cooling_k_s == pytest.approx(-0.25 / 3600.0)
    assert case.sgs.model == "amd"


def test_gabls1_cli_dry_run_does_not_import_solver(capsys) -> None:
    assert main([str(CONFIG), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert 'runner = "gabls1"' in output
    assert 'model = "amd"' in output
    assert "duration_hours = 9.0" in output


def test_gabls1_rejects_non_amd_sgs(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(CONFIG.read_text().replace('model = "amd"', 'model = "mgm"'))
    with pytest.raises(ConfigError, match="'amd' or 'lasd'"):
        load_case(invalid)


def test_gabls1_lasd_case_selects_dynamic_closure() -> None:
    from jaxwind.runners.gabls1.runner import (
        LASD_SCALAR_STABILITY_BUOYANCY_COEFFICIENT,
    )

    case = load_case(LASD_CONFIG)
    assert case.sgs.model == "lasd"
    assert case.sgs.lasd_update_interval == 5
    assert LASD_SCALAR_STABILITY_BUOYANCY_COEFFICIENT == 0.0


def test_most_neutral_and_stable_flux_signs() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from jaxwind.runners.gabls1.most import MoninObukhovWallLaw

    law = MoninObukhovWallLaw(0.1, 0.1, 263.5)
    neutral = law.surface_fluxes(
        jnp.asarray(8.0),
        jnp.asarray(0.0),
        jnp.asarray(265.0),
        jnp.asarray(265.0),
        6.25,
    )
    expected = 0.4 * 8.0 / np.log(6.25 / 0.1)
    assert float(neutral.friction_velocity) == pytest.approx(expected, rel=1e-6)
    assert float(neutral.heat_flux) == pytest.approx(0.0, abs=1e-12)

    stable = law.surface_fluxes(
        jnp.asarray(8.0),
        jnp.asarray(0.0),
        jnp.asarray(266.0),
        jnp.asarray(265.0),
        6.25,
    )
    assert float(stable.heat_flux) < 0.0
    assert 0.0 < float(stable.friction_velocity) < float(neutral.friction_velocity)
    assert float(stable.obukhov_length) > 0.0
