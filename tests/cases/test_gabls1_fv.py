from __future__ import annotations

import json

from tools.run_fv_gabls1 import parse_arguments, run


def test_fv_dry_run_uses_coupled_cooling_surface_and_gmg(capsys) -> None:
    result = run(parse_arguments(["--dry-run"]))
    printed = json.loads(capsys.readouterr().out)

    assert result == printed
    assert result["cells"] == [32, 32, 32]
    assert result["lengths_m"] == [400.0, 400.0, 400.0]
    assert result["pressure_backend"] == "gmg"
    assert result["time_integration"] == "AB2"
    assert result["coriolis_vertical_s"] == 1.39e-4
    assert result["evolved_geostrophic_velocity_m_s"] == [0.0, 0.0]
    assert result["velocity_offset_m_s"] == [8.0, 0.0]
    assert result["surface_scalar_initial"] == 265.0
    assert result["surface_scalar_rate_per_second"] == -0.25 / 3600.0
    assert result["buoyancy_acceleration_per_scalar"] == 9.81 / 263.5


def test_fv_runner_rejects_invalid_controls() -> None:
    arguments = parse_arguments(["--dry-run", "--chunk", "0"])

    try:
        run(arguments)
    except ValueError as error:
        assert "--chunk must be positive" in str(error)
    else:
        raise AssertionError("a zero-sized FV execution chunk was accepted")
