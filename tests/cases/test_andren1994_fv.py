from __future__ import annotations

import json

from tools.run_fv_andren1994 import (
    _steps_to_next_sample,
    parse_arguments,
    run,
)


def test_fv_dry_run_uses_the_canonical_case_and_gmg(capsys) -> None:
    result = run(parse_arguments(["--dry-run"]))
    printed = json.loads(capsys.readouterr().out)

    assert result == printed
    assert result["cells"] == [40, 40, 40]
    assert result["lengths_m"] == [4000.0, 2000.0, 1500.0]
    assert result["time_integration"] == "AB2"
    assert result["pressure_backend"] == "gmg"
    assert result["momentum_closure"] == "AnisotropicMinimumDissipation"
    assert result["coriolis_vertical_s"] == 1.0e-4
    assert result["coriolis_horizontal_s"] == 1.0e-4
    assert result["scalar_surface_flux"] == 1.0e-3


def test_chunks_align_to_the_exact_statistics_schedule() -> None:
    assert _steps_to_next_sample(0, 87_500, 300) == 87_500
    assert _steps_to_next_sample(87_300, 87_500, 300) == 200
    assert _steps_to_next_sample(87_500, 87_500, 300) == 300
    assert _steps_to_next_sample(87_799, 87_500, 300) == 1


def test_fv_runner_rejects_invalid_controls() -> None:
    arguments = parse_arguments(["--dry-run", "--chunk", "0"])

    try:
        run(arguments)
    except ValueError as error:
        assert "--chunk must be positive" in str(error)
    else:
        raise AssertionError("a zero-sized FV execution chunk was accepted")
