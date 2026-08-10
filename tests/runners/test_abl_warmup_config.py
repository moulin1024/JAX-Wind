from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaxwind.cli import main
from jaxwind.runners import load_case
from jaxwind.runners.abl import ConfigError
from jaxwind.runners.abl import load_case as load_warmup_case
from jaxwind.runners.abl import run_case


ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "neutral": ROOT / "runners" / "abl_warmup_neutral",
    "stable": ROOT / "runners" / "abl_warmup_stable",
    "convective": ROOT / "runners" / "abl_warmup_convective",
}
EXPECTED_STABILITY = {
    "neutral": "neutral",
    "stable": "stable",
    "convective": "unstable",
}


@pytest.mark.parametrize("case_kind", tuple(CASES))
def test_pure_warmup_cases_use_one_runner(case_kind: str) -> None:
    case_dir = CASES[case_kind]
    assert (case_dir / "config.toml").is_file()
    assert not tuple(case_dir.glob("*.py"))
    assert "regime" not in (case_dir / "config.toml").read_text()

    configured = load_case(case_dir)
    case = configured.configuration
    assert configured.runner_name == "abl"
    assert case.runner == "abl"
    assert case.workflow == "warmup"
    assert case.stability == EXPECTED_STABILITY[case_kind]
    assert case.sgs.model == "lasd"
    assert case.resolved()["case"]["workflow"] == "warmup"
    assert case.resolved()["derived"]["stability"] == case.stability
    assert case.resolved()["output"]["manifest"] == "warmup_manifest.json"


@pytest.mark.parametrize("case_kind", tuple(CASES))
def test_uniform_cli_dry_runs_every_warmup(case_kind: str, capsys) -> None:
    assert main([str(CASES[case_kind] / "config.toml"), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert 'runner = "abl"' in output
    assert f'stability = "{EXPECTED_STABILITY[case_kind]}"' in output
    assert "regime" not in output
    assert 'workflow = "warmup"' in output


def test_stability_is_derived_from_explicit_thermal_physics() -> None:
    neutral = load_warmup_case(CASES["neutral"] / "config.toml")
    stable = load_warmup_case(CASES["stable"] / "config.toml")
    convective = load_warmup_case(CASES["convective"] / "config.toml")

    assert neutral.flow.momentum_forcing == "pressure_gradient"
    assert neutral.thermal.boundary_condition == "none"
    assert stable.flow.momentum_forcing == "geostrophic"
    assert stable.thermal.surface_cooling_k_s < 0.0
    assert convective.flow.momentum_forcing == "none"
    assert convective.thermal.surface_heat_flux_k_m_s > 0.0
    assert convective.estimated_lasd_trajectory_cfl < 1.0


def test_inconsistent_boundary_choices_are_rejected(tmp_path: Path) -> None:
    source = (CASES["convective"] / "config.toml").read_text()
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        source.replace('model = "neutral_log"', 'model = "monin_obukhov"')
    )
    with pytest.raises(ConfigError, match="thermal boundary 'fixed_surface_flux'"):
        load_warmup_case(invalid)


@pytest.mark.parametrize("field", ("regime", "stability"))
def test_explicit_stability_category_is_rejected(
    field: str,
    tmp_path: Path,
) -> None:
    source = (CASES["neutral"] / "config.toml").read_text()
    invalid = tmp_path / f"explicit-{field}.toml"
    invalid.write_text(
        source.replace("[case]", f'[case]\n{field} = "neutral"')
    )
    with pytest.raises(ConfigError, match="derived from thermal forcing"):
        load_warmup_case(invalid)


def test_unimplemented_workflow_is_rejected(tmp_path: Path) -> None:
    source = (CASES["neutral"] / "config.toml").read_text()
    invalid = tmp_path / "precursor.toml"
    invalid.write_text(source.replace('workflow = "warmup"', 'workflow = "precursor"'))
    with pytest.raises(ConfigError, match="only case.workflow"):
        load_warmup_case(invalid)


@pytest.mark.parametrize("case_kind", tuple(CASES))
def test_warmup_facade_writes_common_manifest(
    case_kind: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = load_warmup_case(CASES[case_kind] / "config.toml")

    def fake_backend(_case, **options):
        output = options["output_dir"]
        output.mkdir(parents=True)
        (output / "checkpoint_latest.npz").write_bytes(b"checkpoint")
        runtime = {"final_step": 2}
        if _case.thermal.boundary_condition == "fixed_surface_flux":
            runtime["surface_heat_flux_k_m_s"] = (
                _case.thermal.surface_heat_flux_k_m_s
            )
        elif _case.thermal.boundary_condition == "prescribed_surface_temperature":
            runtime["surface_heat_flux_k_m_s"] = -0.01
        return {"runtime": runtime}

    if case_kind == "neutral":
        import jaxwind.runners.pressure_driven_warmup.runner as backend
    else:
        import jaxwind.runners.gabls1.runner as backend
    monkeypatch.setattr(backend, "run_case", fake_backend)

    output = tmp_path / case_kind
    summary = run_case(
        case,
        output_dir=output,
        restart=None,
        max_steps=2,
        overwrite=False,
    )

    manifest = json.loads((output / "warmup_manifest.json").read_text())
    assert summary["workflow"]["stability"] == case.stability
    assert summary["schema"] == "jaxwind.abl-warmup.v1"
    assert set(summary) == {
        "case",
        "configuration",
        "physics",
        "runtime",
        "schema",
        "workflow",
    }
    assert summary["case"]["runner"] == "abl"
    assert "regime" not in summary["configuration"]["case"]
    assert summary["configuration"]["derived"]["stability"] == case.stability
    heat_flux = summary["runtime"].get("surface_heat_flux_k_m_s", 0.0)
    expected_buoyancy_flux = (
        case.thermal.gravity_m_s2
        * heat_flux
        / case.thermal.reference_temperature_k
    )
    assert summary["runtime"][
        "surface_buoyancy_flux_m2_s3"
    ] == pytest.approx(expected_buoyancy_flux)
    assert manifest["schema"] == "jaxwind.abl-warmup-manifest.v1"
    assert manifest["stability"] == case.stability
    assert "regime" not in manifest
    assert manifest["checkpoint"]["latest"] == "checkpoint_latest.npz"
    assert manifest["checkpoint"]["final"] is None
    assert manifest["checkpoint"]["layout"] == "z_slab_boussinesq.v1"
    assert manifest["checkpoint"]["statistics"] == "statistics_latest.npz"
    assert manifest["compatible_downstream_workflows"] == [
        "precursor",
        "wind_farm_main",
        "concurrent_precursor_main",
    ]
