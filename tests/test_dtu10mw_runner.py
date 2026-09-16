"""DTU10MW stage schedules, provenance, and small CPU ALM plumbing."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_dtu10mw as runner
from create_dtu10mw_case import FIXED_DT, TIP_SPEED_M_S, STEPS_PER_HOUR
from jaxwind.config.document import load_case
from jaxwind.workflows.engine import check_workflow


def test_production_prepare_main_schedules(tmp_path, monkeypatch):
    monkeypatch.delenv("JAXWIND_DTU10MW_FAST", raising=False)
    prepare, main = tmp_path / "prepare", tmp_path / "main"
    assert runner.main(["prepare", "--output", str(prepare), "--configure-only"]) == 0
    assert runner.main(["main", "--output", str(main), "--prepare-run", str(prepare), "--configure-only"]) == 0
    reference = check_workflow(prepare / "cases/workflow.toml")
    replay = check_workflow(main / "cases/workflow.toml")
    assert reference["order"] == ["warmup", "precursor"]
    assert replay["order"] == ["main"]
    warm, precursor, farm = [p["stages"][name]["time"] for p, name in
        [(reference, "warmup"), (reference, "precursor"), (replay, "main")]]
    assert warm["steps"] * warm["dt_seconds"] == pytest.approx(36000.)
    assert "cfl" not in warm
    assert warm["dt_seconds"] == FIXED_DT
    assert TIP_SPEED_M_S * FIXED_DT / 2. <= .5
    assert warm["steps"] == 10 * STEPS_PER_HOUR
    for settings in (precursor, farm):
        assert settings["steps"] * settings["dt_seconds"] == pytest.approx(3600.)
        assert "cfl" not in settings
    assert precursor["dt_seconds"] == farm["dt_seconds"] == FIXED_DT
    assert all(t["checkpoint_every_seconds"] == pytest.approx(3600.) for t in (warm, precursor, farm))
    case = load_case(main / "cases/case.toml")
    assert case.document["mesh"] == {"cells": [512, 256, 512], "lengths_m": [4096., 2048., 1024.]}
    assert case.document["physics"]["turbine"]["model"] == "openfast-alm"
    assert "turbine" not in load_case(prepare / "cases/case.toml").document["physics"]
    assert Path(replay["stages"]["main"]["inputs"]["checkpoint"]) == prepare / "run/reference/warmup/checkpoint.npz"
    assert runner.main(["prepare", "--output", str(prepare), "--configure-only"]) == 0
    with pytest.raises(ValueError, match="not complete"):
        runner.prepare_binding(prepare, False, True)
    with pytest.raises(ValueError, match="must match"):
        runner.prepare_binding(prepare, True, False)


def test_identity_and_existing_execution_protection(tmp_path):
    args = runner.parser().parse_args(["prepare", "--output", str(tmp_path), "--configure-only"])
    runner.run(args)
    path = tmp_path / "runner.json"
    data = runner.read_json(path)
    data["status"] = "paused"
    runner.save_json(path, data)
    with pytest.raises(ValueError, match="configure-only"):
        runner.run(args)
    args.configure_only = False
    with pytest.raises(ValueError, match="requires --resume"):
        runner.run(args)
    args.max_steps = 0
    with pytest.raises(ValueError, match="positive"):
        runner.run(args)
    assert runner.read_json(path)["status"] == "paused"


def test_main_output_separate_from_prepare(tmp_path):
    args = runner.parser().parse_args(["main", "--output", str(tmp_path),
        "--prepare-run", str(tmp_path / "prepare"), "--configure-only"])
    with pytest.raises(ValueError, match="separate"):
        runner.run(args)


def test_small_alm_prepare_main_resume(tmp_path, monkeypatch):
    # Use the explicitly identified NREL fixture solely to exercise the generic
    # OpenFAST ALM adapter. This does not validate a DTU10MW turbine deck.
    monkeypatch.setenv("JAXWIND_DTU10MW_FAST", str(ROOT / "tests/fixtures/openfast/nrel5mw/NREL5MW_Rigid_Smoke.fst"))
    prepare, main = tmp_path / "prepare", tmp_path / "main"
    prepare_args = ["prepare", "--output", str(prepare), "--backend", "cpu", "--smoke"]
    assert runner.main(prepare_args + ["--max-steps", "2"]) == 3
    assert runner.read_json(prepare / "runner.json")["status"] == "paused"
    assert runner.main(prepare_args + ["--resume"]) == 0
    main_args = ["main", "--output", str(main), "--prepare-run", str(prepare), "--backend", "cpu", "--smoke"]
    assert runner.main(main_args + ["--max-steps", "2"]) == 3
    assert runner.main(main_args + ["--resume"]) == 0
    summary = runner.read_json(main / "run/main/summary.json")
    assert summary["step"] == 4
    assert summary["time_seconds"] == pytest.approx(4 * FIXED_DT, abs=1.e-6)
    assert summary["final_cfl"] < 1.
    meta = runner.read_json(prepare / "run/reference/precursor/inflow/metadata.json")
    assert meta["samples"] == 4
    assert meta["duration_seconds"] == pytest.approx(4 * FIXED_DT)
    # Reject changed upstream provenance on main resume.
    meta["test_changed"] = True
    runner.save_json(prepare / "run/reference/precursor/inflow/metadata.json", meta)
    with pytest.raises(SystemExit, match="recording changed"):
        runner.main(main_args + ["--resume"])
