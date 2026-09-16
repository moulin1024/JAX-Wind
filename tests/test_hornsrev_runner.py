import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_hornsrev as runner


def test_main_requires_prepare():
    with pytest.raises(SystemExit):
        runner.parser().parse_args(["direction", "--output", "unused", "--wind-direction", "270"])


def test_configure_and_binding(tmp_path):
    prepare = tmp_path / "prepare"
    assert runner.main(["prepare", "--output", str(prepare), "--configure-only", "--smoke"]) == 0
    reference, digest = runner.prepare_binding(prepare, True, False)
    assert reference == prepare / "run/reference_smoke"
    assert digest is None
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.prepare_binding(prepare, True, True)
    with pytest.raises(ValueError, match="must match"):
        runner.prepare_binding(prepare, False, False)
    main = tmp_path / "main"
    assert runner.main(["direction", "--output", str(main), "--prepare-run", str(prepare),
                        "--wind-direction", "0", "--configure-only", "--smoke"]) == 0
    document = (main / "cases/wd000p000_smoke/workflow.toml").read_text()
    assert str(reference) in document
    assert runner.read_json(main / "runner.json")["prepare_run"] == str(prepare)


def test_windrose_dispatch(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "run_one", lambda args: seen.append(args) or "complete")
    args = runner.parser().parse_args(["windrose", "--output", str(tmp_path / "rose"),
                                     "--prepare-run", str(tmp_path / "prepare")])
    assert runner.run(args) == "complete"
    assert [a.wind_direction for a in seen] == list(range(0, 360, 30))
    assert len({a.output for a in seen}) == 12
    assert all(a.mode == "direction" and a.prepare_run == args.prepare_run for a in seen)
    seen.clear()
    args.sector_index = 9
    runner.run(args)
    assert [a.wind_direction for a in seen] == [270]
    args.sector_index = 12
    with pytest.raises(ValueError):
        runner.run(args)


def test_configure_preserves_executed_metadata(tmp_path):
    args = runner.parser().parse_args(["prepare", "--output", str(tmp_path), "--configure-only", "--smoke"])
    runner.run(args)
    path = tmp_path / "runner.json"
    manifest = runner.read_json(path)
    manifest["status"] = "complete"
    runner.save_json(path, manifest)
    args.resume = True
    with pytest.raises(ValueError, match="cannot overwrite"):
        runner.run(args)
    assert runner.read_json(path)["status"] == "complete"
