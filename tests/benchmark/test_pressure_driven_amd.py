from __future__ import annotations

from benchmark.PressureDrivenAMD import run as amd_run
from benchmark.PressureDrivenMGM import run as shared_run
from jaxwind.runners.pressure_driven_warmup import load_case


def test_amd_changes_only_the_sgs_choice() -> None:
    mgm = load_case(shared_run.CONFIG, statistics_fraction=0.2)
    amd = shared_run._select_sgs(mgm, "amd")
    assert amd.sgs.model == "amd"
    assert amd.name == mgm.name.replace("mgm", "amd")
    assert amd.domain is mgm.domain
    assert amd.flow is mgm.flow
    assert amd.wall is mgm.wall
    assert amd.time is mgm.time
    assert amd.numerics is mgm.numerics
    assert amd.output is mgm.output
    assert amd.resolved()["sgs"] == {"model": "amd"}


def test_amd_entrypoint_uses_a_separate_output(monkeypatch) -> None:
    captured = {}

    def fake_run_benchmark(argv, **options):
        captured["argv"] = argv
        captured.update(options)
        return 7

    monkeypatch.setattr(amd_run, "run_benchmark", fake_run_benchmark)
    assert amd_run.main(["--allow-cpu"]) == 7
    assert captured["argv"] == ["--allow-cpu"]
    assert captured["sgs_model"] == "amd"
    assert captured["default_output"] == amd_run.DEFAULT_OUTPUT
    assert captured["default_output"] != shared_run.DEFAULT_OUTPUT
