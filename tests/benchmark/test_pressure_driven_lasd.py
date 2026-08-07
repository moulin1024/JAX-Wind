from __future__ import annotations

from benchmark.PressureDrivenLASD import run as lasd_run
from benchmark.PressureDrivenMGM import run as shared_run
from jaxwind.runners.pressure_driven_warmup import load_case


def test_lasd_changes_only_the_sgs_choice() -> None:
    mgm = load_case(shared_run.CONFIG, statistics_fraction=0.2)
    lasd = shared_run._select_sgs(mgm, "lasd")
    assert lasd.sgs.model == "lasd"
    assert lasd.name == mgm.name.replace("mgm", "lasd")
    assert lasd.domain is mgm.domain
    assert lasd.flow is mgm.flow
    assert lasd.wall is mgm.wall
    assert lasd.time is mgm.time
    assert lasd.numerics is mgm.numerics
    assert lasd.output is mgm.output


def test_lasd_entrypoint_uses_a_separate_output(
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_benchmark(argv, **options):
        captured["argv"] = argv
        captured.update(options)
        return 7

    monkeypatch.setattr(lasd_run, "run_benchmark", fake_run_benchmark)
    assert lasd_run.main(["--allow-cpu"]) == 7
    assert captured["argv"] == ["--allow-cpu"]
    assert captured["sgs_model"] == "lasd"
    assert captured["default_output"] == lasd_run.DEFAULT_OUTPUT
    assert captured["default_output"] != shared_run.DEFAULT_OUTPUT
