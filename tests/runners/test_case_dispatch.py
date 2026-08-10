from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jaxwind.cli import main
from jaxwind.runners import (
    ConfiguredCase,
    ExecutionConfig,
    RunnerDefinition,
    load_case,
    run_case,
)


def test_uniform_runner_forwards_configuration_owned_execution(
    tmp_path: Path,
) -> None:
    captured = {}

    def execute(configuration, **options):
        captured["configuration"] = configuration
        captured.update(options)
        options["output_dir"].mkdir(parents=True)
        return {"status": "complete"}

    configuration = SimpleNamespace(
        output=SimpleNamespace(directory=str(tmp_path / "configured-output")),
        resolved=lambda: {"case": {"runner": "fake"}},
    )
    configured = ConfiguredCase(
        config_path=Path("case/config.toml"),
        runner_name="fake",
        runner=RunnerDefinition(
            load_case=lambda _path: configuration,
            run_case=execute,
        ),
        configuration=configuration,
        execution=ExecutionConfig(
            restart_checkpoint=tmp_path / "checkpoint.npz",
            overwrite=True,
        ),
    )
    result = run_case(configured)

    assert result == {"status": "complete"}
    assert captured == {
        "configuration": configuration,
        "output_dir": tmp_path / "configured-output",
        "restart": tmp_path / "checkpoint.npz",
        "max_steps": None,
        "overwrite": True,
    }
    resolved = (tmp_path / "configured-output" / "resolved_config.toml")
    assert 'restart_checkpoint = "' in resolved.read_text()
    assert "overwrite = true" in resolved.read_text()


def test_uniform_runner_uses_safe_execution_defaults(tmp_path: Path) -> None:
    captured = {}

    def execute(_configuration, **options):
        captured.update(options)
        options["output_dir"].mkdir(parents=True)
        return {}

    configuration = SimpleNamespace(
        output=SimpleNamespace(directory=str(tmp_path / "configured-output")),
        resolved=lambda: {"case": {"runner": "fake"}},
    )
    configured = ConfiguredCase(
        config_path=Path("case/config.toml"),
        runner_name="fake",
        runner=RunnerDefinition(
            load_case=lambda _path: configuration,
            run_case=execute,
        ),
        configuration=configuration,
        execution=ExecutionConfig(
            restart_checkpoint=None,
            overwrite=False,
        ),
    )

    run_case(configured)

    assert captured["output_dir"] == tmp_path / "configured-output"
    assert captured["restart"] is None
    assert captured["max_steps"] is None
    assert captured["overwrite"] is False


@pytest.mark.parametrize(
    "arguments",
    (
        ("--max-steps", "1"),
        ("--output-dir", "/tmp/output"),
        ("--restart", "/tmp/checkpoint.npz"),
        ("--overwrite",),
        ("--config", "other.toml"),
    ),
)
def test_cli_rejects_ad_hoc_case_overrides(arguments, capsys) -> None:
    case = Path("runners/abl_warmup_neutral/config.toml")
    with pytest.raises(SystemExit, match="2"):
        main([str(case), *arguments])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_shared_run_api_rejects_overrides(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        run_case("runners/abl_warmup_neutral/config.toml", output_dir=tmp_path)


def test_shared_run_api_requires_explicit_toml_path() -> None:
    with pytest.raises(ValueError, match="explicit TOML"):
        run_case("runners/abl_warmup_neutral")


def test_cli_requires_explicit_toml_path(capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["runners/abl_warmup_neutral", "--dry-run"])
    assert "explicit TOML configuration file" in capsys.readouterr().err


def test_execution_restart_is_read_only_from_toml(tmp_path: Path) -> None:
    source = Path("runners/abl_warmup_neutral/config.toml").read_text()
    configured_path = tmp_path / "restart.toml"
    configured_path.write_text(
        source.replace(
            "overwrite = false",
            'restart_checkpoint = "outputs/source/checkpoint_latest.npz"\n'
            "overwrite = true",
        )
    )
    configured = load_case(configured_path)
    assert configured.execution.restart_checkpoint == Path(
        "outputs/source/checkpoint_latest.npz"
    )
    assert configured.execution.overwrite is True


def test_execution_table_rejects_step_caps(tmp_path: Path) -> None:
    source = Path("runners/abl_warmup_neutral/config.toml").read_text()
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(source.replace("overwrite = false", "max_steps = 1"))
    with pytest.raises(ValueError, match="unsupported.*max_steps"):
        load_case(invalid)
