"""Uniform dispatch for package-owned declarative case runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 development fallback
    import tomli as tomllib

from ._toml import dumps as toml_dumps


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Configuration-owned launch behavior shared by every runner."""

    restart_checkpoint: Path | None
    overwrite: bool

    def resolved(self) -> dict[str, Any]:
        return {
            "restart_checkpoint": (
                None
                if self.restart_checkpoint is None
                else str(self.restart_checkpoint)
            ),
            "overwrite": self.overwrite,
        }


@dataclass(frozen=True, slots=True)
class RunnerDefinition:
    """Lazy CLI-facing hooks for one built-in runner kind."""

    load_case: Callable[[str | Path], Any]
    run_case: Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ConfiguredCase:
    """A validated case paired with the runner selected by its TOML file."""

    config_path: Path
    runner_name: str
    runner: RunnerDefinition
    configuration: Any
    execution: ExecutionConfig

    def resolved_toml(self) -> str:
        resolved = self.configuration.resolved()
        resolved["execution"] = self.execution.resolved()
        return toml_dumps(resolved)

    @property
    def output_directory(self) -> Path:
        return Path(self.configuration.output.directory)


def get_runner(name: str) -> RunnerDefinition:
    """Resolve a configured runner name without importing JAX."""

    if name == "abl":
        from .abl import load_case, run_case

        return RunnerDefinition(load_case=load_case, run_case=run_case)
    if name == "concurrent_precursor_adm":
        from .concurrent_precursor_adm import load_case, run_case

        return RunnerDefinition(load_case=load_case, run_case=run_case)
    if name == "pressure_driven_warmup":
        from .pressure_driven_warmup import load_case, run_case

        return RunnerDefinition(load_case=load_case, run_case=run_case)
    if name == "gabls1":
        from .gabls1 import load_case, run_case

        return RunnerDefinition(load_case=load_case, run_case=run_case)
    if name in (
        "direct_rigid_alm",
        "direct_aeroelastic_alm",
    ):
        from .direct_rigid_alm import load_case, run_case

        return RunnerDefinition(load_case=load_case, run_case=run_case)
    available = (
        "abl, concurrent_precursor_adm, direct_aeroelastic_alm, "
        "direct_rigid_alm, gabls1, pressure_driven_warmup"
    )
    raise ValueError(f"unknown runner {name!r}; available runners: {available}")


def resolve_config_path(path: str | Path) -> Path:
    """Resolve either a case directory or an explicit TOML configuration."""

    source = Path(path)
    config_path = source / "config.toml" if source.is_dir() else source
    if not config_path.is_file():
        raise FileNotFoundError(f"case configuration does not exist: {config_path}")
    return config_path


def _configured_runner_name(config_path: Path) -> str:
    with config_path.open("rb") as stream:
        document: dict[str, Any] = tomllib.load(stream)
    case = document.get("case")
    if not isinstance(case, dict):
        raise ValueError("missing [case] table")
    runner_name = case.get("runner")
    if not isinstance(runner_name, str) or not runner_name:
        raise ValueError("case.runner must be a non-empty string")
    return runner_name


def _execution_config(config_path: Path) -> ExecutionConfig:
    with config_path.open("rb") as stream:
        document: dict[str, Any] = tomllib.load(stream)
    table = document.get("execution", {})
    if not isinstance(table, dict):
        raise ValueError("[execution] must be a table")
    unknown = set(table) - {"restart_checkpoint", "overwrite"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported [execution] keys: {names}")
    restart = table.get("restart_checkpoint")
    if restart is not None and (not isinstance(restart, str) or not restart):
        raise ValueError("execution.restart_checkpoint must be a path string")
    overwrite = table.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise ValueError("execution.overwrite must be boolean")
    return ExecutionConfig(
        restart_checkpoint=None if restart is None else Path(restart),
        overwrite=overwrite,
    )


def load_case(path: str | Path) -> ConfiguredCase:
    """Read and validate any built-in declarative case without importing JAX."""

    config_path = resolve_config_path(path)
    runner_name = _configured_runner_name(config_path)
    runner = get_runner(runner_name)
    return ConfiguredCase(
        config_path=config_path,
        runner_name=runner_name,
        runner=runner,
        configuration=runner.load_case(config_path),
        execution=_execution_config(config_path),
    )


def run_case(
    case: ConfiguredCase | str | Path,
) -> dict[str, Any]:
    """Execute a case using only behavior declared by its TOML file."""

    if isinstance(case, ConfiguredCase):
        configured = case
    else:
        source = Path(case)
        if source.is_dir() or source.suffix.lower() != ".toml":
            raise ValueError("run_case requires an explicit TOML configuration")
        configured = load_case(source)
    result = configured.runner.run_case(
        configured.configuration,
        output_dir=configured.output_directory,
        restart=configured.execution.restart_checkpoint,
        max_steps=None,
        overwrite=configured.execution.overwrite,
    )
    resolved_path = configured.output_directory / "resolved_config.toml"
    resolved_path.write_text(configured.resolved_toml())
    return result


__all__ = [
    "ConfiguredCase",
    "ExecutionConfig",
    "RunnerDefinition",
    "get_runner",
    "load_case",
    "resolve_config_path",
    "run_case",
]
