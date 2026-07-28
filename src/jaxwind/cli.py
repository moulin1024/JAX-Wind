"""Command-line entry point for declarative JAX-Wind case directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 development fallback
    import tomli as tomllib

from .runners import get_runner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jaxwind",
        description=(
            "Run a JAX-Wind case directory. The directory must contain a "
            "config.toml whose [case] table selects a built-in runner."
        ),
    )
    parser.add_argument(
        "case",
        type=Path,
        help="case directory containing config.toml, or a TOML case file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="override the config.toml selected by the case argument",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--restart", type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="cap accepted steps without changing the configured final time",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the resolved TOML without importing JAX",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow a fresh run to replace an existing latest checkpoint",
    )
    return parser


def _resolve_config_path(
    parser: argparse.ArgumentParser,
    case_path: Path,
    override: Path | None,
) -> Path:
    if override is not None:
        config_path = override
    elif case_path.is_dir():
        config_path = case_path / "config.toml"
    else:
        config_path = case_path
    if not config_path.is_file():
        parser.error(f"case configuration does not exist: {config_path}")
    return config_path


def _runner_name(config_path: Path) -> str:
    with config_path.open("rb") as stream:
        document: dict[str, Any] = tomllib.load(stream)
    case = document.get("case")
    if not isinstance(case, dict):
        raise ValueError("missing [case] table")
    runner = case.get("runner")
    if not isinstance(runner, str) or not runner:
        raise ValueError("case.runner must be a non-empty string")
    return runner


def main(argv: list[str] | None = None) -> int:
    """Resolve a case directory and execute its package-owned runner."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")

    config_path = _resolve_config_path(parser, args.case, args.config)
    try:
        runner = get_runner(_runner_name(config_path))
        case = runner.load_case(config_path)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        parser.error(str(error))

    if args.dry_run:
        print(case.resolved_toml(), end="")
        return 0

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(case.output.directory)
    )
    runner.run_case(
        case,
        output_dir=output_dir,
        restart=args.restart,
        max_steps=args.max_steps,
        overwrite=args.overwrite,
    )
    return 0


__all__ = ["main"]
