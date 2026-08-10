"""Command-line entry point for declarative JAX-Wind case directories."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 development fallback
    import tomli as tomllib

from .runners import load_case, resolve_config_path, run_case


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jaxwind",
        description="Run exactly the case declared by a TOML configuration.",
    )
    parser.add_argument(
        "case",
        type=Path,
        help="TOML case file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the resolved TOML without importing JAX",
    )
    return parser


def _resolve_config_path(
    parser: argparse.ArgumentParser,
    case_path: Path,
) -> Path:
    if case_path.is_dir() or case_path.suffix.lower() != ".toml":
        parser.error("launches require an explicit TOML configuration file")
    try:
        return resolve_config_path(case_path)
    except FileNotFoundError as error:
        parser.error(str(error))


def main(argv: list[str] | None = None) -> int:
    """Resolve a TOML case and execute its package-owned runner."""

    parser = _parser()
    args = parser.parse_args(argv)

    config_path = _resolve_config_path(parser, args.case)
    try:
        case = load_case(config_path)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        parser.error(str(error))

    if args.dry_run:
        print(case.resolved_toml(), end="")
        return 0

    run_case(case)
    return 0


__all__ = ["main"]
