"""Run one finite-volume ABL case from a TOML configuration file."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import load_fv_abl
from .evaluate import evaluate, resolved


def parser(
    default_config: Path | None = None,
) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    if default_config is None:
        result.add_argument("config", type=Path)
    else:
        result.add_argument(
            "config", type=Path, nargs="?", default=default_config
        )
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--output", type=Path)
    result.add_argument("--max-steps", type=int)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--chunk", type=int, help=argparse.SUPPRESS)
    result.add_argument(
        "--turbulent-prandtl", type=float, help=argparse.SUPPRESS
    )
    return result


def run(arguments: argparse.Namespace) -> dict:
    if arguments.max_steps is not None and arguments.max_steps < 0:
        raise ValueError("--max-steps must be nonnegative")
    if arguments.chunk is not None and arguments.chunk <= 0:
        raise ValueError("--chunk must be positive")
    if (
        arguments.turbulent_prandtl is not None
        and arguments.turbulent_prandtl <= 0.0
    ):
        raise ValueError("--turbulent-prandtl must be positive")
    case = load_fv_abl(arguments.config)
    if arguments.chunk is not None:
        case = replace(
            case,
            options=replace(case.options, chunk_steps=arguments.chunk),
        )
    if arguments.turbulent_prandtl is not None:
        case = replace(
            case,
            options=replace(
                case.options,
                turbulent_prandtl=arguments.turbulent_prandtl,
            ),
        )
    if arguments.dry_run:
        result = resolved(case)
        print(json.dumps(result, indent=2))
        return result
    return evaluate(
        case,
        output_dir=arguments.output or case.options.output_directory,
        max_steps=arguments.max_steps,
        overwrite=arguments.overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        run(parser().parse_args(argv))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
