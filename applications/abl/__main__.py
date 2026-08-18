"""Evaluate atmospheric-boundary-layer case data through JAX-Wind."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import load_abl
from .evaluate import evaluate, resolved


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "cases" / "Andren1994" / "config.toml"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--lasd-filter-backend",
        choices=("jax", "cufft"),
        default="jax",
        help="select the LASD test-filter implementation",
    )
    parser.add_argument(
        "--lasd-update-interval",
        type=int,
        help="override the number of timesteps between LASD coefficient updates",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case = load_abl(args.config)
    if args.lasd_update_interval is not None:
        if args.lasd_update_interval <= 0:
            raise SystemExit("--lasd-update-interval must be positive")
        case = replace(
            case,
            model=replace(
                case.model,
                momentum=replace(
                    case.model.momentum,
                    sgs=replace(
                        case.model.momentum.sgs,
                        update_interval=args.lasd_update_interval,
                    ),
                ),
            ),
        )
    if args.dry_run:
        print(json.dumps(resolved(case), indent=2))
        return 0
    evaluate(
        case,
        output_dir=args.output or case.output.directory,
        restart=args.restart,
        max_steps=args.max_steps,
        overwrite=args.overwrite,
        lasd_filter_backend=args.lasd_filter_backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
