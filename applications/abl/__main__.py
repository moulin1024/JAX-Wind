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
        "--advection",
        choices=("conservative", "rotational"),
        help="override momentum advection while retaining the physical case data",
    )
    parser.add_argument(
        "--dealiasing",
        choices=("three_halves", "two_thirds", "legacy_two_thirds"),
        help="override the horizontal nonlinear dealiasing rule",
    )
    parser.add_argument(
        "--lasd-filter-backend",
        choices=("jax", "cufft"),
        default="jax",
        help="select the LASD test-filter implementation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case = load_abl(args.config)
    if args.advection is not None:
        from jaxwind.physics import ConservativeAdvection, RotationalAdvection

        advection = (
            RotationalAdvection()
            if args.advection == "rotational"
            else ConservativeAdvection()
        )
        case = replace(
            case,
            model=replace(
                case.model,
                momentum=replace(case.model.momentum, advection=advection),
            ),
        )
    if args.dealiasing is not None:
        case = replace(case, nonlinear_dealiasing=args.dealiasing)
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
