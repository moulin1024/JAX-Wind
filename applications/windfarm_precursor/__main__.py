"""Generate an offline precursor and test its fringe-enforced main domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from applications.pressure_driven_lasd.config import load_case

from .evaluate import evaluate
from .replay import replay_main


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "cases" / "PressureDrivenLASD" / "config.toml"
DEFAULT_RESTART = (
    ROOT
    / "outputs"
    / "pressure_driven_lasd_64x64x64_gpu"
    / "checkpoint_final.npz"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "windfarm_precursor_smoke"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--restart", type=Path, default=DEFAULT_RESTART)
    parser.add_argument(
        "--recording",
        type=Path,
        help="reuse an existing precursor HDF5 file and run only the main domain",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--precursor-steps", type=int, default=4)
    parser.add_argument("--main-steps", type=int)
    parser.add_argument("--fringe-start-fraction", type=float, default=0.75)
    parser.add_argument("--fringe-relaxation-seconds", type=float, default=1.0)
    parser.add_argument("--section", choices=("inflow", "outflow"), default="inflow")
    parser.add_argument("--sample-buffer", type=int, default=512)
    parser.add_argument("--read-buffer", type=int, default=512)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument(
        "--turbine",
        choices=("none", "dtu-10mw-adm"),
        default="none",
    )
    parser.add_argument("--thrust-coefficient-prime", type=float, default=4.0 / 3.0)
    parser.add_argument("--disk-smoothing-width-m", type=float)
    parser.add_argument(
        "--compression",
        choices=("none", "lzf", "gzip"),
        default="none",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case = load_case(args.config)
    main_steps = args.precursor_steps if args.main_steps is None else args.main_steps
    resolved = {
        "case": case.name,
        "restart": str(args.restart),
        "output": str(args.output),
        "precursor_steps": args.precursor_steps,
        "main_steps": main_steps,
        "fringe_start_fraction": args.fringe_start_fraction,
        "fringe_relaxation_seconds": args.fringe_relaxation_seconds,
        "section": args.section,
        "sample_buffer": args.sample_buffer,
        "read_buffer": args.read_buffer,
        "frames": args.frames,
        "gif_fps": args.gif_fps,
        "compression": args.compression,
        "recording": None if args.recording is None else str(args.recording),
        "turbine": args.turbine,
        "thrust_coefficient_prime": args.thrust_coefficient_prime,
        "disk_smoothing_width_m": args.disk_smoothing_width_m,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return 0
    turbine = None
    if args.turbine == "dtu-10mw-adm":
        from jaxwind.windfarm import dtu_10mw_reference_actuator_disk

        smoothing_width_m = (
            case.domain.lx_m / case.domain.nx
            if args.disk_smoothing_width_m is None
            else args.disk_smoothing_width_m
        )
        turbine = dtu_10mw_reference_actuator_disk(
            x_m=0.5 * case.domain.lx_m,
            y_m=0.5 * case.domain.ly_m,
            smoothing_width_m=smoothing_width_m,
            thrust_coefficient_prime=args.thrust_coefficient_prime,
        )
    if args.recording is not None:
        replay_main(
            case,
            restart=args.restart,
            recording=args.recording,
            output_dir=args.output,
            main_steps=main_steps,
            fringe_start_fraction=args.fringe_start_fraction,
            fringe_relaxation_seconds=args.fringe_relaxation_seconds,
            section=args.section,
            read_buffer=args.read_buffer,
            frame_count=args.frames,
            gif_fps=args.gif_fps,
            turbine=turbine,
            overwrite=args.overwrite,
        )
        return 0
    evaluate(
        case,
        restart=args.restart,
        output_dir=args.output,
        precursor_steps=args.precursor_steps,
        main_steps=main_steps,
        fringe_start_fraction=args.fringe_start_fraction,
        fringe_relaxation_seconds=args.fringe_relaxation_seconds,
        section=args.section,
        sample_buffer=args.sample_buffer,
        read_buffer=args.read_buffer,
        compression=None if args.compression == "none" else args.compression,
        frame_count=args.frames,
        gif_fps=args.gif_fps,
        turbine=turbine,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
