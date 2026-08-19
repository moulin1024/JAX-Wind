"""Run the strict CUDA-Fortran offline-precursor wind-farm workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from applications.pressure_driven_lasd.config import load_case

from .evaluate import evaluate
from .legacy_inflow import STRICT_LEGACY_INFLOW
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
    parser.add_argument("--precursor-steps", type=int, default=100)
    parser.add_argument("--main-steps", type=int)
    parser.add_argument(
        "--legacy-inflow-directory",
        type=Path,
        help="directory containing p000_inflow_[uvw].bin from legacy Fortran",
    )
    parser.add_argument("--sample-buffer", type=int, default=8)
    parser.add_argument("--read-buffer", type=int, default=8)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument(
        "--turbine",
        choices=("none", "dtu-10mw-adm", "dtu-10mw-prescribed-adm", "dtu-10mw-ad-bem"),
        default="none",
    )
    parser.add_argument(
        "--openfast-model",
        type=Path,
        help="OpenFAST .fst deck used by dtu-10mw-ad-bem",
    )
    parser.add_argument("--rotor-speed-rpm", type=float)
    parser.add_argument(
        "--ad-bem-smearing-azimuthal-elements",
        type=int,
        default=64,
        help="virtual azimuthal element count in the legacy ADMR width formula",
    )
    parser.add_argument("--blade-pitch-degrees", type=float)
    parser.add_argument("--nacelle-drag-coefficient", type=float, default=1.0)
    parser.add_argument("--tower-drag-coefficient", type=float, default=1.0)
    parser.add_argument("--thrust-coefficient-prime", type=float, default=4.0 / 3.0)
    parser.add_argument(
        "--turbine-x-m",
        type=float,
        help="streamwise turbine position; defaults to the domain midpoint",
    )
    parser.add_argument(
        "--disk-smoothing-width-m",
        type=float,
        help=(
            "fixed width for the simple ADM; retained only as actuator-line "
            "metadata for AD-BEM, whose width follows the legacy element-size formula"
        ),
    )
    parser.add_argument(
        "--body-smoothing-width-m",
        type=float,
        help="separate Gaussian width for nacelle and tower drag",
    )
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
    contract = STRICT_LEGACY_INFLOW
    main_steps = args.precursor_steps if args.main_steps is None else args.main_steps
    resolved = {
        "case": case.name,
        "restart": str(args.restart),
        "output": str(args.output),
        "precursor_steps": args.precursor_steps,
        "main_steps": main_steps,
        "compatibility": "strict-cuda-fortran",
        "inflow_enforcement": "legacy-overwrite",
        "inflow_start_plane": contract.start_plane,
        "inflow_end_plane": contract.end_plane,
        "inflow_update_steps": contract.update_interval_steps,
        "spanwise_cycle_updates": contract.cycle_interval_updates,
        "main_pressure_gradient": "off",
        "legacy_inflow_directory": (
            None if args.legacy_inflow_directory is None else str(args.legacy_inflow_directory)
        ),
        "section": "inflow",
        "sample_buffer": args.sample_buffer,
        "read_buffer": args.read_buffer,
        "frames": args.frames,
        "gif_fps": args.gif_fps,
        "compression": args.compression,
        "recording": None if args.recording is None else str(args.recording),
        "turbine": args.turbine,
        "thrust_coefficient_prime": args.thrust_coefficient_prime,
        "turbine_x_m": args.turbine_x_m,
        "disk_smoothing_width_m": args.disk_smoothing_width_m,
        "body_smoothing_width_m": args.body_smoothing_width_m,
        "openfast_model": None if args.openfast_model is None else str(args.openfast_model),
        "rotor_speed_rpm": args.rotor_speed_rpm,
        "ad_bem_smearing_azimuthal_elements": (
            args.ad_bem_smearing_azimuthal_elements
        ),
        "blade_pitch_degrees": args.blade_pitch_degrees,
        "nacelle_drag_coefficient": args.nacelle_drag_coefficient,
        "tower_drag_coefficient": args.tower_drag_coefficient,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return 0
    turbine = None
    if args.turbine in ("dtu-10mw-adm", "dtu-10mw-prescribed-adm"):
        from jaxwind.windfarm import dtu_10mw_reference_actuator_disk

        smoothing_width_m = (
            case.domain.lx_m / case.domain.nx
            if args.disk_smoothing_width_m is None
            else args.disk_smoothing_width_m
        )
        factory = dtu_10mw_reference_actuator_disk
        if args.turbine == "dtu-10mw-prescribed-adm":
            from jaxwind.windfarm import dtu_10mw_prescribed_actuator_disk
            factory = dtu_10mw_prescribed_actuator_disk
        turbine = factory(
            x_m=(
                0.5 * case.domain.lx_m
                if args.turbine_x_m is None
                else args.turbine_x_m
            ),
            y_m=0.5 * case.domain.ly_m,
            smoothing_width_m=smoothing_width_m,
            **(
                {"thrust_coefficient_prime": args.thrust_coefficient_prime}
                if args.turbine == "dtu-10mw-adm"
                else {}
            ),
            **(
                {
                    "force_x_offset_m": -0.5 * case.domain.lx_m / case.domain.nx,
                    "force_y_offset_m": -0.5 * case.domain.ly_m / case.domain.ny,
                }
                if args.turbine == "dtu-10mw-prescribed-adm"
                else {}
            ),
        )
    elif args.turbine == "dtu-10mw-ad-bem":
        if args.openfast_model is None:
            _parser().error("--openfast-model is required for dtu-10mw-ad-bem")
        from jaxwind.windfarm import (
            RigidBladeElementDisk,
            load_openfast_rigid_turbine,
        )

        rotor = load_openfast_rigid_turbine(args.openfast_model)
        smoothing_width_m = (
            case.domain.lx_m / case.domain.nx
            if args.disk_smoothing_width_m is None
            else args.disk_smoothing_width_m
        )
        turbine = RigidBladeElementDisk(
            rotor=rotor,
            x_m=(
                0.5 * case.domain.lx_m
                if args.turbine_x_m is None
                else args.turbine_x_m
            ),
            y_m=0.5 * case.domain.ly_m,
            smoothing_width_m=smoothing_width_m,
            hub_height_m=rotor.hub_height_m,
            rotor_speed_rpm=args.rotor_speed_rpm,
            smearing_azimuthal_elements=(
                args.ad_bem_smearing_azimuthal_elements
            ),
            pitch_degrees=args.blade_pitch_degrees,
            body_smoothing_width_m=args.body_smoothing_width_m,
            nacelle_drag_coefficient=args.nacelle_drag_coefficient,
            tower_drag_coefficient=args.tower_drag_coefficient,
        )
    if args.recording is not None:
        replay_main(
            case,
            restart=args.restart,
            recording=args.recording,
            output_dir=args.output,
            main_steps=main_steps,
            legacy_inflow_directory=args.legacy_inflow_directory,
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
