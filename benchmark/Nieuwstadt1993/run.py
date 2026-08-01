#!/usr/bin/env python3
"""Run and compare the Nieuwstadt1993 non-spectral AMD benchmark."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parents[1]
DEFAULT_AMD_OUTPUT = (
    ROOT / "benchmark_results" / "nieuwstadt1993_nonspectral_amd_40x40x48"
)
DEFAULT_LASD_OUTPUT = ROOT / "benchmark_results" / "Nieuwstadt1993_new"
REFERENCE_DIR = BENCHMARK_DIR / "reference"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Nieuwstadt et al. (1993) CBL benchmark, compare with "
            "digitized data, and overlay the result on the paper figures."
        )
    )
    parser.add_argument(
        "--solver",
        choices=("amd-nonspectral", "lasd-semantic"),
        default="amd-nonspectral",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small end-to-end smoke test.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label")
    parser.add_argument("--legend-label")
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--dt", type=float, default=1.25)
    parser.add_argument("--steps", type=int, default=9646)
    parser.add_argument("--amd-coefficient", type=float, default=0.212)
    parser.add_argument("--scalar-amd-coefficient", type=float)
    parser.add_argument("--mp5-strength", type=float, default=1.0)
    parser.add_argument("--target-cfl", type=float, default=0.8)
    parser.add_argument("--target-diffusive-cfl", type=float, default=0.5)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--max-run-seconds", type=float)
    parser.add_argument("--lasd-update-interval", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--method", choices=("transpose", "spike"), default="spike")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Reuse an existing output directory and skip the solve.",
    )
    parser.add_argument("--skip-csv", action="store_true")
    parser.add_argument("--skip-overlays", action="store_true")
    return parser.parse_args(argv)


def run_command(command: list[str], env: dict[str, str]) -> None:
    print(f"[benchmark] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_AMD_OUTPUT
            if args.solver == "amd-nonspectral"
            else DEFAULT_LASD_OUTPUT
        )
    if args.run_label is None:
        args.run_label = (
            "JAX-Wind non-spectral AMD 40×40×48"
            if args.solver == "amd-nonspectral"
            else "JAX-Wind semantic LASD 40×40×48"
        )
    if args.legend_label is None:
        args.legend_label = args.run_label
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    python_path = [str(ROOT), str(ROOT / "src")]
    if args.solver == "lasd-semantic":
        pressure_source = Path(
            env.get(
                "JAXWIND_SPECTRAL_FD_SOURCE",
                ROOT / "external" / "bw1000_benchmark",
            )
        ).resolve()
        env["JAXWIND_SPECTRAL_FD_SOURCE"] = str(pressure_source)
        python_path.append(str(pressure_source))
    python_path.append(env.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    if not args.compare_only:
        if args.solver == "amd-nonspectral":
            solve_command = [
                sys.executable,
                str(BENCHMARK_DIR / "run_amd.py"),
                "--output-dir",
                str(output_dir),
                "--nx",
                str(args.nx),
                "--ny",
                str(args.ny),
                "--nz",
                str(args.nz),
                "--dt-max",
                str(args.dt),
                "--sample-every",
                str(args.sample_every),
                "--checkpoint-every",
                str(args.checkpoint_every),
                "--target-cfl",
                str(args.target_cfl),
                "--target-diffusive-cfl",
                str(args.target_diffusive_cfl),
                "--amd-coefficient",
                str(args.amd_coefficient),
                "--scalar-amd-coefficient",
                str(
                    args.amd_coefficient
                    if args.scalar_amd_coefficient is None
                    else args.scalar_amd_coefficient
                ),
                "--mp5-strength",
                str(args.mp5_strength),
                "--seed",
                str(args.seed),
            ]
            if args.dtype == "float32":
                solve_command.append("--single")
            if args.restart is not None:
                solve_command.extend(("--restart", str(args.restart)))
            if args.max_run_seconds is not None:
                solve_command.extend(
                    ("--max-run-seconds", str(args.max_run_seconds))
                )
        else:
            solve_command = [
                sys.executable,
                str(BENCHMARK_DIR / "run_new.py"),
                "--output-dir",
                str(output_dir),
                "--nx",
                str(args.nx),
                "--ny",
                str(args.ny),
                "--nz",
                str(args.nz),
                "--dt",
                str(args.dt),
                "--steps",
                str(args.steps),
                "--sample-every",
                str(args.sample_every),
                "--lasd-update-interval",
                str(args.lasd_update_interval),
                "--dtype",
                args.dtype,
                "--method",
                args.method,
                "--seed",
                str(args.seed),
            ]
        if args.quick:
            solve_command.append("--quick")
        elif args.max_steps is not None:
            solve_command.extend(("--max-steps", str(args.max_steps)))
        run_command(solve_command, env)

    profiles = output_dir / "profiles.csv"
    if not profiles.exists():
        raise SystemExit(
            f"ERROR: missing {profiles}; run without --compare-only or select a valid output directory."
        )

    reference_data = REFERENCE_DIR / "data"
    if not args.skip_csv and reference_data.exists():
        run_command(
            [
                sys.executable,
                str(BENCHMARK_DIR / "compare_csv.py"),
                "--profiles",
                str(profiles),
                "--reference-dir",
                str(reference_data),
                "--output-dir",
                str(output_dir / "comparison"),
                "--run-label",
                args.run_label,
            ],
            env,
        )
    elif not args.skip_csv:
        print("[benchmark] digitized CSV data absent; continuing with paper-image overlays")

    figure_dir = REFERENCE_DIR / "figures"
    figures_available = all(
        (figure_dir / f"fig{index}.png").exists() for index in range(1, 18)
    )
    if not args.skip_overlays and figures_available:
        overlay_name = (
            "Nieuwstadt1993_AMD_complete_overlay.png"
            if args.solver == "amd-nonspectral"
            else "Nieuwstadt1993_LASD_complete_overlay.png"
        )
        run_command(
            [
                sys.executable,
                str(BENCHMARK_DIR / "overlay_figures.py"),
                "--figure-dir",
                str(figure_dir),
                "--result-dir",
                str(output_dir),
                "--output",
                str(BENCHMARK_DIR / overlay_name),
                "--legend-label",
                args.legend_label,
            ],
            env,
        )
    elif not args.skip_overlays:
        print(
            "[benchmark] registered paper figures absent; "
            "continuing without image overlays"
        )

    print(f"[benchmark] complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
