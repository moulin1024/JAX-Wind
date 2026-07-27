#!/usr/bin/env python3
"""Run the Nieuwstadt1993 benchmark with the new semantic JAX-Wind stack."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parents[1]
DEFAULT_OUTPUT = ROOT / "benchmark_results" / "Nieuwstadt1993_new"
REFERENCE_DIR = BENCHMARK_DIR / "reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Nieuwstadt et al. (1993) CBL benchmark, compare with "
            "digitized data, and overlay the result on the paper figures."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small eight-step end-to-end GPU smoke test.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="JAX-Wind new LASD 40×40×48")
    parser.add_argument("--legend-label", default="JAX-Wind new LASD 40×40×48 GPU")
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--dt", type=float, default=1.25)
    parser.add_argument("--steps", type=int, default=9646)
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
    return parser.parse_args()


def run_command(command: list[str], env: dict[str, str]) -> None:
    print(f"[benchmark] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    pressure_source = Path(
        env.get("JAXWIND_SPECTRAL_FD_SOURCE", ROOT / "external" / "bw1000_benchmark")
    ).resolve()
    env["JAXWIND_SPECTRAL_FD_SOURCE"] = str(pressure_source)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(pressure_source), env.get("PYTHONPATH", ""))
    )

    if not args.compare_only:
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

    if not args.skip_overlays:
        run_command(
            [
                sys.executable,
                str(BENCHMARK_DIR / "overlay_figures.py"),
                "--figure-dir",
                str(REFERENCE_DIR / "figures"),
                "--result-dir",
                str(output_dir),
                "--output",
                str(
                    BENCHMARK_DIR / "Nieuwstadt1993_LASD_complete_overlay.png"
                ),
                "--legend-label",
                args.legend_label,
            ],
            env,
        )

    print(f"[benchmark] complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
