#!/usr/bin/env python3
"""Run the Yang et al. rated wind-tunnel actuator-disk benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy" / "jax"))

from benchmark.yang2024.case import (  # noqa: E402
    PAPER_CASE,
    paper_settings,
    resolved_case,
)
from run_single import RUN_DEFAULTS, params_from_settings  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inflow",
        choices=("paper-uniform", "measured-log"),
        default="paper-uniform",
        help=(
            "paper baseline or the fitted measured-profile override "
            "(default: paper-uniform)"
        ),
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "yang2024_r9",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a four-step reduced-grid plumbing check",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved case without importing JAX",
    )
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    return args


def _write_inlet_profile(
    path: Path,
    velocity: np.ndarray,
    params,
) -> None:
    heights = (np.arange(params.nz, dtype=np.float64) + 0.5) * params.dz * params.z_i
    inlet_mean = np.mean(velocity[0, :, :], axis=0)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("height_m", "mean_u_m_s"))
        writer.writerows(zip(heights, inlet_mean, strict=True))


def run_case(args: argparse.Namespace) -> None:
    mode = args.inflow.replace("-", "_")
    case_settings = paper_settings(
        inflow_mode=mode,
        quick=args.quick,
        steps=args.steps,
    )
    if args.dry_run:
        payload = resolved_case(mode)
        payload["solver_settings"] = case_settings
        print(json.dumps(payload, indent=2))
        return

    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from wireles_jax import run
    from wireles_jax.wind_tunnel import actuator_disk_kernel

    settings = dict(RUN_DEFAULTS)
    settings.update(case_settings)
    params = params_from_settings(settings, jnp)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[yang2024:R9] inflow={args.inflow}, "
        f"grid={params.nx}x{params.ny}x{params.nz}, "
        f"steps={params.nsteps}",
        flush=True,
    )
    final_state, diagnostics = run(
        params,
        seed=args.seed,
        log_every=params.c_count,
        status_callback=lambda message: print(message, flush=True),
        log_callback=lambda row: print(
            f"[yang2024:R9] step={int(row.step):6d} "
            f"CFL={max(float(row.cfl_x), float(row.cfl_y), float(row.cfl_z)):.3f}",
            flush=True,
        ),
    )

    velocity = np.asarray(final_state.u, dtype=np.float64)
    disk = np.asarray(actuator_disk_kernel(params), dtype=np.float64)
    disk_velocity = float(np.sum(velocity * disk) / np.sum(disk))
    cell_volume_m3 = params.dx * params.dy * params.dz * params.z_i**3
    loaded_area_m2 = float(np.sum(disk) * cell_volume_m3)
    thrust_n = (
        0.5
        * PAPER_CASE.air_density_kg_m3
        * params.actuator_disk_ct_prime
        * disk_velocity**2
        * loaded_area_m2
    )

    _write_inlet_profile(
        args.output_dir / "final_inlet_profile.csv",
        velocity,
        params,
    )
    summary = resolved_case(mode)
    summary["run"] = {
        "seed": args.seed,
        "quick": args.quick,
        "final_step": int(final_state.step),
        "maximum_cfl": max(
            max(float(row.cfl_x), float(row.cfl_y), float(row.cfl_z))
            for row in diagnostics
        ),
        "final_disk_velocity_m_s": disk_velocity,
        "final_adm_thrust_n": thrust_n,
        "loaded_area_m2": loaded_area_m2,
        "power_validation_available": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[yang2024:R9] results: {args.output_dir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    run_case(parse_args(argv))


if __name__ == "__main__":
    main()
