#!/usr/bin/env python3
"""Minimal neutral log-law ABL with KEP4/MP5 momentum and AMD SGS."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--lx", type=float, default=6.283185307179586)
    parser.add_argument("--ly", type=float, default=3.141592653589793)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--ustar", type=float, default=0.1)
    parser.add_argument("--z0", type=float, default=1.0e-3)
    parser.add_argument("--amd-coefficient", type=float, default=0.212)
    parser.add_argument("--mp5-strength", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument(
        "--target-cfl",
        type=float,
        help="adapt dt each step to this advective CFL",
    )
    parser.add_argument("--pressure-rtol", type=float, default=2.0e-6)
    parser.add_argument("--pressure-max-iterations", type=int, default=80)
    parser.add_argument("--pressure-restart", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--perturbation", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/neutral_loglaw_amd"),
    )
    parser.add_argument("--single", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.nx, args.ny, args.nz, args.steps) <= 0:
        raise SystemExit("grid dimensions and steps must be positive")
    if args.sample_every <= 0 or args.log_every <= 0:
        raise SystemExit("sampling intervals must be positive")
    if args.target_cfl is not None and args.target_cfl <= 0.0:
        raise SystemExit("target CFL must be positive")
    if args.pressure_rtol <= 0.0:
        raise SystemExit("pressure tolerance must be positive")
    if min(args.pressure_max_iterations, args.pressure_restart) <= 0:
        raise SystemExit("pressure iteration controls must be positive")

    from jax import config as jax_config

    if not args.single:
        jax_config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaxwind.momentum import AMDModel, NeutralABLConfig, NeutralABLMomentum
    from jaxwind.pressure import (
        BoundaryCondition,
        FGMRESConfig,
        GMGConfig,
        MatrixFreePoissonSolver,
        PoissonBoundaryConditions,
        RectilinearGrid,
    )

    dtype = jnp.float32 if args.single else jnp.float64
    grid = RectilinearGrid.uniform(
        args.nx,
        args.ny,
        args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.height,
    )
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    pressure_boundaries = PoissonBoundaryConditions(
        periodic,
        periodic,
        periodic,
        periodic,
        neumann,
        neumann,
    )
    pressure = MatrixFreePoissonSolver(
        grid,
        pressure_boundaries,
        dtype=dtype,
        gmg=GMGConfig(
            smoother="auto",
            coarsening="auto",
        ),
        krylov=FGMRESConfig(
            restart=args.pressure_restart,
            max_iterations=args.pressure_max_iterations,
            relative_tolerance=(
                args.pressure_rtol if args.single else min(args.pressure_rtol, 1.0e-9)
            ),
        ),
    )
    config = NeutralABLConfig(
        friction_velocity=args.ustar,
        roughness_length=args.z0,
        mp5_dissipation_strength=args.mp5_strength,
        amd=AMDModel(coefficient=args.amd_coefficient),
    )
    solver = NeutralABLMomentum(grid, pressure, config)
    velocity = solver.initial_log_profile(
        perturbation_amplitude=args.perturbation
    )
    profiles: list[np.ndarray] = []
    tke_profiles: list[np.ndarray] = []
    uw_profiles: list[np.ndarray] = []
    diagnostics = []
    timesteps: list[float] = []
    simulation_time = 0.0
    start = time.perf_counter()

    for step in range(1, args.steps + 1):
        timestep = (
            args.dt
            if args.target_cfl is None
            else solver.timestep_for_cfl(velocity, args.target_cfl)
        )
        velocity = solver.step(
            velocity,
            timestep=timestep,
            time=simulation_time,
        )
        simulation_time += timestep
        timesteps.append(timestep)
        if step % args.sample_every == 0 or step == args.steps:
            mean, tke, minus_uw = solver.plane_statistics(velocity)
            profiles.append(np.asarray(mean[..., 0]))
            tke_profiles.append(np.asarray(tke))
            uw_profiles.append(np.asarray(minus_uw))
        if step % args.log_every == 0 or step == args.steps:
            diagnostic = solver.diagnostic(
                velocity,
                timestep=timestep,
                time=simulation_time,
            )
            diagnostics.append(diagnostic)
            print(
                f"step={step}/{args.steps} t={diagnostic.time:.4f} "
                f"CFL={diagnostic.maximum_cfl:.4f} "
                f"ustar={diagnostic.mean_wall_ustar:.5f} "
                f"div={diagnostic.divergence_norm:.3e} "
                f"nu_amd_max={diagnostic.maximum_amd_viscosity:.3e}"
            )
    jax.block_until_ready(velocity.x)
    elapsed = time.perf_counter() - start

    retained = profiles[max(0, len(profiles) // 2) :]
    retained_tke = tke_profiles[max(0, len(tke_profiles) // 2) :]
    retained_uw = uw_profiles[max(0, len(uw_profiles) // 2) :]
    mean_profile = np.mean(retained, axis=0)
    mean_tke = np.mean(retained_tke, axis=0)
    mean_minus_uw = np.mean(retained_uw, axis=0)
    z = (np.arange(args.nz) + 0.5) * args.height / args.nz
    target = args.ustar / config.von_karman * np.log(z / args.z0)
    fit_mask = (z >= max(2.0 * args.height / args.nz, 5.0 * args.z0)) & (
        z <= 0.5 * args.height
    )
    design = np.column_stack((np.log(z[fit_mask]), np.ones(np.sum(fit_mask))))
    slope, intercept = np.linalg.lstsq(
        design,
        mean_profile[fit_mask],
        rcond=None,
    )[0]
    fitted_ustar = config.von_karman * slope
    fitted_z0 = float(np.exp(-intercept / slope)) if slope > 0.0 else float("nan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / "mean_profile.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "z_m",
                "mean_u_m_s",
                "target_log_u_m_s",
                "resolved_tke_m2_s2",
                "minus_uw_m2_s2",
            )
        )
        writer.writerows(
            zip(z, mean_profile, target, mean_tke, mean_minus_uw)
        )

    summary = {
        "backend": jax.default_backend(),
        "dtype": str(dtype),
        "shape_zyx": grid.shape,
        "steps": args.steps,
        "simulation_time_seconds": simulation_time,
        "fixed_dt_seconds": args.dt if args.target_cfl is None else None,
        "target_cfl": args.target_cfl,
        "minimum_dt_seconds": min(timesteps),
        "maximum_dt_seconds": max(timesteps),
        "elapsed_seconds": elapsed,
        "target_ustar_m_s": args.ustar,
        "target_z0_m": args.z0,
        "fitted_ustar_m_s": float(fitted_ustar),
        "fitted_z0_m": fitted_z0,
        "amd_coefficient": args.amd_coefficient,
        "mp5_dissipation_strength": args.mp5_strength,
        "pressure_relative_tolerance": (
            args.pressure_rtol if args.single else min(args.pressure_rtol, 1.0e-9)
        ),
        "pressure_max_iterations": args.pressure_max_iterations,
        "final": asdict(diagnostics[-1]),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(args.output_dir / ".matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    axis.plot(mean_profile, z, label="AMD LES sampled mean", linewidth=2.0)
    axis.plot(target, z, "--", label="target neutral log law")
    axis.set_xlabel("mean streamwise velocity [m/s]")
    axis.set_ylabel("z [m]")
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.set_title(
        f"KEP4/MP5 + AMD: fitted u*={fitted_ustar:.4f} m/s, "
        f"z0={fitted_z0:.3e} m"
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "loglaw_profile.png", dpi=180)
    plt.close(figure)

    turbulence_figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    axes[0].plot(mean_tke, z, linewidth=2.0)
    axes[0].set_xlabel("resolved TKE [m²/s²]")
    axes[1].plot(mean_minus_uw, z, linewidth=2.0)
    axes[1].set_xlabel("-<u'w'> [m²/s²]")
    for panel in axes:
        panel.set_ylabel("z [m]")
        panel.grid(True, alpha=0.25)
    turbulence_figure.suptitle("Resolved turbulence profiles")
    turbulence_figure.tight_layout()
    turbulence_figure.savefig(
        args.output_dir / "turbulence_profiles.png",
        dpi=180,
    )
    plt.close(turbulence_figure)
    print(f"[done] elapsed={elapsed:.3f}s summary={summary_path}")


if __name__ == "__main__":
    main()
