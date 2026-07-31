#!/usr/bin/env python3
"""Reproduce the neutral Ekman-layer intercomparison of Andren et al. (1994)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import time


INITIAL_U = (
    4.44, 5.92, 6.91, 7.73, 8.43, 9.02, 9.52, 9.93, 10.25, 10.47,
    10.62, 10.70, 10.71, 10.67, 10.59, 10.48, 10.36, 10.24, 10.13, 10.04,
    9.99, 9.96, 9.95, 9.96, 9.98, 9.99, 10.00, 9.99, 9.99, 9.99,
    10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00,
)
INITIAL_V = (
    2.18, 2.67, 2.83, 2.84, 2.75, 2.57, 2.34, 2.06, 1.75, 1.44,
    1.12, 0.82, 0.55, 0.31, 0.12, -0.02, -0.11, -0.16, -0.17, -0.15,
    -0.11, -0.06, -0.02, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01,
    0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
)
INITIAL_TKE = (
    0.365, 0.295, 0.245, 0.205, 0.175, 0.145, 0.120, 0.100, 0.085,
    0.070, 0.055, 0.045, 0.035, 0.025, 0.020, 0.015, 0.010, 0.010,
    0.005, 0.005, 0.005, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-ft", type=float, default=0.1)
    parser.add_argument("--sample-start-ft", type=float, default=0.05)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--target-cfl", type=float, default=0.8)
    parser.add_argument("--target-diffusive-cfl", type=float, default=0.5)
    parser.add_argument("--lasd-update-interval", type=int, default=1)
    parser.add_argument("--lasd-filter-grid-ratio", type=float, default=1.0)
    parser.add_argument("--lasd-maximum-coefficient", type=float, default=0.81)
    parser.add_argument("--mp5-strength", type=float, default=1.0)
    parser.add_argument(
        "--sgs-time-integration",
        choices=("imex_ark3", "explicit"),
        default="imex_ark3",
    )
    parser.add_argument("--pressure-rtol", type=float, default=1.0e-4)
    parser.add_argument("--pressure-max-iterations", type=int, default=20)
    parser.add_argument("--pressure-restart", type=int, default=10)
    parser.add_argument(
        "--linear-solver",
        choices=("pcg", "gmres"),
        default="pcg",
    )
    parser.add_argument(
        "--projection-method",
        choices=("full", "fpj2"),
        default="fpj2",
    )
    parser.add_argument(
        "--fpj2-timestep-ratio-limit",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--krylov-execution",
        choices=("jax", "python"),
        default="jax",
    )
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--max-run-seconds",
        type=float,
        help="pause cleanly at a checkpoint after this much stepping wall time",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/andren1994_40cubed"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_ft <= 0.0:
        raise SystemExit("end-ft must be positive")
    if not 0.0 <= args.sample_start_ft < args.end_ft:
        raise SystemExit("sample-start-ft must lie in [0, end-ft)")
    if min(args.sample_every, args.log_every) <= 0:
        raise SystemExit("sampling intervals must be positive")
    if args.lasd_update_interval <= 0:
        raise SystemExit("LASD update interval must be positive")
    if args.checkpoint_every <= 0:
        raise SystemExit("checkpoint interval must be positive")
    if args.fpj2_timestep_ratio_limit < 1.0:
        raise SystemExit("FPJ-2 timestep ratio limit must be at least one")
    if args.max_run_seconds is not None and args.max_run_seconds <= 0.0:
        raise SystemExit("max-run-seconds must be positive")

    from jax import config as jax_config

    if not args.single:
        jax_config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaxwind.momentum import (
        FPJ2State,
        LASDModel,
        LASDState,
        NeutralABLConfig,
        NeutralABLMomentum,
    )
    from jaxwind.pressure import (
        BoundaryCondition,
        FGMRESConfig,
        GMGConfig,
        MACVelocity,
        MatrixFreePoissonSolver,
        PCGConfig,
        PoissonBoundaryConditions,
        RectilinearGrid,
    )

    nx = ny = nz = 40
    lx, ly, height = 4000.0, 2000.0, 1500.0
    roughness = 0.1
    geostrophic = (10.0, 0.0)
    coriolis = 1.0e-4
    expected_ustar = 0.425
    dtype = jnp.float32 if args.single else jnp.float64

    grid = RectilinearGrid.uniform(
        nx,
        ny,
        nz,
        lx=lx,
        ly=ly,
        lz=height,
    )
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    krylov = (
        PCGConfig(
            max_iterations=args.pressure_max_iterations,
            relative_tolerance=args.pressure_rtol,
            execution=args.krylov_execution,
        )
        if args.linear_solver == "pcg"
        else FGMRESConfig(
            restart=args.pressure_restart,
            max_iterations=args.pressure_max_iterations,
            relative_tolerance=args.pressure_rtol,
            reorthogonalize=False,
            execution=args.krylov_execution,
        )
    )
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions(
            periodic,
            periodic,
            periodic,
            periodic,
            neumann,
            neumann,
        ),
        dtype=dtype,
        gmg=GMGConfig(
            smoother="auto",
            coarsening="auto",
            coarse_smooth=20,
        ),
        krylov=krylov,
    )
    solver = NeutralABLMomentum(
        grid,
        pressure,
        NeutralABLConfig(
            friction_velocity=expected_ustar,
            roughness_length=roughness,
            geostrophic_wind=geostrophic,
            coriolis_vertical=coriolis,
            coriolis_horizontal=coriolis,
            mp5_dissipation_strength=args.mp5_strength,
            sgs_time_integration=args.sgs_time_integration,
            projection_method=args.projection_method,
            fpj2_timestep_ratio_limit=args.fpj2_timestep_ratio_limit,
            lasd=LASDModel(
                filter_grid_ratio=args.lasd_filter_grid_ratio,
                update_interval=args.lasd_update_interval,
                maximum_coefficient=args.lasd_maximum_coefficient,
                x_boundary="periodic",
                y_boundary="periodic",
            ),
        ),
    )

    def sample_profiles_kernel(sample_velocity, lasd_coefficient):
        cells = solver.cell_centered_velocity(sample_velocity)
        mean = jnp.mean(cells, axis=(1, 2))
        fluctuation = cells - mean[:, None, None, :]
        variances = jnp.mean(fluctuation * fluctuation, axis=(1, 2))
        resolved_uw = jnp.mean(
            fluctuation[..., 0] * fluctuation[..., 2],
            axis=(1, 2),
        )
        resolved_vw = jnp.mean(
            fluctuation[..., 1] * fluctuation[..., 2],
            axis=(1, 2),
        )
        stress = solver.sgs_stress(cells, lasd_coefficient)
        sgs_uw = -jnp.mean(stress[..., 0, 2], axis=(1, 2))
        sgs_vw = -jnp.mean(stress[..., 1, 2], axis=(1, 2))
        return mean, variances, resolved_uw, resolved_vw, sgs_uw, sgs_vw

    compiled_sample_profiles = jax.jit(sample_profiles_kernel)

    if args.restart is None:
        velocity = solver.initial_profile(
            jnp.asarray(INITIAL_U, dtype=dtype),
            jnp.asarray(INITIAL_V, dtype=dtype),
            perturbation_tke=jnp.asarray(INITIAL_TKE, dtype=dtype),
            seed=args.seed,
        )
        solver.reset_lasd(velocity)
        samples: list[tuple[np.ndarray, ...]] = []
        timesteps: list[float] = []
        simulation_time = 0.0
        step = 0
    else:
        checkpoint = np.load(args.restart)
        velocity = MACVelocity(
            jnp.asarray(checkpoint["velocity_x"], dtype=dtype),
            jnp.asarray(checkpoint["velocity_y"], dtype=dtype),
            jnp.asarray(checkpoint["velocity_z"], dtype=dtype),
        )
        lasd_state = LASDState(
            *(jnp.asarray(checkpoint[f"lasd_{name}"], dtype=dtype) for name in (
                "coefficient",
                "lm",
                "mm",
                "qn",
                "nn",
                "trajectory_x",
                "trajectory_y",
                "trajectory_z",
            ))
        )
        step = int(checkpoint["step"])
        simulation_time = float(checkpoint["simulation_time"])
        solver.restore_lasd(
            lasd_state,
            accepted_step=int(checkpoint["lasd_step"]),
            interval_time=float(checkpoint["lasd_interval_time"]),
        )
        if (
            args.projection_method == "fpj2"
            and "fpj2_current_pressure" in checkpoint.files
        ):
            solver.restore_fpj2(
                FPJ2State(
                    jnp.asarray(
                        checkpoint["fpj2_current_pressure"],
                        dtype=dtype,
                    ),
                    jnp.asarray(
                        checkpoint["fpj2_previous_pressure"],
                        dtype=dtype,
                    ),
                    float(checkpoint["fpj2_current_timestep"]),
                    float(checkpoint["fpj2_previous_timestep"]),
                    int(checkpoint["fpj2_history_count"]),
                )
            )
        timesteps = list(np.asarray(checkpoint["timesteps"], dtype=float))
        sample_arrays = tuple(
            np.asarray(checkpoint[f"sample_{index}"]) for index in range(6)
        )
        samples = [
            tuple(array[index] for array in sample_arrays)
            for index in range(sample_arrays[0].shape[0])
        ]

    saved_lasd = solver.lasd_state
    saved_lasd_step, saved_interval_time = solver.lasd_progress
    saved_fpj2 = solver.fpj2_state
    compile_start = time.perf_counter()
    warmup_timestep = solver.timestep_for_cfl(
        velocity,
        args.target_cfl,
        args.target_diffusive_cfl,
    )
    compiled_velocity = velocity
    warmup_steps = max(
        args.lasd_update_interval,
        3 if args.projection_method == "fpj2" else 1,
    )
    for warmup_step in range(warmup_steps):
        compiled_velocity = solver.step(
            compiled_velocity,
            timestep=warmup_timestep,
            time=warmup_step * warmup_timestep,
        )
    jax.block_until_ready(compiled_velocity.x)
    if args.restart is None:
        solver.reset_lasd(velocity)
        solver.reset_fpj2()
    else:
        solver.restore_lasd(
            saved_lasd,
            accepted_step=saved_lasd_step,
            interval_time=saved_interval_time,
        )
        if saved_fpj2 is None:
            solver.reset_fpj2()
        else:
            solver.restore_fpj2(saved_fpj2)
    compiled_samples = compiled_sample_profiles(
        velocity,
        solver.lasd_state.coefficient,
    )
    jax.block_until_ready(compiled_samples[0])
    solver.diagnostic(
        velocity,
        timestep=warmup_timestep,
        time=0.0,
    )
    compilation_elapsed = time.perf_counter() - compile_start
    print(f"[compile] kernels ready in {compilation_elapsed:.3f}s")

    diagnostics = []
    final_time = args.end_ft / coriolis
    sample_start = args.sample_start_ft / coriolis
    start = time.perf_counter()

    def sample_profiles() -> tuple[np.ndarray, ...]:
        return tuple(
            np.asarray(value)
            for value in compiled_sample_profiles(
                velocity,
                solver.lasd_state.coefficient,
            )
        )

    def save_checkpoint() -> None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        lasd = solver.lasd_state
        lasd_step, interval_time = solver.lasd_progress
        if samples:
            stacked_samples = tuple(
                np.stack([sample[index] for sample in samples])
                for index in range(6)
            )
        else:
            shapes = ((nz, 3), (nz, 3), (nz,), (nz,), (nz,), (nz,))
            stacked_samples = tuple(
                np.empty((0, *shape), dtype=np.float32) for shape in shapes
            )
        payload = {
            "velocity_x": np.asarray(velocity.x),
            "velocity_y": np.asarray(velocity.y),
            "velocity_z": np.asarray(velocity.z),
            "step": step,
            "simulation_time": simulation_time,
            "lasd_step": lasd_step,
            "lasd_interval_time": interval_time,
            "timesteps": np.asarray(timesteps),
        }
        payload.update(
            {
                f"lasd_{name}": np.asarray(value)
                for name, value in zip(
                    LASDState._fields,
                    lasd,
                    strict=True,
                )
            }
        )
        fpj2 = solver.fpj2_state
        if fpj2 is not None:
            payload.update(
                {
                    "fpj2_current_pressure": np.asarray(
                        fpj2.current_pressure
                    ),
                    "fpj2_previous_pressure": np.asarray(
                        fpj2.previous_pressure
                    ),
                    "fpj2_current_timestep": fpj2.current_timestep,
                    "fpj2_previous_timestep": fpj2.previous_timestep,
                    "fpj2_history_count": fpj2.history_count,
                }
            )
        payload.update(
            {
                f"sample_{index}": values
                for index, values in enumerate(stacked_samples)
            }
        )
        np.savez_compressed(args.output_dir / "checkpoint.npz", **payload)

    while simulation_time < final_time:
        timestep = solver.timestep_for_cfl(
            velocity,
            args.target_cfl,
            args.target_diffusive_cfl,
        )
        timestep = min(timestep, final_time - simulation_time)
        step_lasd_coefficient = solver.lasd_state.coefficient
        velocity = solver.step(
            velocity,
            timestep=timestep,
            time=simulation_time,
        )
        simulation_time += timestep
        timesteps.append(timestep)
        step += 1
        if (
            simulation_time >= sample_start
            and (step % args.sample_every == 0 or simulation_time >= final_time)
        ):
            samples.append(sample_profiles())
        if step % args.log_every == 0 or simulation_time >= final_time:
            diagnostic = solver.diagnostic(
                velocity,
                timestep=timestep,
                time=simulation_time,
            )
            step_diagnostic = solver.diagnostic(
                velocity,
                timestep=timestep,
                time=simulation_time,
                lasd_coefficient=step_lasd_coefficient,
            )
            diagnostics.append(diagnostic)
            print(
                f"step={step} ft={coriolis * simulation_time:.4f}/"
                f"{args.end_ft:g} CFL={diagnostic.maximum_cfl:.4f} "
                f"CFLnu_step={step_diagnostic.maximum_diffusive_cfl:.4f} "
                f"CFLnu_next={diagnostic.maximum_diffusive_cfl:.4f} "
                f"ustar/Ug={diagnostic.mean_wall_ustar / geostrophic[0]:.5f} "
                f"divL2={diagnostic.divergence_norm:.3e} "
                f"nu_lasd_max={diagnostic.maximum_sgs_viscosity:.3e} "
                f"Cs2_mean/max={diagnostic.mean_sgs_coefficient:.3e}/"
                f"{diagnostic.maximum_sgs_coefficient:.3e} "
                f"clipped={diagnostic.clipped_sgs_coefficient_fraction:.3f}"
            )
        if step % args.checkpoint_every == 0:
            save_checkpoint()
            if (
                args.max_run_seconds is not None
                and time.perf_counter() - start >= args.max_run_seconds
            ):
                print(
                    f"[paused] checkpointed step={step} "
                    f"ft={coriolis * simulation_time:.4f}",
                    flush=True,
                )
                return
    jax.block_until_ready(velocity.x)
    save_checkpoint()
    elapsed = time.perf_counter() - start

    averaged = [
        np.mean([sample[index] for sample in samples], axis=0)
        for index in range(len(samples[0]))
    ]
    mean, variances, resolved_uw, resolved_vw, sgs_uw, sgs_vw = averaged
    total_uw = resolved_uw + sgs_uw
    total_vw = resolved_vw + sgs_vw
    z = (np.arange(nz) + 0.5) * height / nz
    speed = np.linalg.norm(mean[:, :2], axis=-1)
    ustar = diagnostics[-1].mean_wall_ustar
    normalized_height = z * coriolis / ustar
    phi_m = 0.4 * z / ustar * np.gradient(speed, z)
    resolved_tke = 0.5 * np.sum(variances, axis=-1)
    integrated_tke = float(
        coriolis * np.trapezoid(resolved_tke, z) / ustar**3
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / "andren1994_profiles.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "z_m",
                "zf_over_ustar",
                "mean_u_m_s",
                "mean_v_m_s",
                "var_u_m2_s2",
                "var_v_m2_s2",
                "var_w_m2_s2",
                "resolved_uw_m2_s2",
                "resolved_vw_m2_s2",
                "sgs_uw_m2_s2",
                "sgs_vw_m2_s2",
                "total_uw_m2_s2",
                "total_vw_m2_s2",
                "phi_m",
            )
        )
        writer.writerows(
            zip(
                z,
                normalized_height,
                mean[:, 0],
                mean[:, 1],
                variances[:, 0],
                variances[:, 1],
                variances[:, 2],
                resolved_uw,
                resolved_vw,
                sgs_uw,
                sgs_vw,
                total_uw,
                total_vw,
                phi_m,
            )
        )

    summary = {
        "reference": "Andren et al. (1994), QJRMS 120, 1457-1484",
        "backend": jax.default_backend(),
        "dtype": str(dtype),
        "shape_zyx": grid.shape,
        "domain_m": [lx, ly, height],
        "roughness_length_m": roughness,
        "geostrophic_wind_m_s": geostrophic,
        "coriolis_vertical_s-1": coriolis,
        "coriolis_horizontal_s-1": coriolis,
        "end_ft": coriolis * simulation_time,
        "sample_start_ft": args.sample_start_ft,
        "steps": step,
        "minimum_dt_seconds": min(timesteps),
        "maximum_dt_seconds": max(timesteps),
        "elapsed_seconds": elapsed,
        "compilation_seconds": compilation_elapsed,
        "sgs_model": "physical-space LASD",
        "lasd_update_interval": args.lasd_update_interval,
        "lasd_filter_grid_ratio": args.lasd_filter_grid_ratio,
        "lasd_sgs_delta_scale": args.lasd_filter_grid_ratio,
        "lasd_maximum_coefficient": args.lasd_maximum_coefficient,
        "lasd_filter": "three-dimensional compact top-hat convolution",
        "lasd_clipped_beta_fallback": True,
        "mp5_dissipation_strength": args.mp5_strength,
        "sgs_time_integration": args.sgs_time_integration,
        "vertical_sgs_diffusion_is_implicit": (
            args.sgs_time_integration == "imex_ark3"
        ),
        "horizontal_sgs_diffusion_is_explicit": True,
        "target_advective_cfl": args.target_cfl,
        "target_diffusive_cfl": args.target_diffusive_cfl,
        "linear_solver": args.linear_solver,
        "pressure_relative_tolerance": args.pressure_rtol,
        "pressure_max_iterations": args.pressure_max_iterations,
        "pressure_restart": args.pressure_restart,
        "krylov_execution": args.krylov_execution,
        "projection_method": args.projection_method,
        "fpj2_timestep_ratio_limit": args.fpj2_timestep_ratio_limit,
        "elapsed_seconds_scope": "final invocation after restart",
        "friction_velocity_m_s": ustar,
        "friction_velocity_over_geostrophic": ustar / geostrophic[0],
        "reference_friction_velocity_ratio_range": [0.0402, 0.0448],
        "normalized_integrated_resolved_tke": integrated_tke,
        "reference_normalized_integrated_total_tke": 0.7,
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

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), sharey=True)
    axes[0, 0].plot(mean[:, 0] / geostrophic[0], normalized_height, label="U/Ug")
    axes[0, 0].plot(mean[:, 1] / geostrophic[0], normalized_height, label="V/Ug")
    axes[0, 0].set_xlabel("mean velocity / Ug")
    axes[0, 0].legend()
    for component, label in enumerate(("u", "v", "w")):
        axes[0, 1].plot(
            variances[:, component] / ustar**2,
            normalized_height,
            label=label,
        )
    axes[0, 1].set_xlabel("resolved variance / u*²")
    axes[0, 1].legend()
    axes[1, 0].plot(total_uw / ustar**2, normalized_height, label="total uw")
    axes[1, 0].plot(total_vw / ustar**2, normalized_height, label="total vw")
    axes[1, 0].set_xlabel("momentum flux / u*²")
    axes[1, 0].legend()
    axes[1, 1].plot(phi_m, normalized_height)
    axes[1, 1].axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].set_xlabel("Phi_M")
    for panel in axes.flat:
        panel.set_ylabel("z f / u*")
        panel.grid(True, alpha=0.25)
        panel.set_ylim(0.0, 0.36)
    figure.suptitle(
        f"Andren 1994 with physical-space LASD: "
        f"ft={coriolis * simulation_time:.3f}, "
        f"u*/Ug={ustar / geostrophic[0]:.4f}"
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "andren1994_profiles.png", dpi=180)
    plt.close(figure)
    print(f"[done] elapsed={elapsed:.3f}s summary={summary_path}")


if __name__ == "__main__":
    main()
