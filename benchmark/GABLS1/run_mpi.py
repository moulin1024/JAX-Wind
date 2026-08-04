#!/usr/bin/env python3
"""Run GABLS1 on a true multi-process non-spectral y-slab decomposition."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
from mpi4py import MPI


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
for source in (ROOT, SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


from benchmark.GABLS1 import diagnostics  # noqa: E402
from benchmark.GABLS1 import run as serial_run  # noqa: E402


CHECKPOINT_SCHEMA = serial_run.CHECKPOINT_SCHEMA


def parse_args(argv: list[str] | None = None):
    args = serial_run.parse_args(argv)
    if args.mesh is not None:
        raise SystemExit(
            "the y-slab runner does not yet support stretched mesh artifacts"
        )
    if args.wall_matching_height is not None:
        raise SystemExit(
            "the y-slab runner does not yet support a custom wall matching height"
        )
    if args.rayleigh_sponge_start_height is not None:
        raise SystemExit(
            "the y-slab runner does not yet support the Rayleigh sponge"
        )
    if args.quick:
        # Four MP5 ranks require at least three owned y cells per rank.
        args.nx = args.ny = args.nz = 16
    return args


def _coordinator_address(communicator) -> str:
    configured = os.environ.get("JAXWIND_COORDINATOR_ADDRESS")
    if configured:
        return configured
    address = None
    if communicator.Get_rank() == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            address = f"127.0.0.1:{listener.getsockname()[1]}"
    return communicator.bcast(address, root=0)


def _initialize_jax(communicator):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    import jax

    jax.distributed.initialize(
        coordinator_address=_coordinator_address(communicator),
        num_processes=communicator.Get_size(),
        process_id=communicator.Get_rank(),
        local_device_ids=[0],
    )
    return jax


def _build_distributed(args, jax):
    import jax.numpy as jnp

    from benchmark.GABLS1.distributed_solver import YSlabAMDBoussinesq
    from jaxwind.pressure import (
        BoundaryCondition,
        GMGConfig,
        PCGConfig,
        PoissonBoundaryConditions,
        RectilinearGrid,
        YSlabConfig,
        YSlabMatrixFreePoissonSolver,
    )

    case = diagnostics.GABLS1Case()
    dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    grid = RectilinearGrid.uniform(
        args.nx,
        args.ny,
        args.nz,
        lx=case.domain,
        ly=case.domain,
        lz=case.domain,
    )
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    pressure = YSlabMatrixFreePoissonSolver(
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
            pre_smooth=args.pressure_smooth,
            post_smooth=args.pressure_smooth,
            coarse_smooth=args.pressure_coarse_smooth,
        ),
        krylov=PCGConfig(
            max_iterations=args.pressure_max_iterations,
            relative_tolerance=args.pressure_rtol,
            execution="jax",
        ),
        distribution=YSlabConfig(
            coarse_cells_per_device=args.y_slab_coarse_cells_per_rank
        ),
    )
    coupled = YSlabAMDBoussinesq(
        grid,
        pressure,
        geostrophic_wind=(case.geostrophic_u, case.geostrophic_v),
        coriolis=case.coriolis,
        roughness_length=case.roughness_length,
        gravity=case.gravity,
        reference_potential_temperature=case.theta_reference,
        surface_potential_temperature=case.theta_initial,
        surface_temperature_tendency=case.surface_cooling_rate,
        amd_coefficient=args.amd_coefficient,
        scalar_amd_coefficient=args.scalar_amd_coefficient,
        mp5_strength=args.mp5_strength,
        coupling_integrator=args.coupling_integrator,
    )
    return coupled, case, dtype


def _build_serial_observer(args, case, dtype):
    from jaxwind.momentum import (
        AMDBoussinesq,
        AMDBoussinesqConfig,
        AMDModel,
        AMDPassiveScalar,
        AMDPassiveScalarModel,
        NeutralABLConfig,
        NeutralABLMomentum,
    )
    from jaxwind.pressure import (
        BoundaryCondition,
        GMGConfig,
        MatrixFreePoissonSolver,
        PCGConfig,
        PoissonBoundaryConditions,
        RectilinearGrid,
    )

    grid = RectilinearGrid.uniform(
        args.nx,
        args.ny,
        args.nz,
        lx=case.domain,
        ly=case.domain,
        lz=case.domain,
    )
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
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
            pre_smooth=args.pressure_smooth,
            post_smooth=args.pressure_smooth,
            coarse_smooth=args.pressure_coarse_smooth,
        ),
        krylov=PCGConfig(
            max_iterations=args.pressure_max_iterations,
            relative_tolerance=args.pressure_rtol,
            execution="jax",
        ),
    )
    momentum = NeutralABLMomentum(
        grid,
        pressure,
        NeutralABLConfig(
            friction_velocity=0.3,
            roughness_length=case.roughness_length,
            pressure_acceleration=0.0,
            geostrophic_wind=(case.geostrophic_u, case.geostrophic_v),
            coriolis_vertical=case.coriolis,
            coriolis_horizontal=0.0,
            mp5_dissipation_strength=args.mp5_strength,
            amd=AMDModel(coefficient=args.amd_coefficient),
            sgs_time_integration="explicit",
        ),
    )
    scalar = AMDPassiveScalar(
        grid,
        AMDPassiveScalarModel(
            coefficient=args.scalar_amd_coefficient,
            lower_surface_flux=0.0,
            upper_surface_flux=0.0,
            mp5_dissipation_strength=args.mp5_strength,
        ),
    )
    return AMDBoussinesq(
        momentum,
        scalar,
        AMDBoussinesqConfig(
            gravity=case.gravity,
            reference_potential_temperature=case.theta_reference,
            surface_potential_temperature=case.theta_initial,
            surface_temperature_tendency=case.surface_cooling_rate,
            thermal_roughness_length=case.roughness_length,
            coupling_integrator=args.coupling_integrator,
        ),
    )


def _local_initial_state(args, coupled, case, dtype, rank: int):
    import jax
    import jax.numpy as jnp

    from jaxwind.pressure import YSlabMACVelocity

    local_y = args.ny // coupled.device_count
    start = rank * local_y
    z = (jnp.arange(args.nz, dtype=dtype) + 0.5) * coupled.dz
    profile = jnp.where(
        z <= case.inversion_base,
        case.theta_initial,
        case.theta_initial + case.inversion_gradient * (z - case.inversion_base),
    )
    perturbation = jax.random.uniform(
        jax.random.PRNGKey(args.seed),
        (args.nz, args.ny, args.nx),
        dtype=dtype,
        minval=-0.1,
        maxval=0.1,
    )
    perturbation -= jnp.mean(perturbation, axis=(1, 2), keepdims=True)
    perturbation *= (z < 50.0)[:, None, None]
    theta = (profile[:, None, None] + perturbation)[:, start : start + local_y, :][None]
    velocity = YSlabMACVelocity(
        jnp.full(
            (1, args.nz, local_y, args.nx + 1),
            case.geostrophic_u,
            dtype=dtype,
        ),
        jnp.full(
            (1, args.nz, local_y + 1, args.nx),
            case.geostrophic_v,
            dtype=dtype,
        ),
        jnp.zeros(
            (1, args.nz + 1, local_y, args.nx),
            dtype=dtype,
        ),
    )
    return coupled.initial_state(velocity, theta)


def _validate_checkpoint(args, checkpoint, coupled) -> None:
    if str(checkpoint["checkpoint_schema"]) != CHECKPOINT_SCHEMA:
        raise SystemExit("restart checkpoint schema is not supported")
    if not np.array_equal(checkpoint["shape_zyx"], coupled.grid.shape):
        raise SystemExit("restart grid shape does not match")
    for key in ("x_faces", "y_faces", "z_faces"):
        if key in checkpoint and not np.array_equal(
            np.asarray(checkpoint[key]),
            np.asarray(getattr(coupled.grid, key)),
        ):
            raise SystemExit(f"restart {key} do not match the active mesh")
    for key in ("amd_coefficient", "scalar_amd_coefficient", "mp5_strength"):
        if not np.isclose(float(checkpoint[key]), getattr(args, key)):
            raise SystemExit(f"restart {key} does not match")
    checkpoint_limiter = (
        str(checkpoint["advection_limiter"])
        if "advection_limiter" in checkpoint
        else "mp5"
    )
    if checkpoint_limiter != "mp5":
        raise SystemExit("cannot restart a non-MP5 advection checkpoint")


def _local_restart_state(args, coupled, dtype, rank: int):
    import jax.numpy as jnp

    from jaxwind.pressure import YSlabMACVelocity

    checkpoint = np.load(args.restart, allow_pickle=False)
    _validate_checkpoint(args, checkpoint, coupled)
    local_y = args.ny // coupled.device_count
    start = rank * local_y
    end = start + local_y
    velocity = YSlabMACVelocity(
        jnp.asarray(checkpoint["velocity_x"][:, start:end, :], dtype=dtype)[None],
        jnp.asarray(checkpoint["velocity_y"][:, start : end + 1, :], dtype=dtype)[None],
        jnp.asarray(checkpoint["velocity_z"][:, start:end, :], dtype=dtype)[None],
    )
    state = coupled.initial_state(
        velocity,
        jnp.asarray(
            checkpoint["potential_temperature"][:, start:end, :],
            dtype=dtype,
        )[None],
        pressure=jnp.asarray(checkpoint["pressure"][:, start:end, :], dtype=dtype)[
            None
        ],
        time=float(checkpoint["time"]),
        step=int(checkpoint["step"]),
        project=False,
    )
    return state, checkpoint


def _assemble_y_faces(pieces: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(
        tuple(piece[:, :-1, :] for piece in pieces[:-1]) + (pieces[-1],),
        axis=1,
    )


def _gather_global_state(communicator, state, observer):
    import jax.numpy as jnp

    from jaxwind.pressure import MACVelocity

    rank = communicator.Get_rank()
    x_parts = communicator.gather(np.asarray(state.velocity.x[0]), root=0)
    y_parts = communicator.gather(np.asarray(state.velocity.y[0]), root=0)
    z_parts = communicator.gather(np.asarray(state.velocity.z[0]), root=0)
    theta_parts = communicator.gather(
        np.asarray(state.potential_temperature[0]),
        root=0,
    )
    pressure_parts = communicator.gather(np.asarray(state.pressure[0]), root=0)
    if rank != 0:
        return None
    velocity = MACVelocity(
        jnp.asarray(np.concatenate(x_parts, axis=1)),
        jnp.asarray(_assemble_y_faces(y_parts)),
        jnp.asarray(np.concatenate(z_parts, axis=1)),
    )
    return observer.initial_state(
        velocity,
        jnp.asarray(np.concatenate(theta_parts, axis=1)),
        pressure=jnp.asarray(np.concatenate(pressure_parts, axis=1)),
        time=state.time,
        step=state.step,
    )


def run(args) -> dict[str, float | int | str] | None:
    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    size = communicator.Get_size()
    if size != 4:
        raise SystemExit("GABLS1 y-slab runner currently requires exactly 4 ranks")
    if args.ny % size or args.ny // size < 3:
        raise SystemExit("ny must divide four ranks with at least 3 cells per slab")

    jax = _initialize_jax(communicator)
    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    coupled, case, dtype = _build_distributed(args, jax)
    observer = _build_serial_observer(args, case, dtype) if rank == 0 else None
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    communicator.Barrier()

    if args.restart is None:
        state = _local_initial_state(args, coupled, case, dtype, rank)
        samples: list[dict] = []
        time_rows: list[dict] = []
        timesteps: list[float] = []
        max_cfl = 0.0
        max_diffusive_cfl = 0.0
        max_divergence = 0.0
        max_scalar_budget_residual = 0.0
    else:
        state, checkpoint = _local_restart_state(args, coupled, dtype, rank)
        if rank == 0:
            samples = serial_run._unpack_records(checkpoint, "samples")
            time_rows = serial_run._unpack_records(checkpoint, "time_rows")
            timesteps = list(np.asarray(checkpoint["timesteps"], dtype=float))
            max_cfl = float(checkpoint["max_cfl"])
            max_diffusive_cfl = float(checkpoint["max_diffusive_cfl"])
            max_divergence = float(checkpoint["max_divergence"])
            max_scalar_budget_residual = float(checkpoint["max_scalar_budget_residual"])
        else:
            samples = []
            time_rows = []
            timesteps = []
            max_cfl = max_diffusive_cfl = max_divergence = 0.0
            max_scalar_budget_residual = 0.0

    compile_start = time.perf_counter()
    compiled = coupled.step(state, timestep=min(args.dt_max, 0.25))
    jax.block_until_ready(compiled.velocity.x)
    compilation_s = time.perf_counter() - compile_start
    if rank == 0:
        print(
            f"[compile] four-rank GABLS1 y-slab ready in {compilation_s:.3f}s",
            flush=True,
        )

    final_time = args.end_hours * 3600.0
    sample_start_time = args.sample_start_hours * 3600.0
    next_sample_time = (
        math.floor(state.time / args.sample_interval_seconds) + 1
    ) * args.sample_interval_seconds
    start_simulation_time = state.time
    start_wall = time.perf_counter()
    stopped_early = False
    cached_global_step = -1
    cached_global_state = None

    def global_state():
        nonlocal cached_global_step, cached_global_state
        if cached_global_step != state.step:
            cached_global_state = _gather_global_state(
                communicator,
                state,
                observer,
            )
            cached_global_step = state.step
        return cached_global_state

    def save_checkpoint() -> None:
        observed = global_state()
        if rank != 0:
            return
        payload: dict[str, object] = {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "parallel_layout": "mpi-y-slab",
            "parallel_processes": size,
            "shape_zyx": np.asarray(coupled.grid.shape),
            "x_faces": np.asarray(coupled.grid.x_faces),
            "y_faces": np.asarray(coupled.grid.y_faces),
            "z_faces": np.asarray(coupled.grid.z_faces),
            "velocity_x": np.asarray(observed.velocity.x),
            "velocity_y": np.asarray(observed.velocity.y),
            "velocity_z": np.asarray(observed.velocity.z),
            "potential_temperature": np.asarray(observed.potential_temperature),
            "pressure": np.asarray(observed.pressure),
            "time": observed.time,
            "step": observed.step,
            "timesteps": np.asarray(timesteps),
            "amd_coefficient": args.amd_coefficient,
            "scalar_amd_coefficient": args.scalar_amd_coefficient,
            "mp5_strength": args.mp5_strength,
            "advection_limiter": "mp5",
            "max_cfl": max_cfl,
            "max_diffusive_cfl": max_diffusive_cfl,
            "max_divergence": max_divergence,
            "max_scalar_budget_residual": max_scalar_budget_residual,
        }
        serial_run._pack_records(payload, "samples", samples)
        serial_run._pack_records(payload, "time_rows", time_rows)
        destination = args.output_dir / "checkpoint.npz"
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, destination)

    def append_diagnostics() -> None:
        observed = global_state()
        if rank != 0:
            return
        statistics = diagnostics.snapshot_statistics(observer, observed)
        nonfinite = [
            key
            for key, value in statistics.items()
            if not np.all(np.isfinite(value)) and key != "obukhov_length"
        ]
        if nonfinite:
            raise FloatingPointError("non-finite diagnostic: " + ", ".join(nonfinite))
        time_rows.append(
            {
                "step": float(state.step),
                "time_s": state.time,
                "time_hours": state.time / 3600.0,
                "boundary_layer_height": float(statistics["boundary_layer_height"]),
                "surface_temperature": case.surface_temperature(state.time),
                "surface_heat_flux": float(statistics["surface_heat_flux"]),
                "friction_velocity": float(statistics["friction_velocity"]),
                "obukhov_length": float(statistics["obukhov_length"]),
                "maximum_abs_w": float(statistics["maximum_abs_w"]),
                "jet_speed": float(statistics["jet_speed"]),
                "jet_height": float(statistics["jet_height"]),
            }
        )
        if state.time >= sample_start_time:
            samples.append(statistics)

    while state.time < final_time:
        advective_rate, momentum_rate, scalar_rate = coupled.rates(state)
        timestep = min(
            args.target_cfl / advective_rate,
            (
                args.target_diffusive_cfl / momentum_rate
                if momentum_rate > 0.0
                else math.inf
            ),
            (
                args.target_diffusive_cfl / scalar_rate
                if scalar_rate > 0.0
                else math.inf
            ),
            args.dt_max,
            final_time - state.time,
        )
        next_step = state.step + 1
        final_after_step = state.time + timestep >= final_time - 1.0e-12
        max_steps_after_step = (
            args.max_steps is not None and next_step >= args.max_steps
        )
        metrics_due = (
            next_step % args.metrics_every == 0
            or next_step % args.log_every == 0
            or next_step % args.checkpoint_every == 0
            or final_after_step
            or max_steps_after_step
        )
        if metrics_due:
            theta_local_before = np.asarray(state.potential_temperature[0])
            theta_sum_before = communicator.allreduce(
                float(np.sum(theta_local_before, dtype=np.float64)),
                op=MPI.SUM,
            )
            if args.coupling_integrator == "strang":
                flux_before = coupled.surface_layer_fluxes(state)
                heat_sum_before = communicator.allreduce(
                    float(np.sum(np.asarray(flux_before.heat_flux))),
                    op=MPI.SUM,
                )
        state = coupled.step(state, timestep=timestep)
        cached_global_step = -1
        if metrics_due:
            flux_after = coupled.surface_layer_fluxes(state)
            heat_sum_after = communicator.allreduce(
                float(np.sum(np.asarray(flux_after.heat_flux))),
                op=MPI.SUM,
            )
            theta_sum_after = communicator.allreduce(
                float(
                    np.sum(
                        np.asarray(state.potential_temperature[0]),
                        dtype=np.float64,
                    )
                ),
                op=MPI.SUM,
            )
            theta_before = theta_sum_before / (args.nx * args.ny * args.nz)
            theta_after = theta_sum_after / (args.nx * args.ny * args.nz)
            heat_flux_after = heat_sum_after / (args.nx * args.ny)
            if coupled.last_surface_heat_flux_quadrature is None:
                heat_flux_before = heat_sum_before / (args.nx * args.ny)
                integrated_heat_flux = 0.5 * (heat_flux_before + heat_flux_after)
            else:
                integrated_heat_flux = float(
                    np.mean(np.asarray(coupled.last_surface_heat_flux_quadrature))
                )
            budget_residual = abs(
                theta_after
                - theta_before
                - timestep * integrated_heat_flux / case.domain
            )
            divergence = coupled.divergence_norm(state.velocity)
        cfl = timestep * advective_rate
        diffusive_cfl = timestep * max(momentum_rate, scalar_rate)
        if rank == 0:
            timesteps.append(timestep)
            max_cfl = max(max_cfl, cfl)
            max_diffusive_cfl = max(max_diffusive_cfl, diffusive_cfl)
            if metrics_due:
                max_divergence = max(max_divergence, divergence)
                max_scalar_budget_residual = max(
                    max_scalar_budget_residual,
                    budget_residual,
                )
        final = state.time >= final_time
        if state.time + 1.0e-9 >= next_sample_time or final:
            append_diagnostics()
            while next_sample_time <= state.time + 1.0e-9:
                next_sample_time += args.sample_interval_seconds
        if rank == 0 and (state.step % args.log_every == 0 or final):
            print(
                f"step={state.step} time={state.time / 3600.0:.4f}/"
                f"{args.end_hours:g}h CFL={cfl:.4f} "
                f"CFLnu={diffusive_cfl:.4f} divL2={divergence:.3e} "
                f"Q0={heat_flux_after:.3e} "
                f"theta_budget={budget_residual:.3e} "
                + serial_run._eta_log_fields(
                    start_wall=start_wall,
                    start_simulation_time=start_simulation_time,
                    simulation_time=state.time,
                    final_simulation_time=final_time,
                ),
                flush=True,
            )
        if state.step % args.checkpoint_every == 0 or final:
            save_checkpoint()
        if args.max_steps is not None and state.step >= args.max_steps:
            stopped_early = state.time < final_time
            if rank == 0:
                need_diagnostic = not time_rows or time_rows[-1]["step"] != float(
                    state.step
                )
            else:
                need_diagnostic = None
            need_diagnostic = communicator.bcast(need_diagnostic, root=0)
            if need_diagnostic:
                append_diagnostics()
            save_checkpoint()
            break
        if (
            args.max_run_seconds is not None
            and time.perf_counter() - start_wall >= args.max_run_seconds
        ):
            stopped_early = True
            save_checkpoint()
            break

    if rank == 0 and not samples:
        need_sample = True
    else:
        need_sample = False
    need_sample = communicator.bcast(need_sample, root=0)
    if need_sample:
        observed = global_state()
        if rank == 0:
            samples.append(diagnostics.snapshot_statistics(observer, observed))

    runtime_s = time.perf_counter() - start_wall
    summary = None
    if rank == 0:
        reference_dir = args.reference_dir if args.reference_dir.exists() else None
        summary = diagnostics.save_outputs(
            args.output_dir,
            samples=samples,
            time_rows=time_rows,
            reference_dir=reference_dir,
            metadata={
                "solver": "non-spectral MAC + multi-host y-slab GMG/PCG",
                "sgs_model": "AMD",
                "parallel_layout": "1x4 y-slab",
                "parallel_processes": size,
                "nx": args.nx,
                "ny": args.ny,
                "nz": args.nz,
                "grid_spacing_m": case.domain / args.nx,
                "end_time_hours": state.time / 3600.0,
                "runtime_s": runtime_s,
                "compilation_s": compilation_s,
                "stopped_early": str(stopped_early).lower(),
                "max_cfl": max_cfl,
                "max_diffusive_cfl": max_diffusive_cfl,
                "max_divergence": max_divergence,
                "max_scalar_budget_residual": max_scalar_budget_residual,
                "accepted_metrics_interval_steps": args.metrics_every,
                "amd_coefficient": args.amd_coefficient,
                "scalar_amd_coefficient": args.scalar_amd_coefficient,
                "mp5_dissipation_strength": args.mp5_strength,
                "advection_dissipation_strength": args.mp5_strength,
                "advection_limiter": "mp5",
                "projection_method": "full",
                "coupling_integrator": args.coupling_integrator,
                "pressure_smooth": args.pressure_smooth,
            },
        )
        resolved = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        resolved["parallel_layout"] = "1x4 y-slab"
        (args.output_dir / "resolved_config.json").write_text(
            json.dumps(resolved, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    communicator.Barrier()
    jax.distributed.shutdown()
    return summary


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
