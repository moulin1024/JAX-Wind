"""Run synchronized precursor and turbine domains from a developed warmup."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import CaseConfig
from ..pressure_driven_warmup.runner import _configure_source_paths


def _block_pair(paired) -> None:
    paired.precursor.fields.velocity.x.payload.block_until_ready()
    paired.main.fields.velocity.x.payload.block_until_ready()


def _directional_cfl(state, *, dt: float, grid, jnp) -> tuple[float, float, float]:
    velocity = state.fields.velocity
    return (
        float(dt * jnp.max(jnp.abs(velocity.x.payload)) / grid.dx),
        float(dt * jnp.max(jnp.abs(velocity.y.payload)) / grid.dy),
        float(dt * jnp.max(jnp.abs(velocity.z.owned.payload)) / grid.dz),
    )


def _save_main_velocity_sample(
    path: Path,
    state,
    *,
    paired_step: int,
    dt_seconds: float,
    velocity_scale: float,
    shape: tuple[int, int, int],
    domain_metadata: dict[str, float | int],
    jax,
) -> None:
    """Atomically save one global turbine-domain velocity field in SI units."""

    accepted_step = int(state.clock.step)
    velocity = state.fields.velocity

    def physical_array(payload) -> np.ndarray:
        addressable = np.asarray(jax.device_get(payload), dtype=np.float32)
        return (addressable * np.float32(velocity_scale)).reshape(shape)

    metadata = {
        "schema": "jaxwind.concurrent-adm.main-velocity.v1",
        "domain": "main",
        "representation": "global-z-y-x",
        "units": "m/s",
        "field_locations": {
            "u_m_s": "cell",
            "v_m_s": "cell",
            "w_upper_m_s": "upper-z-face",
        },
        "paired_step": paired_step,
        "accepted_step": accepted_step,
        "wake_time_seconds": paired_step * dt_seconds,
        "physical_time_seconds": accepted_step * dt_seconds,
        "grid": domain_metadata,
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            metadata=np.asarray(json.dumps(metadata)),
            u_m_s=physical_array(velocity.x.payload),
            v_m_s=physical_array(velocity.y.payload),
            w_upper_m_s=physical_array(velocity.z.owned.payload),
        )
    os.replace(temporary, path)


def run_case(
    case: CaseConfig,
    *,
    output_dir: Path,
    restart: Path | None,
    max_steps: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Advance a live precursor beside a pure-thrust ADM turbine domain."""

    _configure_source_paths()
    import jax

    jax.config.update("jax_enable_x64", case.base.numerics.dtype == "float64")
    import jax.numpy as jnp
    from spectral_fd import runtime_from_initialized_jax

    from jaxwind.domain import (
        DistributionSpec,
        EqualZSlab,
        MeshAxis,
        MeshTopology,
        ScaleSystem,
        UniformGrid,
        VerticalBoundary,
    )
    from jaxwind.effects import (
        SideBySideStreamLauncher,
        ZSlabCheckpointLayout,
        load_boussinesq_checkpoint,
        save_boussinesq_checkpoint,
    )
    from jaxwind.integrators import (
        AB2Config,
        ConcurrentPrecursorState,
        cold_start_boussinesq,
        serial_pair,
        step_concurrent_boussinesq_precursor,
    )
    from jaxwind.interpreters.jax_zslab import build_zslab_interpreter
    from jaxwind.physics import (
        BoussinesqModel,
        BoussinesqVectorField,
        ConcurrentPrecursorFringe,
        ConcurrentPrecursorLasdAcceptedStepEvent,
        ConservativeAdvection,
        ConservativeScalarAdvection,
        DryFlowModel,
        FilteredNeutralLogWall,
        KinematicPressureGradient,
        LagrangianScaleDependentDynamic,
        LagrangianScaleDependentScalarFlux,
        LasdAcceptedStepEvent,
        NoActuatorDisk,
        NoBuoyancy,
        NoFringe,
        NoRayleighDamping,
        NoRotation,
        PureThrustActuatorDisk,
        ScalarFluxBoundary,
        WindTunnelBoussinesqVectorField,
        WindTunnelModel,
    )
    from jaxwind.pressure import build_spectral_fd_pressure_adapter

    if jax.process_count() != 1:
        raise RuntimeError("concurrent ADM currently supports one JAX process")
    shard_count = jax.device_count()
    if case.base.domain.nz % shard_count:
        raise RuntimeError("nz must be divisible by the number of JAX devices")
    addressable_shards = tuple(range(shard_count))

    output_dir.mkdir(parents=True, exist_ok=True)
    precursor_latest = output_dir / "precursor_checkpoint_latest.npz"
    main_latest = output_dir / "main_checkpoint_latest.npz"
    field_sample_interval = case.output.field_sample_every_steps
    fields_dir = output_dir / "fields"
    if (
        restart is None
        and (precursor_latest.exists() or main_latest.exists())
        and not overwrite
    ):
        raise FileExistsError(
            f"{output_dir} already contains paired checkpoints; "
            "use --restart or --overwrite"
        )
    if (
        restart is None
        and field_sample_interval is not None
        and fields_dir.is_dir()
        and next(fields_dir.glob("main_velocity_*.npz"), None) is not None
        and not overwrite
    ):
        raise FileExistsError(
            f"{fields_dir} already contains field samples; "
            "use --restart or --overwrite"
        )
    if not case.warmup.checkpoint.is_file():
        raise FileNotFoundError(case.warmup.checkpoint)
    (output_dir / "resolved_config.toml").write_text(case.resolved_toml())
    if field_sample_interval is not None:
        fields_dir.mkdir(parents=True, exist_ok=True)

    base = case.base
    physical_grid = UniformGrid(
        base.domain.nx,
        base.domain.ny,
        base.domain.nz,
        base.domain.lx_m,
        base.domain.ly_m,
        base.domain.lz_m,
    )
    scales = ScaleSystem(
        base.flow.forcing_height_m,
        base.flow.friction_velocity_m_s,
    )
    grid = scales.to_execution_grid(physical_grid)
    decomposition = EqualZSlab(
        grid,
        MeshTopology((MeshAxis("z", shard_count),)),
        DistributionSpec.z_slab(),
    )
    algebra = build_zslab_interpreter(
        decomposition,
        addressable_shards=addressable_shards,
    )
    runtime = runtime_from_initialized_jax(jax)
    pressure_arguments = {
        "decomposition": decomposition,
        "addressable_shards": addressable_shards,
        "runtime": runtime,
        "dtype": base.numerics.dtype,
        "method": base.numerics.pressure_method,
    }
    precursor_pressure = build_spectral_fd_pressure_adapter(**pressure_arguments)
    main_pressure = build_spectral_fd_pressure_adapter(**pressure_arguments)

    momentum_sgs = LagrangianScaleDependentDynamic(
        filter_grid_ratio=base.sgs.filter_grid_ratio,
        test_filter_ratio=base.sgs.test_filter_ratio,
        update_interval=base.sgs.update_interval_steps,
        timescale_coefficient=base.sgs.timescale_coefficient,
        initial_coefficient=base.sgs.initial_coefficient,
        minimum_coefficient=base.sgs.minimum_coefficient,
        maximum_coefficient=base.sgs.maximum_coefficient,
    )
    scalar_sgs = LagrangianScaleDependentScalarFlux()
    boussinesq_model = BoussinesqModel(
        DryFlowModel(
            ConservativeAdvection(),
            KinematicPressureGradient(
                scales.to_execution_acceleration(
                    base.flow.pressure_acceleration_m_s2
                )
            ),
            FilteredNeutralLogWall(
                scales.to_execution_length(base.flow.roughness_length_m),
                von_karman=base.flow.von_karman,
                filter_grid_ratio=base.wall.filter_grid_ratio,
                test_filter_ratio=base.wall.test_filter_ratio,
            ),
            momentum_sgs,
            NoRotation(),
        ),
        ConservativeScalarAdvection(),
        scalar_sgs,
        NoBuoyancy(),
        NoRayleighDamping(),
        ScalarFluxBoundary(),
    )
    base_vector_field = BoussinesqVectorField(algebra, boussinesq_model)
    precursor_vector_field = WindTunnelBoussinesqVectorField(
        algebra,
        base_vector_field,
        WindTunnelModel(NoActuatorDisk(), NoFringe()),
    )
    turbine = case.turbine
    concurrent_fringe = ConcurrentPrecursorFringe(
        scales.to_execution_length(case.fringe.start_x_m),
        scales.to_execution_time(case.fringe.relaxation_time_seconds),
        scales.to_execution_length(case.fringe.rise_width_m),
        scales.to_execution_length(case.fringe.fall_width_m),
    )
    main_vector_field = WindTunnelBoussinesqVectorField(
        algebra,
        base_vector_field,
        WindTunnelModel(
            PureThrustActuatorDisk(
                scales.to_execution_length(turbine.location_m[0]),
                scales.to_execution_length(turbine.location_m[1]),
                scales.to_execution_length(turbine.hub_height_m),
                scales.to_execution_length(turbine.diameter_m),
                turbine.local_thrust_coefficient,
                scales.to_execution_length(case.normal_smoothing_width_m),
                scales.to_execution_length(case.transverse_smoothing_width_m),
                yaw_degrees=0.0,
                filtered_velocity_correction=True,
            ),
            concurrent_fringe,
        ),
    )
    integrator_config = AB2Config(
        scales.to_execution_time(base.time.dt_seconds)
    )
    checkpoint_layout = ZSlabCheckpointLayout(
        decomposition,
        addressable_shards,
        jnp.asarray,
    )
    closure_fingerprint = momentum_sgs.fingerprint + "|" + scalar_sgs.fingerprint

    if restart is None:
        precursor = load_boussinesq_checkpoint(
            case.warmup.checkpoint,
            layout=checkpoint_layout,
            config=integrator_config,
            scale_fingerprint=scales.fingerprint,
            closure_fingerprint=closure_fingerprint,
        )
        main = cold_start_boussinesq(
            precursor.fields,
            clock=precursor.clock,
            config=integrator_config,
        )
        initial_source = str(case.warmup.checkpoint)
    else:
        restart_dir = restart
        if not restart_dir.is_dir():
            raise ValueError("--restart must name a paired checkpoint directory")
        precursor = load_boussinesq_checkpoint(
            restart_dir / "precursor_checkpoint_latest.npz",
            layout=checkpoint_layout,
            config=integrator_config,
            scale_fingerprint=scales.fingerprint,
            closure_fingerprint=closure_fingerprint,
        )
        main = load_boussinesq_checkpoint(
            restart_dir / "main_checkpoint_latest.npz",
            layout=checkpoint_layout,
            config=integrator_config,
            scale_fingerprint=scales.fingerprint,
            closure_fingerprint=closure_fingerprint,
        )
        initial_source = str(restart_dir)
    paired = ConcurrentPrecursorState(precursor, main)
    initial_step = paired.precursor.clock.step

    steps_to_run = (
        case.concurrent.steps
        if max_steps is None
        else min(case.concurrent.steps, max_steps)
    )
    boundary = lambda _clock, _environment: VerticalBoundary(0.0, 0.0)
    precursor_closure_event = LasdAcceptedStepEvent(
        algebra,
        boussinesq_model,
        integrator_config.dt,
    )
    main_closure_event = ConcurrentPrecursorLasdAcceptedStepEvent(
        algebra,
        boussinesq_model,
        integrator_config.dt,
        concurrent_fringe,
    )

    execution = case.concurrent.launch
    if execution == "auto":
        execution = "cuda-streams" if jax.default_backend() == "gpu" else "serial"
    launcher = None
    launch_pair = serial_pair
    if execution in ("threads", "cuda-streams"):
        launcher = SideBySideStreamLauncher(
            execution_streams=execution == "cuda-streams"
        )
        launch_pair = launcher

    history_path = output_dir / "history.csv"
    history_stream = history_path.open("w", newline="")
    history_fields = (
        "paired_step",
        "accepted_step",
        "physical_time_seconds",
        "precursor_cfl",
        "main_cfl",
        "lasd_trajectory_cfl",
        "precursor_maximum_divergence",
        "main_maximum_divergence",
        "maximum_main_precursor_delta_u_m_s",
        "minimum_main_precursor_delta_u_m_s",
        "elapsed_seconds",
    )
    writer = csv.DictWriter(history_stream, fieldnames=history_fields)
    writer.writeheader()

    latest: dict[str, float] = {}
    field_samples_written = 0
    started = time.perf_counter()
    try:
        for paired_step in range(1, steps_to_run + 1):
            should_log = (
                paired_step % case.output.log_every_steps == 0
                or paired_step == steps_to_run
            )
            result = step_concurrent_boussinesq_precursor(
                paired,
                config=integrator_config,
                precursor_vector_field=precursor_vector_field,
                main_vector_field=main_vector_field,
                normal_boundary=boundary,
                algebra=algebra,
                precursor_pressure_solver=precursor_pressure,
                main_pressure_solver=main_pressure,
                precursor_closure_event=precursor_closure_event,
                main_closure_event=main_closure_event,
                launch_pair=launch_pair,
                compute_projection_residual=should_log,
            )
            paired = result.state
            if (
                field_sample_interval is not None
                and paired_step % field_sample_interval == 0
            ):
                sample_path = (
                    fields_dir / f"main_velocity_{paired_step:06d}.npz"
                )
                _save_main_velocity_sample(
                    sample_path,
                    paired.main,
                    paired_step=paired_step,
                    dt_seconds=base.time.dt_seconds,
                    velocity_scale=scales.velocity,
                    shape=(
                        base.domain.nz,
                        base.domain.ny,
                        base.domain.nx,
                    ),
                    domain_metadata={
                        "nx": base.domain.nx,
                        "ny": base.domain.ny,
                        "nz": base.domain.nz,
                        "lx_m": base.domain.lx_m,
                        "ly_m": base.domain.ly_m,
                        "lz_m": base.domain.lz_m,
                    },
                    jax=jax,
                )
                field_samples_written += 1
                print(
                    f"field_sample={sample_path} "
                    f"paired_step={paired_step}/{case.concurrent.steps}",
                    flush=True,
                )
            if should_log:
                _block_pair(paired)
                precursor_cfl_components = _directional_cfl(
                    paired.precursor,
                    dt=integrator_config.dt,
                    grid=grid,
                    jnp=jnp,
                )
                main_cfl_components = _directional_cfl(
                    paired.main,
                    dt=integrator_config.dt,
                    grid=grid,
                    jnp=jnp,
                )
                precursor_cfl = max(precursor_cfl_components)
                main_cfl = max(main_cfl_components)
                maximum_cfl = max(precursor_cfl, main_cfl)
                lasd_cfl = maximum_cfl * base.sgs.update_interval_steps
                if maximum_cfl >= base.numerics.cfl_abort:
                    raise RuntimeError(
                        f"CFL abort limit reached: {maximum_cfl:.4f} >= "
                        f"{base.numerics.cfl_abort:.4f}"
                    )
                if lasd_cfl >= base.numerics.lasd_trajectory_cfl_abort:
                    raise RuntimeError(
                        f"LASD trajectory CFL abort limit reached: {lasd_cfl:.4f} >= "
                        f"{base.numerics.lasd_trajectory_cfl_abort:.4f}"
                    )
                delta_u = scales.from_execution_velocity(
                    paired.main.fields.velocity.x.payload
                    - paired.precursor.fields.velocity.x.payload
                )
                latest = {
                    "precursor_cfl": precursor_cfl,
                    "main_cfl": main_cfl,
                    "lasd_trajectory_cfl": lasd_cfl,
                    "precursor_maximum_divergence": float(
                        jnp.max(
                            jnp.abs(
                                result.diagnostic.precursor.projection.divergence.payload
                            )
                        )
                    ),
                    "main_maximum_divergence": float(
                        jnp.max(
                            jnp.abs(
                                result.diagnostic.main.projection.divergence.payload
                            )
                        )
                    ),
                    "maximum_main_precursor_delta_u_m_s": float(
                        jnp.max(delta_u)
                    ),
                    "minimum_main_precursor_delta_u_m_s": float(
                        jnp.min(delta_u)
                    ),
                }
                row = {
                    "paired_step": paired_step,
                    "accepted_step": paired.main.clock.step,
                    "physical_time_seconds": (
                        paired.main.clock.step * base.time.dt_seconds
                    ),
                    **latest,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                writer.writerow(row)
                history_stream.flush()
                print(
                    f"paired_step={paired_step}/{case.concurrent.steps} "
                    f"accepted_step={paired.main.clock.step} "
                    f"mode={execution} CFL={maximum_cfl:.3f} "
                    f"delta_u_min={latest['minimum_main_precursor_delta_u_m_s']:.4e}m/s "
                    f"elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )

            should_checkpoint = (
                paired_step % case.output.checkpoint_every_steps == 0
                or paired_step == steps_to_run
            )
            if should_checkpoint:
                save_boussinesq_checkpoint(
                    precursor_latest,
                    paired.precursor,
                    scale_fingerprint=scales.fingerprint,
                )
                save_boussinesq_checkpoint(
                    main_latest,
                    paired.main,
                    scale_fingerprint=scales.fingerprint,
                )
    finally:
        history_stream.close()
        if launcher is not None:
            launcher.close()

    _block_pair(paired)
    runtime_seconds = time.perf_counter() - started
    main_diagnostic = result.diagnostic.main.vector_field
    summary = {
        **case.resolved(),
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_devices": shard_count,
            "execution": execution,
            "initial_source": initial_source,
            "initial_step": initial_step,
            "steps_run": steps_to_run,
            "final_step": paired.main.clock.step,
            "runtime_seconds": runtime_seconds,
            "field_samples_written": field_samples_written,
            "lasd_closure_fringe_enabled": True,
            "actuator_disk_enabled": main_diagnostic.actuator_disk_enabled,
            "concurrent_fringe_enabled": main_diagnostic.concurrent_fringe_enabled,
            **latest,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary
