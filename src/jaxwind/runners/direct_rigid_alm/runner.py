"""Execute a cold neutral-flow rigid actuator-line smoke case."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from ..pressure_driven_warmup.runner import _configure_source_paths
from .diagnostics import (
    _blade_positions_m,
    _capture_flow_frame,
    _diagnostics,
    _initial_velocity,
    _save_flow_frames,
)
from .models import CaseConfig


def run_case(
    case: CaseConfig,
    *,
    output_dir: Path,
    restart: Path | None,
    max_steps: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Run a cold-start rigid or modal actuator line without a warmup."""

    if restart is not None:
        raise ValueError("direct_rigid_alm does not accept a warmup or restart")
    _configure_source_paths()
    import jax

    jax.config.update("jax_enable_x64", case.numerics.dtype == "float64")
    import jax.numpy as jnp
    from spectral_fd import runtime_from_initialized_jax

    from jaxwind.domain import (
        AcceptedClock,
        DistributionSpec,
        EqualZSlab,
        MeshAxis,
        MeshTopology,
        ScaleSystem,
        UniformGrid,
        VerticalBoundary,
    )
    from jaxwind.integrators import AB2Config, cold_start, step
    from jaxwind.interpreters.jax_zslab import build_zslab_interpreter
    from jaxwind.openfast import build_modal_blade_model
    from jaxwind.physics import (
        ConservativeAdvection,
        DryFlowModel,
        DryFlowVectorField,
        KinematicPressureGradient,
        NeutralLogWall,
        NoRotation,
        StaticSmagorinsky,
        WindTunnelModel,
        WindTunnelVectorField,
    )
    from jaxwind.pressure import build_spectral_fd_pressure_adapter

    if jax.process_count() != 1:
        raise RuntimeError(
            "direct ALM currently supports one JAX process"
        )
    shard_count = jax.device_count()
    if case.domain.nz % shard_count:
        raise RuntimeError("nz must be divisible by the JAX device count")
    addressable_shards = tuple(range(shard_count))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"{summary_path} exists; set execution.overwrite = true "
            "in the TOML configuration"
        )
    (output_dir / "resolved_config.toml").write_text(
        case.resolved_toml()
    )

    physical_grid = UniformGrid(
        case.domain.nx,
        case.domain.ny,
        case.domain.nz,
        case.domain.lx_m,
        case.domain.ly_m,
        case.domain.lz_m,
    )
    scales = ScaleSystem(
        case.flow.forcing_height_m,
        case.flow.friction_velocity_m_s,
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
    pressure_solver = build_spectral_fd_pressure_adapter(
        decomposition,
        addressable_shards=addressable_shards,
        runtime=runtime_from_initialized_jax(jax),
        dtype=case.numerics.dtype,
        method=case.numerics.pressure_method,
    )
    flow_model = DryFlowModel(
        ConservativeAdvection(),
        KinematicPressureGradient(
            scales.to_execution_acceleration(
                case.flow.pressure_acceleration_m_s2
            )
        ),
        NeutralLogWall(
            scales.to_execution_length(
                case.flow.roughness_length_m
            ),
            von_karman=case.flow.von_karman,
        ),
        StaticSmagorinsky(case.sgs.coefficient),
        NoRotation(),
    )
    base_line = case.turbine.openfast.to_actuator_line(
        scales=scales,
        x_m=case.turbine.location_m[0],
        y_m=case.turbine.location_m[1],
        smoothing_width_m=case.turbine.smoothing_width_m,
        hub_height_m=case.turbine.hub_height_m,
        rotor_speed_rpm=case.turbine.rotor_speed_rpm,
        pitch_degrees=case.turbine.pitch_degrees,
        yaw_degrees=case.turbine.yaw_degrees,
        initial_azimuth_degrees=(
            case.turbine.initial_azimuth_degrees
        ),
    )
    dry_vector_field = DryFlowVectorField(algebra, flow_model)
    modal_model = None
    modal_state = None
    modal_openfast = case.turbine.modal_openfast
    if case.aeroelastic.enabled:
        if modal_openfast is None:
            raise RuntimeError("aeroelastic case has no modal OpenFAST data")
        modal_model = build_modal_blade_model(
            modal_openfast,
            element_radii_m=case.turbine.openfast.element_radii_m,
            element_widths_m=case.turbine.openfast.element_widths_m,
            rotor_speed_rpm=case.turbine.rotor_speed_rpm,
        )
        modal_state = modal_model.initial_state()
    integrator_config = AB2Config(
        scales.to_execution_time(case.time.dt_seconds)
    )
    initial = _initial_velocity(
        jnp=jnp,
        case=case,
        decomposition=decomposition,
        addressable_shards=addressable_shards,
        scales=scales,
    )
    state = cold_start(
        initial,
        clock=AcceptedClock(0.0, 0),
        config=integrator_config,
    )
    vector_field = WindTunnelVectorField(
        algebra,
        dry_vector_field,
        WindTunnelModel(actuator_line=base_line),
    )
    frame_interval = case.output.flow_slice_every_steps
    frame_times: list[float] = []
    rotor_frames: list[np.ndarray] = []
    hub_frames: list[np.ndarray] = []
    blade_position_frames: list[np.ndarray] = []
    if frame_interval is not None:
        rotor_frame, hub_frame = _capture_flow_frame(
            state,
            case=case,
            scales=scales,
            jax=jax,
            jnp=jnp,
        )
        frame_times.append(0.0)
        rotor_frames.append(rotor_frame)
        hub_frames.append(hub_frame)
        blade_position_frames.append(
            _blade_positions_m(
                case,
                modal_model=modal_model,
                modal_state=modal_state,
                time_seconds=0.0,
            )
        )
    steps_to_run = (
        case.time.steps
        if max_steps is None
        else min(case.time.steps, max_steps)
    )
    boundary = lambda _clock, _environment: VerticalBoundary(0.0, 0.0)
    history_path = output_dir / "history.csv"
    history_stream = history_path.open("w", newline="")
    fieldnames = (
        "step",
        "time_seconds",
        "cfl_x",
        "cfl_y",
        "cfl_z",
        "maximum_cfl",
        "maximum_execution_divergence",
        "maximum_divergence_s_inv",
        "fields_finite",
        "maximum_u_m_s",
        "maximum_v_m_s",
        "maximum_w_m_s",
        "rotor_thrust_n",
        "rotor_torque_n_m",
        "maximum_tip_deflection_m",
        "maximum_flap_tip_deflection_m",
        "maximum_edge_tip_deflection_m",
        "maximum_modal_velocity_m_s",
        "elapsed_seconds",
    )
    writer = csv.DictWriter(history_stream, fieldnames=fieldnames)
    writer.writeheader()

    started = time.perf_counter()
    latest: dict[str, float | bool] = {}
    latest_aeroelastic: dict[str, float] = {
        "rotor_thrust_n": 0.0,
        "rotor_torque_n_m": 0.0,
        "maximum_tip_deflection_m": 0.0,
        "maximum_flap_tip_deflection_m": 0.0,
        "maximum_edge_tip_deflection_m": 0.0,
        "maximum_modal_velocity_m_s": 0.0,
    }
    modal_times_seconds: list[float] = []
    modal_displacements_m: list[np.ndarray] = []
    modal_velocities_m_s: list[np.ndarray] = []
    modal_generalized_forces_n: list[np.ndarray] = []
    if modal_state is not None:
        modal_times_seconds.append(0.0)
        modal_displacements_m.append(modal_state.displacement_m.copy())
        modal_velocities_m_s.append(modal_state.velocity_m_s.copy())
        modal_generalized_forces_n.append(
            np.zeros_like(modal_state.displacement_m)
        )
    line_enabled = False
    try:
        for local_step in range(1, steps_to_run + 1):
            generalized_force_n = None
            if modal_model is not None and modal_state is not None:
                line = modal_model.deform_actuator_line(
                    base_line,
                    modal_state,
                    scales=scales,
                )
                line_diagnostic = algebra.actuator_line_diagnostic(
                    state.velocity,
                    line,
                    time=state.clock.time,
                )
                (
                    force_per_density,
                    positions,
                    tangents,
                    normals,
                ) = jax.device_get(
                    (
                        line_diagnostic.force_on_fluid_per_density,
                        line_diagnostic.positions,
                        line_diagnostic.tangents,
                        line_diagnostic.normals,
                    )
                )
                force_scale = (
                    scales.velocity**2 * scales.length**2
                )
                force_on_blade_n = (
                    -case.aeroelastic.air_density_kg_m3
                    * np.asarray(force_per_density)
                    * force_scale
                )
                generalized_force_n = modal_model.generalized_forces(
                    force_on_blade_n,
                    np.asarray(normals),
                    np.asarray(tangents),
                    gravity_m_s2=case.aeroelastic.gravity_m_s2,
                )
                tilt = math.radians(base_line.tilt_degrees)
                yaw = math.radians(base_line.yaw_degrees)
                rotor_normal = np.asarray(
                    (
                        math.cos(tilt) * math.cos(yaw),
                        math.cos(tilt) * math.sin(yaw),
                        math.sin(tilt),
                    )
                )
                physical_positions = (
                    np.asarray(positions) * scales.length
                )
                hub = np.asarray(
                    (
                        base_line.x,
                        base_line.y,
                        base_line.z,
                    )
                ) * scales.length
                moments = np.cross(
                    physical_positions - hub,
                    force_on_blade_n,
                )
                latest_aeroelastic["rotor_thrust_n"] = float(
                    np.sum(force_on_blade_n @ rotor_normal)
                )
                latest_aeroelastic["rotor_torque_n_m"] = float(
                    np.sum(moments @ rotor_normal)
                )
                vector_field = WindTunnelVectorField(
                    algebra,
                    dry_vector_field,
                    WindTunnelModel(actuator_line=line),
                )
            result = step(
                state,
                config=integrator_config,
                environment=None,
                vector_field=vector_field,
                normal_boundary=boundary,
                algebra=algebra,
                pressure_solver=pressure_solver,
                compute_projection_residual=True,
            )
            state = result.state
            state.velocity.x.payload.block_until_ready()
            latest = _diagnostics(
                state,
                divergence=result.diagnostic.projection.divergence.payload,
                case=case,
                scales=scales,
                jnp=jnp,
            )
            line_enabled = (
                result.diagnostic.vector_field.actuator_line_enabled
            )
            if not latest["fields_finite"]:
                raise RuntimeError("non-finite velocity after ALM smoke step")
            if latest["maximum_cfl"] >= case.numerics.cfl_abort:
                raise RuntimeError(
                    "CFL abort limit reached: "
                    f"{latest['maximum_cfl']:.4f} >= "
                    f"{case.numerics.cfl_abort:.4f}"
                )
            if (
                modal_model is not None
                and modal_state is not None
                and generalized_force_n is not None
            ):
                modal_state, modal_diagnostic = modal_model.advance(
                    modal_state,
                    generalized_force_n,
                    dt_seconds=case.time.dt_seconds,
                )
                latest_aeroelastic.update(
                    {
                        "maximum_tip_deflection_m": (
                            modal_diagnostic.maximum_tip_deflection_m
                        ),
                        "maximum_flap_tip_deflection_m": float(
                            np.max(
                                np.abs(
                                    modal_diagnostic.flap_tip_deflection_m
                                )
                            )
                        ),
                        "maximum_edge_tip_deflection_m": float(
                            np.max(
                                np.abs(
                                    modal_diagnostic.edge_tip_deflection_m
                                )
                            )
                        ),
                        "maximum_modal_velocity_m_s": float(
                            np.max(np.abs(modal_state.velocity_m_s))
                        ),
                    }
                )
                if (
                    not np.all(np.isfinite(modal_state.displacement_m))
                    or not np.all(np.isfinite(modal_state.velocity_m_s))
                ):
                    raise RuntimeError(
                        "non-finite modal state after aeroelastic step"
                    )
                if (
                    modal_diagnostic.maximum_tip_deflection_m
                    >= case.aeroelastic.maximum_tip_deflection_m
                ):
                    raise RuntimeError(
                        "tip-deflection abort limit reached: "
                        f"{modal_diagnostic.maximum_tip_deflection_m:.3f} >= "
                        f"{case.aeroelastic.maximum_tip_deflection_m:.3f} m"
                    )
                modal_times_seconds.append(
                    state.clock.step * case.time.dt_seconds
                )
                modal_displacements_m.append(
                    modal_state.displacement_m.copy()
                )
                modal_velocities_m_s.append(
                    modal_state.velocity_m_s.copy()
                )
                modal_generalized_forces_n.append(
                    generalized_force_n.copy()
                )
            accepted_step = state.clock.step
            if frame_interval is not None and (
                accepted_step % frame_interval == 0
                or local_step == steps_to_run
            ):
                rotor_frame, hub_frame = _capture_flow_frame(
                    state,
                    case=case,
                    scales=scales,
                    jax=jax,
                    jnp=jnp,
                )
                frame_times.append(
                    accepted_step * case.time.dt_seconds
                )
                rotor_frames.append(rotor_frame)
                hub_frames.append(hub_frame)
                blade_position_frames.append(
                    _blade_positions_m(
                        case,
                        modal_model=modal_model,
                        modal_state=modal_state,
                        time_seconds=(
                            accepted_step * case.time.dt_seconds
                        ),
                    )
                )
            if (
                accepted_step % case.output.log_every_steps == 0
                or local_step == steps_to_run
            ):
                row = {
                    "step": accepted_step,
                    "time_seconds": (
                        accepted_step * case.time.dt_seconds
                    ),
                    **latest,
                    **latest_aeroelastic,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                writer.writerow(row)
                history_stream.flush()
                print(
                    f"step={accepted_step}/{case.time.steps} "
                    f"time={row['time_seconds']:.3f}s "
                    f"CFL={latest['maximum_cfl']:.4f} "
                    f"div={latest['maximum_execution_divergence']:.3e} "
                    f"tip={latest_aeroelastic['maximum_tip_deflection_m']:.3f}m "
                    f"elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
    finally:
        history_stream.close()

    flow_frames_path: Path | None = None
    if frame_times:
        flow_frames_path = output_dir / "flow_slices.npz"
        _save_flow_frames(
            flow_frames_path,
            case=case,
            times_seconds=frame_times,
            rotor_planes=rotor_frames,
            hub_planes=hub_frames,
            blade_positions_m=blade_position_frames,
        )
    modal_history_path: Path | None = None
    if modal_model is not None and modal_state is not None:
        modal_history_path = output_dir / "aeroelastic_history.npz"
        temporary = modal_history_path.with_name(
            f".{modal_history_path.name}.tmp"
        )
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema=np.asarray(
                    "jaxwind.openfast-modal-aeroelastic-history.v1"
                ),
                times_seconds=np.asarray(modal_times_seconds),
                mode_names=np.asarray(modal_model.mode_names),
                displacement_m=np.stack(modal_displacements_m),
                velocity_m_s=np.stack(modal_velocities_m_s),
                generalized_force_n=np.stack(
                    modal_generalized_forces_n
                ),
                natural_frequencies_hz=(
                    modal_model.natural_frequencies_hz
                ),
            )
        temporary.replace(modal_history_path)
    summary = {
        **case.resolved(),
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_devices": shard_count,
            "cold_start": True,
            "warmup_checkpoint_used": False,
            "steps_run": steps_to_run,
            "final_step": state.clock.step,
            "final_time_seconds": (
                state.clock.step * case.time.dt_seconds
            ),
            "runtime_seconds": time.perf_counter() - started,
            "actuator_line_enabled": line_enabled,
            "aeroelastic_coupling_enabled": modal_model is not None,
            "turbine_computation_backend": (
                "jax_modal"
                if modal_model is not None
                else "jax_rigid"
            ),
            "aeroelastic_history_file": (
                None
                if modal_history_path is None
                else str(modal_history_path)
            ),
            "modal_natural_frequencies_hz": (
                []
                if modal_model is None
                else [
                    float(value)
                    for value in modal_model.natural_frequencies_hz
                ]
            ),
            "flow_slice_frames": len(frame_times),
            "flow_slices_file": (
                None
                if flow_frames_path is None
                else str(flow_frames_path)
            ),
            **latest,
            **latest_aeroelastic,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary
