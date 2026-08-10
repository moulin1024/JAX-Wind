"""Execute the GABLS1 stable-boundary-layer benchmark on the semantic solver."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np

from .config import CaseConfig
from .most import MoninObukhovWallLaw
from .profiles import ProfileStatistics, plot_profiles, write_profiles
from .surface import is_fixed_surface_flux as _is_fixed_surface_flux
from .surface import physical_arrays as _physical_arrays
from .surface import stability as _stability
from .surface import surface_fluxes as _surface_fluxes
from .surface import velocity_scale as _velocity_scale


HERE = Path(__file__).resolve().parent
CHECKOUT_ROOT = HERE.parents[3]
PADDING_RATIO = 1.5
LASD_SCALAR_STABILITY_BUOYANCY_COEFFICIENT = 0.0


def _configure_pressure_source() -> None:
    configured = Path(
        os.environ.get(
            "JAXWIND_SPECTRAL_FD_SOURCE",
            CHECKOUT_ROOT.parent / "bw1000_benchmark",
        )
    )
    candidates = (
        configured,
        CHECKOUT_ROOT.parent / "bw1000_benchmark",
        CHECKOUT_ROOT / "external" / "bw1000_benchmark",
    )
    pressure_source = next(
        (path for path in candidates if (path / "spectral_fd").is_dir()),
        configured,
    )
    if str(pressure_source) not in sys.path:
        sys.path.insert(0, str(pressure_source))


def _initial_fields(
    *,
    jax,
    jnp,
    case,
    physical_grid,
    decomposition,
    mechanical_scales,
    thermal_scales,
    algebra,
    pressure_solver,
    config,
):
    from jaxwind.domain import (
        Accepted,
        AddressableField,
        Candidate,
        Cell,
        PotentialTemperaturePerturbation,
        VerticalBoundary,
        VerticalVelocity,
        XVelocity,
        YVelocity,
        ZFace,
    )
    from jaxwind.interpreters.jax_zslab import ZFaceFieldContext
    from jaxwind.operators import VelocityVector, project
    from jaxwind.physics import BoussinesqFields

    dtype = getattr(jnp, case.numerics.dtype)
    shape = (1, case.domain.nz, case.domain.ny, case.domain.nx)
    z = (jnp.arange(case.domain.nz, dtype=dtype) + 0.5) * case.domain.dz_m
    temperature_profile = jnp.where(
        z <= case.thermal.inversion_height_m,
        case.thermal.initial_temperature_k,
        case.thermal.initial_temperature_k
        + case.thermal.inversion_gradient_k_m
        * (z - case.thermal.inversion_height_m),
    )
    noise = jax.random.uniform(
        jax.random.PRNGKey(case.numerics.seed),
        shape,
        minval=-case.thermal.perturbation_amplitude_k,
        maxval=case.thermal.perturbation_amplitude_k,
        dtype=dtype,
    )
    noise -= jnp.mean(noise, axis=(0, 2, 3), keepdims=True)
    noise *= (z < case.thermal.perturbation_height_m)[None, :, None, None]
    temperature_perturbation = (
        temperature_profile - case.thermal.initial_temperature_k
    )[None, :, None, None] + noise

    # Integrate in coordinates translating with the geostrophic wind.  This is
    # the exact Galilean form of the periodic GABLS1 equations and prevents the
    # fixed-step AB2 method from needlessly advecting the grid-scale thermal
    # seed at 8 m/s before turbulence and SGS dissipation have developed.
    u = jnp.zeros(shape, dtype=dtype)
    v = jnp.zeros(shape, dtype=dtype)
    if _is_fixed_surface_flux(case) and case.thermal.perturbation_amplitude_k:
        z_face = (jnp.arange(case.domain.nz, dtype=dtype) + 1.0) * case.domain.dz_m
        taper = jnp.maximum(
            1.0 - z_face / max(case.thermal.inversion_height_m, case.domain.dz_m),
            0.0,
        )
        normalized_noise = noise / case.thermal.perturbation_amplitude_k
        w = (
            0.1
            * _velocity_scale(case)
            * normalized_noise
            * taper[None, :, None, None]
        )
    else:
        w = jnp.zeros(shape, dtype=dtype)
    cell_regions = decomposition.regions(Cell)
    face_regions = decomposition.regions(ZFace)
    candidate = VelocityVector(
        AddressableField(
            XVelocity,
            Cell,
            cell_regions,
            Candidate,
            mechanical_scales.to_execution_velocity(u),
        ),
        AddressableField(
            YVelocity,
            Cell,
            cell_regions,
            Candidate,
            mechanical_scales.to_execution_velocity(v),
        ),
        ZFaceFieldContext(
            AddressableField(
                VerticalVelocity,
                ZFace,
                face_regions,
                Candidate,
                mechanical_scales.to_execution_velocity(w),
            ),
            jnp.zeros((case.domain.ny, case.domain.nx), dtype=dtype),
        ),
    )
    velocity = project(
        candidate,
        dt=config.dt,
        normal_boundary=VerticalBoundary(0.0, 0.0),
        algebra=algebra,
        pressure_solver=pressure_solver,
    ).velocity
    scalar = AddressableField(
        PotentialTemperaturePerturbation,
        Cell,
        cell_regions,
        Accepted,
        thermal_scales.to_execution_potential_temperature(
            temperature_perturbation
        ),
    )
    return BoussinesqFields(velocity, scalar)


def _stable_vector_field(
    *,
    algebra,
    model,
    case,
    mechanical_scales,
    thermal_scales,
    wall_law,
    jax,
    jnp,
):
    from jaxwind.physics import (
        BoussinesqDiagnostic,
        BoussinesqTendency,
        BoussinesqVectorFieldResult,
    )

    def surface_sources(x_payload, y_payload, theta_payload, execution_time):
        physical_time = mechanical_scales.from_execution_time(execution_time)
        surface_temperature = (
            case.thermal.initial_temperature_k
            + case.thermal.surface_cooling_k_s * physical_time
        )
        mean_u = case.flow.geostrophic_u_m_s + (
            mechanical_scales.from_execution_velocity(jnp.mean(x_payload[:, 0]))
        )
        mean_v = case.flow.geostrophic_v_m_s + (
            mechanical_scales.from_execution_velocity(jnp.mean(y_payload[:, 0]))
        )
        mean_theta = case.thermal.initial_temperature_k + (
            thermal_scales.from_execution_potential_temperature(
                jnp.mean(theta_payload[:, 0])
            )
        )
        fluxes = wall_law.surface_fluxes(
            mean_u,
            mean_v,
            mean_theta,
            surface_temperature,
            0.5 * case.domain.dz_m,
        )
        wall_x = mechanical_scales.to_execution_acceleration(
            -fluxes.stress_x / case.domain.dz_m
        )
        wall_y = mechanical_scales.to_execution_acceleration(
            -fluxes.stress_y / case.domain.dz_m
        )
        scalar_surface = thermal_scales.to_execution_temperature_tendency(
            fluxes.heat_flux / case.domain.dz_m
        )
        return wall_x, wall_y, scalar_surface

    surface_sources = jax.jit(surface_sources)

    class StableVectorField:
        def __call__(self, evaluation):
            fields = evaluation.velocity
            wall_x, wall_y, scalar_surface = surface_sources(
                fields.velocity.x.payload,
                fields.velocity.y.payload,
                fields.potential_temperature.payload,
                evaluation.time.time,
            )
            fused_tendency = algebra.fused_boussinesq_tendency(
                fields,
                model,
                wall_acceleration=(wall_x, wall_y),
                scalar_surface_source=scalar_surface,
            )
            if fused_tendency is not None:
                return BoussinesqVectorFieldResult(
                    fused_tendency,
                    BoussinesqDiagnostic(evaluation.time),
                )

            context = algebra.boussinesq_context(fields)
            momentum = algebra.momentum_context(context)
            wall_x_payload = (
                jnp.zeros_like(fields.velocity.x.payload).at[:, 0].set(wall_x)
            )
            wall_y_payload = (
                jnp.zeros_like(fields.velocity.y.payload).at[:, 0].set(wall_y)
            )
            wall = algebra._dry_tendency(
                wall_x_payload,
                wall_y_payload,
                jnp.zeros_like(fields.velocity.z.owned.payload),
            )
            scalar_surface_tendency = algebra._scalar_tendency(
                context,
                jnp.zeros_like(fields.potential_temperature.payload)
                .at[:, 0]
                .set(scalar_surface),
            )
            momentum_tendency = algebra.combine_tendencies(
                (
                    algebra.advection_tendency(
                        momentum,
                        model.momentum.advection,
                        model.momentum.wall,
                    ),
                    algebra.pressure_gradient_tendency(
                        momentum, model.momentum.pressure_gradient
                    ),
                    wall,
                    algebra.sgs_tendency(
                        momentum,
                        model.momentum.sgs,
                        model.momentum.wall,
                    ),
                    algebra.coriolis_geostrophic_tendency(
                        momentum, model.momentum.rotation
                    ),
                    algebra.buoyancy_tendency(context, model.buoyancy),
                    algebra.rayleigh_damping_tendency(
                        context, model.rayleigh_damping
                    ),
                )
            )
            scalar_tendency = algebra.combine_scalar_tendencies(
                (
                    algebra.scalar_advection_tendency(
                        context, model.scalar_advection
                    ),
                    algebra.scalar_sgs_tendency(
                        context,
                        model.momentum.sgs,
                        model.scalar_sgs,
                        model.scalar_boundary,
                    ),
                    scalar_surface_tendency,
                )
            )
            return BoussinesqVectorFieldResult(
                BoussinesqTendency(momentum_tendency, scalar_tendency),
                BoussinesqDiagnostic(evaluation.time),
            )

    return StableVectorField()


def _snapshot(
    state,
    *,
    case,
    mechanical_scales,
    thermal_scales,
    algebra,
    model,
    wall_law,
    jnp,
    jax,
):
    u_j, v_j, w_j, w_upper_j, theta_j = _physical_arrays(
        state.fields, case, mechanical_scales, thermal_scales, jnp
    )
    u, v, w, w_upper, theta = (
        np.asarray(jax.device_get(value), dtype=np.float64)
        for value in (u_j, v_j, w_j, w_upper_j, theta_j)
    )
    mean_u = np.mean(u, axis=(1, 2))
    mean_v = np.mean(v, axis=(1, 2))
    mean_w = np.mean(w, axis=(1, 2))
    mean_theta = np.mean(theta, axis=(1, 2))
    u_face = np.concatenate(
        (0.5 * (u[:-1] + u[1:]), u[-1:]), axis=0
    )
    v_face = np.concatenate(
        (0.5 * (v[:-1] + v[1:]), v[-1:]), axis=0
    )
    theta_face = np.concatenate(
        (0.5 * (theta[:-1] + theta[1:]), theta[-1:]), axis=0
    )
    w_upper_prime = w_upper - np.mean(w_upper, axis=(1, 2), keepdims=True)
    uw_resolved = np.zeros(case.domain.nz + 1)
    vw_resolved = np.zeros(case.domain.nz + 1)
    wtheta_resolved = np.zeros(case.domain.nz + 1)
    uw_resolved[1:] = np.mean(
        (u_face - np.mean(u_face, axis=(1, 2), keepdims=True)) * w_upper_prime,
        axis=(1, 2),
    )
    vw_resolved[1:] = np.mean(
        (v_face - np.mean(v_face, axis=(1, 2), keepdims=True)) * w_upper_prime,
        axis=(1, 2),
    )
    wtheta_resolved[1:] = np.mean(
        (
            theta_face
            - np.mean(theta_face, axis=(1, 2), keepdims=True)
        )
        * w_upper_prime,
        axis=(1, 2),
    )

    fluxes, surface_temperature = _surface_fluxes(
        state.fields,
        state.clock.time,
        case=case,
        mechanical_scales=mechanical_scales,
        thermal_scales=thermal_scales,
        wall_law=wall_law,
        jnp=jnp,
    )
    context = algebra.boussinesq_context(state.fields)
    lasd_diagnostic = algebra.lasd_diagnostic_fields(
        context,
        model.momentum.sgs,
        model.scalar_sgs,
        model.scalar_boundary,
    )
    maximum_eddy_viscosity = float(
        jax.device_get(
            jnp.max(lasd_diagnostic.momentum_diffusivity)
            * mechanical_scales.kinematic_viscosity
        )
    )
    maximum_scalar_diffusivity = float(
        jax.device_get(
            jnp.max(lasd_diagnostic.scalar_diffusivity)
            * mechanical_scales.kinematic_viscosity
        )
    )
    diffusive_cfl = (
        2.0
        * case.time.dt_seconds
        * maximum_scalar_diffusivity
        * (
            1.0 / case.domain.dx_m**2
            + 1.0 / case.domain.dy_m**2
            + 1.0 / case.domain.dz_m**2
        )
    )
    trajectory_cfl = (
        case.time.dt_seconds
        * case.sgs.lasd_update_interval
        * float(
            jax.device_get(
                jnp.max(
                    jnp.abs(u_j - case.flow.geostrophic_u_m_s)
                    / case.domain.dx_m
                    + jnp.abs(v_j - case.flow.geostrophic_v_m_s)
                    / case.domain.dy_m
                    + jnp.abs(w_j) / case.domain.dz_m
                )
            )
        )
    )
    txz, tyz = algebra.sgs_vertical_flux(
        context.momentum, model.momentum.sgs
    )
    txz = mechanical_scales.kinematic_pressure * txz[0]
    tyz = mechanical_scales.kinematic_pressure * tyz[0]
    uw_sgs = np.empty(case.domain.nz + 1)
    vw_sgs = np.empty(case.domain.nz + 1)
    uw_sgs[0] = -float(jnp.mean(fluxes.stress_x))
    vw_sgs[0] = -float(jnp.mean(fluxes.stress_y))
    uw_sgs[1:] = np.asarray(jax.device_get(jnp.mean(txz, axis=(1, 2))))
    vw_sgs[1:] = np.asarray(jax.device_get(jnp.mean(tyz, axis=(1, 2))))

    standard_scalar_tendency = algebra.scalar_sgs_tendency(
        context,
        model.momentum.sgs,
        model.scalar_sgs,
        model.scalar_boundary,
    ).payload
    fixed_flux = (
        getattr(case.thermal, "boundary_condition", "")
        == "fixed_surface_flux"
    )
    surface_source = jnp.where(
        fixed_flux,
        jnp.zeros_like(standard_scalar_tendency),
        jnp.zeros_like(standard_scalar_tendency).at[:, 0].set(
            thermal_scales.to_execution_temperature_tendency(
                fluxes.heat_flux / case.domain.dz_m
            )
        )
    )
    total_plane_tendency = jnp.mean(
        standard_scalar_tendency + surface_source,
        axis=(0, 2, 3),
    )
    lower_flux_execution = thermal_scales.to_execution_temperature_flux(
        jnp.mean(fluxes.heat_flux)
    )
    execution_dz = case.domain.dz_m / mechanical_scales.length
    upper_flux_execution = lower_flux_execution - execution_dz * jnp.cumsum(
        total_plane_tendency
    )
    wtheta_sgs = np.empty(case.domain.nz + 1)
    wtheta_sgs[0] = float(jnp.mean(fluxes.heat_flux))
    wtheta_sgs[1:] = np.asarray(
        jax.device_get(
            thermal_scales.from_execution_temperature_flux(
                upper_flux_execution
            )
        )
    )
    values = {
        "u": mean_u,
        "v": mean_v,
        "w": mean_w,
        "theta": mean_theta,
        "u2": np.mean(u * u, axis=(1, 2)),
        "v2": np.mean(v * v, axis=(1, 2)),
        "w2": np.mean(w * w, axis=(1, 2)),
        "theta2": np.mean(theta * theta, axis=(1, 2)),
        "uw_resolved": uw_resolved,
        "vw_resolved": vw_resolved,
        "uw_sgs": uw_sgs,
        "vw_sgs": vw_sgs,
        "wtheta_resolved": wtheta_resolved,
        "wtheta_sgs": wtheta_sgs,
    }
    diagnostics = {
        "surface_temperature_k": float(surface_temperature),
        "surface_heat_flux_k_m_s": float(jnp.mean(fluxes.heat_flux)),
        "friction_velocity_m_s": float(
            jnp.sqrt(jnp.mean(fluxes.friction_velocity**2))
        ),
        "mean_obukhov_length_m": float(
            jnp.mean(
                jnp.where(
                    jnp.isfinite(fluxes.obukhov_length),
                    fluxes.obukhov_length,
                    0.0,
                )
            )
        ),
        "maximum_eddy_viscosity_m2_s": maximum_eddy_viscosity,
        "scalar_diffusive_cfl": diffusive_cfl,
        "lasd_trajectory_cfl": trajectory_cfl,
    }
    return values, diagnostics


def run_case(
    case: CaseConfig,
    *,
    output_dir: Path,
    restart: Path | None,
    max_steps: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    _configure_pressure_source()
    import jax

    jax.config.update("jax_enable_x64", case.numerics.dtype == "float64")
    import jax.numpy as jnp
    from spectral_fd import runtime_from_initialized_jax

    from jaxwind.domain import (
        AcceptedClock,
        BoussinesqScaleSystem,
        DistributionSpec,
        EqualZSlab,
        MeshAxis,
        MeshTopology,
        ScaleSystem,
        UniformGrid,
        VerticalBoundary,
    )
    from jaxwind.effects import (
        ZSlabCheckpointLayout,
        load_boussinesq_checkpoint,
        save_boussinesq_checkpoint,
    )
    from jaxwind.integrators import AB2Config, cold_start_boussinesq, step_boussinesq
    from jaxwind.interpreters.jax_zslab import build_zslab_interpreter
    from jaxwind.physics import (
        BoussinesqModel,
        BoussinesqVectorField,
        ConservativeAdvection,
        ConservativeScalarAdvection,
        CoriolisGeostrophic,
        DryFlowModel,
        KinematicPressureGradient,
        LagrangianScaleDependentDynamic,
        LagrangianScaleDependentScalarFlux,
        LasdAcceptedStepEvent,
        LinearBoussinesqBuoyancy,
        NeutralLogWall,
        NoRotation,
        NoRayleighDamping,
        ScalarFluxBoundary,
    )
    from jaxwind.pressure import build_spectral_fd_pressure_adapter

    if jax.device_count() != 1:
        raise RuntimeError(
            "the stratified ABL runner currently requires one JAX device"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_latest.npz"
    statistics_path = output_dir / "statistics_latest.npz"
    if restart is None and checkpoint_path.exists() and not overwrite:
        raise FileExistsError(
            f"{checkpoint_path} exists; configure "
            "execution.restart_checkpoint or execution.overwrite"
        )
    if restart is not None and not restart.exists():
        raise FileNotFoundError(restart)
    (output_dir / "resolved_config.toml").write_text(case.resolved_toml())

    physical_grid = UniformGrid(
        case.domain.nx,
        case.domain.ny,
        case.domain.nz,
        case.domain.lx_m,
        case.domain.ly_m,
        case.domain.lz_m,
    )
    mechanical_scales = ScaleSystem(case.domain.lz_m, _velocity_scale(case))
    thermal_scales = BoussinesqScaleSystem(mechanical_scales, 1.0)
    grid = mechanical_scales.to_execution_grid(physical_grid)
    decomposition = EqualZSlab(
        grid,
        MeshTopology((MeshAxis("z", 1),)),
        DistributionSpec.z_slab(),
    )
    algebra = build_zslab_interpreter(
        decomposition,
        addressable_shards=(0,),
        nonlinear_padding_ratio=PADDING_RATIO,
    )
    pressure_solver = build_spectral_fd_pressure_adapter(
        decomposition,
        addressable_shards=(0,),
        runtime=runtime_from_initialized_jax(jax),
        dtype=case.numerics.dtype,
        method=case.numerics.pressure_method,
    )
    buoyancy_coefficient = thermal_scales.to_execution_buoyancy_coefficient(
        gravity=case.thermal.gravity_m_s2,
        reference_potential_temperature=case.thermal.reference_temperature_k,
    )
    momentum_sgs = LagrangianScaleDependentDynamic(
        filter_grid_ratio=1.5,
        update_interval=case.sgs.lasd_update_interval,
    )
    scalar_stability = (
        buoyancy_coefficient
        if _is_fixed_surface_flux(case)
        else LASD_SCALAR_STABILITY_BUOYANCY_COEFFICIENT
    )
    scalar_sgs = LagrangianScaleDependentScalarFlux(
        # The dynamic scalar coefficient already responds to the resolved
        # stratification. Applying the additional Richardson multiplier
        # collapses near-wall mixing and can reverse the diagnosed stability.
        stability_buoyancy_coefficient=scalar_stability,
        stability_beta=30.0,
        stability_power=2.0,
    )
    rotation = (
        NoRotation()
        if _is_fixed_surface_flux(case)
        else CoriolisGeostrophic(
            mechanical_scales.to_execution_inverse_time(case.flow.coriolis_s),
            0.0,
            0.0,
        )
    )
    scalar_boundary = (
        ScalarFluxBoundary(
            thermal_scales.to_execution_temperature_flux(
                case.thermal.surface_heat_flux_k_m_s
            ),
            0.0,
        )
        if _is_fixed_surface_flux(case)
        else ScalarFluxBoundary()
    )
    model = BoussinesqModel(
        DryFlowModel(
            ConservativeAdvection(),
            KinematicPressureGradient(0.0, 0.0),
            NeutralLogWall(
                mechanical_scales.to_execution_length(
                    case.flow.roughness_length_m
                ),
                case.flow.von_karman,
            ),
            momentum_sgs,
            rotation,
        ),
        ConservativeScalarAdvection(),
        scalar_sgs,
        LinearBoussinesqBuoyancy(buoyancy_coefficient),
        NoRayleighDamping(),
        scalar_boundary,
    )
    wall_law = MoninObukhovWallLaw(
        case.flow.roughness_length_m,
        case.thermal.thermal_roughness_length_m,
        case.thermal.reference_temperature_k,
        gravity=case.thermal.gravity_m_s2,
        von_karman=case.flow.von_karman,
    )
    config = AB2Config(mechanical_scales.to_execution_time(case.time.dt_seconds))
    fixed_surface_flux = _is_fixed_surface_flux(case)
    surface_fingerprint = (
        "fixed-heat-flux-neutral-wall-v1"
        if fixed_surface_flux
        else "businger-dyer-plane-prescribed-temperature-v3"
    )
    frame_fingerprint = (
        "stationary-v1"
        if fixed_surface_flux
        else "geostrophic-translating-v1"
    )
    physics_fingerprint = (
        momentum_sgs.fingerprint
        + "|"
        + scalar_sgs.fingerprint
        + f"|surface={surface_fingerprint}"
        + f"|stability={_stability(case)}"
        + "|thermal-boundary="
        + getattr(
            case.thermal,
            "boundary_condition",
            "prescribed_surface_temperature",
        )
        + f"|frame={frame_fingerprint}"
        + "|advection=conservative|dealiasing=three-halves-padding"
    )
    checkpoint_layout = ZSlabCheckpointLayout(
        decomposition, (0,), jnp.asarray
    )
    closure_fingerprint = momentum_sgs.fingerprint + "|" + scalar_sgs.fingerprint
    if restart is None:
        fields = _initial_fields(
            jax=jax,
            jnp=jnp,
            case=case,
            physical_grid=physical_grid,
            decomposition=decomposition,
            mechanical_scales=mechanical_scales,
            thermal_scales=thermal_scales,
            algebra=algebra,
            pressure_solver=pressure_solver,
            config=config,
        )
        fields = algebra.initialize_lasd_closure(fields, model)
        state = cold_start_boussinesq(
            fields, clock=AcceptedClock(0.0, 0), config=config
        )
        statistics = ProfileStatistics(case.domain.nz)
    else:
        state = load_boussinesq_checkpoint(
            restart,
            layout=checkpoint_layout,
            config=config,
            scale_fingerprint=thermal_scales.fingerprint,
            closure_fingerprint=closure_fingerprint,
            physics_fingerprint=physics_fingerprint,
        )
        statistics = (
            ProfileStatistics.load(statistics_path, case.domain.nz)
            if statistics_path.exists()
            else ProfileStatistics(case.domain.nz)
        )
    remaining = case.time.steps - state.clock.step
    if remaining < 0:
        raise ValueError("restart is beyond the configured ABL final time")
    steps_to_run = remaining if max_steps is None else min(remaining, max_steps)
    vector_field = (
        BoussinesqVectorField(algebra, model)
        if fixed_surface_flux
        else _stable_vector_field(
            algebra=algebra,
            model=model,
            case=case,
            mechanical_scales=mechanical_scales,
            thermal_scales=thermal_scales,
            wall_law=wall_law,
            jax=jax,
            jnp=jnp,
        )
    )
    closure_event = LasdAcceptedStepEvent(algebra, model, config.dt)

    history_name = "history.csv" if case.runner == "abl" else "time_series.csv"
    history_path = output_dir / history_name
    append_history = restart is not None and history_path.exists()
    history_stream = history_path.open("a" if append_history else "w", newline="")
    history_fields = (
        "step",
        "time_hours",
        "cfl",
        "maximum_divergence_s",
        "surface_temperature_k",
        "surface_heat_flux_k_m_s",
        "friction_velocity_m_s",
        "mean_obukhov_length_m",
        "maximum_eddy_viscosity_m2_s",
        "scalar_diffusive_cfl",
        "lasd_trajectory_cfl",
        "elapsed_seconds",
    )
    history_writer = csv.DictWriter(history_stream, fieldnames=history_fields)
    if not append_history:
        history_writer.writeheader()
    started = time.perf_counter()
    latest: dict[str, float] = {}
    initial_step = state.clock.step
    timing_warmup_steps = min(
        steps_to_run,
        2 * case.sgs.lasd_update_interval,
    )
    timing_warmup_elapsed: float | None = None
    timing_end_step = max(timing_warmup_steps, steps_to_run - 1)
    timing_end_elapsed: float | None = None
    solver_elapsed: float | None = None
    try:
        for local_step in range(1, steps_to_run + 1):
            next_step = state.clock.step + 1
            final_local = local_step == steps_to_run
            should_log = next_step % case.output.log_every_steps == 0 or final_local
            result = step_boussinesq(
                state,
                config=config,
                environment=None,
                vector_field=vector_field,
                normal_boundary=lambda _clock, _environment: VerticalBoundary(
                    0.0, 0.0
                ),
                algebra=algebra,
                pressure_solver=pressure_solver,
                closure_event=closure_event,
                compute_projection_residual=should_log,
            )
            state = result.state
            if local_step == timing_warmup_steps:
                state.fields.velocity.x.payload.block_until_ready()
                timing_warmup_elapsed = time.perf_counter() - started
            if local_step == timing_end_step:
                state.fields.velocity.x.payload.block_until_ready()
                timing_end_elapsed = time.perf_counter() - started
            if final_local:
                state.fields.velocity.x.payload.block_until_ready()
                solver_elapsed = time.perf_counter() - started
            should_sample = (
                state.clock.step >= case.time.sample_start_step
                and (state.clock.step - case.time.sample_start_step)
                % case.output.sample_every_steps
                == 0
            ) or final_local
            if should_sample:
                values, surface = _snapshot(
                    state,
                    case=case,
                    mechanical_scales=mechanical_scales,
                    thermal_scales=thermal_scales,
                    algebra=algebra,
                    model=model,
                    wall_law=wall_law,
                    jnp=jnp,
                    jax=jax,
                )
                statistics.sample(values)
            if should_log:
                u, v, w, _w_upper, _theta = _physical_arrays(
                    state.fields, case, mechanical_scales, thermal_scales, jnp
                )
                cfl = float(
                    case.time.dt_seconds
                    * jnp.max(
                        jnp.abs(u) / case.domain.dx_m
                        + jnp.abs(v) / case.domain.dy_m
                        + jnp.abs(w) / case.domain.dz_m
                    )
                )
                if not should_sample:
                    _values, surface = _snapshot(
                        state,
                        case=case,
                        mechanical_scales=mechanical_scales,
                        thermal_scales=thermal_scales,
                        algebra=algebra,
                        model=model,
                        wall_law=wall_law,
                        jnp=jnp,
                        jax=jax,
                    )
                divergence = result.diagnostic.projection.divergence.payload
                divergence.block_until_ready()
                latest = {
                    "cfl": cfl,
                    "maximum_divergence_s": float(
                        jnp.max(jnp.abs(divergence))
                        * mechanical_scales.inverse_time
                    ),
                    **surface,
                }
                if cfl >= case.numerics.cfl_warning:
                    warnings.warn(
                        f"ABL CFL {cfl:.3f} exceeds configured warning "
                        f"{case.numerics.cfl_warning:.3f}",
                        stacklevel=1,
                    )
                row = {
                    "step": state.clock.step,
                    "time_hours": mechanical_scales.from_execution_time(
                        state.clock.time
                    )
                    / 3600.0,
                    **latest,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                history_writer.writerow(row)
                history_stream.flush()
                print(
                    f"step={state.clock.step}/{case.time.steps} "
                    f"t={row['time_hours']:.3f}h "
                    f"u*={row['friction_velocity_m_s']:.4f} "
                    f"Q0={row['surface_heat_flux_k_m_s']:.5f} "
                    f"CFL={cfl:.3f} "
                    f"diff-CFL={row['scalar_diffusive_cfl']:.3f} "
                    f"trajectory-CFL={row['lasd_trajectory_cfl']:.3f} "
                    f"elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            if (
                state.clock.step % case.output.checkpoint_every_steps == 0
                or final_local
            ):
                save_boussinesq_checkpoint(
                    checkpoint_path,
                    state,
                    scale_fingerprint=thermal_scales.fingerprint,
                    physics_fingerprint=physics_fingerprint,
                )
                statistics.save(statistics_path)
    finally:
        history_stream.close()

    if steps_to_run == 0:
        save_boussinesq_checkpoint(
            checkpoint_path,
            state,
            scale_fingerprint=thermal_scales.fingerprint,
            physics_fingerprint=physics_fingerprint,
        )
    if statistics.count:
        write_profiles(output_dir, case, statistics)
        plot_profiles(output_dir, case)
    if state.clock.step == case.time.steps:
        save_boussinesq_checkpoint(
            output_dir / "checkpoint_final.npz",
            state,
            scale_fingerprint=thermal_scales.fingerprint,
            physics_fingerprint=physics_fingerprint,
        )
    elapsed = time.perf_counter() - started
    post_warmup_steps = timing_end_step - timing_warmup_steps
    post_warmup_elapsed = (
        timing_end_elapsed - timing_warmup_elapsed
        if (
            timing_end_elapsed is not None
            and timing_warmup_elapsed is not None
            and post_warmup_steps
        )
        else None
    )
    post_warmup_rate = (
        post_warmup_steps / post_warmup_elapsed if post_warmup_elapsed else None
    )
    summary = {
        "schema": (
            "jaxwind.abl-warmup.v1"
            if case.runner == "abl"
            else "jaxwind.gabls1.v1"
        ),
        "case": case.resolved(),
        "physics": {
            "momentum_sgs": "LASD",
            "scalar_sgs": "LASD dynamic scalar flux",
            "stability": _stability(case),
            "surface": surface_fingerprint,
            "reference_frame": (
                (
                    "stationary"
                    if fixed_surface_flux
                    else "geostrophic translating"
                )
            ),
            "momentum_advection": "conservative",
            "scalar_advection": "conservative",
            "dealiasing": "three-halves-padding",
            "nonlinear_padding_ratio": PADDING_RATIO,
            "fingerprint": physics_fingerprint,
        },
        "runtime": {
            "jax_backend": jax.default_backend(),
            "initial_step": initial_step,
            "steps_run": steps_to_run,
            "final_step": state.clock.step,
            "final_time_hours": mechanical_scales.from_execution_time(
                state.clock.time
            )
            / 3600.0,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps_to_run / elapsed if elapsed else math.inf,
            "solver_elapsed_seconds": solver_elapsed,
            "timing_warmup_steps": timing_warmup_steps,
            "timing_warmup_elapsed_seconds": timing_warmup_elapsed,
            "timing_end_step": timing_end_step,
            "timing_end_elapsed_seconds": timing_end_elapsed,
            "post_warmup_steps": post_warmup_steps,
            "post_warmup_elapsed_seconds": post_warmup_elapsed,
            "post_warmup_steps_per_second": post_warmup_rate,
            "profile_samples": statistics.count,
            "reached_final_time": state.clock.step == case.time.steps,
            **latest,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if case.runner != "abl":
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary
