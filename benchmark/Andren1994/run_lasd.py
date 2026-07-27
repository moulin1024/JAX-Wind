#!/usr/bin/env python3
"""Run Andrén et al. (1994) with momentum/scalar LASD as an external fifth model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
PRESSURE_SOURCE = Path(
    os.environ.get("WIRELES_SPECTRAL_FD_SOURCE", ROOT.parent / "bw1000_benchmark")
)
for source in (ROOT, SOURCE, PRESSURE_SOURCE):
    if source.exists() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from benchmark.Andren1994 import run as andren  # noqa: E402
from benchmark.Andren1994 import fig13_budget  # noqa: E402
from benchmark.NeutralEkman import run as neutral_runner  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from spectral_fd import runtime_from_initialized_jax  # noqa: E402

from wireles.domain import (  # noqa: E402
    Accepted,
    AcceptedClock,
    AddressableField,
    Cell,
    DistributionSpec,
    EqualZSlab,
    MeshAxis,
    MeshTopology,
    PassiveScalarConcentration,
    PassiveScalarScaleSystem,
    ScaleSystem,
    UniformGrid,
    VerticalBoundary,
)
from wireles.effects import (  # noqa: E402
    ZSlabCheckpointLayout,
    load_boussinesq_checkpoint,
    save_boussinesq_checkpoint,
)
from wireles.integrators import (  # noqa: E402
    AB2Config,
    cold_start_boussinesq,
    step_boussinesq,
)
from wireles.interpreters.jax_zslab import build_zslab_interpreter  # noqa: E402
from wireles.operators import project  # noqa: E402
from wireles.physics import (  # noqa: E402
    BoussinesqFields,
    BoussinesqModel,
    BoussinesqVectorField,
    ConservativeAdvection,
    ConservativeScalarAdvection,
    CoriolisGeostrophic,
    DiagnosticLasdConstants,
    DryFlowModel,
    KinematicPressureGradient,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LasdAcceptedStepEvent,
    NeutralLogWall,
    NoBuoyancy,
    NoRayleighDamping,
    ScalarFluxBoundary,
)
from wireles.pressure import build_spectral_fd_pressure_adapter  # noqa: E402


HERE = Path(__file__).resolve().parent
SURFACE_SCALAR_FLUX = 1.0e-3
AIR_DENSITY = 1.0
ANDREN_DIAGNOSTIC_CONSTANTS = DiagnosticLasdConstants(horizontal_homogeneous_wall=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--dt", type=float, default=0.8)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--sample-every", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=600)
    parser.add_argument("--checkpoint-every", type=int, default=6000)
    parser.add_argument("--lasd-update-interval", type=int, default=5)
    parser.add_argument("--max-cfl-warning", type=float, default=0.25)
    parser.add_argument("--method", choices=("transpose", "spike"), default="spike")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--fig13-budget",
        action="store_true",
        help="sample the complete resolved scalar-flux budget used by paper Fig. 13",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.nx = args.ny = args.nz = 8
        args.dt = 0.25
        args.hours = 8.0 * args.dt / 3600.0
        args.sample_every = 1
        args.log_every = 4
        args.checkpoint_every = 4
    elif args.hours is None:
        args.hours = andren.CANONICAL_HOURS
    if args.output is None:
        name = "lasd_quick" if args.quick else "lasd_40x40x40"
        args.output = HERE / "results" / name
    if min(args.nx, args.ny, args.nz) <= 1:
        parser.error("all grid dimensions must exceed one")
    if args.dt <= 0.0 or args.hours <= 0.0:
        parser.error("dt and hours must be positive")
    if min(args.sample_every, args.log_every, args.checkpoint_every) <= 0:
        parser.error("sampling, logging, and checkpoint intervals must be positive")
    if args.lasd_update_interval <= 0:
        parser.error("LASD update interval must be positive")
    return args


def _plane_profile(values) -> np.ndarray:
    return np.asarray(jax.device_get(jnp.mean(values, axis=(0, 2, 3))))


def _w_at_cells(velocity):
    upper = velocity.z.owned.payload
    lower_plane = jnp.broadcast_to(
        jnp.asarray(velocity.z.lower_boundary, dtype=upper.dtype),
        upper.shape[2:],
    )
    lower = jnp.concatenate((lower_plane[None, None], upper[:, :-1]), axis=1)
    return 0.5 * (lower + upper)


def _x_spectrum(values, level: int):
    signal = values[:, level] - jnp.mean(values[:, level], axis=-1, keepdims=True)
    coefficients = jnp.fft.rfft(signal, axis=-1) / signal.shape[-1]
    energy = jnp.mean(jnp.abs(coefficients) ** 2, axis=(0, 1))
    if values.shape[-1] % 2 == 0:
        factors = jnp.concatenate(
            (
                jnp.ones((1,), dtype=energy.dtype),
                2.0 * jnp.ones((energy.size - 2,), dtype=energy.dtype),
                jnp.ones((1,), dtype=energy.dtype),
            )
        )
    else:
        factors = jnp.concatenate(
            (
                jnp.ones((1,), dtype=energy.dtype),
                2.0 * jnp.ones((energy.size - 1,), dtype=energy.dtype),
            )
        )
    return np.asarray(jax.device_get(energy * factors))


def instantaneous_diagnostics(
    state,
    divergence,
    *,
    physical_grid,
    mechanical_scales,
    scalar_scales,
    algebra,
    model,
    args,
):
    fields = state.fields
    u = mechanical_scales.from_execution_velocity(fields.velocity.x.payload)
    v = mechanical_scales.from_execution_velocity(fields.velocity.y.payload)
    w = mechanical_scales.from_execution_velocity(_w_at_cells(fields.velocity))
    scalar = scalar_scales.from_execution_concentration(
        fields.potential_temperature.payload
    )
    mean_u = _plane_profile(u)
    mean_v = _plane_profile(v)
    mean_w = _plane_profile(w)
    mean_scalar = _plane_profile(scalar)
    u_fluctuation = u - jnp.asarray(mean_u)[None, :, None, None]
    v_fluctuation = v - jnp.asarray(mean_v)[None, :, None, None]
    w_fluctuation = w - jnp.asarray(mean_w)[None, :, None, None]
    scalar_fluctuation = scalar - jnp.asarray(mean_scalar)[None, :, None, None]

    speed0 = jnp.hypot(u[:, 0], v[:, 0])
    reference_height = 0.5 * physical_grid.dz
    drag = (0.4 / math.log(reference_height / 0.1)) ** 2
    tau_x = -drag * speed0 * u[:, 0]
    tau_y = -drag * speed0 * v[:, 0]
    ustar = math.sqrt(math.hypot(float(jnp.mean(tau_x)), float(jnp.mean(tau_y))))

    context = algebra.boussinesq_context(fields)
    diagnostic = algebra.lasd_diagnostic_fields(
        context,
        model.momentum.sgs,
        model.scalar_sgs,
        model.scalar_boundary,
        constants=ANDREN_DIAGNOSTIC_CONSTANTS,
        wall=model.momentum.wall,
    )
    sgs_tke = diagnostic.sgs_tke * mechanical_scales.velocity**2
    scalar_variance_numerator = (
        diagnostic.scalar_variance_numerator
        * scalar_scales.concentration**2
        * mechanical_scales.velocity
    )
    momentum_diffusivity = (
        diagnostic.momentum_diffusivity * mechanical_scales.kinematic_viscosity
    )
    scalar_diffusivity = (
        diagnostic.scalar_diffusivity * mechanical_scales.kinematic_viscosity
    )
    scalar_flux_upper = scalar_scales.from_execution_concentration_flux(
        diagnostic.scalar_flux_z
    )
    lower_scalar_flux = jnp.concatenate(
        (
            jnp.full_like(
                scalar_flux_upper[:, :1],
                SURFACE_SCALAR_FLUX / AIR_DENSITY,
            ),
            scalar_flux_upper[:, :-1],
        ),
        axis=1,
    )
    sgs_scalar_flux = 0.5 * (lower_scalar_flux + scalar_flux_upper)

    txz_upper, tyz_upper = algebra.sgs_vertical_flux(
        context.momentum,
        model.momentum.sgs,
    )
    txz_upper = txz_upper * mechanical_scales.kinematic_pressure
    tyz_upper = tyz_upper * mechanical_scales.kinematic_pressure
    lower_txz = jnp.concatenate((tau_x[:, None], txz_upper[:, :-1]), axis=1)
    lower_tyz = jnp.concatenate((tau_y[:, None], tyz_upper[:, :-1]), axis=1)

    resolved_tke = _plane_profile(
        0.5 * (u_fluctuation**2 + v_fluctuation**2 + w_fluctuation**2)
    )
    e_sgs = _plane_profile(sgs_tke)
    plane_scalar_variance = np.maximum(
        _plane_profile(scalar_variance_numerator)
        / np.sqrt(np.maximum(e_sgs, np.finfo(e_sgs.dtype).tiny)),
        0.0,
    )
    integrated_resolved = float(np.sum(resolved_tke) * physical_grid.dz)
    integrated_sgs = float(np.sum(e_sgs) * physical_grid.dz)
    cfl = float(
        args.dt
        * jnp.max(
            jnp.abs(u) / physical_grid.dx
            + jnp.abs(v) / physical_grid.dy
            + jnp.abs(w) / physical_grid.dz
        )
    )
    z = (np.arange(physical_grid.nz) + 0.5) * physical_grid.dz
    target_level = int(np.argmin(np.abs(z * andren.F_CORIOLIS / ustar - 0.1)))
    modes = np.arange(physical_grid.nx // 2 + 1, dtype=np.float64)
    profiles = {
        "u": mean_u,
        "v": mean_v,
        "w": mean_w,
        "u2": _plane_profile(u * u),
        "v2": _plane_profile(v * v),
        "w2": _plane_profile(w * w),
        "scalar": mean_scalar,
        "scalar2": _plane_profile(scalar * scalar),
        "resolved_u_variance": _plane_profile(u_fluctuation**2),
        "resolved_v_variance": _plane_profile(v_fluctuation**2),
        "resolved_w_variance": _plane_profile(w_fluctuation**2),
        "resolved_scalar_variance": _plane_profile(scalar_fluctuation**2),
        "resolved_tke": resolved_tke,
        "resolved_uw": _plane_profile(u_fluctuation * w_fluctuation),
        "resolved_vw": _plane_profile(v_fluctuation * w_fluctuation),
        "resolved_wc": _plane_profile(w_fluctuation * scalar_fluctuation),
        "sgs_tke": e_sgs,
        "sgs_scalar_variance": plane_scalar_variance,
        "sgs_uw": _plane_profile(0.5 * (lower_txz + txz_upper)),
        "sgs_vw": _plane_profile(0.5 * (lower_tyz + tyz_upper)),
        "sgs_wc": _plane_profile(sgs_scalar_flux),
        "momentum_diffusivity": _plane_profile(momentum_diffusivity),
        "scalar_diffusivity": _plane_profile(scalar_diffusivity),
        "momentum_coefficient": _plane_profile(
            fields.closure.momentum.coefficient.payload
        ),
        "scalar_coefficient": _plane_profile(fields.closure.scalar.coefficient.payload),
        "spectrum_mode": modes,
        "spectrum_u": _x_spectrum(u, target_level),
        "spectrum_v": _x_spectrum(v, target_level),
        "spectrum_w": _x_spectrum(w, target_level),
        "spectrum_scalar": _x_spectrum(scalar, target_level),
        "spectrum_height_m": np.full_like(modes, z[target_level]),
    }
    history = {
        "time_seconds": mechanical_scales.from_execution_time(state.clock.time),
        "step": state.clock.step,
        "ustar": ustar,
        "cfl": cfl,
        "lasd_cfl": cfl * args.lasd_update_interval,
        "max_divergence": float(
            jnp.max(jnp.abs(divergence)) * mechanical_scales.inverse_time
        ),
        "top_u": float(mean_u[-1]),
        "top_v": float(mean_v[-1]),
        "near_wall_u": float(mean_u[0]),
        "near_wall_v": float(mean_v[0]),
        "integrated_resolved_tke_m3_s2": integrated_resolved,
        "integrated_sgs_tke_m3_s2": integrated_sgs,
        "integrated_total_tke_m3_s2": integrated_resolved + integrated_sgs,
    }
    return history, profiles


def _write_statistics_state(
    path: Path, times: list[float], samples: list[dict]
) -> None:
    if not samples:
        return
    arrays = {
        name: np.stack([sample[name] for sample in samples]) for name in samples[0]
    }
    np.savez(path, profile_times=np.asarray(times), **arrays)


def _load_statistics_state(path: Path):
    if not path.exists():
        return [], []
    with np.load(path, allow_pickle=False) as archive:
        times = np.asarray(archive["profile_times"]).tolist()
        names = tuple(name for name in archive.files if name != "profile_times")
        samples = [
            {name: np.array(archive[name][index], copy=True) for name in names}
            for index in range(len(times))
        ]
    return times, samples


def _read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return [
            {name: float(value) for name, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _save_progress(
    args,
    state,
    history_rows,
    profile_times,
    profile_samples,
    *,
    scale_fingerprint,
    closure_fingerprint,
    budget_times=None,
    budget_samples=None,
):
    save_boussinesq_checkpoint(
        args.output / "checkpoint.npz",
        state,
        scale_fingerprint=scale_fingerprint,
    )
    neutral_runner.write_csv(args.output / "history.csv", history_rows)
    _write_statistics_state(
        args.output / "statistics_samples.npz",
        profile_times,
        profile_samples,
    )
    if budget_times is not None and budget_samples is not None:
        fig13_budget.write_samples(
            args.output / "fig13_budget_samples.npz",
            budget_times,
            budget_samples,
        )
    if state.fields.closure.configuration_fingerprint != closure_fingerprint:
        raise ValueError("saved LASD closure fingerprint changed unexpectedly")


def _average_profile_samples(samples: list[dict]) -> dict:
    """Average instantaneous statistics without folding mean drift into variance."""
    if not samples:
        raise ValueError("at least one profile sample is required")
    averaged = {
        name: np.mean([sample[name] for sample in samples], axis=0)
        for name in samples[0]
    }
    for quantity in ("u", "v", "w", "scalar"):
        if quantity not in averaged:
            continue
        variance = f"resolved_{quantity}_variance"
        if variance not in averaged:
            averaged[variance] = np.mean(
                [
                    np.maximum(
                        sample[f"{quantity}2"] - sample[quantity] ** 2,
                        0.0,
                    )
                    for sample in samples
                ],
                axis=0,
            )
    return averaged


def _write_profiles(output: Path, averaged: dict, statistics_ustar: float) -> None:
    z = (np.arange(averaged["u"].size) + 0.5) * (1500.0 / averaged["u"].size)
    u = averaged["u"]
    v = averaged["v"]
    w = averaged["w"]
    resolved_u_variance = averaged["resolved_u_variance"]
    resolved_v_variance = averaged["resolved_v_variance"]
    resolved_w_variance = averaged["resolved_w_variance"]
    resolved_scalar_variance = averaged["resolved_scalar_variance"]
    component_sgs = (2.0 / 3.0) * averaged["sgs_tke"]
    resolved_tke = 0.5 * (
        resolved_u_variance + resolved_v_variance + resolved_w_variance
    )
    cstar = (SURFACE_SCALAR_FLUX / AIR_DENSITY) / statistics_ustar
    height = z * andren.F_CORIOLIS / statistics_ustar
    scalar_gradient = np.gradient(averaged["scalar"], z)
    phi_c = -0.4 * z * scalar_gradient / cstar
    phi_c[0] = 1.0
    velocity_gradient = np.hypot(np.gradient(u, z), np.gradient(v, z))
    phi_m = 0.4 * z * velocity_gradient / statistics_ustar
    phi_m[0] = 1.0
    ustar2 = statistics_ustar**2
    scalar_flux_scale = statistics_ustar * cstar
    columns = {
        "z_m": z,
        "z_f_over_ustar": height,
        "u_m_s": u,
        "v_m_s": v,
        "w_m_s": w,
        "scalar_kg_m3": averaged["scalar"],
        "phi_m": phi_m,
        "phi_c": phi_c,
        "resolved_u_variance_over_ustar2": resolved_u_variance / ustar2,
        "resolved_v_variance_over_ustar2": resolved_v_variance / ustar2,
        "resolved_w_variance_over_ustar2": resolved_w_variance / ustar2,
        "sgs_component_variance_over_ustar2": component_sgs / ustar2,
        "total_u_variance_over_ustar2": (resolved_u_variance + component_sgs) / ustar2,
        "total_v_variance_over_ustar2": (resolved_v_variance + component_sgs) / ustar2,
        "total_w_variance_over_ustar2": (resolved_w_variance + component_sgs) / ustar2,
        "resolved_tke_over_ustar2": resolved_tke / ustar2,
        "sgs_tke_over_ustar2": averaged["sgs_tke"] / ustar2,
        "total_tke_over_ustar2": (resolved_tke + averaged["sgs_tke"]) / ustar2,
        "resolved_uw_over_ustar2": averaged["resolved_uw"] / ustar2,
        "resolved_vw_over_ustar2": averaged["resolved_vw"] / ustar2,
        "sgs_uw_over_ustar2": averaged["sgs_uw"] / ustar2,
        "sgs_vw_over_ustar2": averaged["sgs_vw"] / ustar2,
        "total_uw_over_ustar2": (averaged["resolved_uw"] + averaged["sgs_uw"]) / ustar2,
        "total_vw_over_ustar2": (averaged["resolved_vw"] + averaged["sgs_vw"]) / ustar2,
        "resolved_scalar_variance_over_cstar2": resolved_scalar_variance / cstar**2,
        "sgs_scalar_variance_over_cstar2": averaged["sgs_scalar_variance"] / cstar**2,
        "total_scalar_variance_over_cstar2": (
            resolved_scalar_variance + averaged["sgs_scalar_variance"]
        )
        / cstar**2,
        "resolved_wc_over_ustar_cstar": averaged["resolved_wc"] / scalar_flux_scale,
        "sgs_wc_over_ustar_cstar": averaged["sgs_wc"] / scalar_flux_scale,
        "total_wc_over_ustar_cstar": (averaged["resolved_wc"] + averaged["sgs_wc"])
        / scalar_flux_scale,
        "momentum_diffusivity_m2_s": averaged["momentum_diffusivity"],
        "scalar_diffusivity_m2_s": averaged["scalar_diffusivity"],
        "momentum_lasd_coefficient": averaged["momentum_coefficient"],
        "scalar_lasd_coefficient": averaged["scalar_coefficient"],
    }
    matrix = np.column_stack(tuple(columns.values()))
    np.savetxt(
        output / "profiles.csv",
        matrix,
        delimiter=",",
        header=",".join(columns),
        comments="",
    )
    np.savetxt(
        output / "normalized_profiles.csv",
        matrix,
        delimiter=",",
        header=",".join(columns),
        comments="",
    )

    modes = averaged["spectrum_mode"]
    selected = modes > 0.0
    k = 2.0 * np.pi * modes[selected] / 4000.0
    spectrum_columns = {
        "k_ustar_over_f": k * statistics_ustar / andren.F_CORIOLIS,
        "kEu_over_ustar2": modes[selected] * averaged["spectrum_u"][selected] / ustar2,
        "kEv_over_ustar2": modes[selected] * averaged["spectrum_v"][selected] / ustar2,
        "kEw_over_ustar2": modes[selected] * averaged["spectrum_w"][selected] / ustar2,
        "kEc_over_cstar2": modes[selected]
        * averaged["spectrum_scalar"][selected]
        / cstar**2,
        "sample_height_m": averaged["spectrum_height_m"][selected],
    }
    np.savetxt(
        output / "spectra.csv",
        np.column_stack(tuple(spectrum_columns.values())),
        delimiter=",",
        header=",".join(spectrum_columns),
        comments="",
    )


def run(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    if jax.device_count() != 1:
        raise RuntimeError("the Andrén LASD runner currently requires one JAX device")
    dtype = getattr(jnp, args.dtype)
    physical_grid = UniformGrid(args.nx, args.ny, args.nz, 4000.0, 2000.0, 1500.0)
    mechanical_scales = ScaleSystem(1500.0, 10.0)
    scalar_scales = PassiveScalarScaleSystem(mechanical_scales, 1.0)
    grid = mechanical_scales.to_execution_grid(physical_grid)
    decomposition = EqualZSlab(
        grid,
        MeshTopology((MeshAxis("z", 1),)),
        DistributionSpec.z_slab(),
    )
    algebra = build_zslab_interpreter(decomposition, addressable_shards=(0,))
    pressure_solver = build_spectral_fd_pressure_adapter(
        decomposition,
        addressable_shards=(0,),
        runtime=runtime_from_initialized_jax(jax),
        dtype=args.dtype,
        method=args.method,
    )
    momentum_lasd = LagrangianScaleDependentDynamic(
        update_interval=args.lasd_update_interval
    )
    scalar_lasd = LagrangianScaleDependentScalarFlux()
    scalar_boundary = ScalarFluxBoundary(
        scalar_scales.to_execution_concentration_flux(
            SURFACE_SCALAR_FLUX / AIR_DENSITY
        ),
        0.0,
    )
    model = BoussinesqModel(
        DryFlowModel(
            ConservativeAdvection(),
            KinematicPressureGradient(0.0, 0.0),
            NeutralLogWall(mechanical_scales.to_execution_length(0.1), 0.4),
            momentum_lasd,
            CoriolisGeostrophic(
                mechanical_scales.to_execution_inverse_time(andren.F_CORIOLIS),
                1.0,
                0.0,
                mechanical_scales.to_execution_inverse_time(andren.F_CORIOLIS),
            ),
        ),
        ConservativeScalarAdvection(),
        scalar_lasd,
        NoBuoyancy(),
        NoRayleighDamping(),
        scalar_boundary,
    )
    config = AB2Config(mechanical_scales.to_execution_time(args.dt))
    closure_fingerprint = momentum_lasd.fingerprint + "|" + scalar_lasd.fingerprint
    scale_fingerprint = scalar_scales.fingerprint
    if args.restart is None:
        candidate = andren.andren_initial_velocity(
            physical_grid,
            decomposition,
            mechanical_scales,
            dtype,
            args,
        )
        projected = project(
            candidate,
            dt=config.dt,
            normal_boundary=VerticalBoundary(0.0, 0.0),
            algebra=algebra,
            pressure_solver=pressure_solver,
        )
        scalar = AddressableField(
            PassiveScalarConcentration,
            Cell,
            decomposition.regions(Cell),
            Accepted,
            jnp.zeros(
                (1, physical_grid.nz, physical_grid.ny, physical_grid.nx),
                dtype=dtype,
            ),
        )
        fields = algebra.initialize_lasd_closure(
            BoussinesqFields(projected.velocity, scalar),
            model,
        )
        state = cold_start_boussinesq(
            fields,
            clock=AcceptedClock(0.0, 0),
            config=config,
        )
        history_rows: list[dict] = []
        profile_times: list[float] = []
        profile_samples: list[dict] = []
    else:
        state = load_boussinesq_checkpoint(
            args.restart,
            layout=ZSlabCheckpointLayout(decomposition, (0,), jnp.asarray),
            config=config,
            scale_fingerprint=scale_fingerprint,
            closure_fingerprint=closure_fingerprint,
        )
        source = args.restart.parent
        history_rows = _read_history(source / "history.csv")
        profile_times, profile_samples = _load_statistics_state(
            source / "statistics_samples.npz"
        )

    if args.fig13_budget and args.restart is not None:
        budget_times, budget_samples = fig13_budget.load_samples(
            args.restart.parent / "fig13_budget_samples.npz"
        )
    else:
        budget_times, budget_samples = [], []

    vector_field = BoussinesqVectorField(algebra, model)
    closure_event = LasdAcceptedStepEvent(algebra, model, config.dt)
    target_steps = int(round(args.hours * 3600.0 / args.dt))
    requested_steps = target_steps - state.clock.step
    if requested_steps <= 0:
        raise ValueError("target time does not exceed the checkpoint time")
    warned_cfl = False
    warned_lasd = False
    started = time.perf_counter()
    last_result = None
    initial_step = state.clock.step
    for iteration in range(requested_steps):
        last_result = step_boussinesq(
            state,
            config=config,
            environment=None,
            vector_field=vector_field,
            normal_boundary=lambda _clock, _environment: VerticalBoundary(0.0, 0.0),
            algebra=algebra,
            pressure_solver=pressure_solver,
            closure_event=closure_event,
        )
        state = last_result.state
        final_iteration = iteration + 1 == requested_steps
        if state.clock.step % args.sample_every == 0 or final_iteration:
            divergence = last_result.diagnostic.projection.divergence.payload
            divergence.block_until_ready()
            history, profiles = instantaneous_diagnostics(
                state,
                divergence,
                physical_grid=physical_grid,
                mechanical_scales=mechanical_scales,
                scalar_scales=scalar_scales,
                algebra=algebra,
                model=model,
                args=args,
            )
            if not all(math.isfinite(value) for value in history.values()):
                raise FloatingPointError(
                    f"non-finite diagnostic at accepted step {state.clock.step}"
                )
            history_rows.append(history)
            profile_times.append(history["time_seconds"])
            profile_samples.append(profiles)
            if args.fig13_budget:
                budget_samples.append(
                    fig13_budget.observe(
                        state,
                        last_result.diagnostic.projection.pressure,
                        vector_field=vector_field,
                        algebra=algebra,
                        model=model,
                        physical_grid=physical_grid,
                        mechanical_scales=mechanical_scales,
                        scalar_scales=scalar_scales,
                        diagnostic_constants=ANDREN_DIAGNOSTIC_CONSTANTS,
                    )
                )
                budget_times.append(history["time_seconds"])
            if history["cfl"] > args.max_cfl_warning and not warned_cfl:
                warnings.warn(
                    f"CFL {history['cfl']:.3f} exceeds {args.max_cfl_warning:.3f}",
                    stacklevel=1,
                )
                warned_cfl = True
            if history["lasd_cfl"] >= 1.0 and not warned_lasd:
                warnings.warn(
                    f"LASD trajectory CFL {history['lasd_cfl']:.3f} is not below one",
                    stacklevel=1,
                )
                warned_lasd = True
        if state.clock.step % args.checkpoint_every == 0 or final_iteration:
            _save_progress(
                args,
                state,
                history_rows,
                profile_times,
                profile_samples,
                scale_fingerprint=scale_fingerprint,
                closure_fingerprint=closure_fingerprint,
                budget_times=budget_times if args.fig13_budget else None,
                budget_samples=budget_samples if args.fig13_budget else None,
            )
        if state.clock.step % args.log_every == 0:
            latest = history_rows[-1] if history_rows else None
            if latest is not None:
                print(
                    f"step={state.clock.step} "
                    f"t={latest['time_seconds'] / 3600.0:.3f} h "
                    f"u*={latest['ustar']:.4f} CFL={latest['cfl']:.3f} "
                    f"LASD-CFL={latest['lasd_cfl']:.3f} "
                    f"elapsed={time.perf_counter() - started:.1f} s",
                    flush=True,
                )
    if last_result is None:
        raise RuntimeError("benchmark executed no steps")

    final_time = mechanical_scales.from_execution_time(state.clock.time)
    averaging_start = (
        0.0
        if args.quick
        else max(7.0 / andren.F_CORIOLIS, final_time - 3.0 / andren.F_CORIOLIS)
    )
    selected = [
        sample
        for sample_time, sample in zip(profile_times, profile_samples, strict=True)
        if sample_time >= averaging_start
    ]
    if not selected:
        selected = [profile_samples[-1]]
    averaged = _average_profile_samples(selected)
    selected_history = [
        row for row in history_rows if row["time_seconds"] >= averaging_start
    ] or [history_rows[-1]]
    statistics_ustar = float(np.mean([row["ustar"] for row in selected_history]))
    _write_profiles(args.output, averaged, statistics_ustar)
    if args.fig13_budget:
        budget = fig13_budget.averaged_budget(
            budget_times,
            budget_samples,
            ustar=statistics_ustar,
            dz=physical_grid.dz,
        )
        fig13_budget.write_profile(args.output / "fig13_budget_profiles.csv", budget)
    elapsed = time.perf_counter() - started
    reference = json.loads(andren.REFERENCE_RESULTS.read_text())
    published = np.asarray(tuple(reference["ustar_over_ug"].values()))
    ratio = statistics_ustar / andren.GEOSTROPHIC_SPEED
    cstar = (SURFACE_SCALAR_FLUX / AIR_DENSITY) / statistics_ustar
    normalized_sgs_scalar_variance = averaged["sgs_scalar_variance"] / cstar**2
    normalized_integrated_total_tke = (
        andren.F_CORIOLIS
        * np.asarray([row["integrated_total_tke_m3_s2"] for row in selected_history])
        / statistics_ustar**3
    )
    summary = {
        "schema": "wireles.andren1994.lasd-passive-scalar.v1",
        "case": {
            "citation": "Andren et al. (1994)",
            "comparison_role": "external fifth SGS closure family",
            "passive_scalar_surface_flux_kg_m2_s": SURFACE_SCALAR_FLUX,
            "air_density_kg_m3": AIR_DENSITY,
            "diagnostic_sgs_energy": True,
            "diagnostic_sgs_scalar_variance": True,
        },
        "grid": {
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
            "lx": 4000.0,
            "ly": 2000.0,
            "lz": 1500.0,
        },
        "physics": {
            "momentum_sgs": momentum_lasd.fingerprint,
            "scalar_sgs": scalar_lasd.fingerprint,
            "surface_scalar_flux": SURFACE_SCALAR_FLUX,
            "neutral_log_wall_roughness_m": 0.1,
            "neutral_log_wall_von_karman": 0.4,
            "sgs_energy_kind": (
                "diagnostic local equilibrium with log-wall shear, not prognostic"
            ),
            "sgs_scalar_variance_kind": (
                "horizontal-budget diagnostic of full q_i grad_i c with "
                "flux-consistent wall gradient"
            ),
            "diagnostic_fingerprint": ANDREN_DIAGNOSTIC_CONSTANTS.fingerprint,
        },
        "runtime": {
            "dtype": args.dtype,
            "dt": args.dt,
            "completed_hours": final_time / 3600.0,
            "accepted_steps": state.clock.step,
            "elapsed_seconds": elapsed,
            "steps_per_second": (state.clock.step - initial_step) / elapsed,
            "samples_averaged": len(selected),
        },
        "comparison": {
            "statistics_ustar_m_s": statistics_ustar,
            "ustar_over_ug": ratio,
            "published_ustar_over_ug_min": float(published.min()),
            "published_ustar_over_ug_max": float(published.max()),
            "inside_published_envelope": bool(
                published.min() <= ratio <= published.max()
            ),
            "mean_normalized_integrated_total_tke": float(
                np.mean(normalized_integrated_total_tke)
            ),
            "first_cell_diagnostic_sgs_scalar_variance_over_cstar2": float(
                normalized_sgs_scalar_variance[0]
            ),
            "maximum_diagnostic_sgs_scalar_variance_over_cstar2": float(
                np.max(normalized_sgs_scalar_variance)
            ),
            "model_count_interpretation": (
                "four paper SGS/code families plus WIRE-LES LASD; "
                "the paper plots two Mason variants"
            ),
        },
        "acceptance": {
            "finite": all(np.all(np.isfinite(value)) for value in averaged.values()),
            "maximum_cfl": max(row["cfl"] for row in history_rows),
            "maximum_lasd_cfl": max(row["lasd_cfl"] for row in history_rows),
            "maximum_divergence": max(row["max_divergence"] for row in history_rows),
        },
        "final": history_rows[-1],
        "reference": reference,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "configuration.json").write_text(
        json.dumps(
            vars(args)
            | {
                "output": str(args.output),
                "restart": None if args.restart is None else str(args.restart),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
