#!/usr/bin/env python3
"""Run Nieuwstadt et al. (1993) with the new semantic JAX-Wind stack."""

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
    os.environ.get(
        "JAXWIND_SPECTRAL_FD_SOURCE",
        ROOT / "external" / "bw1000_benchmark",
    )
)
for source in (ROOT, SOURCE, PRESSURE_SOURCE):
    if source.exists() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from spectral_fd import runtime_from_initialized_jax  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    Accepted,
    AcceptedClock,
    AddressableField,
    BoussinesqScaleSystem,
    Candidate,
    Cell,
    DistributionSpec,
    EqualZSlab,
    MeshAxis,
    MeshTopology,
    PotentialTemperaturePerturbation,
    ScaleSystem,
    UniformGrid,
    VerticalBoundary,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from jaxwind.integrators import AB2Config, cold_start_boussinesq, step_boussinesq  # noqa: E402
from jaxwind.interpreters.jax_zslab import (  # noqa: E402
    ZFaceFieldContext,
    build_zslab_interpreter,
)
from jaxwind.operators import VelocityVector, project  # noqa: E402
from jaxwind.physics import (  # noqa: E402
    BoussinesqFields,
    BoussinesqModel,
    BoussinesqVectorField,
    ConservativeAdvection,
    ConservativeScalarAdvection,
    DiagnosticLasdConstants,
    DryFlowModel,
    KinematicPressureGradient,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LasdAcceptedStepEvent,
    LinearBoussinesqBuoyancy,
    NeutralLogWall,
    NoRayleighDamping,
    NoRotation,
    ScalarFluxBoundary,
)
from jaxwind.pressure import build_spectral_fd_pressure_adapter  # noqa: E402
from jaxwind.runners._toml import dumps as toml_dumps  # noqa: E402


HERE = Path(__file__).resolve().parent
THETA0 = 300.0
GRAVITY = 9.81
SURFACE_THETA_FLUX = 0.06
ZI0 = 1600.0
INITIAL_ZI_FRACTION = 0.844
ZI_SEARCH_MAX_FRACTION = 1.15
STABLE_THETA_GRADIENT = 0.003
ROUGHNESS_LENGTH = 0.16
SPECTRUM_LEVEL_FRACTIONS = np.asarray((0.2, 0.6, 1.0), dtype=np.float64)
SGS_DISSIPATION_COEFFICIENT = 0.93
NIEUWSTADT_DIAGNOSTIC_CONSTANTS = DiagnosticLasdConstants(
    horizontal_homogeneous_wall=True
)


def convective_scales(zi: float) -> tuple[float, float, float]:
    wstar = (GRAVITY * SURFACE_THETA_FLUX * zi / THETA0) ** (1.0 / 3.0)
    theta_star = SURFACE_THETA_FLUX / wstar
    return wstar, theta_star, zi / wstar


WSTAR0, THETA_STAR0, TSTAR0 = convective_scales(ZI0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--dt", type=float, default=1.25)
    parser.add_argument("--steps", type=int, default=9646)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=96)
    parser.add_argument("--lasd-update-interval", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--method", choices=("transpose", "spike"), default="spike")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "Nieuwstadt1993_new",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.nx = args.ny = 8
        args.nz = 8
        args.dt = 0.25
        args.steps = 8
        args.sample_every = 1
        args.log_every = 1
        args.lasd_update_interval = 2
    if args.max_steps is not None:
        args.steps = args.max_steps
    if min(args.nx, args.ny, args.nz) <= 1:
        parser.error("all grid dimensions must exceed one")
    if args.dt <= 0.0 or args.steps <= 0:
        parser.error("dt and steps must be positive")
    if min(args.sample_every, args.log_every, args.lasd_update_interval) <= 0:
        parser.error("sampling, logging, and LASD intervals must be positive")
    return args


def _plane_profile(value) -> np.ndarray:
    return np.asarray(jax.device_get(jnp.mean(value, axis=(0, 2, 3))), dtype=np.float64)


def _w_at_cells(velocity):
    upper = velocity.z.owned.payload
    lower_plane = jnp.broadcast_to(
        jnp.asarray(velocity.z.lower_boundary, dtype=upper.dtype),
        upper.shape[2:],
    )
    lower = jnp.concatenate((lower_plane[None, None], upper[:, :-1]), axis=1)
    return 0.5 * (lower + upper)


def _radial_spectrum(
    field: np.ndarray,
    dx: float,
    dy: float,
    edges: np.ndarray,
) -> np.ndarray:
    nx, ny = field.shape
    transformed = np.fft.fft2(field) / (nx * ny)
    energy = np.abs(transformed) ** 2
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2) * ZI0
    bins = np.digitize(radius.ravel(), edges) - 1
    valid = (bins >= 0) & (bins < edges.size - 1)
    result = np.bincount(
        bins[valid], weights=energy.ravel()[valid], minlength=edges.size - 1
    )
    return result[: edges.size - 1]


def _initial_fields(
    *,
    args: argparse.Namespace,
    physical_grid: UniformGrid,
    decomposition: EqualZSlab,
    mechanical_scales: ScaleSystem,
    thermal_scales: BoussinesqScaleSystem,
    algebra,
    pressure_solver,
    config: AB2Config,
):
    dtype = getattr(jnp, args.dtype)
    shape = (1, physical_grid.nz, physical_grid.ny, physical_grid.nx)
    key = jax.random.PRNGKey(args.seed)
    random_theta = jax.random.uniform(
        key,
        shape,
        minval=-0.5,
        maxval=0.5,
        dtype=dtype,
    )
    z = (jnp.arange(physical_grid.nz, dtype=dtype) + 0.5) * physical_grid.dz
    z_face = (jnp.arange(physical_grid.nz, dtype=dtype) + 1.0) * physical_grid.dz
    initial_zi = INITIAL_ZI_FRACTION * ZI0
    cell_weight = jnp.maximum(1.0 - z / initial_zi, 0.0)
    face_weight = jnp.maximum(1.0 - z_face / initial_zi, 0.0)
    theta_perturbation = jnp.where(
        (z < initial_zi)[None, :, None, None],
        0.1 * random_theta * cell_weight[None, :, None, None] * THETA_STAR0,
        (z - initial_zi)[None, :, None, None] * STABLE_THETA_GRADIENT,
    )
    w = jnp.where(
        (z_face < initial_zi)[None, :, None, None],
        0.1 * random_theta * face_weight[None, :, None, None] * WSTAR0,
        0.0,
    )
    zeros = jnp.zeros(shape, dtype=dtype)
    cell_regions = decomposition.regions(Cell)
    face_regions = decomposition.regions(ZFace)
    candidate = VelocityVector(
        AddressableField(XVelocity, Cell, cell_regions, Candidate, zeros),
        AddressableField(YVelocity, Cell, cell_regions, Candidate, zeros),
        ZFaceFieldContext(
            AddressableField(
                VerticalVelocity,
                ZFace,
                face_regions,
                Candidate,
                mechanical_scales.to_execution_velocity(w),
            ),
            jnp.zeros((physical_grid.ny, physical_grid.nx), dtype=dtype),
        ),
    )
    projected = project(
        candidate,
        dt=config.dt,
        normal_boundary=VerticalBoundary(0.0, 0.0),
        algebra=algebra,
        pressure_solver=pressure_solver,
    )
    scalar = AddressableField(
        PotentialTemperaturePerturbation,
        Cell,
        cell_regions,
        Accepted,
        thermal_scales.to_execution_potential_temperature(theta_perturbation),
    )
    return BoussinesqFields(projected.velocity, scalar)


def _physical_arrays(state, pressure, mechanical_scales, thermal_scales):
    velocity = state.fields.velocity
    u = mechanical_scales.from_execution_velocity(velocity.x.payload)
    v = mechanical_scales.from_execution_velocity(velocity.y.payload)
    w = mechanical_scales.from_execution_velocity(_w_at_cells(velocity))
    theta = THETA0 + thermal_scales.from_execution_potential_temperature(
        state.fields.potential_temperature.payload
    )
    p = mechanical_scales.kinematic_pressure * pressure.payload
    return u, v, w, theta, p


def snapshot_statistics(
    state,
    pressure,
    *,
    physical_grid: UniformGrid,
    mechanical_scales: ScaleSystem,
    thermal_scales: BoussinesqScaleSystem,
    algebra,
    model,
    spectrum_edges: np.ndarray,
) -> dict:
    u_jax, v_jax, w_jax, theta_jax, p_jax = _physical_arrays(
        state, pressure, mechanical_scales, thermal_scales
    )
    u = np.asarray(jax.device_get(u_jax[0])).transpose(1, 2, 0)
    v = np.asarray(jax.device_get(v_jax[0])).transpose(1, 2, 0)
    w = np.asarray(jax.device_get(w_jax[0])).transpose(1, 2, 0)
    theta = np.asarray(jax.device_get(theta_jax[0])).transpose(1, 2, 0)
    pressure_physical = np.asarray(jax.device_get(p_jax[0])).transpose(1, 2, 0)
    z = (np.arange(physical_grid.nz) + 0.5) * physical_grid.dz

    u_mean = np.mean(u, axis=(0, 1))
    v_mean = np.mean(v, axis=(0, 1))
    w_mean = np.mean(w, axis=(0, 1))
    theta_mean = np.mean(theta, axis=(0, 1))
    p_mean = np.mean(pressure_physical, axis=(0, 1))
    up = u - u_mean[None, None]
    vp = v - v_mean[None, None]
    wp = w - w_mean[None, None]
    thetap = theta - theta_mean[None, None]
    pp = pressure_physical - p_mean[None, None]

    context = algebra.boussinesq_context(state.fields)
    diagnostic = algebra.lasd_diagnostic_fields(
        context,
        model.momentum.sgs,
        model.scalar_sgs,
        model.scalar_boundary,
        constants=NIEUWSTADT_DIAGNOSTIC_CONSTANTS,
        wall=model.momentum.wall,
    )
    if not bool(jnp.all(jnp.isfinite(diagnostic.scalar_variance))):
        debug = {}
        for name in (
            "momentum_diffusivity",
            "scalar_diffusivity",
            "scalar_flux_x",
            "scalar_flux_y",
            "scalar_flux_z",
            "sgs_tke",
            "scalar_variance_numerator",
            "scalar_variance",
        ):
            values = getattr(diagnostic, name)
            finite = jnp.isfinite(values)
            debug[name] = {
                "nonfinite": int(values.size - jnp.count_nonzero(finite)),
                "finite_min": float(jnp.min(jnp.where(finite, values, jnp.inf))),
                "finite_max": float(jnp.max(jnp.where(finite, values, -jnp.inf))),
            }
        raise FloatingPointError(
            "non-finite new-stack LASD scalar-variance diagnostic: "
            + json.dumps(debug, sort_keys=True)
        )
    e_sgs_field = diagnostic.sgs_tke * mechanical_scales.velocity**2
    theta_var_sgs_field = (
        diagnostic.scalar_variance
        * thermal_scales.potential_temperature_difference**2
    )
    upper_scalar_flux = thermal_scales.from_execution_temperature_flux(
        diagnostic.scalar_flux_z
    )
    lower_scalar_flux = jnp.concatenate(
        (
            jnp.full_like(upper_scalar_flux[:, :1], SURFACE_THETA_FLUX),
            upper_scalar_flux[:, :-1],
        ),
        axis=1,
    )
    sgs_heat_flux = _plane_profile(0.5 * (lower_scalar_flux + upper_scalar_flux))
    e_sgs = _plane_profile(e_sgs_field)
    theta_var_sgs = _plane_profile(theta_var_sgs_field)

    velocity_face = state.fields.velocity.z.owned.payload
    theta_upper = context.arrays.theta_upper
    w_face_prime = velocity_face - jnp.mean(
        velocity_face, axis=(0, 2, 3), keepdims=True
    )
    theta_face_prime = theta_upper - jnp.mean(
        theta_upper, axis=(0, 2, 3), keepdims=True
    )
    resolved_upper = (
        w_face_prime
        * theta_face_prime
        * mechanical_scales.velocity
        * thermal_scales.potential_temperature_difference
    )
    resolved_upper_profile = _plane_profile(resolved_upper)
    resolved_heat_flux = 0.5 * np.concatenate(
        (resolved_upper_profile[:1], resolved_upper_profile[:-1] + resolved_upper_profile[1:])
    )
    total_heat_flux = resolved_heat_flux + sgs_heat_flux
    # The weak gravity-wave flux above the capping inversion can contain a
    # deeper instantaneous pocket than the entrainment-zone minimum.  The
    # paper case grows only a few percent over zi0, so keep the bulk-height
    # diagnostic on the physically contiguous primary inversion.
    inversion_search = z <= ZI_SEARCH_MAX_FRACTION * ZI0
    zi_index = np.flatnonzero(inversion_search)[
        np.argmin(total_heat_flux[inversion_search])
    ]
    zi = float(z[zi_index])
    wstar, theta_star, _ = convective_scales(zi)

    resolved_tke = 0.5 * np.mean(up * up + vp * vp + wp * wp, axis=(0, 1))
    energy_profile = resolved_tke + e_sgs
    energy_bl = float(np.mean(energy_profile[z <= zi]))
    component_sgs = (2.0 / 3.0) * e_sgs
    delta = (physical_grid.dx * physical_grid.dy * physical_grid.dz) ** (1.0 / 3.0)
    epsilon_zi = SGS_DISSIPATION_COEFFICIENT * np.maximum(e_sgs, 0.0) ** 1.5 / delta * zi

    updraft = wp > 0.0
    count = np.sum(updraft, axis=(0, 1))
    alpha_u = count / float(physical_grid.nx * physical_grid.ny)
    w_u = np.divide(
        np.sum(np.where(updraft, w, 0.0), axis=(0, 1)),
        count,
        out=np.zeros_like(w_mean),
        where=count > 0,
    )
    theta_u_excess = np.divide(
        np.sum(np.where(updraft, thetap, 0.0), axis=(0, 1)),
        count,
        out=np.zeros_like(theta_mean),
        where=count > 0,
    )
    resolved_energy = 0.5 * (up * up + vp * vp + wp * wp)
    w_transport = np.mean(wp * resolved_energy, axis=(0, 1))
    p_transport = np.mean(pp * wp, axis=(0, 1))

    level_indices = np.asarray(
        [int(np.argmin(np.abs(z - fraction * zi))) for fraction in SPECTRUM_LEVEL_FRACTIONS]
    )
    spectra_u = []
    spectra_w = []
    spectra_theta = []
    for level in level_indices:
        spectra_u.append(
            0.5
            * (
                _radial_spectrum(up[:, :, level], physical_grid.dx, physical_grid.dy, spectrum_edges)
                + _radial_spectrum(vp[:, :, level], physical_grid.dx, physical_grid.dy, spectrum_edges)
            )
        )
        spectra_w.append(
            _radial_spectrum(wp[:, :, level], physical_grid.dx, physical_grid.dy, spectrum_edges)
        )
        spectra_theta.append(
            _radial_spectrum(thetap[:, :, level], physical_grid.dx, physical_grid.dy, spectrum_edges)
        )

    return {
        "zi": zi,
        "wstar": wstar,
        "theta_star": theta_star,
        "energy_bl": energy_bl,
        "theta_mean": theta_mean,
        "heat_flux": total_heat_flux,
        "heat_flux_resolved": resolved_heat_flux,
        "heat_flux_sgs": sgs_heat_flux,
        "epsilon_zi": epsilon_zi,
        "sgs_tke": e_sgs,
        "u_var_resolved": np.mean(up * up, axis=(0, 1)),
        "v_var_resolved": np.mean(vp * vp, axis=(0, 1)),
        "w_var_resolved": np.mean(wp * wp, axis=(0, 1)),
        "component_var_sgs": component_sgs,
        "theta_var_resolved": np.mean(thetap * thetap, axis=(0, 1)),
        "theta_var_sgs": theta_var_sgs,
        "p_var": np.mean(pp * pp, axis=(0, 1)),
        "w3": np.mean(wp * wp * wp, axis=(0, 1)),
        "w_transport": w_transport,
        "p_transport": p_transport,
        "alpha_u": alpha_u,
        "w_u": w_u,
        "theta_u_excess": theta_u_excess,
        "spectra_u": np.asarray(spectra_u),
        "spectra_w": np.asarray(spectra_w),
        "spectra_theta": np.asarray(spectra_theta),
        "spectrum_level_z": z[level_indices],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots_from_files(output: Path) -> None:
    """Render the complete paper-aligned diagnostic set from saved outputs."""
    mpl_cache = output / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "profiles.csv").open(newline="") as stream:
        profile_rows = list(csv.DictReader(stream))
    with (output / "time_series.csv").open(newline="") as stream:
        time_rows = list(csv.DictReader(stream))
    profiles = {
        key: np.asarray([float(row[key]) for row in profile_rows])
        for key in profile_rows[0]
    }
    times = {
        key: np.asarray([float(row[key]) for row in time_rows])
        for key in time_rows[0]
    }
    stats = np.load(output / "benchmark_stats.npz")
    z_zi = profiles["z_over_zi"]

    def finish(path: str) -> None:
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output / path, dpi=180)
        plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(times["time_over_tstar0"], times["energy_bl_over_wstar0_sq"])
    plt.xlabel(r"$t/t_{*0}$")
    plt.ylabel(r"$\langle E\rangle/w_{*0}^2$")
    finish("fig01_energy_time.png")

    plt.figure(figsize=(6, 4))
    plt.plot(profiles["heat_flux_total_over_qs"], z_zi, label="total")
    plt.plot(profiles["heat_flux_resolved_over_qs"], z_zi, "--", label="resolved")
    plt.plot(profiles["heat_flux_sgs_over_qs"], z_zi, ":", label="SGS")
    plt.axvline(0.0, color="0.4", lw=0.8)
    plt.xlabel(r"$\langle w'\theta'\rangle/Q_s$")
    plt.ylabel(r"$z/z_i$")
    plt.legend(fontsize=8)
    finish("fig02_heat_flux.png")

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharey=True)
    panels = (
        ("w_var_over_wstar_sq", "w_var_resolved_over_wstar_sq", "w_var_sgs_over_wstar_sq", r"$\langle w'^2\rangle/w_*^2$"),
        ("horizontal_var_over_wstar_sq", "horizontal_var_resolved_over_wstar_sq", "horizontal_var_sgs_over_wstar_sq", r"$\langle u_h'^2\rangle/w_*^2$"),
        ("theta_var_over_thetastar_sq", "theta_var_resolved_over_thetastar_sq", "theta_var_sgs_over_thetastar_sq", r"$\langle\theta'^2\rangle/\theta_*^2$"),
    )
    for axis, (total, resolved, sgs, label) in zip(axes.ravel()[:3], panels, strict=True):
        axis.plot(profiles[total], z_zi, label="total")
        axis.plot(profiles[resolved], z_zi, "--", label="resolved")
        axis.plot(profiles[sgs], z_zi, ":", label="SGS")
        axis.set_xlabel(label)
        axis.legend(fontsize=7)
    axes.ravel()[3].plot(profiles["p_var_over_wstar4"], z_zi)
    axes.ravel()[3].set_xlabel(r"$\langle p'^2\rangle/w_*^4$")
    for axis in axes.ravel():
        axis.set_ylabel(r"$z/z_i$")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "fig03_06_variances.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    axes[0].plot(profiles["w3_over_wstar3"], z_zi)
    axes[0].set_xlabel(r"$\langle w'^3\rangle/w_*^3$")
    axes[1].plot(profiles["skewness"], z_zi)
    axes[1].set_xlabel(r"$Sk_w$")
    for axis in axes:
        axis.set_ylabel(r"$z/z_i$")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "fig07_08_higher_moments.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(6, 4))
    plt.plot(profiles["epsilon_zi_over_wstar3"], z_zi)
    plt.xlabel(r"$\langle\epsilon\rangle z_i/w_*^3$")
    plt.ylabel(r"$z/z_i$")
    plt.xlim(left=0.0)
    finish("fig09_dissipation.png")
    (output / "fig09_11_energy_budget_terms.png").write_bytes(
        (output / "fig09_dissipation.png").read_bytes()
    )

    plt.figure(figsize=(6, 4))
    plt.plot(profiles["buoyancy_production"], z_zi, label="buoyancy")
    plt.plot(profiles["d_w_transport"], z_zi, label=r"$d\langle w'E'\rangle/dz$")
    plt.plot(profiles["d_p_transport"], z_zi, label=r"$d\langle p'w'\rangle/dz$")
    plt.xlabel("normalized budget term")
    plt.ylabel(r"$z/z_i$")
    plt.legend(fontsize=8)
    finish("fig10_11_energy_budget_terms.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, key, label in zip(
        axes,
        ("alpha_u", "w_u_over_wstar", "theta_u_excess_over_thetastar"),
        (r"$\alpha_u$", r"$w_u/w_*$", r"$(\theta_u-\langle\theta\rangle)/\theta_*$"),
        strict=True,
    ):
        axis.plot(profiles[key], z_zi)
        axis.set_xlabel(label)
        axis.set_ylabel(r"$z/z_i$")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "fig12_14_conditional_updrafts.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    kzi = np.asarray(stats["spectrum_kzi"])
    for spectrum_key, axis, title in zip(
        ("spectrum_u", "spectrum_w", "spectrum_theta"),
        axes,
        ("horizontal velocity", "vertical velocity", "temperature"),
        strict=True,
    ):
        values = np.asarray(stats[spectrum_key])
        for index, fraction in enumerate(SPECTRUM_LEVEL_FRACTIONS):
            mask = (kzi > 0.0) & (values[index] > 0.0)
            axis.loglog(kzi[mask], values[index, mask], label=f"z={fraction:.1f} zi")
        axis.set_title(title)
        axis.set_xlabel(r"$kz_{i0}$")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("radial spectrum")
    fig.tight_layout()
    fig.savefig(output / "fig15_17_spectra.png", dpi=180)
    plt.close(fig)


def save_outputs(
    output: Path,
    *,
    args,
    physical_grid,
    time_rows,
    selected,
    runtime_s,
    max_cfl,
    max_lasd_cfl,
    max_divergence,
    spectrum_edges,
) -> dict[str, float]:
    averaged = {
        key: np.mean([sample[key] for sample in selected], axis=0)
        for key in selected[0]
        if key not in ("zi", "wstar", "theta_star", "energy_bl")
    }
    zi_mean = float(np.mean([sample["zi"] for sample in selected]))
    wstar_mean = float(np.mean([sample["wstar"] for sample in selected]))
    theta_star_mean = float(np.mean([sample["theta_star"] for sample in selected]))
    z = (np.arange(physical_grid.nz) + 0.5) * physical_grid.dz
    component_sgs = averaged["component_var_sgs"]
    w_var_resolved = averaged["w_var_resolved"]
    w_var = w_var_resolved + component_sgs
    u_var = averaged["u_var_resolved"] + component_sgs
    v_var = averaged["v_var_resolved"] + component_sgs
    theta_var = averaged["theta_var_resolved"] + averaged["theta_var_sgs"]
    skewness = np.divide(
        averaged["w3"],
        np.maximum(w_var_resolved, 0.0) ** 1.5,
        out=np.zeros_like(w_var_resolved),
        where=w_var_resolved > 0.0,
    )
    transport_norm = wstar_mean**3 / zi_mean
    d_w_transport = np.gradient(averaged["w_transport"], z) / transport_norm
    d_p_transport = np.gradient(averaged["p_transport"], z) / transport_norm
    profiles = []
    for index, height in enumerate(z):
        profiles.append(
            {
                "z": height,
                "z_over_zi": height / zi_mean,
                "theta_mean": averaged["theta_mean"][index],
                "heat_flux_over_qs": averaged["heat_flux"][index] / SURFACE_THETA_FLUX,
                "heat_flux_resolved_over_qs": averaged["heat_flux_resolved"][index] / SURFACE_THETA_FLUX,
                "heat_flux_sgs_over_qs": averaged["heat_flux_sgs"][index] / SURFACE_THETA_FLUX,
                "heat_flux_total_over_qs": averaged["heat_flux"][index] / SURFACE_THETA_FLUX,
                "epsilon_zi_over_wstar3": averaged["epsilon_zi"][index] / wstar_mean**3,
                "w_var_over_wstar_sq": w_var[index] / wstar_mean**2,
                "w_var_resolved_over_wstar_sq": w_var_resolved[index] / wstar_mean**2,
                "w_var_sgs_over_wstar_sq": component_sgs[index] / wstar_mean**2,
                "horizontal_var_over_wstar_sq": 0.5 * (u_var[index] + v_var[index]) / wstar_mean**2,
                "horizontal_var_resolved_over_wstar_sq": 0.5 * (averaged["u_var_resolved"][index] + averaged["v_var_resolved"][index]) / wstar_mean**2,
                "horizontal_var_sgs_over_wstar_sq": component_sgs[index] / wstar_mean**2,
                "sgs_tke_over_wstar_sq": averaged["sgs_tke"][index] / wstar_mean**2,
                "theta_var_over_thetastar_sq": theta_var[index] / theta_star_mean**2,
                "theta_var_resolved_over_thetastar_sq": averaged["theta_var_resolved"][index] / theta_star_mean**2,
                "theta_var_sgs_over_thetastar_sq": averaged["theta_var_sgs"][index] / theta_star_mean**2,
                "p_var_over_wstar4": averaged["p_var"][index] / wstar_mean**4,
                "w3_over_wstar3": averaged["w3"][index] / wstar_mean**3,
                "skewness": skewness[index],
                "alpha_u": averaged["alpha_u"][index],
                "w_u_over_wstar": averaged["w_u"][index] / wstar_mean,
                "theta_u_excess_over_thetastar": averaged["theta_u_excess"][index] / theta_star_mean,
                "buoyancy_production": (GRAVITY / THETA0) * averaged["heat_flux"][index] / transport_norm,
                "d_w_transport": d_w_transport[index],
                "d_p_transport": d_p_transport[index],
            }
        )
    _write_csv(output / "profiles.csv", profiles)
    _write_csv(output / "time_series.csv", time_rows)

    inversion_search = z <= ZI_SEARCH_MAX_FRACTION * ZI0
    entrainment_flux = float(np.min(averaged["heat_flux"][inversion_search]))
    zi_flux_min = float(
        z[
            np.flatnonzero(inversion_search)[
                np.argmin(averaged["heat_flux"][inversion_search])
            ]
        ]
    )
    entrainment_ratio_instantaneous_mean = float(
        np.mean(
            [
                -np.min(sample["heat_flux"][inversion_search])
                / SURFACE_THETA_FLUX
                for sample in selected
            ]
        )
    )
    mixed = z <= 0.8 * zi_mean
    summary = {
        "sample_count": float(len(selected)),
        "zi0": ZI0,
        "wstar0": WSTAR0,
        "theta_star0": THETA_STAR0,
        "tstar0": TSTAR0,
        "zi_mean": zi_mean,
        "zi_flux_min": zi_flux_min,
        "zi_over_zi0": zi_mean / ZI0,
        "wstar_mean": wstar_mean,
        "wstar_over_wstar0": wstar_mean / WSTAR0,
        "theta_star_mean": theta_star_mean,
        "entrainment_ratio": -entrainment_flux / SURFACE_THETA_FLUX,
        "entrainment_ratio_instantaneous_mean": entrainment_ratio_instantaneous_mean,
        "theta_mixed_layer_mean": float(np.mean(averaged["theta_mean"][mixed])),
        "max_cfl": max_cfl,
        "max_lasd_cfl": max_lasd_cfl,
        "max_divergence": max_divergence,
        "runtime_s": runtime_s,
    }
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("quantity", "value"))
        writer.writerows(summary.items())

    spectra = {
        "u": averaged["spectra_u"],
        "w": averaged["spectra_w"],
        "theta": averaged["spectra_theta"],
    }
    np.savez(
        output / "benchmark_stats.npz",
        z=z,
        z_over_zi=z / zi_mean,
        spectrum_kzi=0.5 * (spectrum_edges[:-1] + spectrum_edges[1:]),
        spectrum_level_z=averaged["spectrum_level_z"],
        spectrum_level_fraction=SPECTRUM_LEVEL_FRACTIONS,
        spectrum_u=spectra["u"],
        spectrum_w=spectra["w"],
        spectrum_theta=spectra["theta"],
        **{f"profile_{key}": value for key, value in averaged.items()},
        **summary,
    )
    (output / "resolved_config.toml").write_text(
        toml_dumps(
            vars(args)
            | {
                "output_dir": str(args.output_dir),
                "implementation": "new semantic src/jaxwind",
                "pressure_api": "external spectral_fd runtime API",
                "integrator": "AB2",
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
                "scalar_stability_correction": {
                    "beta": 30.0,
                    "power": 2.0,
                },
            }
        )
    )
    make_plots_from_files(output)
    return summary


def run(args: argparse.Namespace) -> dict[str, float]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if jax.device_count() != 1:
        raise RuntimeError("the Nieuwstadt new-stack runner currently requires one JAX device")
    physical_grid = UniformGrid(args.nx, args.ny, args.nz, 6400.0, 6400.0, 2400.0)
    mechanical_scales = ScaleSystem(ZI0, WSTAR0)
    thermal_scales = BoussinesqScaleSystem(mechanical_scales, THETA_STAR0)
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
    buoyancy_coefficient = thermal_scales.to_execution_buoyancy_coefficient(
        gravity=GRAVITY,
        reference_potential_temperature=THETA0,
    )
    scalar_lasd = LagrangianScaleDependentScalarFlux(
        stability_buoyancy_coefficient=buoyancy_coefficient,
        stability_beta=30.0,
        stability_power=2.0,
    )
    model = BoussinesqModel(
        DryFlowModel(
            ConservativeAdvection(),
            KinematicPressureGradient(0.0, 0.0),
            NeutralLogWall(mechanical_scales.to_execution_length(ROUGHNESS_LENGTH)),
            momentum_lasd,
            NoRotation(),
        ),
        ConservativeScalarAdvection(),
        scalar_lasd,
        LinearBoussinesqBuoyancy(buoyancy_coefficient),
        NoRayleighDamping(),
        ScalarFluxBoundary(
            thermal_scales.to_execution_temperature_flux(SURFACE_THETA_FLUX),
            0.0,
        ),
    )
    config = AB2Config(mechanical_scales.to_execution_time(args.dt))
    fields = _initial_fields(
        args=args,
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
        fields,
        clock=AcceptedClock(0.0, 0),
        config=config,
    )
    vector_field = BoussinesqVectorField(algebra, model)
    closure_event = LasdAcceptedStepEvent(algebra, model, config.dt)
    maximum_mode = math.sqrt(2.0) * math.pi * max(args.nx, args.ny) * ZI0 / 6400.0
    spectrum_edges = np.linspace(0.0, maximum_mode, args.nx // 2 + 2)
    time_rows: list[dict] = []
    samples: list[tuple[float, dict]] = []
    max_cfl = 0.0
    max_lasd_cfl = 0.0
    max_divergence = 0.0
    started = time.perf_counter()
    for _ in range(args.steps):
        result = step_boussinesq(
            state,
            config=config,
            environment=None,
            vector_field=vector_field,
            normal_boundary=lambda _clock, _environment: VerticalBoundary(0.0, 0.0),
            algebra=algebra,
            pressure_solver=pressure_solver,
            closure_event=closure_event,
        )
        state = result.state
        final = state.clock.step == args.steps
        if state.clock.step % args.sample_every == 0 or final:
            divergence = result.diagnostic.projection.divergence.payload
            divergence.block_until_ready()
            u, v, w, _, _ = _physical_arrays(
                state,
                result.diagnostic.projection.pressure,
                mechanical_scales,
                thermal_scales,
            )
            directional_cfl = float(
                args.dt
                * jnp.max(
                    jnp.maximum(
                        jnp.maximum(
                            jnp.abs(u) / physical_grid.dx,
                            jnp.abs(v) / physical_grid.dy,
                        ),
                        jnp.abs(w) / physical_grid.dz,
                    )
                )
            )
            lasd_cfl = directional_cfl * args.lasd_update_interval
            divergence_si = float(
                jnp.max(jnp.abs(divergence)) * mechanical_scales.inverse_time
            )
            max_cfl = max(max_cfl, directional_cfl)
            max_lasd_cfl = max(max_lasd_cfl, lasd_cfl)
            max_divergence = max(max_divergence, divergence_si)
            statistics = snapshot_statistics(
                state,
                result.diagnostic.projection.pressure,
                physical_grid=physical_grid,
                mechanical_scales=mechanical_scales,
                thermal_scales=thermal_scales,
                algebra=algebra,
                model=model,
                spectrum_edges=spectrum_edges,
            )
            time_s = mechanical_scales.from_execution_time(state.clock.time)
            time_rows.append(
                {
                    "step": state.clock.step,
                    "time_s": time_s,
                    "time_over_tstar0": time_s / TSTAR0,
                    "zi": statistics["zi"],
                    "zi_over_zi0": statistics["zi"] / ZI0,
                    "wstar": statistics["wstar"],
                    "energy_bl_over_wstar0_sq": statistics["energy_bl"] / WSTAR0**2,
                }
            )
            samples.append((time_s / TSTAR0, statistics))
            nonfinite = [
                name
                for name, value in statistics.items()
                if not np.all(np.isfinite(value))
            ]
            if nonfinite:
                raise FloatingPointError(
                    f"non-finite diagnostic at accepted step {state.clock.step}: "
                    + ", ".join(nonfinite)
                )
            if lasd_cfl >= 1.0:
                warnings.warn(
                    f"LASD trajectory CFL {lasd_cfl:.3f} is not below one",
                    stacklevel=1,
                )
        if state.clock.step % args.log_every == 0 or final:
            latest = time_rows[-1] if time_rows else None
            if latest is not None:
                print(
                    f"step={state.clock.step} t/t*={latest['time_over_tstar0']:.3f} "
                    f"zi/zi0={latest['zi_over_zi0']:.4f} CFL={max_cfl:.4f} "
                    f"LASD-CFL={max_lasd_cfl:.4f} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    runtime_s = time.perf_counter() - started
    selected = [sample for normalized_time, sample in samples if 10.0 <= normalized_time <= 11.0]
    if not selected:
        selected = [sample for _, sample in samples]
    summary = save_outputs(
        args.output_dir,
        args=args,
        physical_grid=physical_grid,
        time_rows=time_rows,
        selected=selected,
        runtime_s=runtime_s,
        max_cfl=max_cfl,
        max_lasd_cfl=max_lasd_cfl,
        max_divergence=max_divergence,
        spectrum_edges=spectrum_edges,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
