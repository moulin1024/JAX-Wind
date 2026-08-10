#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parents[1]
JAX_DIR = ROOT / "legacy" / "jax"
sys.path.insert(0, str(JAX_DIR))

from run_single import (
    CONFIG_KEYS,
    RUN_DEFAULTS,
    dtype_for_precision,
    scaled_grid_lengths,
    scaled_time_step,
    sgs_dtype_for_precision,
)

LASD_SGS_DISSIPATION_COEFFICIENT = 0.93
SGS_SCALAR_VARIANCE_COEFFICIENT = 2.02
SPECTRUM_LEVEL_FRACTIONS = np.asarray((0.2, 0.6, 1.0), dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Nieuwstadt et al. (1993) dry CBL benchmark and generate paper-style diagnostics."
    )
    parser.add_argument("--config", type=Path, default=BENCHMARK_DIR / "configs" / "lasd_scalar.toml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results" / "Nieuwstadt1993")
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--average-start-tstar", type=float, default=10.0)
    parser.add_argument("--average-end-tstar", type=float, default=11.0)
    parser.add_argument("--max-steps", type=int, help="Optional shorter run length for smoke tests.")
    parser.add_argument("--steps", type=int, help="Override [time].steps.")
    parser.add_argument("--dt", type=float, help="Override physical time step in seconds.")
    parser.add_argument("--smag-cs", type=float, help="Override fixed Smagorinsky Cs.")
    parser.add_argument("--prandtl-t", type=float, help="Override turbulent Prandtl number for scalar fixed_prandtl path.")
    parser.add_argument(
        "--scalar-vertical-scheme",
        choices=("centered", "weno3", "weno5z"),
        help="Override the conservative vertical scalar face reconstruction.",
    )
    parser.add_argument("--coriolis-f", type=float, help="Override physical Coriolis parameter f in s^-1.")
    parser.add_argument("--geostrophic-u", type=float, help="Override physical geostrophic wind U_g in m/s.")
    parser.add_argument("--geostrophic-v", type=float, help="Override physical geostrophic wind V_g in m/s.")
    parser.add_argument("--wall-stress-model", choices=("dynamic_neutral", "dynamic-neutral", "prescribed_ustar", "prescribed-ustar"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-jit", action="store_false", dest="use_jit", default=None)
    return parser.parse_args()


def load_settings(path: Path) -> dict:
    settings = dict(RUN_DEFAULTS)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw.pop("benchmark", {})
    for section, section_values in raw.items():
        if section not in CONFIG_KEYS:
            valid = ", ".join((*sorted(CONFIG_KEYS), "benchmark"))
            raise ValueError(
                f"Unknown config section [{section}]. Valid sections: {valid}"
            )
        if not isinstance(section_values, dict):
            raise ValueError(f"Config section [{section}] must be a table.")
        key_map = CONFIG_KEYS[section]
        for key, value in section_values.items():
            if key not in key_map:
                valid = ", ".join(sorted(key_map))
                raise ValueError(
                    f"Unknown config key [{section}].{key}. Valid keys: {valid}"
                )
            settings[key_map[key]] = value
    if not isinstance(benchmark, dict):
        raise ValueError("Config section [benchmark] must be a table.")
    unknown_benchmark = set(benchmark) - {"initial_zi_fraction"}
    if unknown_benchmark:
        raise ValueError(
            "Unknown config key(s) in [benchmark]: "
            + ", ".join(sorted(unknown_benchmark))
        )
    settings["benchmark_initial_zi_fraction"] = float(
        benchmark.get("initial_zi_fraction", 0.844)
    )
    return settings


def build_params(settings: dict, jnp):
    from wireles_jax import Params

    lx_scaled, ly_scaled, lz_scaled, z_i = scaled_grid_lengths(settings)
    dt_scaled = scaled_time_step(settings, z_i)
    solver_dtype = dtype_for_precision(settings["precision"], jnp)
    return Params(
        nx=settings["nx"],
        ny=settings["ny"],
        nz=settings["nz"],
        lx=lx_scaled,
        ly=ly_scaled,
        lz=lz_scaled,
        nsteps=settings["steps"],
        dt=dt_scaled,
        c_count=settings["log_every"],
        u_fric=settings["u_fric"],
        zo=settings["zo"],
        bl_height=settings["bl_height"],
        z_i=z_i,
        vonk=settings["vonk"],
        pressure_force=settings["pressure_force"],
        pressure_force_height=settings["pressure_force_height"],
        coriolis_f=settings["coriolis_f"],
        geostrophic_u=settings["geostrophic_u"],
        geostrophic_v=settings["geostrophic_v"],
        initial_condition=settings["initial_condition"],
        momentum_wall_model=settings["momentum_wall_model"],
        wall_stress_model=settings["wall_stress_model"],
        initial_velocity_noise=settings["initial_velocity_noise"],
        molecular_viscosity=settings["molecular_viscosity"],
        molecular_diffusivity=settings["molecular_diffusivity"],
        rayleigh_number=settings["rayleigh_number"],
        rayleigh_prandtl=settings["rayleigh_prandtl"],
        fgr=settings["fgr"],
        tfr=settings["tfr"],
        sgs_model=settings["sgs_model"],
        cs_count=settings["cs_count"],
        smagorinsky_cs=settings["smag_cs"],
        sgs_delta_scale=settings["sgs_delta_scale"],
        time_scheme=settings["time_scheme"],
        projection_mode=settings["projection_mode"],
        horizontal_dealias=settings["horizontal_dealias"],
        pressure_filter_nyquist=settings["pressure_filter_nyquist"],
        top_boundary_condition=settings["top_boundary_condition"],
        radiation_brunt_vaisala=settings["radiation_brunt_vaisala"],
        thermo_enabled=settings["thermo_enabled"],
        moisture_enabled=settings["moisture_enabled"],
        theta0=settings["theta0"],
        g=settings["g"],
        theta_bc=settings["theta_bc"],
        theta_profile=settings["theta_profile"],
        theta_top_gradient=settings["theta_top_gradient"],
        theta_bottom=settings["theta_bottom"],
        theta_top=settings["theta_top"],
        theta_initial_gradient=settings["theta_initial_gradient"],
        theta_perturbation_amplitude=settings["theta_perturbation_amplitude"],
        theta_perturbation_height=settings["theta_perturbation_height"],
        cbl_mixed_layer_height=settings["cbl_mixed_layer_height"],
        cbl_inversion_strength=settings["cbl_inversion_strength"],
        cbl_inversion_thickness=settings["cbl_inversion_thickness"],
        cbl_free_atmosphere_gradient=settings["cbl_free_atmosphere_gradient"],
        surface_theta_flux=settings["surface_theta_flux"],
        qv0=settings["qv0"],
        qv_initial_gradient=settings["qv_initial_gradient"],
        surface_qv_flux=settings["surface_qv_flux"],
        qv_floor=settings["qv_floor"],
        scalar_sgs_model=settings["scalar_sgs_model"],
        prandtl_t=settings["prandtl_t"],
        schmidt_t=settings["schmidt_t"],
        scalar_stability_correction=settings["scalar_stability_correction"],
        scalar_stability_beta=settings["scalar_stability_beta"],
        scalar_stability_power=settings["scalar_stability_power"],
        scalar_lasd_min=settings["scalar_lasd_min"],
        scalar_lasd_max=settings["scalar_lasd_max"],
        scalar_vertical_scheme=settings["scalar_vertical_scheme"],
        dtype=solver_dtype,
        sgs_dtype=sgs_dtype_for_precision(settings["sgs_precision"], solver_dtype, jnp),
        use_jit=settings["use_jit"],
    )


def convective_scales(params, zi: float) -> tuple[float, float, float]:
    wstar = ((params.g / params.theta0) * params.surface_theta_flux * zi) ** (1.0 / 3.0)
    theta_star = params.surface_theta_flux / wstar
    tstar = zi / wstar
    return float(wstar), float(theta_star), float(tstar)


def physical_z(params) -> np.ndarray:
    centers = (np.arange(params.nz, dtype=np.float64) + 0.5) * float(params.dz)
    return centers * float(params.z_i)


def alternating_mode_amplitude(
    values: np.ndarray,
    z_over_zi: np.ndarray,
    lower: float = 0.07,
    upper: float = 0.35,
) -> float:
    """Amplitude of the detrended vertical 2-dz mode in the lower CBL."""
    mask = (z_over_zi >= lower) & (z_over_zi <= upper)
    selected_z = np.asarray(z_over_zi[mask], dtype=np.float64)
    selected_values = np.asarray(values[mask], dtype=np.float64)
    if selected_values.size < 4:
        return float("nan")
    degree = min(2, selected_values.size - 2)
    trend = np.polyval(np.polyfit(selected_z, selected_values, degree), selected_z)
    residual = selected_values - trend
    alternating = (-1.0) ** np.arange(selected_values.size)
    return float(np.dot(residual, alternating) / np.dot(alternating, alternating))


def radial_spectrum(field: np.ndarray, dx: float, dy: float, zi0: float, edges: np.ndarray) -> np.ndarray:
    """Return a Parseval-consistent spectrum per unit dimensionless wavenumber.

    ``np.fft.rfft2`` stores only nonnegative y wavenumbers, so modes away
    from zero and the y Nyquist mode represent both signs of ``ky``.  Sum
    those Hermitian-weighted modal energies in each radial shell and divide
    by the shell width in ``k * zi0``.  The resulting spectrum ``Phi`` obeys

        sum(Phi * delta(k * zi0)) = mean((field - mean(field))**2)

    whenever ``edges`` cover all energetic modes.
    """
    nx, ny = field.shape
    q = field - np.mean(field)
    qhat = np.fft.rfft2(q)
    power = (np.abs(qhat) ** 2) / float(nx * ny) ** 2
    hermitian_weight = np.full_like(power, 2.0, dtype=np.float64)
    hermitian_weight[:, 0] = 1.0
    if ny % 2 == 0:
        hermitian_weight[:, -1] = 1.0
    weighted_power = power * hermitian_weight
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(ny, d=dy)
    kzi = np.sqrt(kx[:, None] * kx[:, None] + ky[None, :] * ky[None, :]) * zi0
    bin_index = np.digitize(kzi.ravel(), edges) - 1
    valid = (bin_index >= 0) & (bin_index < edges.size - 1)
    shell_energy = np.bincount(
        bin_index[valid],
        weights=weighted_power.ravel()[valid],
        minlength=edges.size - 1,
    ).astype(np.float64)
    return shell_energy / np.diff(edges)


def _with_interior_np(template: np.ndarray, inner: np.ndarray) -> np.ndarray:
    del template
    return inner


def _ddxy_filter_np(q: np.ndarray, params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_hat = np.fft.rfft2(q, axes=(0, 1))
    if params.horizontal_dealias:
        x_mode = np.abs(np.fft.fftfreq(params.nx, d=1.0) * params.nx)
        y_mode = np.fft.rfftfreq(params.ny, d=1.0) * params.ny
        cutoff_x = np.rint(params.nx / (2.0 * params.fgr))
        cutoff_y = np.rint(params.ny / (2.0 * params.fgr))
        keep = (x_mode[:, None] < cutoff_x) & (y_mode[None, :] < cutoff_y)
        q_hat = q_hat * keep[:, :, None]
    kx = 2.0 * np.pi * np.fft.fftfreq(params.nx, d=float(params.dx))
    ky = 2.0 * np.pi * np.fft.rfftfreq(params.ny, d=float(params.dy))
    if params.nx % 2 == 0:
        kx[params.nx // 2] = 0.0
    if params.ny % 2 == 0:
        ky[-1] = 0.0
    q_filtered = np.fft.irfft2(q_hat, s=(params.nx, params.ny), axes=(0, 1)).real
    dqdx = np.fft.irfft2(1j * kx[:, None, None] * q_hat, s=(params.nx, params.ny), axes=(0, 1)).real
    dqdy = np.fft.irfft2(1j * ky[None, :, None] * q_hat, s=(params.nx, params.ny), axes=(0, 1)).real
    return _with_interior_np(q, q_filtered), _with_interior_np(q, dqdx), _with_interior_np(q, dqdy)


def _ddz_uv_np(q: np.ndarray, params) -> np.ndarray:
    if q.shape[2] == 1:
        return np.zeros_like(q)
    out = np.empty_like(q)
    out[:, :, 0] = (q[:, :, 1] - q[:, :, 0]) / float(params.dz)
    out[:, :, -1] = (q[:, :, -1] - q[:, :, -2]) / float(params.dz)
    if q.shape[2] > 2:
        out[:, :, 1:-1] = (q[:, :, 2:] - q[:, :, :-2]) / (2.0 * float(params.dz))
    return out


def _ddz_w_np(q: np.ndarray, params) -> np.ndarray:
    lower = np.concatenate((np.zeros_like(q[:, :, :1]), q[:, :, :-1]), axis=2)
    return (q - lower) / float(params.dz)


def _ddz_uv_face_np(q: np.ndarray, params) -> np.ndarray:
    out = np.zeros_like(q)
    if q.shape[2] > 1:
        out[:, :, :-1] = (
            q[:, :, 1:] - q[:, :, :-1]
        ) / float(params.dz)
    return out


def _filter_2d_wall_np(q: np.ndarray, params) -> np.ndarray:
    q_hat = np.fft.rfft2(q, axes=(0, 1))
    x_mode = np.fft.fftfreq(params.nx, d=1.0) * params.nx
    y_mode = np.fft.rfftfreq(params.ny, d=1.0) * params.ny
    cutoff_x = np.floor(params.nx / (2.0 * params.fgr * params.tfr))
    cutoff_y = np.floor(params.ny / (2.0 * params.fgr * params.tfr))
    keep = (
        (np.abs(x_mode)[:, None] < cutoff_x)
        & (y_mode[None, :] < cutoff_y)
    )
    return np.fft.irfft2(
        np.where(keep[:, :, None], q_hat, 0.0),
        s=(params.nx, params.ny),
        axes=(0, 1),
    ).real


def _wall_gradient_np(
    u: np.ndarray,
    v: np.ndarray,
    params,
) -> tuple[np.ndarray, np.ndarray]:
    if params.momentum_wall_model != "abl":
        zeros = np.zeros((params.nx, params.ny), dtype=u.dtype)
        return zeros, zeros
    u0 = _filter_2d_wall_np(u[:, :, :1], params)[:, :, 0]
    v0 = _filter_2d_wall_np(v[:, :, :1], params)[:, :, 0]
    speed = np.sqrt(u0 * u0 + v0 * v0)
    denom = np.log(float(params.wall_ref_height) / float(params.zo))
    valid = (speed > 1.0e-12) & (abs(denom) > 1.0e-12)
    safe_speed = np.where(speed > 1.0e-12, speed, 1.0)
    if params.wall_stress_model == "prescribed_ustar":
        ustar = np.where(valid, float(params.u_fric), 0.0)
    else:
        ustar = np.where(
            valid,
            speed * float(params.vonk) / (denom if abs(denom) > 1.0e-12 else 1.0),
            0.0,
        )
    shear_denom = safe_speed * float(params.vonk) * 0.5 * float(params.dz)
    return (
        np.where(valid, u0 * ustar / shear_denom, 0.0),
        np.where(valid, v0 * ustar / shear_denom, 0.0),
    )


def _momentum_vertical_gradients_np(
    u: np.ndarray,
    v: np.ndarray,
    params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dudz_face = _ddz_uv_face_np(u, params)
    dvdz_face = _ddz_uv_face_np(v, params)
    if params.momentum_wall_model == "abl":
        fr1 = 1.0 / np.log(3.0) - 1.0
        dudz_face[:, :, 0] += fr1 * np.mean(dudz_face[:, :, 0])
        dvdz_face[:, :, 0] += fr1 * np.mean(dvdz_face[:, :, 0])
    dudz0, dvdz0 = _wall_gradient_np(u, v, params)
    dudz_lower = np.concatenate(
        (dudz0[:, :, None], dudz_face[:, :, :-1]), axis=2
    )
    dvdz_lower = np.concatenate(
        (dvdz0[:, :, None], dvdz_face[:, :, :-1]), axis=2
    )
    dudz_center = 0.5 * (dudz_lower + dudz_face)
    dvdz_center = 0.5 * (dvdz_lower + dvdz_face)
    dudz_center[:, :, 0] = dudz0
    dvdz_center[:, :, 0] = dvdz0
    return dudz_center, dvdz_center, dudz_face, dvdz_face


def _scalar_center_dz_np(q: np.ndarray, params) -> np.ndarray:
    return _ddz_uv_np(q, params)


def _avg_next_np(q: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (0.5 * (q[:, :, :-1] + q[:, :, 1:]), q[:, :, -1:]), axis=2
    )


def _avg_prev_np(q: np.ndarray) -> np.ndarray:
    lower = np.concatenate((np.zeros_like(q[:, :, :1]), q[:, :, :-1]), axis=2)
    return 0.5 * (lower + q)


def _diagnostic_lasd_sgs_tke(
    cs2: np.ndarray,
    scalar_c_theta: np.ndarray,
    strain_mag: np.ndarray,
    dtheta_dz: np.ndarray,
    scalar_stability: np.ndarray,
    params,
) -> np.ndarray:
    """Diagnose LASD SGS energy from local production-dissipation balance.

    The diagnostic uses the shear and buoyancy production implied by the
    *actual* dynamic LASD momentum and heat coefficients, then balances their
    positive sum against a fixed equilibrium dissipation closure,

        P = nu_t |S|^2 - kappa_t N^2,
        epsilon = C_e e^(3/2) / Delta,
        e = max(P Delta / C_e, 0)^(2/3).

    ``C_e = 0.93`` matches the coefficient used when this diagnostic was
    introduced. This adds no prognostic SGS state and remains well behaved
    when dynamic ``C_s^2`` is large near the wall. Solver derivatives use
    z/zi, hence ``N^2`` carries one factor of zi.
    """
    delta2 = float(params.sgs_delta) ** 2
    momentum_length2 = np.maximum(np.asarray(cs2, dtype=np.float64), 0.0) * delta2
    heat_length2 = (
        np.maximum(np.asarray(scalar_c_theta, dtype=np.float64), 0.0)
        * delta2
        * np.asarray(scalar_stability, dtype=np.float64)
    )
    n2_scaled = (
        float(params.z_i)
        * float(params.g)
        / float(params.theta_v0)
        * np.asarray(dtheta_dz, dtype=np.float64)
    )
    strain = np.asarray(strain_mag, dtype=np.float64)
    production_internal = strain * (
        momentum_length2 * strain**2 - heat_length2 * n2_scaled
    )
    dissipation_coefficient = LASD_SGS_DISSIPATION_COEFFICIENT
    mask = np.ones_like(production_internal)
    return (
        np.maximum(production_internal * float(params.sgs_delta) / dissipation_coefficient, 0.0)
        ** (2.0 / 3.0)
        * mask
    )


def _diagnostic_scalar_stability(
    strain_mag: np.ndarray,
    dtheta_dz_center: np.ndarray,
    params,
) -> np.ndarray:
    """Reproduce the solver's cell-centred scalar stability correction."""
    strain = np.asarray(strain_mag, dtype=np.float64)
    if not params.scalar_stability_correction:
        return np.ones_like(strain)
    n2_scaled = (
        float(params.z_i)
        * float(params.g)
        / float(params.theta_v0)
        * np.asarray(dtheta_dz_center, dtype=np.float64)
    )
    ri = np.maximum(n2_scaled, 0.0) / np.maximum(strain * strain, 1.0e-24)
    return (1.0 + params.scalar_stability_beta * ri) ** (-params.scalar_stability_power)


def diagnostic_sgs_profiles(
    state,
    params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not params.thermo_enabled:
        zeros = np.zeros(params.nz, dtype=np.float64)
        return zeros, zeros, zeros, zeros

    u, dudx, dudy = _ddxy_filter_np(np.asarray(state.u), params)
    v, dvdx, dvdy = _ddxy_filter_np(np.asarray(state.v), params)
    w, dwdx, dwdy = _ddxy_filter_np(np.asarray(state.w), params)
    dudz, dvdz, _, _ = _momentum_vertical_gradients_np(u, v, params)
    dwdz = _ddz_w_np(w, params)

    s11 = dudx
    s22 = dvdy
    s33 = dwdz
    s12 = 0.5 * (dudy + dvdx)
    s13 = 0.5 * (dudz + _avg_prev_np(dwdx))
    s23 = 0.5 * (dvdz + _avg_prev_np(dwdy))
    sij_sij = s11 * s11 + s22 * s22 + s33 * s33 + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23)
    strain_mag = np.sqrt(np.maximum(2.0 * sij_sij, 0.0))

    theta = np.asarray(state.theta)
    _, dtheta_dx, dtheta_dy = _ddxy_filter_np(theta, params)
    dtheta_dz = _ddz_uv_np(theta, params)
    dtheta_dz_center = _scalar_center_dz_np(theta, params)
    delta = params.sgs_delta
    mask = np.ones_like(dtheta_dz)
    coeff = np.asarray(state.scalar_c[..., 0], dtype=np.float64)
    stability = _diagnostic_scalar_stability(strain_mag, dtheta_dz_center, params)
    kappa = (
        coeff * delta * delta * strain_mag + params.molecular_diffusivity_internal
    ) * stability * mask
    cs2 = np.asarray(state.cs2, dtype=np.float64)
    nu_t_internal = cs2 * delta * delta * strain_mag
    epsilon_zi = nu_t_internal * strain_mag * strain_mag * mask
    e_sgs = _diagnostic_lasd_sgs_tke(
        cs2,
        coeff,
        strain_mag,
        dtheta_dz_center,
        stability,
        params,
    )
    kappa_upper = _avg_next_np(kappa)
    dtheta_upper = np.empty_like(theta)
    dtheta_upper[:, :, :-1] = (
        theta[:, :, 1:] - theta[:, :, :-1]
    ) / float(params.dz)
    dtheta_upper[:, :, -1] = (
        0.0
        if params.theta_top_gradient is None
        else params.theta_top_gradient * params.z_i
    )
    qz = -kappa_upper * dtheta_upper
    bottom_qz = params.surface_theta_flux if params.theta_bc == "flux" else 0.0
    qz_lower = np.concatenate(
        (np.full_like(qz[:, :, :1], bottom_qz), qz[:, :, :-1]), axis=2
    )
    qz_center = 0.5 * (qz_lower + qz)
    heat_flux_sgs = np.mean(qz_center, axis=(0, 1)).astype(np.float64)

    scalar_length = delta * np.sqrt(
        np.maximum(coeff * stability, 0.0)
    )
    sqrt_e = np.sqrt(np.maximum(e_sgs, 0.0))
    valid_e = sqrt_e > np.finfo(np.float64).tiny
    theta_var_sgs = np.divide(
        -2.0
        * scalar_length
        * qz_center
        * dtheta_dz_center,
        SGS_SCALAR_VARIANCE_COEFFICIENT * np.where(valid_e, sqrt_e, 1.0),
        out=np.zeros_like(e_sgs),
        where=valid_e,
    )
    theta_var_sgs = np.maximum(theta_var_sgs, 0.0)

    epsilon_zi_profile = np.mean(epsilon_zi, axis=(0, 1)).astype(np.float64)
    e_sgs_profile = np.mean(e_sgs, axis=(0, 1)).astype(np.float64)
    theta_var_sgs_profile = np.mean(theta_var_sgs, axis=(0, 1)).astype(
        np.float64
    )
    return (
        heat_flux_sgs,
        epsilon_zi_profile,
        e_sgs_profile,
        theta_var_sgs_profile,
    )


def diagnostic_sgs_heat_flux_profile(state, params) -> np.ndarray:
    return diagnostic_sgs_profiles(state, params)[0]


def snapshot_statistics(
    state,
    params,
    z: np.ndarray,
    spectrum_edges: np.ndarray,
    spectrum_level_fractions: np.ndarray,
    sgs_heat_flux: np.ndarray | None = None,
    epsilon_zi: np.ndarray | None = None,
    sgs_tke_profile: np.ndarray | None = None,
    sgs_theta_var_profile: np.ndarray | None = None,
) -> dict:
    u = np.asarray(state.u)
    v = np.asarray(state.v)
    w = _avg_prev_np(np.asarray(state.w))
    p = np.asarray(state.p)
    theta = np.asarray(state.theta)

    u_mean = np.mean(u, axis=(0, 1))
    v_mean = np.mean(v, axis=(0, 1))
    w_mean = np.mean(w, axis=(0, 1))
    p_mean = np.mean(p, axis=(0, 1))
    theta_mean = np.mean(theta, axis=(0, 1))

    up = u - u_mean[None, None, :]
    vp = v - v_mean[None, None, :]
    wp = w - w_mean[None, None, :]
    pp = p - p_mean[None, None, :]
    thetap = theta - theta_mean[None, None, :]

    theta_face = _avg_next_np(theta)
    theta_face_prime = theta_face - np.mean(theta_face, axis=(0, 1), keepdims=True)
    w_face = np.asarray(state.w)
    w_face_prime = w_face - np.mean(w_face, axis=(0, 1), keepdims=True)
    heat_flux_resolved_face = np.mean(w_face_prime * theta_face_prime, axis=(0, 1))
    heat_flux_resolved = 0.5 * np.concatenate(
        (
            heat_flux_resolved_face[:1],
            heat_flux_resolved_face[:-1] + heat_flux_resolved_face[1:],
        )
    )
    if sgs_heat_flux is None:
        sgs_heat_flux = np.zeros_like(heat_flux_resolved)
    heat_flux_sgs = np.asarray(sgs_heat_flux, dtype=np.float64)
    heat_flux_total = heat_flux_resolved + heat_flux_sgs
    heat_flux = heat_flux_total
    if epsilon_zi is None:
        epsilon_zi = np.zeros_like(heat_flux)
    epsilon_zi = np.asarray(epsilon_zi, dtype=np.float64)
    zi = float(z[int(np.argmin(heat_flux))]) if np.any(np.isfinite(heat_flux)) else float(params.z_i)
    if not np.isfinite(zi) or zi <= 0.0:
        zi = float(params.z_i)
    wstar, theta_star, _ = convective_scales(params, zi)

    e = 0.5 * (up * up + vp * vp + wp * wp)
    if sgs_tke_profile is None:
        e_sgs_profile = np.zeros(e.shape[-1], dtype=np.float64)
    else:
        e_sgs_profile = np.asarray(sgs_tke_profile, dtype=np.float64)
    e_profile = np.mean(e, axis=(0, 1)) + e_sgs_profile
    bl_mask = z <= zi
    energy_bl = float(np.mean(e_profile[bl_mask])) if np.any(bl_mask) else float(np.mean(e_profile))

    updraft = wp > 0.0
    updraft_count = np.sum(updraft, axis=(0, 1))
    grid_count = float(u.shape[0] * u.shape[1])
    alpha_u = updraft_count / grid_count
    w_u = np.divide(
        np.sum(np.where(updraft, w, 0.0), axis=(0, 1)),
        updraft_count,
        out=np.zeros_like(w_mean),
        where=updraft_count > 0,
    )
    theta_u_excess = np.divide(
        np.sum(np.where(updraft, thetap, 0.0), axis=(0, 1)),
        updraft_count,
        out=np.zeros_like(theta_mean),
        where=updraft_count > 0,
    )

    w_transport = np.mean(wp * e, axis=(0, 1))
    p_transport = np.mean(pp * wp, axis=(0, 1))

    spectrum_level_indices = np.asarray(
        [
            int(np.argmin(np.abs(z - fraction * zi)))
            for fraction in spectrum_level_fractions
        ],
        dtype=int,
    )
    spectra_u = []
    spectra_w = []
    spectra_theta = []
    dx = float(params.dx * params.z_i)
    dy = float(params.dy * params.z_i)
    for k in spectrum_level_indices:
        spectra_u.append(0.5 * (radial_spectrum(up[:, :, k], dx, dy, params.z_i, spectrum_edges) + radial_spectrum(vp[:, :, k], dx, dy, params.z_i, spectrum_edges)))
        spectra_w.append(radial_spectrum(wp[:, :, k], dx, dy, params.z_i, spectrum_edges))
        spectra_theta.append(radial_spectrum(thetap[:, :, k], dx, dy, params.z_i, spectrum_edges))

    u_var_resolved = np.mean(up * up, axis=(0, 1))
    v_var_resolved = np.mean(vp * vp, axis=(0, 1))
    w_var_resolved = np.mean(wp * wp, axis=(0, 1))
    theta_var_resolved = np.mean(thetap * thetap, axis=(0, 1))
    theta_var_sgs = (
        np.zeros_like(theta_var_resolved)
        if sgs_theta_var_profile is None
        else np.asarray(sgs_theta_var_profile, dtype=np.float64)
    )
    component_sgs_var = (2.0 / 3.0) * e_sgs_profile
    return {
        "zi": zi,
        "wstar": wstar,
        "theta_star": theta_star,
        "energy_bl": energy_bl,
        "u_mean": u_mean,
        "v_mean": v_mean,
        "w_mean": w_mean,
        "theta_mean": theta_mean,
        "p_mean": p_mean,
        "heat_flux": heat_flux,
        "heat_flux_resolved": heat_flux_resolved,
        "heat_flux_sgs": heat_flux_sgs,
        "heat_flux_total": heat_flux_total,
        "epsilon_zi": epsilon_zi,
        "sgs_tke": e_sgs_profile,
        "u_var_resolved": u_var_resolved,
        "v_var_resolved": v_var_resolved,
        "w_var_resolved": w_var_resolved,
        "component_var_sgs": component_sgs_var,
        "u_var": u_var_resolved + component_sgs_var,
        "v_var": v_var_resolved + component_sgs_var,
        "w_var": w_var_resolved + component_sgs_var,
        "theta_var_resolved": theta_var_resolved,
        "theta_var_sgs": theta_var_sgs,
        "theta_var": theta_var_resolved + theta_var_sgs,
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
        "spectrum_level_z": z[spectrum_level_indices],
    }


def add_profile(accumulator: dict[str, np.ndarray], stats: dict) -> None:
    for key, value in stats.items():
        if isinstance(value, np.ndarray) and key not in {
            "spectra_u",
            "spectra_w",
            "spectra_theta",
            "spectrum_level_z",
        }:
            accumulator[key] += value


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(
    output_dir: Path,
    params,
    z: np.ndarray,
    spectrum_edges: np.ndarray,
    spectrum_level_z: np.ndarray,
    time_rows: list[dict],
    averaged: dict[str, np.ndarray],
    summary: dict[str, float],
    spectra: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    z_over_zi = z / summary["zi_mean"]
    wstar = summary["wstar_mean"]
    theta_star = summary["theta_star_mean"]
    heat_flux = averaged["heat_flux"]
    heat_flux_resolved = averaged.get("heat_flux_resolved", heat_flux)
    heat_flux_sgs = averaged.get("heat_flux_sgs", np.zeros_like(heat_flux))
    epsilon_zi = averaged.get("epsilon_zi", np.zeros_like(heat_flux))
    w_var = averaged["w_var"]
    h_var = 0.5 * (averaged["u_var"] + averaged["v_var"])
    theta_var = averaged["theta_var"]
    theta_var_resolved = averaged["theta_var_resolved"]
    theta_var_sgs = averaged["theta_var_sgs"]
    p_var = averaged["p_var"]
    w3 = averaged["w3"]
    w_var_resolved = averaged["w_var_resolved"]
    skew = np.divide(
        w3,
        w_var_resolved ** 1.5,
        out=np.zeros_like(w3),
        where=w_var_resolved > 0.0,
    )
    transport_norm = wstar**3 / summary["zi_mean"]
    d_w_transport = np.gradient(averaged["w_transport"], z) / transport_norm
    d_p_transport = np.gradient(averaged["p_transport"], z) / transport_norm
    buoyancy_prod = (params.g / params.theta0) * heat_flux / transport_norm

    write_csv(
        output_dir / "time_series.csv",
        time_rows,
        ["step", "time_s", "time_over_tstar0", "zi", "zi_over_zi0", "wstar", "energy_bl_over_wstar0_sq"],
    )
    profile_rows = []
    for k, zk in enumerate(z):
        profile_rows.append(
            {
                "z": zk,
                "z_over_zi": z_over_zi[k],
                "theta_mean": averaged["theta_mean"][k],
                "heat_flux_over_qs": heat_flux[k] / params.surface_theta_flux,
                "heat_flux_resolved_over_qs": heat_flux_resolved[k] / params.surface_theta_flux,
                "heat_flux_sgs_over_qs": heat_flux_sgs[k] / params.surface_theta_flux,
                "heat_flux_total_over_qs": heat_flux[k] / params.surface_theta_flux,
                "epsilon_zi_over_wstar3": epsilon_zi[k] / wstar**3,
                "w_var_over_wstar_sq": w_var[k] / wstar**2,
                "w_var_resolved_over_wstar_sq": averaged["w_var_resolved"][k] / wstar**2,
                "w_var_sgs_over_wstar_sq": averaged["component_var_sgs"][k] / wstar**2,
                "horizontal_var_over_wstar_sq": h_var[k] / wstar**2,
                "horizontal_var_resolved_over_wstar_sq": 0.5
                * (averaged["u_var_resolved"][k] + averaged["v_var_resolved"][k])
                / wstar**2,
                "horizontal_var_sgs_over_wstar_sq": averaged["component_var_sgs"][k] / wstar**2,
                "sgs_tke_over_wstar_sq": averaged["sgs_tke"][k] / wstar**2,
                "theta_var_over_thetastar_sq": theta_var[k] / theta_star**2,
                "theta_var_resolved_over_thetastar_sq": theta_var_resolved[k]
                / theta_star**2,
                "theta_var_sgs_over_thetastar_sq": theta_var_sgs[k]
                / theta_star**2,
                "p_var_over_wstar4": p_var[k] / wstar**4,
                "w3_over_wstar3": w3[k] / wstar**3,
                "skewness": skew[k],
                "alpha_u": averaged["alpha_u"][k],
                "w_u_over_wstar": averaged["w_u"][k] / wstar,
                "theta_u_excess_over_thetastar": averaged["theta_u_excess"][k] / theta_star,
                "buoyancy_production": buoyancy_prod[k],
                "d_w_transport": d_w_transport[k],
                "d_p_transport": d_p_transport[k],
            }
        )
    write_csv(output_dir / "profiles.csv", profile_rows, list(profile_rows[0]))

    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])

    np.savez(
        output_dir / "benchmark_stats.npz",
        z=z,
        z_over_zi=z_over_zi,
        spectrum_kzi=0.5 * (spectrum_edges[:-1] + spectrum_edges[1:]),
        spectrum_level_z=spectrum_level_z,
        spectrum_level_fraction=SPECTRUM_LEVEL_FRACTIONS,
        **{f"profile_{key}": value for key, value in averaged.items()},
        **{f"spectrum_{key}": value for key, value in spectra.items()},
        **summary,
    )
    make_plots(
        output_dir,
        params,
        z,
        z_over_zi,
        time_rows,
        averaged,
        summary,
        spectra,
        spectrum_edges,
        spectrum_level_z,
    )


def make_plots(
    output_dir: Path,
    params,
    z: np.ndarray,
    z_over_zi: np.ndarray,
    time_rows: list[dict],
    averaged: dict[str, np.ndarray],
    summary: dict[str, float],
    spectra: dict[str, np.ndarray],
    spectrum_edges: np.ndarray,
    spectrum_level_z: np.ndarray,
) -> None:
    mpl_cache_dir = output_dir / ".matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wstar = summary["wstar_mean"]
    theta_star = summary["theta_star_mean"]
    transport_norm = wstar**3 / summary["zi_mean"]
    time_x = np.asarray([row["time_over_tstar0"] for row in time_rows])
    energy = np.asarray([row["energy_bl_over_wstar0_sq"] for row in time_rows])

    plt.figure(figsize=(6, 4))
    plt.plot(time_x, energy, marker="o", ms=3)
    plt.xlabel(r"$t/t_{*0}$")
    plt.ylabel(r"$\langle E\rangle / w_{*0}^2$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fig01_energy_time.png", dpi=180)
    plt.close()

    heat_flux = averaged["heat_flux"]
    heat_flux_resolved = averaged.get("heat_flux_resolved", heat_flux)
    heat_flux_sgs = averaged.get("heat_flux_sgs", np.zeros_like(heat_flux))
    epsilon_zi = averaged.get("epsilon_zi", np.zeros_like(heat_flux))

    plt.figure(figsize=(6, 4))
    plt.plot(heat_flux / params.surface_theta_flux, z / params.z_i, label="total")
    plt.plot(heat_flux_resolved / params.surface_theta_flux, z / params.z_i, "--", label="resolved")
    plt.plot(heat_flux_sgs / params.surface_theta_flux, z / params.z_i, ":", label="SGS")
    plt.axvline(0.0, color="0.4", lw=0.8)
    plt.xlabel(r"$\langle w'\theta'\rangle + q_\theta^{sgs}$ / $Q_s$")
    plt.ylabel(r"$z/z_{i0}$")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "fig02_heat_flux.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharey=True)
    axes = axes.ravel()
    axes[0].plot(averaged["w_var"] / wstar**2, z_over_zi, label="total")
    axes[0].plot(averaged["w_var_resolved"] / wstar**2, z_over_zi, "--", label="resolved")
    axes[0].plot(averaged["component_var_sgs"] / wstar**2, z_over_zi, ":", label="SGS")
    axes[0].set_xlabel(r"$\langle w'^2\rangle/w_*^2$")
    axes[1].plot(0.5 * (averaged["u_var"] + averaged["v_var"]) / wstar**2, z_over_zi, label="total")
    axes[1].plot(
        0.5 * (averaged["u_var_resolved"] + averaged["v_var_resolved"]) / wstar**2,
        z_over_zi,
        "--",
        label="resolved",
    )
    axes[1].plot(averaged["component_var_sgs"] / wstar**2, z_over_zi, ":", label="SGS")
    axes[1].set_xlabel(r"$0.5(\langle u'^2\rangle+\langle v'^2\rangle)/w_*^2$")
    axes[2].plot(
        averaged["theta_var"] / theta_star**2,
        z_over_zi,
        label="total",
    )
    axes[2].plot(
        averaged["theta_var_resolved"] / theta_star**2,
        z_over_zi,
        "--",
        label="resolved",
    )
    axes[2].plot(
        averaged["theta_var_sgs"] / theta_star**2,
        z_over_zi,
        ":",
        label="SGS",
    )
    axes[2].set_xlabel(r"$\langle \theta'^2\rangle/\theta_*^2$")
    axes[3].plot(averaged["p_var"] / wstar**4, z_over_zi)
    axes[3].set_xlabel(r"$\langle p'^2\rangle/w_*^4$")
    for ax in axes:
        ax.set_ylabel(r"$z/z_i$")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig03_06_variances.png", dpi=180)
    plt.close(fig)

    w_var = averaged["w_var_resolved"]
    w3 = averaged["w3"]
    skew = np.divide(w3, w_var ** 1.5, out=np.zeros_like(w3), where=w_var > 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    axes[0].plot(w3 / wstar**3, z_over_zi)
    axes[0].set_xlabel(r"$\langle w'^3\rangle/w_*^3$")
    axes[1].plot(skew, z_over_zi)
    axes[1].set_xlabel(r"$Sk_w$")
    for ax in axes:
        ax.set_ylabel(r"$z/z_i$")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig07_08_higher_moments.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(6, 4))
    plt.plot(epsilon_zi / wstar**3, z_over_zi)
    plt.xlabel(r"$\langle\epsilon\rangle z_i / w_*^3$")
    plt.ylabel(r"$z/z_i$")
    plt.xlim(left=0.0)
    plt.ylim(0.0, 1.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fig09_dissipation.png", dpi=180)
    plt.savefig(output_dir / "fig09_11_energy_budget_terms.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot((params.g / params.theta0) * heat_flux / transport_norm, z_over_zi, label="buoyancy production")
    plt.plot(np.gradient(averaged["w_transport"], z) / transport_norm, z_over_zi, label=r"$d\langle w'E'\rangle/dz$")
    plt.plot(np.gradient(averaged["p_transport"], z) / transport_norm, z_over_zi, label=r"$d\langle p'w'\rangle/dz$")
    plt.xlabel(r"normalized budget term")
    plt.ylabel(r"$z/z_i$")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "fig10_11_energy_budget_terms.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    axes[0].plot(averaged["alpha_u"], z_over_zi)
    axes[0].set_xlabel(r"$\alpha_u$")
    axes[1].plot(averaged["w_u"] / wstar, z_over_zi)
    axes[1].set_xlabel(r"$w_u/w_*$")
    axes[2].plot(averaged["theta_u_excess"] / theta_star, z_over_zi)
    axes[2].set_xlabel(r"$(\theta_u-\langle\theta\rangle)/\theta_*$")
    for ax in axes:
        ax.set_ylabel(r"$z/z_i$")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig12_14_conditional_updrafts.png", dpi=180)
    plt.close(fig)

    kzi = 0.5 * (spectrum_edges[:-1] + spectrum_edges[1:])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for i, level_fraction in enumerate(SPECTRUM_LEVEL_FRACTIONS):
        label = f"z={level_fraction:.1f} zi"
        mask = (kzi > 0.0) & (spectra["u"][i] > 0.0)
        axes[0].loglog(kzi[mask], spectra["u"][i][mask], label=label)
        mask = (kzi > 0.0) & (spectra["w"][i] > 0.0)
        axes[1].loglog(kzi[mask], spectra["w"][i][mask], label=label)
        mask = (kzi > 0.0) & (spectra["theta"][i] > 0.0)
        axes[2].loglog(kzi[mask], spectra["theta"][i][mask], label=label)
    axes[0].set_title("horizontal velocity")
    axes[1].set_title("vertical velocity")
    axes[2].set_title("temperature")
    for ax in axes:
        ax.set_xlabel(r"$k z_{i0}$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("radial spectrum")
    fig.tight_layout()
    fig.savefig(output_dir / "fig15_17_spectra.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    try:
        settings = load_settings(args.config)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"ERROR: failed to load config: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    if args.max_steps is not None:
        settings["steps"] = args.max_steps
    if args.steps is not None:
        settings["steps"] = args.steps
    if args.dt is not None:
        settings["dt"] = args.dt
    if args.smag_cs is not None:
        settings["smag_cs"] = args.smag_cs
    if args.prandtl_t is not None:
        settings["prandtl_t"] = args.prandtl_t
    if args.scalar_vertical_scheme is not None:
        settings["scalar_vertical_scheme"] = args.scalar_vertical_scheme
    if args.coriolis_f is not None:
        settings["coriolis_f"] = args.coriolis_f
    if args.geostrophic_u is not None:
        settings["geostrophic_u"] = args.geostrophic_u
    if args.geostrophic_v is not None:
        settings["geostrophic_v"] = args.geostrophic_v
    if args.wall_stress_model is not None:
        settings["wall_stress_model"] = args.wall_stress_model
    if args.use_jit is not None:
        settings["use_jit"] = args.use_jit
    if args.sample_every <= 0:
        raise SystemExit("ERROR: --sample-every must be positive")

    if settings["precision"] == "float64" or settings["sgs_precision"] == "float64":
        from jax import config as jax_config

        jax_config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp

    from wireles_jax.diagnostics import diagnostics, lasd_cfl_number, validate_cfl, validate_lasd_cfl
    from wireles_jax.grid import make_operators
    from wireles_jax.timestep import step
    from initial_condition import initial_benchmark_state

    params = build_params(settings, jnp)
    if not params.thermo_enabled or params.surface_theta_flux <= 0.0:
        raise SystemExit("ERROR: this benchmark requires thermo enabled and positive surface_theta_flux")
    lasd_active = params.sgs_model == "lasd" or (
        params.thermo_enabled and params.scalar_sgs_model == "lasd"
    )

    wstar0, theta_star0, tstar0 = convective_scales(params, params.z_i)
    start_step = int(np.ceil(args.average_start_tstar * tstar0 / params.dt_physical))
    end_step = int(np.floor(args.average_end_tstar * tstar0 / params.dt_physical))
    end_step = min(end_step, params.nsteps)
    if start_step >= end_step:
        print(
            f"WARNING: average window [{start_step}, {end_step}] is empty for this run; "
            "using the final sampled interval instead.",
            flush=True,
        )

    ops = make_operators(params)
    state = initial_benchmark_state(
        params,
        seed=args.seed,
        initial_zi_fraction=settings["benchmark_initial_zi_fraction"],
    )
    initial_diag = jax.block_until_ready(diagnostics(state, params, ops))
    max_cfl = validate_cfl(initial_diag)
    max_lasd_cfl = validate_lasd_cfl(initial_diag, params) if lasd_active else 0.0
    if params.use_jit:
        step_jit = jax.jit(lambda s: step(s, params, ops))
        compile_start = time.perf_counter()
        print(f"[precompile] lowering Nieuwstadt1993 step kernel for {params.nx}x{params.ny}x{params.nz}", flush=True)
        lowered = step_jit.lower(state)
        print(f"[precompile] compiling step kernel (lowered in {time.perf_counter() - compile_start:.1f}s)", flush=True)
        step_fn = lowered.compile()
        print(f"[precompile] done in {time.perf_counter() - compile_start:.1f}s", flush=True)
    else:
        step_fn = lambda s: step(s, params, ops)

    z = physical_z(params)
    spectrum_edges = np.linspace(0.0, np.pi * params.z_i / min(params.dx * params.z_i, params.dy * params.z_i), 31)
    spectrum_level_fractions = SPECTRUM_LEVEL_FRACTIONS

    profile_sums: defaultdict[str, np.ndarray] = defaultdict(lambda: np.zeros(params.nz, dtype=np.float64))
    profile_sample_steps: list[int] = []
    profile_samples: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    spectra_sums = {
        "u": np.zeros((spectrum_level_fractions.size, spectrum_edges.size - 1), dtype=np.float64),
        "w": np.zeros((spectrum_level_fractions.size, spectrum_edges.size - 1), dtype=np.float64),
        "theta": np.zeros((spectrum_level_fractions.size, spectrum_edges.size - 1), dtype=np.float64),
    }
    spectrum_level_z_sum = np.zeros_like(spectrum_level_fractions)
    profile_count = 0
    time_rows: list[dict] = []
    zi_samples: list[float] = []

    def sample(step_number: int, force_average: bool = False) -> None:
        nonlocal profile_count, spectrum_level_z_sum
        ready_state = jax.block_until_ready(state)
        (
            sgs_heat_flux,
            epsilon_zi,
            sgs_tke_profile,
            sgs_theta_var_profile,
        ) = diagnostic_sgs_profiles(ready_state, params)
        stats = snapshot_statistics(
            ready_state,
            params,
            z,
            spectrum_edges,
            spectrum_level_fractions,
            sgs_heat_flux=sgs_heat_flux,
            epsilon_zi=epsilon_zi,
            sgs_tke_profile=sgs_tke_profile,
            sgs_theta_var_profile=sgs_theta_var_profile,
        )
        time_s = step_number * params.dt_physical
        zi = stats["zi"]
        time_rows.append(
            {
                "step": step_number,
                "time_s": time_s,
                "time_over_tstar0": time_s / tstar0,
                "zi": zi,
                "zi_over_zi0": zi / params.z_i,
                "wstar": stats["wstar"],
                "energy_bl_over_wstar0_sq": stats["energy_bl"] / wstar0**2,
            }
        )
        in_average_window = start_step <= step_number <= end_step if start_step < end_step else force_average
        if in_average_window:
            add_profile(profile_sums, stats)
            profile_sample_steps.append(step_number)
            for key, value in stats.items():
                if isinstance(value, np.ndarray) and value.shape == (params.nz,):
                    profile_samples[key].append(np.asarray(value, dtype=np.float64).copy())
            spectra_sums["u"] += stats["spectra_u"]
            spectra_sums["w"] += stats["spectra_w"]
            spectra_sums["theta"] += stats["spectra_theta"]
            spectrum_level_z_sum += stats["spectrum_level_z"]
            zi_samples.append(zi)
            profile_count += 1

    print(
        "[case] "
        f"Qs={params.surface_theta_flux:.4g} K m/s, zi0={params.z_i:.1f} m, "
        f"wstar0={wstar0:.4f} m/s, theta_star0={theta_star0:.4f} K, tstar0={tstar0:.1f} s",
        flush=True,
    )
    print(
        f"[case] steps={params.nsteps}, dt={params.dt_physical:.3f}s, "
        f"average_steps={start_step}:{end_step}, sample_every={args.sample_every}",
        flush=True,
    )

    run_start = time.perf_counter()
    sample(0)
    for n in range(params.nsteps):
        lasd_update = ((n + 1) % params.cs_count) == 0
        if lasd_update and lasd_active:
            update_diag = jax.block_until_ready(diagnostics(state, params, ops))
            max_cfl = max(max_cfl, validate_cfl(update_diag))
            max_lasd_cfl = max(max_lasd_cfl, validate_lasd_cfl(update_diag, params))
        state = step_fn(state)
        step_number = n + 1
        if step_number % args.sample_every == 0 or step_number == params.nsteps:
            diag = jax.block_until_ready(diagnostics(state, params, ops))
            current_cfl = validate_cfl(diag)
            max_cfl = max(max_cfl, current_cfl)
            lasd_cfl = lasd_cfl_number(diag, params) if lasd_active else 0.0
            if lasd_active:
                max_lasd_cfl = max(max_lasd_cfl, lasd_cfl)
            elapsed = time.perf_counter() - run_start
            total = elapsed * params.nsteps / step_number
            remaining = max(0.0, total - elapsed)
            print(
                f"{step_number:5d} rest_s={remaining:8.1f} total_s={total:8.1f} "
                f"t/t*={step_number * params.dt_physical / tstar0:6.2f} "
                f"div={float(diag.div_max):.3e} cfl={current_cfl:.4f}"
                + (f" lasd_cfl={lasd_cfl:.4f}" if lasd_active else ""),
                flush=True,
            )
            sample(step_number, force_average=step_number == params.nsteps)

    if profile_count == 0:
        raise SystemExit("ERROR: no samples collected in the averaging window")

    averaged = {key: value / profile_count for key, value in profile_sums.items()}
    spectra = {key: value / profile_count for key, value in spectra_sums.items()}
    spectrum_level_z = spectrum_level_z_sum / profile_count
    heat_flux = averaged["heat_flux"]
    zi_flux = float(z[int(np.argmin(heat_flux))])
    theta_gradient = np.gradient(averaged["theta_mean"], z)
    zi_theta_gradient = float(z[int(np.argmax(theta_gradient))])
    zi_sample_mean = float(np.mean(zi_samples))
    zi_mean = zi_sample_mean
    wstar_mean, theta_star_mean, tstar_mean = convective_scales(params, zi_mean)
    entrainment_ratio = -float(np.min(heat_flux)) / params.surface_theta_flux
    z_over_zi_mean = z / zi_mean
    summary = {
        "sample_count": float(profile_count),
        "average_start_step": float(start_step),
        "average_end_step": float(end_step),
        "zi0": float(params.z_i),
        "wstar0": float(wstar0),
        "theta_star0": float(theta_star0),
        "tstar0": float(tstar0),
        "zi_mean": zi_mean,
        "zi_flux_min": zi_flux,
        "zi_theta_gradient_max": zi_theta_gradient,
        "zi_instant_sample_mean": zi_sample_mean,
        "zi_over_zi0": zi_mean / params.z_i,
        "wstar_mean": wstar_mean,
        "wstar_over_wstar0": wstar_mean / wstar0,
        "theta_star_mean": theta_star_mean,
        "tstar_mean": tstar_mean,
        "entrainment_ratio": entrainment_ratio,
        "theta_mixed_layer_mean": float(np.mean(averaged["theta_mean"][z < 0.8 * zi_mean])),
        "lasd_sgs_energy_dissipation_coefficient": float(
            LASD_SGS_DISSIPATION_COEFFICIENT
        ),
        "heat_flux_alternating_over_qs": abs(
            alternating_mode_amplitude(heat_flux / params.surface_theta_flux, z_over_zi_mean)
        ),
        "theta_mean_alternating_K": abs(
            alternating_mode_amplitude(averaged["theta_mean"], z_over_zi_mean)
        ),
        "max_cfl": max_cfl,
        "max_lasd_cfl": max_lasd_cfl,
        "runtime_s": float(time.perf_counter() - run_start),
    }
    save_outputs(
        args.output_dir,
        params,
        z,
        spectrum_edges,
        spectrum_level_z,
        time_rows,
        averaged,
        summary,
        spectra,
    )
    np.savez(
        args.output_dir / "profile_samples.npz",
        step=np.asarray(profile_sample_steps, dtype=np.int64),
        **{key: np.stack(values, axis=0) for key, values in profile_samples.items()},
    )
    print("[summary]", flush=True)
    for key in (
        "zi_over_zi0",
        "wstar_over_wstar0",
        "entrainment_ratio",
        "theta_mixed_layer_mean",
        "heat_flux_alternating_over_qs",
        "theta_mean_alternating_K",
        "max_cfl",
        "max_lasd_cfl",
        "sample_count",
    ):
        print(f"  {key}: {summary[key]:.6g}", flush=True)
    print(f"[output] wrote diagnostics to {args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
