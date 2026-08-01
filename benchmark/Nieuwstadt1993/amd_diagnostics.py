"""Paper-facing diagnostics for the non-spectral Nieuwstadt AMD solver."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True, slots=True)
class NieuwstadtCase:
    theta0: float = 300.0
    gravity: float = 9.81
    surface_theta_flux: float = 0.06
    zi0: float = 1600.0
    zi_search_max_fraction: float = 1.15
    spectrum_level_fractions: tuple[float, ...] = (0.2, 0.6, 1.0)

    @property
    def wstar0(self) -> float:
        return (
            self.gravity
            * self.surface_theta_flux
            * self.zi0
            / self.theta0
        ) ** (1.0 / 3.0)

    @property
    def theta_star0(self) -> float:
        return self.surface_theta_flux / self.wstar0

    @property
    def tstar0(self) -> float:
        return self.zi0 / self.wstar0

    def convective_scales(self, zi: float) -> tuple[float, float]:
        wstar = (
            self.gravity * self.surface_theta_flux * zi / self.theta0
        ) ** (1.0 / 3.0)
        return wstar, self.surface_theta_flux / wstar


def _plane(value) -> np.ndarray:
    return np.asarray(jax.device_get(jnp.mean(value, axis=(1, 2))), dtype=np.float64)


def _cell_velocity(velocity) -> jax.Array:
    return jnp.stack(
        (
            0.5 * (velocity.x[..., 1:] + velocity.x[..., :-1]),
            0.5 * (velocity.y[:, 1:, :] + velocity.y[:, :-1, :]),
            0.5 * (velocity.z[1:] + velocity.z[:-1]),
        ),
        axis=-1,
    )


def _radial_spectrum(
    field: np.ndarray,
    dx: float,
    dy: float,
    edges: np.ndarray,
    zi0: float,
) -> np.ndarray:
    ny, nx = field.shape
    transformed = np.fft.fft2(field) / (nx * ny)
    energy = np.abs(transformed) ** 2
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    radius = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2) * zi0
    bins = np.digitize(radius.ravel(), edges) - 1
    valid = (bins >= 0) & (bins < edges.size - 1)
    result = np.bincount(
        bins[valid],
        weights=energy.ravel()[valid],
        minlength=edges.size - 1,
    )
    return result[: edges.size - 1]


def snapshot_statistics(
    state,
    coupled,
    diagnostic_kernel,
    *,
    case: NieuwstadtCase,
    spectrum_edges: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Return one complete host-side paper-statistics snapshot."""
    cells_jax = _cell_velocity(state.velocity)
    theta_jax = state.potential_temperature
    fields = diagnostic_kernel(state)
    jax.block_until_ready(fields.sgs_tke)

    cells = np.asarray(jax.device_get(cells_jax), dtype=np.float64)
    theta_perturbation = np.asarray(jax.device_get(theta_jax), dtype=np.float64)
    theta = case.theta0 + theta_perturbation
    pressure = np.asarray(jax.device_get(state.pressure), dtype=np.float64)
    z = (
        np.arange(coupled.grid.shape[0], dtype=np.float64) + 0.5
    ) * coupled.momentum.dz

    mean = np.mean(cells, axis=(1, 2))
    fluctuation = cells - mean[:, None, None, :]
    theta_mean = np.mean(theta, axis=(1, 2))
    theta_fluctuation = theta - theta_mean[:, None, None]
    pressure_mean = np.mean(pressure, axis=(1, 2))
    pressure_fluctuation = pressure - pressure_mean[:, None, None]

    theta_face = np.zeros(
        (theta.shape[0] + 1, theta.shape[1], theta.shape[2]),
        dtype=np.float64,
    )
    theta_face[1:-1] = 0.5 * (theta[:-1] + theta[1:])
    theta_face[0] = theta[0]
    theta_face[-1] = theta[-1]
    w_face = np.asarray(jax.device_get(state.velocity.z), dtype=np.float64)
    w_face_fluctuation = w_face - np.mean(w_face, axis=(1, 2), keepdims=True)
    theta_face_fluctuation = theta_face - np.mean(
        theta_face,
        axis=(1, 2),
        keepdims=True,
    )
    resolved_heat_flux_face = np.mean(
        w_face_fluctuation * theta_face_fluctuation,
        axis=(1, 2),
    )
    resolved_heat_flux = 0.5 * (
        resolved_heat_flux_face[:-1] + resolved_heat_flux_face[1:]
    )
    scalar_flux_z = np.asarray(
        jax.device_get(fields.scalar_flux_z),
        dtype=np.float64,
    )
    sgs_heat_flux = np.mean(
        0.5 * (scalar_flux_z[:-1] + scalar_flux_z[1:]),
        axis=(1, 2),
    )
    total_heat_flux = resolved_heat_flux + sgs_heat_flux
    inversion_search = z <= case.zi_search_max_fraction * case.zi0
    local_index = int(np.argmin(total_heat_flux[inversion_search]))
    zi_index = int(np.flatnonzero(inversion_search)[local_index])
    zi = float(z[zi_index])
    wstar, theta_star = case.convective_scales(zi)

    sgs_tke = _plane(fields.sgs_tke)
    resolved_tke = 0.5 * np.mean(
        np.sum(fluctuation * fluctuation, axis=-1),
        axis=(1, 2),
    )
    energy_bl = float(np.mean((resolved_tke + sgs_tke)[z <= zi]))
    w_fluctuation = fluctuation[..., 2]
    updraft = w_fluctuation > 0.0
    count = np.sum(updraft, axis=(1, 2))
    alpha_u = count / float(coupled.grid.shape[1] * coupled.grid.shape[2])
    w_u = np.divide(
        np.sum(np.where(updraft, cells[..., 2], 0.0), axis=(1, 2)),
        count,
        out=np.zeros_like(count, dtype=np.float64),
        where=count > 0,
    )
    theta_u_excess = np.divide(
        np.sum(np.where(updraft, theta_fluctuation, 0.0), axis=(1, 2)),
        count,
        out=np.zeros_like(count, dtype=np.float64),
        where=count > 0,
    )
    resolved_energy = 0.5 * np.sum(fluctuation * fluctuation, axis=-1)
    w_transport = np.mean(w_fluctuation * resolved_energy, axis=(1, 2))
    p_transport = np.mean(pressure_fluctuation * w_fluctuation, axis=(1, 2))

    level_fractions = np.asarray(case.spectrum_level_fractions)
    level_indices = np.asarray(
        [int(np.argmin(np.abs(z - fraction * zi))) for fraction in level_fractions]
    )
    spectra_u = []
    spectra_w = []
    spectra_theta = []
    for level in level_indices:
        spectra_u.append(
            0.5
            * (
                _radial_spectrum(
                    fluctuation[level, ..., 0],
                    coupled.momentum.dx,
                    coupled.momentum.dy,
                    spectrum_edges,
                    case.zi0,
                )
                + _radial_spectrum(
                    fluctuation[level, ..., 1],
                    coupled.momentum.dx,
                    coupled.momentum.dy,
                    spectrum_edges,
                    case.zi0,
                )
            )
        )
        spectra_w.append(
            _radial_spectrum(
                fluctuation[level, ..., 2],
                coupled.momentum.dx,
                coupled.momentum.dy,
                spectrum_edges,
                case.zi0,
            )
        )
        spectra_theta.append(
            _radial_spectrum(
                theta_fluctuation[level],
                coupled.momentum.dx,
                coupled.momentum.dy,
                spectrum_edges,
                case.zi0,
            )
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
        "sgs_tke": sgs_tke,
        "scalar_variance_numerator": _plane(fields.scalar_variance_numerator),
        "u_var_resolved": np.mean(fluctuation[..., 0] ** 2, axis=(1, 2)),
        "v_var_resolved": np.mean(fluctuation[..., 1] ** 2, axis=(1, 2)),
        "w_var_resolved": np.mean(fluctuation[..., 2] ** 2, axis=(1, 2)),
        "theta_var_resolved": np.mean(theta_fluctuation**2, axis=(1, 2)),
        "p_var": np.mean(pressure_fluctuation**2, axis=(1, 2)),
        "w3": np.mean(w_fluctuation**3, axis=(1, 2)),
        "w_transport": w_transport,
        "p_transport": p_transport,
        "alpha_u": alpha_u,
        "w_u": w_u,
        "theta_u_excess": theta_u_excess,
        "amd_dissipation": _plane(fields.amd_energy_dissipation),
        "mp5_dissipation": _plane(fields.mp5_energy_dissipation),
        "amd_scalar_dissipation": _plane(fields.amd_scalar_dissipation),
        "mp5_scalar_dissipation": _plane(fields.mp5_scalar_dissipation),
        "momentum_diffusivity": _plane(fields.momentum_diffusivity),
        "scalar_diffusivity": _plane(fields.scalar_diffusivity),
        "spectra_u": np.asarray(spectra_u),
        "spectra_w": np.asarray(spectra_w),
        "spectra_theta": np.asarray(spectra_theta),
        "spectrum_level_z": z[level_indices],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(
    output: Path,
    *,
    args,
    case: NieuwstadtCase,
    coupled,
    time_rows: list[dict],
    selected: list[dict],
    runtime_s: float,
    max_cfl: float,
    max_diffusive_cfl: float,
    max_divergence: float,
    max_scalar_budget_error: float,
    spectrum_edges: np.ndarray,
) -> dict[str, float | str]:
    """Average samples and write the paper-compatible output contract."""
    if not selected:
        raise ValueError("at least one statistics sample is required")
    scalar_keys = {"zi", "wstar", "theta_star", "energy_bl"}
    averaged = {
        key: np.mean([np.asarray(sample[key]) for sample in selected], axis=0)
        for key in selected[0]
        if key not in scalar_keys
    }
    zi_mean = float(np.mean([sample["zi"] for sample in selected]))
    wstar_mean = float(np.mean([sample["wstar"] for sample in selected]))
    theta_star_mean = float(
        np.mean([sample["theta_star"] for sample in selected])
    )
    nz = coupled.grid.shape[0]
    z = (np.arange(nz) + 0.5) * coupled.momentum.dz
    component_sgs = (2.0 / 3.0) * averaged["sgs_tke"]
    theta_var_sgs = np.maximum(
        averaged["scalar_variance_numerator"]
        / np.sqrt(
            np.maximum(
                averaged["sgs_tke"],
                np.finfo(np.float64).tiny,
            )
        ),
        0.0,
    )
    w_var_resolved = averaged["w_var_resolved"]
    w_var = w_var_resolved + component_sgs
    u_var = averaged["u_var_resolved"] + component_sgs
    v_var = averaged["v_var_resolved"] + component_sgs
    theta_var = averaged["theta_var_resolved"] + theta_var_sgs
    skewness = np.divide(
        averaged["w3"],
        np.maximum(w_var_resolved, 0.0) ** 1.5,
        out=np.zeros_like(w_var_resolved),
        where=w_var_resolved > 0.0,
    )
    transport_norm = wstar_mean**3 / zi_mean
    d_w_transport = np.gradient(averaged["w_transport"], z) / transport_norm
    d_p_transport = np.gradient(averaged["p_transport"], z) / transport_norm
    total_dissipation = averaged["amd_dissipation"] + averaged["mp5_dissipation"]

    profiles = []
    for index, height in enumerate(z):
        profiles.append(
            {
                "z": height,
                "z_over_zi": height / zi_mean,
                "theta_mean": averaged["theta_mean"][index],
                "heat_flux_over_qs": averaged["heat_flux"][index]
                / case.surface_theta_flux,
                "heat_flux_resolved_over_qs": averaged["heat_flux_resolved"][index]
                / case.surface_theta_flux,
                "heat_flux_sgs_over_qs": averaged["heat_flux_sgs"][index]
                / case.surface_theta_flux,
                "heat_flux_total_over_qs": averaged["heat_flux"][index]
                / case.surface_theta_flux,
                "epsilon_zi_over_wstar3": total_dissipation[index]
                * zi_mean
                / wstar_mean**3,
                "amd_epsilon_zi_over_wstar3": averaged["amd_dissipation"][index]
                * zi_mean
                / wstar_mean**3,
                "mp5_epsilon_zi_over_wstar3": averaged["mp5_dissipation"][index]
                * zi_mean
                / wstar_mean**3,
                "w_var_over_wstar_sq": w_var[index] / wstar_mean**2,
                "w_var_resolved_over_wstar_sq": w_var_resolved[index]
                / wstar_mean**2,
                "w_var_sgs_over_wstar_sq": component_sgs[index] / wstar_mean**2,
                "horizontal_var_over_wstar_sq": 0.5
                * (u_var[index] + v_var[index])
                / wstar_mean**2,
                "horizontal_var_resolved_over_wstar_sq": 0.5
                * (
                    averaged["u_var_resolved"][index]
                    + averaged["v_var_resolved"][index]
                )
                / wstar_mean**2,
                "horizontal_var_sgs_over_wstar_sq": component_sgs[index]
                / wstar_mean**2,
                "sgs_tke_over_wstar_sq": averaged["sgs_tke"][index]
                / wstar_mean**2,
                "theta_var_over_thetastar_sq": theta_var[index]
                / theta_star_mean**2,
                "theta_var_resolved_over_thetastar_sq": averaged[
                    "theta_var_resolved"
                ][index]
                / theta_star_mean**2,
                "theta_var_sgs_over_thetastar_sq": theta_var_sgs[index]
                / theta_star_mean**2,
                "p_var_over_wstar4": averaged["p_var"][index] / wstar_mean**4,
                "w3_over_wstar3": averaged["w3"][index] / wstar_mean**3,
                "skewness": skewness[index],
                "alpha_u": averaged["alpha_u"][index],
                "w_u_over_wstar": averaged["w_u"][index] / wstar_mean,
                "theta_u_excess_over_thetastar": averaged["theta_u_excess"][
                    index
                ]
                / theta_star_mean,
                "buoyancy_production": case.gravity
                / case.theta0
                * averaged["heat_flux"][index]
                / transport_norm,
                "d_w_transport": d_w_transport[index],
                "d_p_transport": d_p_transport[index],
                "momentum_diffusivity_m2_s": averaged["momentum_diffusivity"][
                    index
                ],
                "scalar_diffusivity_m2_s": averaged["scalar_diffusivity"][index],
                "amd_scalar_dissipation": averaged["amd_scalar_dissipation"][
                    index
                ],
                "mp5_scalar_dissipation": averaged["mp5_scalar_dissipation"][
                    index
                ],
            }
        )
    _write_csv(output / "profiles.csv", profiles)
    _write_csv(output / "time_series.csv", time_rows)

    inversion_search = z <= case.zi_search_max_fraction * case.zi0
    entrainment_flux = float(np.min(averaged["heat_flux"][inversion_search]))
    zi_flux_min = float(
        z[np.flatnonzero(inversion_search)[np.argmin(averaged["heat_flux"][inversion_search])]]
    )
    mixed = z <= 0.8 * zi_mean
    summary: dict[str, float | str] = {
        "schema": "jaxwind.nieuwstadt1993.nonspectral-amd.v1",
        "solver": "non-spectral MAC + matrix-free GMG/PCG",
        "sgs_model": "AMD",
        "sample_count": float(len(selected)),
        "zi0": case.zi0,
        "wstar0": case.wstar0,
        "theta_star0": case.theta_star0,
        "tstar0": case.tstar0,
        "zi_mean": zi_mean,
        "zi_flux_min": zi_flux_min,
        "zi_over_zi0": zi_mean / case.zi0,
        "wstar_mean": wstar_mean,
        "wstar_over_wstar0": wstar_mean / case.wstar0,
        "theta_star_mean": theta_star_mean,
        "entrainment_ratio": -entrainment_flux / case.surface_theta_flux,
        "theta_mixed_layer_mean": float(np.mean(averaged["theta_mean"][mixed])),
        "max_cfl": max_cfl,
        "max_diffusive_cfl": max_diffusive_cfl,
        "max_divergence": max_divergence,
        "max_scalar_budget_error": max_scalar_budget_error,
        "runtime_s": runtime_s,
        "amd_coefficient": float(args.amd_coefficient),
        "scalar_amd_coefficient": float(args.scalar_amd_coefficient),
        "mp5_dissipation_strength": float(args.mp5_strength),
    }
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("quantity", "value"))
        writer.writerows(summary.items())

    np.savez(
        output / "benchmark_stats.npz",
        z=z,
        z_over_zi=z / zi_mean,
        spectrum_kzi=0.5 * (spectrum_edges[:-1] + spectrum_edges[1:]),
        spectrum_level_z=averaged["spectrum_level_z"],
        spectrum_level_fraction=np.asarray(case.spectrum_level_fractions),
        spectrum_u=averaged["spectra_u"],
        spectrum_w=averaged["spectra_w"],
        spectrum_theta=averaged["spectra_theta"],
        **{f"profile_{key}": value for key, value in averaged.items()},
        **summary,
    )
    (output / "resolved_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "output_dir": str(args.output_dir),
                "implementation": "non-spectral MAC AMD Boussinesq",
                "pressure": "matrix-free GMG-preconditioned Krylov",
                "coupling": "Strang scalar-half / projected momentum / scalar-half",
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    make_plots_from_files(output)
    return summary


def make_plots_from_files(output: Path) -> None:
    """Render the paper-aligned standalone diagnostic set."""
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
    plt.plot(profiles["heat_flux_sgs_over_qs"], z_zi, ":", label="AMD SGS")
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
        axis.plot(profiles[sgs], z_zi, ":", label="AMD SGS")
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
    plt.plot(profiles["epsilon_zi_over_wstar3"], z_zi, label="total")
    plt.plot(profiles["amd_epsilon_zi_over_wstar3"], z_zi, "--", label="AMD")
    plt.plot(profiles["mp5_epsilon_zi_over_wstar3"], z_zi, ":", label="MP5")
    plt.xlabel(r"$\langle\epsilon\rangle z_i/w_*^3$")
    plt.ylabel(r"$z/z_i$")
    plt.legend(fontsize=8)
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
    for key, axis, title in zip(
        ("spectrum_u", "spectrum_w", "spectrum_theta"),
        axes,
        ("horizontal velocity", "vertical velocity", "temperature"),
        strict=True,
    ):
        values = np.asarray(stats[key])
        for index, fraction in enumerate(stats["spectrum_level_fraction"]):
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


__all__ = [
    "NieuwstadtCase",
    "make_plots_from_files",
    "save_outputs",
    "snapshot_statistics",
]
