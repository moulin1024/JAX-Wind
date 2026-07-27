#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Nieuwstadt et al. (1993) CBL diagnostics from C++ profile CSV output.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/largeeddy1993_lasd_diagnostics"))
    parser.add_argument("--output-dir", type=Path, help="Directory for PNG outputs. Defaults to --input-dir.")
    parser.add_argument("--title", default="C++ LASD")
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, float]:
    summary: dict[str, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            summary[row["quantity"]] = float(row["value"])
    return summary


def load_profiles(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise SystemExit(f"No profile rows found in {path}")
    return np.atleast_1d(data)


def finish_profile_plot(path: Path, title: str) -> None:
    plt.title(title)
    plt.ylabel(r"$z/z_i$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles(input_dir / "profiles.csv")
    summary = load_summary(input_dir / "summary.csv")
    face_heat_flux_path = input_dir / "heat_flux_faces.csv"
    face_heat_flux = load_profiles(face_heat_flux_path) if face_heat_flux_path.exists() else None
    z_over_zi = profiles["z_over_zi"]
    z_over_zi0 = profiles["z_over_zi0"]
    title_suffix = (
        f"{args.title}: zi/zi0={summary['zi_over_zi0']:.3f}, "
        f"w*/w*0={summary['wstar_over_wstar0']:.3f}, "
        f"entr={summary['entrainment_ratio']:.3f}"
    )

    plt.figure(figsize=(6, 4))
    if face_heat_flux is not None:
        heat_flux_z = face_heat_flux["z_over_zi0"]
        plt.plot(face_heat_flux["heat_flux_over_qs"], heat_flux_z, label="total")
        plt.plot(face_heat_flux["heat_flux_resolved_over_qs"], heat_flux_z, "--", label="resolved")
        plt.plot(face_heat_flux["heat_flux_sgs_over_qs"], heat_flux_z, ":", label="SGS")
    else:
        plt.plot(profiles["heat_flux_total_over_qs"], z_over_zi0, label="total")
        plt.plot(profiles["heat_flux_resolved_over_qs"], z_over_zi0, "--", label="resolved")
        plt.plot(profiles["heat_flux_sgs_over_qs"], z_over_zi0, ":", label="SGS")
    plt.axvline(0.0, color="0.4", lw=0.8)
    plt.xlabel(r"$\langle w'\theta'\rangle + q_\theta^{sgs}$ / $Q_s$")
    plt.ylabel(r"$z/z_{i0}$")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.title(title_suffix)
    plt.tight_layout()
    plt.savefig(output_dir / "fig02_heat_flux.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharey=True)
    axes = axes.ravel()
    axes[0].plot(profiles["w_var_over_wstar_sq"], z_over_zi)
    axes[0].set_xlabel(r"$\langle w'^2\rangle/w_*^2$")
    axes[1].plot(profiles["horizontal_var_over_wstar_sq"], z_over_zi)
    axes[1].set_xlabel(r"$0.5(\langle u'^2\rangle+\langle v'^2\rangle)/w_*^2$")
    axes[2].plot(profiles["theta_var_over_thetastar_sq"], z_over_zi)
    axes[2].set_xlabel(r"$\langle \theta'^2\rangle/\theta_*^2$")
    axes[3].plot(profiles["p_var_over_wstar4"], z_over_zi)
    axes[3].set_xlabel(r"$\langle p'^2\rangle/w_*^4$")
    for ax in axes:
        ax.set_ylabel(r"$z/z_i$")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title_suffix)
    fig.tight_layout()
    fig.savefig(output_dir / "fig03_06_variances.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    axes[0].plot(profiles["w3_over_wstar3"], z_over_zi)
    axes[0].set_xlabel(r"$\langle w'^3\rangle/w_*^3$")
    axes[1].plot(profiles["skewness"], z_over_zi)
    axes[1].set_xlabel(r"$Sk_w$")
    for ax in axes:
        ax.set_ylabel(r"$z/z_i$")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title_suffix)
    fig.tight_layout()
    fig.savefig(output_dir / "fig07_08_higher_moments.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(6, 4))
    plt.plot(profiles["epsilon_zi_over_wstar3"], z_over_zi)
    plt.xlabel(r"$\langle\epsilon\rangle z_i / w_*^3$")
    plt.xlim(left=0.0)
    plt.ylim(0.0, 1.5)
    finish_profile_plot(output_dir / "fig09_dissipation.png", title_suffix)

    plt.figure(figsize=(6, 4))
    plt.plot(profiles["buoyancy_production"], z_over_zi, label="buoyancy production")
    plt.plot(profiles["d_w_transport"], z_over_zi, label=r"$d\langle w'E'\rangle/dz$")
    plt.plot(profiles["d_p_transport"], z_over_zi, label=r"$d\langle p'w'\rangle/dz$")
    plt.xlabel(r"normalized budget term")
    plt.legend(fontsize=8)
    finish_profile_plot(output_dir / "fig10_11_energy_budget_terms.png", title_suffix)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    axes[0].plot(profiles["alpha_u"], z_over_zi)
    axes[0].set_xlabel(r"$\alpha_u$")
    axes[1].plot(profiles["w_u_over_wstar"], z_over_zi)
    axes[1].set_xlabel(r"$w_u/w_*$")
    axes[2].plot(profiles["theta_u_excess_over_thetastar"], z_over_zi)
    axes[2].set_xlabel(r"$(\theta_u-\langle\theta\rangle)/\theta_*$")
    for ax in axes:
        ax.set_ylabel(r"$z/z_i$")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title_suffix)
    fig.tight_layout()
    fig.savefig(output_dir / "fig12_14_conditional_updrafts.png", dpi=180)
    plt.close(fig)

    print(f"Read profiles from: {input_dir}")
    print(f"Wrote figures to: {output_dir}")


if __name__ == "__main__":
    main()
