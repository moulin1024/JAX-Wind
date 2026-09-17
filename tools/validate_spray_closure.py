#!/usr/bin/env python3
"""Reproduce the independently referenced canonical round-jet comparison.

This assesses flux-space integral closure and conservative coarse-plane
integration. It does not run the 3D LES or validate inertial spray dispersion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.spray_closure import (
    RoundJetFlux,
    advance_round_jet,
    round_jet_gaussian,
    round_jet_plane_fluxes,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    ref_path = root / "cases/SprayClosureValidation/reference.json"
    ref = json.loads(ref_path.read_text())
    alpha = ref["calibration"]["alpha_top_hat"]
    jax.config.update("jax_enable_x64", True)
    # Normalized top-hat source fluxes, D=U0=rho=1. Only asymptotic slopes
    # are assessed; no nozzle profile or measured virtual origin is imposed.
    source = RoundJetFlux(
        np.pi / 4, np.pi / 4, 3e5 * np.pi / 4, jnp.array([0.99, 0.01]) * np.pi / 4
    )
    intake_y = jnp.array([0.98, 0.02])
    advance = jax.jit(
        lambda f, ds: advance_round_jet(f, ds, 1.0, 3.1e5, intake_y, alpha=alpha)
    )
    runs = []
    target = ref["holdout"]
    threshold = ref["assessment"]["relative_error_threshold"]
    for ds in (8.0, 4.0, 1.0, 0.25):
        q = source
        x, uc, widths, masses = [0.0], [], [], []
        for i in range(round(120 / ds) + 1):
            u, b = round_jet_gaussian(q, 1.0)
            uc.append(float(u))
            widths.append(float(b * jnp.sqrt(jnp.log(2.0))))
            masses.append(float(q.mass))
            if i < round(120 / ds):
                q = advance(q, ds)
                x.append((i + 1) * ds)
        x = np.asarray(x)
        selected = (x >= 40) & (x <= 100)
        decay = 1 / np.polyfit(x[selected], 1 / np.asarray(uc)[selected], 1)[0]
        spread = np.polyfit(x[selected], np.asarray(widths)[selected], 1)[0]
        mass_slope = np.polyfit(x[selected], np.asarray(masses)[selected], 1)[0]
        entrainment_alpha = mass_slope / (2 * np.sqrt(np.pi * float(q.momentum)))
        predictions = {
            "decay_coefficient": decay,
            "half_width_slope": spread,
            "alpha_top_hat": entrainment_alpha,
        }
        errors = {
            key: abs(value / target[key] - 1) for key, value in predictions.items()
        }
        runs.append(
            {
                "axial_step_over_D": ds,
                "predictions": predictions,
                "relative_errors": errors,
                "screening_pass": all(e <= threshold for e in errors.values()),
            }
        )
    projection = []
    _, radius = round_jet_gaussian(q, 1.0)
    for width_over_cell in (0.25, 0.5, 1.0, 2.0, 4.0):
        spacing = float(radius) / width_over_cell
        edges = jnp.arange(-32, 33) * spacing
        projected = round_jet_plane_fluxes(
            q, 1.0, edges, edges, center=(0.37 * spacing, -0.21 * spacing)
        )
        errors = [
            abs(float(jnp.sum(p)) / float(f) - 1) for p, f in zip(projected[:3], q[:3])
        ]
        species_error = float(
            jnp.max(jnp.abs(jnp.sum(projected.species, axis=(1, 2)) / q.species - 1))
        )
        projection.append(
            {
                "radius_over_cell": width_over_cell,
                "max_relative_flux_error": max(*errors, species_error),
            }
        )
    profile_ref = target["normalized_velocity_profile"]
    eta = np.linspace(profile_ref["eta_min"], profile_ref["eta_max"], 201)
    observed_shape = (
        profile_ref["c0"] + profile_ref["c2"] * eta**2 + profile_ref["c4"] * eta**4
    ) * np.exp(-profile_ref["A"] * eta**2)
    radius_slope = runs[-1]["predictions"]["half_width_slope"] / np.sqrt(np.log(2.0))
    predicted_shape = np.exp(-((eta / radius_slope) ** 2))
    local_errors = np.abs(predicted_shape / observed_shape - 1)
    profile_assessment = {
        "reference_type": "published experimental fit; not raw points",
        "coordinate": profile_ref["coordinate"],
        "eta_range": [float(eta[0]), float(eta[-1])],
        "max_local_relative_error": float(np.max(local_errors)),
        "max_error_eta": float(eta[np.argmax(local_errors)]),
        "rms_error_over_centerline_velocity": float(
            np.sqrt(np.mean((predicted_shape - observed_shape) ** 2))
        ),
        "screening_pass": bool(np.max(local_errors) <= threshold),
    }
    np.savetxt(
        args.output / "radial_profile.csv",
        np.column_stack((eta, observed_shape, predicted_shape, local_errors)),
        delimiter=",",
        header="eta,experiment_fit_U_over_Uc,model_U_over_Uc,local_relative_error",
        comments="",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].plot(eta, observed_shape, label="Independent experimental fit")
    axes[0].plot(eta, predicted_shape, "--", label="Gaussian integral model")
    axes[0].set(xlabel="r / (x - x0)", ylabel="U / Ucentre")
    axes[0].legend(fontsize=8)
    axes[1].plot(eta, 100 * local_errors)
    axes[1].axhline(100 * threshold, color="black", linestyle=":", label=f"{100 * threshold:g}% screen")
    axes[1].set(xlabel="r / (x - x0)", ylabel="Local relative error (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Held-out jet profile: published fit, not raw measurement points")
    fig.savefig(args.output / "radial_profile.png", dpi=180)
    fig.savefig(args.output / "radial_profile.svg")
    plt.close(fig)
    source_files = [
        ref_path,
        root / "src/jaxwind/spray_closure.py",
        Path(__file__).resolve(),
        root / "tests/physics/test_spray_closure.py",
        root / "tools/validate_spray_closure.sbatch",
    ]
    report = {
        "scope": ref["scope"],
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "calibration": ref["calibration"],
        "holdout": target,
        "assessment": ref["assessment"],
        "axial_step_runs": runs,
        "coarse_plane_flux_integration": projection,
        "integral_summary_screening_pass": all(r["screening_pass"] for r in runs),
        "radial_profile": profile_assessment,
        "experimental_screening_pass": all(r["screening_pass"] for r in runs)
        and profile_assessment["screening_pass"],
        "dispersion_validation": "pending independent particle-transport reference",
        "production_spray_validated": False,
        "waterjet_enabled": False,
        "sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for name, value in runs[-1]["predictions"].items():
        rows.append(
            f"| {name} | {value:.8g} | {target[name]:.8g} | {100 * runs[-1]['relative_errors'][name]:.3f}% |"
        )
    message = "\n".join(
        [
            "# Canonical entrainment component assessment",
            "",
            f"Job {report['slurm_job_id']}; backend {report['backend']}.",
            "",
            "Fixed Ricou–Spalding top-hat alpha=0.08; independent Hussein et al. LDA/FHW summaries.",
            f"The {100 * threshold:g}% engineering screen is not a measurement confidence interval.",
            "",
            "| Observable | Prediction | Experiment | Relative error |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            f"Integral-summary screens pass: {report['integral_summary_screening_pass']}.",
            f"Radial-profile screen passes: {profile_assessment['screening_pass']}.",
            f"Maximum local radial-profile error: {100 * profile_assessment['max_local_relative_error']:.3f}%.",
            f"Complete experimental screen passes: {report['experimental_screening_pass']}.",
            "Axial steps: 8, 4, 1, 0.25 D. This tests the exact integral advance, not LES convergence.",
            f"Maximum coarse-plane flux error: {max(p['max_relative_flux_error'] for p in projection):.3e}.",
            "",
            "Particle dispersion remains unvalidated against independent data. The homogeneous",
            "stochastic update is numerically verified separately; no inhomogeneous drift or",
            "two-way SGS energy coupling is implemented. The production waterjet is unchanged.",
            "No evaporation, confined-coflow, near-nozzle, or waterjet accuracy claim follows.",
            "",
        ]
    )
    (args.output / "report.md").write_text(message)
    print(message)
    if not report["experimental_screening_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
