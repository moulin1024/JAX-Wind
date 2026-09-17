"""Assess a frozen measured-profile calibration against unchanged holdout data."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from jaxwind.jet_profiles import gaussian_mixture_moments, round_jet_gaussian_mixture
from jaxwind.spray_closure import RoundJetFlux


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "cases/SprayClosureValidation/measured_profile_protocol.json"
    reference_path = root / "cases/SprayClosureValidation/reference.json"
    protocol = json.loads(protocol_path.read_text())
    reference = json.loads(reference_path.read_text())
    calibration = json.loads(args.calibration.read_text())
    if (
        calibration["protocol"] != protocol
        or not calibration["source_fit_only"]
        or calibration["holdout_accessed"]
    ):
        raise ValueError("Calibration provenance/role mismatch")
    jax.config.update("jax_enable_x64", True)
    target = reference["holdout"]
    p = target["normalized_velocity_profile"]
    eta = np.linspace(p["eta_min"], p["eta_max"], 201)
    observed = (p["c0"] + p["c2"] * eta**2 + p["c4"] * eta**4) * np.exp(
        -p["A"] * eta**2
    )
    threshold = reference["assessment"]["relative_error_threshold"]
    rows = []
    profiles = []
    for name in protocol["assessment"]["candidates"]:
        family = (
            "published_gaussian"
            if name == "published_gaussian"
            else "single_gaussian"
            if name.startswith("single")
            else "two_gaussian_mixture"
        )
        fit = calibration["families"][family]
        weights, coefficients = map(jnp.asarray, (fit["weights"], fit["coefficients"]))
        i1, i2 = map(float, gaussian_mixture_moments(weights, coefficients))
        alpha = (
            protocol["assessment"]["alpha_shape_ricou"]
            if name.endswith("ricou")
            else i1 / np.sqrt(2 * i2)
        )
        momentum = np.pi / 4
        mass = 2 * alpha * np.sqrt(np.pi * momentum)
        uc, length = map(
            float,
            round_jet_gaussian_mixture(
                RoundJetFlux(mass, momentum, 0.0, jnp.array([mass])),
                1.0,
                weights,
                coefficients,
            ),
        )
        f = lambda x, fit=fit: sum(
            w * np.exp(-c * x * x) for w, c in zip(fit["weights"], fit["coefficients"])
        )
        half = brentq(lambda x, f=f: f(x) - 0.5, 0.0, 10.0)
        prediction = f(eta / length)
        metrics = {
            "decay_coefficient": uc,
            "half_width_slope": length * half,
            "alpha_top_hat": alpha,
        }
        errors = {k: abs(v / target[k] - 1) for k, v in metrics.items()}
        local = np.abs(prediction / observed - 1)
        cutoff = calibration["source_support"][1]
        tail_mass = (
            sum(
                w * np.exp(-c * cutoff**2) / (2 * c)
                for w, c in zip(fit["weights"], fit["coefficients"])
            )
            / i1
        )
        rows.append(
            {
                "name": name,
                "predictions": metrics,
                "relative_errors": errors,
                "max_profile_relative_error": float(local.max()),
                "max_error_eta": float(eta[np.argmax(local)]),
                "profile_rms_over_Uc": float(
                    np.sqrt(np.mean((prediction - observed) ** 2))
                ),
                "unobserved_source_tail_mass_fraction": float(tail_mass),
                "source_weighted_rms_over_Uc": fit["source_weighted_rms_over_Uc"],
                "screening_pass": bool(max(*errors.values(), local.max()) <= threshold),
            }
        )
        profiles.append(prediction)
    source_path = args.calibration.parent / "measured_profile.csv"
    source = np.loadtxt(source_path, delimiter=",", skiprows=1)
    # Restrict the source-vs-target comparison to actual archived radial support.
    inside = (eta >= source[0, 0]) & (eta <= source[-1, 0])
    interp = np.interp(eta[inside], source[:, 0], source[:, 1])
    comparison = {
        "interpretation": "Secondary author-provided experimental curve, not raw data or a confidence interval. No source-origin/width adjustment fitted to holdout.",
        "source_half_width": calibration["source_half_width"],
        "max_profile_relative_difference": float(
            np.max(np.abs(interp / observed[inside] - 1))
        ),
        "last_eta": float(eta[-1]),
        "last_eta_source": float(np.interp(eta[-1], source[:, 0], source[:, 1])),
        "last_eta_holdout": float(observed[-1]),
        "last_eta_holdout_screen_interval": [
            float((1 - threshold) * observed[-1]),
            float((1 + threshold) * observed[-1]),
        ],
    }
    files = [
        protocol_path,
        reference_path,
        Path(__file__),
        root / "tools/fit_measured_jet_profile.py",
        root / "tools/assess_measured_jet_profile.sbatch",
        root / "src/jaxwind/jet_profiles.py",
        root / "src/jaxwind/spray_closure.py",
        root / "tests/physics/test_jet_profiles.py",
    ]
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "backend": jax.default_backend(),
        "threshold": threshold,
        "source_comparison": comparison,
        "results": rows,
        "calibration_sha256": sha(args.calibration),
        "source_profile_sha256": sha(source_path),
        "sha256": {str(p.relative_to(root)): sha(p) for p in files},
        "any_candidate_passes": any(r["screening_pass"] for r in rows),
        "production_enabled": False,
        "measured_spray_validated": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(
        args.output / "holdout_profiles.csv",
        np.column_stack((eta, observed, *profiles)),
        delimiter=",",
        header=",".join(["eta", "holdout", *[r["name"] for r in rows]]),
        comments="",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(eta, observed, "k-", label="Hussein holdout")
    axes[0].plot(
        source[:, 0], source[:, 1], "o", ms=3, label="Archived PL measured curve"
    )
    for row, curve in zip(rows, profiles):
        axes[0].plot(eta, curve, label=row["name"])
        axes[1].plot(eta, 100 * np.abs(curve / observed - 1), label=row["name"])
    axes[0].set(xlabel="Similarity radius", ylabel="U/Uc", xlim=(0, 0.2))
    axes[1].set(xlabel="Similarity radius", ylabel="Local relative error (%)")
    axes[1].axhline(100 * threshold, color="k", ls=":")
    axes[0].legend(fontsize=7)
    fig.savefig(args.output / "profiles.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"source_comparison": comparison, "results": rows}, indent=2))
    if not report["any_candidate_passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
