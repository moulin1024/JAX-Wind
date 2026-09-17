#!/usr/bin/env python3
"""Evaluate a frozen independent-DNS calibration against the unchanged holdout."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    calibration = json.loads(args.calibration.read_text())
    reference_path = root / "cases/SprayClosureValidation/reference.json"
    reference = json.loads(reference_path.read_text())
    protocol_path = root / "cases/SprayClosureValidation/dns_profile_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol == calibration["protocol"]
    assert calibration["source_fit_only"] and not calibration["holdout_accessed"]
    jax.config.update("jax_enable_x64", True)
    target = reference["holdout"]
    p = target["normalized_velocity_profile"]
    eta = np.linspace(p["eta_min"], p["eta_max"], 201)
    observed = (p["c0"] + p["c2"] * eta**2 + p["c4"] * eta**4) * np.exp(
        -p["A"] * eta**2
    )
    threshold = reference["assessment"]["relative_error_threshold"]
    results = []
    profiles = []
    for name in protocol["assessment"]["candidates"]:
        family = (
            "single_gaussian" if name.startswith("single") else "two_gaussian_mixture"
        )
        fit = calibration["families"][family]
        weights, coefficients = map(jnp.asarray, (fit["weights"], fit["coefficients"]))
        i1, i2 = map(float, gaussian_mixture_moments(weights, coefficients))
        alpha = (
            protocol["assessment"]["alpha_shape_ricou"]
            if name.endswith("ricou")
            else i1 / np.sqrt(2 * i2)
        )
        # Unit nozzle diameter/speed/density; a self-similar section at x-x0=1.
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
        f = lambda s, fit=fit: sum(
            w * np.exp(-c * s * s) for w, c in zip(fit["weights"], fit["coefficients"])
        )
        half = brentq(lambda s, f=f: f(s) - 0.5, 0.0, 10.0)
        predicted = f(eta / length)
        metrics = {
            "decay_coefficient": uc,
            "half_width_slope": length * half,
            "alpha_top_hat": alpha,
        }
        errors = {k: abs(v / target[k] - 1) for k, v in metrics.items()}
        local = np.abs(predicted / observed - 1)
        passed = bool(max(*errors.values(), max(local)) <= threshold)
        results.append(
            {
                "name": name,
                "family": family,
                "alpha": alpha,
                "predictions": metrics,
                "relative_errors": errors,
                "max_profile_relative_error": float(max(local)),
                "max_error_eta": float(eta[np.argmax(local)]),
                "profile_rms_over_Uc": float(
                    np.sqrt(np.mean((predicted - observed) ** 2))
                ),
                "screening_pass": passed,
            }
        )
        profiles.append(predicted)
    source_path = args.calibration.parent / "dns_pooled_profile.csv"
    source = np.loadtxt(source_path, delimiter=",", skiprows=1)
    source_on_holdout = [np.interp(eta, source[:, 0], source[:, i]) for i in (1, 2, 3)]
    source_mean, source_min, source_max = source_on_holdout
    source_comparison = {
        "interpretation": "Range across unique DNS source curves; NOT an uncertainty/confidence interval or independent realizations. Different study, Reynolds number, and development range.",
        "max_mean_profile_relative_difference": float(
            np.max(np.abs(source_mean / observed - 1))
        ),
        "last_eta": float(eta[-1]),
        "last_eta_dns_range": [float(source_min[-1]), float(source_max[-1])],
        "last_eta_dns_mean": float(source_mean[-1]),
        "last_eta_holdout": float(observed[-1]),
        "last_eta_holdout_screen_interval": [
            float((1 - threshold) * observed[-1]),
            float((1 + threshold) * observed[-1]),
        ],
    }
    files = [
        reference_path,
        protocol_path,
        Path(__file__),
        root / "tools/fit_dns_jet_profile.py",
        root / "src/jaxwind/jet_profiles.py",
        root / "src/jaxwind/spray_closure.py",
        root / "tests/physics/test_jet_profiles.py",
        root / "tools/assess_dns_jet_profile.sbatch",
    ]
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "backend": jax.default_backend(),
        "source_comparison": source_comparison,
        "source_profile_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "threshold": threshold,
        "results": results,
        "any_candidate_passes": any(r["screening_pass"] for r in results),
        "measured_spray_validated": False,
        "production_enabled": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(
        args.output / "holdout_profiles.csv",
        np.column_stack((eta, observed, *profiles)),
        delimiter=",",
        header=",".join(["eta", "holdout", *[r["name"] for r in results]]),
        comments="",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(eta, observed, "k-", label="Hussein holdout")
    axes[0].fill_between(
        eta, source_min, source_max, color="gray", alpha=0.25, label="DNS curve range"
    )
    for r, f in zip(results, profiles):
        axes[0].plot(eta, f, label=r["name"])
        axes[1].plot(eta, 100 * np.abs(f / observed - 1), label=r["name"])
    axes[1].axhline(100 * threshold, color="k", linestyle=":")
    axes[0].set(xlabel="r/(x-x0)", ylabel="U/Uc")
    axes[1].set(xlabel="r/(x-x0)", ylabel="Local relative error (%)")
    axes[0].legend(fontsize=7)
    fig.savefig(args.output / "profiles.png", dpi=150)
    plt.close(fig)
    print(json.dumps(results, indent=2))
    if not report["any_candidate_passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
