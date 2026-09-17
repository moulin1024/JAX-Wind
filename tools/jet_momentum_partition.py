"""Separate source-only integral calibration and frozen-holdout assessment."""

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

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "cases/SprayClosureValidation/momentum_partition_protocol.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(source_path, output):
    # This process never opens reference.json or an assessment report.
    source = json.loads(source_path.read_text())
    if not source["source_fit_only"] or source["holdout_accessed"]:
        raise ValueError("Not source-only calibration")
    protocol = json.loads(PROTOCOL.read_text())
    if source["protocol"] != json.loads(
        (
            ROOT / "cases/SprayClosureValidation/measured_profile_protocol.json"
        ).read_text()
    ):
        raise ValueError("Measured-source protocol mismatch")
    b = protocol["source"]["decay_coefficient"]
    families = {}
    for name in protocol["source"]["families"]:
        fit = source["families"][name]
        w, c = map(jnp.asarray, (fit["weights"], fit["coefficients"]))
        i1, i2 = map(float, gaussian_mixture_moments(w, c))
        chi = 8 * b * b * i2
        alpha = 2 * b * i1
        if not np.isfinite(chi) or chi <= 0:
            raise ValueError("Invalid mean momentum fraction")
        families[name] = {
            "weights": fit["weights"],
            "coefficients": fit["coefficients"],
            "I1": i1,
            "I2": i2,
            "mean_momentum_fraction": chi,
            "alpha_source": alpha,
            "stress_pressure_remainder_fraction": 1 - chi,
        }
    report = {
        "protocol": protocol,
        "families": families,
        "source_fit_only": True,
        "holdout_accessed": False,
        "measured_calibration_sha256": sha(source_path),
        "protocol_sha256": sha(PROTOCOL),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(families, indent=2))


def assess(calibration_path, output):
    calibration = json.loads(calibration_path.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    if (
        calibration["protocol"] != protocol
        or not calibration["source_fit_only"]
        or calibration["holdout_accessed"]
    ):
        raise ValueError("Calibration role mismatch")
    reference_path = ROOT / "cases/SprayClosureValidation/reference.json"
    reference = json.loads(reference_path.read_text())
    target = reference["holdout"]
    p = target["normalized_velocity_profile"]
    eta = np.linspace(p["eta_min"], p["eta_max"], 201)
    observed = (p["c0"] + p["c2"] * eta**2 + p["c4"] * eta**4) * np.exp(
        -p["A"] * eta**2
    )
    threshold = reference["assessment"]["relative_error_threshold"]
    rows = []
    curves = []
    for family in protocol["source"]["families"]:
        fit = calibration["families"][family]
        w, c = map(jnp.asarray, (fit["weights"], fit["coefficients"]))
        f = lambda x, fit=fit: sum(
            w * np.exp(-c * x * x) for w, c in zip(fit["weights"], fit["coefficients"])
        )
        half = brentq(lambda x, f=f: f(x) - 0.5, 0.0, 10.0)
        for mode in protocol["assessment"]["entrainment_modes"]:
            alpha = (
                fit["alpha_source"]
                if mode == "source_integral"
                else protocol["assessment"]["alpha_ricou"]
            )
            total = np.pi / 4
            mean = fit["mean_momentum_fraction"] * total
            mass = 2 * alpha * np.sqrt(np.pi * total)
            uc, length = map(
                float,
                round_jet_gaussian_mixture(
                    RoundJetFlux(mass, mean, 0.0, jnp.array([mass])), 1.0, w, c
                ),
            )
            if mode == "source_integral":
                np.testing.assert_allclose(
                    [uc, length],
                    [protocol["source"]["decay_coefficient"], 1.0],
                    rtol=2e-14,
                )
            predicted = f(eta / length)
            metrics = {
                "decay_coefficient": uc,
                "half_width_slope": length * half,
                "alpha_top_hat": alpha,
            }
            errors = {k: abs(v / target[k] - 1) for k, v in metrics.items()}
            local = np.abs(predicted / observed - 1)
            rows.append(
                {
                    "name": family + "_" + mode,
                    "predictions": metrics,
                    "relative_errors": errors,
                    "mean_momentum_flux": mean,
                    "stress_pressure_flux": total - mean,
                    "total_momentum_flux": total,
                    "mean_momentum_fraction": fit["mean_momentum_fraction"],
                    "max_profile_relative_error": float(local.max()),
                    "max_error_eta": float(eta[np.argmax(local)]),
                    "profile_rms_over_Uc": float(
                        np.sqrt(np.mean((predicted - observed) ** 2))
                    ),
                    "screening_pass": bool(
                        max(*errors.values(), local.max()) <= threshold
                    ),
                }
            )
            curves.append(predicted)
    files = [
        PROTOCOL,
        reference_path,
        Path(__file__),
        ROOT / "tools/assess_jet_momentum_partition.sbatch",
        ROOT / "src/jaxwind/jet_profiles.py",
        ROOT / "src/jaxwind/spray_closure.py",
    ]
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "results": rows,
        "threshold": threshold,
        "calibration_sha256": sha(calibration_path),
        "sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "any_candidate_passes": any(r["screening_pass"] for r in rows),
        "production_enabled": False,
        "measured_spray_validated": False,
        "source_integral_identity_passed": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(
        output / "holdout_profiles.csv",
        np.column_stack((eta, observed, *curves)),
        delimiter=",",
        header=",".join(["eta", "holdout", *[r["name"] for r in rows]]),
        comments="",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(eta, observed, "k-", label="Hussein holdout")
    for row, curve in zip(rows, curves):
        label = row["name"].replace("_", " ")
        axes[0].plot(eta, curve, label=label)
        axes[1].plot(eta, 100 * np.abs(curve / observed - 1))
    axes[0].set(xlabel="Similarity radius", ylabel="U/Uc", xlim=(0, 0.2))
    axes[1].set(
        xlabel="Similarity radius", ylabel="Local relative error (%)", xlim=(0, 0.2)
    )
    axes[1].axhline(100 * threshold, color="k", linestyle=":")
    axes[0].legend(fontsize=6)
    fig.savefig(output / "profiles.png", dpi=160)
    plt.close(fig)
    print(json.dumps(rows, indent=2))
    if not report["any_candidate_passes"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "assess"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    (calibrate if args.mode == "calibrate" else assess)(args.input, args.output)


if __name__ == "__main__":
    main()
