#!/usr/bin/env python3
"""Assess fixed independent profile candidates; never fit the held-out target."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.jet_profiles import round_jet_squared_lorentzian
from jaxwind.spray_closure import RoundJetFlux, advance_round_jet, round_jet_gaussian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    ref_path = root / "cases/SprayClosureValidation/reference.json"
    candidates_path = root / "cases/SprayClosureValidation/profile_candidates.json"
    reference = json.loads(ref_path.read_text())
    protocol = json.loads(candidates_path.read_text())
    target = reference["holdout"]
    profile = target["normalized_velocity_profile"]
    threshold = reference["assessment"]["relative_error_threshold"]
    eta = np.linspace(profile["eta_min"], profile["eta_max"], 201)
    measured = (
        profile["c0"] + profile["c2"] * eta**2 + profile["c4"] * eta**4
    ) * np.exp(-profile["A"] * eta**2)
    jax.config.update("jax_enable_x64", True)
    predictions = []
    results = []
    for spec in protocol["candidates"]:
        alpha = spec.get("alpha", None)
        if alpha is None:
            alpha = 1 / np.sqrt(2 * spec["gaussian_C"])
        lorentz = spec["profile"] == "squared_lorentzian"
        reconstruct = round_jet_squared_lorentzian if lorentz else round_jet_gaussian
        width_factor = np.sqrt(np.sqrt(2) - 1) if lorentz else np.sqrt(np.log(2))
        source = RoundJetFlux(np.pi / 4, np.pi / 4, 0.0, jnp.array([np.pi / 4]))
        advance = jax.jit(
            lambda q, ds, alpha=alpha: advance_round_jet(
                q, ds, 1.0, 0.0, jnp.array([1.0]), alpha=alpha
            )
        )
        step_results = []
        for ds in (8.0, 4.0, 1.0, 0.25):
            q = source
            x = np.arange(round(120 / ds) + 1) * ds
            inv_uc = []
            widths = []
            masses = []
            for i, _ in enumerate(x):
                uc, b = reconstruct(q, 1.0)
                inv_uc.append(1 / float(uc))
                widths.append(float(b) * width_factor)
                masses.append(float(q.mass))
                if i < len(x) - 1:
                    q = advance(q, ds)
            selected = (x >= 40) & (x <= 100)
            decay = 1 / np.polyfit(x[selected], np.asarray(inv_uc)[selected], 1)[0]
            spread = np.polyfit(x[selected], np.asarray(widths)[selected], 1)[0]
            intake = np.polyfit(x[selected], np.asarray(masses)[selected], 1)[0] / np.pi
            metrics = {
                "decay_coefficient": decay,
                "half_width_slope": spread,
                "alpha_top_hat": intake,
            }
            errors = {k: abs(v / target[k] - 1) for k, v in metrics.items()}
            slope = spread / width_factor
            predicted = (
                (1 + (eta / slope) ** 2) ** -2
                if lorentz
                else np.exp(-((eta / slope) ** 2))
            )
            local = np.abs(predicted / measured - 1)
            step_results.append(
                {
                    "ds_over_D": ds,
                    "predictions": metrics,
                    "relative_errors": errors,
                    "max_profile_relative_error": float(np.max(local)),
                    "profile_rms_over_Uc": float(
                        np.sqrt(np.mean((predicted - measured) ** 2))
                    ),
                    "screening_pass": bool(
                        max(*errors.values(), np.max(local)) <= threshold
                    ),
                }
            )
        predictions.append(predicted)
        results.append(
            {
                "candidate": spec,
                "alpha_used": float(alpha),
                "axial_steps": step_results,
                "screening_pass": all(r["screening_pass"] for r in step_results),
            }
        )
    np.savetxt(
        args.output / "profiles.csv",
        np.column_stack([eta, measured, *predictions]),
        delimiter=",",
        header=",".join(["eta", "holdout", *[r["candidate"]["name"] for r in results]]),
        comments="",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(eta, measured, "k-", label="Held-out experimental fit")
    for result, predicted in zip(results, predictions):
        label = result["candidate"]["name"]
        axes[0].plot(eta, predicted, label=label)
        axes[1].plot(eta, 100 * np.abs(predicted / measured - 1), label=label)
    axes[1].axhline(100 * threshold, color="k", linestyle=":")
    axes[0].set(xlabel="r / (x - x0)", ylabel="U / Uc")
    axes[1].set(xlabel="r / (x - x0)", ylabel="Local relative error (%)")
    axes[0].legend(fontsize=7)
    fig.savefig(args.output / "profiles.png", dpi=160)
    plt.close(fig)
    files = [
        ref_path,
        candidates_path,
        Path(__file__),
        root / "src/jaxwind/jet_profiles.py",
        root / "src/jaxwind/spray_closure.py",
        root / "tests/physics/test_jet_profiles.py",
        root / "tools/assess_jet_profiles.sbatch",
    ]
    report = {
        "job_id": os.getenv("SLURM_JOB_ID"),
        "backend": jax.default_backend(),
        "protocol": protocol,
        "threshold": threshold,
        "results": results,
        "physical_spray_validated": False,
        "any_candidate_passes": any(r["screening_pass"] for r in results),
        "sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for r in results:
        last = r["axial_steps"][-1]
        m = last["predictions"]
        rows.append(
            f"| {r['candidate']['name']} | {m['decay_coefficient']:.5f} | {m['half_width_slope']:.6f} | {r['alpha_used']:.6f} | {100 * last['max_profile_relative_error']:.3f}% | {r['screening_pass']} |"
        )
    text = "\n".join(
        [
            "# Independent profile-candidate assessment",
            "",
            f"Gpudev job {report['job_id']}; {report['backend']}. No held-out fitting.",
            "",
            "| Candidate | Decay B | Width slope | Alpha | Maximum profile error | All screens pass |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            *rows,
            "",
            f"Published fits, not raw observations. {100 * threshold:g}% engineering screen; no uncertainty intervals available.",
            "Huck C transfers a small-droplet gas proxy from a separate two-phase experiment; applicability is unproven.",
            "No candidate is enabled in production waterjet cases. This does not validate measured droplet transport or LES coupling.",
            "",
        ]
    )
    (args.output / "report.md").write_text(text)
    print(text)
    if not report["any_candidate_passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
