"""Assess the source-frozen 1990 Pope convention against the unchanged holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = (
        root / "cases/SprayClosureValidation/pope_1990_profile_protocol.json"
    )
    protocol = json.loads(protocol_path.read_text())
    for line in (args.solutions / "source.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if digest(root / name.strip()) != expected:
            raise ValueError(f"solver/protocol changed since qualification: {name}")
    qualified = []
    provenance = {str(protocol_path): digest(protocol_path)}
    # All candidate gates precede access to the held-out experimental values.
    for normal, name in enumerate(protocol["candidates"]):
        label = f"standard_0.79_{normal}"
        report_path = args.solutions / (label + ".json")
        profile_path = args.solutions / (label + "_6.npz")
        result = json.loads(report_path.read_text())
        final = result["stages"][-1]
        if not result["qualification"].get("numerical_pass", False):
            raise ValueError(f"{name}: numerical qualification failed")
        if result["holdout_accessed"] or final["constants"] != protocol["constants"]:
            raise ValueError(f"{name}: protocol/source mismatch")
        source_error = abs(
            final["half_width"] / protocol["source_gate"]["half_width_slope"] - 1
        )
        if source_error > protocol["source_gate"]["relative_tolerance"]:
            raise ValueError(f"{name}: independent source-rate gate failed")
        data = np.load(profile_path)
        qualified.append((name, final, data["eta"], data["F"], result["qualification"]))
        provenance.update({str(p): digest(p) for p in (report_path, profile_path)})
    reference_path = root / "cases/SprayClosureValidation/reference.json"
    reference = json.loads(reference_path.read_text())
    provenance[str(reference_path)] = digest(reference_path)
    target = reference["holdout"]
    shape = target["normalized_velocity_profile"]
    eta = np.linspace(shape["eta_min"], shape["eta_max"], 201)
    observed = (shape["c0"] + shape["c2"] * eta**2 + shape["c4"] * eta**4) * np.exp(
        -shape["A"] * eta**2
    )
    if np.any(observed <= 0):
        raise ValueError("relative-error denominator is not strictly positive")
    threshold = reference["assessment"]["relative_error_threshold"]
    rows = []
    curves = []
    for name, final, positions, predicted, qualification in qualified:
        np.testing.assert_array_equal(positions, eta)
        metrics = {
            "decay_coefficient": final["decay_coefficient"],
            "half_width_slope": final["half_width"],
            "alpha_top_hat": final["alpha"],
        }
        errors = {key: abs(value / target[key] - 1) for key, value in metrics.items()}
        local = abs(predicted / observed - 1)
        rows.append(
            {
                "name": name,
                "predictions": metrics,
                "relative_errors": errors,
                "source_rate_relative_error": final["historical_rate_relative_error"],
                "numerical_qualification": qualification,
                "max_profile_relative_error": float(local.max()),
                "max_error_eta": float(eta[local.argmax()]),
                "profile_rms_over_Uc": float(
                    np.sqrt(np.mean((predicted - observed) ** 2))
                ),
                "screening_pass": bool(max(local.max(), *errors.values()) <= threshold),
            }
        )
        curves.append(predicted)
    args.output.mkdir(parents=True, exist_ok=True)
    document = {
        "protocol": protocol,
        "input_sha256": provenance,
        "relative_error_threshold": threshold,
        "results": rows,
        "experimental_uncertainty": f"{100 * threshold:g}% engineering screen, not a confidence interval",
        "source_reproduction_scope": "source spreading agreement, not a claim of reproducing the undocumented 1978 implementation",
        "production_default_changed": False,
    }
    (args.output / "assessment.json").write_text(json.dumps(document, indent=2) + "\n")
    np.savetxt(
        args.output / "profiles.csv",
        np.column_stack([eta, observed, *curves]),
        delimiter=",",
        header="eta,Hussein1994,shear_only,normal_strain",
        comments="",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(eta, observed, "k", label="Hussein 1994 holdout")
    for label, f in zip(["Shear production", "Normal strain added"], curves):
        axes[0].plot(eta, f, label=label)
        axes[1].plot(eta, 100 * (f / observed - 1), label=label)
    axes[1].axhspan(-100 * threshold, 100 * threshold, color="0.9", label=f"{100 * threshold:g}% screen")
    axes[1].axhline(0, color="k", linewidth=0.5)
    for axis in axes:
        axis.set_xlabel(r"$r/(x-x_0)$")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"$U/U_c$")
    axes[1].set_ylabel("Signed local error (%)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Independent 1990 Pope-convention profile assessment")
    fig.savefig(args.output / "profile-assessment.png", dpi=160)
    fig.savefig(args.output / "profile-assessment.pdf")
    plt.close(fig)
    print(
        json.dumps({"results": rows, "output": str(args.output)}, indent=2), flush=True
    )


if __name__ == "__main__":
    main()
