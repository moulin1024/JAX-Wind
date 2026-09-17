"""Independent screening after source/numerical gates; retain rejected controls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = root/"cases/SprayClosureValidation/constrained_profile_assessment_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    for line in (args.solutions/"source.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if digest(root/name.strip()) != expected:
            raise ValueError(f"source changed after qualification: {name}")
    source = args.solutions/"assessment.json"
    document = json.loads(source.read_text())
    if document["holdout_accessed"]:
        raise ValueError("source-only qualification required")
    provenance = {str(p): digest(p) for p in [source, protocol_path]}
    eligible, rejected = [], []
    for c3 in protocol["frozen_cases"]:
        case = next(c for c in document["cases"] if c["c3"] == c3)
        final = case["stages"][-1]
        if final["constants"] != {**protocol["coefficients"], "c3": c3}:
            raise ValueError("coefficient protocol mismatch")
        reason = None
        if not case["qualification"].get("numerical_pass", False):
            reason = "numerical qualification failed"
        elif abs(final["half_width"]/protocol["source_widths"][str(c3)]-1) > protocol["source_relative_tolerance"]:
            reason = "independent source-width gate failed"
        if reason:
            rejected.append({"c3": c3, "reason": reason, "qualification": case["qualification"],
                "source_width_relative_error": final.get("source_width_relative_error"), "holdout_assessed": False})
            continue
        index = len(case["stages"])-1
        state = args.solutions/f"constrained_{c3}_{index}.npz"
        baseline = args.baseline/f"standard_{c3}_1_6.npz"
        data, old = np.load(state), np.load(baseline)
        np.testing.assert_array_equal(data["eta"], old["eta"])
        eligible.append((c3, final, data["eta"], data["F"], old["F"]))
        provenance.update({str(p): digest(p) for p in [state, baseline]})
    if not eligible:
        raise ValueError("no source-qualified candidate; holdout remains unopened")
    reference_path = root/"cases/SprayClosureValidation/reference.json"
    reference = json.loads(reference_path.read_text())
    provenance[str(reference_path)] = digest(reference_path)
    target = reference["holdout"]
    shape = target["normalized_velocity_profile"]
    eta = np.linspace(shape["eta_min"], shape["eta_max"], 201)
    observed = (shape["c0"]+shape["c2"]*eta**2+shape["c4"]*eta**4)*np.exp(-shape["A"]*eta**2)
    threshold = reference["assessment"]["relative_error_threshold"]
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for c3, final, positions, predicted, old in eligible:
        np.testing.assert_array_equal(eta, positions)
        metrics = {"decay_coefficient": final["decay_coefficient"], "half_width_slope": final["half_width"], "alpha_top_hat": final["alpha"]}
        errors = {key: abs(value/target[key]-1) for key, value in metrics.items()}
        local = abs(predicted/observed-1)
        rows.append({"c3": c3, "predictions": metrics, "relative_errors": errors,
            "maximum_profile_relative_error": float(local.max()), "maximum_error_eta": float(eta[local.argmax()]),
            "maximum_profile_change_from_unconstrained_over_Uc": float(abs(predicted-old).max()),
            "profile_rms_over_Uc": float(np.sqrt(np.mean((predicted-observed)**2))),
            "screening_pass": bool(max(local.max(), *errors.values()) <= threshold)})
        np.savetxt(args.output/f"profiles_{c3}.csv", np.column_stack([eta, observed, old, predicted]),
                   delimiter=",", header="eta,Hussein1994,unconstrained,constrained", comments="")
    result = {"results": rows, "rejected_without_holdout": rejected, "input_sha256": provenance,
              "threshold": threshold, "production_enabled": False, "uncertainty": protocol["uncertainty"]}
    (args.output/"assessment.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
