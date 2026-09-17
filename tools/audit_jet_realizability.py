"""Check local momentum-compatible PSD feasibility; never changes a profile."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_jet_stress import fields
from jet_realizability import maximal_realizable_fraction, momentum_stress_affine
from jet_stress import velocity_gradient
from scipy.integrate import simpson


def assess(path, constants, count):
    eta, integral, f, fp, k, n, cut = fields(path, constants, count, edge_graded=True)
    keep = (eta <= cut) & (k > 1e-12 * k.max())
    eta, integral, f, fp, k, n = (v[keep] for v in (eta, integral, f, fp, k, n))
    rows = []
    for radial_only in [True, False]:
        a, b = momentum_stress_affine(eta, integral, f, k, n, radial_shear_only=radial_only)
        a, b = a / k[:, None, None], b / k[:, None, None]
        original = np.linalg.eigvalsh(a + b)[:, 0]
        violating = original < -1e-10
        fractions = np.ones_like(k)
        infeasible, indeterminate = [], []
        for i in np.flatnonzero(violating):
            result = maximal_realizable_fraction(a[i], b[i])
            if result["feasible"] is False:
                infeasible.append(int(i))
            elif result["feasible"] is None:
                indeterminate.append(int(i))
            else:
                fractions[i] = result["fraction"]
        updated = np.linalg.eigvalsh(a + fractions[:, None, None] * b)[:, 0]
        gradient = velocity_gradient(eta, integral, f, fp, radial_shear_only=radial_only)
        strain = (gradient + np.swapaxes(gradient, -1, -2)) / 2
        maximum = np.linalg.eigvalsh(strain)[:, -1]
        # Exact PSD cap for the old prescribed gradient, then re-enforce momentum.
        fixed_gradient_fraction = np.minimum(1, k / (3 * n * maximum))
        naive = np.linalg.eigvalsh(a + fixed_gradient_fraction[:, None, None] * b)[:, 0]
        rows.append({
            "gradient": "radial_shear_only" if radial_only else "complete_self_similar",
            "solved_domain_outer_eta": float(cut),
            "samples": int(k.size), "violating_samples": int(violating.sum()),
            "infeasible_samples": len(infeasible), "indeterminate_samples": len(indeterminate),
            "infeasible_eta_range": [float(eta[infeasible].min()), float(eta[infeasible].max())]
                if infeasible else None,
            "minimum_original_eigenvalue_over_k": float(original.min()),
            "minimum_candidate_viscosity_fraction": float(fractions.min()),
            "minimum_candidate_eigenvalue_over_k": float(updated.min()),
            "maximum_candidate_relative_gradient_change": float((1 / fractions - 1).max()),
            "fixed_gradient_cap_recomputed_minimum_eigenvalue_over_k": float(naive.min()),
            "fixed_gradient_cap_recomputed_violating_samples": int((naive < -1e-10).sum()),
            "fraction_solved_integrated_k_affected": float(simpson(eta*k*violating, x=eta) / simpson(eta*k, x=eta)),
            "transport_equations_resolved": False, "profile_changed": False,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-solutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.edge_solutions / "assessment.json"
    report = json.loads(source.read_text())
    if report["holdout_accessed"]:
        raise ValueError("source-only input required")
    hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    cases = []
    for case in report["cases"]:
        accepted = [r for r in case["refinements"] if not r.get("numerically_rejected", False)]
        last = accepted[-1]
        if not last["solver"]["basic_numerical_checks_pass"]:
            raise ValueError("accepted numerical state required")
        state = args.edge_solutions / f'{case["label"]}_{last["delta"]}.npz'
        hashes[str(state)] = hashlib.sha256(state.read_bytes()).hexdigest()
        rows = [assess(state, last["solver"]["constants"], count) for count in [8193, 16385]]
        cases.append({"label": case["label"], "delta": last["delta"], "checks": rows})
        print(json.dumps({"label": case["label"], "fine": rows[-1]}), flush=True)
    (args.output / "assessment.json").write_text(json.dumps({
        "cases": cases, "input_sha256": hashes, "holdout_accessed": False,
        "closure_implemented": False, "physical_validation_pass": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
