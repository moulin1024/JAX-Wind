"""Distinguish a computed covariance violation from appended-tail artifacts."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from audit_jet_stress import assess
from kepsilon_similarity import PopeConstants
from scipy.interpolate import PchipInterpolator
from solve_kepsilon_profile import solve


def continuation_seed(seed, constants):
    """Extend the numerical initializer with the same leading edge powers.

    This avoids polynomial extrapolation of logarithmic fields beyond their
    previous solve interval. It changes the starting guess, not the equations.
    """
    edge = float(np.exp(seed.p[0]))
    last_z = seed.x[-1]
    last = seed.sol(np.array([last_z]))[:, 0]
    exponent = 1 / (2 * constants.sigma_k - constants.sigma_e)
    kpower, epower = constants.sigma_k * exponent, constants.sigma_e * exponent
    tail_length = edge * (1 - last_z)
    viscosity = constants.c_mu * np.exp(2 * last[2] - last[4])

    def value(z):
        result = seed.sol(np.minimum(z, last_z))
        tail = z > last_z
        ratio = (1 - z[tail]) / (1 - last_z)
        result[0, tail] = (
            last[0]
            + last[1]
            * (
                edge * tail_length / (exponent + 1) * (1 - ratio ** (exponent + 1))
                - tail_length**2 / (exponent + 2) * (1 - ratio ** (exponent + 2))
            )
            / edge**2
        )
        result[1, tail] = last[1] * ratio**exponent
        result[2, tail] = last[2] + kpower * np.log(ratio)
        result[4, tail] = last[4] + epower * np.log(ratio)
        factor = -edge * z[tail] * viscosity * exponent / tail_length
        result[3, tail] = factor * np.exp(result[2, tail])
        result[5, tail] = factor * np.exp(result[4, tail])
        return result

    return SimpleNamespace(p=seed.p, sol=value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    hashes = {}
    for c3 in [0.0, 0.79]:
        for normal in [False, True]:
            constants = PopeConstants(c1=1.44, c2=1.92, c3=c3)
            label = f"standard_{c3}_{int(normal)}"
            initial = args.solutions / (label + "_6.npz")
            document = json.loads((args.solutions / (label + ".json")).read_text())
            if (
                not document["qualification"]["numerical_pass"]
                or document["holdout_accessed"]
            ):
                raise ValueError("numerically qualified source-only state required")
            data = np.load(initial)
            seed = SimpleNamespace(
                p=np.array([np.log(float(data["edge"]))]),
                x=data["z"],
                sol=PchipInterpolator(data["z"], data["y"], axis=1),
            )
            hashes[str(initial)] = hashlib.sha256(initial.read_bytes()).hexdigest()
            rows = []
            for delta in [0.0025, 0.001, 0.0005, 0.00025, 0.0001]:
                result, report, eta, f = solve(
                    constants,
                    normal,
                    delta,
                    1e-8,
                    800,
                    None,
                    continuation_seed(seed, constants),
                )
                if not report["basic_numerical_checks_pass"]:
                    rows.append(
                        {"delta": delta, "solver": report, "numerically_rejected": True}
                    )
                    print(
                        json.dumps(
                            {
                                "case": label,
                                "delta": delta,
                                "numerically_rejected": True,
                            }
                        ),
                        flush=True,
                    )
                    break
                seed = result
                state = args.output / f"{label}_{delta}.npz"
                np.savez(
                    state, z=seed.x, y=seed.y, edge=np.exp(seed.p[0]), eta=eta, F=f
                )
                fine = assess(state, asdict(constants), 8193, edge_graded=True)
                row = {"delta": delta, "solver": report, "stress": fine}
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "case": label,
                            "delta": delta,
                            "core_min_eigenvalues": [
                                mode["minimum_eigenvalue_inside_solved_domain"]
                                for mode in fine["modes"]
                            ],
                            "core_violation": [
                                mode["negative_eigenvalue_inside_solved_domain"]
                                for mode in fine["modes"]
                            ],
                        }
                    ),
                    flush=True,
                )
            if not rows[-1].get("numerically_rejected", False):
                rows[-1]["quadrature_check"] = assess(
                    state, asdict(constants), 16385, edge_graded=True
                )
            cases.append({"label": label, "refinements": rows})
        (args.output / "assessment.json").write_text(
            json.dumps(
                {
                    "cases": cases,
                    "seed_sha256": hashes,
                    "holdout_accessed": False,
                    "covariance_repaired": False,
                },
                indent=2,
            )
            + "\n"
        )
    if any(
        case["refinements"][-1].get("numerically_rejected", False) for case in cases
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
