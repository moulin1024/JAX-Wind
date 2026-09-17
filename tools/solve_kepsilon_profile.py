"""Source-only free-edge similarity solve initialized from a numerical march.

Published coefficient conventions remain distinct. Trial exponent safeguards
must be inactive in accepted solutions. No held-out experiment is read here.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from kepsilon_similarity import PopeConstants, similarity_rhs
from scipy.integrate import quad, solve_bvp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


def initial(march_directory, c3, delta, nodes):
    tag = f"source_{float(c3)}_1000_0.0001875_300_1e-07_80_1"
    state_path = march_directory / ("march_" + tag + ".npz")
    report_path = march_directory / ("report_" + tag + ".json")
    data = np.load(state_path)
    report = json.loads(report_path.read_text())
    if report["failed"] or not report["final_x"] == 300:
        raise ValueError("a completed source-only march is required")
    distance = float(data["x"]) - report["virtual_origin"]
    psi, u, k, e, r = (data[key] for key in ("psi", "u", "k", "e", "r"))
    uc = u[0] - psi[0] * (u[1] - u[0]) / (psi[1] - psi[0])
    eta, f, k, e = r / distance, u / uc, k / uc**2, e * distance / uc**3
    integral = psi / (uc * distance**2)
    n = 0.09 * k * k / e
    mask = (f > 0.002) & (f < 0.02)
    if np.count_nonzero(mask) < 3:
        mask = (f > 0.03) & (f < 0.1)
    if np.count_nonzero(mask) < 3:
        raise ValueError("insufficient numerical edge samples for initialization")
    edge = float(np.median(eta[mask] * (1 + n[mask] / (0.7 * integral[mask]))))
    radius = np.r_[0, eta]
    functions = [
        PchipInterpolator(radius, np.r_[axis, value])
        for axis, value in [(0, integral), (1, f), (k[0], k), (e[0], e)]
    ]
    z = np.linspace(0, 1 - delta, nodes)
    eta = edge * z
    ii, ff, kk, ee = (function(eta) for function in functions)
    nn = 0.09 * kk * kk / ee
    qk = eta * nn * functions[2].derivative()(eta)
    qe = eta * nn / 1.3 * functions[3].derivative()(eta)
    qk[0] = qe[0] = 0
    y = np.array([ii / edge**2, ff, np.log(kk), qk, np.log(ee), qe])
    if not np.all(np.isfinite(y)) or not np.isfinite(edge):
        raise ValueError("nonfinite numerical initializer")
    provenance = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (state_path, report_path)
    }
    return z, y, np.array([np.log(edge)]), provenance


def solve(constants, normal, delta, tol, nodes, march_directory, seed=None):
    if seed is None:
        z, y, parameter, provenance = initial(
            march_directory, constants.c3, delta, nodes
        )
    else:
        z = np.linspace(0, 1 - delta, nodes)
        y, parameter, provenance = seed.sol(z), seed.p, {}
    exponent = 1 / (2 * constants.sigma_k - constants.sigma_e)

    def rhs(z, y, parameter):
        edge = np.exp(np.clip(parameter[0], -5, 3))
        k, e = np.exp(np.clip(y[[2, 4]], -40, 10))
        physical = np.array([edge**2 * y[0], y[1], k, y[3], e, y[5]])
        # Removable-axis extension during Newton trials. Boundary equations
        # independently require the actual I,Qk,Qe axis values to be zero.
        physical[np.ix_([0, 3, 5], z == 0)] = 0
        derivative = similarity_rhs(edge * z, physical, constants, normal_strain=normal)
        return np.array(
            [
                derivative[0] / edge,
                edge * derivative[1],
                edge * derivative[2] / k,
                edge * derivative[3],
                edge * derivative[4] / e,
                edge * derivative[5],
            ]
        )

    def boundary(a, b, parameter):
        edge = np.exp(np.clip(parameter[0], -5, 3))
        k, e = np.exp(np.clip(b[[2, 4]], -40, 10))
        n = constants.c_mu * k * k / e
        return np.array(
            [
                a[0],
                a[1] - 1,
                a[3],
                a[5],
                n - edge**2 * b[0] * delta / ((1 - delta) * exponent),
                b[3] + edge**2 * b[0] * k,
                b[5] + edge**2 * b[0] * e,
            ]
        )

    result = solve_bvp(
        rhs, boundary, z, y, p=parameter, tol=tol, bc_tol=1e-10, max_nodes=20000
    )
    edge = float(np.exp(result.p[0]))
    cut = edge * (1 - delta)
    tail_amplitude = float(result.y[1, -1])

    def profile(eta):
        eta = np.asarray(eta)
        inside = result.sol(np.minimum(eta, cut) / edge)[1]
        tail = tail_amplitude * np.maximum((edge - eta) / (edge * delta), 0) ** exponent
        return np.where(eta <= cut, inside, tail)

    eta = np.linspace(0, 0.2, 201)
    dense = result.sol(np.linspace(0, 1 - delta, 10001))
    bound = float(np.max(abs(boundary(result.y[:, 0], result.y[:, -1], result.p))))
    safeguards_inactive = bool(
        -5 < result.p[0] < 3
        and np.all(dense[[2, 4]] > -40)
        and np.all(dense[[2, 4]] < 10)
    )
    basic = bool(
        result.status == 0
        and safeguards_inactive
        and np.all(np.isfinite(dense))
        and np.all(dense[1] > 0)
        and np.max(np.diff(dense[1])) <= 1e-12
        and np.min(dense[0]) >= -1e-12
        and bound <= 1e-9
        and cut >= eta[-1]
        and np.max(result.rms_residuals) <= 1e-6
    )
    report = {
        "constants": asdict(constants),
        "normal_strain": normal,
        "delta": delta,
        "tolerance": tol,
        "initial_nodes": nodes,
        "final_nodes": result.x.size,
        "status": result.status,
        "message": result.message,
        "maximum_collocation_residual": float(np.max(result.rms_residuals)),
        "boundary_residual": bound,
        "safeguards_inactive": safeguards_inactive,
        "basic_numerical_checks_pass": basic,
        "edge": edge,
        "uncomputed_tail_start": cut,
        "tail_profile_value": tail_amplitude,
        "tail_model": "leading free-edge power, checked by delta refinement",
        "holdout_accessed": False,
        "initializer_hashes": provenance,
    }
    if basic:
        i1 = sum(
            quad(lambda t: t * float(profile(t)), lo, hi, epsabs=1e-12)[0]
            for lo, hi in [(0, cut), (cut, edge)]
        )
        i2 = sum(
            quad(lambda t: t * float(profile(t)) ** 2, lo, hi, epsabs=1e-12)[0]
            for lo, hi in [(0, cut), (cut, edge)]
        )
        width = brentq(lambda t: float(profile(t)) - 0.5, 0, cut, xtol=1e-13)
        b = 1 / np.sqrt(8 * i2)
        target = 0.125 if constants.c3 == 0 else 0.086
        report.update(
            I1=i1,
            I2=i2,
            half_width=width,
            decay_coefficient=b,
            alpha=2 * b * i1,
            historical_source_rate=target,
            historical_rate_relative_error=abs(width / target - 1),
        )
    return result, report, eta, profile(eta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--march-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair", choices=["pope_printed", "standard"], required=True)
    parser.add_argument("--c3", type=float, choices=[0, 0.79], required=True)
    parser.add_argument("--normal-strain", action="store_true")
    args = parser.parse_args()
    constants = PopeConstants(
        c1=1.45 if args.pair == "pope_printed" else 1.44,
        c2=1.90 if args.pair == "pope_printed" else 1.92,
        c3=args.c3,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    label = f"{args.pair}_{args.c3}_{int(args.normal_strain)}"
    stages = [(d, 1e-6, 400) for d in [0.08, 0.04, 0.02, 0.01, 0.005]] + [
        (0.005, 1e-8, 400),
        (0.005, 1e-8, 800),
    ]
    seed = None
    rows, profiles = [], []
    for index, (delta, tol, nodes) in enumerate(stages):
        seed, report, eta, f = solve(
            constants, args.normal_strain, delta, tol, nodes, args.march_directory, seed
        )
        rows.append(report)
        profiles.append(f)
        np.savez(
            args.output / f"{label}_{index}.npz",
            eta=eta,
            F=f,
            z=seed.x,
            y=seed.y,
            edge=np.exp(seed.p[0]),
        )
        print(json.dumps(report), flush=True)
        if not report["basic_numerical_checks_pass"]:
            break
    qualification = {
        "complete": len(rows) == len(stages)
        and all(r["basic_numerical_checks_pass"] for r in rows)
    }
    if qualification["complete"]:
        checks = []
        for name, earlier, later in [
            ("edge", 3, 4),
            ("tolerance", 4, 5),
            ("initial_mesh", 5, 6),
        ]:
            metrics = ["half_width", "I1", "I2"]
            change = max(abs(rows[later][k] / rows[earlier][k] - 1) for k in metrics)
            shape = float(np.max(abs(profiles[later] - profiles[earlier])))
            checks.append(
                {
                    "check": name,
                    "maximum_relative_metric_change": change,
                    "maximum_profile_change_over_Uc": shape,
                    "pass": change <= 0.002 and shape <= 1e-4,
                }
            )
        qualification.update(
            refinements=checks, numerical_pass=all(c["pass"] for c in checks)
        )
    document = {
        "coefficient_pair": args.pair,
        "stages": rows,
        "qualification": qualification,
        "holdout_accessed": False,
        "physical_validation_pass": False,
    }
    (args.output / f"{label}.json").write_text(json.dumps(document, indent=2) + "\n")
    if not qualification.get("numerical_pass", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
