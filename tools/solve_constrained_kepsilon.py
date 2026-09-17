"""Source-only solve with consistent constrained timescale and new edge powers."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from constrained_kepsilon import fields, rhs
from jet_stress import modeled_stress, velocity_gradient
from kepsilon_similarity import PopeConstants
from scipy.integrate import quad, solve_bvp
from scipy.integrate._bvp import estimate_rms_residuals
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


def physical(z, y, edge):
    k, e = np.exp(np.clip(y[[2, 4]], -40, 10))
    state = np.array([np.maximum(edge**2*y[0], 0), np.maximum(y[1], 1e-20), k, y[3], e, y[5]])
    state[np.ix_([0, 3, 5], z == 0)] = 0
    return state


def solve(seed, constants, delta, tol, nodes):
    exponent = 1/constants.sigma_k
    z = np.unique(np.r_[np.linspace(0, 0.99, nodes//2), 1-np.geomspace(0.01, delta, nodes//2)])
    y = seed.sol(np.minimum(z, seed.x[-1]))
    if z[-1] > seed.x[-1]:
        tail = z > seed.x[-1]
        ratio = (1-z[tail])/(1-seed.x[-1])
        last = seed.sol(np.array([seed.x[-1]]))[:, 0]
        edge = np.exp(seed.p[0])
        length = edge*(1-seed.x[-1])
        y[0, tail] = last[0]+last[1]*(edge*length/(exponent+1)*(1-ratio**(exponent+1))
            -length**2/(exponent+2)*(1-ratio**(exponent+2)))/edge**2
        y[1, tail] = last[1]*ratio**exponent
        y[2, tail] = last[2]+np.log(ratio)
        y[4, tail] = last[4]+constants.sigma_e/constants.sigma_k*np.log(ratio)
        y[3, tail] = -edge**2*y[0, tail]*np.exp(y[2, tail])
        y[5, tail] = -edge**2*y[0, tail]*np.exp(y[4, tail])

    def spatial_equation(z, y, parameter):
        edge = np.exp(np.clip(parameter[0], -5, 3))
        state = physical(z, y, edge)
        derivative = rhs(edge*z, state, constants, trial=True)
        return np.array([derivative[0]/edge, edge*derivative[1],
                         edge*derivative[2]/state[2], edge*derivative[3],
                         edge*derivative[4]/state[4], edge*derivative[5]])

    def boundary(a, b, parameter):
        edge = np.exp(np.clip(parameter[0], -5, 3))
        state = physical(np.array([1-delta]), b[:, None], edge)
        ii, _, k, qk, e, qe = state[:, 0]
        ii = max(ii, 1e-20)
        n = fields(np.array([edge*(1-delta)]), state, constants, trial=True)["viscosity"][0]
        return np.array([a[0], a[1]-1, a[3], a[5],
            n/(ii*delta/((1-delta)*exponent))-1, qk/(ii*k)+1, qe/(ii*e)+1])

    def equation(t, y, parameter):
        z = -np.expm1(-t)
        return np.exp(-t)*spatial_equation(z, y, parameter)

    result = solve_bvp(equation, boundary, -np.log1p(-z), y, p=seed.p,
                       tol=tol/10, bc_tol=1e-10, max_nodes=30000)
    transformed_residual = float(np.max(result.rms_residuals))
    transformed_sol = result.sol

    def spatial_sol(z, nu=0):
        t = -np.log1p(-np.asarray(z))
        if nu == 0:
            return transformed_sol(t)
        if nu == 1:
            return transformed_sol(t, 1)/(1-np.asarray(z))
        raise ValueError("only value and first derivative are supported")

    result.sol = spatial_sol
    result.x = -np.expm1(-result.x)
    h = np.diff(result.x)
    middle = result.x[:-1]+h/2
    f_middle = spatial_equation(middle, result.sol(middle), result.p)
    r_middle = result.sol(middle, 1)-f_middle
    result.rms_residuals = estimate_rms_residuals(
        spatial_equation, result.sol, result.x, h, result.p, r_middle, f_middle)

    bound = float(np.max(abs(boundary(result.y[:, 0], result.y[:, -1], result.p))))
    return assess_solution(result, constants, delta, tol, nodes, bound,
                           "t=-log(1-z)", transformed_residual)


def assess_solution(result, constants, delta, tol, nodes, bound, coordinate, transformed_residual):
    exponent = 1/constants.sigma_k
    edge = float(np.exp(result.p[0]))
    zz = np.unique(np.r_[np.linspace(0, 0.99, 5001), 1-np.geomspace(0.01, delta, 5001), result.x])
    dense = result.sol(zz)
    state = physical(zz, dense, edge)
    value = fields(edge*zz, state, constants, trial=True)
    gradient = velocity_gradient(edge*zz, state[0], state[1], value["fp"], radial_shear_only=True)
    stress = modeled_stress(state[2], value["viscosity"], gradient)
    minimum = float(np.min(np.linalg.eigvalsh(stress)[:, 0]/state[2]))
    safeguards = bool(-5 < result.p[0] < 3 and np.all(dense[[2, 4]] > -40)
        and np.all(dense[[2, 4]] < 10) and np.all(dense[1] > 1e-20)
        and np.min(dense[0]) >= -1e-15 and np.all(value["feasible"]))
    basic = bool(result.status == 0 and safeguards and bound <= 1e-9
        and np.max(result.rms_residuals) <= 1e-6 and minimum >= -1e-10
        and np.max(np.diff(dense[1])) <= 1e-12 and edge*(1-delta) >= 0.2)
    row = {"constants": asdict(constants), "delta": delta, "tolerance": tol,
        "initial_nodes": nodes, "final_nodes": result.x.size, "status": result.status,
        "coordinate": coordinate, "transformed_tolerance": tol/10,
        "transformed_max_residual": transformed_residual,
        "message": result.message, "max_residual": float(np.max(result.rms_residuals)),
        "boundary_residual": bound, "safeguards_inactive": safeguards,
        "basic_numerical_pass": basic, "minimum_eigenvalue_over_K": minimum,
        "edge": edge, "edge_viscosity_fraction": float(value["fraction"][-1]),
        "edge_constraint_active": bool(value["fraction"][-1] < 1-1e-8),
        "minimum_viscosity_fraction": float(value["fraction"].min())}
    cut = edge*(1-delta)

    def profile(eta):
        return np.where(np.asarray(eta) <= cut, result.sol(np.minimum(eta, cut)/edge)[1],
            result.y[1, -1]*np.maximum((edge-np.asarray(eta))/(edge*delta), 0)**exponent)

    eta = np.linspace(0, 0.2, 201)
    if basic:
        i1 = sum(quad(lambda t: t*float(profile(t)), lo, hi, epsabs=1e-12)[0]
                 for lo, hi in [(0, cut), (cut, edge)])
        i2 = sum(quad(lambda t: t*float(profile(t))**2, lo, hi, epsabs=1e-12)[0]
                 for lo, hi in [(0, cut), (cut, edge)])
        width = brentq(lambda t: float(profile(t))-0.5, 0, cut, xtol=1e-13)
        row.update(I1=i1, I2=i2, half_width=width, decay_coefficient=1/np.sqrt(8*i2),
            alpha=2*i1/np.sqrt(8*i2), source_width_relative_error=abs(width/(0.086 if constants.c3 else 0.125)-1))
    return result, row, eta, profile(eta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-solutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-transition", action="store_true")
    parser.add_argument("--constrained-seeds", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_cases, hashes = [], {}
    backend = solve
    if args.split_transition:
        from constrained_kepsilon_piecewise import solve as backend
    for c3 in [0.0, 0.79]:
        constants = PopeConstants(c1=1.44, c2=1.92, c3=c3)
        source = args.edge_solutions/f"standard_{c3}_1_0.0001.npz"
        if args.constrained_seeds:
            manifest = args.constrained_seeds/"assessment.json"
            prior = json.loads(manifest.read_text())
            selected = next(case for case in prior["cases"] if case["c3"] == c3)
            if prior["holdout_accessed"] or not selected["stages"][5]["basic_numerical_pass"]:
                raise ValueError("accepted source-only constrained initializer required")
            source = args.constrained_seeds/f"constrained_{c3}_5.npz"
            hashes[str(manifest)] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        data = np.load(source)
        seed = SimpleNamespace(x=data["z"], p=np.array([np.log(float(data["edge"]))]),
                               sol=PchipInterpolator(data["z"], data["y"], axis=1))
        stages = [(d, 1e-6, 400) for d in [0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025]]
        stages += [(0.000025, 1e-8, 400), (0.000025, 1e-8, 800)]
        comparisons = [("edge", 4, 5), ("tolerance", 5, 6), ("mesh", 6, 7)]
        if args.split_transition:
            stages = [(d, 1e-6, 400) for d in [0.000025, 0.0000125, 0.00000625, 0.000003125]]
            stages += [(0.000003125, 1e-8, 400), (0.000003125, 1e-8, 800)]
            comparisons = [("edge", 2, 3), ("tolerance", 3, 4), ("mesh", 4, 5)]
        rows, profiles = [], []
        for index, (delta, tol, nodes) in enumerate(stages):
            seed, row, eta, f = backend(seed, constants, delta, tol, nodes)
            rows.append(row)
            profiles.append(f)
            np.savez(args.output/f"constrained_{c3}_{index}.npz", z=seed.x, y=seed.y,
                     edge=np.exp(seed.p[0]), eta=eta, F=f)
            print(json.dumps({"c3": c3, **row}), flush=True)
            if not row["basic_numerical_pass"]:
                break
        qualification = {"complete": len(rows) == len(stages) and all(r["basic_numerical_pass"] for r in rows)}
        if qualification["complete"]:
            checks = []
            for name, before, after in comparisons:
                change = max(abs(rows[after][key]/rows[before][key]-1) for key in ["I1", "I2", "half_width"])
                shape = float(np.max(abs(profiles[after]-profiles[before])))
                checks.append({"name": name, "relative_metric_change": change, "profile_change": shape,
                               "pass": change <= 0.002 and shape <= 1e-4})
            qualification.update(checks=checks, numerical_pass=all(c["pass"] for c in checks)
                and all(r["edge_constraint_active"] for r in rows[-4:]),
                source_width_pass=rows[-1]["source_width_relative_error"] <= 0.02)
        all_cases.append({"c3": c3, "stages": rows, "qualification": qualification})
        (args.output/"assessment.json").write_text(json.dumps({"cases": all_cases, "input_sha256": hashes,
            "holdout_accessed": False, "production_enabled": False}, indent=2)+"\n")
    if not all(c["qualification"].get("numerical_pass", False) for c in all_cases):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
