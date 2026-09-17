"""Audit modeled momentum/pressure and covariance without accessing a holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from jet_stress import modeled_stress, thin_jet_pressure, velocity_gradient
from scipy.integrate import simpson
from scipy.interpolate import PchipInterpolator


def fields(path, constants, count, *, edge_graded=False):
    data = np.load(path)
    edge = float(data["edge"])
    z, y = data["z"], data["y"]
    cut = edge * z[-1]
    eta = np.linspace(0, edge, count)
    if edge_graded:
        eta = np.unique(
            np.r_[
                np.linspace(0, 0.99 * edge, count // 2),
                edge * (1 - np.geomspace(0.01, 1e-8, count // 2)),
                cut,
                edge,
            ]
        )
    value = PchipInterpolator(z, y, axis=1)(np.minimum(eta, cut) / edge)
    integral, f = edge**2 * value[0], value[1]
    k, e = np.exp(value[[2, 4]])
    n = constants["c_mu"] * k * k / e
    fp = -np.divide(integral * f, eta * n, out=np.zeros_like(eta), where=eta > 0)
    tail = eta > cut
    s0 = edge - cut
    ratio = np.maximum((edge - eta[tail]) / s0, 0)
    m = 1 / (2 * constants["sigma_k"] - constants["sigma_e"])
    p = constants["sigma_k"] * m
    q = constants["sigma_e"] * m
    fend, kend, eend = y[1, -1], np.exp(y[2, -1]), np.exp(y[4, -1])
    nend = constants["c_mu"] * kend * kend / eend
    f[tail] = fend * ratio**m
    fp[tail] = -m * fend / s0 * ratio ** (m - 1)
    k[tail], e[tail], n[tail] = kend * ratio**p, eend * ratio**q, nend * ratio
    integral[tail] = edge**2 * y[0, -1] + fend * (
        edge * s0 / (m + 1) * (1 - ratio ** (m + 1))
        - s0**2 / (m + 2) * (1 - ratio ** (m + 2))
    )
    return eta, integral, f, fp, k, n, cut


def assess(path, constants, count, *, edge_graded=False):
    eta, integral, f, fp, k, n, cut = fields(
        path, constants, count, edge_graded=edge_graded
    )
    i2 = float(simpson(eta * f * f, x=eta))
    integral_k = float(simpson(eta * k, x=eta))
    peak_k = float(k.max())
    modes = []
    for radial_only in [True, False]:
        gradient = velocity_gradient(
            eta, integral, f, fp, radial_shear_only=radial_only
        )
        stress = modeled_stress(k, n, gradient)
        pressure = thin_jet_pressure(eta, stress[:, 1, 1], stress[:, 2, 2])
        normal = stress[:, 0, 0] - (stress[:, 1, 1] + stress[:, 2, 2]) / 2
        correction = float(simpson(eta * normal, x=eta))
        direct = float(simpson(eta * (stress[:, 0, 0] + pressure), x=eta))
        eigenvalues = np.linalg.eigvalsh(stress)
        minimum = eigenvalues[:, 0]
        violating = minimum < -1e-10 * peak_k
        energetic = k > 1e-12 * peak_k
        computed = eta <= cut
        modes.append(
            {
                "gradient": "radial_shear_only"
                if radial_only
                else "complete_self_similar",
                "stress_pressure_integral": correction,
                "stress_pressure_over_mean_momentum": correction / i2,
                "pressure_identity_error": direct - correction,
                "pressure_identity_relative_to_mean_momentum": (direct - correction)
                / i2,
                "trace_error_over_peak_k": float(
                    np.max(abs(np.trace(stress, axis1=1, axis2=2) - 2 * k)) / peak_k
                ),
                "minimum_eigenvalue": float(minimum.min()),
                "minimum_eigenvalue_over_local_k": float(
                    np.min(minimum[energetic] / k[energetic])
                ),
                "minimum_eigenvalue_eta": float(eta[minimum.argmin()]),
                "negative_eigenvalue_detected": bool(np.any(violating)),
                "negative_eigenvalue_inside_solved_domain": bool(
                    np.any(violating & computed)
                ),
                "minimum_eigenvalue_inside_solved_domain": float(
                    minimum[computed].min()
                ),
                "minimum_eigenvalue_over_k_inside_solved_domain": float(
                    np.min(minimum[computed & energetic] / k[computed & energetic])
                ),
                "fraction_integrated_k_in_violating_solved_region": float(
                    simpson(eta * k * violating * computed, x=eta) / integral_k
                ),
                "violating_eta_range": [
                    float(eta[violating].min()),
                    float(eta[violating].max()),
                ]
                if np.any(violating)
                else None,
                "fraction_integrated_k_in_violating_region": float(
                    simpson(eta * k * violating, x=eta) / integral_k
                ),
            }
        )
    return {
        "radial_samples": eta.size,
        "edge_graded": edge_graded,
        "solved_domain_outer_eta": cut,
        "I2": i2,
        "integral_k": integral_k,
        "modes": modes,
        "mean_momentum_renormalized": False,
        "holdout_accessed": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solutions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    hashes = {}
    for pair in ["pope_printed", "standard"]:
        for c3 in [0.0, 0.79]:
            for normal in [0, 1]:
                label = f"{pair}_{c3}_{normal}"
                source = a.solutions / (label + ".json")
                document = json.loads(source.read_text())
                if (
                    not document["qualification"].get("numerical_pass", False)
                    or document["holdout_accessed"]
                ):
                    raise ValueError("numerically qualified source-only state required")
                constants = document["stages"][-1]["constants"]
                checks = []
                for stage, count in [(6, 1025), (6, 2049), (6, 4097), (3, 4097)]:
                    state = a.solutions / f"{label}_{stage}.npz"
                    checks.append({"stage": stage, **assess(state, constants, count)})
                    hashes[str(state)] = hashlib.sha256(state.read_bytes()).hexdigest()
                hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
                row = {
                    "label": label,
                    "constants": constants,
                    "normal_strain_production": bool(normal),
                    "checks": checks,
                }
                rows.append(row)
                print(json.dumps({"label": label, "fine": checks[2]}), flush=True)
    (a.output / "assessment.json").write_text(
        json.dumps(
            {
                "inputs_sha256": hashes,
                "results": rows,
                "holdout_accessed": False,
                "mean_momentum_renormalized": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
