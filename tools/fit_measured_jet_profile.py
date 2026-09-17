"""Fit only the separately attributed measured axial-profile overlay."""

import argparse
import hashlib
import json
import re
from io import StringIO
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "cases/SprayClosureValidation/measured_profile_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    source = protocol["calibration"]
    fit = protocol["fit"]
    path = args.data / source["file"]
    if sha(path) != source["file_sha256"]:
        raise ValueError("Source hash mismatch")
    blocks = [s for s in re.split(r"\n\s*\n", path.read_text()) if s.strip()]
    if len(blocks) != source["expected_blocks"]:
        raise ValueError("Unexpected source layout")
    curves = [np.loadtxt(StringIO(blocks[i])) for i in source["axial_overlay_blocks"]]
    if any(c.shape != (source["points_per_block"], 2) for c in curves):
        raise ValueError("Unexpected axial overlay dimensions")
    if not np.array_equal(*curves):
        raise ValueError("Expected exact duplicate export")
    data = curves[0]
    if not np.isfinite(data).all() or not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError("Invalid/nonmonotone curve")
    eta, y = data.T
    if not 0 <= eta[0] < 1e-4 or eta[-1] < 0.249 or not 0.95 < y[0] < 1.05:
        raise ValueError("Unexpected support or normalization")
    if not np.all(np.diff(y) < 0):
        raise ValueError("Expected monotone axial profile for half-width audit")
    half = float(np.interp(0.5, y[::-1], eta[::-1]))
    if (
        abs(half / source["reported_half_width"] - 1)
        > source["source_half_width_relative_audit_tolerance"]
    ):
        raise ValueError("Overlay half-width disagrees with primary definition")
    # Endpoint half-interval quadrature; source values themselves are unchanged.
    widths = np.r_[np.diff(eta)[0] / 2, (eta[2:] - eta[:-2]) / 2, np.diff(eta)[-1] / 2]
    weights = widths / widths.sum()
    residual_weight = np.sqrt(weights)
    single = least_squares(
        lambda p: residual_weight * (np.exp(-np.exp(p[0]) * eta**2) - y),
        [np.log(fit["single_start"])],
        bounds=tuple([np.log(v)] for v in fit["single_bounds"]),
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
        max_nfev=10000,
    )
    if not single.success:
        raise RuntimeError(single.message)
    mixture = lambda p: (
        p[0] * np.exp(-np.exp(p[1] + p[2]) * eta**2)
        + (1 - p[0]) * np.exp(-np.exp(p[1]) * eta**2)
    )
    starts = []
    for a, c, ratio in fit["starts"]:
        initial = [a, c, ratio]
        result = least_squares(
            lambda p: residual_weight * (mixture(p) - y),
            [a, np.log(c), np.log(ratio)],
            bounds=(
                [
                    fit["bounds"]["a"][0],
                    np.log(fit["bounds"]["c2"][0]),
                    np.log(fit["bounds"]["ratio"][0]),
                ],
                [
                    fit["bounds"]["a"][1],
                    np.log(fit["bounds"]["c2"][1]),
                    np.log(fit["bounds"]["ratio"][1]),
                ],
            ),
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=10000,
        )
        if not result.success:
            raise RuntimeError(result.message)
        a, lc, lr = result.x
        starts.append(
            {
                "start": initial,
                "weights": [float(a), float(1 - a)],
                "coefficients": [float(np.exp(lc + lr)), float(np.exp(lc))],
                "cost": float(result.cost),
                "active_mask": result.active_mask.tolist(),
                "optimality": float(result.optimality),
                "nfev": result.nfev,
            }
        )
    best = min(starts, key=lambda r: r["cost"])
    families = {
        "published_gaussian": {
            "weights": [1.0],
            "coefficients": [source["published_gaussian_coefficient"]],
        },
        "single_gaussian": {
            "weights": [1.0],
            "coefficients": [float(np.exp(single.x[0]))],
        },
        "two_gaussian_mixture": {k: best[k] for k in ("weights", "coefficients")},
    }
    for family in families.values():
        predicted = sum(
            w * np.exp(-c * eta**2)
            for w, c in zip(family["weights"], family["coefficients"])
        )
        family["source_weighted_rms_over_Uc"] = float(
            np.sqrt(np.sum(weights * (predicted - y) ** 2))
        )
        family["source_max_absolute_error_over_Uc"] = float(
            np.max(np.abs(predicted - y))
        )
    report = {
        "protocol": protocol,
        "source_fit_only": True,
        "holdout_accessed": False,
        "used_block": 26,
        "excluded_duplicate_block": 27,
        "points": len(eta),
        "source_half_width": half,
        "source_support": [float(eta[0]), float(eta[-1])],
        "all_mixture_starts": starts,
        "families": families,
        "data_sha256": sha(path),
        "sha256": {
            str(p.relative_to(root)): sha(p) for p in (protocol_path, Path(__file__))
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(
        args.output / "measured_profile.csv",
        np.column_stack((eta, y, weights)),
        delimiter=",",
        header="eta,U_over_Uc,quadrature_weight",
        comments="",
    )
    print(json.dumps(families, indent=2))


if __name__ == "__main__":
    main()
