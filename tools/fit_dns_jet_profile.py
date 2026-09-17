#!/usr/bin/env python3
"""Fit only independent stationary DNS curves; never open holdout data."""

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
    protocol_path = root / "cases/SprayClosureValidation/dns_profile_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    spec, fit = protocol["calibration"], protocol["fit"]
    path = args.data / spec["file"]
    if sha(path) != spec["file_sha256"]:
        raise ValueError("Calibration source hash mismatch")
    raw = [b for b in re.split(r"\n\s*\n", path.read_text()) if b.strip()]
    if len(raw) != 55:
        raise ValueError("Unexpected source curve layout")
    eta = np.linspace(fit["eta_min"], fit["eta_max"], fit["points"])
    curves = []
    seen = {}
    duplicates = []
    used = []
    for index in spec["block_indices"]:
        data = np.loadtxt(StringIO(raw[index]))
        if data.shape != (spec["points_per_block"], 2) or not np.isfinite(data).all():
            raise ValueError(f"Invalid DNS block {index}")
        if (
            not np.all(np.diff(data[:, 0]) > 0)
            or data[0, 0] > eta[0]
            or data[-1, 0] < eta[-1]
        ):
            raise ValueError("Nonmonotone or incomplete radial support")
        if not 0.95 < data[0, 1] < 1.05:
            raise ValueError(
                "Expected normalized axial velocity, not radial/other field"
            )
        digest = hashlib.sha256(data.tobytes()).hexdigest()
        if digest in seen:
            duplicates.append({"block": index, "duplicate_of": seen[digest]})
            continue
        seen[digest] = index
        used.append(index)
        curves.append(np.interp(eta, data[:, 0], data[:, 1]))
    curves = np.asarray(curves)
    mean = curves.mean(axis=0)

    def mixture(p):
        a, log_c2, log_ratio = p
        c2, ratio = np.exp(log_c2), np.exp(log_ratio)
        return a * np.exp(-c2 * ratio * eta**2) + (1 - a) * np.exp(-c2 * eta**2)

    fits = []
    for a, c2, ratio in fit["starts"]:
        initial = [a, c2, ratio]
        result = least_squares(
            lambda p: mixture(p) - mean,
            [a, np.log(c2), np.log(ratio)],
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
        fits.append(
            {
                "start": initial,
                "parameters": result.x.tolist(),
                "weights": [float(a), float(1 - a)],
                "coefficients": [float(np.exp(lc + lr)), float(np.exp(lc))],
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "active_mask": result.active_mask.tolist(),
                "nfev": result.nfev,
            }
        )
    best = min(fits, key=lambda r: r["cost"])
    single = least_squares(
        lambda p: np.exp(-np.exp(p[0]) * eta**2) - mean,
        [np.log(75.0)],
        bounds=([np.log(1.0)], [np.log(100000.0)]),
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
    )
    if not single.success:
        raise RuntimeError(single.message)
    families = {
        "single_gaussian": {
            "weights": [1.0],
            "coefficients": [float(np.exp(single.x[0]))],
        },
        "two_gaussian_mixture": {
            "weights": best["weights"],
            "coefficients": best["coefficients"],
        },
    }
    for result in families.values():
        predicted = sum(
            w * np.exp(-c * eta**2)
            for w, c in zip(result["weights"], result["coefficients"])
        )
        result["pooled_rms_over_Uc"] = float(np.sqrt(np.mean((predicted - mean) ** 2)))
        result["curve_rms_over_Uc"] = np.sqrt(
            np.mean((curves - predicted) ** 2, axis=1)
        ).tolist()
    report = {
        "protocol": protocol,
        "used_blocks": used,
        "duplicate_blocks": duplicates,
        "families": families,
        "all_mixture_starts": fits,
        "source_spread_max_over_Uc": float(np.ptp(curves, axis=0).max()),
        "source_fit_only": True,
        "holdout_accessed": False,
        "sha256": {
            str(p.relative_to(root)): sha(p) for p in [protocol_path, Path(__file__)]
        },
        "data_sha256": sha(path),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(
        args.output / "dns_pooled_profile.csv",
        np.column_stack((eta, mean, curves.min(axis=0), curves.max(axis=0))),
        delimiter=",",
        header="eta,mean,min,max",
        comments="",
    )
    print(json.dumps(families, indent=2))


if __name__ == "__main__":
    main()
