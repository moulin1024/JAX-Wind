"""Conditional axial-chord calibration; no correction of raw PDA measurements.

For a cylinder of radius R with uniform transverse impact y in [-R,R],
ell=2*sqrt(R**2-y**2). Thus E(ell**2)=8*R**2/3. A Gaussian threshold
model motivates R(D)**2=a*log(D_um)+b; this specialization is derived here,
not a reproduction of an unavailable published calibration implementation.
"""

import numpy as np


def _vectors(diameter_m, axial_velocity, transit_s):
    d, u, t = (
        np.asarray(q, dtype=float) for q in (diameter_m, axial_velocity, transit_s)
    )
    if (
        d.ndim != 1
        or not d.size
        or u.shape != d.shape
        or t.shape != d.shape
        or not all(np.all(np.isfinite(q)) for q in (d, u, t))
        or np.any(d <= 0)
        or np.any(t <= 0)
    ):
        raise ValueError("expected finite vectors, positive diameter and transit time")
    return d * 1e6, np.abs(u) * t * 1e6


def fit_radius_squared(diameter_m, axial_velocity, transit_s):
    """Unconstrained moment fit; return coefficients, never hide a bad slope."""
    d, length = _vectors(diameter_m, axial_velocity, transit_s)
    x, target = np.log(d), 3 / 8 * length**2
    delta = x - np.mean(x)
    denominator = np.sum(delta**2)
    if denominator <= np.finfo(float).eps * len(x):
        raise ValueError("diameter variation is required for radius calibration")
    a = np.sum(delta * (target - np.mean(target))) / denominator
    b = np.mean(target) - a * np.mean(x)
    return np.array([a, b])


def radius_squared(coefficients, diameter_m):
    return coefficients[0] * np.log(np.asarray(diameter_m) * 1e6) + coefficients[1]


def admissible(coefficients, diameter_range_m):
    q = np.asarray(coefficients)
    bounds = np.asarray(diameter_range_m)
    if q.shape != (2,) or bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise ValueError("expected two coefficients and finite diameter bounds")
    if bounds[0] <= 0 or bounds[1] <= bounds[0]:
        raise ValueError("invalid diameter range")
    return bool(
        np.all(np.isfinite(q)) and q[0] > 0 and np.all(radius_squared(q, bounds) > 0)
    )


def chord_cdf(normalized_chord):
    """CDF of ell/R for uniform one-dimensional impact parameter."""
    z = np.asarray(normalized_chord)
    interior = np.clip(z / 2, 0, 1)
    return np.where(z <= 0, 0, 1 - np.sqrt(1 - interior**2))


def assess_chords(coefficients, diameter_m, axial_velocity, transit_s):
    d, length = _vectors(diameter_m, axial_velocity, transit_s)
    r2 = radius_squared(coefficients, d * 1e-6)
    if not np.all(np.isfinite(r2)) or np.any(r2 <= 0):
        raise ValueError("nonpositive detection radius; do not clip to fit data")
    normalized = length / np.sqrt(r2)
    cdf = chord_cdf(np.sort(normalized))
    n = len(cdf)
    distance = max(
        np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n)
    )
    return {
        "events": n,
        "support_exceedances": int(np.sum(normalized > 2)),
        "support_exceedance_fraction": float(np.mean(normalized > 2)),
        "cdf_distance": float(distance),
        "normalized_mean": float(np.mean(normalized)),
        "normalized_second_moment": float(np.mean(normalized**2)),
        "relative_first_moment_error": float(np.mean(normalized) / (np.pi / 2) - 1),
        "relative_second_moment_error": float(np.mean(normalized**2) / (8 / 3) - 1),
        "normalized_quantiles_50_95_99_100": np.quantile(
            normalized, [0.5, 0.95, 0.99, 1]
        ).tolist(),
    }


def temporal_training_mask(arrival_s, blocks):
    """Even equal-duration blocks train; odd blocks assess source stability."""
    t = np.asarray(arrival_s, dtype=float)
    if (
        t.ndim != 1
        or len(t) < 2
        or not np.all(np.isfinite(t))
        or np.any(np.diff(t) < 0)
        or t[-1] <= t[0]
        or not isinstance(blocks, int)
        or blocks < 2
        or blocks % 2
    ):
        raise ValueError("ordered nonconstant times and an even block count required")
    edges = np.linspace(t[0], t[-1], blocks + 1)[1:-1]
    # Equal mathematical boundaries can differ by a few ulps after unit/origin
    # changes. Assign those ties to the right block, without a physical-time
    # tolerance; reject windows whose block spacing cannot resolve that tie.
    tolerance = 8 * np.spacing(max(float(np.max(np.abs(t))), np.finfo(float).tiny))
    if (t[-1] - t[0]) / blocks <= 2 * tolerance:
        raise ValueError("time origin precision cannot resolve block boundaries")
    index = np.searchsorted(edges, t + tolerance, side="right")
    return index % 2 == 0


def fit_nonnegative_radius_squared(diameter_m, axial_velocity, transit_s, lower_m):
    """Exact two-parameter nonnegative least squares in log(D/lower) basis.

    Radius squared is a*log(D/lower)+c with a,c >= 0. The boundary c=0
    means the detection radius tends to zero at the declared lower diameter;
    no event is discarded or given a clipped radius to accommodate the fit.
    """
    d, length = _vectors(diameter_m, axial_velocity, transit_s)
    if not np.isfinite(lower_m) or lower_m <= 0 or np.any(d * 1e-6 < lower_m):
        raise ValueError("fit diameters must lie above a positive lower bound")
    x, target = np.log(d / (lower_m * 1e6)), 3 / 8 * length**2
    if np.sum(x**2) == 0:
        raise ValueError("diameter variation above lower bound required")
    unbounded = fit_radius_squared(diameter_m, axial_velocity, transit_s)
    a, c = unbounded[0], unbounded[1] + unbounded[0] * np.log(lower_m * 1e6)
    candidates = [(0.0, np.mean(target)), (np.sum(x * target) / np.sum(x**2), 0.0)]
    if a >= 0 and c >= 0:
        candidates.append((a, c))
    a, c = min(candidates, key=lambda q: np.sum((q[0] * x + q[1] - target) ** 2))
    return np.array([a, c - a * np.log(lower_m * 1e6)])
