"""Independent implementation of Racz et al. (2022) Appendix A gas proxy.

Method DOI 10.1016/j.ijmultiphaseflow.2022.104260. Public author filtering_v2.m
was inspected to disambiguate scaled-MAD filtering and equal-diameter ties.
No author source code is included here. This estimates a two-component speed
magnitude from selected droplet events, NOT a measured gas field or SGS energy.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GasProxy:
    preprocessed: np.ndarray
    selected: np.ndarray
    history: tuple
    mean_speed: float
    last_cutoff_m: float | None


def scaled_mad_inliers(values):
    """MATLAB isoutlier default: three scaled median absolute deviations."""
    x = np.asarray(values)
    median = np.median(x)
    mad = np.median(np.abs(x - median)) / 0.6744897501960817
    return np.abs(x - median) <= 3 * mad


def estimate_gas_proxy(
    diameter,
    velocity,
    *,
    density,
    viscosity,
    length_scale,
    stokes_limit=0.1,
    minimum_diameter=0.5145e-6,
):
    """Return original-record masks and complete monotonically shrinking history.

    diameter is metres (n,), velocity is m/s (n,ncomponents). Filtering is for
    this estimator only: the full liquid population must remain intact. At each
    iteration retain Stk <= limit AND diameter <= the smallest violating size.
    This matches the public implementation's treatment of equal-size ties.
    Removed records never re-enter. Stopping requires no retained violation,
    recomputed using the final selected mean. Empty populations fail explicitly.
    """
    d = np.asarray(diameter, dtype=float)
    v = np.asarray(velocity, dtype=float)
    parameters = np.array([density, viscosity, length_scale, stokes_limit])
    if np.any(~np.isfinite(parameters)) or np.any(parameters <= 0):
        raise ValueError("Density, viscosity, length and Stokes limit must be positive")
    if not np.isfinite(minimum_diameter) or minimum_diameter < 0:
        raise ValueError("Invalid minimum diameter")
    if d.ndim != 1 or v.ndim != 2 or v.shape[0] != len(d) or v.shape[1] < 1:
        raise ValueError("Expected diameter (n,) and velocity (n,ncomponents)")
    valid = (
        np.isfinite(d)
        & (d > 0)
        & (d >= minimum_diameter)
        & np.all(np.isfinite(v), axis=1)
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("No valid samples for gas estimation")
    indices = indices[scaled_mad_inliers(d[indices])]
    preprocessed = np.zeros(len(d), dtype=bool)
    preprocessed[indices] = True
    speed = np.linalg.norm(v, axis=1)
    coefficient = density / (18 * viscosity * length_scale)
    history = []
    cutoff = None
    for _ in range(len(indices) + 1):
        if not len(indices):
            raise ValueError("Stokes filtering removed all samples")
        mean = float(np.mean(speed[indices]))
        stokes = d[indices] ** 2 * np.abs(mean - speed[indices]) * coefficient
        bad = stokes > stokes_limit
        history.append(
            {
                "count": len(indices),
                "mean_speed_m_s": mean,
                "maximum_stokes": float(np.max(stokes)),
            }
        )
        if not np.any(bad):
            selected = np.zeros(len(d), dtype=bool)
            selected[indices] = True
            return GasProxy(preprocessed, selected, tuple(history), mean, cutoff)
        cutoff = float(np.min(d[indices[bad]]))
        indices = indices[(~bad) & (d[indices] <= cutoff)]
    raise RuntimeError("Non-convergent gas proxy filter")
