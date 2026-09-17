"""Necessary axial transit-support tests; no effective-volume calibration."""

import numpy as np


def axial_support_ratios(
    axial_velocity,
    transit,
    diameter,
    *,
    axial_extent,
    velocity_allowance=0.0,
    diameter_allowance=0.0,
    finite_size=False,
):
    """Ratio >1 contradicts the specified hard-support transit scenario.

    A particle moving at constant axial speed u through a volume with axial
    full extent L has |u|*TT <= L, regardless of its transverse velocity or
    impact parameter. Finite sphere overlap expands the necessary bound to
    L+D. Optical thresholds and transit-time definitions require calibration.
    """
    u, t, d = np.broadcast_arrays(
        np.asarray(axial_velocity, float),
        np.asarray(transit, float),
        np.asarray(diameter, float),
    )
    if (
        not np.all(np.isfinite(u))
        or not np.all(np.isfinite(t))
        or not np.all(np.isfinite(d))
        or np.any(t <= 0)
        or np.any(d <= 0)
        or not np.isfinite(axial_extent)
        or axial_extent <= 0
        or not np.isfinite(velocity_allowance)
        or velocity_allowance < 0
        or not np.isfinite(diameter_allowance)
        or diameter_allowance < 0
    ):
        raise ValueError("invalid support inputs")
    displacement = np.maximum(np.abs(u) - velocity_allowance, 0) * t
    support = axial_extent + (d + diameter_allowance if finite_size else 0)
    return displacement / support


def scenario_ratios(samples, protocol):
    a = np.asarray(samples)
    common = {"axial_extent": protocol["nominal_axial_extent_m"]}
    allowance = protocol["generous_axial_velocity_allowance_m_s"]
    return {
        "point_nominal": axial_support_ratios(a[:, 3], a[:, 2], a[:, 7], **common),
        "finite_size_nominal": axial_support_ratios(
            a[:, 3], a[:, 2], a[:, 7], **common, finite_size=True
        ),
        "point_velocity_allowance": axial_support_ratios(
            a[:, 3], a[:, 2], a[:, 7], **common, velocity_allowance=allowance
        ),
        "finite_size_velocity_diameter_allowance": axial_support_ratios(
            a[:, 3],
            a[:, 2],
            a[:, 7],
            **common,
            finite_size=True,
            velocity_allowance=allowance,
            diameter_allowance=protocol["reported_diameter_uncertainty_m"],
        ),
    }
