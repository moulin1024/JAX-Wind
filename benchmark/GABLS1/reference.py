"""Readers and ensemble interpolation for official GABLS1 LES data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SET_COLUMNS = {
    "A": ("z", "u_mean", "v_mean", "theta_mean"),
    "B": (
        "z",
        "u_var_resolved",
        "v_var_resolved",
        "w_var_resolved",
        "w_skewness",
        "sgs_tke",
        "theta_var_resolved",
    ),
    "C": (
        "z",
        "uw_resolved",
        "uw_sgs",
        "vw_resolved",
        "vw_sgs",
        "wtheta_resolved",
        "wtheta_sgs",
        "utheta_resolved",
        "utheta_sgs",
        "vtheta_resolved",
        "vtheta_sgs",
    ),
    "D": (
        "z",
        "shear_production_resolved",
        "shear_production_sgs",
        "buoyancy_production",
        "transport_total",
        "dissipation",
        "storage",
    ),
    "E": (
        "time_s",
        "boundary_layer_height",
        "surface_heat_flux",
        "friction_velocity",
        "obukhov_length",
        "maximum_abs_w",
    ),
}


@dataclass(frozen=True)
class ReferenceSet:
    """One participant's formatted GABLS dataset."""

    participant: str
    description: str
    values: dict[str, np.ndarray]


def read_reference_set(path: Path) -> ReferenceSet:
    """Read the original Fortran-record formatted A--E data file."""
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 3:
        raise ValueError(f"reference file is incomplete: {path}")
    set_name = path.name.split("_")[1][0]
    if set_name not in SET_COLUMNS:
        raise ValueError(f"cannot determine GABLS set from {path.name}")
    columns = SET_COLUMNS[set_name]
    count = int(lines[1])
    parsed = []
    for token in " ".join(lines[2:]).split():
        if token.upper().startswith("NAN"):
            parsed.append(np.nan)
        else:
            parsed.append(float(token.replace("D", "E").replace("d", "e")))
    flat = np.asarray(parsed, dtype=float)
    expected = count * len(columns)
    if flat.size != expected:
        raise ValueError(
            f"expected {expected} values in {path}, found {flat.size}"
        )
    arrays = flat.reshape(len(columns), count)
    arrays[(arrays < -1.0e6) | np.isclose(arrays, -9999.0)] = np.nan
    participant = path.parent.name
    return ReferenceSet(
        participant,
        lines[0].strip(),
        dict(zip(columns, arrays, strict=True)),
    )


def load_period_sets(
    reference_dir: Path,
    set_name: str,
    period: int = 9,
) -> list[ReferenceSet]:
    """Load every participant available for a profile set and hour."""
    if set_name not in "ABCD":
        raise ValueError("profile set must be one of A, B, C, or D")
    pattern = f"*/*_{set_name}{period}_*.dat"
    paths = sorted(reference_dir.glob(pattern))
    return [read_reference_set(path) for path in paths]


def load_time_series(reference_dir: Path) -> list[ReferenceSet]:
    """Load set E time series for every available participant."""
    return [
        read_reference_set(path)
        for path in sorted(reference_dir.glob("*/*_E_*.dat"))
    ]


def ensemble_on_grid(
    datasets: list[ReferenceSet],
    coordinate: str,
    target: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Interpolate participants and return pointwise ensemble statistics."""
    if not datasets:
        return {}
    variables = tuple(
        name for name in datasets[0].values if name != coordinate
    )
    result: dict[str, dict[str, np.ndarray]] = {}
    for variable in variables:
        interpolated = []
        for dataset in datasets:
            x = np.asarray(dataset.values[coordinate], dtype=float)
            y = np.asarray(dataset.values[variable], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            if x.size < 2:
                continue
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            values = np.interp(target, x, y)
            values[(target < x[0]) | (target > x[-1])] = np.nan
            interpolated.append(values)
        if not interpolated:
            continue
        stack = np.stack(interpolated)
        result[variable] = {
            "mean": np.nanmean(stack, axis=0),
            "minimum": np.nanmin(stack, axis=0),
            "maximum": np.nanmax(stack, axis=0),
            "standard_deviation": np.nanstd(stack, axis=0),
            "count": np.sum(np.isfinite(stack), axis=0),
        }
    return result


__all__ = [
    "ReferenceSet",
    "SET_COLUMNS",
    "ensemble_on_grid",
    "load_period_sets",
    "load_time_series",
    "read_reference_set",
]
