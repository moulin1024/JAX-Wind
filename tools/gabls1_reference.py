"""Read and interpolate the archived official GABLS1 participant records."""

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
    participant: str
    description: str
    values: dict[str, np.ndarray]


def read_reference_set(path: Path) -> ReferenceSet:
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
        parsed.append(
            np.nan
            if token.upper().startswith("NAN")
            else float(token.replace("D", "E").replace("d", "e"))
        )
    flat = np.asarray(parsed, dtype=float)
    expected = count * len(columns)
    if flat.size != expected:
        raise ValueError(f"expected {expected} values in {path}, found {flat.size}")
    arrays = flat.reshape(len(columns), count)
    arrays[(arrays < -1.0e6) | np.isclose(arrays, -9999.0)] = np.nan
    return ReferenceSet(
        path.parent.name,
        lines[0].strip(),
        dict(zip(columns, arrays, strict=True)),
    )


def load_period_sets(
    reference_dir: Path,
    set_name: str,
    period: int = 9,
) -> list[ReferenceSet]:
    if set_name not in "ABCD":
        raise ValueError("profile set must be A, B, C, or D")
    return [
        read_reference_set(path)
        for path in sorted(reference_dir.glob(f"*/*_{set_name}{period}_*.dat"))
    ]


def interpolate_values(
    dataset: ReferenceSet,
    coordinate: str,
    variable: str,
    target: np.ndarray,
) -> np.ndarray:
    x = np.asarray(dataset.values[coordinate], dtype=float)
    y = np.asarray(dataset.values[variable], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    result = np.full_like(np.asarray(target, dtype=float), np.nan)
    if x.size < 2:
        return result
    order = np.argsort(x)
    x, y = x[order], y[order]
    inside = (target >= x[0]) & (target <= x[-1])
    result[inside] = np.interp(target[inside], x, y)
    return result


def ensemble_statistics(stack: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(all="ignore"):
        return {
            "mean": np.nanmean(stack, axis=0),
            "minimum": np.nanmin(stack, axis=0),
            "maximum": np.nanmax(stack, axis=0),
            "standard_deviation": np.nanstd(stack, axis=0),
            "count": np.sum(np.isfinite(stack), axis=0),
        }


def ensemble_on_grid(
    datasets: list[ReferenceSet],
    target: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    if not datasets:
        raise ValueError("no GABLS1 participant records were found")
    variables = tuple(name for name in datasets[0].values if name != "z")
    result = {}
    for variable in variables:
        stack = np.stack(
            [interpolate_values(data, "z", variable, target) for data in datasets]
        )
        if np.any(np.isfinite(stack)):
            result[variable] = ensemble_statistics(stack)
    return result


__all__ = [
    "ReferenceSet",
    "ensemble_on_grid",
    "ensemble_statistics",
    "interpolate_values",
    "load_period_sets",
    "read_reference_set",
]
