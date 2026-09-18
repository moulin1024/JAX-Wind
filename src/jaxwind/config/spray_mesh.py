"""Optional physical-coordinate clustering for the inertial spray benchmark."""

import numpy as np

from jaxwind.domain import (
    AnalyticalGrid,
    IdentityMapping,
    SinhMapping,
    TanhMapping,
    UniformGrid,
)


def build_spray_grid(mesh, *, spray_model="inertial"):
    """Build a tensor-product mesh; mappings are not an original-paper mesh.

    Mapping locations are physical metres. Zero strengths retain the exact
    uniform-grid path. The Fluent-style parcel implementation still requires
    a uniform mesh and must not silently consume these coordinates.
    """
    cells, lengths = tuple(mesh["cells"]), tuple(mesh["lengths_m"])
    uniform = UniformGrid(*cells, *lengths)
    table = mesh.get("mapping")
    if table is None:
        return uniform
    if not isinstance(table, dict) or set(table) != {"types", "focus_m", "strength"}:
        raise ValueError("mesh.mapping requires exactly types, focus_m and strength")
    kinds = table["types"]
    if (
        not isinstance(kinds, list)
        or len(kinds) != 3
        or any(k not in ("uniform", "sinh", "tanh") for k in kinds)
    ):
        raise ValueError(
            "mesh.mapping.types must contain three uniform/sinh/tanh names"
        )
    focus, strength = (
        np.asarray(table[k], dtype=float) for k in ("focus_m", "strength")
    )
    if (
        focus.shape != (3,)
        or strength.shape != (3,)
        or not np.isfinite(focus).all()
        or not np.isfinite(strength).all()
    ):
        raise ValueError(
            "mapping focus_m and strength must contain three finite values"
        )
    if np.any(focus < 0) or np.any(focus > lengths) or np.any(strength < 0):
        raise ValueError(
            "mapping focus must lie in the domain and strength must be nonnegative"
        )
    mappings = []
    for kind, f, s, length in zip(kinds, focus, strength, lengths):
        if kind == "uniform":
            if s != 0:
                raise ValueError("uniform mapping strength must be zero")
            mappings.append(IdentityMapping())
        else:
            mappings.append(
                SinhMapping(f / length, s)
                if kind == "sinh"
                else TanhMapping(s, f / length)
            )
    if not np.any(strength):
        return uniform
    if spray_model != "inertial":
        raise ValueError(
            "stretched water-spray grids currently require spray_model=inertial"
        )
    return AnalyticalGrid(*cells, *lengths, *mappings)
