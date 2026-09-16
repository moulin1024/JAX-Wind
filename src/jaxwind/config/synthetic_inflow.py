"""Measured-profile Mann inflow settings for direct open atmospheric runs."""

from __future__ import annotations

import math


def reference_profile(path):
    import numpy as np

    table = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    required = {"height_mm", "wind_speed_m_s", "turbulence_intensity_percent"}
    if not required <= set(table.dtype.names or ()) or len(table) < 2:
        raise ValueError(
            "reference_profile needs height_mm, wind_speed_m_s and turbulence_intensity_percent"
        )
    z, u, ti = (
        table[name]
        for name in ("height_mm", "wind_speed_m_s", "turbulence_intensity_percent")
    )
    if (
        not all(np.isfinite(values).all() for values in (z, u, ti))
        or not (np.diff(z) > 0).all()
        or (z < 0).any()
        or (u <= 0).any()
        or (ti < 0).any()
    ):
        raise ValueError("invalid measured inflow profile")
    return z / 1000, u, ti / 100


def validate_mann_inflow(document):
    table = document["physics"]["inflow"]
    if isinstance(table, dict) and table.get("model") == "uniform":
        if set(table) != {"model", "speed_m_s", "lateral_boundary"}:
            raise ValueError("uniform inflow requires model, speed_m_s, lateral_boundary")
        speed = table["speed_m_s"]
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
            raise ValueError("uniform inflow speed_m_s must be finite and positive")
        if table["lateral_boundary"] != "outflow":
            raise ValueError("uniform inflow currently requires lateral_boundary = outflow")
        if document["numerics"]["pressure_backend"] != "gmg" or document["numerics"].get("time_integration") not in {"rk3", "fast-rk3"}:
            raise ValueError("uniform farm inflow requires GMG and rk3 or fast-rk3")
        if "cfl" in document["time"]:
            raise ValueError("uniform farm inflow currently requires a fixed timestep")
        if "wind_farm" not in document["physics"] or "cooling" in document["physics"] or "surface_scalar" in document["physics"]:
            raise ValueError("uniform inflow supports neutral wind-farm main cases only")
        flow = document["physics"].get("flow", {})
        if any(flow.get("pressure_acceleration_m_s2", [0., 0.])) or any(flow.get("coriolis_s", [0., 0.])):
            raise ValueError("uniform inflow requires zero pressure forcing and Coriolis")
        if document["physics"].get("scalar", {}).get("buoyancy_acceleration_per_unit", 0.) != 0.:
            raise ValueError("uniform inflow requires neutral scalar physics")
        if document.get("initial_conditions", {}).get("checkpoint"):
            raise ValueError("uniform inflow starts directly from uniform flow; use resume for continuation")
        return
    required = {
        "model",
        "reference_profile",
        "box_cells",
        "box_lengths_m",
        "length_scale_m",
        "gamma",
        "seed",
        "advection_height_m",
    }
    if not isinstance(table, dict) or table.keys() != required:
        raise ValueError(
            "physics.inflow requires exactly: " + ", ".join(sorted(required))
        )
    if table["model"] != "mann":
        raise ValueError("physics.inflow.model must be mann")
    if (
        not isinstance(table["reference_profile"], str)
        or not table["reference_profile"]
    ):
        raise ValueError("reference_profile must be a path")
    if "cfl" in document["time"]:
        raise ValueError("Mann inflow currently requires fixed timesteps")
    if document["numerics"]["pressure_backend"] != "gmg":
        raise ValueError("direct Mann inflow requires GMG pressure")
    if document["numerics"].get("time_integration") not in {"ab2", "fast-rk3"}:
        raise ValueError("direct Mann inflow requires ab2 or fast-rk3")
    for name in ("box_cells", "box_lengths_m"):
        values = table[name]
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"{name} requires three values")
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
            if name == "box_cells" and (type(value) is not int or value < 3):
                raise ValueError("box_cells must be integers >= 3")
    for name in ("length_scale_m", "gamma", "advection_height_m"):
        value = table[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (name != "gamma" and value == 0)
        ):
            raise ValueError(f"{name} is outside its valid range")
    if type(table["seed"]) is not int or table["seed"] < 0:
        raise ValueError("inflow.seed must be a nonnegative integer")
    if any(
        a < b
        for a, b in zip(table["box_lengths_m"][1:], document["mesh"]["lengths_m"][1:])
    ):
        raise ValueError("Mann box must cover the inlet")
    if document.get("initial_conditions", {}).get("checkpoint"):
        raise ValueError(
            "direct Mann inflow starts from the measured mean; use resume for continuation"
        )
    if "cooling" in document["physics"] or "surface_scalar" in document["physics"]:
        raise ValueError(
            "direct Mann inflow currently supports neutral flow without cooling"
        )
    scalar = document["physics"].get("scalar", {})
    if scalar.get("buoyancy_acceleration_per_unit", 0) != 0:
        raise ValueError("Mann inflow requires neutral flow")
