"""Strict schema for a shared-rotor, independently controlled periodic farm."""
import math


def validate_wind_farm(document):
    physics = document["physics"]
    farm = physics["wind_farm"]
    if not isinstance(farm, dict) or set(farm) != {"controller", "layout"}:
        raise ValueError("wind_farm requires exactly controller and layout")
    if ("inflow" in physics and physics["inflow"].get("model") != "uniform") or "cooling" in physics:
        raise ValueError("wind_farm supports periodic or uniform-open direct runs, without cooling")
    if document.get("initial_conditions", {}).get("operation") not in (None, "simulation", "open-inflow"):
        raise ValueError("controlled wind_farm requires simulation or open-inflow operation")
    template = physics.get("turbine", {})
    if template.get("model") not in ("openfast-ad-bem", "hitsz-r9-ad-bem"):
        raise ValueError("wind_farm requires an AD-BEM physics.turbine template")
    if any(template.get(key, 1.) != 0. for key in ("nacelle_drag_coefficient", "tower_drag_coefficient")):
        raise ValueError("controlled wind_farm currently requires disabled nacelle/tower drag")
    flow = physics.get("flow", {})
    if any(flow.get("advection_frame_velocity_m_s", [0., 0.])):
        raise ValueError("controlled wind_farm requires a stationary advection frame")
    control = farm["controller"]
    if not isinstance(control, dict):
        raise ValueError("wind_farm.controller must be a table")
    lookup = control.get("model") == "wind-speed-lookup"
    if "inflow" in physics and not lookup:
        raise ValueError("open uniform farms currently require wind-speed-lookup control")
    common = {"model", "wind_filter_seconds", "probe_distance_diameters"}
    required = common | ({"wind_speed_m_s", "rpm", "power_w", "cut_in_m_s", "cut_out_m_s",
                          "rpm_source", "power_source"} if lookup else
                         {"target_tsr", "response_seconds", "minimum_rpm", "maximum_rpm",
                          "maximum_acceleration_rpm_s"})
    if set(control) != required:
        raise ValueError("wind_farm.controller requires: " + ", ".join(sorted(required)))
    if control["model"] not in ("ideal-tsr", "wind-speed-lookup"):
        raise ValueError("wind_farm.controller.model must be ideal-tsr or wind-speed-lookup")
    def number(value, name, zero=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
            raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    numeric = required - {"model", "wind_speed_m_s", "rpm", "power_w", "rpm_source", "power_source"}
    for key in numeric:
        number(control[key], key, zero=key == "minimum_rpm")
    if lookup:
        for key in ("wind_speed_m_s", "rpm", "power_w"):
            values = control[key]
            if not isinstance(values, list) or len(values) < 2:
                raise ValueError(f"{key} must contain at least two values")
            for value in values:
                number(value, key, zero=True)
        speeds = control["wind_speed_m_s"]
        if any(a >= b for a, b in zip(speeds, speeds[1:])):
            raise ValueError("wind_speed_m_s must be strictly increasing")
        if any(len(control[key]) != len(speeds) for key in ("rpm", "power_w")):
            raise ValueError("wind_speed_m_s, rpm, and power_w must have equal lengths")
        if not speeds[0] <= control["cut_in_m_s"] < control["cut_out_m_s"] <= speeds[-1]:
            raise ValueError("lookup table must cover the complete cut-in/cut-out interval")
        for key in ("rpm_source", "power_source"):
            if not isinstance(control[key], str) or not control[key].strip():
                raise ValueError(f"{key} must document the table provenance")
    elif control["minimum_rpm"] >= control["maximum_rpm"]:
        raise ValueError("minimum_rpm must be less than maximum_rpm")
    layout = farm["layout"]
    if not isinstance(layout, list) or not layout:
        raise ValueError("wind_farm.layout must be a nonempty array of tables")
    ids, positions = set(), set()
    lx, ly, lz = document["mesh"]["lengths_m"]
    for row in layout:
        if not isinstance(row, dict) or set(row) != {"id", "x_m", "y_m", "hub_height_m", "initial_rpm"}:
            raise ValueError("each wind_farm.layout row requires id, x_m, y_m, hub_height_m, initial_rpm")
        name = row["id"]
        if not isinstance(name, str) or not name or not all(c.isalnum() or c in "_-" for c in name) or name in ids:
            raise ValueError("turbine IDs must be unique nonempty alphanumeric/underscore/hyphen strings")
        ids.add(name)
        for key in ("x_m", "y_m", "hub_height_m", "initial_rpm"):
            number(row[key], key, zero=key == "initial_rpm")
        if not (row["x_m"] < lx and row["y_m"] < ly and row["hub_height_m"] < lz):
            raise ValueError("turbine location is outside the domain")
        pos = (row["x_m"], row["y_m"], row["hub_height_m"])
        if pos in positions:
            raise ValueError("duplicate turbine locations")
        positions.add(pos)
        if not lookup and not control["minimum_rpm"] <= row["initial_rpm"] <= control["maximum_rpm"]:
            raise ValueError("initial_rpm must be within controller RPM limits")
