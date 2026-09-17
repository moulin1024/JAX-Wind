"""Strict configuration for shared atmospheric humidity and water injection."""

import math
from dataclasses import dataclass, field, fields

from jaxwind.physics.moisture import MoistureConfig
from .fluent_dpm import DPMOptions, load_dpm


@dataclass(frozen=True)
class WaterSprayOptions:
    mass_flow_rate_kg_s: float
    streamwise_offset_m: float
    standard_deviation_m: tuple[float, float, float]
    droplet_diameter_m: float = 50.0e-6
    ramp_time_s: float = 0.0
    model: str = "entrained"
    dpm: DPMOptions | None = None


@dataclass(frozen=True)
class AtmosphericMoistureOptions:
    ambient_relative_humidity: float = 0.5
    temperature_offset_k: float = 300.0
    reference_temperature_k: float = 300.0
    thermodynamics: MoistureConfig = field(default_factory=MoistureConfig)


def load_moisture(document):
    table = document.get("moisture")
    spray_table = document.get("water_spray")
    if table is None:
        if spray_table is not None:
            raise ValueError("water_spray requires physics.moisture")
        return None, None
    if not isinstance(table, dict):
        raise ValueError("physics.moisture must be a table")
    allowed = {f.name for f in fields(AtmosphericMoistureOptions)} - {"thermodynamics"}
    allowed |= {"pressure_pa", "dry_air_density_kg_m3"}
    if table.keys() - allowed:
        raise ValueError(
            "unknown physics.moisture keys: " + str(table.keys() - allowed)
        )

    def number(source, key, default=None):
        value = source.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be a finite number")
        return float(value)

    moisture = AtmosphericMoistureOptions(
        ambient_relative_humidity=number(table, "ambient_relative_humidity", 0.5),
        temperature_offset_k=number(table, "temperature_offset_k", 300.0),
        reference_temperature_k=number(table, "reference_temperature_k", 300.0),
        thermodynamics=MoistureConfig(
            pressure=number(table, "pressure_pa", 100_000.0),
            dry_air_density=number(table, "dry_air_density_kg_m3", 1.225),
        ),
    )
    if not 0.0 <= moisture.ambient_relative_humidity <= 1.0:
        raise ValueError("ambient_relative_humidity must lie in [0, 1]")
    if moisture.temperature_offset_k < 0 or moisture.reference_temperature_k <= 0:
        raise ValueError(
            "temperature offset must be nonnegative and reference temperature positive"
        )
    if spray_table is None:
        return moisture, None
    if not isinstance(spray_table, dict):
        raise ValueError("physics.water_spray must be a table")
    allowed = {f.name for f in fields(WaterSprayOptions)}
    model = spray_table.get("model", "entrained")
    if model not in ("entrained", "fluent-dpm"):
        raise ValueError("water_spray.model must be entrained or fluent-dpm")
    required = {"mass_flow_rate_kg_s", "streamwise_offset_m"}
    required |= {"dpm"} if model == "fluent-dpm" else {"standard_deviation_m"}
    if model == "entrained" and "dpm" in spray_table:
        raise ValueError("DPM options require model=fluent-dpm")
    if spray_table.keys() - allowed or required - spray_table.keys():
        raise ValueError("water_spray has unknown or missing settings")
    widths = spray_table.get("standard_deviation_m", (1.0, 1.0, 1.0))
    if not isinstance(widths, (list, tuple)) or len(widths) != 3:
        raise ValueError("water_spray.standard_deviation_m needs three positive widths")
    widths = tuple(number({"width": v}, "width") for v in widths)
    spray = WaterSprayOptions(
        number(spray_table, "mass_flow_rate_kg_s"),
        number(spray_table, "streamwise_offset_m"),
        widths,
        number(spray_table, "droplet_diameter_m", 50.0e-6),
        number(spray_table, "ramp_time_s", 0.0),
        model,
        load_dpm(spray_table["dpm"]) if model == "fluent-dpm" else None,
    )
    if min(spray.streamwise_offset_m, spray.droplet_diameter_m, *widths) <= 0:
        raise ValueError("water spray offset, diameter and widths must be positive")
    if min(spray.mass_flow_rate_kg_s, spray.ramp_time_s) < 0:
        raise ValueError("water spray flow and ramp must be nonnegative")
    if moisture.reference_temperature_k < moisture.thermodynamics.freezing_temperature:
        raise ValueError("water spray currently requires a warm atmosphere")
    return moisture, spray
