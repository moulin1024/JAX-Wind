"""Explicit configuration for the Fluent-reference atmospheric water DPM."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DPMOptions:
    diameters_m: tuple[float, ...]
    mass_fractions: tuple[float, ...]
    injection_velocity_m_s: tuple[float, float, float]
    injection_temperature_k: float
    capacity: int = 256
    tracking_substeps: int = 4
    max_path_segments: int = 64
    max_eddy_intervals: int = 128
    dispersion: str = "drw"
    random_lifetime: bool = False
    seed: int = 1729
    les_model: str = "fluent-smagorinsky"
    amd_length_scale_m: float | None = None
    smagorinsky_constant: float = 0.1
    von_karman: float = 0.41
    vaporization: str = "diffusion-controlled"

    def __post_init__(self):
        if not self.diameters_m or len(self.diameters_m) != len(self.mass_fractions):
            raise ValueError(
                "DPM diameters and mass fractions must have equal nonzero length"
            )
        if any(
            not math.isfinite(v) or v <= 0
            for v in (*self.diameters_m, *self.mass_fractions)
        ):
            raise ValueError("DPM diameters and weights must be finite and positive")
        if abs(sum(self.mass_fractions) - 1) > 1e-12:
            raise ValueError(
                "DPM mass fractions must sum to one; no silent renormalization"
            )
        if len(self.injection_velocity_m_s) != 3 or not all(
            math.isfinite(v) for v in self.injection_velocity_m_s
        ):
            raise ValueError("DPM injection velocity needs three finite components")
        if not 273.15 <= self.injection_temperature_k < 373.15:
            raise ValueError("DPM currently supports warm, nonboiling water")
        for name in (
            "capacity",
            "tracking_substeps",
            "max_path_segments",
            "max_eddy_intervals",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"DPM {name} must be a positive integer")
        if self.capacity < len(self.diameters_m):
            raise ValueError("DPM capacity must fit an injection batch")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("DPM seed must be a nonnegative 31-bit integer")
        if type(self.random_lifetime) is not bool or self.dispersion not in (
            "off",
            "drw",
        ):
            raise ValueError("invalid DPM stochastic tracking options")
        if not 0 < self.smagorinsky_constant < 1 or not 0 < self.von_karman < 1:
            raise ValueError("invalid DPM LES constants")
        if self.les_model not in ("fluent-smagorinsky", "amd-inferred"):
            raise ValueError("invalid DPM LES model")
        if self.les_model == "amd-inferred":
            length = self.amd_length_scale_m
            if (
                isinstance(length, bool)
                or not isinstance(length, (int, float))
                or not math.isfinite(length)
                or length <= 0
            ):
                raise ValueError(
                    "AMD-inferred DPM requires an explicit positive amd_length_scale_m"
                )
        elif self.amd_length_scale_m is not None:
            raise ValueError("amd_length_scale_m requires les_model=amd-inferred")
        if self.vaporization not in (
            "diffusion-controlled",
            "convection-diffusion-controlled",
        ):
            raise ValueError("invalid DPM vaporization model")


def load_dpm(table):
    if not isinstance(table, dict) or table.keys() - {
        f.name for f in fields(DPMOptions)
    }:
        raise ValueError("unknown DPM settings")
    values = dict(table)
    for key in ("diameters_m", "mass_fractions", "injection_velocity_m_s"):
        if key not in values or not isinstance(values[key], (tuple, list)):
            raise ValueError(f"DPM requires {key}")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in values[key]
        ):
            raise ValueError(f"DPM {key} must contain numbers")
        values[key] = tuple(values[key])
    if "injection_temperature_k" not in values or isinstance(
        values["injection_temperature_k"], bool
    ):
        raise ValueError("DPM requires an injection temperature")
    for key in ("injection_temperature_k", "smagorinsky_constant", "von_karman"):
        if key in values and (
            isinstance(values[key], bool)
            or not isinstance(values[key], (int, float))
            or not math.isfinite(values[key])
        ):
            raise ValueError(f"DPM {key} must be a finite number")
    return DPMOptions(**values)
