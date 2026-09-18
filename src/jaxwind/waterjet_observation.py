"""Explicit downstream sensor mapping for an ideal adiabatic spray separator.

The separator removes liquid only; gas enthalpy and vapor continue through the
resolved downstream duct. No perfect mixing, sensor wetting, plate heat exchange
or measured separation efficiency is implied by this idealization.
"""

import math
from dataclasses import dataclass, fields
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np

from .physics.moisture import saturation_vapor_pressure_water


@dataclass(frozen=True)
class WaterjetObservation:
    sensor_x_m: float
    separator_model: str = "ideal-adiabatic-trap"
    sensor_y_m: tuple = (0.0975, 0.2925, 0.4875)
    sensor_z_m: tuple = (0.0975, 0.2925, 0.4875)

    @classmethod
    def from_table(cls, table, grid, dpm):
        if not isinstance(table, dict) or table.keys() - {f.name for f in fields(cls)}:
            raise ValueError("DPM waterjet requires an explicit case.observation table")
        if "sensor_x_m" not in table:
            raise ValueError("DPM waterjet requires sensor_x_m")
        obj = cls(**table)
        if obj.separator_model != "ideal-adiabatic-trap" or dpm.eliminator_x_m is None:
            raise ValueError(
                "waterjet observation requires an ideal-adiabatic-trap separator"
            )
        if any(
            not isinstance(v, (list, tuple)) or len(v) != 3
            for v in (obj.sensor_y_m, obj.sensor_z_m)
        ):
            raise ValueError("waterjet requires a 3 by 3 sensor grid")
        for values, centers in (
            ([obj.sensor_x_m], grid.x_centers),
            (obj.sensor_y_m, grid.y_centers),
            (obj.sensor_z_m, grid.z_centers),
        ):
            if any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not centers[0] <= v <= centers[-1]
                for v in values
            ):
                raise ValueError(
                    "sensors must lie within cell centres; no extrapolation"
                )
        if len(obj.sensor_y_m) != 3 or len(obj.sensor_z_m) != 3:
            raise ValueError("waterjet requires a 3 by 3 sensor grid")
        if any(
            b <= a
            for values in (obj.sensor_y_m, obj.sensor_z_m)
            for a, b in pairwise(values)
        ):
            raise ValueError(
                "sensor coordinates must increase in bottom/top and left/right order"
            )
        left = max(
            0,
            np.searchsorted(np.asarray(grid.x_centers), obj.sensor_x_m, side="right")
            - 1,
        )
        if float(grid.x_centers[left]) < dpm.eliminator_x_m:
            raise ValueError(
                "sensor interpolation must use only downstream separator cells"
            )
        return obj


def sample_plane(field, grid, observation):
    """Trilinear paired point samples, ordered bottom/middle/top then left/right."""
    plane = jax.vmap(
        jax.vmap(lambda row: jnp.interp(observation.sensor_x_m, grid.x_centers, row))
    )(field)
    return jnp.stack(
        [
            jnp.interp(
                z,
                grid.z_centers,
                jax.vmap(lambda row, y=y: jnp.interp(y, grid.y_centers, row))(plane),
            )
            for z in observation.sensor_z_m
            for y in observation.sensor_y_m
        ]
    )


def wet_bulb_c(dry_c, vapor_ratio, pressure, gas_constant_ratio):
    """Invert the same ventilated-psychrometer assumption as the inlet.

    This is a virtual dry sensor, not a prediction of liquid wetting bias.
    Supersaturation has no unsaturated wet-bulb solution; return NaN so the
    unsupported observation cannot silently pass a comparison.
    """
    partial = pressure * vapor_ratio / (gas_constant_ratio + vapor_ratio)

    def residual(wet):
        return (
            saturation_vapor_pressure_water(wet + 273.15)
            - 0.00066 * (1 + 0.00115 * wet) * pressure * (dry_c - wet)
            - partial
        )

    low = jnp.full_like(dry_c, -80.0)
    high = dry_c

    def bisect(_, bounds):
        lo, hi = bounds
        mid = (lo + hi) / 2
        return jnp.where(residual(mid) < 0, mid, lo), jnp.where(
            residual(mid) < 0, hi, mid
        )

    low, high = jax.lax.fori_loop(0, 50, bisect, (low, high))
    return jnp.where(residual(dry_c) >= 0, (low + high) / 2, jnp.nan)
