"""Minimal physical-unit actuator-disk turbine parameterization."""

from __future__ import annotations

from dataclasses import dataclass
import math

from jaxwind.domain import ScaleSystem
from jaxwind.physics import PureThrustActuatorDisk


DTU_10MW_HUB_HEIGHT_M = 119.0
DTU_10MW_ROTOR_DIAMETER_M = 178.3


@dataclass(frozen=True, slots=True)
class SimpleActuatorDisk:
    """Uniform, non-rotating disk normal to the streamwise direction.

    Geometry is specified in metres. ``thrust_coefficient_prime`` is based on
    the velocity sampled at the disk, so the force per unit density is
    ``-0.5 * C_T' * u_disk**2 * area``. One Gaussian smoothing width is used
    in both the streamwise and transverse directions.
    """

    x_m: float
    y_m: float
    hub_height_m: float
    rotor_diameter_m: float
    thrust_coefficient_prime: float
    smoothing_width_m: float

    def __post_init__(self) -> None:
        values = (
            self.x_m,
            self.y_m,
            self.hub_height_m,
            self.rotor_diameter_m,
            self.thrust_coefficient_prime,
            self.smoothing_width_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("simple actuator-disk parameters must be finite")
        if self.hub_height_m < 0.0:
            raise ValueError("actuator-disk hub height must be nonnegative")
        if self.rotor_diameter_m <= 0.0:
            raise ValueError("actuator-disk rotor diameter must be positive")
        if self.thrust_coefficient_prime < 0.0:
            raise ValueError("actuator-disk thrust coefficient must be nonnegative")
        if self.smoothing_width_m <= 0.0:
            raise ValueError("actuator-disk smoothing width must be positive")

    def to_actuator_disk(self, *, scales: ScaleSystem) -> PureThrustActuatorDisk:
        """Lower the SI turbine to the solver's execution-unit forcing choice."""

        if not isinstance(scales, ScaleSystem):
            raise TypeError("simple actuator-disk lowering requires ScaleSystem")
        smoothing_width = scales.to_execution_length(self.smoothing_width_m)
        return PureThrustActuatorDisk(
            x=scales.to_execution_length(self.x_m),
            y=scales.to_execution_length(self.y_m),
            z=scales.to_execution_length(self.hub_height_m),
            diameter=scales.to_execution_length(self.rotor_diameter_m),
            thrust_coefficient_prime=self.thrust_coefficient_prime,
            normal_smoothing_width=smoothing_width,
            transverse_smoothing_width=smoothing_width,
            hub_diameter=0.0,
            yaw_degrees=0.0,
            filtered_velocity_correction=True,
        )


def dtu_10mw_reference_actuator_disk(
    *,
    x_m: float,
    y_m: float,
    smoothing_width_m: float,
    thrust_coefficient_prime: float = 4.0 / 3.0,
) -> SimpleActuatorDisk:
    """Return a fixed-loading ADM with the DTU 10-MW reference geometry.

    The 178.3 m rotor diameter and 119 m hub height follow the DTU reference
    turbine. The default ``C_T'=4/3`` is an explicit constant-loading ADM
    assumption; this model does not reproduce the turbine controller or its
    wind-speed-dependent thrust curve.
    """

    return SimpleActuatorDisk(
        x_m=x_m,
        y_m=y_m,
        hub_height_m=DTU_10MW_HUB_HEIGHT_M,
        rotor_diameter_m=DTU_10MW_ROTOR_DIAMETER_M,
        thrust_coefficient_prime=thrust_coefficient_prime,
        smoothing_width_m=smoothing_width_m,
    )


__all__ = [
    "DTU_10MW_HUB_HEIGHT_M",
    "DTU_10MW_ROTOR_DIAMETER_M",
    "SimpleActuatorDisk",
    "dtu_10mw_reference_actuator_disk",
]
