"""Surface scalar evolution and atmosphere--surface exchange programs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, NamedTuple


@dataclass(frozen=True, slots=True)
class NoSurfaceTransfer:
    """Use the independently prescribed momentum and scalar boundaries."""


@dataclass(frozen=True, slots=True)
class MoninObukhovSurfaceTransfer:
    """Coupled Businger--Dyer exchange for an evolving surface scalar."""

    scalar_roughness_length: float
    surface_scalar_initial: float
    surface_scalar_rate: float = 0.0
    x_velocity_offset: float = 0.0
    y_velocity_offset: float = 0.0
    positive_zeta_momentum_slope: float = 4.8
    positive_zeta_scalar_slope: float = 7.8
    negative_zeta_momentum_coefficient: float = 16.0
    negative_zeta_scalar_coefficient: float = 16.0
    iterations: int = 12
    relaxation: float = 0.5
    maximum_abs_zeta: float = 10.0

    def __post_init__(self) -> None:
        positive = (
            self.scalar_roughness_length,
            self.positive_zeta_momentum_slope,
            self.positive_zeta_scalar_slope,
            self.negative_zeta_momentum_coefficient,
            self.negative_zeta_scalar_coefficient,
            self.maximum_abs_zeta,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("surface-transfer constants must be finite and positive")
        if not all(
            math.isfinite(value)
            for value in (
                self.surface_scalar_initial,
                self.surface_scalar_rate,
                self.x_velocity_offset,
                self.y_velocity_offset,
            )
        ):
            raise ValueError("surface-scalar evolution and velocity offsets must be finite")
        if self.iterations <= 0:
            raise ValueError("surface-transfer iterations must be positive")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("surface-transfer relaxation must lie in (0, 1]")


class SurfaceTransferResult(NamedTuple):
    """Exchange values in execution units; stress points into the fluid."""

    stress_x: Any
    stress_y: Any
    scalar_flux: Any
    friction_velocity: Any
    scalar_scale: Any
    obukhov_length: Any
    surface_scalar: Any
    wall_x_acceleration: Any
    wall_y_acceleration: Any
    scalar_surface_source: Any


__all__ = [
    "MoninObukhovSurfaceTransfer",
    "NoSurfaceTransfer",
    "SurfaceTransferResult",
]
