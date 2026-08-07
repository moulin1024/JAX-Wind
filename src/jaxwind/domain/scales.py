"""Array-independent coherent scales for nondimensional JAX execution."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .grid import UniformGrid


@dataclass(frozen=True, slots=True)
class ScaleSystem:
    """Two-base incompressible mechanical scale system.

    Public case values remain SI.  Length and velocity determine time,
    acceleration, inverse-time, pressure-per-density, and viscosity scales.
    """

    length: float
    velocity: float
    version: str = "jaxwind.mechanical-scales.v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length scale must be finite and positive")
        if not math.isfinite(self.velocity) or self.velocity <= 0.0:
            raise ValueError("velocity scale must be finite and positive")
        if not self.version:
            raise ValueError("scale-system version must be non-empty")

    @property
    def time(self) -> float:
        return self.length / self.velocity

    @property
    def acceleration(self) -> float:
        return self.velocity * self.velocity / self.length

    @property
    def inverse_time(self) -> float:
        return self.velocity / self.length

    @property
    def kinematic_pressure(self) -> float:
        return self.velocity * self.velocity

    @property
    def kinematic_viscosity(self) -> float:
        return self.length * self.velocity

    @property
    def fingerprint(self) -> str:
        return (
            f"{self.version}|length={float(self.length).hex()}"
            f"|velocity={float(self.velocity).hex()}"
        )

    def to_execution_length(self, value):
        return value / self.length

    def from_execution_length(self, value):
        return value * self.length

    def to_execution_velocity(self, value):
        return value / self.velocity

    def from_execution_velocity(self, value):
        return value * self.velocity

    def to_execution_time(self, value):
        return value / self.time

    def from_execution_time(self, value):
        return value * self.time

    def to_execution_acceleration(self, value):
        return value / self.acceleration

    def from_execution_acceleration(self, value):
        return value * self.acceleration

    def to_execution_kinematic_viscosity(self, value):
        return value / self.kinematic_viscosity

    def from_execution_kinematic_viscosity(self, value):
        return value * self.kinematic_viscosity

    def to_execution_inverse_time(self, value):
        return value / self.inverse_time

    def from_execution_inverse_time(self, value):
        return value * self.inverse_time

    def to_execution_inverse_time_squared(self, value):
        return value / (self.inverse_time * self.inverse_time)

    def from_execution_inverse_time_squared(self, value):
        return value * self.inverse_time * self.inverse_time

    def to_execution_grid(self, grid: UniformGrid) -> UniformGrid:
        return UniformGrid(
            grid.nx,
            grid.ny,
            grid.nz,
            self.to_execution_length(grid.lx),
            self.to_execution_length(grid.ly),
            self.to_execution_length(grid.lz),
        )


@dataclass(frozen=True, slots=True)
class BoussinesqScaleSystem:
    """Mechanical scales plus one potential-temperature-difference scale."""

    mechanical: ScaleSystem
    potential_temperature_difference: float
    version: str = "jaxwind.boussinesq-scales.v1"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.potential_temperature_difference)
            or self.potential_temperature_difference <= 0.0
        ):
            raise ValueError("temperature-difference scale must be finite and positive")

    @property
    def fingerprint(self) -> str:
        return (
            f"{self.version}|mechanical={self.mechanical.fingerprint}"
            f"|potential_temperature_difference="
            f"{float(self.potential_temperature_difference).hex()}"
        )

    def to_execution_potential_temperature(self, value):
        return value / self.potential_temperature_difference

    def from_execution_potential_temperature(self, value):
        return value * self.potential_temperature_difference

    def to_execution_temperature_tendency(self, value):
        return value / (self.potential_temperature_difference / self.mechanical.time)

    def from_execution_temperature_tendency(self, value):
        return value * (self.potential_temperature_difference / self.mechanical.time)

    def to_execution_temperature_flux(self, value):
        """Lower a kinematic potential-temperature flux to execution units."""
        return value / (
            self.potential_temperature_difference * self.mechanical.velocity
        )

    def from_execution_temperature_flux(self, value):
        """Lift an execution scalar flux to canonical K m s-1 units."""
        return value * (
            self.potential_temperature_difference * self.mechanical.velocity
        )

    def to_execution_buoyancy_coefficient(
        self,
        *,
        gravity: float,
        reference_potential_temperature: float,
    ) -> float:
        if not math.isfinite(gravity) or gravity <= 0.0:
            raise ValueError("gravity must be finite and positive")
        if (
            not math.isfinite(reference_potential_temperature)
            or reference_potential_temperature <= 0.0
        ):
            raise ValueError("reference potential temperature must be positive")
        physical_per_execution_temperature = (
            gravity
            * self.potential_temperature_difference
            / reference_potential_temperature
        )
        return physical_per_execution_temperature / self.mechanical.acceleration


@dataclass(frozen=True, slots=True)
class PassiveScalarScaleSystem:
    """Mechanical scales plus one passive mass-concentration scale."""

    mechanical: ScaleSystem
    concentration: float
    version: str = "jaxwind.passive-scalar-scales.v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.concentration) or self.concentration <= 0.0:
            raise ValueError("concentration scale must be finite and positive")

    @property
    def fingerprint(self) -> str:
        return (
            f"{self.version}|mechanical={self.mechanical.fingerprint}"
            f"|concentration={float(self.concentration).hex()}"
        )

    def to_execution_concentration(self, value):
        return value / self.concentration

    def from_execution_concentration(self, value):
        return value * self.concentration

    def to_execution_concentration_tendency(self, value):
        return value / (self.concentration / self.mechanical.time)

    def from_execution_concentration_tendency(self, value):
        return value * (self.concentration / self.mechanical.time)

    def to_execution_concentration_flux(self, value):
        return value / (self.concentration * self.mechanical.velocity)

    def from_execution_concentration_flux(self, value):
        return value * self.concentration * self.mechanical.velocity
