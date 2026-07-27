"""Pure wind-tunnel forcing choices for actuator-disk LES.

The configurations in this module contain no array implementation.  The
interpreter owns coordinates, reductions, and the distributed array layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

from .boussinesq import BoussinesqTendency


@dataclass(frozen=True, slots=True)
class NoActuatorDisk:
    """Explicit absence of turbine forcing."""


@dataclass(frozen=True, slots=True)
class PureThrustActuatorDisk:
    """Uniform, non-rotating actuator disk in execution units.

    ``thrust_coefficient_prime`` is based on the disk-normal velocity.  The
    ``normal_smoothing_width`` and ``transverse_smoothing_width`` use the
    Gaussian convention ``exp(-(x / epsilon)**2)``.  The geometric disk is
    convolved with this anisotropic Gaussian and discretely renormalized so
    the integrated force is independent of grid alignment.
    """

    x: float
    y: float
    z: float
    diameter: float
    thrust_coefficient_prime: float
    normal_smoothing_width: float
    transverse_smoothing_width: float
    hub_diameter: float = 0.0
    yaw_degrees: float = 0.0
    filtered_velocity_correction: bool = True

    def __post_init__(self) -> None:
        finite = (
            self.x,
            self.y,
            self.z,
            self.diameter,
            self.thrust_coefficient_prime,
            self.normal_smoothing_width,
            self.transverse_smoothing_width,
            self.hub_diameter,
            self.yaw_degrees,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("actuator-disk parameters must be finite")
        if self.diameter <= 0.0:
            raise ValueError("actuator-disk diameter must be positive")
        if self.thrust_coefficient_prime < 0.0:
            raise ValueError("local thrust coefficient must be nonnegative")
        if min(
            self.normal_smoothing_width,
            self.transverse_smoothing_width,
        ) <= 0.0:
            raise ValueError("actuator-disk smoothing widths must be positive")
        if self.hub_diameter < 0.0 or self.hub_diameter >= self.diameter:
            raise ValueError("hub diameter must lie in [0, rotor diameter)")


@dataclass(frozen=True, slots=True)
class NoFringe:
    """Explicit absence of downstream fringe forcing."""


@dataclass(frozen=True, slots=True)
class ConcurrentPrecursorFringe:
    """Relax the downstream fringe toward a concurrent precursor field."""

    start_x: float
    relaxation_time: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_x) or self.start_x < 0.0:
            raise ValueError("fringe start must be finite and nonnegative")
        if not math.isfinite(self.relaxation_time) or self.relaxation_time <= 0.0:
            raise ValueError("fringe relaxation time must be finite and positive")


@dataclass(frozen=True, slots=True)
class WindTunnelModel:
    """Independent actuator-disk and fringe choices."""

    actuator_disk: NoActuatorDisk | PureThrustActuatorDisk = NoActuatorDisk()
    fringe: NoFringe | ConcurrentPrecursorFringe = NoFringe()

    def __post_init__(self) -> None:
        if not isinstance(
            self.actuator_disk, (NoActuatorDisk, PureThrustActuatorDisk)
        ):
            raise TypeError("unsupported actuator-disk choice")
        if not isinstance(self.fringe, (NoFringe, ConcurrentPrecursorFringe)):
            raise TypeError("unsupported wind-tunnel fringe choice")


@dataclass(frozen=True, slots=True)
class ConcurrentPrecursorEnvironment:
    """Same-layout precursor velocity sampled at the main evaluation time."""

    velocity: Any


@dataclass(frozen=True, slots=True)
class WindTunnelDiagnostic:
    base: Any
    actuator_disk_enabled: bool
    concurrent_fringe_enabled: bool


@dataclass(frozen=True, slots=True)
class WindTunnelVectorFieldResult:
    tendency: Any
    diagnostic: WindTunnelDiagnostic


class WindTunnelAlgebra(Protocol):
    def wind_tunnel_tendency(
        self,
        velocity: Any,
        model: WindTunnelModel,
        environment: Any,
    ) -> Any: ...

    def combine_tendencies(self, tendencies: tuple[Any, ...]) -> Any: ...


@dataclass(frozen=True, slots=True)
class WindTunnelVectorField:
    """Add wind-tunnel forcing to an arbitrary dry momentum vector field."""

    algebra: WindTunnelAlgebra
    base: Any
    model: WindTunnelModel

    def __call__(self, evaluation: Any) -> WindTunnelVectorFieldResult:
        base = self.base(evaluation)
        disk_enabled = isinstance(self.model.actuator_disk, PureThrustActuatorDisk)
        fringe_enabled = isinstance(self.model.fringe, ConcurrentPrecursorFringe)
        if disk_enabled or fringe_enabled:
            forcing = self.algebra.wind_tunnel_tendency(
                evaluation.velocity,
                self.model,
                evaluation.environment,
            )
            tendency = self.algebra.combine_tendencies((base.tendency, forcing))
        else:
            tendency = base.tendency
        return WindTunnelVectorFieldResult(
            tendency,
            WindTunnelDiagnostic(
                base.diagnostic,
                disk_enabled,
                fringe_enabled,
            ),
        )


@dataclass(frozen=True, slots=True)
class WindTunnelBoussinesqVectorField:
    """Add wind-tunnel momentum forcing to a velocity--scalar vector field.

    The scalar tendency and closure state remain owned by the wrapped
    Boussinesq program.  Only its momentum tendency is augmented by the
    actuator disk and/or concurrent-precursor fringe.
    """

    algebra: WindTunnelAlgebra
    base: Any
    model: WindTunnelModel

    def __call__(self, evaluation: Any) -> WindTunnelVectorFieldResult:
        base = self.base(evaluation)
        disk_enabled = isinstance(self.model.actuator_disk, PureThrustActuatorDisk)
        fringe_enabled = isinstance(self.model.fringe, ConcurrentPrecursorFringe)
        momentum = base.tendency.velocity
        if disk_enabled or fringe_enabled:
            forcing = self.algebra.wind_tunnel_tendency(
                evaluation.velocity.velocity,
                self.model,
                evaluation.environment,
            )
            momentum = self.algebra.combine_tendencies((momentum, forcing))
        return WindTunnelVectorFieldResult(
            BoussinesqTendency(
                momentum,
                base.tendency.potential_temperature,
            ),
            WindTunnelDiagnostic(
                base.diagnostic,
                disk_enabled,
                fringe_enabled,
            ),
        )
