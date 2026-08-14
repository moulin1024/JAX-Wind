"""Pure wind-tunnel forcing choices for actuator-disk and actuator-line LES.

The configurations in this module contain no array implementation.  The
solver algebra owns coordinates, reductions, and the distributed array layout.
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
class NoActuatorLine:
    """Explicit absence of rotating blade-element forcing."""


@dataclass(frozen=True, slots=True)
class BladeElementActuatorLine:
    """Rigid or small-deflection modal actuator lines with tabulated airfoils.

    Geometry and angular velocity are expressed in the solver's execution
    units.  Each blade element carries one radial quadrature width, chord,
    twist, and airfoil-table index.  All polars share ``polar_alpha_degrees``;
    data importers may form that common grid by exactly resampling piecewise
    linear source tables onto their union.

    The azimuth convention is blade 1 pointing upward at zero degrees.
    Positive angular velocity advances it toward the rotor-plane horizontal
    basis.  Gaussian interpolation and projection use
    ``exp(-(distance / smoothing_width)**2)`` and are discretely normalized.
    """

    x: float
    y: float
    z: float
    blade_count: int
    hub_radius: float
    tip_radius: float
    angular_velocity: float
    smoothing_width: float
    element_radii: tuple[float, ...]
    element_widths: tuple[float, ...]
    element_chords: tuple[float, ...]
    element_twist_degrees: tuple[float, ...]
    element_airfoil_ids: tuple[int, ...]
    polar_alpha_degrees: tuple[float, ...]
    polar_lift_coefficients: tuple[tuple[float, ...], ...]
    polar_drag_coefficients: tuple[tuple[float, ...], ...]
    pitch_degrees: float = 0.0
    yaw_degrees: float = 0.0
    tilt_degrees: float = 0.0
    precone_degrees: float = 0.0
    initial_azimuth_degrees: float = 0.0
    tip_loss: bool = True
    root_loss: bool = True
    element_flap_displacements: tuple[float, ...] = ()
    element_edge_displacements: tuple[float, ...] = ()
    element_flap_slopes: tuple[float, ...] = ()
    element_edge_slopes: tuple[float, ...] = ()
    element_flap_velocities: tuple[float, ...] = ()
    element_edge_velocities: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        scalar_values = (
            self.x,
            self.y,
            self.z,
            self.hub_radius,
            self.tip_radius,
            self.angular_velocity,
            self.smoothing_width,
            self.pitch_degrees,
            self.yaw_degrees,
            self.tilt_degrees,
            self.precone_degrees,
            self.initial_azimuth_degrees,
        )
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("actuator-line parameters must be finite")
        if isinstance(self.blade_count, bool) or self.blade_count <= 0:
            raise ValueError("actuator line requires a positive blade count")
        if self.hub_radius < 0.0 or self.tip_radius <= self.hub_radius:
            raise ValueError("actuator-line radii must satisfy 0 <= hub < tip")
        if self.smoothing_width <= 0.0:
            raise ValueError("actuator-line smoothing width must be positive")

        element_count = len(self.element_radii)
        element_arrays = (
            self.element_widths,
            self.element_chords,
            self.element_twist_degrees,
            self.element_airfoil_ids,
        )
        if element_count == 0 or any(
            len(values) != element_count for values in element_arrays
        ):
            raise ValueError("actuator-line element arrays must have equal length")
        if not all(
            math.isfinite(value)
            for values in (
                self.element_radii,
                self.element_widths,
                self.element_chords,
                self.element_twist_degrees,
            )
            for value in values
        ):
            raise ValueError("actuator-line element data must be finite")
        if any(
            right <= left
            for left, right in zip(self.element_radii, self.element_radii[1:])
        ):
            raise ValueError("actuator-line element radii must be strictly increasing")
        if (
            self.element_radii[0] < self.hub_radius
            or self.element_radii[-1] > self.tip_radius
        ):
            raise ValueError("actuator-line elements must lie between hub and tip")
        if min(self.element_widths) <= 0.0 or min(self.element_chords) <= 0.0:
            raise ValueError("actuator-line widths and chords must be positive")
        point_count = self.blade_count * element_count
        deformation_arrays = (
            self.element_flap_displacements,
            self.element_edge_displacements,
            self.element_flap_slopes,
            self.element_edge_slopes,
            self.element_flap_velocities,
            self.element_edge_velocities,
        )
        if any(
            values and len(values) != point_count
            for values in deformation_arrays
        ):
            raise ValueError(
                "actuator-line deformation arrays must be empty or contain "
                "one value per blade element"
            )
        if not all(
            math.isfinite(value)
            for values in deformation_arrays
            for value in values
        ):
            raise ValueError("actuator-line deformation values must be finite")

        alpha_count = len(self.polar_alpha_degrees)
        polar_count = len(self.polar_lift_coefficients)
        if alpha_count < 2 or polar_count == 0:
            raise ValueError("actuator line requires nonempty aerodynamic polars")
        if len(self.polar_drag_coefficients) != polar_count:
            raise ValueError("actuator-line lift and drag polar counts must match")
        if any(
            right <= left
            for left, right in zip(
                self.polar_alpha_degrees,
                self.polar_alpha_degrees[1:],
            )
        ):
            raise ValueError("polar angles of attack must be strictly increasing")
        if not all(
            len(row) == alpha_count
            for table in (
                self.polar_lift_coefficients,
                self.polar_drag_coefficients,
            )
            for row in table
        ):
            raise ValueError("all actuator-line polar rows must share the alpha grid")
        if not all(
            math.isfinite(value)
            for table in (
                self.polar_lift_coefficients,
                self.polar_drag_coefficients,
            )
            for row in table
            for value in row
        ):
            raise ValueError("actuator-line polar coefficients must be finite")
        if any(
            value < 0.0
            for row in self.polar_drag_coefficients
            for value in row
        ):
            raise ValueError("actuator-line drag coefficients must be nonnegative")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < polar_count
            for index in self.element_airfoil_ids
        ):
            raise ValueError("actuator-line airfoil index is outside the polar table")


@dataclass(frozen=True, slots=True)
class NoFringe:
    """Explicit absence of downstream fringe forcing."""


@dataclass(frozen=True, slots=True)
class ConcurrentPrecursorFringe:
    """Relax the downstream fringe toward a concurrent precursor field."""

    start_x: float
    relaxation_time: float
    rise_width: float | None = None
    fall_width: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_x) or self.start_x < 0.0:
            raise ValueError("fringe start must be finite and nonnegative")
        if not math.isfinite(self.relaxation_time) or self.relaxation_time <= 0.0:
            raise ValueError("fringe relaxation time must be finite and positive")
        widths = (self.rise_width, self.fall_width)
        if (widths[0] is None) != (widths[1] is None):
            raise ValueError("fringe rise and fall widths must be specified together")
        if widths[0] is not None and (
            not all(math.isfinite(value) for value in widths)
            or min(widths) <= 0.0
        ):
            raise ValueError("fringe rise and fall widths must be finite and positive")

    def resolved_widths(self, end_x: float) -> tuple[float, float]:
        available = end_x - self.start_x
        if not math.isfinite(end_x) or available <= 0.0:
            raise ValueError("fringe start must lie before the periodic seam")
        if self.rise_width is None:
            return 0.5 * available, 0.5 * available
        assert self.fall_width is not None
        if self.rise_width + self.fall_width > available:
            raise ValueError("fringe rise and fall widths exceed the fringe region")
        return self.rise_width, self.fall_width


@dataclass(frozen=True, slots=True)
class WindTunnelModel:
    """Independent turbine and fringe choices."""

    actuator_disk: NoActuatorDisk | PureThrustActuatorDisk = NoActuatorDisk()
    fringe: NoFringe | ConcurrentPrecursorFringe = NoFringe()
    actuator_line: NoActuatorLine | BladeElementActuatorLine = (
        NoActuatorLine()
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.actuator_disk, (NoActuatorDisk, PureThrustActuatorDisk)
        ):
            raise TypeError("unsupported actuator-disk choice")
        if not isinstance(self.fringe, (NoFringe, ConcurrentPrecursorFringe)):
            raise TypeError("unsupported wind-tunnel fringe choice")
        if not isinstance(
            self.actuator_line,
            (
                NoActuatorLine,
                BladeElementActuatorLine,
            ),
        ):
            raise TypeError("unsupported actuator-line choice")
        if isinstance(
            self.actuator_disk,
            PureThrustActuatorDisk,
        ) and isinstance(
            self.actuator_line,
            BladeElementActuatorLine,
        ):
            raise ValueError("actuator disk and actuator line are mutually exclusive")


@dataclass(frozen=True, slots=True)
class ConcurrentPrecursorEnvironment:
    """Same-layout precursor velocity sampled at the main evaluation time."""

    velocity: Any
    closure: Any | None = None


@dataclass(frozen=True, slots=True)
class ConcurrentPrecursorLasdEventDiagnostic:
    """LASD update plus confirmation that precursor memory was imposed."""

    lasd: Any
    closure_relaxed: bool = True


@dataclass(frozen=True, slots=True)
class ConcurrentPrecursorLasdAcceptedStepEvent:
    """Relax main LASD memory at a synchronized accepted-step boundary."""

    algebra: Any
    model: Any
    dt: float
    fringe: ConcurrentPrecursorFringe

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("LASD event dt must be finite and positive")
        if not isinstance(self.fringe, ConcurrentPrecursorFringe):
            raise TypeError("concurrent LASD event requires a precursor fringe")

    def __call__(self, fields: Any, clock: Any, environment: Any) -> tuple[Any, Any]:
        if (
            not isinstance(environment, ConcurrentPrecursorEnvironment)
            or environment.closure is None
        ):
            raise TypeError(
                "concurrent LASD event requires precursor closure memory"
            )
        relaxed = self.algebra.relax_lasd_closure(
            fields,
            environment.closure,
            self.fringe,
            self.dt,
        )
        prepared, diagnostic = self.algebra.prepare_lasd_closure(
            relaxed,
            self.model,
            clock,
            self.dt,
        )
        return prepared, ConcurrentPrecursorLasdEventDiagnostic(diagnostic)


@dataclass(frozen=True, slots=True)
class WindTunnelDiagnostic:
    base: Any
    actuator_disk_enabled: bool
    concurrent_fringe_enabled: bool
    actuator_line_enabled: bool = False


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
        evaluation_time: Any | None = None,
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
        line_enabled = isinstance(
            self.model.actuator_line,
            BladeElementActuatorLine,
        )
        fringe_enabled = isinstance(self.model.fringe, ConcurrentPrecursorFringe)
        if disk_enabled or line_enabled or fringe_enabled:
            forcing = self.algebra.wind_tunnel_tendency(
                evaluation.velocity,
                self.model,
                evaluation.environment,
                evaluation.time,
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
                line_enabled,
            ),
        )


@dataclass(frozen=True, slots=True)
class WindTunnelBoussinesqVectorField:
    """Add wind-tunnel momentum forcing to a velocity--scalar vector field.

    The scalar tendency and closure state remain owned by the wrapped
    Boussinesq program.  Only its momentum tendency is augmented by the
    actuator disk, actuator line, and/or concurrent-precursor fringe.
    """

    algebra: WindTunnelAlgebra
    base: Any
    model: WindTunnelModel

    def _combine(
        self,
        evaluation: Any,
        base: Any,
    ) -> WindTunnelVectorFieldResult:
        disk_enabled = isinstance(self.model.actuator_disk, PureThrustActuatorDisk)
        line_enabled = isinstance(
            self.model.actuator_line,
            BladeElementActuatorLine,
        )
        fringe_enabled = isinstance(self.model.fringe, ConcurrentPrecursorFringe)
        momentum = base.tendency.velocity
        if disk_enabled or line_enabled or fringe_enabled:
            forcing = self.algebra.wind_tunnel_tendency(
                evaluation.velocity.velocity,
                self.model,
                evaluation.environment,
                getattr(evaluation, "time", None),
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
                line_enabled,
            ),
        )

    def __call__(self, evaluation: Any) -> WindTunnelVectorFieldResult:
        return self._combine(evaluation, self.base(evaluation))

    def evaluate_prepared(
        self,
        evaluation: Any,
        momentum_context: Any,
    ) -> WindTunnelVectorFieldResult:
        evaluate_prepared = getattr(self.base, "evaluate_prepared", None)
        if evaluate_prepared is None:
            raise TypeError("wind-tunnel base does not support prepared evaluation")
        return self._combine(
            evaluation,
            evaluate_prepared(evaluation, momentum_context),
        )
