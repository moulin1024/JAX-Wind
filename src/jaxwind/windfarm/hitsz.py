"""HITSZ 1:100 wind-tunnel rotor with digitized HITSZ001 polars."""

from __future__ import annotations

from dataclasses import dataclass
import math

from jaxwind.domain import ScaleSystem
from jaxwind.physics import BladeElementActuatorDisk, NacelleTowerDrag


HITSZ_ROTOR_DIAMETER_M = 1.26
HITSZ_HUB_HEIGHT_M = 0.876
HITSZ_R9_ROTOR_SPEED_RPM = 480.0

# Marker centres digitized from Yang, Lin & Zhou (2024), Fig. 10. Chords in
# that figure are prototype metres and are reduced by the reported 1:100
# length scale below. The first and last stations bound the modeled span;
# AD-BEM elements are placed halfway between adjacent stations.
_NORMALIZED_RADIAL_STATIONS = tuple(0.04 * index for index in range(1, 26))
_PROTOTYPE_CHORDS_M = (
    3.24699, 3.63253, 3.93373, 4.15060, 4.28313,
    4.35542, 4.35542, 4.30723, 4.18675, 4.03012,
    3.83735, 3.62048, 3.36747, 3.09036, 2.80120,
    2.50000, 2.19880, 1.89759, 1.60843, 1.34337,
    1.09036, 0.87349, 0.68072, 0.53614, 0.43976,
)
_TWIST_DEGREES = (
    13.30843, 13.32530, 13.30843, 13.34217, 13.30843,
    13.30843, 11.45301, 10.30602, 9.05783, 7.80964,
    6.56145, 5.34699, 4.23373, 3.22169, 2.31084,
    1.56867, 0.94458, 0.48916, 0.18554, 0.11807,
    0.05060, 0.05060, 0.05060, 0.01687, 0.05060,
)

# E1/E4 marker centres digitized from Fig. 9 at Re=4.6e4. These are raster
# readings, not original XFOIL tables; the versioned case CSV records that
# provenance explicitly.
_POLAR_ALPHA_DEGREES = tuple(float(value) for value in range(-10, 31))
_POLAR_LIFT = (
    -0.8242, -0.7728, -0.5990, -0.4765, -0.3659,
    -0.2830, -0.2395, -0.2277, -0.1328, -0.0578,
    0.1240, 0.3314, 0.4993, 0.6691, 0.7719,
    0.8193, 0.9081, 1.0563, 1.1116, 1.2183,
    1.3249, 1.3723, 1.4632, 1.5067, 1.5383,
    1.5778, 1.6173, 1.6291, 1.6449, 1.6252,
    1.6528, 1.6568, 1.7477, 1.8030, 1.6173,
    1.6449, 1.6528, 1.6686, 1.6805, 1.7160,
    1.7714,
)
_POLAR_DRAG = (
    0.0395, 0.0869, 0.1362, 0.1481, 0.1481,
    0.1441, 0.1362, 0.1283, 0.1145, 0.1145,
    0.1283, 0.1244, 0.1244, 0.1204, 0.1244,
    0.1244, 0.1481, 0.1165, 0.1737, 0.1856,
    0.2053, 0.2053, 0.2014, 0.2113, 0.2053,
    0.2014, 0.2448, 0.2665, 0.2725, 0.2922,
    0.3080, 0.2942, 0.3277, 0.3751, 0.3554,
    0.4146, 0.3830, 0.3909, 0.4146, 0.4265,
    0.4423,
)


def _element_geometry() -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    radius = 0.5 * HITSZ_ROTOR_DIAMETER_M
    radii = tuple(
        0.5 * (left + right) * radius
        for left, right in zip(
            _NORMALIZED_RADIAL_STATIONS,
            _NORMALIZED_RADIAL_STATIONS[1:],
        )
    )
    widths = tuple(
        (right - left) * radius
        for left, right in zip(
            _NORMALIZED_RADIAL_STATIONS,
            _NORMALIZED_RADIAL_STATIONS[1:],
        )
    )
    chords = tuple(
        0.01 * 0.5 * (left + right)
        for left, right in zip(_PROTOTYPE_CHORDS_M, _PROTOTYPE_CHORDS_M[1:])
    )
    twists = tuple(
        0.5 * (left + right)
        for left, right in zip(_TWIST_DEGREES, _TWIST_DEGREES[1:])
    )
    return radii, widths, chords, twists


@dataclass(frozen=True, slots=True)
class HITSZR9BladeElementDisk:
    """Positioned SI-unit, azimuthally averaged R9 HITSZ rotor."""

    x_m: float
    y_m: float
    smoothing_width_m: float
    hub_height_m: float = HITSZ_HUB_HEIGHT_M
    rotor_speed_rpm: float = HITSZ_R9_ROTOR_SPEED_RPM
    pitch_degrees: float = 0.0
    smearing_azimuthal_elements: int = 64
    nacelle_length_m: float = 0.18
    nacelle_diameter_m: float = 0.05
    nacelle_drag_coefficient: float = 1.0
    tower_base_diameter_m: float = 0.04
    tower_top_diameter_m: float = 0.04
    tower_drag_coefficient: float = 1.0
    body_smoothing_width_m: float | None = None
    power_coefficient: float = 0.459

    model_name = "HITSZ R9 azimuthally averaged AD-BEM"
    rotor_source = "Yang, Lin & Zhou (2024), Figs. 9-10"
    blade_count = 3
    radial_stations = 24
    rotating_blade_element = True

    def __post_init__(self) -> None:
        values = (
            self.x_m, self.y_m, self.smoothing_width_m, self.hub_height_m,
            self.rotor_speed_rpm, self.pitch_degrees, self.nacelle_length_m,
            self.nacelle_diameter_m, self.nacelle_drag_coefficient,
            self.tower_base_diameter_m, self.tower_top_diameter_m,
            self.tower_drag_coefficient,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("HITSZ turbine parameters must be finite")
        if min(
            self.smoothing_width_m,
            self.hub_height_m,
            self.rotor_speed_rpm,
            self.nacelle_length_m,
            self.nacelle_diameter_m,
            self.tower_base_diameter_m,
            self.tower_top_diameter_m,
        ) <= 0.0:
            raise ValueError("HITSZ turbine dimensions and speed must be positive")
        if (
            isinstance(self.smearing_azimuthal_elements, bool)
            or self.smearing_azimuthal_elements <= 0
        ):
            raise ValueError("AD-BEM requires a positive azimuthal smearing count")

    @property
    def rotor_diameter_m(self) -> float:
        return HITSZ_ROTOR_DIAMETER_M

    @property
    def element_smoothing_widths_m(self) -> tuple[float, ...]:
        radii, widths, _, _ = _element_geometry()
        angle = 2.0 * math.pi / self.smearing_azimuthal_elements
        return tuple(
            math.hypot(radius * angle, width)
            for radius, width in zip(radii, widths)
        )

    def to_actuator_disk(self, *, scales: ScaleSystem) -> BladeElementActuatorDisk:
        radii, widths, chords, twists = _element_geometry()
        tip_radius = 0.5 * HITSZ_ROTOR_DIAMETER_M
        hub_radius = _NORMALIZED_RADIAL_STATIONS[0] * tip_radius
        angular_velocity = self.rotor_speed_rpm * 2.0 * math.pi / 60.0
        return BladeElementActuatorDisk(
            x=scales.to_execution_length(self.x_m),
            y=scales.to_execution_length(self.y_m),
            z=scales.to_execution_length(self.hub_height_m),
            blade_count=self.blade_count,
            hub_radius=scales.to_execution_length(hub_radius),
            tip_radius=scales.to_execution_length(tip_radius),
            angular_velocity=scales.to_execution_inverse_time(angular_velocity),
            smoothing_width=scales.to_execution_length(self.smoothing_width_m),
            element_radii=tuple(scales.to_execution_length(v) for v in radii),
            element_widths=tuple(scales.to_execution_length(v) for v in widths),
            element_chords=tuple(scales.to_execution_length(v) for v in chords),
            element_twist_degrees=twists,
            element_airfoil_ids=(0,) * len(radii),
            polar_alpha_degrees=_POLAR_ALPHA_DEGREES,
            polar_lift_coefficients=(_POLAR_LIFT,),
            polar_drag_coefficients=(_POLAR_DRAG,),
            pitch_degrees=self.pitch_degrees,
            tip_loss=True,
            root_loss=True,
            smearing_azimuthal_elements=self.smearing_azimuthal_elements,
        )

    def to_nacelle_tower(self, *, scales: ScaleSystem) -> NacelleTowerDrag:
        width = (
            self.smoothing_width_m
            if self.body_smoothing_width_m is None
            else self.body_smoothing_width_m
        )
        return NacelleTowerDrag(
            x=scales.to_execution_length(self.x_m),
            y=scales.to_execution_length(self.y_m),
            hub_height=scales.to_execution_length(self.hub_height_m),
            nacelle_length=scales.to_execution_length(self.nacelle_length_m),
            nacelle_diameter=scales.to_execution_length(self.nacelle_diameter_m),
            nacelle_drag_coefficient=self.nacelle_drag_coefficient,
            tower_base_diameter=scales.to_execution_length(
                self.tower_base_diameter_m
            ),
            tower_top_diameter=scales.to_execution_length(
                self.tower_top_diameter_m
            ),
            tower_drag_coefficient=self.tower_drag_coefficient,
            smoothing_width=scales.to_execution_length(width),
        )
