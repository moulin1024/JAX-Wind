"""Import a rigid, quasi-steady actuator line from an OpenFAST input deck.

This adapter intentionally reads only the OpenFAST fields needed by the
JAX-Wind rigid actuator-line model.  It follows referenced ElastoDyn, AeroDyn,
blade, and airfoil files, while retaining the ordinary OpenFAST files as the
source of truth.  Structural flexibility, controllers, dynamic inflow, and
unsteady aerodynamics are not reimplemented here.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, fields
import math
from pathlib import Path

from jaxwind.domain import ScaleSystem
from jaxwind.physics import (
    BladeElementActuatorDisk,
    BladeElementActuatorLine,
    NacelleTowerDrag,
)

from .errors import OpenFASTInputError
from .parser import (
    _Line,
    _boolean_value,
    _finite_number,
    _float_value,
    _integer_value,
    _optional_boolean_value,
    _optional_integer_value,
    _path_value,
    _positive_number,
    _read_lines,
    _rows_after,
    _value,
)


@dataclass(frozen=True, slots=True)
class OpenFASTAirfoilPolar:
    """One AeroDyn airfoil polar in the file's SI/degree convention."""

    source: Path
    alpha_degrees: tuple[float, ...]
    lift_coefficients: tuple[float, ...]
    drag_coefficients: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OpenFASTRigidTurbine:
    """Physical rigid-rotor data resolved from one OpenFAST input deck."""

    source: Path
    aerodyn_source: Path
    elastodyn_source: Path
    blade_source: Path
    airfoil_sources: tuple[Path, ...]
    blade_count: int
    hub_radius_m: float
    tip_radius_m: float
    hub_height_m: float
    rotor_speed_rpm: float
    pitch_degrees: float
    yaw_degrees: float
    shaft_tilt_degrees: float
    precone_degrees: float
    initial_azimuth_degrees: float
    mirror_rotor: bool
    element_radii_m: tuple[float, ...]
    element_widths_m: tuple[float, ...]
    element_chords_m: tuple[float, ...]
    element_twist_degrees: tuple[float, ...]
    element_airfoil_ids: tuple[int, ...]
    polar_alpha_degrees: tuple[float, ...]
    polar_lift_coefficients: tuple[tuple[float, ...], ...]
    polar_drag_coefficients: tuple[tuple[float, ...], ...]
    tip_loss: bool
    root_loss: bool
    compatibility_notes: tuple[str, ...]

    @property
    def angular_velocity_rad_s(self) -> float:
        """Signed rigid-rotor speed in radians per second."""

        direction = -1.0 if self.mirror_rotor else 1.0
        return direction * self.rotor_speed_rpm * 2.0 * math.pi / 60.0

    def to_actuator_line(
        self,
        *,
        scales: ScaleSystem,
        x_m: float,
        y_m: float,
        smoothing_width_m: float,
        hub_height_m: float | None = None,
        rotor_speed_rpm: float | None = None,
        pitch_degrees: float | None = None,
        yaw_degrees: float | None = None,
        initial_azimuth_degrees: float | None = None,
        tip_loss: bool | None = None,
        root_loss: bool | None = None,
    ) -> BladeElementActuatorLine:
        """Lower physical OpenFAST data to an execution-unit actuator line.

        ``x_m`` and ``y_m`` locate the rotor apex in the LES domain.  OpenFAST
        supplies the default hub height and orientation.  Runtime operating
        point overrides are explicit because a rigid first slice has no
        ServoDyn controller or structural state.
        """

        speed_rpm = (
            self.rotor_speed_rpm
            if rotor_speed_rpm is None
            else _finite_number(rotor_speed_rpm, "rotor_speed_rpm")
        )
        signed_speed = speed_rpm * 2.0 * math.pi / 60.0
        if self.mirror_rotor:
            signed_speed = -signed_speed
        return BladeElementActuatorLine(
            x=scales.to_execution_length(
                _finite_number(x_m, "actuator-line x_m")
            ),
            y=scales.to_execution_length(
                _finite_number(y_m, "actuator-line y_m")
            ),
            z=scales.to_execution_length(
                self.hub_height_m
                if hub_height_m is None
                else _finite_number(hub_height_m, "hub_height_m")
            ),
            blade_count=self.blade_count,
            hub_radius=scales.to_execution_length(self.hub_radius_m),
            tip_radius=scales.to_execution_length(self.tip_radius_m),
            angular_velocity=scales.to_execution_inverse_time(signed_speed),
            smoothing_width=scales.to_execution_length(
                _positive_number(smoothing_width_m, "smoothing_width_m")
            ),
            element_radii=tuple(
                scales.to_execution_length(value)
                for value in self.element_radii_m
            ),
            element_widths=tuple(
                scales.to_execution_length(value)
                for value in self.element_widths_m
            ),
            element_chords=tuple(
                scales.to_execution_length(value)
                for value in self.element_chords_m
            ),
            element_twist_degrees=self.element_twist_degrees,
            element_airfoil_ids=self.element_airfoil_ids,
            polar_alpha_degrees=self.polar_alpha_degrees,
            polar_lift_coefficients=self.polar_lift_coefficients,
            polar_drag_coefficients=self.polar_drag_coefficients,
            pitch_degrees=(
                self.pitch_degrees
                if pitch_degrees is None
                else _finite_number(pitch_degrees, "pitch_degrees")
            ),
            yaw_degrees=(
                self.yaw_degrees
                if yaw_degrees is None
                else _finite_number(yaw_degrees, "yaw_degrees")
            ),
            # OpenFAST's positive shaft tilt rises toward an upwind rotor.
            # JAX-Wind stores the downstream rotor-normal elevation.
            tilt_degrees=-self.shaft_tilt_degrees,
            precone_degrees=self.precone_degrees,
            initial_azimuth_degrees=(
                self.initial_azimuth_degrees
                if initial_azimuth_degrees is None
                else _finite_number(
                    initial_azimuth_degrees,
                    "initial_azimuth_degrees",
                )
            ),
            tip_loss=self.tip_loss if tip_loss is None else bool(tip_loss),
            root_loss=self.root_loss if root_loss is None else bool(root_loss),
        )

    def to_actuator_disk_bem(self, **kwargs) -> BladeElementActuatorDisk:
        """Lower the rigid rotor to an azimuthally averaged AD-BEM disk.

        Shaft tilt, precone, and instantaneous azimuth do not survive the
        annular average.  Blade geometry, polars, operating point, physical
        blade count, and Prandtl loss settings are identical to the ALM form.
        """

        line = self.to_actuator_line(**kwargs)
        values = {
            field.name: getattr(line, field.name)
            for field in fields(BladeElementActuatorLine)
        }
        values.update(
            tilt_degrees=0.0,
            precone_degrees=0.0,
            initial_azimuth_degrees=0.0,
        )
        return BladeElementActuatorDisk(**values)


@dataclass(frozen=True, slots=True)
class RigidBladeElementDisk:
    """A positioned SI-unit AD-BEM turbine backed by an OpenFAST rotor deck."""

    rotor: OpenFASTRigidTurbine
    x_m: float
    y_m: float
    smoothing_width_m: float
    hub_height_m: float
    rotor_speed_rpm: float | None = None
    pitch_degrees: float | None = None
    nacelle_length_m: float = 15.0
    nacelle_diameter_m: float = 6.0
    nacelle_drag_coefficient: float = 1.0
    tower_base_diameter_m: float = 8.3
    tower_top_diameter_m: float = 5.5
    tower_drag_coefficient: float = 1.0
    body_smoothing_width_m: float | None = None

    model_name = "DTU-10MW azimuthally averaged AD-BEM"

    @property
    def rotor_diameter_m(self) -> float:
        return 2.0 * self.rotor.tip_radius_m

    def to_actuator_disk(self, *, scales: ScaleSystem) -> BladeElementActuatorDisk:
        return self.rotor.to_actuator_disk_bem(
            scales=scales,
            x_m=self.x_m,
            y_m=self.y_m,
            smoothing_width_m=self.smoothing_width_m,
            hub_height_m=self.hub_height_m,
            rotor_speed_rpm=self.rotor_speed_rpm,
            pitch_degrees=self.pitch_degrees,
            yaw_degrees=0.0,
        )

    def to_nacelle_tower(self, *, scales: ScaleSystem) -> NacelleTowerDrag:
        """Lower the configured nacelle and tower to solver execution units."""

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
            smoothing_width=scales.to_execution_length(
                self.smoothing_width_m
                if self.body_smoothing_width_m is None
                else self.body_smoothing_width_m
            ),
        )


@dataclass(frozen=True, slots=True)
class _BladeData:
    source: Path
    spans: tuple[float, ...]
    chords: tuple[float, ...]
    twists: tuple[float, ...]
    airfoil_ids: tuple[int, ...]
    curve_offsets: tuple[float, ...]
    sweep_offsets: tuple[float, ...]
    curve_angles: tuple[float, ...]

    def aerodynamic_signature(self) -> tuple[tuple[float, ...], ...]:
        return (
            self.spans,
            self.chords,
            self.twists,
            tuple(float(value) for value in self.airfoil_ids),
            self.curve_offsets,
            self.sweep_offsets,
            self.curve_angles,
        )


def _collect_paths(
    lines: tuple[_Line, ...],
    key: str,
    *,
    count: int,
    path: Path,
) -> tuple[Path, ...]:
    first, line_index = _value(lines, key, path)
    tokens = [first]
    for line in lines[line_index + 1 :]:
        if len(tokens) == count:
            break
        if len(line.tokens) == 1:
            tokens.append(line.tokens[0])
        else:
            break
    if len(tokens) != count:
        raise OpenFASTInputError(
            f"{path}: {key} declares {count} paths but only "
            f"{len(tokens)} were found"
        )
    resolved = []
    for token in tokens:
        candidate = Path(token)
        resolved.append(
            candidate
            if candidate.is_absolute()
            else (path.parent / candidate).resolve()
        )
    return tuple(resolved)


def _blade_column(
    headers: tuple[str, ...],
    *names: str,
) -> int | None:
    normalized = tuple(header.casefold() for header in headers)
    for name in names:
        try:
            return normalized.index(name.casefold())
        except ValueError:
            pass
    return None


def _load_blade(path: Path) -> _BladeData:
    lines = _read_lines(path)
    count = _integer_value(lines, "NumBlNds", path)
    if count < 2:
        raise OpenFASTInputError(f"{path}: NumBlNds must be at least two")
    _, count_line = _value(lines, "NumBlNds", path)
    required = ("BlSpn", "BlTwist", "BlChord", "BlAFID")
    header_index: int | None = None
    for index, line in enumerate(lines[count_line + 1 :], start=count_line + 1):
        folded = {token.casefold() for token in line.tokens}
        if all(name.casefold() in folded for name in required):
            header_index = index
            break
    if header_index is None:
        raise OpenFASTInputError(f"{path}: AeroDyn blade column header is missing")
    headers = lines[header_index].tokens
    indices = {
        "span": _blade_column(headers, "BlSpn"),
        "twist": _blade_column(headers, "BlTwist", "BlTwst"),
        "chord": _blade_column(headers, "BlChord"),
        "airfoil": _blade_column(headers, "BlAFID"),
        "curve": _blade_column(headers, "BlCrvAC"),
        "sweep": _blade_column(headers, "BlSwpAC"),
        "curve_angle": _blade_column(headers, "BlCrvAng"),
    }
    required_indices = tuple(
        indices[name] for name in ("span", "twist", "chord", "airfoil")
    )
    if any(index is None for index in required_indices):
        raise OpenFASTInputError(f"{path}: required AeroDyn blade columns are missing")
    column_count = max(
        index for index in indices.values() if index is not None
    ) + 1
    rows = _rows_after(
        lines,
        header_index,
        count=count,
        columns=column_count,
        path=path,
        context="blade-node",
    )

    def values(name: str, default: float = 0.0) -> tuple[float, ...]:
        index = indices[name]
        return (
            tuple(default for _ in rows)
            if index is None
            else tuple(row[index] for row in rows)
        )

    airfoil_values = values("airfoil")
    airfoil_ids = tuple(int(value) for value in airfoil_values)
    if any(float(index) != value for index, value in zip(airfoil_ids, airfoil_values)):
        raise OpenFASTInputError(f"{path}: BlAFID values must be integers")
    return _BladeData(
        source=path,
        spans=values("span"),
        chords=values("chord"),
        twists=values("twist"),
        airfoil_ids=airfoil_ids,
        curve_offsets=values("curve"),
        sweep_offsets=values("sweep"),
        curve_angles=values("curve_angle"),
    )


def _load_airfoil(
    path: Path,
    *,
    alpha_column: int,
    lift_column: int,
    drag_column: int,
) -> OpenFASTAirfoilPolar:
    lines = _read_lines(path)
    interpolation, _ = _value(lines, "InterpOrd", path)
    if interpolation.casefold() not in ("default", "1"):
        raise OpenFASTInputError(
            f"{path}: InterpOrd={interpolation!r} is not compatible with "
            "JAX-Wind's piecewise-linear polar interpolation"
        )
    tables = _integer_value(lines, "NumTabs", path)
    if tables < 1:
        raise OpenFASTInputError(f"{path}: NumTabs must be positive")
    count = _integer_value(lines, "NumAlf", path)
    if count < 1:
        raise OpenFASTInputError(f"{path}: NumAlf must be positive")
    _, count_line = _value(lines, "NumAlf", path)
    columns = max(alpha_column, lift_column, drag_column) + 1
    rows = _rows_after(
        lines,
        count_line,
        count=count,
        columns=columns,
        path=path,
        context="airfoil-polar",
    )
    alpha = tuple(row[alpha_column] for row in rows)
    lift = tuple(row[lift_column] for row in rows)
    drag = tuple(row[drag_column] for row in rows)
    if len(alpha) == 1:
        alpha = (-180.0, 180.0)
        lift = (lift[0], lift[0])
        drag = (drag[0], drag[0])
    if any(right <= left for left, right in zip(alpha, alpha[1:])):
        raise OpenFASTInputError(
            f"{path}: first airfoil table angles must be strictly increasing"
        )
    if any(value < 0.0 for value in drag):
        raise OpenFASTInputError(f"{path}: drag coefficients must be nonnegative")
    return OpenFASTAirfoilPolar(path, alpha, lift, drag)


def _resample(
    source_x: tuple[float, ...],
    source_y: tuple[float, ...],
    target_x: tuple[float, ...],
) -> tuple[float, ...]:
    values = []
    for target in target_x:
        if target <= source_x[0]:
            values.append(source_y[0])
            continue
        if target >= source_x[-1]:
            values.append(source_y[-1])
            continue
        upper = bisect_right(source_x, target)
        lower = upper - 1
        fraction = (target - source_x[lower]) / (
            source_x[upper] - source_x[lower]
        )
        values.append(
            source_y[lower] + fraction * (source_y[upper] - source_y[lower])
        )
    return tuple(values)


def _element_widths(
    spans: tuple[float, ...],
    blade_span: float,
    path: Path,
) -> tuple[float, ...]:
    if any(right <= left for left, right in zip(spans, spans[1:])):
        raise OpenFASTInputError(f"{path}: BlSpn values must be strictly increasing")
    tolerance = max(1.0e-8, 1.0e-6 * blade_span)
    if abs(spans[0]) > tolerance or abs(spans[-1] - blade_span) > tolerance:
        raise OpenFASTInputError(
            f"{path}: BlSpn must begin at 0 and end at TipRad-HubRad "
            f"({blade_span:g} m)"
        )
    boundaries = (
        (0.0,)
        + tuple(
            0.5 * (left + right)
            for left, right in zip(spans, spans[1:])
        )
        + (blade_span,)
    )
    return tuple(
        right - left for left, right in zip(boundaries, boundaries[1:])
    )


def _equal_operating_values(
    lines: tuple[_Line, ...],
    stem: str,
    count: int,
    path: Path,
) -> float:
    values = tuple(
        _float_value(lines, f"{stem}({index})", path)
        for index in range(1, count + 1)
    )
    if any(not math.isclose(value, values[0], abs_tol=1.0e-12) for value in values[1:]):
        raise OpenFASTInputError(
            f"{path}: per-blade {stem} values differ; the first rigid model "
            "requires identical blades"
        )
    return values[0]


def load_openfast_rigid_turbine(
    input_file: str | Path,
) -> OpenFASTRigidTurbine:
    """Resolve an OpenFAST input deck into a rigid JAX-Wind turbine.

    The supported compatibility surface is an OpenFAST primary ``.fst`` file
    referencing ElastoDyn and AeroDyn v15-style input, blade, and airfoil
    files.  AeroDyn's first polar table is used, matching ``AFTabMod = 1``.
    """

    source = Path(input_file).resolve()
    fst = _read_lines(source)
    number_of_rotors = _optional_integer_value(
        fst,
        "NRotors",
        source,
        default=1,
    )
    if number_of_rotors != 1:
        raise OpenFASTInputError(
            f"{source}: only one OpenFAST rotor is supported, got {number_of_rotors}"
        )
    aerodyn_path = _path_value(fst, "AeroFile", source)
    elastodyn_path = _path_value(fst, "EDFile", source)
    mirror_rotor = _optional_boolean_value(
        fst,
        "MirrorRotor",
        source,
        default=False,
    )

    elastodyn = _read_lines(elastodyn_path)
    blade_count = _integer_value(elastodyn, "NumBl", elastodyn_path)
    if blade_count < 1:
        raise OpenFASTInputError(f"{elastodyn_path}: NumBl must be positive")
    hub_radius = _float_value(elastodyn, "HubRad", elastodyn_path)
    tip_radius = _float_value(elastodyn, "TipRad", elastodyn_path)
    if hub_radius < 0.0 or tip_radius <= hub_radius:
        raise OpenFASTInputError(
            f"{elastodyn_path}: radii must satisfy 0 <= HubRad < TipRad"
        )
    rotor_speed = _float_value(elastodyn, "RotSpeed", elastodyn_path)
    pitch = _equal_operating_values(
        elastodyn,
        "BlPitch",
        blade_count,
        elastodyn_path,
    )
    precone = _equal_operating_values(
        elastodyn,
        "PreCone",
        blade_count,
        elastodyn_path,
    )
    yaw = _float_value(elastodyn, "NacYaw", elastodyn_path)
    shaft_tilt = _float_value(elastodyn, "ShftTilt", elastodyn_path)
    azimuth = _float_value(elastodyn, "Azimuth", elastodyn_path)
    azimuth_blade_one_up = _float_value(
        elastodyn,
        "AzimB1Up",
        elastodyn_path,
    )
    tower_height = _float_value(elastodyn, "TowerHt", elastodyn_path)
    tower_to_shaft = _float_value(elastodyn, "Twr2Shft", elastodyn_path)
    overhang = _float_value(elastodyn, "OverHang", elastodyn_path)
    hub_height = (
        tower_height
        + tower_to_shaft
        - overhang * math.sin(math.radians(shaft_tilt))
    )

    aerodyn = _read_lines(aerodyn_path)
    aftab_mode = _integer_value(aerodyn, "AFTabMod", aerodyn_path)
    if aftab_mode != 1:
        raise OpenFASTInputError(
            f"{aerodyn_path}: AFTabMod={aftab_mode} is unsupported; "
            "the rigid model currently uses one polar table per airfoil"
        )
    alpha_column = _integer_value(aerodyn, "InCol_Alfa", aerodyn_path) - 1
    lift_column = _integer_value(aerodyn, "InCol_Cl", aerodyn_path) - 1
    drag_column = _integer_value(aerodyn, "InCol_Cd", aerodyn_path) - 1
    if min(alpha_column, lift_column, drag_column) < 0:
        raise OpenFASTInputError(
            f"{aerodyn_path}: Alfa, Cl, and Cd input columns must be positive"
        )
    airfoil_count = _integer_value(aerodyn, "NumAFfiles", aerodyn_path)
    if airfoil_count < 1:
        raise OpenFASTInputError(f"{aerodyn_path}: NumAFfiles must be positive")
    airfoil_paths = _collect_paths(
        aerodyn,
        "AFNames",
        count=airfoil_count,
        path=aerodyn_path,
    )
    blade_paths = tuple(
        _path_value(aerodyn, f"ADBlFile({index})", aerodyn_path)
        for index in range(1, blade_count + 1)
    )
    blades = tuple(_load_blade(path) for path in blade_paths)
    signature = blades[0].aerodynamic_signature()
    if any(blade.aerodynamic_signature() != signature for blade in blades[1:]):
        raise OpenFASTInputError(
            f"{aerodyn_path}: blade files differ; the first rigid model "
            "requires identical blade geometry and airfoil assignment"
        )
    blade = blades[0]
    if min(blade.chords) <= 0.0:
        raise OpenFASTInputError(f"{blade.source}: BlChord values must be positive")
    if any(
        index < 1 or index > airfoil_count for index in blade.airfoil_ids
    ):
        raise OpenFASTInputError(
            f"{blade.source}: BlAFID lies outside the AeroDyn AFNames table"
        )

    polars = tuple(
        _load_airfoil(
            path,
            alpha_column=alpha_column,
            lift_column=lift_column,
            drag_column=drag_column,
        )
        for path in airfoil_paths
    )
    alpha_grid = tuple(
        sorted({value for polar in polars for value in polar.alpha_degrees})
    )
    lift_tables = tuple(
        _resample(
            polar.alpha_degrees,
            polar.lift_coefficients,
            alpha_grid,
        )
        for polar in polars
    )
    drag_tables = tuple(
        _resample(
            polar.alpha_degrees,
            polar.drag_coefficients,
            alpha_grid,
        )
        for polar in polars
    )

    wake_mode = _integer_value(aerodyn, "Wake_Mod", aerodyn_path)
    unsteady_mode = _integer_value(aerodyn, "UA_Mod", aerodyn_path)
    notes = [
        "ElastoDyn structural flexibility and ServoDyn control are not advanced; "
        "the imported rotor speed, pitch, yaw, and geometry remain rigid.",
    ]
    if wake_mode != 0:
        notes.append(
            f"AeroDyn Wake_Mod={wake_mode} is not reproduced; resolved LES "
            "velocity is sampled directly at the actuator line."
        )
    if unsteady_mode != 0:
        notes.append(
            f"AeroDyn UA_Mod={unsteady_mode} is not reproduced; Cl/Cd use "
            "quasi-steady piecewise-linear interpolation."
        )
    if any(
        abs(value) > 1.0e-12
        for values in (
            blade.curve_offsets,
            blade.sweep_offsets,
            blade.curve_angles,
        )
        for value in values
    ):
        notes.append(
            "BlCrvAC, BlSwpAC, and BlCrvAng are parsed but not yet represented; "
            "the imported actuator lines follow the preconed pitch axes."
        )
    return OpenFASTRigidTurbine(
        source=source,
        aerodyn_source=aerodyn_path,
        elastodyn_source=elastodyn_path,
        blade_source=blade.source,
        airfoil_sources=airfoil_paths,
        blade_count=blade_count,
        hub_radius_m=hub_radius,
        tip_radius_m=tip_radius,
        hub_height_m=hub_height,
        rotor_speed_rpm=rotor_speed,
        pitch_degrees=pitch,
        yaw_degrees=yaw,
        shaft_tilt_degrees=shaft_tilt,
        precone_degrees=precone,
        initial_azimuth_degrees=azimuth - azimuth_blade_one_up,
        mirror_rotor=mirror_rotor,
        element_radii_m=tuple(
            hub_radius + span for span in blade.spans
        ),
        element_widths_m=_element_widths(
            blade.spans,
            tip_radius - hub_radius,
            blade.source,
        ),
        element_chords_m=blade.chords,
        element_twist_degrees=blade.twists,
        element_airfoil_ids=tuple(index - 1 for index in blade.airfoil_ids),
        polar_alpha_degrees=alpha_grid,
        polar_lift_coefficients=lift_tables,
        polar_drag_coefficients=drag_tables,
        tip_loss=_boolean_value(aerodyn, "TipLoss", aerodyn_path),
        root_loss=_boolean_value(aerodyn, "HubLoss", aerodyn_path),
        compatibility_notes=tuple(notes),
    )
