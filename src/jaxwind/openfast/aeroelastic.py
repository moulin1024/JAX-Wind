"""OpenFAST-compatible small-deflection modal blade aeroelasticity.

This module implements the ElastoDyn blade compatibility surface: two
flapwise polynomial modes and one edgewise polynomial mode per blade.  It
uses the distributed mass and bending-stiffness tables from ``BldFile`` and
advances the modal coordinates with an average-acceleration Newmark scheme.

It is intentionally not a reimplementation of the complete ElastoDyn or
BeamDyn multibody systems.  Rotor speed, hub, nacelle, tower, drivetrain,
pitch, and yaw remain prescribed by the rigid turbine adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

import numpy as np

from jaxwind.domain import ScaleSystem
from jaxwind.physics import BladeElementActuatorLine

from .errors import OpenFASTInputError
from .parser import (
    _boolean_value,
    _find,
    _float_value,
    _path_value,
    _read_lines,
    _rows_after,
)
from .rigid import OpenFASTRigidTurbine, load_openfast_rigid_turbine


_MODE_NAMES = ("flap1", "flap2", "edge1")
_MODE_DIRECTIONS = ("flap", "flap", "edge")


@dataclass(frozen=True, slots=True)
class OpenFASTBladeStructure:
    """One ElastoDyn individual-blade structural input file."""

    source: Path
    station_fractions: tuple[float, ...]
    structural_twist_degrees: tuple[float, ...]
    mass_density_kg_m: tuple[float, ...]
    flap_stiffness_n_m2: tuple[float, ...]
    edge_stiffness_n_m2: tuple[float, ...]
    flap_damping_ratios: tuple[float, float]
    edge_damping_ratio: float
    flap_stiffness_tuners: tuple[float, float]
    mass_tuner: float
    flap_stiffness_tuner: float
    edge_stiffness_tuner: float
    flap_mode_coefficients: tuple[tuple[float, ...], tuple[float, ...]]
    edge_mode_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.station_fractions)
        distributed = (
            self.structural_twist_degrees,
            self.mass_density_kg_m,
            self.flap_stiffness_n_m2,
            self.edge_stiffness_n_m2,
        )
        if count < 2 or any(len(values) != count for values in distributed):
            raise ValueError("ElastoDyn blade tables must have equal lengths")
        if (
            self.station_fractions[0] != 0.0
            or self.station_fractions[-1] != 1.0
            or any(
                right <= left
                for left, right in zip(
                    self.station_fractions,
                    self.station_fractions[1:],
                )
            )
        ):
            raise ValueError(
                "ElastoDyn BlFract must increase strictly from zero to one"
            )
        if min(self.mass_density_kg_m) <= 0.0:
            raise ValueError("ElastoDyn blade mass density must be positive")
        if min(self.flap_stiffness_n_m2) <= 0.0:
            raise ValueError("ElastoDyn flap stiffness must be positive")
        if min(self.edge_stiffness_n_m2) <= 0.0:
            raise ValueError("ElastoDyn edge stiffness must be positive")

    def signature(self) -> tuple[Any, ...]:
        """Return all physical data used to require identical blades."""

        return (
            self.station_fractions,
            self.structural_twist_degrees,
            self.mass_density_kg_m,
            self.flap_stiffness_n_m2,
            self.edge_stiffness_n_m2,
            self.flap_damping_ratios,
            self.edge_damping_ratio,
            self.flap_stiffness_tuners,
            self.mass_tuner,
            self.flap_stiffness_tuner,
            self.edge_stiffness_tuner,
            self.flap_mode_coefficients,
            self.edge_mode_coefficients,
        )


@dataclass(frozen=True, slots=True)
class OpenFASTModalTurbine:
    """Rigid aerodynamic data plus supported ElastoDyn blade modes."""

    rigid: OpenFASTRigidTurbine
    blade_sources: tuple[Path, ...]
    blade_structure: OpenFASTBladeStructure
    enabled_modes: tuple[bool, bool, bool]
    initial_out_of_plane_tip_deflection_m: float
    initial_in_plane_tip_deflection_m: float
    compatibility_notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModalBladeState:
    """Per-blade generalized displacements, rates, and accelerations."""

    displacement_m: np.ndarray
    velocity_m_s: np.ndarray
    acceleration_m_s2: np.ndarray


@dataclass(frozen=True, slots=True)
class ModalBladeDiagnostics:
    """Small structural output set suitable for runner histories."""

    generalized_force_n: np.ndarray
    flap_tip_deflection_m: np.ndarray
    edge_tip_deflection_m: np.ndarray
    maximum_tip_deflection_m: float


@dataclass(frozen=True, slots=True)
class ModalBladeModel:
    """Linear ElastoDyn-style blade modes at a prescribed rotor speed."""

    blade_count: int
    mode_names: tuple[str, ...]
    mode_directions: tuple[str, ...]
    element_mode_shapes: np.ndarray
    element_mode_slopes_per_m: np.ndarray
    element_structural_twist_radians: np.ndarray
    element_structural_twist_slopes_per_m: np.ndarray
    element_masses_kg: np.ndarray
    mass_matrix_kg: np.ndarray
    damping_matrix_n_s_m: np.ndarray
    stiffness_matrix_n_m: np.ndarray
    natural_frequencies_hz: np.ndarray
    initial_out_of_plane_tip_deflection_m: float
    initial_in_plane_tip_deflection_m: float

    @property
    def mode_count(self) -> int:
        return len(self.mode_names)

    def initial_state(self) -> ModalBladeState:
        """Construct the OpenFAST tip-deflection initial condition."""

        displacement = np.zeros((self.blade_count, self.mode_count))
        for mode_index, name in enumerate(self.mode_names):
            if name == "flap1":
                displacement[:, mode_index] = (
                    self.initial_out_of_plane_tip_deflection_m
                )
            elif name == "edge1":
                displacement[:, mode_index] = (
                    self.initial_in_plane_tip_deflection_m
                )
        velocity = np.zeros_like(displacement)
        acceleration = np.stack(
            [
                np.linalg.solve(
                    self.mass_matrix_kg,
                    -self.stiffness_matrix_n_m @ blade_displacement,
                )
                for blade_displacement in displacement
            ]
        )
        return ModalBladeState(displacement, velocity, acceleration)

    def deformation_fields(
        self,
        state: ModalBladeState,
    ) -> dict[str, np.ndarray]:
        """Evaluate modal displacement, slope, and rate at AeroDyn nodes."""

        flap_indices = tuple(
            index
            for index, direction in enumerate(self.mode_directions)
            if direction == "flap"
        )
        edge_indices = tuple(
            index
            for index, direction in enumerate(self.mode_directions)
            if direction == "edge"
        )

        def modal_sum(values: np.ndarray, indices: tuple[int, ...], shapes):
            if not indices:
                return np.zeros(
                    (self.blade_count, self.element_mode_shapes.shape[1])
                )
            return values[:, indices] @ shapes[np.asarray(indices)]

        structural_flap = modal_sum(
            state.displacement_m,
            flap_indices,
            self.element_mode_shapes,
        )
        structural_edge = modal_sum(
            state.displacement_m,
            edge_indices,
            self.element_mode_shapes,
        )
        structural_flap_slope = modal_sum(
            state.displacement_m,
            flap_indices,
            self.element_mode_slopes_per_m,
        )
        structural_edge_slope = modal_sum(
            state.displacement_m,
            edge_indices,
            self.element_mode_slopes_per_m,
        )
        structural_flap_rate = modal_sum(
            state.velocity_m_s,
            flap_indices,
            self.element_mode_shapes,
        )
        structural_edge_rate = modal_sum(
            state.velocity_m_s,
            edge_indices,
            self.element_mode_shapes,
        )

        twist = self.element_structural_twist_radians[None, :]
        twist_slope = self.element_structural_twist_slopes_per_m[None, :]
        cosine = np.cos(twist)
        sine = np.sin(twist)
        flap = cosine * structural_flap - sine * structural_edge
        edge = sine * structural_flap + cosine * structural_edge
        flap_slope = (
            cosine * structural_flap_slope
            - sine * structural_edge_slope
            - twist_slope
            * (sine * structural_flap + cosine * structural_edge)
        )
        edge_slope = (
            sine * structural_flap_slope
            + cosine * structural_edge_slope
            + twist_slope
            * (cosine * structural_flap - sine * structural_edge)
        )
        flap_rate = (
            cosine * structural_flap_rate - sine * structural_edge_rate
        )
        edge_rate = (
            sine * structural_flap_rate + cosine * structural_edge_rate
        )
        return {
            "flap_displacements_m": flap,
            "edge_displacements_m": edge,
            "flap_slopes": flap_slope,
            "edge_slopes": edge_slope,
            "flap_velocities_m_s": flap_rate,
            "edge_velocities_m_s": edge_rate,
        }

    def deform_actuator_line(
        self,
        line: BladeElementActuatorLine,
        state: ModalBladeState,
        *,
        scales: ScaleSystem,
    ) -> BladeElementActuatorLine:
        """Attach the current modal fields to an execution-unit line."""

        fields = self.deformation_fields(state)

        def flattened(name: str, conversion) -> tuple[float, ...]:
            values = conversion(fields[name]).reshape(-1)
            return tuple(float(value) for value in values)

        return replace(
            line,
            element_flap_displacements=flattened(
                "flap_displacements_m",
                scales.to_execution_length,
            ),
            element_edge_displacements=flattened(
                "edge_displacements_m",
                scales.to_execution_length,
            ),
            element_flap_slopes=flattened(
                "flap_slopes",
                lambda values: values,
            ),
            element_edge_slopes=flattened(
                "edge_slopes",
                lambda values: values,
            ),
            element_flap_velocities=flattened(
                "flap_velocities_m_s",
                scales.to_execution_velocity,
            ),
            element_edge_velocities=flattened(
                "edge_velocities_m_s",
                scales.to_execution_velocity,
            ),
        )

    def generalized_forces(
        self,
        force_on_blade_n: np.ndarray,
        section_normals: np.ndarray,
        section_tangents: np.ndarray,
        *,
        gravity_m_s2: float,
    ) -> np.ndarray:
        """Project aerodynamic and gravity loads onto generalized modes."""

        element_count = self.element_mode_shapes.shape[1]
        forces = np.asarray(force_on_blade_n).reshape(
            (self.blade_count, element_count, 3)
        )
        normals = np.asarray(section_normals).reshape(
            (self.blade_count, element_count, 3)
        )
        tangents = np.asarray(section_tangents).reshape(
            (self.blade_count, element_count, 3)
        )
        gravity = np.asarray((0.0, 0.0, -gravity_m_s2))
        forces = forces + self.element_masses_kg[None, :, None] * gravity
        rotor_flap_force = np.sum(forces * normals, axis=2)
        rotor_edge_force = np.sum(forces * tangents, axis=2)
        twist = self.element_structural_twist_radians[None, :]
        cosine = np.cos(twist)
        sine = np.sin(twist)
        structural_flap_force = (
            cosine * rotor_flap_force + sine * rotor_edge_force
        )
        structural_edge_force = (
            -sine * rotor_flap_force + cosine * rotor_edge_force
        )
        result = np.empty((self.blade_count, self.mode_count))
        for mode_index, direction in enumerate(self.mode_directions):
            sectional = (
                structural_flap_force
                if direction == "flap"
                else structural_edge_force
            )
            result[:, mode_index] = np.sum(
                sectional * self.element_mode_shapes[mode_index][None, :],
                axis=1,
            )
        return result

    def advance(
        self,
        state: ModalBladeState,
        generalized_force_n: np.ndarray,
        *,
        dt_seconds: float,
    ) -> tuple[ModalBladeState, ModalBladeDiagnostics]:
        """Advance all blade modes with average-acceleration Newmark."""

        beta = 0.25
        gamma = 0.5
        dt = float(dt_seconds)
        q_predict = (
            state.displacement_m
            + dt * state.velocity_m_s
            + dt * dt * (0.5 - beta) * state.acceleration_m_s2
        )
        v_predict = (
            state.velocity_m_s
            + dt * (1.0 - gamma) * state.acceleration_m_s2
        )
        effective = (
            self.mass_matrix_kg
            + gamma * dt * self.damping_matrix_n_s_m
            + beta * dt * dt * self.stiffness_matrix_n_m
        )
        acceleration = np.stack(
            [
                np.linalg.solve(
                    effective,
                    force
                    - self.damping_matrix_n_s_m @ velocity
                    - self.stiffness_matrix_n_m @ displacement,
                )
                for force, velocity, displacement in zip(
                    generalized_force_n,
                    v_predict,
                    q_predict,
                )
            ]
        )
        displacement = q_predict + beta * dt * dt * acceleration
        velocity = v_predict + gamma * dt * acceleration
        accepted = ModalBladeState(displacement, velocity, acceleration)
        flap = np.zeros(self.blade_count)
        edge = np.zeros(self.blade_count)
        for index, direction in enumerate(self.mode_directions):
            if direction == "flap":
                flap += displacement[:, index]
            else:
                edge += displacement[:, index]
        diagnostic = ModalBladeDiagnostics(
            np.asarray(generalized_force_n),
            flap,
            edge,
            float(np.max(np.sqrt(flap * flap + edge * edge))),
        )
        return accepted, diagnostic


def _optional_float(lines, key: str, path: Path, default: float) -> float:
    try:
        _find(lines, key, path)
    except OpenFASTInputError:
        return default
    return _float_value(lines, key, path)


def _load_structure(path: Path) -> OpenFASTBladeStructure:
    lines = _read_lines(path)
    count_value = _float_value(lines, "NBlInpSt", path)
    count = int(count_value)
    if count_value != count or count < 2:
        raise OpenFASTInputError(f"{path}: NBlInpSt must be an integer >= 2")
    _, count_line = next(
        (line.number, index)
        for index, line in enumerate(lines)
        if any(token.casefold() == "nblinpst" for token in line.tokens)
    )
    header_index: int | None = None
    required = {
        "blfract",
        "strctwst",
        "bmassden",
        "flpstff",
        "edgstff",
    }
    for index, line in enumerate(lines[count_line + 1 :], start=count_line + 1):
        if required.issubset({token.casefold() for token in line.tokens}):
            header_index = index
            break
    if header_index is None:
        raise OpenFASTInputError(
            f"{path}: distributed blade property header is missing"
        )
    headers = tuple(token.casefold() for token in lines[header_index].tokens)
    indices = tuple(headers.index(name) for name in required)
    rows = _rows_after(
        lines,
        header_index,
        count=count,
        columns=max(indices) + 1,
        path=path,
        context="ElastoDyn blade-property",
    )

    def column(name: str) -> tuple[float, ...]:
        index = headers.index(name)
        return tuple(row[index] for row in rows)

    def coefficients(stem: str) -> tuple[float, ...]:
        return tuple(
            _float_value(lines, f"{stem}({power})", path)
            for power in range(2, 7)
        )

    mass_tuner = _optional_float(lines, "AdjBlMs", path, 1.0)
    flap_tuner = _optional_float(lines, "AdjFlSt", path, 1.0)
    edge_tuner = _optional_float(lines, "AdjEdSt", path, 1.0)
    if min(mass_tuner, flap_tuner, edge_tuner) <= 0.0:
        raise OpenFASTInputError(
            f"{path}: blade adjustment factors must be positive"
        )
    return OpenFASTBladeStructure(
        source=path,
        station_fractions=column("blfract"),
        structural_twist_degrees=column("strctwst"),
        mass_density_kg_m=column("bmassden"),
        flap_stiffness_n_m2=column("flpstff"),
        edge_stiffness_n_m2=column("edgstff"),
        flap_damping_ratios=(
            _float_value(lines, "BldFlDmp(1)", path) / 100.0,
            _float_value(lines, "BldFlDmp(2)", path) / 100.0,
        ),
        edge_damping_ratio=(
            _float_value(lines, "BldEdDmp(1)", path) / 100.0
        ),
        flap_stiffness_tuners=(
            _optional_float(lines, "FlStTunr(1)", path, 1.0),
            _optional_float(lines, "FlStTunr(2)", path, 1.0),
        ),
        mass_tuner=mass_tuner,
        flap_stiffness_tuner=flap_tuner,
        edge_stiffness_tuner=edge_tuner,
        flap_mode_coefficients=(
            coefficients("BldFl1Sh"),
            coefficients("BldFl2Sh"),
        ),
        edge_mode_coefficients=coefficients("BldEdgSh"),
    )


def load_openfast_modal_turbine(
    input_file: str | Path,
) -> OpenFASTModalTurbine:
    """Load supported ElastoDyn blade modes through an OpenFAST primary file."""

    rigid = load_openfast_rigid_turbine(input_file)
    path = rigid.elastodyn_source
    lines = _read_lines(path)
    enabled = (
        _boolean_value(lines, "FlapDOF1", path),
        _boolean_value(lines, "FlapDOF2", path),
        _boolean_value(lines, "EdgeDOF", path),
    )
    if not any(enabled):
        raise OpenFASTInputError(
            f"{path}: modal coupling requires FlapDOF1, FlapDOF2, "
            "or EdgeDOF"
        )
    blade_sources = tuple(
        _path_value(lines, f"BldFile({index})", path)
        for index in range(1, rigid.blade_count + 1)
    )
    structures = tuple(_load_structure(source) for source in blade_sources)
    signature = structures[0].signature()
    if any(structure.signature() != signature for structure in structures[1:]):
        raise OpenFASTInputError(
            f"{path}: modal coupling currently requires identical "
            "ElastoDyn blade files"
        )
    notes = (
        "ElastoDyn blade FlapDOF1, FlapDOF2, and EdgeDOF flags are "
        "supported with small-deflection modal coupling.",
        "Tower, platform, drivetrain, generator, pitch, yaw, teeter, "
        "ServoDyn, and BeamDyn degrees of freedom remain prescribed.",
        "Coupling is partitioned and explicit at accepted CFD steps; "
        "OpenFAST tight-coupling iterations are not reproduced.",
    )
    return OpenFASTModalTurbine(
        rigid=rigid,
        blade_sources=blade_sources,
        blade_structure=structures[0],
        enabled_modes=enabled,
        initial_out_of_plane_tip_deflection_m=_optional_float(
            lines,
            "OoPDefl",
            path,
            0.0,
        ),
        initial_in_plane_tip_deflection_m=_optional_float(
            lines,
            "IPDefl",
            path,
            0.0,
        ),
        compatibility_notes=notes,
    )


def _shape(coefficients: tuple[float, ...], fraction: np.ndarray) -> np.ndarray:
    return sum(
        coefficient * fraction**power
        for power, coefficient in enumerate(coefficients, start=2)
    )


def _shape_derivative(
    coefficients: tuple[float, ...],
    fraction: np.ndarray,
    span_m: float,
) -> np.ndarray:
    return sum(
        power * coefficient * fraction ** (power - 1)
        for power, coefficient in enumerate(coefficients, start=2)
    ) / span_m


def _shape_second_derivative(
    coefficients: tuple[float, ...],
    fraction: np.ndarray,
    span_m: float,
) -> np.ndarray:
    return sum(
        power
        * (power - 1)
        * coefficient
        * fraction ** (power - 2)
        for power, coefficient in enumerate(coefficients, start=2)
    ) / (span_m * span_m)


def _trapz(values: np.ndarray, coordinate: np.ndarray) -> float:
    return float(np.trapezoid(values, coordinate))


def build_modal_blade_model(
    turbine: OpenFASTModalTurbine,
    *,
    element_radii_m: tuple[float, ...],
    element_widths_m: tuple[float, ...],
    rotor_speed_rpm: float,
) -> ModalBladeModel:
    """Reduce distributed ElastoDyn blade properties to active modal DOFs."""

    rigid = turbine.rigid
    structure = turbine.blade_structure
    span_m = rigid.tip_radius_m - rigid.hub_radius_m
    integration_fraction = np.linspace(0.0, 1.0, 2001)
    integration_span = integration_fraction * span_m
    source_fraction = np.asarray(structure.station_fractions)
    mass_density = (
        np.interp(
            integration_fraction,
            source_fraction,
            structure.mass_density_kg_m,
        )
        * structure.mass_tuner
    )
    flap_stiffness = (
        np.interp(
            integration_fraction,
            source_fraction,
            structure.flap_stiffness_n_m2,
        )
        * structure.flap_stiffness_tuner
    )
    edge_stiffness = (
        np.interp(
            integration_fraction,
            source_fraction,
            structure.edge_stiffness_n_m2,
        )
        * structure.edge_stiffness_tuner
    )
    coefficient_sets = (
        structure.flap_mode_coefficients[0],
        structure.flap_mode_coefficients[1],
        structure.edge_mode_coefficients,
    )
    damping = (
        structure.flap_damping_ratios[0],
        structure.flap_damping_ratios[1],
        structure.edge_damping_ratio,
    )
    stiffness_tuners = (
        structure.flap_stiffness_tuners[0],
        structure.flap_stiffness_tuners[1],
        1.0,
    )
    active_indices = tuple(
        index for index, enabled in enumerate(turbine.enabled_modes) if enabled
    )
    shapes = np.stack(
        [_shape(coefficient_sets[index], integration_fraction) for index in active_indices]
    )
    slopes = np.stack(
        [
            _shape_derivative(
                coefficient_sets[index],
                integration_fraction,
                span_m,
            )
            for index in active_indices
        ]
    )
    curvatures = np.stack(
        [
            _shape_second_derivative(
                coefficient_sets[index],
                integration_fraction,
                span_m,
            )
            for index in active_indices
        ]
    )
    directions = tuple(_MODE_DIRECTIONS[index] for index in active_indices)
    count = len(active_indices)
    mass_matrix = np.zeros((count, count))
    stiffness_matrix = np.zeros((count, count))
    omega = float(rotor_speed_rpm) * 2.0 * math.pi / 60.0
    centrifugal_integrand = (
        mass_density
        * omega**2
        * (rigid.hub_radius_m + integration_span)
    )
    centrifugal_tension = np.zeros_like(centrifugal_integrand)
    increments = 0.5 * (
        centrifugal_integrand[:-1] + centrifugal_integrand[1:]
    ) * np.diff(integration_span)
    centrifugal_tension[:-1] = np.cumsum(increments[::-1])[::-1]
    for left in range(count):
        for right in range(count):
            if directions[left] != directions[right]:
                continue
            mass_matrix[left, right] = _trapz(
                mass_density * shapes[left] * shapes[right],
                integration_span,
            )
            bending_stiffness = (
                flap_stiffness
                if directions[left] == "flap"
                else edge_stiffness
            )
            elastic = _trapz(
                bending_stiffness
                * curvatures[left]
                * curvatures[right],
                integration_span,
            )
            centrifugal = _trapz(
                centrifugal_tension * slopes[left] * slopes[right],
                integration_span,
            )
            stiffness_matrix[left, right] = elastic + centrifugal
    for local_index, source_index in enumerate(active_indices):
        stiffness_matrix[local_index, local_index] *= (
            stiffness_tuners[source_index]
        )
    damping_matrix = np.zeros_like(stiffness_matrix)
    for local_index, source_index in enumerate(active_indices):
        damping_matrix[local_index, local_index] = (
            2.0
            * damping[source_index]
            * math.sqrt(
                mass_matrix[local_index, local_index]
                * stiffness_matrix[local_index, local_index]
            )
        )
    eigenvalues = np.linalg.eigvals(
        np.linalg.solve(mass_matrix, stiffness_matrix)
    )
    frequencies = np.sqrt(np.maximum(np.real(eigenvalues), 0.0)) / (
        2.0 * math.pi
    )

    element_fraction = (
        np.asarray(element_radii_m) - rigid.hub_radius_m
    ) / span_m
    element_shapes = np.stack(
        [_shape(coefficient_sets[index], element_fraction) for index in active_indices]
    )
    element_slopes = np.stack(
        [
            _shape_derivative(
                coefficient_sets[index],
                element_fraction,
                span_m,
            )
            for index in active_indices
        ]
    )
    twist = np.deg2rad(
        np.interp(
            element_fraction,
            source_fraction,
            structure.structural_twist_degrees,
        )
    )
    twist_slopes = np.gradient(
        twist,
        np.asarray(element_radii_m),
        edge_order=1,
    )
    element_mass_density = (
        np.interp(
            element_fraction,
            source_fraction,
            structure.mass_density_kg_m,
        )
        * structure.mass_tuner
    )
    element_masses = element_mass_density * np.asarray(element_widths_m)
    return ModalBladeModel(
        blade_count=rigid.blade_count,
        mode_names=tuple(_MODE_NAMES[index] for index in active_indices),
        mode_directions=directions,
        element_mode_shapes=element_shapes,
        element_mode_slopes_per_m=element_slopes,
        element_structural_twist_radians=twist,
        element_structural_twist_slopes_per_m=twist_slopes,
        element_masses_kg=element_masses,
        mass_matrix_kg=mass_matrix,
        damping_matrix_n_s_m=damping_matrix,
        stiffness_matrix_n_m=stiffness_matrix,
        natural_frequencies_hz=np.sort(frequencies),
        initial_out_of_plane_tip_deflection_m=(
            turbine.initial_out_of_plane_tip_deflection_m
        ),
        initial_in_plane_tip_deflection_m=(
            turbine.initial_in_plane_tip_deflection_m
        ),
    )
