"""Build turbine definitions and conservative forcing."""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import replace
from jaxwind.config.stages import FiniteVolumeWorkflow, TurbineOptions

def _openfast_path(options: TurbineOptions) -> Path:
    environment = options.model_environment
    if environment is None:
        raise ValueError("the native HITSZ rotor does not use an OpenFAST path")
    value = os.environ.get(environment)
    if not value:
        raise ValueError(f"set {environment} to the OpenFAST .fst model")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"OpenFAST model does not exist: {path}")
    return path


def _lower_actuator_line(turbine, options: TurbineOptions, scales):
    line = turbine.to_actuator_line(
        scales=scales,
        initial_azimuth_degrees=options.initial_azimuth_degrees,
    )
    factor = options.smoothing_width_chord_factor
    if factor is None:
        return line
    return replace(
        line,
        element_gaussian_widths=tuple(
            factor * chord for chord in line.element_chords
        ),
    )


def build_turbine_definition(workflow: FiniteVolumeWorkflow):
    options = workflow.turbine
    if options is None:
        return None
    if options.model == "thrust-only-adm":
        from jaxwind.windfarm.actuator_disk import SimpleActuatorDisk
        return SimpleActuatorDisk(x_m=options.x_m, y_m=options.y_m, hub_height_m=options.hub_height_m, rotor_diameter_m=options.rotor_diameter_m, thrust_coefficient_prime=options.thrust_coefficient, smoothing_width_m=options.smoothing_width_m, prescribed_inflow_velocity_m_s=options.prescribed_inflow_velocity_m_s, prescribed_thrust_coefficient=options.thrust_coefficient)
    from jaxwind.windfarm import (
        HITSZR9BladeElementDisk,
        RigidBladeElementDisk,
        load_openfast_rigid_turbine,
    )

    common = {
        "x_m": options.x_m,
        "y_m": options.y_m,
        "smoothing_width_m": options.smoothing_width_m,
        "hub_height_m": options.hub_height_m,
        "rotor_speed_rpm": options.rotor_speed_rpm,
        "pitch_degrees": options.blade_pitch_degrees,
        "smearing_azimuthal_elements": options.smearing_azimuthal_elements,
        "body_smoothing_width_m": options.body_smoothing_width_m,
        "nacelle_drag_coefficient": options.nacelle_drag_coefficient,
        "tower_drag_coefficient": options.tower_drag_coefficient,
    }
    if options.model.startswith("hitsz-r9-"):
        turbine = HITSZR9BladeElementDisk(**common)
    else:
        rotor = load_openfast_rigid_turbine(_openfast_path(options))
        turbine = RigidBladeElementDisk(rotor=rotor, **common)

    from jaxwind.domain import ScaleSystem

    grid = workflow.case.physical.physical_grid
    scales = ScaleSystem(1.0, 1.0)
    rotor = (
        _lower_actuator_line(turbine, options, scales)
        if options.model.endswith("alm")
        else turbine.to_actuator_disk(scales=scales)
    )
    if not 0.0 < rotor.x < grid.lx:
        raise ValueError("finite-volume turbine x position is outside the domain")
    if not 0.0 < rotor.y < grid.ly:
        raise ValueError("finite-volume turbine y position is outside the domain")
    if rotor.z - rotor.tip_radius <= 0.0:
        raise ValueError("finite-volume turbine rotor intersects the lower boundary")
    if rotor.z + rotor.tip_radius >= grid.lz:
        raise ValueError("finite-volume turbine rotor intersects the upper boundary")
    return turbine


def build_turbine_forcing(workflow: FiniteVolumeWorkflow):
    turbine = build_turbine_definition(workflow)
    if turbine is None:
        return None
    from jaxwind.domain import ScaleSystem
    from jaxwind import build_adbem_forcing, build_actuator_line_forcing
    scales = ScaleSystem(1.0, 1.0)
    grid = workflow.case.physical.physical_grid
    if workflow.turbine.model == "thrust-only-adm":
        from jaxwind.turbine import build_thrust_only_adm_forcing
        return build_thrust_only_adm_forcing(grid, turbine.to_actuator_disk(scales=scales), periodic_y=True)
    body = turbine.to_nacelle_tower(scales=scales)
    if body.nacelle_drag_coefficient == 0.0 and body.tower_drag_coefficient == 0.0:
        body = None
    if workflow.turbine.model.endswith("alm"):
        line = _lower_actuator_line(
            turbine,
            workflow.turbine,
            scales,
        )
        return build_actuator_line_forcing(grid, line, body)
    return build_adbem_forcing(
        grid,
        turbine.to_actuator_disk(scales=scales),
        body,
        minimum_normal_smoothing_width=workflow.turbine.minimum_normal_smoothing_width_m,
        momentum_stabilization_coefficient=workflow.turbine.momentum_stabilization_coefficient,
    )


def combine_forcings(*forcings):
    """Add independently conservative momentum sources component-wise."""

    active = tuple(forcing for forcing in forcings if forcing is not None)
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    from jaxwind import StaggeredVelocity

    def combined(velocity, time):
        total = active[0](velocity, time)
        for forcing in active[1:]:
            other = forcing(velocity, time)
            total = StaggeredVelocity(
                total.x + other.x,
                total.y + other.y,
                total.z + other.z,
            )
        return total

    return combined
