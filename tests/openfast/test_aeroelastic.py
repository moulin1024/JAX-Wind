from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jaxwind.domain import ScaleSystem
from jaxwind.openfast import (
    build_modal_blade_model,
    load_openfast_modal_turbine,
)


ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    ROOT
    / "tests"
    / "fixtures"
    / "openfast"
    / "nrel5mw"
    / "NREL5MW_Rigid_Smoke.fst"
)


def _model():
    turbine = load_openfast_modal_turbine(INPUT)
    return turbine, build_modal_blade_model(
        turbine,
        element_radii_m=turbine.rigid.element_radii_m,
        element_widths_m=turbine.rigid.element_widths_m,
        rotor_speed_rpm=turbine.rigid.rotor_speed_rpm,
    )


def test_loads_elastodyn_blade_modes_and_distributed_properties() -> None:
    turbine, model = _model()

    structure = turbine.blade_structure
    assert turbine.enabled_modes == (True, True, True)
    assert len(structure.station_fractions) == 49
    assert structure.station_fractions[0] == 0.0
    assert structure.station_fractions[-1] == 1.0
    assert structure.mass_tuner == pytest.approx(1.04536)
    assert model.mode_names == ("flap1", "flap2", "edge1")
    assert model.mass_matrix_kg.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(model.mass_matrix_kg) > 0.0)
    assert np.all(np.linalg.eigvalsh(model.stiffness_matrix_n_m) > 0.0)
    assert model.natural_frequencies_hz == pytest.approx(
        (0.7336, 1.1222, 2.0429),
        rel=2.0e-3,
    )


def test_modal_state_deforms_line_and_newmark_step_is_finite() -> None:
    turbine, model = _model()
    scales = ScaleSystem(length=512.0, velocity=0.4)
    line = turbine.rigid.to_actuator_line(
        scales=scales,
        x_m=256.0,
        y_m=256.0,
        smoothing_width_m=4.0,
    )
    state = model.initial_state()
    displacement = state.displacement_m.copy()
    displacement[:, 0] = (1.0, 2.0, 3.0)
    state = replace(state, displacement_m=displacement)

    deformed = model.deform_actuator_line(line, state, scales=scales)

    count = turbine.rigid.blade_count * len(turbine.rigid.element_radii_m)
    assert len(deformed.element_flap_displacements) == count
    assert max(deformed.element_flap_displacements) == pytest.approx(
        3.0 / scales.length
    )
    generalized_force = np.full((3, 3), 1.0e4)
    advanced, diagnostic = model.advance(
        state,
        generalized_force,
        dt_seconds=0.05,
    )
    assert np.all(np.isfinite(advanced.displacement_m))
    assert diagnostic.maximum_tip_deflection_m > 0.0


def test_generalized_load_projection_includes_aerodynamics_and_gravity() -> None:
    turbine, model = _model()
    element_count = len(turbine.rigid.element_radii_m)
    point_count = turbine.rigid.blade_count * element_count
    force = np.zeros((point_count, 3))
    force[:, 0] = 100.0
    normals = np.zeros_like(force)
    normals[:, 0] = 1.0
    tangents = np.zeros_like(force)
    tangents[:, 1] = 1.0

    generalized = model.generalized_forces(
        force,
        normals,
        tangents,
        gravity_m_s2=0.0,
    )

    assert generalized.shape == (3, 3)
    assert np.all(generalized[:, 0] > 0.0)
    assert np.all(np.isfinite(generalized))
