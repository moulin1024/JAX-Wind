from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import jax.numpy as jnp

from jaxwind._jax.actuator_line import blade_element_kinematic_forces
from jaxwind.domain import ScaleSystem
from jaxwind.windfarm import HITSZR9BladeElementDisk


ROOT = Path(__file__).resolve().parents[2]


def test_hitsz_r9_lowers_digitized_geometry_and_polar() -> None:
    turbine = HITSZR9BladeElementDisk(
        x_m=12.0,
        y_m=3.0,
        smoothing_width_m=0.0625,
        body_smoothing_width_m=0.125,
    )
    disk = turbine.to_actuator_disk(scales=ScaleSystem(length=1.0, velocity=1.0))

    assert disk.blade_count == 3
    assert len(disk.element_radii) == 24
    assert disk.tip_radius == 0.63
    assert disk.hub_radius == pytest.approx(0.0252)
    assert disk.angular_velocity == pytest.approx(16.0 * math.pi)
    assert disk.element_radii[0] == pytest.approx(0.0378)
    assert disk.element_radii[-1] == pytest.approx(0.6174)
    assert disk.element_chords[0] == pytest.approx(0.0343976)
    assert disk.element_chords[-1] == pytest.approx(0.0048795)
    assert disk.polar_alpha_degrees == tuple(float(v) for v in range(-10, 31))
    assert disk.polar_lift_coefficients[0][33] == pytest.approx(1.8030)
    assert disk.polar_drag_coefficients[0][0] == pytest.approx(0.0395)
    assert min(turbine.element_smoothing_widths_m) == pytest.approx(
        0.02547177983420705
    )
    assert max(turbine.element_smoothing_widths_m) == pytest.approx(
        0.06564288451029125
    )

    body = turbine.to_nacelle_tower(
        scales=ScaleSystem(length=1.0, velocity=1.0)
    )
    assert body.tower_base_diameter == 0.04
    assert body.tower_top_diameter == 0.04
    assert body.smoothing_width == 0.125


def test_versioned_hitsz_polar_matches_execution_table() -> None:
    path = (
        ROOT
        / "cases"
        / "HITSZWindTunnel"
        / "reference"
        / "hitsz001_polar_digitized.csv"
    )
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    disk = HITSZR9BladeElementDisk(12.0, 3.0, 0.0625).to_actuator_disk(
        scales=ScaleSystem(length=1.0, velocity=1.0)
    )

    assert tuple(float(row["angle_of_attack_degrees"]) for row in rows) == (
        disk.polar_alpha_degrees
    )
    assert tuple(float(row["lift_coefficient"]) for row in rows) == (
        disk.polar_lift_coefficients[0]
    )
    assert tuple(float(row["drag_coefficient"]) for row in rows) == (
        disk.polar_drag_coefficients[0]
    )


def test_hitsz_r9_freestream_thrust_is_close_to_measured_r9() -> None:
    disk = HITSZR9BladeElementDisk(12.0, 3.0, 0.0625).to_actuator_disk(
        scales=ScaleSystem(length=1.0, velocity=1.0)
    )
    speed = 3.3514673026030772
    count = len(disk.element_radii)
    sampled = jnp.column_stack(
        (jnp.full(count, speed), jnp.zeros(count), jnp.zeros(count))
    )
    tangent = jnp.tile(jnp.asarray((0.0, 1.0, 0.0)), (count, 1))
    forces, *_ = blade_element_kinematic_forces(
        sampled,
        tangent,
        disk.angular_velocity * jnp.asarray(disk.element_radii),
        jnp.asarray((1.0, 0.0, 0.0)),
        element_radii=disk.element_radii,
        element_widths=disk.element_widths,
        element_chords=disk.element_chords,
        element_twist_degrees=disk.element_twist_degrees,
        element_airfoil_ids=disk.element_airfoil_ids,
        blade_count=disk.blade_count,
        hub_radius=disk.hub_radius,
        tip_radius=disk.tip_radius,
        pitch_degrees=disk.pitch_degrees,
        polar_alpha_degrees=disk.polar_alpha_degrees,
        polar_lift_coefficients=disk.polar_lift_coefficients,
        polar_drag_coefficients=disk.polar_drag_coefficients,
        tip_loss=disk.tip_loss,
        root_loss=disk.root_loss,
    )
    thrust_per_density = -float(jnp.sum(forces[:, 0])) * disk.blade_count
    area = math.pi * disk.tip_radius**2
    coefficient = thrust_per_density / (0.5 * speed**2 * area)

    assert coefficient == pytest.approx(0.87868, rel=2.0e-5)
    assert abs(coefficient - 0.810) < 0.1
