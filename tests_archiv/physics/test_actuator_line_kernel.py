from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind_archiv.interpreters._jax_actuator_line import (  # noqa: E402
    actuator_line_deformed_kinematics,
    actuator_line_geometry,
    blade_element_kinematic_forces,
)


def test_rigid_geometry_rotates_blades_and_preserves_radius() -> None:
    positions, tangents, speed, normal = actuator_line_geometry(
        x=1.0,
        y=2.0,
        z=3.0,
        blade_count=2,
        element_radii=(0.5, 1.0),
        angular_velocity=2.0,
        time=0.0,
        yaw_degrees=0.0,
        tilt_degrees=0.0,
        precone_degrees=0.0,
        initial_azimuth_degrees=0.0,
        dtype=jnp.float64,
    )

    assert positions == pytest.approx(
        jnp.asarray(
            (
                (1.0, 2.0, 3.5),
                (1.0, 2.0, 4.0),
                (1.0, 2.0, 2.5),
                (1.0, 2.0, 2.0),
            )
        )
    )
    assert tangents == pytest.approx(
        jnp.asarray(
            (
                (0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, -1.0, 0.0),
            )
        )
    )
    assert speed == pytest.approx(jnp.asarray((1.0, 2.0, 1.0, 2.0)))
    assert normal == pytest.approx(jnp.asarray((1.0, 0.0, 0.0)))


def test_deformed_geometry_moves_sections_and_adds_modal_velocity() -> None:
    positions, tangents, velocity, normals, spans = (
        actuator_line_deformed_kinematics(
            x=0.0,
            y=0.0,
            z=0.0,
            blade_count=1,
            element_radii=(1.0,),
            angular_velocity=2.0,
            time=0.0,
            yaw_degrees=0.0,
            tilt_degrees=0.0,
            precone_degrees=0.0,
            initial_azimuth_degrees=0.0,
            flap_displacements=(0.2,),
            edge_displacements=(0.1,),
            flap_slopes=(0.05,),
            edge_slopes=(0.0,),
            flap_velocities=(0.3,),
            edge_velocities=(0.4,),
            dtype=jnp.float64,
        )
    )

    assert positions[0] == pytest.approx((0.2, 0.1, 1.0))
    assert velocity[0] == pytest.approx((0.3, 2.4, -0.2))
    assert float(jnp.linalg.norm(tangents[0])) == pytest.approx(1.0)
    assert float(jnp.linalg.norm(normals[0])) == pytest.approx(1.0)
    assert float(jnp.linalg.norm(spans[0])) == pytest.approx(1.0)


def test_quasi_steady_blade_force_is_finite_and_opposes_streamwise_flow() -> None:
    force, alpha, lift, drag, loss = blade_element_kinematic_forces(
        sampled_velocity=jnp.asarray(((1.0, 0.0, 0.0),)),
        tangents=jnp.asarray(((0.0, 1.0, 0.0),)),
        tangential_speed=jnp.asarray((4.0,)),
        normal=jnp.asarray((1.0, 0.0, 0.0)),
        element_radii=jnp.asarray((0.75,)),
        element_widths=jnp.asarray((0.1,)),
        element_chords=jnp.asarray((0.2,)),
        element_twist_degrees=jnp.asarray((5.0,)),
        element_airfoil_ids=jnp.asarray((0,)),
        blade_count=3,
        hub_radius=0.1,
        tip_radius=1.0,
        pitch_degrees=0.0,
        polar_alpha_degrees=(-180.0, 0.0, 180.0),
        polar_lift_coefficients=((0.0, 0.0, 0.0),),
        polar_drag_coefficients=((0.1, 0.1, 0.1),),
        tip_loss=False,
        root_loss=False,
    )

    assert bool(jnp.all(jnp.isfinite(force)))
    assert float(force[0, 0]) < 0.0
    assert float(alpha[0]) == pytest.approx(
        math_degrees_atan2(1.0, 4.0) - 5.0
    )
    assert float(lift[0]) == 0.0
    assert float(drag[0]) == 0.1
    assert float(loss[0]) == 1.0


def math_degrees_atan2(y: float, x: float) -> float:
    return float(jnp.rad2deg(jnp.arctan2(y, x)))
