from __future__ import annotations

import pytest

from jaxwind.domain import ScaleSystem
from jaxwind.physics import PureThrustActuatorDisk, WindTunnelModel
from jaxwind.windfarm import (
    DTU_10MW_HUB_HEIGHT_M,
    DTU_10MW_ROTOR_DIAMETER_M,
    SimpleActuatorDisk,
    dtu_10mw_reference_actuator_disk,
    dtu_10mw_prescribed_actuator_disk,
)


def _disk(**overrides) -> SimpleActuatorDisk:
    values = {
        "x_m": 400.0,
        "y_m": 250.0,
        "hub_height_m": 90.0,
        "rotor_diameter_m": 120.0,
        "thrust_coefficient_prime": 4.0 / 3.0,
        "smoothing_width_m": 10.0,
    }
    values.update(overrides)
    return SimpleActuatorDisk(**values)


def test_lowers_si_geometry_to_existing_force_conserving_disk() -> None:
    disk = _disk().to_actuator_disk(scales=ScaleSystem(100.0, 10.0))

    assert isinstance(disk, PureThrustActuatorDisk)
    assert disk.x == 4.0
    assert disk.y == 2.5
    assert disk.z == 0.9
    assert disk.diameter == 1.2
    assert disk.thrust_coefficient_prime == pytest.approx(4.0 / 3.0)
    assert disk.normal_smoothing_width == 0.1
    assert disk.transverse_smoothing_width == 0.1
    assert disk.hub_diameter == 0.0
    assert disk.yaw_degrees == 0.0
    assert disk.filtered_velocity_correction
    assert WindTunnelModel(actuator_disk=disk).actuator_disk is disk


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hub_height_m": -1.0}, "hub height"),
        ({"rotor_diameter_m": 0.0}, "rotor diameter"),
        ({"thrust_coefficient_prime": -0.1}, "thrust coefficient"),
        ({"smoothing_width_m": 0.0}, "smoothing width"),
        ({"x_m": float("nan")}, "must be finite"),
    ],
)
def test_rejects_invalid_physical_parameters(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _disk(**overrides)


def test_requires_mechanical_scales_for_lowering() -> None:
    with pytest.raises(TypeError, match="ScaleSystem"):
        _disk().to_actuator_disk(scales=object())


def test_dtu_10mw_factory_uses_reference_geometry_and_explicit_loading() -> None:
    disk = dtu_10mw_reference_actuator_disk(
        x_m=1000.0,
        y_m=500.0,
        smoothing_width_m=25.0,
    )

    assert disk.hub_height_m == DTU_10MW_HUB_HEIGHT_M == 119.0
    assert disk.rotor_diameter_m == DTU_10MW_ROTOR_DIAMETER_M == 178.3
    assert disk.thrust_coefficient_prime == pytest.approx(4.0 / 3.0)


def test_dtu_10mw_prescribed_factory_matches_legacy_loading() -> None:
    disk = dtu_10mw_prescribed_actuator_disk(x_m=1000.0, y_m=512.0)
    lowered = disk.to_actuator_disk(scales=ScaleSystem(100.0, 10.0))

    assert lowered.thrust_coefficient_prime == 0.0
    assert lowered.prescribed_inflow_velocity == pytest.approx(1.108514881)
    assert lowered.prescribed_thrust_coefficient == pytest.approx(0.840)
    assert not lowered.filtered_velocity_correction or lowered.prescribed_inflow_velocity > 0.0
