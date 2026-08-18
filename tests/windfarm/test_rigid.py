from __future__ import annotations

import math
from pathlib import Path

import pytest

from jaxwind.domain import ScaleSystem
from jaxwind.windfarm import (
    OpenFASTInputError,
    RigidBladeElementDisk,
    load_openfast_rigid_turbine,
)

def _write_openfast_deck(
    root: Path,
    *,
    mirror_rotor: bool = False,
    aftab_mode: int = 1,
    second_blade_pitch: float = 1.0,
) -> Path:
    (root / "model.fst").write_text(
        f"""
False Echo
1 NRotors
{"True" if mirror_rotor else "False"} MirrorRotor
"ElastoDyn.dat" EDFile
"AeroDyn.dat" AeroFile
""".strip()
        + "\n"
    )
    (root / "ElastoDyn.dat").write_text(
        f"""
3 NumBl
50.0 TipRad
2.0 HubRad
-2.0 PreCone(1)
-2.0 PreCone(2)
-2.0 PreCone(3)
1.0 BlPitch(1)
{second_blade_pitch} BlPitch(2)
1.0 BlPitch(3)
7.5 Azimuth
12.0 RotSpeed
3.0 NacYaw
5.0 ShftTilt
2.5 AzimB1Up
-4.0 OverHang
2.0 Twr2Shft
86.0 TowerHt
""".strip()
        + "\n"
    )
    (root / "AeroDyn.dat").write_text(
        f"""
0 Wake_Mod
True TipLoss
False HubLoss
0 UA_Mod
{aftab_mode} AFTabMod
1 InCol_Alfa
2 InCol_Cl
3 InCol_Cd
0 InCol_Cm
2 NumAFfiles
"airfoils/af-root.dat" AFNames
"airfoils/af-tip.dat"
"blade.dat" ADBlFile(1)
"blade.dat" ADBlFile(2)
"blade.dat" ADBlFile(3)
""".strip()
        + "\n"
    )
    (root / "blade.dat").write_text(
        """
3 NumBlNds
BlSpn BlCrvAC BlSwpAC BlCrvAng BlTwist BlChord BlAFID
(m) (m) (m) (deg) (deg) (m) (-)
0.0  0.0 0.0 0.0 10.0 3.0 1
24.0 0.1 0.0 0.0  5.0 2.0 1
48.0 0.0 0.0 0.0  0.0 1.0 2
""".strip()
        + "\n"
    )
    airfoils = root / "airfoils"
    airfoils.mkdir()
    (airfoils / "af-root.dat").write_text(
        """
DEFAULT InterpOrd
0 NumCoords
1 NumTabs
3 NumAlf
-10.0 -1.0 0.10
  0.0  0.0 0.01
 10.0  1.0 0.10
""".strip()
        + "\n"
    )
    (airfoils / "af-tip.dat").write_text(
        """
1 InterpOrd
0 NumCoords
1 NumTabs
3 NumAlf
-5.0 -0.4 0.06
 0.0  0.0 0.02
 5.0  0.4 0.06
""".strip()
        + "\n"
    )
    return root / "model.fst"


def test_loads_openfast_deck_and_resamples_airfoils(tmp_path: Path) -> None:
    source = _write_openfast_deck(tmp_path)

    turbine = load_openfast_rigid_turbine(source)

    assert turbine.source == source
    assert turbine.blade_count == 3
    assert turbine.hub_radius_m == 2.0
    assert turbine.tip_radius_m == 50.0
    assert turbine.hub_height_m == pytest.approx(
        88.0 + 4.0 * math.sin(math.radians(5.0))
    )
    assert turbine.initial_azimuth_degrees == 5.0
    assert turbine.element_radii_m == (2.0, 26.0, 50.0)
    assert turbine.element_widths_m == (12.0, 24.0, 12.0)
    assert turbine.element_chords_m == (3.0, 2.0, 1.0)
    assert turbine.element_airfoil_ids == (0, 0, 1)
    assert turbine.polar_alpha_degrees == (-10.0, -5.0, 0.0, 5.0, 10.0)
    assert turbine.polar_lift_coefficients[0] == (
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
    )
    assert turbine.polar_lift_coefficients[1] == (
        -0.4,
        -0.4,
        0.0,
        0.4,
        0.4,
    )
    assert turbine.tip_loss
    assert not turbine.root_loss
    assert any("BlCrvAC" in note for note in turbine.compatibility_notes)


def test_lowers_openfast_si_data_to_execution_units(tmp_path: Path) -> None:
    source = _write_openfast_deck(tmp_path, mirror_rotor=True)
    turbine = load_openfast_rigid_turbine(source)
    scales = ScaleSystem(length=100.0, velocity=10.0)

    line = turbine.to_actuator_line(
        scales=scales,
        x_m=400.0,
        y_m=100.0,
        smoothing_width_m=4.0,
        rotor_speed_rpm=9.0,
        pitch_degrees=2.0,
    )

    assert line.x == 4.0
    assert line.y == 1.0
    assert line.z == pytest.approx(turbine.hub_height_m / 100.0)
    assert line.hub_radius == 0.02
    assert line.tip_radius == 0.5
    assert line.angular_velocity == pytest.approx(
        -(9.0 * 2.0 * math.pi / 60.0) / scales.inverse_time
    )
    assert line.smoothing_width == 0.04
    assert line.element_radii == (0.02, 0.26, 0.5)
    assert line.element_widths == (0.12, 0.24, 0.12)
    assert line.pitch_degrees == 2.0
    assert line.yaw_degrees == 3.0
    assert line.tilt_degrees == -5.0
    assert line.precone_degrees == -2.0

    disk = turbine.to_actuator_disk_bem(
        scales=scales,
        x_m=400.0,
        y_m=100.0,
        smoothing_width_m=4.0,
        yaw_degrees=0.0,
    )
    assert disk.blade_count == line.blade_count
    assert disk.element_chords == line.element_chords
    assert disk.tilt_degrees == 0.0
    assert disk.precone_degrees == 0.0

    configured = RigidBladeElementDisk(
        rotor=turbine,
        x_m=400.0,
        y_m=100.0,
        smoothing_width_m=4.0,
        hub_height_m=turbine.hub_height_m,
    )
    body = configured.to_nacelle_tower(scales=scales)
    assert body.nacelle_length == 0.15
    assert body.tower_base_diameter == pytest.approx(0.083)
    assert body.tower_top_diameter == pytest.approx(0.055)


def test_rejects_per_blade_operating_point_difference(tmp_path: Path) -> None:
    source = _write_openfast_deck(tmp_path, second_blade_pitch=2.0)

    with pytest.raises(OpenFASTInputError, match="per-blade BlPitch"):
        load_openfast_rigid_turbine(source)


def test_rejects_multiple_aerodyn_polar_tables(tmp_path: Path) -> None:
    source = _write_openfast_deck(tmp_path, aftab_mode=2)

    with pytest.raises(OpenFASTInputError, match="AFTabMod=2"):
        load_openfast_rigid_turbine(source)
