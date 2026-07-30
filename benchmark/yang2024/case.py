"""Configuration for the Yang, Lin & Zhou rated wind-tunnel benchmark."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = CASE_DIR / "reference"
OPERATING_POINTS_PATH = REFERENCE_DIR / "operating_points.csv"
INFLOW_PROFILE_PATH = REFERENCE_DIR / "inflow_profile_override.csv"


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    condition: str
    wind_speed_m_s: float
    design_rotor_speed_rpm: float
    test_rotor_speed_rpm: float
    tip_speed_ratio: float
    target_ct: float
    target_cp: float
    measured_ct: float | None
    measured_cp: float | None
    mean_thrust_n: float | None
    mean_torque_n_m: float | None


@dataclass(frozen=True, slots=True)
class InflowFit:
    von_karman: float
    friction_velocity_m_s: float
    roughness_length_m: float
    minimum_height_m: float
    maximum_height_m: float
    minimum_turbulence_intensity: float
    maximum_turbulence_intensity: float
    maximum_reconstruction_error_m_s: float

    def velocity_m_s(
        self,
        height_m: float,
        *,
        friction_velocity_m_s: float | None = None,
    ) -> float:
        if height_m <= self.roughness_length_m:
            raise ValueError("height must exceed the roughness length")
        friction_velocity = (
            self.friction_velocity_m_s
            if friction_velocity_m_s is None
            else friction_velocity_m_s
        )
        return (
            friction_velocity
            / self.von_karman
            * math.log(height_m / self.roughness_length_m)
        )

    def friction_velocity_for_hub_speed(
        self,
        hub_speed_m_s: float,
        hub_height_m: float,
    ) -> float:
        if hub_speed_m_s <= 0.0:
            raise ValueError("hub speed must be positive")
        if hub_height_m <= self.roughness_length_m:
            raise ValueError("hub height must exceed the roughness length")
        return (
            self.von_karman
            * hub_speed_m_s
            / math.log(hub_height_m / self.roughness_length_m)
        )


@dataclass(frozen=True, slots=True)
class PaperCase:
    test_section_m: tuple[float, float, float] = (24.0, 6.0, 3.6)
    grid: tuple[int, int, int] = (384, 96, 64)
    rotor_diameter_m: float = 1.26
    hub_height_m: float = 0.876
    turbine_x_m: float = 12.0
    turbine_y_m: float = 3.0
    length_scale: float = 1.0 / 100.0
    time_scale: float = 1.0 / 40.0
    reported_blockage_ratio: float = 0.058
    reported_turbulence_ceiling: float = 0.01
    rated_condition: str = "R9"
    air_density_kg_m3: float = 1.225
    molecular_viscosity_m2_s: float = 1.5e-5

    @property
    def test_section_area_m2(self) -> float:
        return self.test_section_m[1] * self.test_section_m[2]

    @property
    def rotor_area_m2(self) -> float:
        return math.pi * (0.5 * self.rotor_diameter_m) ** 2

    @property
    def geometric_blockage_ratio(self) -> float:
        return self.rotor_area_m2 / self.test_section_area_m2

    @property
    def spacing_m(self) -> tuple[float, float, float]:
        return tuple(
            length / cells
            for length, cells in zip(
                self.test_section_m,
                self.grid,
                strict=True,
            )
        )


def _optional_float(value: str) -> float | None:
    return None if not value.strip() else float(value)


def load_operating_points(
    path: Path = OPERATING_POINTS_PATH,
) -> dict[str, OperatingPoint]:
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        result = {}
        for row in rows:
            point = OperatingPoint(
                condition=row["condition"],
                wind_speed_m_s=float(row["wind_speed_m_s"]),
                design_rotor_speed_rpm=float(row["design_rotor_speed_rpm"]),
                test_rotor_speed_rpm=float(row["test_rotor_speed_rpm"]),
                tip_speed_ratio=float(row["tip_speed_ratio"]),
                target_ct=float(row["target_ct"]),
                target_cp=float(row["target_cp"]),
                measured_ct=_optional_float(row["measured_ct"]),
                measured_cp=_optional_float(row["measured_cp"]),
                mean_thrust_n=_optional_float(row["mean_thrust_n"]),
                mean_torque_n_m=_optional_float(row["mean_torque_n_m"]),
            )
            result[point.condition] = point
    return result


def load_inflow_fit(
    path: Path = INFLOW_PROFILE_PATH,
    *,
    von_karman: float = 0.4,
) -> InflowFit:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError("inflow fit requires at least two height levels")

    heights_m = [float(row["Height_mm"]) * 1.0e-3 for row in rows]
    log_heights = [math.log(height) for height in heights_m]
    fit_velocity = [float(row["LogFit_WindSpeed_m_s"]) for row in rows]
    turbulence = [0.01 * float(row["TurbulenceIntensity_percent"]) for row in rows]

    mean_x = sum(log_heights) / len(log_heights)
    mean_y = sum(fit_velocity) / len(fit_velocity)
    slope = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(log_heights, fit_velocity, strict=True)
    ) / sum((x - mean_x) ** 2 for x in log_heights)
    intercept = mean_y - slope * mean_x
    roughness_length = math.exp(-intercept / slope)
    reconstructed = [slope * value + intercept for value in log_heights]

    return InflowFit(
        von_karman=von_karman,
        friction_velocity_m_s=von_karman * slope,
        roughness_length_m=roughness_length,
        minimum_height_m=min(heights_m),
        maximum_height_m=max(heights_m),
        minimum_turbulence_intensity=min(turbulence),
        maximum_turbulence_intensity=max(turbulence),
        maximum_reconstruction_error_m_s=max(
            abs(actual - predicted)
            for actual, predicted in zip(
                fit_velocity,
                reconstructed,
                strict=True,
            )
        ),
    )


def local_thrust_coefficient(thrust_coefficient: float) -> float:
    """Convert freestream ``C_T`` to disk-local ``C_T'``."""
    if not 0.0 <= thrust_coefficient < 1.0:
        raise ValueError("thrust_coefficient must lie in [0, 1)")
    induction = 0.5 * (1.0 - math.sqrt(1.0 - thrust_coefficient))
    return thrust_coefficient / (1.0 - induction) ** 2


PAPER_CASE = PaperCase()
OPERATING_POINTS = load_operating_points()
INFLOW_FIT = load_inflow_fit()


def _inflow_mode(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in {"paper_uniform", "measured_log"}:
        raise ValueError("inflow mode must be 'paper_uniform' or 'measured_log'")
    return normalized


def paper_settings(
    *,
    condition: str = "R9",
    inflow_mode: str = "paper_uniform",
    quick: bool = False,
    steps: int | None = None,
    sample_every: int | None = None,
) -> dict[str, object]:
    """Return settings consumed by ``legacy/jax/run_single.py``.

    The first runnable milestone is limited to R9 because that is the only
    operating point for which the paper text reports a measured ``C_T < 1``.
    """
    try:
        operating_point = OPERATING_POINTS[condition.upper()]
    except KeyError as error:
        supported = ", ".join(OPERATING_POINTS)
        raise ValueError(f"operating condition must be one of {supported}") from error
    if operating_point.measured_ct is None:
        raise ValueError(
            "the pure-thrust ADM milestone is limited to R9; the paper "
            f"does not tabulate a measured C_T for {operating_point.condition}"
        )

    mode = _inflow_mode(inflow_mode)
    nx, ny, nz = PAPER_CASE.grid
    if quick:
        nx, ny, nz = 96, 24, 18

    if steps is None:
        run_steps = 4 if quick else (6_000 if mode == "paper_uniform" else 4_500)
    else:
        if steps <= 0:
            raise ValueError("steps must be positive")
        run_steps = steps
    log_every = (1 if quick else 100) if sample_every is None else sample_every
    if log_every <= 0:
        raise ValueError("sample_every must be positive")

    lx, ly, lz = PAPER_CASE.test_section_m
    friction_velocity = INFLOW_FIT.friction_velocity_for_hub_speed(
        operating_point.wind_speed_m_s,
        PAPER_CASE.hub_height_m,
    )
    uniform = mode == "paper_uniform"

    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "z_i": lz,
        "steps": run_steps,
        # The paper does not report a solver time step. This benchmark choice
        # keeps the initial directional CFL below 0.08 on the production grid.
        "dt": 1.0e-3,
        "log_every": log_every,
        "u_fric": friction_velocity,
        "zo": INFLOW_FIT.roughness_length_m,
        # Extrapolate the measured log fit through the complete tunnel height.
        "bl_height": lz,
        "vonk": INFLOW_FIT.von_karman,
        "pressure_force": 0.0 if uniform else None,
        "pressure_force_height": None if uniform else lz,
        "uniform_u": operating_point.wind_speed_m_s,
        "uniform_v": 0.0,
        "horizontal_homogeneous": False,
        "initial_condition": "uniform_flow" if uniform else "log_law",
        "momentum_wall_model": "free_slip" if uniform else "abl",
        "wall_stress_model": ("dynamic_neutral" if uniform else "prescribed_ustar"),
        "initial_velocity_noise": 0.0,
        "molecular_viscosity": PAPER_CASE.molecular_viscosity_m2_s,
        "sgs_model": "smagorinsky" if quick else "lasd",
        "cs_count": 10,
        "time_scheme": "ab2",
        "projection_mode": "stage",
        "top_boundary_condition": "rigid_lid",
        "actuator_disk_enabled": True,
        "actuator_disk_x": PAPER_CASE.turbine_x_m,
        "actuator_disk_y": PAPER_CASE.turbine_y_m,
        "actuator_disk_z": PAPER_CASE.hub_height_m,
        "actuator_disk_diameter": PAPER_CASE.rotor_diameter_m,
        "actuator_disk_hub_diameter": 0.0,
        "actuator_disk_ct_prime": local_thrust_coefficient(operating_point.measured_ct),
        "actuator_disk_thickness": 2.0 * lx / nx,
        "actuator_disk_yaw_degrees": 0.0,
        # The current static fringe has a uniform target. It is appropriate
        # for the paper baseline, but would erase the measured log profile.
        "fringe_enabled": uniform,
        "fringe_start_x": 20.0,
        "fringe_timescale": 0.25,
        "fringe_target_u": operating_point.wind_speed_m_s,
        "fringe_target_v": 0.0,
        "precision": "float64",
        "sgs_precision": "float32",
        "use_jit": not quick,
    }


def resolved_case(inflow_mode: str = "paper_uniform") -> dict[str, object]:
    """Return benchmark provenance and derived physical choices."""
    mode = _inflow_mode(inflow_mode)
    point = OPERATING_POINTS[PAPER_CASE.rated_condition]
    scaled_ustar = INFLOW_FIT.friction_velocity_for_hub_speed(
        point.wind_speed_m_s,
        PAPER_CASE.hub_height_m,
    )
    return {
        "paper": {
            "citation": ("Yang, Lin & Zhou, Renewable Energy 220 (2024), 119625"),
            "doi": "10.1016/j.renene.2023.119625",
        },
        "scope": "R9 rated pure-thrust actuator-disk milestone",
        "test_section_m": list(PAPER_CASE.test_section_m),
        "test_section_provenance": (
            "user-supplied facility dimensions, consistent with the "
            "paper's reported 5.8% blockage"
        ),
        "benchmark_grid": list(PAPER_CASE.grid),
        "benchmark_spacing_m": list(PAPER_CASE.spacing_m),
        "rotor_diameter_m": PAPER_CASE.rotor_diameter_m,
        "hub_height_m": PAPER_CASE.hub_height_m,
        "turbine_location_m": [
            PAPER_CASE.turbine_x_m,
            PAPER_CASE.turbine_y_m,
        ],
        "computed_blockage_ratio": PAPER_CASE.geometric_blockage_ratio,
        "reported_blockage_ratio": PAPER_CASE.reported_blockage_ratio,
        "operating_point": {
            "condition": point.condition,
            "wind_speed_m_s": point.wind_speed_m_s,
            "rotor_speed_rpm": point.test_rotor_speed_rpm,
            "measured_ct": point.measured_ct,
            "measured_cp": point.measured_cp,
        },
        "inflow": {
            "mode": mode,
            "base_fit_friction_velocity_m_s": (INFLOW_FIT.friction_velocity_m_s),
            "rated_scaled_friction_velocity_m_s": scaled_ustar,
            "roughness_length_m": INFLOW_FIT.roughness_length_m,
            "roughness_length_mm": (1.0e3 * INFLOW_FIT.roughness_length_m),
            "profile_height_range_m": [
                INFLOW_FIT.minimum_height_m,
                INFLOW_FIT.maximum_height_m,
            ],
            "measured_turbulence_intensity_range": [
                INFLOW_FIT.minimum_turbulence_intensity,
                INFLOW_FIT.maximum_turbulence_intensity,
            ],
        },
        "limitations": [
            "HITSZ001 polar tables are not published; no actuator-line model",
            "pure-thrust ADM cannot validate the reported power coefficient",
            "measured-log mode does not synthesize the measured turbulence",
        ],
    }
