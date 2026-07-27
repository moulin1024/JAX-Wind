"""Paper parameters for Lin & Porté-Agel, Energies 12 (2019), 4574."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PaperCase:
    domain: tuple[float, float, float] = (6.4, 0.8, 0.4)
    grid: tuple[int, int, int] = (128, 64, 32)
    boundary_layer_height: float = 0.4
    hub_velocity: float = 4.88
    hub_turbulence_intensity: float = 0.07
    roughness_length: float = 0.022e-3
    friction_velocity: float = 0.22
    rotor_diameter: float = 0.15
    hub_height: float = 0.125
    turbine_x: float = 3.2
    turbine_y: float = 0.4
    yaw_degrees: tuple[float, ...] = (10.0, 20.0, 30.0)
    profile_x_over_d: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0)


PAPER_CASE = PaperCase()

# These measured coefficients are the standard-ADM inputs for the three
# Bastankhah--Porté-Agel wind-tunnel operating points reproduced by the paper.
THRUST_COEFFICIENT_BY_YAW = {
    10.0: 0.78,
    20.0: 0.73,
    30.0: 0.66,
}


def local_thrust_coefficient(thrust_coefficient: float) -> float:
    """Convert freestream ``C_T`` to disk-local ``C_T'`` by momentum theory."""
    if not 0.0 <= thrust_coefficient < 1.0:
        raise ValueError("thrust_coefficient must lie in [0, 1)")
    induction = 0.5 * (1.0 - math.sqrt(1.0 - thrust_coefficient))
    return thrust_coefficient / (1.0 - induction) ** 2


def paper_settings(
    yaw_degrees: float,
    *,
    quick: bool = False,
    steps: int | None = None,
    sample_every: int | None = None,
) -> dict[str, object]:
    """Return settings consumed by ``legacy/jax/run_single.py``."""
    try:
        thrust_coefficient = THRUST_COEFFICIENT_BY_YAW[float(yaw_degrees)]
    except KeyError as exc:
        supported = ", ".join(f"{yaw:g}" for yaw in PAPER_CASE.yaw_degrees)
        raise ValueError(f"paper yaw must be one of {supported} degrees") from exc

    nx, ny, nz = PAPER_CASE.grid
    run_steps = 19_000 if steps is None else steps
    log_every = 100 if sample_every is None else sample_every
    if quick:
        nx, ny, nz = 32, 16, 16
        run_steps = 8 if steps is None else steps
        log_every = 1 if sample_every is None else sample_every

    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "lx": PAPER_CASE.domain[0],
        "ly": PAPER_CASE.domain[1],
        "lz": PAPER_CASE.domain[2],
        "z_i": PAPER_CASE.boundary_layer_height,
        "steps": run_steps,
        # The paper does not report dt. This is the stable value retained by
        # the archived WiRE wind-tunnel setup in this repository.
        "dt": 2.5e-4,
        "log_every": log_every,
        "u_fric": PAPER_CASE.friction_velocity,
        "zo": PAPER_CASE.roughness_length,
        "bl_height": PAPER_CASE.boundary_layer_height,
        "pressure_force": None,
        "uniform_u": PAPER_CASE.hub_velocity,
        "uniform_v": 0.0,
        "horizontal_homogeneous": False,
        "initial_condition": "log_law",
        "momentum_wall_model": "abl",
        "wall_stress_model": "dynamic_neutral",
        "initial_velocity_noise": 0.01,
        "sgs_model": "smagorinsky" if quick else "lasd",
        "cs_count": 10,
        "time_scheme": "ab2",
        "projection_mode": "stage",
        "top_boundary_condition": "rigid_lid",
        "actuator_disk_enabled": True,
        "actuator_disk_x": PAPER_CASE.turbine_x,
        "actuator_disk_y": PAPER_CASE.turbine_y,
        "actuator_disk_z": PAPER_CASE.hub_height,
        "actuator_disk_diameter": PAPER_CASE.rotor_diameter,
        "actuator_disk_hub_diameter": 0.0,
        "actuator_disk_ct_prime": local_thrust_coefficient(thrust_coefficient),
        "actuator_disk_thickness": 0.05,
        "actuator_disk_yaw_degrees": float(yaw_degrees),
        # Restore the downstream flow before the periodic seam. This is the
        # single-domain first milestone; concurrent turbulent precursor
        # coupling is intentionally left for the next benchmark stage.
        "fringe_enabled": True,
        "fringe_start_x": 5.4,
        "fringe_timescale": 0.1,
        "fringe_target_u": PAPER_CASE.hub_velocity,
        "fringe_target_v": 0.0,
        "precision": "float64",
        "sgs_precision": "float32",
        "use_jit": not quick,
    }
