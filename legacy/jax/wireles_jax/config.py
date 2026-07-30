from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import jax.numpy as jnp


AB2_DEFAULT_DT = 1.0e-3
RK4_DEFAULT_DT = 10.0 * AB2_DEFAULT_DT
RK3_DEFAULT_DT = RK4_DEFAULT_DT


def default_dt_for_time_scheme(time_scheme: str) -> float:
    if time_scheme == "rk4":
        return RK4_DEFAULT_DT
    if time_scheme == "rk3":
        return RK3_DEFAULT_DT
    if time_scheme == "ab2":
        return AB2_DEFAULT_DT
    raise ValueError(f"Unsupported time_scheme: {time_scheme}")


@dataclass(frozen=True)
class Params:
    nx: int = 64
    ny: int = 64
    nz: int = 64
    lx: float = 2.0 * 3.141592653589793
    ly: float = 2.0 * 3.141592653589793
    lz: float = 1.0
    z_i: float = 1.0
    dt: float | None = None
    nsteps: int = 100
    c_count: int = 10
    u_fric: float = 0.4
    zo: float = 5.0e-3
    bl_height: float = 1.0
    vonk: float = 0.4
    pressure_force: float | None = None
    pressure_force_height: float | None = None
    coriolis_f: float = 0.0
    geostrophic_u: float = 0.0
    geostrophic_v: float = 0.0
    uniform_u: float = 0.0
    uniform_v: float = 0.0
    horizontal_homogeneous: bool = True
    initial_condition: str = "default"
    momentum_wall_model: str = "abl"
    wall_stress_model: str = "dynamic_neutral"
    initial_velocity_noise: float = 0.01
    molecular_viscosity: float = 0.0
    molecular_diffusivity: float = 0.0
    rayleigh_number: float | None = None
    rayleigh_prandtl: float = 1.0
    fgr: float = 1.5
    tfr: float = 2.0
    sgs_model: str = "smagorinsky"
    cs_count: int = 10
    lasd_scale_dependent: bool = True
    momentum_lasd_scale_dependent: bool | None = None
    scalar_lasd_scale_dependent: bool | None = None
    lasd_invalid_beta_fallback: bool = False
    lasd_clipped_beta_fallback: bool = False
    smagorinsky_cs: float = 0.16
    sgs_delta_scale: float | None = None
    time_scheme: str = "ab2"
    projection_mode: str = "stage"
    horizontal_dealias: bool = True
    pressure_filter_nyquist: bool = False
    sharded_pressure_solver: str = "transpose"
    top_boundary_condition: str = "rigid_lid"
    radiation_brunt_vaisala: float | None = None
    sponge_enabled: bool = False
    sponge_start_height: float = 0.0
    sponge_timescale: float = 0.0
    sponge_power: float = 2.0
    sponge_target: str = "geostrophic"
    thermo_enabled: bool = False
    moisture_enabled: bool = False
    theta0: float = 300.0
    g: float = 9.81
    buoyancy_reference: str = "plane_mean"
    theta_bc: str = "flux"
    theta_profile: str = "linear"
    theta_top_gradient: float | None = None
    theta_bottom: float | None = None
    theta_top: float | None = None
    theta_initial_gradient: float = 0.0
    theta_perturbation_amplitude: float = 0.0
    theta_perturbation_height: float | None = None
    cbl_mixed_layer_height: float | None = None
    cbl_inversion_strength: float = 0.0
    cbl_inversion_thickness: float = 100.0
    cbl_free_atmosphere_gradient: float = 0.0
    surface_theta_flux: float = 0.0
    qv0: float = 0.0
    qv_initial_gradient: float = 0.0
    surface_qv_flux: float = 0.0
    qv_floor: float = 0.0
    surface_pressure: float = 100000.0
    initial_perturbation_height: float = 0.0
    scalar_sgs_model: str = "lasd"
    prandtl_t: float = 0.74
    schmidt_t: float = 0.74
    scalar_stability_correction: bool = False
    scalar_stability_beta: float = 10.0
    scalar_stability_power: float = 2.0
    scalar_lasd_min: float = 0.0
    scalar_lasd_max: float = 1.0
    scalar_vertical_scheme: str = "centered"
    actuator_disk_enabled: bool = False
    actuator_disk_x: float = 0.0
    actuator_disk_y: float = 0.0
    actuator_disk_z: float = 0.0
    actuator_disk_diameter: float = 1.0
    actuator_disk_hub_diameter: float = 0.0
    actuator_disk_ct_prime: float = 1.0
    actuator_disk_thickness: float = 0.1
    actuator_disk_yaw_degrees: float = 0.0
    cold_source_enabled: bool = False
    cold_source_x: float = 0.0
    cold_source_y: float = 0.0
    cold_source_z: float = 0.0
    cold_source_sigma_x: float = 0.1
    cold_source_sigma_r: float = 0.1
    cold_source_momentum_flux: float = 0.0
    cold_source_cooling_power: float = 0.0
    cold_source_density: float = 1.2
    cold_source_heat_capacity: float = 1005.0
    fringe_enabled: bool = False
    fringe_start_x: float = 0.0
    fringe_timescale: float = 1.0
    # Retained for restart/config compatibility.  The classic C-infinity
    # fringe shape has no tunable exponent.
    fringe_power: float = 2.0
    fringe_target_u: float = 0.0
    fringe_target_v: float = 0.0
    fringe_target_theta: float | None = None
    dtype: Any = jnp.float32
    sgs_dtype: Any | None = None
    use_jit: bool = True

    def __post_init__(self) -> None:
        if self.sgs_dtype is None:
            object.__setattr__(self, "sgs_dtype", self.dtype)
        if self.dt is None:
            object.__setattr__(self, "dt", default_dt_for_time_scheme(self.time_scheme))
        else:
            default_dt_for_time_scheme(self.time_scheme)
        if self.projection_mode not in {"stage", "final"}:
            raise ValueError(f"Unsupported projection_mode: {self.projection_mode}")
        sharded_pressure_solver = str(self.sharded_pressure_solver).lower()
        if sharded_pressure_solver not in {"transpose", "spike"}:
            raise ValueError(
                "sharded_pressure_solver must be 'transpose' or 'spike', "
                f"got {self.sharded_pressure_solver!r}"
            )
        object.__setattr__(
            self, "sharded_pressure_solver", sharded_pressure_solver
        )
        top_boundary = str(self.top_boundary_condition).lower()
        top_boundary_aliases = {
            "rigid_lid": "rigid_lid",
            "rigid-lid": "rigid_lid",
            "rigid": "rigid_lid",
            "klemp_durran": "klemp_durran",
            "klemp-durran": "klemp_durran",
            "radiation": "klemp_durran",
            "radiative": "klemp_durran",
        }
        if top_boundary not in top_boundary_aliases:
            raise ValueError(f"Unsupported top_boundary_condition: {self.top_boundary_condition}")
        object.__setattr__(self, "top_boundary_condition", top_boundary_aliases[top_boundary])
        if self.radiation_brunt_vaisala is not None and self.radiation_brunt_vaisala <= 0.0:
            raise ValueError(
                "radiation_brunt_vaisala must be positive, "
                f"got {self.radiation_brunt_vaisala:.6e}"
            )
        if self.top_boundary_condition == "klemp_durran":
            top_gradient = self.theta_top_gradient
            if self.radiation_brunt_vaisala is None and (top_gradient is None or top_gradient <= 0.0):
                raise ValueError(
                    "Klemp-Durran radiation requires either a positive "
                    "radiation_brunt_vaisala or theta_top_gradient"
                )
        if self.fgr <= 0.0:
            raise ValueError(f"fgr must be positive, got {self.fgr:.6e}")
        if self.tfr <= 0.0:
            raise ValueError(f"tfr must be positive, got {self.tfr:.6e}")
        sgs_model = str(self.sgs_model).lower()
        aliases = {
            "1": "smagorinsky",
            "smag": "smagorinsky",
            "smagorinsky": "smagorinsky",
            "classic": "smagorinsky",
            "3": "lasd",
            "lasd": "lasd",
            "porte_agel_sd": "porte_agel_sd",
            "porte-agel-sd": "porte_agel_sd",
            "porte_agel": "porte_agel_sd",
        }
        if sgs_model not in aliases:
            raise ValueError(f"Unsupported sgs_model: {self.sgs_model}")
        object.__setattr__(self, "sgs_model", aliases[sgs_model])
        if self.cs_count <= 0:
            raise ValueError(f"cs_count must be positive, got {self.cs_count}")
        if self.sgs_delta_scale is None:
            object.__setattr__(self, "sgs_delta_scale", self.fgr ** (2.0 / 3.0))
        elif self.sgs_delta_scale <= 0.0:
            raise ValueError(f"sgs_delta_scale must be positive, got {self.sgs_delta_scale:.6e}")
        if self.zo <= 0.0:
            raise ValueError(f"zo must be positive, got {self.zo:.6e}")
        initial_condition = str(self.initial_condition).lower()
        initial_condition_aliases = {
            "default": "default",
            "wireles": "default",
            "log_law": "default",
            "uniform": "uniform",
            "plug": "uniform",
            "pressure_driven": "uniform",
            "pressure-driven": "uniform",
            "geostrophic": "geostrophic",
            "geostrophic_wind": "geostrophic",
            "geostrophic-wind": "geostrophic",
            "ekman": "geostrophic",
            "neutral_ekman": "geostrophic",
            "uniform_flow": "uniform_flow",
            "uniform-flow": "uniform_flow",
            "wind_tunnel": "uniform_flow",
            "wind-tunnel": "uniform_flow",
        }
        if initial_condition not in initial_condition_aliases:
            raise ValueError(f"Unsupported initial_condition: {self.initial_condition}")
        object.__setattr__(self, "initial_condition", initial_condition_aliases[initial_condition])
        buoyancy_reference = str(self.buoyancy_reference).lower()
        buoyancy_reference_aliases = {
            "plane_mean": "plane_mean",
            "plane-mean": "plane_mean",
            "horizontal_mean": "plane_mean",
            "ambient": "ambient",
            "fixed_ambient": "ambient",
            "fixed-ambient": "ambient",
        }
        if buoyancy_reference not in buoyancy_reference_aliases:
            raise ValueError(f"Unsupported buoyancy_reference: {self.buoyancy_reference}")
        object.__setattr__(
            self, "buoyancy_reference", buoyancy_reference_aliases[buoyancy_reference]
        )
        momentum_wall_model = str(self.momentum_wall_model).lower()
        wall_aliases = {
            "abl": "abl",
            "log": "abl",
            "log_law": "abl",
            "free_slip": "free_slip",
            "free-slip": "free_slip",
            "freeslip": "free_slip",
            "no_slip": "no_slip",
            "no-slip": "no_slip",
            "noslip": "no_slip",
        }
        if momentum_wall_model not in wall_aliases:
            raise ValueError(f"Unsupported momentum_wall_model: {self.momentum_wall_model}")
        object.__setattr__(self, "momentum_wall_model", wall_aliases[momentum_wall_model])
        wall_stress_model = str(self.wall_stress_model).lower()
        wall_stress_aliases = {
            "dynamic": "dynamic_neutral",
            "dynamic_neutral": "dynamic_neutral",
            "dynamic-neutral": "dynamic_neutral",
            "neutral": "dynamic_neutral",
            "prescribed": "prescribed_ustar",
            "prescribed_ustar": "prescribed_ustar",
            "prescribed-ustar": "prescribed_ustar",
            "fixed": "prescribed_ustar",
            "fixed_ustar": "prescribed_ustar",
        }
        if wall_stress_model not in wall_stress_aliases:
            raise ValueError(f"Unsupported wall_stress_model: {self.wall_stress_model}")
        object.__setattr__(self, "wall_stress_model", wall_stress_aliases[wall_stress_model])
        domain_height = self.lz * self.z_i
        if self.pressure_force_height is not None and not (
            0.0 < self.pressure_force_height <= domain_height
        ):
            raise ValueError(
                "pressure_force_height must lie in (0, domain height], "
                f"got {self.pressure_force_height:.6e} for "
                f"domain height {domain_height:.6e}"
            )
        if self.initial_velocity_noise < 0.0:
            raise ValueError(f"initial_velocity_noise must be non-negative, got {self.initial_velocity_noise:.6e}")
        if self.molecular_viscosity < 0.0:
            raise ValueError(f"molecular_viscosity must be non-negative, got {self.molecular_viscosity:.6e}")
        if self.molecular_diffusivity < 0.0:
            raise ValueError(f"molecular_diffusivity must be non-negative, got {self.molecular_diffusivity:.6e}")
        if self.rayleigh_prandtl <= 0.0:
            raise ValueError(f"rayleigh_prandtl must be positive, got {self.rayleigh_prandtl:.6e}")
        if self.rayleigh_number is not None and self.rayleigh_number <= 0.0:
            raise ValueError(f"rayleigh_number must be positive, got {self.rayleigh_number:.6e}")
        sponge_target = str(self.sponge_target).lower()
        sponge_target_aliases = {
            "geostrophic": "geostrophic",
            "geostrophic_wind": "geostrophic",
            "geostrophic-wind": "geostrophic",
            "plane_mean": "plane_mean",
            "plane-mean": "plane_mean",
            "horizontal_mean": "plane_mean",
            "horizontal-mean": "plane_mean",
        }
        if sponge_target not in sponge_target_aliases:
            raise ValueError(f"Unsupported sponge_target: {self.sponge_target}")
        object.__setattr__(self, "sponge_target", sponge_target_aliases[sponge_target])
        if self.sponge_enabled:
            domain_height = self.lz * self.z_i
            if self.sponge_start_height < 0.0 or self.sponge_start_height >= domain_height:
                raise ValueError(
                    "sponge_start_height must lie in [0, domain height), "
                    f"got {self.sponge_start_height:.6e} for domain height {domain_height:.6e}"
                )
            if self.sponge_timescale <= 0.0:
                raise ValueError(
                    f"sponge_timescale must be positive, got {self.sponge_timescale:.6e}"
                )
            if self.sponge_power <= 0.0:
                raise ValueError(f"sponge_power must be positive, got {self.sponge_power:.6e}")
        if self.theta0 <= 0.0:
            raise ValueError(f"theta0 must be positive, got {self.theta0:.6e}")
        theta_bc = str(self.theta_bc).lower()
        theta_bc_aliases = {
            "flux": "flux",
            "neumann": "flux",
            "dirichlet": "dirichlet",
            "fixed": "dirichlet",
            "fixed_temperature": "dirichlet",
        }
        if theta_bc not in theta_bc_aliases:
            raise ValueError(f"Unsupported theta_bc: {self.theta_bc}")
        object.__setattr__(self, "theta_bc", theta_bc_aliases[theta_bc])
        if self.theta_bc == "dirichlet" and (self.theta_bottom is None or self.theta_top is None):
            raise ValueError("theta_bc='dirichlet' requires theta_bottom and theta_top.")
        theta_profile = str(self.theta_profile).lower()
        theta_profile_aliases = {
            "linear": "linear",
            "gradient": "linear",
            "uniform": "linear",
            "rayleigh": "linear",
            "deardorff": "deardorff_cbl",
            "deardorff_cbl": "deardorff_cbl",
            "cbl": "deardorff_cbl",
            "convective_boundary_layer": "deardorff_cbl",
        }
        if theta_profile not in theta_profile_aliases:
            raise ValueError(f"Unsupported theta_profile: {self.theta_profile}")
        object.__setattr__(self, "theta_profile", theta_profile_aliases[theta_profile])
        domain_height = self.lz * self.z_i
        if self.theta_profile == "deardorff_cbl":
            mixed_layer_height = self.cbl_mixed_layer_height
            if mixed_layer_height is None:
                mixed_layer_height = self.bl_height
            if mixed_layer_height <= 0.0 or mixed_layer_height >= domain_height:
                raise ValueError(
                    "cbl_mixed_layer_height must lie inside the domain, "
                    f"got {mixed_layer_height:.6e} for domain height {domain_height:.6e}"
                )
            object.__setattr__(self, "cbl_mixed_layer_height", mixed_layer_height)
            if self.cbl_inversion_strength < 0.0:
                raise ValueError(
                    "cbl_inversion_strength must be non-negative, "
                    f"got {self.cbl_inversion_strength:.6e}"
                )
            if self.cbl_inversion_thickness <= 0.0:
                raise ValueError(
                    "cbl_inversion_thickness must be positive, "
                    f"got {self.cbl_inversion_thickness:.6e}"
                )
            if self.cbl_inversion_thickness >= domain_height:
                raise ValueError(
                    "cbl_inversion_thickness must be smaller than the domain height, "
                    f"got {self.cbl_inversion_thickness:.6e} for {domain_height:.6e}"
                )
            if self.cbl_free_atmosphere_gradient < 0.0:
                raise ValueError(
                    "cbl_free_atmosphere_gradient should be non-negative for a capped CBL, "
                    f"got {self.cbl_free_atmosphere_gradient:.6e}"
                )
        if self.theta_perturbation_amplitude < 0.0:
            raise ValueError(
                "theta_perturbation_amplitude must be non-negative, "
                f"got {self.theta_perturbation_amplitude:.6e}"
            )
        if self.theta_perturbation_height is not None and self.theta_perturbation_height <= 0.0:
            raise ValueError(
                "theta_perturbation_height must be positive when set, "
                f"got {self.theta_perturbation_height:.6e}"
            )
        if self.prandtl_t <= 0.0:
            raise ValueError(f"prandtl_t must be positive, got {self.prandtl_t:.6e}")
        if self.schmidt_t <= 0.0:
            raise ValueError(f"schmidt_t must be positive, got {self.schmidt_t:.6e}")
        if self.scalar_stability_beta < 0.0:
            raise ValueError(
                "scalar_stability_beta must be non-negative, "
                f"got {self.scalar_stability_beta:.6e}"
            )
        if self.scalar_stability_power <= 0.0:
            raise ValueError(
                "scalar_stability_power must be positive, "
                f"got {self.scalar_stability_power:.6e}"
            )
        if self.qv_floor < 0.0:
            raise ValueError(f"qv_floor must be non-negative, got {self.qv_floor:.6e}")
        if self.surface_pressure <= 0.0:
            raise ValueError(
                f"surface_pressure must be positive, got {self.surface_pressure:.6e}"
            )
        if self.initial_perturbation_height < 0.0:
            raise ValueError(
                "initial_perturbation_height must be non-negative, "
                f"got {self.initial_perturbation_height:.6e}"
            )
        scalar_sgs_model = str(self.scalar_sgs_model).lower()
        scalar_aliases = {
            "fixed": "fixed_prandtl",
            "fixed_prandtl": "fixed_prandtl",
            "fixed-prandtl": "fixed_prandtl",
            "prandtl": "fixed_prandtl",
            "lasd": "lasd",
            "dynamic": "lasd",
            "dynamic_lasd": "lasd",
            "porte_agel_sd": "porte_agel_sd",
            "porte-agel-sd": "porte_agel_sd",
            "porte_agel": "porte_agel_sd",
        }
        if scalar_sgs_model not in scalar_aliases:
            raise ValueError(f"Unsupported scalar_sgs_model: {self.scalar_sgs_model}")
        object.__setattr__(self, "scalar_sgs_model", scalar_aliases[scalar_sgs_model])
        if self.sgs_model == "porte_agel_sd" and self.scalar_sgs_model == "lasd":
            raise ValueError(
                "momentum porte_agel_sd repurposes the LASD history slots; "
                "use scalar_sgs_model='porte_agel_sd' or 'fixed_prandtl'"
            )
        if self.scalar_lasd_min < 0.0:
            raise ValueError(f"scalar_lasd_min must be non-negative, got {self.scalar_lasd_min:.6e}")
        if self.scalar_lasd_max < self.scalar_lasd_min:
            raise ValueError(
                "scalar_lasd_max must be greater than or equal to scalar_lasd_min, "
                f"got {self.scalar_lasd_max:.6e} < {self.scalar_lasd_min:.6e}"
            )
        scalar_vertical_scheme = str(self.scalar_vertical_scheme).lower()
        scalar_vertical_aliases = {
            "centered": "centered",
            "central": "centered",
            "weno3": "weno3",
            "weno-3": "weno3",
            "weno5z": "weno5z",
            "weno5-z": "weno5z",
            "weno-z5": "weno5z",
        }
        if scalar_vertical_scheme not in scalar_vertical_aliases:
            raise ValueError(f"Unsupported scalar_vertical_scheme: {self.scalar_vertical_scheme}")
        object.__setattr__(
            self,
            "scalar_vertical_scheme",
            scalar_vertical_aliases[scalar_vertical_scheme],
        )
        if self.momentum_wall_model == "abl" and self.wall_ref_height <= self.zo:
            raise ValueError(
                "Invalid wall-model parameters: first off-wall reference height "
                f"0.5*dz*z_i = {self.wall_ref_height:.6e} must be larger than "
                f"zo = {self.zo:.6e} so log(z/zo) is positive. "
                f"For this grid use --zo below {self.wall_ref_height:.6e}, "
                "or increase --z-i/--lz."
            )
        if not self.horizontal_homogeneous:
            if self.sgs_model == "porte_agel_sd":
                raise ValueError(
                    "porte_agel_sd momentum SGS uses horizontal-plane averaging "
                    "and is invalid when horizontal_homogeneous=false; use LASD"
                )
            if self.scalar_sgs_model == "porte_agel_sd":
                raise ValueError(
                    "porte_agel_sd scalar SGS uses horizontal-plane averaging "
                    "and is invalid when horizontal_homogeneous=false; use LASD"
                )
            if self.thermo_enabled and self.buoyancy_reference == "plane_mean":
                raise ValueError(
                    "plane-mean buoyancy is invalid when horizontal_homogeneous=false; "
                    "use buoyancy_reference='ambient'"
                )
            if self.sponge_enabled and self.sponge_target == "plane_mean":
                raise ValueError(
                    "plane-mean sponge is invalid when horizontal_homogeneous=false"
                )
        domain_x = self.lx * self.z_i
        domain_y = self.ly * self.z_i
        domain_z = self.lz * self.z_i
        if self.actuator_disk_enabled:
            if self.actuator_disk_diameter <= 0.0:
                raise ValueError("actuator_disk_diameter must be positive")
            if not 0.0 <= self.actuator_disk_hub_diameter < self.actuator_disk_diameter:
                raise ValueError(
                    "actuator_disk_hub_diameter must lie in [0, actuator_disk_diameter)"
                )
            if self.actuator_disk_ct_prime < 0.0:
                raise ValueError("actuator_disk_ct_prime must be non-negative")
            if self.actuator_disk_thickness <= 0.0:
                raise ValueError("actuator_disk_thickness must be positive")
            if not math.isfinite(self.actuator_disk_yaw_degrees):
                raise ValueError("actuator_disk_yaw_degrees must be finite")
            if not (0.0 <= self.actuator_disk_x < domain_x):
                raise ValueError("actuator_disk_x must lie inside the physical domain")
            if not (0.0 <= self.actuator_disk_y < domain_y):
                raise ValueError("actuator_disk_y must lie inside the physical domain")
            if not (0.0 < self.actuator_disk_z < domain_z):
                raise ValueError("actuator_disk_z must lie inside the physical domain")
        if self.cold_source_enabled:
            if self.cold_source_sigma_x <= 0.0 or self.cold_source_sigma_r <= 0.0:
                raise ValueError("cold-source Gaussian widths must be positive")
            if self.cold_source_momentum_flux < 0.0:
                raise ValueError("cold_source_momentum_flux must be non-negative")
            if self.cold_source_cooling_power < 0.0:
                raise ValueError("cold_source_cooling_power must be non-negative")
            if self.cold_source_density <= 0.0 or self.cold_source_heat_capacity <= 0.0:
                raise ValueError("cold-source density and heat capacity must be positive")
            if not (0.0 <= self.cold_source_x < domain_x):
                raise ValueError("cold_source_x must lie inside the physical domain")
            if not (0.0 <= self.cold_source_y < domain_y):
                raise ValueError("cold_source_y must lie inside the physical domain")
            if not (0.0 < self.cold_source_z < domain_z):
                raise ValueError("cold_source_z must lie inside the physical domain")
        if self.fringe_enabled:
            if not 0.0 <= self.fringe_start_x < domain_x:
                raise ValueError("fringe_start_x must lie inside the physical domain")
            if self.fringe_timescale <= 0.0:
                raise ValueError("fringe_timescale must be positive")
            if self.fringe_power <= 0.0:
                raise ValueError("fringe_power must be positive")

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def dz(self) -> float:
        return self.lz / self.nz

    @property
    def sgs_delta(self) -> float:
        return self.sgs_delta_scale * (self.dx * self.dy * self.dz) ** (1.0 / 3.0)

    @property
    def wall_ref_height(self) -> float:
        return 0.5 * self.dz * self.z_i

    @property
    def dt_physical(self) -> float:
        return self.dt * self.z_i

    @property
    def l_r(self) -> int:
        return max(1, int(round(self.lx / self.ly)))

    @property
    def driving_pressure_force(self) -> float:
        if self.pressure_force is not None:
            return self.pressure_force
        return self.u_fric * self.u_fric / self.forcing_height

    @property
    def pressure_ustar(self) -> float:
        """Friction-velocity scale implied by the integrated body force."""
        return max(self.driving_pressure_force * self.forcing_height, 0.0) ** 0.5

    @property
    def coriolis_f_internal(self) -> float:
        return self.coriolis_f * self.z_i

    @property
    def radiation_brunt_vaisala_physical(self) -> float:
        """Brunt--Vaisala frequency used by the hydrostatic radiation condition."""
        if self.radiation_brunt_vaisala is not None:
            return self.radiation_brunt_vaisala
        if self.theta_top_gradient is None:
            return 0.0
        return max(self.g * self.theta_top_gradient / self.theta_v0, 0.0) ** 0.5

    @property
    def radiation_brunt_vaisala_internal(self) -> float:
        return self.radiation_brunt_vaisala_physical * self.z_i

    @property
    def forcing_height(self) -> float:
        physical_height = (
            self.bl_height
            if self.pressure_force_height is None
            else self.pressure_force_height
        )
        return physical_height / self.z_i

    @property
    def theta_v0(self) -> float:
        qv_ref = self.qv0 if self.moisture_enabled else 0.0
        return self.theta0 * (1.0 + 0.61 * qv_ref)

    @property
    def transported_surface_qv_flux(self) -> float:
        return self.surface_qv_flux

    @property
    def rayleigh_delta_theta(self) -> float:
        if self.theta_bottom is None or self.theta_top is None:
            return 0.0
        return abs(self.theta_bottom - self.theta_top)

    @property
    def rayleigh_molecular_diffusivity(self) -> float:
        if self.rayleigh_number is None:
            return self.molecular_diffusivity
        delta_theta = self.rayleigh_delta_theta
        if delta_theta <= 0.0:
            raise ValueError("rayleigh_number requires theta_bottom and theta_top to differ.")
        height = self.lz * self.z_i
        freefall2 = self.g * delta_theta * height / self.theta_v0
        return (freefall2 * height * height / (self.rayleigh_number * self.rayleigh_prandtl)) ** 0.5

    @property
    def rayleigh_molecular_viscosity(self) -> float:
        if self.rayleigh_number is None:
            return self.molecular_viscosity
        return self.rayleigh_prandtl * self.rayleigh_molecular_diffusivity

    @property
    def molecular_diffusivity_internal(self) -> float:
        return self.rayleigh_molecular_diffusivity / self.z_i

    @property
    def molecular_viscosity_internal(self) -> float:
        return self.rayleigh_molecular_viscosity / self.z_i

    def with_overrides(self, **kwargs: Any) -> "Params":
        return replace(self, **kwargs)
