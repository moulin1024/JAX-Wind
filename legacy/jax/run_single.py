#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


RUN_DEFAULTS = {
    "nx": 32,
    "ny": 32,
    "nz": 32,
    "lx": 2.0 * 3.141592653589793,
    "ly": 2.0 * 3.141592653589793,
    "lz": 1.0,
    "steps": 10,
    "dt": None,
    "log_every": 1,
    "u_fric": 0.4,
    "zo": 5.0e-3,
    "bl_height": 1.0,
    "z_i": None,
    "vonk": 0.4,
    "pressure_force": None,
    "coriolis_f": 0.0,
    "geostrophic_u": 0.0,
    "geostrophic_v": 0.0,
    "uniform_u": 0.0,
    "uniform_v": 0.0,
    "horizontal_homogeneous": True,
    "initial_condition": "default",
    "momentum_wall_model": "abl",
    "wall_stress_model": "dynamic_neutral",
    "initial_velocity_noise": 0.01,
    "molecular_viscosity": 0.0,
    "molecular_diffusivity": 0.0,
    "rayleigh_number": None,
    "rayleigh_prandtl": 1.0,
    "fgr": 1.5,
    "tfr": 2.0,
    "sgs_model": "smagorinsky",
    "cs_count": 10,
    "lasd_scale_dependent": True,
    "momentum_lasd_scale_dependent": None,
    "scalar_lasd_scale_dependent": None,
    "lasd_invalid_beta_fallback": False,
    "lasd_clipped_beta_fallback": False,
    "smag_cs": 0.16,
    "sgs_delta_scale": None,
    "time_scheme": "ab2",
    "projection_mode": "stage",
    "horizontal_dealias": True,
    "pressure_filter_nyquist": False,
    "sharded_pressure_solver": "transpose",
    "top_boundary_condition": "rigid_lid",
    "radiation_brunt_vaisala": None,
    "sponge_enabled": False,
    "sponge_start_height": 0.0,
    "sponge_timescale": 0.0,
    "sponge_power": 2.0,
    "sponge_target": "geostrophic",
    "thermo_enabled": False,
    "moisture_enabled": False,
    "theta0": 300.0,
    "g": 9.81,
    "buoyancy_reference": "plane_mean",
    "theta_bc": "flux",
    "theta_profile": "linear",
    "theta_top_gradient": None,
    "theta_bottom": None,
    "theta_top": None,
    "theta_initial_gradient": 0.0,
    "theta_perturbation_amplitude": 0.0,
    "theta_perturbation_height": None,
    "cbl_mixed_layer_height": None,
    "cbl_inversion_strength": 0.0,
    "cbl_inversion_thickness": 100.0,
    "cbl_free_atmosphere_gradient": 0.0,
    "surface_theta_flux": 0.0,
    "qv0": 0.0,
    "qv_initial_gradient": 0.0,
    "surface_qv_flux": 0.0,
    "qv_floor": 0.0,
    "surface_pressure": 100000.0,
    "initial_perturbation_height": 0.0,
    "scalar_sgs_model": "lasd",
    "prandtl_t": 0.74,
    "schmidt_t": 0.74,
    "scalar_stability_correction": False,
    "scalar_stability_beta": 10.0,
    "scalar_stability_power": 2.0,
    "scalar_lasd_min": 0.0,
    "scalar_lasd_max": 1.0,
    "scalar_vertical_scheme": "centered",
    "actuator_disk_enabled": False,
    "actuator_disk_x": 0.0,
    "actuator_disk_y": 0.0,
    "actuator_disk_z": 0.0,
    "actuator_disk_diameter": 1.0,
    "actuator_disk_hub_diameter": 0.0,
    "actuator_disk_ct_prime": 1.0,
    "actuator_disk_thickness": 0.1,
    "actuator_disk_yaw_degrees": 0.0,
    "cold_source_enabled": False,
    "cold_source_x": 0.0,
    "cold_source_y": 0.0,
    "cold_source_z": 0.0,
    "cold_source_sigma_x": 0.1,
    "cold_source_sigma_r": 0.1,
    "cold_source_momentum_flux": 0.0,
    "cold_source_cooling_power": 0.0,
    "cold_source_density": 1.2,
    "cold_source_heat_capacity": 1005.0,
    "fringe_enabled": False,
    "fringe_start_x": 0.0,
    "fringe_timescale": 1.0,
    "fringe_target_u": 0.0,
    "fringe_target_v": 0.0,
    "fringe_target_theta": None,
    "precision": "float64",
    "sgs_precision": "float32",
    "use_jit": True,
    "checkpoint": None,
    "dump_fields": False,
    "field_dump_start_step": 0,
    "field_output_dir": None,
    "field_output_prefix": "fields",
    "profile": False,
    "profile_steps": None,
    "profile_warmup": 0,
}


CONFIG_KEYS = {
    "grid": {
        "nx": "nx",
        "ny": "ny",
        "nz": "nz",
        "lx": "lx",
        "ly": "ly",
        "lz": "lz",
        "z_i": "z_i",
    },
    "time": {
        "steps": "steps",
        "nsteps": "steps",
        "dt": "dt",
        "log_every": "log_every",
        "c_count": "log_every",
        "scheme": "time_scheme",
        "time_scheme": "time_scheme",
    },
    "physics": {
        "u_fric": "u_fric",
        "zo": "zo",
        "bl_height": "bl_height",
        "vonk": "vonk",
        "pressure_force": "pressure_force",
        "coriolis_f": "coriolis_f",
        "geostrophic_u": "geostrophic_u",
        "geostrophic_v": "geostrophic_v",
        "uniform_u": "uniform_u",
        "uniform_v": "uniform_v",
        "horizontal_homogeneous": "horizontal_homogeneous",
        "initial_condition": "initial_condition",
        "momentum_wall_model": "momentum_wall_model",
        "wall_stress_model": "wall_stress_model",
        "initial_velocity_noise": "initial_velocity_noise",
        "molecular_viscosity": "molecular_viscosity",
        "molecular_diffusivity": "molecular_diffusivity",
        "rayleigh_number": "rayleigh_number",
        "rayleigh_prandtl": "rayleigh_prandtl",
        "initial_perturbation_height": "initial_perturbation_height",
    },
    "wall_filter": {
        "fgr": "fgr",
        "tfr": "tfr",
    },
    "sgs": {
        "model": "sgs_model",
        "sgs_model": "sgs_model",
        "cs_count": "cs_count",
        "scale_dependent": "lasd_scale_dependent",
        "lasd_scale_dependent": "lasd_scale_dependent",
        "momentum_scale_dependent": "momentum_lasd_scale_dependent",
        "momentum_lasd_scale_dependent": "momentum_lasd_scale_dependent",
        "scalar_scale_dependent": "scalar_lasd_scale_dependent",
        "scalar_lasd_scale_dependent": "scalar_lasd_scale_dependent",
        "invalid_beta_fallback": "lasd_invalid_beta_fallback",
        "lasd_invalid_beta_fallback": "lasd_invalid_beta_fallback",
        "clipped_beta_fallback": "lasd_clipped_beta_fallback",
        "lasd_clipped_beta_fallback": "lasd_clipped_beta_fallback",
        "smag_cs": "smag_cs",
        "smagorinsky_cs": "smag_cs",
        "delta_scale": "sgs_delta_scale",
        "sgs_delta_scale": "sgs_delta_scale",
        "precision": "sgs_precision",
        "sgs_precision": "sgs_precision",
    },
    "numerics": {
        "projection_mode": "projection_mode",
        "horizontal_dealias": "horizontal_dealias",
        "pressure_filter_nyquist": "pressure_filter_nyquist",
        "sharded_pressure_solver": "sharded_pressure_solver",
        "top_boundary_condition": "top_boundary_condition",
        "radiation_brunt_vaisala": "radiation_brunt_vaisala",
    },
    "sponge": {
        "enabled": "sponge_enabled",
        "start_height": "sponge_start_height",
        "timescale": "sponge_timescale",
        "power": "sponge_power",
        "target": "sponge_target",
    },
    "thermo": {
        "enabled": "thermo_enabled",
        "thermo_enabled": "thermo_enabled",
        "moisture_enabled": "moisture_enabled",
        "theta0": "theta0",
        "g": "g",
        "buoyancy_reference": "buoyancy_reference",
        "theta_bc": "theta_bc",
        "theta_profile": "theta_profile",
        "theta_top_gradient": "theta_top_gradient",
        "theta_bottom": "theta_bottom",
        "theta_top": "theta_top",
        "theta_initial_gradient": "theta_initial_gradient",
        "theta_perturbation_amplitude": "theta_perturbation_amplitude",
        "theta_perturbation_height": "theta_perturbation_height",
        "cbl_mixed_layer_height": "cbl_mixed_layer_height",
        "cbl_inversion_strength": "cbl_inversion_strength",
        "cbl_inversion_thickness": "cbl_inversion_thickness",
        "cbl_free_atmosphere_gradient": "cbl_free_atmosphere_gradient",
        "surface_theta_flux": "surface_theta_flux",
        "qv0": "qv0",
        "qv_initial_gradient": "qv_initial_gradient",
        "surface_qv_flux": "surface_qv_flux",
        "qv_floor": "qv_floor",
        "surface_pressure": "surface_pressure",
        "molecular_diffusivity": "molecular_diffusivity",
        "scalar_sgs_model": "scalar_sgs_model",
        "prandtl_t": "prandtl_t",
        "schmidt_t": "schmidt_t",
        "scalar_stability_correction": "scalar_stability_correction",
        "scalar_stability_beta": "scalar_stability_beta",
        "scalar_stability_power": "scalar_stability_power",
        "scalar_lasd_min": "scalar_lasd_min",
        "scalar_lasd_max": "scalar_lasd_max",
        "scalar_vertical_scheme": "scalar_vertical_scheme",
    },
    "actuator_disk": {
        "enabled": "actuator_disk_enabled",
        "x": "actuator_disk_x",
        "y": "actuator_disk_y",
        "z": "actuator_disk_z",
        "diameter": "actuator_disk_diameter",
        "hub_diameter": "actuator_disk_hub_diameter",
        "ct_prime": "actuator_disk_ct_prime",
        "thickness": "actuator_disk_thickness",
        "yaw_degrees": "actuator_disk_yaw_degrees",
    },
    "cold_source": {
        "enabled": "cold_source_enabled",
        "x": "cold_source_x",
        "y": "cold_source_y",
        "z": "cold_source_z",
        "sigma_x": "cold_source_sigma_x",
        "sigma_r": "cold_source_sigma_r",
        "momentum_flux": "cold_source_momentum_flux",
        "cooling_power": "cold_source_cooling_power",
        "density": "cold_source_density",
        "heat_capacity": "cold_source_heat_capacity",
    },
    "fringe": {
        "enabled": "fringe_enabled",
        "start_x": "fringe_start_x",
        "timescale": "fringe_timescale",
        "target_u": "fringe_target_u",
        "target_v": "fringe_target_v",
        "target_theta": "fringe_target_theta",
    },
    "runtime": {
        "precision": "precision",
        "sgs_precision": "sgs_precision",
        "use_jit": "use_jit",
    },
    "output": {
        "checkpoint": "checkpoint",
        "dump_fields": "dump_fields",
        "field_dump_start_step": "field_dump_start_step",
        "field_output_dir": "field_output_dir",
        "field_output_prefix": "field_output_prefix",
    },
    "postprocess": {
        "dump_fields": "dump_fields",
        "field_dump_start_step": "field_dump_start_step",
        "field_output_dir": "field_output_dir",
        "field_output_prefix": "field_output_prefix",
    },
    "profiling": {
        "enabled": "profile",
        "profile": "profile",
        "steps": "profile_steps",
        "profile_steps": "profile_steps",
        "warmup": "profile_warmup",
        "profile_warmup": "profile_warmup",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-process JAX WiRE-LES prototype.")
    parser.add_argument("--config", type=Path, help="TOML run configuration.")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nz", type=int)
    parser.add_argument("--lx", type=float)
    parser.add_argument("--ly", type=float)
    parser.add_argument("--lz", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--dt", type=float, help="Physical time step in seconds; normalized internally by z_i.")
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--u-fric", type=float)
    parser.add_argument("--zo", type=float)
    parser.add_argument("--bl-height", type=float)
    parser.add_argument("--z-i", type=float)
    parser.add_argument("--pressure-force", type=float)
    parser.add_argument("--coriolis-f", type=float, help="Physical Coriolis parameter f in s^-1.")
    parser.add_argument("--geostrophic-u", type=float, help="Physical geostrophic wind U_g in m/s.")
    parser.add_argument("--geostrophic-v", type=float, help="Physical geostrophic wind V_g in m/s.")
    parser.add_argument(
        "--initial-condition",
        choices=(
            "default",
            "wireles",
            "log_law",
            "uniform",
            "plug",
            "pressure_driven",
            "pressure-driven",
            "geostrophic",
            "geostrophic_wind",
            "geostrophic-wind",
            "ekman",
            "neutral_ekman",
            "uniform_flow",
            "uniform-flow",
            "wind_tunnel",
            "wind-tunnel",
        ),
    )
    parser.add_argument("--momentum-wall-model", choices=("abl", "free_slip", "free-slip", "no_slip", "no-slip"))
    parser.add_argument("--wall-stress-model", choices=("dynamic_neutral", "dynamic-neutral", "prescribed_ustar", "prescribed-ustar"))
    parser.add_argument("--initial-velocity-noise", type=float)
    parser.add_argument("--molecular-viscosity", type=float)
    parser.add_argument("--molecular-diffusivity", type=float)
    parser.add_argument("--rayleigh-number", type=float)
    parser.add_argument("--rayleigh-prandtl", type=float)
    parser.add_argument("--fgr", type=float)
    parser.add_argument("--tfr", type=float)
    parser.add_argument("--sgs-model", choices=("smagorinsky", "lasd"))
    parser.add_argument("--cs-count", type=int)
    parser.add_argument("--smag-cs", type=float)
    parser.add_argument("--sgs-delta-scale", type=float)
    parser.add_argument("--time-scheme", choices=("rk3", "rk4", "ab2"))
    parser.add_argument("--projection-mode", choices=("stage", "final"))
    parser.add_argument("--horizontal-dealias", action="store_true", dest="horizontal_dealias", default=None)
    parser.add_argument("--no-horizontal-dealias", action="store_false", dest="horizontal_dealias")
    parser.add_argument("--pressure-filter-nyquist", action="store_true", dest="pressure_filter_nyquist", default=None)
    parser.add_argument("--no-pressure-filter-nyquist", action="store_false", dest="pressure_filter_nyquist")
    parser.add_argument(
        "--top-boundary-condition",
        choices=("rigid_lid", "rigid-lid", "klemp_durran", "klemp-durran", "radiation"),
    )
    parser.add_argument(
        "--radiation-brunt-vaisala",
        type=float,
        help="Optional physical Brunt-Vaisala frequency N in s^-1 for the Klemp-Durran top.",
    )
    parser.add_argument("--sponge", action="store_true", dest="sponge_enabled", default=None)
    parser.add_argument("--no-sponge", action="store_false", dest="sponge_enabled")
    parser.add_argument("--sponge-start-height", type=float)
    parser.add_argument("--sponge-timescale", type=float)
    parser.add_argument("--sponge-power", type=float)
    parser.add_argument("--sponge-target", choices=("geostrophic", "plane_mean", "plane-mean"))
    parser.add_argument("--thermo", action="store_true", dest="thermo_enabled", default=None)
    parser.add_argument("--no-thermo", action="store_false", dest="thermo_enabled")
    parser.add_argument("--moisture", action="store_true", dest="moisture_enabled", default=None)
    parser.add_argument("--no-moisture", action="store_false", dest="moisture_enabled")
    parser.add_argument("--theta0", type=float)
    parser.add_argument("--g", type=float)
    parser.add_argument("--theta-bc", choices=("flux", "neumann", "dirichlet", "fixed", "fixed_temperature"))
    parser.add_argument(
        "--theta-profile",
        choices=("linear", "gradient", "uniform", "deardorff", "deardorff_cbl", "cbl"),
    )
    parser.add_argument("--theta-top-gradient", type=float)
    parser.add_argument("--theta-bottom", type=float)
    parser.add_argument("--theta-top", type=float)
    parser.add_argument("--theta-initial-gradient", type=float)
    parser.add_argument("--theta-perturbation-amplitude", type=float)
    parser.add_argument("--theta-perturbation-height", type=float)
    parser.add_argument("--cbl-mixed-layer-height", type=float)
    parser.add_argument("--cbl-inversion-strength", type=float)
    parser.add_argument("--cbl-inversion-thickness", type=float)
    parser.add_argument("--cbl-free-atmosphere-gradient", type=float)
    parser.add_argument("--surface-theta-flux", type=float)
    parser.add_argument("--qv0", type=float)
    parser.add_argument("--qv-initial-gradient", type=float)
    parser.add_argument("--surface-qv-flux", type=float)
    parser.add_argument("--qv-floor", type=float)
    parser.add_argument(
        "--scalar-sgs-model",
        choices=("lasd", "porte_agel_sd", "porte-agel-sd", "fixed_prandtl", "fixed-prandtl"),
    )
    parser.add_argument("--prandtl-t", type=float)
    parser.add_argument("--schmidt-t", type=float)
    parser.add_argument("--scalar-stability-correction", action="store_true", dest="scalar_stability_correction", default=None)
    parser.add_argument("--no-scalar-stability-correction", action="store_false", dest="scalar_stability_correction")
    parser.add_argument("--scalar-stability-beta", type=float)
    parser.add_argument("--scalar-stability-power", type=float)
    parser.add_argument("--scalar-lasd-min", type=float)
    parser.add_argument("--scalar-lasd-max", type=float)
    parser.add_argument("--precision", choices=("float64", "float32"))
    parser.add_argument("--sgs-precision", choices=("default", "float64", "float32"))
    parser.add_argument("--single", action="store_const", const="float32", dest="precision")
    parser.add_argument("--no-jit", action="store_false", dest="use_jit", default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dump-fields", action="store_true", dest="dump_fields", default=None)
    parser.add_argument("--no-dump-fields", action="store_false", dest="dump_fields")
    parser.add_argument("--field-dump-start-step", type=int)
    parser.add_argument("--field-output-dir", type=Path)
    parser.add_argument("--field-output-prefix")
    parser.add_argument("--profile", action="store_true", dest="profile", default=None)
    parser.add_argument("--no-profile", action="store_false", dest="profile")
    parser.add_argument("--profile-steps", type=int, help="Optional profiled run length; defaults to [time].steps.")
    parser.add_argument("--profile-warmup", type=int, help="Completed initial steps excluded from profile averages.")
    return parser.parse_args()


def load_config_file(path: Path) -> dict:
    with path.open("rb") as handle:
        data = tomllib.load(handle)

    values = {}
    for section, section_values in data.items():
        if section not in CONFIG_KEYS:
            valid = ", ".join(sorted(CONFIG_KEYS))
            raise ValueError(f"Unknown config section [{section}]. Valid sections: {valid}")
        if not isinstance(section_values, dict):
            raise ValueError(f"Config section [{section}] must be a table.")

        key_map = CONFIG_KEYS[section]
        for key, value in section_values.items():
            if key not in key_map:
                valid = ", ".join(sorted(key_map))
                raise ValueError(f"Unknown config key [{section}].{key}. Valid keys: {valid}")
            values[key_map[key]] = value

    return values


def merged_settings(args: argparse.Namespace) -> dict:
    settings = dict(RUN_DEFAULTS)
    if args.config is not None:
        settings.update(load_config_file(args.config))

    for key in RUN_DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value
    if settings["checkpoint"] is not None:
        settings["checkpoint"] = Path(settings["checkpoint"])
    if settings["field_output_dir"] is not None:
        settings["field_output_dir"] = Path(settings["field_output_dir"])
    if settings["field_dump_start_step"] < 0:
        raise ValueError("field_dump_start_step must be non-negative.")
    if settings["profile_steps"] is not None and settings["profile_steps"] <= 0:
        raise ValueError("profile_steps must be positive.")
    if settings["profile_warmup"] < 0:
        raise ValueError("profile_warmup must be non-negative.")
    return settings


def dtype_for_precision(precision: str, jnp):
    if precision == "float32":
        return jnp.float32
    if precision == "float64":
        return jnp.float64
    raise ValueError(f"Unsupported precision: {precision}")


def sgs_dtype_for_precision(precision: str, solver_dtype, jnp):
    if precision == "default":
        return solver_dtype
    return dtype_for_precision(precision, jnp)


def scaled_grid_lengths(settings: dict) -> tuple[float, float, float, float]:
    lx_physical = float(settings["lx"])
    ly_physical = float(settings["ly"])
    lz_physical = float(settings["lz"])
    z_i = float(settings["z_i"] if settings["z_i"] is not None else lz_physical)
    if lx_physical <= 0.0 or ly_physical <= 0.0 or lz_physical <= 0.0:
        raise ValueError("Physical grid lengths lx, ly, and lz must all be positive.")
    if z_i <= 0.0:
        raise ValueError(f"z_i must be positive, got {z_i:.6e}")
    return lx_physical / z_i, ly_physical / z_i, lz_physical / z_i, z_i


def scaled_time_step(settings: dict, z_i: float) -> float | None:
    if settings["dt"] is None:
        return None
    dt_physical = float(settings["dt"])
    if dt_physical <= 0.0:
        raise ValueError(f"dt must be positive, got {dt_physical:.6e}")
    return dt_physical / z_i


def format_diagnostic(diag, cs_count: int = 1) -> str:
    lasd_cfl = cs_count * max(float(diag.cfl_x), float(diag.cfl_y), float(diag.cfl_z))
    return (
        f"{int(diag.step):5d} "
        f"{float(diag.remaining_s):9.1f} "
        f"{float(diag.total_s):9.1f} "
        f"{float(diag.elapsed_s):9.1f} "
        f"{float(diag.ustar):11.4e} "
        f"{float(diag.ke_max):11.4e} "
        f"{float(diag.div_max):11.4e} "
        f"{float(diag.cfl_x):10.4f} "
        f"{float(diag.cfl_y):10.4f} "
        f"{float(diag.cfl_z):10.4f} "
        f"{lasd_cfl:10.4f} "
        f"{float(diag.theta_v_min):11.4e} "
        f"{float(diag.qv_min):11.4e} "
        f"{float(diag.qv_floor_hits):9.0f}"
    )


LOG_HEADER = (
    " step    rest_s   total_s elapsed_s      ustar        ke_max       div_max       "
    "cfl_x       cfl_y       cfl_z    lasd_cfl  theta_v_min      qv_min qv_floor"
)

def _avg(rows, attr: str) -> float:
    n = float(len(rows))
    if n == 0.0:
        return 0.0
    return sum(getattr(row, attr) for row in rows) / n


def print_profile_report(rows, simulated_steps: int, warmup_steps: int) -> None:
    if not rows:
        print("[profile] no measured steps; reduce profile_warmup or increase steps", flush=True)
        return
    solver_ms = _avg(rows, "solver_ms")
    total_ms = _avg(rows, "total_ms")
    module_groups = (
        (
            "rhs",
            (
                ("velocity_xy_derivatives", _avg(rows, "velocity_xy_ms")),
                ("wall_z_derivatives", _avg(rows, "wall_z_ms")),
                ("convection", _avg(rows, "convection_ms")),
                ("sgs_strain", _avg(rows, "sgs_strain_ms")),
                ("lasd_coefficients", _avg(rows, "lasd_coefficients_ms")),
                ("sgs_stress", _avg(rows, "sgs_stress_ms")),
                ("sgs_total", _avg(rows, "sgs_ms")),
                ("stress_divergence", _avg(rows, "stress_divergence_ms")),
                ("rhs_assembly", _avg(rows, "rhs_assembly_ms")),
                ("rhs_total", _avg(rows, "rhs_ms")),
            ),
        ),
        (
            "time",
            (
                ("ab_update", _avg(rows, "ab_update_ms")),
            ),
        ),
        (
            "projection",
            (
                ("projection_divergence", _avg(rows, "projection_divergence_ms")),
                ("pressure_solve", _avg(rows, "pressure_solve_ms")),
                ("pressure_gradient_ifft", _avg(rows, "pressure_gradient_ms")),
                ("projection_update", _avg(rows, "projection_update_ms")),
                ("projection_total", _avg(rows, "projection_ms")),
            ),
        ),
        (
            "state",
            (
                ("state_pack", _avg(rows, "state_pack_ms")),
            ),
        ),
    )
    print("", flush=True)
    print("[profile] average module time per completed AB2 step", flush=True)
    print(
        f"[profile] simulated_steps={simulated_steps} "
        f"measured_steps={len(rows)} warmup_excluded={warmup_steps}",
        flush=True,
    )
    print(" group       module                         avg_ms    pct_solver", flush=True)
    for group, modules in module_groups:
        for name, avg_ms in modules:
            pct = 100.0 * avg_ms / solver_ms if solver_ms > 0.0 else 0.0
            print(f" {group:<10} {name:<28} {avg_ms:9.3f} {pct:11.1f}", flush=True)
    print(f" {'total':<10} {'solver_total':<28} {solver_ms:9.3f} {100.0:11.1f}", flush=True)
    print(f" {'total':<10} {'diagnostics':<28} {_avg(rows, 'diagnostics_ms'):9.3f} {'':>11}", flush=True)
    print(f" {'total':<10} {'profile_total':<28} {total_ms:9.3f} {'':>11}", flush=True)
    print(f"[profile] max div_max over measured steps: {max(row.div_max for row in rows):.4e}", flush=True)


def params_from_settings(settings: dict, jnp):
    """Build solver parameters from the shared CLI/TOML settings mapping."""
    from wireles_jax import Params

    lx_scaled, ly_scaled, lz_scaled, z_i = scaled_grid_lengths(settings)
    dt_scaled = scaled_time_step(settings, z_i)
    solver_dtype = dtype_for_precision(settings["precision"], jnp)
    return Params(
        nx=settings["nx"],
        ny=settings["ny"],
        nz=settings["nz"],
        lx=lx_scaled,
        ly=ly_scaled,
        lz=lz_scaled,
        nsteps=settings["steps"],
        dt=dt_scaled,
        c_count=settings["log_every"],
        u_fric=settings["u_fric"],
        zo=settings["zo"],
        bl_height=settings["bl_height"],
        z_i=z_i,
        vonk=settings["vonk"],
        pressure_force=settings["pressure_force"],
        coriolis_f=settings["coriolis_f"],
        geostrophic_u=settings["geostrophic_u"],
        geostrophic_v=settings["geostrophic_v"],
        uniform_u=settings["uniform_u"],
        uniform_v=settings["uniform_v"],
        horizontal_homogeneous=settings["horizontal_homogeneous"],
        initial_condition=settings["initial_condition"],
        momentum_wall_model=settings["momentum_wall_model"],
        wall_stress_model=settings["wall_stress_model"],
        initial_velocity_noise=settings["initial_velocity_noise"],
        molecular_viscosity=settings["molecular_viscosity"],
        molecular_diffusivity=settings["molecular_diffusivity"],
        rayleigh_number=settings["rayleigh_number"],
        rayleigh_prandtl=settings["rayleigh_prandtl"],
        fgr=settings["fgr"],
        tfr=settings["tfr"],
        sgs_model=settings["sgs_model"],
        cs_count=settings["cs_count"],
        lasd_scale_dependent=settings["lasd_scale_dependent"],
        momentum_lasd_scale_dependent=settings["momentum_lasd_scale_dependent"],
        scalar_lasd_scale_dependent=settings["scalar_lasd_scale_dependent"],
        lasd_invalid_beta_fallback=settings["lasd_invalid_beta_fallback"],
        lasd_clipped_beta_fallback=settings["lasd_clipped_beta_fallback"],
        smagorinsky_cs=settings["smag_cs"],
        sgs_delta_scale=settings["sgs_delta_scale"],
        time_scheme=settings["time_scheme"],
        projection_mode=settings["projection_mode"],
        horizontal_dealias=settings["horizontal_dealias"],
        pressure_filter_nyquist=settings["pressure_filter_nyquist"],
        sharded_pressure_solver=settings["sharded_pressure_solver"],
        top_boundary_condition=settings["top_boundary_condition"],
        radiation_brunt_vaisala=settings["radiation_brunt_vaisala"],
        sponge_enabled=settings["sponge_enabled"],
        sponge_start_height=settings["sponge_start_height"],
        sponge_timescale=settings["sponge_timescale"],
        sponge_power=settings["sponge_power"],
        sponge_target=settings["sponge_target"],
        thermo_enabled=settings["thermo_enabled"],
        moisture_enabled=settings["moisture_enabled"],
        theta0=settings["theta0"],
        g=settings["g"],
        buoyancy_reference=settings["buoyancy_reference"],
        theta_bc=settings["theta_bc"],
        theta_profile=settings["theta_profile"],
        theta_top_gradient=settings["theta_top_gradient"],
        theta_bottom=settings["theta_bottom"],
        theta_top=settings["theta_top"],
        theta_initial_gradient=settings["theta_initial_gradient"],
        theta_perturbation_amplitude=settings["theta_perturbation_amplitude"],
        theta_perturbation_height=settings["theta_perturbation_height"],
        cbl_mixed_layer_height=settings["cbl_mixed_layer_height"],
        cbl_inversion_strength=settings["cbl_inversion_strength"],
        cbl_inversion_thickness=settings["cbl_inversion_thickness"],
        cbl_free_atmosphere_gradient=settings["cbl_free_atmosphere_gradient"],
        surface_theta_flux=settings["surface_theta_flux"],
        qv0=settings["qv0"],
        qv_initial_gradient=settings["qv_initial_gradient"],
        surface_qv_flux=settings["surface_qv_flux"],
        qv_floor=settings["qv_floor"],
        surface_pressure=settings["surface_pressure"],
        initial_perturbation_height=settings["initial_perturbation_height"],
        scalar_sgs_model=settings["scalar_sgs_model"],
        prandtl_t=settings["prandtl_t"],
        schmidt_t=settings["schmidt_t"],
        scalar_stability_correction=settings["scalar_stability_correction"],
        scalar_stability_beta=settings["scalar_stability_beta"],
        scalar_stability_power=settings["scalar_stability_power"],
        scalar_lasd_min=settings["scalar_lasd_min"],
        scalar_lasd_max=settings["scalar_lasd_max"],
        scalar_vertical_scheme=settings["scalar_vertical_scheme"],
        actuator_disk_enabled=settings["actuator_disk_enabled"],
        actuator_disk_x=settings["actuator_disk_x"],
        actuator_disk_y=settings["actuator_disk_y"],
        actuator_disk_z=settings["actuator_disk_z"],
        actuator_disk_diameter=settings["actuator_disk_diameter"],
        actuator_disk_hub_diameter=settings["actuator_disk_hub_diameter"],
        actuator_disk_ct_prime=settings["actuator_disk_ct_prime"],
        actuator_disk_thickness=settings["actuator_disk_thickness"],
        actuator_disk_yaw_degrees=settings["actuator_disk_yaw_degrees"],
        cold_source_enabled=settings["cold_source_enabled"],
        cold_source_x=settings["cold_source_x"],
        cold_source_y=settings["cold_source_y"],
        cold_source_z=settings["cold_source_z"],
        cold_source_sigma_x=settings["cold_source_sigma_x"],
        cold_source_sigma_r=settings["cold_source_sigma_r"],
        cold_source_momentum_flux=settings["cold_source_momentum_flux"],
        cold_source_cooling_power=settings["cold_source_cooling_power"],
        cold_source_density=settings["cold_source_density"],
        cold_source_heat_capacity=settings["cold_source_heat_capacity"],
        fringe_enabled=settings["fringe_enabled"],
        fringe_start_x=settings["fringe_start_x"],
        fringe_timescale=settings["fringe_timescale"],
        fringe_target_u=settings["fringe_target_u"],
        fringe_target_v=settings["fringe_target_v"],
        fringe_target_theta=settings["fringe_target_theta"],
        dtype=solver_dtype,
        sgs_dtype=sgs_dtype_for_precision(settings["sgs_precision"], solver_dtype, jnp),
        use_jit=settings["use_jit"],
    )


def main() -> None:
    args = parse_args()
    try:
        settings = merged_settings(args)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"ERROR: failed to load config: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if settings["precision"] == "float64" or settings["sgs_precision"] == "float64":
        from jax import config as jax_config

        jax_config.update("jax_enable_x64", True)

    import jax.numpy as jnp

    from wireles_jax import run
    from wireles_jax.io import save_npz, save_velocity_h5

    try:
        params = params_from_settings(settings, jnp)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if settings["profile"]:
        from wireles_jax.profiling import profile_ab2

        try:
            state, rows = profile_ab2(
                params,
                warmup_steps=settings["profile_warmup"],
                profile_steps=settings["profile_steps"],
                status_callback=lambda message: print(message, flush=True),
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(2) from None
        simulated_steps = settings["profile_steps"] if settings["profile_steps"] is not None else params.nsteps
        print_profile_report(rows, simulated_steps, settings["profile_warmup"])
        if settings["checkpoint"]:
            save_npz(settings["checkpoint"], state)
        return

    header_printed = False

    def print_diagnostic(diag) -> None:
        nonlocal header_printed
        if not header_printed:
            print(LOG_HEADER, flush=True)
            header_printed = True
        print(format_diagnostic(diag, params.cs_count), flush=True)

    field_output_dir = settings["field_output_dir"] or Path("jax_fields")

    def dump_velocity_fields(state, diag) -> None:
        if not settings["dump_fields"]:
            return
        if int(diag.step) < settings["field_dump_start_step"]:
            return
        path = field_output_dir / f"{settings['field_output_prefix']}_step_{int(diag.step):06d}.h5"
        save_velocity_h5(path, state, params, diag)

    try:
        state, _ = run(
            params,
            log_every=settings["log_every"],
            log_callback=print_diagnostic,
            log_state_callback=dump_velocity_fields,
            status_callback=lambda message: print(message, flush=True),
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    if settings["checkpoint"]:
        save_npz(settings["checkpoint"], state)


if __name__ == "__main__":
    main()
