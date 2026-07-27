#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


JAX_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(JAX_ROOT))


LOG_HEADER = " step    rest_s   total_s elapsed_s      ustar        ke_max       div_max       cfl_x       cfl_y       cfl_z"


RUN_DEFAULTS = {
    "nx": 32,
    "ny": 32,
    "nz": 33,
    "lx": 2.0 * 3.141592653589793,
    "ly": 2.0 * 3.141592653589793,
    "lz": 1.0,
    "dt": 1.0e-4,
    "steps": 10,
    "log_every": 1,
    "time_scheme": "rk3",
    "u_fric": 0.4,
    "zo": 1.0e-4,
    "vonk": 0.4,
    "pressure_force": "balanced",
    "nu": 0.0,
    "initial_perturbation_amp": 0.25,
    "spinup_forcing_accel": 0.0,
    "spinup_forcing_steps": 0,
    "sgs_model": "wall_smagorinsky",
    "fgr": 1.5,
    "tfr": 2.0,
    "cs_count": 10,
    "lasd_cheb_filter": False,
    "lasd_cheb_filter_alpha": 36.0,
    "lasd_cheb_filter_order": 8,
    "smagorinsky_cs": 0.16,
    "dynamic_smagorinsky_cs_max": 0.23,
    "dynamic_smagorinsky_cs_floor": 0.08,
    "dynamic_smagorinsky_relaxation": 0.05,
    "dynamic_smagorinsky_smooth_passes": 2,
    "wall_model": "loglaw",
    "wall_ref_height": None,
    "wall_filter": True,
    "wall_stress_treatment": "weak_flux",
    "convective_form": "skew",
    "horizontal_dealias": True,
    "vertical_dealias": True,
    "solution_filter": True,
    "vertical_dealias_cutoff_ratio": 2.0 / 3.0,
    "vertical_dealias_filter_alpha": 18.0,
    "vertical_dealias_filter_order": 8,
    "precision": "float64",
    "sgs_precision": "float32",
    "use_jit": True,
    "seed": 0,
    "dump_fields": False,
    "field_clean_output": False,
    "field_dump_start_step": 0,
    "field_output_dir": None,
    "field_output_prefix": "fields",
}


CONFIG_KEYS = {
    "grid": {
        "nx": "nx",
        "ny": "ny",
        "nz": "nz",
        "lx": "lx",
        "ly": "ly",
        "lz": "lz",
    },
    "time": {
        "steps": "steps",
        "nsteps": "steps",
        "dt": "dt",
        "log_every": "log_every",
        "scheme": "time_scheme",
        "time_scheme": "time_scheme",
    },
    "physics": {
        "u_fric": "u_fric",
        "zo": "zo",
        "vonk": "vonk",
        "pressure_force": "pressure_force",
        "nu": "nu",
        "initial_perturbation_amp": "initial_perturbation_amp",
        "spinup_forcing_accel": "spinup_forcing_accel",
        "spinup_forcing_steps": "spinup_forcing_steps",
    },
    "sgs": {
        "model": "sgs_model",
        "sgs_model": "sgs_model",
        "fgr": "fgr",
        "tfr": "tfr",
        "cs_count": "cs_count",
        "smag_cs": "smagorinsky_cs",
        "smagorinsky_cs": "smagorinsky_cs",
        "dynamic_cs_max": "dynamic_smagorinsky_cs_max",
        "dynamic_smagorinsky_cs_max": "dynamic_smagorinsky_cs_max",
        "dynamic_cs_floor": "dynamic_smagorinsky_cs_floor",
        "dynamic_smagorinsky_cs_floor": "dynamic_smagorinsky_cs_floor",
        "dynamic_relaxation": "dynamic_smagorinsky_relaxation",
        "dynamic_smagorinsky_relaxation": "dynamic_smagorinsky_relaxation",
        "dynamic_smooth_passes": "dynamic_smagorinsky_smooth_passes",
        "dynamic_smagorinsky_smooth_passes": "dynamic_smagorinsky_smooth_passes",
        "precision": "sgs_precision",
        "sgs_precision": "sgs_precision",
        "lasd_cheb_filter": "lasd_cheb_filter",
        "lasd_cheb_filter_alpha": "lasd_cheb_filter_alpha",
        "lasd_cheb_filter_order": "lasd_cheb_filter_order",
    },
    "wall": {
        "model": "wall_model",
        "wall_model": "wall_model",
        "ref_height": "wall_ref_height",
        "wall_ref_height": "wall_ref_height",
        "filter": "wall_filter",
        "wall_filter": "wall_filter",
        "stress_treatment": "wall_stress_treatment",
        "wall_stress_treatment": "wall_stress_treatment",
    },
    "numerics": {
        "convective_form": "convective_form",
        "horizontal_dealias": "horizontal_dealias",
        "vertical_dealias": "vertical_dealias",
        "solution_filter": "solution_filter",
        "vertical_dealias_cutoff_ratio": "vertical_dealias_cutoff_ratio",
        "vertical_dealias_filter_alpha": "vertical_dealias_filter_alpha",
        "vertical_dealias_filter_order": "vertical_dealias_filter_order",
    },
    "runtime": {
        "precision": "precision",
        "sgs_precision": "sgs_precision",
        "use_jit": "use_jit",
        "seed": "seed",
    },
    "output": {
        "dump_fields": "dump_fields",
        "field_clean_output": "field_clean_output",
        "field_dump_start_step": "field_dump_start_step",
        "field_output_dir": "field_output_dir",
        "field_output_prefix": "field_output_prefix",
    },
    "postprocess": {
        "dump_fields": "dump_fields",
        "field_clean_output": "field_clean_output",
        "field_dump_start_step": "field_dump_start_step",
        "field_output_dir": "field_output_dir",
        "field_output_prefix": "field_output_prefix",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-GPU Fourier/compact-FD wireles2 prototype.")
    parser.add_argument("--config", type=Path, help="TOML run configuration.")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nz", type=int, help="Number of uniform vertical nodes, including both walls.")
    parser.add_argument("--lx", type=float)
    parser.add_argument("--ly", type=float)
    parser.add_argument("--lz", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--time-scheme", choices=("ab2", "rk3"))
    parser.add_argument("--u-fric", type=float)
    parser.add_argument("--zo", type=float)
    parser.add_argument("--vonk", type=float)
    parser.add_argument("--pressure-force")
    parser.add_argument("--nu", type=float)
    parser.add_argument("--initial-perturbation-amp", type=float)
    parser.add_argument("--spinup-forcing-accel", type=float)
    parser.add_argument("--spinup-forcing-steps", type=int)
    parser.add_argument("--sgs-model", choices=("smagorinsky", "wall_smagorinsky", "dynamic_smagorinsky", "lasd"))
    parser.add_argument("--fgr", type=float)
    parser.add_argument("--tfr", type=float)
    parser.add_argument("--cs-count", type=int)
    parser.add_argument("--lasd-cheb-filter", action="store_true", dest="lasd_cheb_filter", default=None)
    parser.add_argument("--no-lasd-cheb-filter", action="store_false", dest="lasd_cheb_filter")
    parser.add_argument("--lasd-cheb-filter-alpha", type=float)
    parser.add_argument("--lasd-cheb-filter-order", type=int)
    parser.add_argument("--smag-cs", type=float, dest="smagorinsky_cs")
    parser.add_argument("--dynamic-cs-max", type=float, dest="dynamic_smagorinsky_cs_max")
    parser.add_argument("--dynamic-cs-floor", type=float, dest="dynamic_smagorinsky_cs_floor")
    parser.add_argument("--dynamic-relaxation", type=float, dest="dynamic_smagorinsky_relaxation")
    parser.add_argument("--dynamic-smooth-passes", type=int, dest="dynamic_smagorinsky_smooth_passes")
    parser.add_argument("--wall-model", choices=("none", "loglaw", "fixed_ustar", "balanced_loglaw"))
    parser.add_argument("--wall-ref-height", type=float)
    parser.add_argument("--wall-filter", action="store_true", dest="wall_filter", default=None)
    parser.add_argument("--no-wall-filter", action="store_false", dest="wall_filter")
    parser.add_argument("--wall-stress-treatment", choices=("weak_flux", "lifting", "strong_flux"))
    parser.add_argument("--convective-form", choices=("skew", "advective"))
    parser.add_argument("--horizontal-dealias", action="store_true", dest="horizontal_dealias", default=None)
    parser.add_argument("--no-horizontal-dealias", action="store_false", dest="horizontal_dealias")
    parser.add_argument("--vertical-dealias", action="store_true", dest="vertical_dealias", default=None)
    parser.add_argument("--no-vertical-dealias", action="store_false", dest="vertical_dealias")
    parser.add_argument("--solution-filter", action="store_true", dest="solution_filter", default=None)
    parser.add_argument("--no-solution-filter", action="store_false", dest="solution_filter")
    parser.add_argument("--vertical-dealias-cutoff-ratio", type=float)
    parser.add_argument("--vertical-dealias-filter-alpha", type=float)
    parser.add_argument("--vertical-dealias-filter-order", type=int)
    parser.add_argument("--precision", choices=("float64", "float32"))
    parser.add_argument("--sgs-precision", choices=("float64", "float32", "default"))
    parser.add_argument("--single", action="store_const", const="float32", dest="precision")
    parser.add_argument("--no-jit", action="store_false", dest="use_jit", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dump-fields", action="store_true", dest="dump_fields", default=None)
    parser.add_argument("--no-dump-fields", action="store_false", dest="dump_fields")
    parser.add_argument("--field-clean-output", action="store_true", dest="field_clean_output", default=None)
    parser.add_argument("--no-field-clean-output", action="store_false", dest="field_clean_output")
    parser.add_argument("--field-dump-start-step", type=int)
    parser.add_argument("--field-output-dir", type=Path)
    parser.add_argument("--field-output-prefix")
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
    if settings["field_output_dir"] is not None:
        settings["field_output_dir"] = Path(settings["field_output_dir"])
    if settings["field_dump_start_step"] < 0:
        raise ValueError("field_dump_start_step must be non-negative.")

    return settings


def dtype_for_precision(precision: str, jnp):
    if precision == "float64":
        return jnp.float64
    if precision == "float32":
        return jnp.float32
    raise ValueError(f"Unsupported precision {precision}")


def format_diagnostic(diag) -> str:
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
        f"{float(diag.cfl_z):10.4f}"
    )


def main() -> None:
    args = parse_args()
    try:
        settings = merged_settings(args)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if settings["precision"] == "float64":
        from jax import config as jax_config

        jax_config.update("jax_enable_x64", True)

    import jax.numpy as jnp

    from wireles2 import Params, run
    from wireles2.io import save_velocity_h5

    try:
        solver_dtype = dtype_for_precision(settings["precision"], jnp)
        sgs_precision = settings["sgs_precision"]
        sgs_dtype = solver_dtype if sgs_precision == "default" else dtype_for_precision(sgs_precision, jnp)
        params = Params(
            nx=settings["nx"],
            ny=settings["ny"],
            nz=settings["nz"],
            lx=settings["lx"],
            ly=settings["ly"],
            lz=settings["lz"],
            dt=settings["dt"],
            nsteps=settings["steps"],
            log_every=settings["log_every"],
            time_scheme=settings["time_scheme"],
            u_fric=settings["u_fric"],
            zo=settings["zo"],
            vonk=settings["vonk"],
            pressure_force=settings["pressure_force"],
            nu=settings["nu"],
            initial_perturbation_amp=settings["initial_perturbation_amp"],
            spinup_forcing_accel=settings["spinup_forcing_accel"],
            spinup_forcing_steps=settings["spinup_forcing_steps"],
            sgs_model=settings["sgs_model"],
            fgr=settings["fgr"],
            tfr=settings["tfr"],
            cs_count=settings["cs_count"],
            lasd_cheb_filter=settings["lasd_cheb_filter"],
            lasd_cheb_filter_alpha=settings["lasd_cheb_filter_alpha"],
            lasd_cheb_filter_order=settings["lasd_cheb_filter_order"],
            smagorinsky_cs=settings["smagorinsky_cs"],
            dynamic_smagorinsky_cs_max=settings["dynamic_smagorinsky_cs_max"],
            dynamic_smagorinsky_cs_floor=settings["dynamic_smagorinsky_cs_floor"],
            dynamic_smagorinsky_relaxation=settings["dynamic_smagorinsky_relaxation"],
            dynamic_smagorinsky_smooth_passes=settings["dynamic_smagorinsky_smooth_passes"],
            wall_model=settings["wall_model"],
            wall_ref_height=settings["wall_ref_height"],
            wall_filter=settings["wall_filter"],
            wall_stress_treatment=settings["wall_stress_treatment"],
            convective_form=settings["convective_form"],
            horizontal_dealias=settings["horizontal_dealias"],
            vertical_dealias=settings["vertical_dealias"],
            solution_filter=settings["solution_filter"],
            vertical_dealias_cutoff_ratio=settings["vertical_dealias_cutoff_ratio"],
            vertical_dealias_filter_alpha=settings["vertical_dealias_filter_alpha"],
            vertical_dealias_filter_order=settings["vertical_dealias_filter_order"],
            dtype=solver_dtype,
            sgs_dtype=sgs_dtype,
            use_jit=settings["use_jit"],
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    header_printed = False

    def log(diag) -> None:
        nonlocal header_printed
        if not header_printed:
            print(LOG_HEADER, flush=True)
            header_printed = True
        print(format_diagnostic(diag), flush=True)

    field_output_dir = settings["field_output_dir"] or Path("wireles2_fields")
    if settings["dump_fields"] and settings["field_clean_output"]:
        removed = 0
        for path in field_output_dir.glob(f"{settings['field_output_prefix']}_step_*.h5"):
            path.unlink()
            removed += 1
        if removed:
            print(f"[output] removed {removed} existing field dump(s) from {field_output_dir}", flush=True)

    def dump_velocity_fields(state, ops, diag) -> None:
        if not settings["dump_fields"]:
            return
        if int(diag.step) < settings["field_dump_start_step"]:
            return
        path = field_output_dir / f"{settings['field_output_prefix']}_step_{int(diag.step):06d}.h5"
        save_velocity_h5(path, state, params, ops, diag)

    run(
        params,
        log_callback=log,
        log_state_callback=dump_velocity_fields,
        status_callback=lambda message: print(message, flush=True),
        seed=settings["seed"],
    )


if __name__ == "__main__":
    main()
