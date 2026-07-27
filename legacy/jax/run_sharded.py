#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

JAX_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(JAX_ROOT))

from run_single import (  # noqa: E402
    LOG_HEADER,
    dtype_for_precision,
    format_diagnostic,
    merged_settings,
    scaled_grid_lengths,
    scaled_time_step,
    sgs_dtype_for_precision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the distributed z-sharded JAX WiRE-LES solver.")
    parser.add_argument("--config", type=Path, help="TOML run configuration.")
    parser.add_argument("--devices", type=int, help="Number of global JAX devices to use.")
    parser.add_argument("--coordinator-address", help="JAX coordinator as host:port.")
    parser.add_argument("--num-processes", type=int, help="Number of distributed JAX processes.")
    parser.add_argument("--process-id", type=int, help="Zero-based distributed JAX process id.")
    parser.add_argument(
        "--local-device-ids",
        help="Comma-separated local device ids assigned to this process.",
    )
    parser.add_argument("--seed", type=int, default=0)
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
        ),
    )
    parser.add_argument("--momentum-wall-model", choices=("abl", "free_slip", "free-slip", "no_slip", "no-slip"))
    parser.add_argument("--wall-stress-model", choices=("dynamic_neutral", "dynamic-neutral", "prescribed_ustar", "prescribed-ustar"))
    parser.add_argument("--initial-velocity-noise", type=float)
    parser.add_argument("--fgr", type=float)
    parser.add_argument("--tfr", type=float)
    parser.add_argument("--sgs-model", choices=("smagorinsky", "lasd"))
    parser.add_argument("--cs-count", type=int)
    parser.add_argument("--smag-cs", type=float)
    parser.add_argument("--time-scheme", choices=("rk4", "ab2"))
    parser.add_argument("--projection-mode", choices=("stage", "final"))
    parser.add_argument("--horizontal-dealias", action="store_true", dest="horizontal_dealias", default=None)
    parser.add_argument("--no-horizontal-dealias", action="store_false", dest="horizontal_dealias")
    parser.add_argument("--pressure-filter-nyquist", action="store_true", dest="pressure_filter_nyquist", default=None)
    parser.add_argument("--no-pressure-filter-nyquist", action="store_false", dest="pressure_filter_nyquist")
    parser.add_argument(
        "--sharded-pressure-solver",
        choices=("transpose", "spike"),
    )
    parser.add_argument("--precision", choices=("float64", "float32"))
    parser.add_argument("--sgs-precision", choices=("default", "float64", "float32"))
    parser.add_argument("--single", action="store_const", const="float32", dest="precision")
    parser.add_argument("--no-jit", action="store_false", dest="use_jit", default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--thermo", action="store_true", dest="thermo_enabled", default=None)
    parser.add_argument("--no-thermo", action="store_false", dest="thermo_enabled")
    parser.add_argument("--theta0", type=float)
    parser.add_argument("--g", type=float)
    parser.add_argument("--theta-bc", choices=("flux",))
    parser.add_argument("--theta-profile", choices=("linear", "deardorff_cbl"))
    parser.add_argument("--theta-top-gradient", type=float)
    parser.add_argument("--theta-initial-gradient", type=float)
    parser.add_argument("--theta-perturbation-amplitude", type=float)
    parser.add_argument("--theta-perturbation-height", type=float)
    parser.add_argument("--surface-theta-flux", type=float)
    parser.add_argument("--scalar-sgs-model", choices=("fixed_prandtl", "fixed-prandtl"))
    parser.add_argument("--prandtl-t", type=float)
    parser.add_argument("--scalar-vertical-scheme", choices=("centered",))
    return parser.parse_args()


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

    import jax

    distributed_args = (
        args.coordinator_address,
        args.num_processes,
        args.process_id,
    )
    if any(value is not None for value in distributed_args):
        if not all(value is not None for value in distributed_args):
            raise SystemExit(
                "ERROR: --coordinator-address, --num-processes, and --process-id must be supplied together."
            )
        local_device_ids = None
        if args.local_device_ids:
            local_device_ids = [int(value) for value in args.local_device_ids.split(",")]
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
            local_device_ids=local_device_ids,
        )

    import jax.numpy as jnp

    from wireles_jax import Params
    from wireles_jax.timestep_sharded import run_sharded

    try:
        lx_scaled, ly_scaled, lz_scaled, z_i = scaled_grid_lengths(settings)
        dt_scaled = scaled_time_step(settings, z_i)
        solver_dtype = dtype_for_precision(settings["precision"], jnp)
        params = Params(
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
            smagorinsky_cs=settings["smag_cs"],
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
            scalar_sgs_model=settings["scalar_sgs_model"],
            prandtl_t=settings["prandtl_t"],
            schmidt_t=settings["schmidt_t"],
            scalar_stability_correction=settings["scalar_stability_correction"],
            scalar_stability_beta=settings["scalar_stability_beta"],
            scalar_stability_power=settings["scalar_stability_power"],
            scalar_lasd_min=settings["scalar_lasd_min"],
            scalar_lasd_max=settings["scalar_lasd_max"],
            scalar_vertical_scheme=settings["scalar_vertical_scheme"],
            dtype=solver_dtype,
            sgs_dtype=sgs_dtype_for_precision(settings["sgs_precision"], solver_dtype, jnp),
            use_jit=settings["use_jit"],
        )
        if settings["checkpoint"] is not None:
            raise ValueError("Checkpoint output is not implemented for sharded interior-only state yet.")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    is_primary = jax.process_index() == 0
    header_printed = False

    def print_diagnostic(diag) -> None:
        nonlocal header_printed
        if not is_primary:
            return
        if not header_printed:
            print(LOG_HEADER, flush=True)
            header_printed = True
        print(format_diagnostic(diag, params.cs_count), flush=True)

    try:
        run_sharded(
            params,
            num_devices=args.devices,
            log_every=settings["log_every"],
            log_callback=print_diagnostic,
            status_callback=(
                (lambda message: print(message, flush=True)) if is_primary else None
            ),
            seed=args.seed,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
