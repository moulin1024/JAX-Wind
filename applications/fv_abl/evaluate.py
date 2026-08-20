"""Compose and execute the generic finite-volume ABL core."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from .config import FiniteVolumeCase
from .diagnostics import (
    ProfileAccumulator,
    RadialAccumulator,
    bulk_metrics,
    initial_fields,
    reference_comparison,
    steps_to_next_sample,
    write_profiles,
    write_radial_spectra,
    write_streamwise_spectra,
)


def _physical_rotation(case) -> tuple[float, float, float, float]:
    rotation = case.model.momentum.rotation
    if not hasattr(rotation, "coriolis_parameter"):
        return 0.0, 0.0, 0.0, 0.0
    scales = case.mechanical_scales
    return (
        scales.from_execution_inverse_time(rotation.coriolis_parameter),
        scales.from_execution_inverse_time(
            rotation.horizontal_coriolis_parameter
        ),
        scales.from_execution_velocity(rotation.geostrophic_x_velocity),
        scales.from_execution_velocity(rotation.geostrophic_y_velocity),
    )


def resolved(configured: FiniteVolumeCase) -> dict:
    """Return the fully lowered, JSON-serializable FV composition."""

    case = configured.physical
    options = configured.options
    grid = case.physical_grid
    scales = case.mechanical_scales
    scalar_scales = case.scalar_scales
    momentum = case.model.momentum
    vertical_f, horizontal_f, geostrophic_u, geostrophic_v = (
        _physical_rotation(case)
    )
    pressure = momentum.pressure_gradient
    surface = case.model.surface_transfer
    if not hasattr(surface, "scalar_roughness_length"):
        surface = None
    result = {
        "case": case.name,
        "citation": case.citation,
        "discretization": "staggered finite volume",
        "time_integration": options.time_integration.upper(),
        "pressure_backend": options.pressure_backend,
        "momentum_closure": "AnisotropicMinimumDissipation",
        "scalar_closure": "eddy diffusivity",
        "turbulent_prandtl": options.turbulent_prandtl,
        "cells": [grid.nx, grid.ny, grid.nz],
        "lengths_m": [grid.lx, grid.ly, grid.lz],
        "dt_seconds": case.dt_seconds,
        "dt_interpretation": (
            "maximum" if options.cfl_ceiling is not None else "fixed"
        ),
        "cfl_ceiling": options.cfl_ceiling,
        "steps": case.steps,
        "dtype": case.pressure.dtype,
        "chunk_steps": options.chunk_steps,
        "spectrum_diagnostic": options.spectrum_diagnostic,
        "output_directory": str(options.output_directory),
        "gmg_tolerance": options.gmg_tolerance,
        "gmg_presweeps": options.gmg_presweeps,
        "gmg_postsweeps": options.gmg_postsweeps,
        "roughness_length_m": scales.from_execution_length(
            momentum.wall.roughness_length
        ),
        "coriolis_vertical_s": vertical_f,
        "coriolis_horizontal_s": horizontal_f,
        "evolved_geostrophic_velocity_m_s": [
            geostrophic_u,
            geostrophic_v,
        ],
        "geostrophic_velocity_m_s": [
            geostrophic_u + case.advection_frame_velocity_m_s[0],
            geostrophic_v + case.advection_frame_velocity_m_s[1],
        ],
        "velocity_offset_m_s": list(case.advection_frame_velocity_m_s),
        "pressure_acceleration_m_s2": [
            scales.from_execution_acceleration(pressure.x_acceleration),
            scales.from_execution_acceleration(pressure.y_acceleration),
        ],
        "scalar_reference": scalar_scales.reference_value,
        "scalar_surface_flux": scalar_scales.from_execution_flux(
            case.model.scalar_boundary.lower_flux
        ),
        "buoyancy_acceleration_per_scalar": (
            scalar_scales.from_execution_buoyancy_coefficient(
                case.model.buoyancy.acceleration_per_temperature
            )
        ),
        "sample_start_step": case.output.sample_start_step,
        "sample_every_steps": case.output.sample_every_steps,
        "spectrum_heights_m": list(
            case.diagnostic_reference.spectrum_heights_m
        ),
    }
    if surface is not None:
        result.update(
            {
                "momentum_roughness_m": result["roughness_length_m"],
                "scalar_roughness_m": scales.from_execution_length(
                    surface.scalar_roughness_length
                ),
                "surface_scalar_initial": (
                    scalar_scales.from_execution_scalar(
                        surface.surface_scalar_initial
                    )
                ),
                "surface_scalar_rate_per_second": (
                    surface.surface_scalar_rate
                    * scalar_scales.magnitude
                    / scales.time
                ),
            }
        )
    return result


class SurfaceStatistics:
    def __init__(self) -> None:
        self.count = 0
        self.scalar_flux_sum = 0.0
        self.obukhov_sum = 0.0
        self.surface_scalar_sum = 0.0

    def sample(self, exchange) -> None:
        self.count += 1
        self.scalar_flux_sum += float(exchange.scalar_flux)
        self.obukhov_sum += float(exchange.obukhov_length)
        self.surface_scalar_sum += float(exchange.surface_scalar)

    def means(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {
            "surface_scalar_flux": self.scalar_flux_sum / self.count,
            "surface_scalar": self.surface_scalar_sum / self.count,
            "obukhov_length_m": self.obukhov_sum / self.count,
        }


def _sample(
    solution,
    accumulator,
    diagnostic,
    jax,
    grid,
    spectrum_level: int,
    surface_statistics: SurfaceStatistics,
) -> None:
    sampled = diagnostic(
        solution.velocity,
        solution.pressure,
        solution.scalar,
        solution.time,
    )
    sampled = jax.device_get(sampled)
    fields, profiles, ustar, exchange = sampled
    if isinstance(accumulator, RadialAccumulator):
        accumulator.sample(fields, profiles, ustar=float(ustar), grid=grid)
    else:
        accumulator.sample(
            fields,
            profiles,
            ustar=float(ustar),
            spectrum_level=spectrum_level,
        )
    if exchange is not None:
        surface_statistics.sample(exchange)


def evaluate(
    configured: FiniteVolumeCase,
    *,
    output_dir: Path,
    max_steps: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Execute one configured FV ABL case and write common diagnostics."""

    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    case = configured.physical
    options = configured.options
    configuration = resolved(configured)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_case.json").write_text(
        json.dumps(configuration, indent=2) + "\n",
        encoding="utf-8",
    )

    import jax

    jax.config.update("jax_enable_x64", case.pressure.dtype == "float64")
    import jax.numpy as jnp
    from jaxwind.fv import (
        AnisotropicMinimumDissipation,
        CELL_CENTRE,
        LOCAL,
        CoriolisGeostrophic,
        FlowModel,
        LinearBoussinesqBuoyancy,
        MoninObukhovSurface,
        MoninObukhovWall,
        PassiveScalar,
        StaggeredVelocity,
        atmospheric_history_diagnostics,
        atmospheric_profile_diagnostics,
        build_atmospheric_run,
        build_atmospheric_step,
        build_pressure_poisson,
        coupled_surface_exchange,
        courant_number,
        divergence,
        friction_velocity,
        initial_atmospheric_solution,
        monin_obukhov_boundaries,
        project,
    )

    grid = case.physical_grid
    u, v, w, scalar_field = initial_fields(case, jax, jnp)
    offset_u, offset_v = case.advection_frame_velocity_m_s
    velocity = StaggeredVelocity(u - offset_u, v - offset_v, w)
    gmg_config = (
        {
            "presweeps": options.gmg_presweeps,
            "postsweeps": options.gmg_postsweeps,
            **(
                {}
                if options.gmg_tolerance is None
                else {"tolerance": options.gmg_tolerance}
            ),
        }
        if options.pressure_backend == "gmg"
        else None
    )
    poisson = build_pressure_poisson(
        grid,
        backend=options.pressure_backend,
        dtype=case.pressure.dtype,
        config=gmg_config,
    )
    velocity, _ = project(velocity, poisson, 1.0)
    boundaries = monin_obukhov_boundaries()
    subfilter = AnisotropicMinimumDissipation()

    vertical_f = configuration["coriolis_vertical_s"]
    rotation = None
    if vertical_f != 0.0:
        evolved = configuration["evolved_geostrophic_velocity_m_s"]
        rotation = CoriolisGeostrophic(
            vertical_f,
            evolved[0],
            evolved[1],
            configuration["coriolis_horizontal_s"],
        )
    pressure_force = configuration["pressure_acceleration_m_s2"]
    wall = None
    coupled_surface = case.model.surface_transfer
    if not hasattr(coupled_surface, "scalar_roughness_length"):
        coupled_surface = None
    if coupled_surface is None:
        wall = MoninObukhovWall(
            configuration["roughness_length_m"],
            von_karman=case.model.momentum.wall.von_karman,
            sampling=CELL_CENTRE,
            averaging=LOCAL,
        )
    momentum = FlowModel(
        body_force=(pressure_force[0], pressure_force[1], 0.0),
        subfilter=subfilter,
        surface=wall,
        rotation=rotation,
    )
    scalar = PassiveScalar(
        lower_flux=configuration["scalar_surface_flux"],
        turbulent_prandtl=options.turbulent_prandtl,
    )
    coefficient = configuration["buoyancy_acceleration_per_scalar"]
    buoyancy = (
        LinearBoussinesqBuoyancy(coefficient)
        if coefficient != 0.0
        else None
    )
    surface = None
    if coupled_surface is not None:
        surface = MoninObukhovSurface(
            momentum_roughness=configuration["momentum_roughness_m"],
            scalar_roughness=configuration["scalar_roughness_m"],
            surface_scalar_initial=configuration["surface_scalar_initial"],
            surface_scalar_rate=configuration[
                "surface_scalar_rate_per_second"
            ],
            x_velocity_offset=offset_u,
            y_velocity_offset=offset_v,
            buoyancy_coefficient=coefficient,
            von_karman=case.model.momentum.wall.von_karman,
            positive_zeta_momentum_slope=(
                coupled_surface.positive_zeta_momentum_slope
            ),
            positive_zeta_scalar_slope=(
                coupled_surface.positive_zeta_scalar_slope
            ),
            negative_zeta_momentum_coefficient=(
                coupled_surface.negative_zeta_momentum_coefficient
            ),
            negative_zeta_scalar_coefficient=(
                coupled_surface.negative_zeta_scalar_coefficient
            ),
            iterations=coupled_surface.iterations,
            relaxation=coupled_surface.relaxation,
            maximum_abs_zeta=coupled_surface.maximum_abs_zeta,
        )
    step = build_atmospheric_step(
        grid,
        boundaries,
        poisson,
        momentum,
        scalar,
        buoyancy,
        surface,
        scheme=options.time_integration,
    )
    advance = build_atmospheric_run(step)
    solution = initial_atmospheric_solution(
        grid,
        velocity,
        scalar_field,
        dtype=case.pressure.dtype,
    )

    if surface is None:
        def diagnostic(velocity, pressure, scalar_field, execution_time):
            del execution_time
            fields, profiles = atmospheric_profile_diagnostics(
                velocity,
                pressure,
                scalar_field,
                grid,
                boundaries,
                wall,
                scalar,
                subfilter,
            )
            ustar = friction_velocity(velocity, grid, wall)
            return fields, profiles, ustar, None

        history_diagnostic = jax.jit(
            lambda current: atmospheric_history_diagnostics(
                current,
                grid,
                wall,
                coriolis=vertical_f,
                geostrophic_u=configuration[
                    "geostrophic_velocity_m_s"
                ][0],
                geostrophic_v=configuration[
                    "geostrophic_velocity_m_s"
                ][1],
            )
        )
        exchange_diagnostic = None
    else:
        def diagnostic(velocity, pressure, scalar_field, execution_time):
            exchange = coupled_surface_exchange(
                velocity,
                scalar_field,
                execution_time,
                grid,
                surface,
            )
            fields, profiles = atmospheric_profile_diagnostics(
                velocity,
                pressure,
                scalar_field,
                grid,
                boundaries,
                None,
                scalar,
                subfilter,
                x_velocity_offset=offset_u,
                y_velocity_offset=offset_v,
                lower_stress_x=exchange.stress_x,
                lower_stress_y=exchange.stress_y,
                lower_scalar_flux=exchange.scalar_flux,
            )
            return fields, profiles, exchange.friction_velocity, exchange

        history_diagnostic = None
        exchange_diagnostic = jax.jit(
            lambda current, scalar_field, execution_time: (
                coupled_surface_exchange(
                    current,
                    scalar_field,
                    execution_time,
                    grid,
                    surface,
                )
            )
        )
    profile_diagnostic = jax.jit(diagnostic)

    z = (np.arange(grid.nz, dtype=np.float64) + 0.5) * grid.dz
    spectrum_level = int(
        np.argmin(
            np.abs(z - case.diagnostic_reference.spectrum_heights_m[0])
        )
    )
    accumulator = (
        RadialAccumulator(case)
        if options.spectrum_diagnostic == "radial"
        else ProfileAccumulator(grid.nz, grid.nx)
    )
    surface_statistics = SurfaceStatistics()
    target_steps = case.steps if max_steps is None else min(case.steps, max_steps)
    history_fields = (
        "step",
        "time_hours",
        "maximum_cfl",
        "ustar_m_s",
        "maximum_divergence_s",
        "mean_scalar",
        "surface_scalar",
        "surface_scalar_flux",
        "obukhov_length_m",
        "maximum_abs_w_m_s",
        "surface_uw_m2_s2",
        "surface_vw_m2_s2",
        "momentum_stationarity_cu",
        "momentum_stationarity_cv",
        "integrated_resolved_tke_m3_s2",
        "integrated_sgs_tke_m3_s2",
        "integrated_total_tke_m3_s2",
        "elapsed_seconds",
        "milliseconds_per_step",
    )
    elapsed_total = 0.0
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as history_stream:
        writer = csv.DictWriter(history_stream, fieldnames=history_fields)
        writer.writeheader()
        while int(solution.step) < target_steps:
            current = int(solution.step)
            block = min(options.chunk_steps, target_steps - current)
            block = min(
                block,
                steps_to_next_sample(
                    current,
                    case.output.sample_start_step,
                    case.output.sample_every_steps,
                ),
            )
            started = time.perf_counter()
            solution = advance(solution, case.dt_seconds, block)
            jax.block_until_ready(solution.velocity.x)
            elapsed = time.perf_counter() - started
            elapsed_total += elapsed
            completed = int(solution.step)
            maximum_divergence = float(
                jnp.max(jnp.abs(divergence(solution.velocity, grid)))
            )
            maximum_cfl = float(
                courant_number(solution.velocity, grid, case.dt_seconds)
            )
            row = {
                "step": completed,
                "time_hours": float(solution.time) / 3600.0,
                "maximum_cfl": maximum_cfl,
                "maximum_divergence_s": maximum_divergence,
                "mean_scalar": float(jnp.mean(solution.scalar)),
                "maximum_abs_w_m_s": float(
                    jnp.max(jnp.abs(solution.velocity.z))
                ),
                "elapsed_seconds": elapsed,
                "milliseconds_per_step": 1000.0 * elapsed / block,
            }
            if exchange_diagnostic is None:
                row.update(
                    {
                        name: float(value)
                        for name, value in jax.device_get(
                            history_diagnostic(solution.velocity)
                        ).items()
                    }
                )
                row["ustar_m_s"] = float(
                    friction_velocity(solution.velocity, grid, wall)
                )
            else:
                exchange = jax.device_get(
                    exchange_diagnostic(
                        solution.velocity,
                        solution.scalar,
                        solution.time,
                    )
                )
                row.update(
                    {
                        "ustar_m_s": float(exchange.friction_velocity),
                        "surface_scalar": float(exchange.surface_scalar),
                        "surface_scalar_flux": float(exchange.scalar_flux),
                        "obukhov_length_m": float(exchange.obukhov_length),
                    }
                )
            writer.writerow(row)
            history_stream.flush()
            if (
                completed >= case.output.sample_start_step
                and (completed - case.output.sample_start_step)
                % case.output.sample_every_steps
                == 0
            ):
                _sample(
                    solution,
                    accumulator,
                    profile_diagnostic,
                    jax,
                    grid,
                    spectrum_level,
                    surface_statistics,
                )
            status = (
                f"step {completed:6d}/{target_steps}  "
                f"t {float(solution.time) / 3600.0:7.3f} h  "
                f"CFL {maximum_cfl:.3f}  div {maximum_divergence:.2e}  "
                f"{1000.0 * elapsed / block:.2f} ms/step"
            )
            print(status, flush=True)

    if accumulator.count == 0:
        _sample(
            solution,
            accumulator,
            profile_diagnostic,
            jax,
            grid,
            spectrum_level,
            surface_statistics,
        )
    write_profiles(output_dir / "profiles.csv", case, accumulator)
    if options.spectrum_diagnostic == "streamwise":
        write_streamwise_spectra(
            output_dir / "spectra.csv", case, accumulator
        )
    elif options.spectrum_diagnostic == "radial":
        write_radial_spectra(
            output_dir / "radial_spectra.csv", case, accumulator
        )

    final_divergence = float(
        jnp.max(jnp.abs(divergence(solution.velocity, grid)))
    )
    diagnostic_metrics: dict[str, float] = {
        "surface_friction_velocity_m_s": accumulator.ustar,
    }
    comparison = None
    if options.spectrum_diagnostic == "radial":
        diagnostic_metrics.update(bulk_metrics(case, accumulator))
        comparison = reference_comparison(case, diagnostic_metrics)
    geostrophic = configuration["geostrophic_velocity_m_s"]
    geostrophic_speed = math.hypot(*geostrophic)
    if geostrophic_speed:
        diagnostic_metrics["surface_friction_velocity_ratio"] = (
            accumulator.ustar / geostrophic_speed
        )
    diagnostic_metrics.update(surface_statistics.means())
    summary = {
        "case": case.name,
        "solver": {
            "discretization": "finite-volume",
            "pressure_backend": options.pressure_backend,
            "time_integration": options.time_integration.upper(),
            "momentum_closure": options.momentum_closure.upper(),
            "scalar_closure": "AMD eddy diffusivity",
            "surface_exchange": (
                "Monin-Obukhov Businger-Dyer"
                if surface is not None
                else "Monin-Obukhov wall stress"
            ),
        },
        "physics": {
            "geostrophic_velocity_m_s": geostrophic,
            "coriolis_vertical_s": vertical_f,
            "coriolis_horizontal_s": configuration[
                "coriolis_horizontal_s"
            ],
            "scalar_surface_flux": configuration["scalar_surface_flux"],
            "buoyancy_acceleration_per_scalar": coefficient,
            **(
                {
                    "surface_scalar_rate_per_second": configuration[
                        "surface_scalar_rate_per_second"
                    ]
                }
                if surface is not None
                else {}
            ),
        },
        "diagnostic_reference": {
            "length_m": case.diagnostic_reference.length_m,
            "velocity_m_s": case.diagnostic_reference.velocity_m_s,
            "scalar": case.diagnostic_reference.scalar,
            "inversion_search_max_height_m": (
                case.diagnostic_reference.inversion_search_max_height_m
            ),
            "spectrum_heights_m": list(
                case.diagnostic_reference.spectrum_heights_m
            ),
        },
        "runtime": {
            "step": int(solution.step),
            "time_seconds": float(solution.time),
            "elapsed_seconds": elapsed_total,
            "maximum_divergence_s": final_divergence,
            "profile_samples": accumulator.count,
        },
        "diagnostic_metrics": diagnostic_metrics,
        **({"comparison": comparison} if comparison is not None else {}),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir}", flush=True)
    return summary


__all__ = ["SurfaceStatistics", "evaluate", "resolved"]
