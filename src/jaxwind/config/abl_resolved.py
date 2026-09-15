"""Resolved atmospheric settings, without simulation construction."""
from __future__ import annotations
import numpy as np
from .abl import FiniteVolumeCase

def _physical_rotation(case) -> tuple[float, float, float, float]:
    vertical, horizontal = case.coriolis_s
    if vertical == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        vertical,
        horizontal,
        case.geostrophic_velocity_m_s[0] - case.advection_frame_velocity_m_s[0],
        case.geostrophic_velocity_m_s[1] - case.advection_frame_velocity_m_s[1],
    )


def resolved(configured: FiniteVolumeCase) -> dict:
    """Return the fully lowered, JSON-serializable FV composition."""

    case = configured.physical
    options = configured.options
    grid = case.physical_grid
    vertical_f, horizontal_f, geostrophic_u, geostrophic_v = (
        _physical_rotation(case)
    )
    surface = case.surface_scalar
    result = {
        "case": case.name,
        "citation": case.citation,
        "discretization": "staggered finite volume",
        "time_integration": options.time_integration.upper(),
        "pressure_backend": options.pressure_backend,
        "fft_method": options.fft_method,
        "fft_thomas_chunk": options.fft_thomas_chunk,
        "fft_spike_block_size": options.fft_spike_block_size,
        "momentum_closure": "AnisotropicMinimumDissipation",
        "scalar_closure": "eddy diffusivity",
        "turbulent_prandtl": options.turbulent_prandtl,
        "scalar_advection_scheme": options.scalar_advection_scheme,
        "momentum_advection_scheme": options.momentum_advection_scheme,
        "cells": [grid.nx, grid.ny, grid.nz],
        "lengths_m": [grid.lx, grid.ly, grid.lz],
        "grid_uniform": grid.is_uniform,
        "minimum_cell_widths_m": [
            float(np.min(grid.x_widths)),
            float(np.min(grid.y_widths)),
            float(np.min(grid.z_widths)),
        ],
        "maximum_cell_widths_m": [
            float(np.max(grid.x_widths)),
            float(np.max(grid.y_widths)),
            float(np.max(grid.z_widths)),
        ],
        "dt_seconds": case.dt_seconds,
        "dt_interpretation": (
            "maximum" if options.cfl_ceiling is not None else "fixed"
        ),
        "cfl_ceiling": options.cfl_ceiling,
        "steps": case.steps,
        "dtype": case.dtype,
        "chunk_steps": options.chunk_steps,
        "spectrum_diagnostic": options.spectrum_diagnostic,
        "output_directory": str(options.output_directory),
        "gmg_tolerance": options.gmg_tolerance,
        "gmg_presweeps": options.gmg_presweeps,
        "gmg_postsweeps": options.gmg_postsweeps,
        "gmg_anisotropy_aware": options.gmg_anisotropy_aware,
        "roughness_length_m": case.roughness_length_m,
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
        "pressure_acceleration_m_s2": list(case.pressure_acceleration_m_s2),
        "scalar_reference": case.scalar_reference_value,
        "scalar_surface_flux": case.scalar_surface_flux,
        "buoyancy_acceleration_per_scalar": case.buoyancy_acceleration_per_scalar,
        "sample_start_step": case.sample_start_step,
        "sample_every_steps": case.sample_every_steps,
        "spectrum_heights_m": list(
            case.diagnostic_reference.spectrum_heights_m
        ),
    }
    if surface is not None:
        result.update(
            {
                "momentum_roughness_m": result["roughness_length_m"],
                "scalar_roughness_m": surface.roughness_length_m,
                "surface_scalar_initial": surface.initial_value,
                "surface_scalar_rate_per_second": surface.rate_per_second,
            }
        )
    return result
