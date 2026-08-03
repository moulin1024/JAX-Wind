"""Paper-facing diagnostics for the non-spectral Andrén AMD runner.

The SGS energy and scalar variance below are local-equilibrium observations of
the eddy-viscosity/diffusivity fields.  They are deliberately kept separate
from the prognostic velocity and passive scalar so the output cannot imply that
JAX-Wind advances an SGS-energy equation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.pressure import mac_pressure_gradient


PROFILE_NAMES = (
    "u",
    "v",
    "w",
    "scalar",
    "resolved_u_variance",
    "resolved_v_variance",
    "resolved_w_variance",
    "resolved_scalar_variance",
    "resolved_tke",
    "resolved_uw",
    "resolved_vw",
    "resolved_wc",
    "sgs_tke",
    "sgs_scalar_variance_numerator",
    "sgs_uw",
    "sgs_vw",
    "sgs_wc",
    "momentum_diffusivity",
    "scalar_diffusivity",
    "wp_modified_pressure",
    "modified_pressure_std",
    "resolved_tke_sgs_dissipation",
    "spectrum_mode",
    "spectrum_u",
    "spectrum_v",
    "spectrum_w",
    "spectrum_scalar",
    "spectrum_height_m",
)

BUDGET_NAMES = (
    "uw_production",
    "uw_subgrid",
    "uw_transport",
    "uw_pressure",
    "uw_coriolis",
    "uw_resolved_flux",
    "wc_production",
    "wc_subgrid",
    "wc_transport",
    "wc_pressure",
    "wc_coriolis",
    "wc_resolved_flux",
)


def _plane(value):
    return jnp.mean(value, axis=(1, 2))


def _faces_to_cells(faces):
    return 0.5 * (faces[:-1] + faces[1:])


def _spectrum(value, level: int):
    signal = value[level] - jnp.mean(value[level], axis=-1, keepdims=True)
    coefficients = jnp.fft.rfft(signal, axis=-1) / signal.shape[-1]
    energy = jnp.mean(jnp.abs(coefficients) ** 2, axis=0)
    if value.shape[-1] % 2 == 0:
        factors = jnp.concatenate(
            (
                jnp.ones((1,), dtype=energy.dtype),
                2.0 * jnp.ones((energy.size - 2,), dtype=energy.dtype),
                jnp.ones((1,), dtype=energy.dtype),
            )
        )
    else:
        factors = jnp.concatenate(
            (
                jnp.ones((1,), dtype=energy.dtype),
                2.0 * jnp.ones((energy.size - 1,), dtype=energy.dtype),
            )
        )
    return energy * factors


def build_profile_kernel(
    solver,
    scalar_solver,
    *,
    diagnostic_ce: float,
    diagnostic_cc: float,
    spectrum_level: int,
):
    """Build one compiled, non-mutating paper-statistics observer."""

    dz = solver.dz
    delta = (solver.dx * solver.dy * solver.dz) ** (1.0 / 3.0)

    def kernel(velocity, scalar, pressure, sgs_coefficient, wall_velocity):
        cells = solver.cell_centered_velocity(velocity)
        mean = _plane(cells)
        fluctuation = cells - mean[:, None, None, :]
        scalar_mean = _plane(scalar)
        scalar_fluctuation = scalar - scalar_mean[:, None, None]
        variances = _plane(fluctuation * fluctuation)
        resolved_uw = _plane(fluctuation[..., 0] * fluctuation[..., 2])
        resolved_vw = _plane(fluctuation[..., 1] * fluctuation[..., 2])
        resolved_wc = _plane(fluctuation[..., 2] * scalar_fluctuation)

        velocity_gradient = solver.velocity_gradient(cells)
        momentum_diffusivity = solver.sgs_viscosity(
            cells,
            sgs_coefficient,
            gradient=velocity_gradient,
        )
        if solver.lasd_closure is None:
            momentum_diffusivity = jnp.maximum(
                momentum_diffusivity
                - solver.config.amd.molecular_viscosity,
                0.0,
            )
        sgs_tke = solver.diagnostic_sgs_tke(
            cells,
            sgs_coefficient,
            gradient=velocity_gradient,
            dissipation_coefficient=diagnostic_ce,
        )
        momentum_faces = solver.vertical_sgs_stress_flux(
            cells,
            sgs_coefficient,
            gradient=velocity_gradient,
            wall_velocity=wall_velocity,
        )
        # The solver face quantity is inward-normal traction.  Andrén plots the
        # physical turbulent flux <u_i w>, hence the minus sign.
        momentum_flux = -_faces_to_cells(momentum_faces)

        (
            scalar_diffusivity,
            scalar_gradient,
            scalar_flux_x,
            scalar_flux_y,
            scalar_flux_z,
        ) = scalar_solver.sgs_fluxes(scalar, velocity_gradient)
        scalar_flux_at_cells = _faces_to_cells(scalar_flux_z)
        lower_gradient = jnp.concatenate(
            (
                jnp.where(
                    jnp.mean(scalar_diffusivity[0]) > 0.0,
                    -scalar_solver.model.lower_surface_flux
                    / jnp.maximum(
                        jnp.mean(scalar_diffusivity[0]),
                        jnp.finfo(scalar.dtype).tiny,
                    ),
                    0.0,
                )[None, None, None]
                * jnp.ones_like(scalar[:1]),
                (scalar[1:] - scalar[:-1]) / dz,
            ),
            axis=0,
        )
        upper_gradient = jnp.concatenate(
            (
                (scalar[1:] - scalar[:-1]) / dz,
                jnp.zeros_like(scalar[-1:]),
            ),
            axis=0,
        )
        diagnostic_scalar_gradient_z = 0.5 * (
            lower_gradient + upper_gradient
        )
        scalar_dissipation = -(
            scalar_flux_x * scalar_gradient[..., 0]
            + scalar_flux_y * scalar_gradient[..., 1]
            + scalar_flux_at_cells * diagnostic_scalar_gradient_z
        )
        strain = 0.5 * (
            velocity_gradient + jnp.swapaxes(velocity_gradient, -1, -2)
        )
        strain_magnitude = jnp.sqrt(
            2.0 * jnp.einsum("...ij,...ij->...", strain, strain)
        )
        effective_scalar_coefficient = scalar_diffusivity / jnp.maximum(
            delta**2 * strain_magnitude,
            jnp.finfo(scalar.dtype).tiny,
        )
        scalar_length = delta * jnp.sqrt(
            jnp.maximum(effective_scalar_coefficient, 0.0)
        )
        scalar_variance_numerator = (
            2.0 * scalar_length * scalar_dissipation / diagnostic_cc
        )

        pressure_fluctuation = pressure - _plane(pressure)[:, None, None]
        resolved_tke = 0.5 * jnp.sum(variances, axis=-1)
        modes = jnp.arange(cells.shape[2] // 2 + 1, dtype=cells.dtype)
        values = (
            mean[:, 0],
            mean[:, 1],
            mean[:, 2],
            scalar_mean,
            variances[:, 0],
            variances[:, 1],
            variances[:, 2],
            _plane(scalar_fluctuation**2),
            resolved_tke,
            resolved_uw,
            resolved_vw,
            resolved_wc,
            _plane(sgs_tke),
            _plane(scalar_variance_numerator),
            _plane(momentum_flux[..., 0]),
            _plane(momentum_flux[..., 1]),
            _plane(scalar_flux_at_cells),
            _plane(momentum_diffusivity),
            _plane(scalar_diffusivity),
            _plane(fluctuation[..., 2] * pressure_fluctuation),
            jnp.sqrt(_plane(pressure_fluctuation**2)),
            _plane(
                solver.resolved_tke_sgs_dissipation(
                    cells,
                    sgs_coefficient,
                    gradient=velocity_gradient,
                )
            ),
            modes,
            _spectrum(cells[..., 0], spectrum_level),
            _spectrum(cells[..., 1], spectrum_level),
            _spectrum(cells[..., 2], spectrum_level),
            _spectrum(scalar, spectrum_level),
            jnp.full_like(modes, (spectrum_level + 0.5) * dz),
        )
        return values

    return jax.jit(kernel)


def build_history_kernel(solver, *, diagnostic_ce: float):
    def kernel(velocity, sgs_coefficient, wall_velocity):
        cells = solver.cell_centered_velocity(velocity)
        mean = _plane(cells)
        fluctuation = cells - mean[:, None, None, :]
        resolved_tke = 0.5 * _plane(jnp.sum(fluctuation**2, axis=-1))
        gradient = solver.velocity_gradient(cells)
        sgs_tke = _plane(
            solver.diagnostic_sgs_tke(
                cells,
                sgs_coefficient,
                gradient=gradient,
                dissipation_coefficient=diagnostic_ce,
            )
        )
        faces = solver.vertical_sgs_stress_flux(
            cells,
            sgs_coefficient,
            gradient=gradient,
            wall_velocity=wall_velocity,
        )
        wall_flux = -jnp.mean(faces[0], axis=(0, 1))
        ustar = jnp.sqrt(jnp.hypot(wall_flux[0], wall_flux[1]))
        integrated_resolved = jnp.sum(resolved_tke) * solver.dz
        integrated_sgs = jnp.sum(sgs_tke) * solver.dz
        tiny = jnp.finfo(cells.dtype).eps
        cu = -solver.config.coriolis_vertical * (
            jnp.sum(mean[:, 1] - solver.config.geostrophic_wind[1]) * solver.dz
        ) / jnp.where(jnp.abs(wall_flux[0]) > tiny, wall_flux[0], jnp.nan)
        cv = solver.config.coriolis_vertical * (
            jnp.sum(mean[:, 0] - solver.config.geostrophic_wind[0]) * solver.dz
        ) / jnp.where(jnp.abs(wall_flux[1]) > tiny, wall_flux[1], jnp.nan)
        return ustar, integrated_resolved, integrated_sgs, cu, cv

    return jax.jit(kernel)


def build_budget_kernel(solver, scalar_solver):
    pressure_boundaries = solver.pressure_solver.operator.boundaries

    def kernel(velocity, scalar, pressure, sgs_coefficient, wall_velocity):
        cells = solver.cell_centered_velocity(velocity)
        mean = _plane(cells)
        fluctuation = cells - mean[:, None, None, :]
        scalar_mean = _plane(scalar)
        scalar_fluctuation = scalar - scalar_mean[:, None, None]
        velocity_gradient = solver.velocity_gradient(cells)

        advection = solver.conservative_advection(velocity, cells)
        if solver.config.mp5_dissipation_strength > 0.0:
            advection += solver.advection_dissipation(velocity, cells)
        momentum_sgs = solver.sgs_tendency(
            cells,
            sgs_coefficient,
            gradient=velocity_gradient,
            wall_velocity=wall_velocity,
        )
        forcing = solver.forcing_tendency(cells)
        pressure_faces = mac_pressure_gradient(
            pressure,
            solver.grid,
            pressure_boundaries,
        )
        pressure_tendency = -solver.cell_centered_velocity(pressure_faces)

        uw_advective = _plane(
            fluctuation[..., 0] * advection[..., 2]
            + fluctuation[..., 2] * advection[..., 0]
        )
        w_variance = _plane(fluctuation[..., 2] ** 2)
        uw_production = -w_variance * jnp.gradient(mean[:, 0], solver.dz)
        uw_transport = uw_advective - uw_production
        uw_subgrid = _plane(
            fluctuation[..., 0] * momentum_sgs[..., 2]
            + fluctuation[..., 2] * momentum_sgs[..., 0]
        )
        uw_pressure = _plane(
            fluctuation[..., 0] * pressure_tendency[..., 2]
            + fluctuation[..., 2] * pressure_tendency[..., 0]
        )
        uw_coriolis = _plane(
            fluctuation[..., 0] * forcing[..., 2]
            + fluctuation[..., 2] * forcing[..., 0]
        )

        scalar_advection = scalar_solver.advective_tendency(scalar, velocity)
        scalar_sgs = scalar_solver.sgs_tendency(scalar, velocity_gradient)
        wc_advective = _plane(
            scalar_fluctuation * advection[..., 2]
            + fluctuation[..., 2] * scalar_advection
        )
        wc_production = -w_variance * jnp.gradient(scalar_mean, solver.dz)
        wc_transport = wc_advective - wc_production
        wc_subgrid = _plane(
            scalar_fluctuation * momentum_sgs[..., 2]
            + fluctuation[..., 2] * scalar_sgs
        )
        wc_pressure = _plane(
            scalar_fluctuation * pressure_tendency[..., 2]
        )
        wc_coriolis = _plane(scalar_fluctuation * forcing[..., 2])
        return (
            uw_production,
            uw_subgrid,
            uw_transport,
            uw_pressure,
            uw_coriolis,
            _plane(fluctuation[..., 0] * fluctuation[..., 2]),
            wc_production,
            wc_subgrid,
            wc_transport,
            wc_pressure,
            wc_coriolis,
            _plane(fluctuation[..., 2] * scalar_fluctuation),
        )

    return jax.jit(kernel)


def average_samples(samples: list[tuple[np.ndarray, ...]]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("at least one profile sample is required")
    return {
        name: np.mean([sample[index] for sample in samples], axis=0)
        for index, name in enumerate(PROFILE_NAMES)
    }


def diagnostic_scalar_variance(averaged: dict[str, np.ndarray]) -> np.ndarray:
    """Apply the paper diagnostic after horizontal/time averaging."""
    return np.maximum(
        averaged["sgs_scalar_variance_numerator"]
        / np.sqrt(
            np.maximum(
                averaged["sgs_tke"],
                np.finfo(averaged["sgs_tke"].dtype).tiny,
            )
        ),
        0.0,
    )


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def averaged_budget(
    times: list[float],
    samples: list[tuple[np.ndarray, ...]],
    *,
    ustar: float,
    scalar_surface_flux: float,
    coriolis: float,
    dz: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if len(samples) < 2:
        raise ValueError("budget tendency requires at least two samples")
    elapsed = times[-1] - times[0]
    if elapsed <= 0.0:
        raise ValueError("budget sample times must increase")
    raw = {
        name: np.mean([sample[index] for sample in samples], axis=0)
        for index, name in enumerate(BUDGET_NAMES)
    }

    def finish(prefix: str, flux_scale: float):
        terms = {
            name: raw[f"{prefix}_{name}"]
            for name in ("production", "subgrid", "transport", "pressure", "coriolis")
        }
        tendency = (
            samples[-1][BUDGET_NAMES.index(f"{prefix}_resolved_flux")]
            - samples[0][BUDGET_NAMES.index(f"{prefix}_resolved_flux")]
        ) / elapsed
        rhs = sum(terms.values(), np.zeros_like(tendency))
        scale = coriolis * flux_scale
        result = {name: value / scale for name, value in terms.items()}
        result["tendency"] = tendency / scale
        result["closure_residual"] = (tendency - rhs) / scale
        z = (np.arange(tendency.size) + 0.5) * dz
        result["height"] = z * coriolis / ustar
        return result

    return (
        finish("uw", ustar**2),
        finish("wc", scalar_surface_flux),
    )


def write_budget(path: Path, budget: dict[str, np.ndarray]) -> None:
    names = (
        "height",
        "production",
        "subgrid",
        "transport",
        "pressure",
        "coriolis",
        "tendency",
        "closure_residual",
    )
    np.savetxt(
        path,
        np.column_stack([budget[name] for name in names]),
        delimiter=",",
        header=",".join(names),
        comments="",
    )
