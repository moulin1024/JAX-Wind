"""Paper-aligned diagnostics for finite-volume atmospheric flow.

The prognostic solver deliberately carries only the fields needed to advance
the equations.  This module reconstructs turbulence statistics from those
fields at output times, using the same staggered interpolations and AMD
closure quantities as the momentum and scalar operators.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxwind.domain.grid import UniformGrid

from .operators import cell_velocity
from .scalar import PassiveScalar
from .sgs import (
    AnisotropicMinimumDissipation,
    cell_gradients,
    eddy_viscosity,
    edge_gradients,
)
from .state import Boundaries, StaggeredVelocity
from .wall import MoninObukhovWall, surface_stress


def _plane_mean(values: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(values, axis=(-2, -1))


def _cell_fluctuations(
    values: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    mean = _plane_mean(values)
    return mean, values - mean[:, None, None]


def _xz_edge_viscosity(viscosity: jnp.ndarray) -> jnp.ndarray:
    rolled = jnp.roll(viscosity, 1, axis=2)
    interior = 0.25 * (
        viscosity[:-1] + viscosity[1:] + rolled[:-1] + rolled[1:]
    )
    wall = jnp.zeros_like(viscosity[:1])
    return jnp.concatenate((wall, interior, wall), axis=0)


def _yz_edge_viscosity(viscosity: jnp.ndarray) -> jnp.ndarray:
    rolled = jnp.roll(viscosity, 1, axis=1)
    interior = 0.25 * (
        viscosity[:-1] + viscosity[1:] + rolled[:-1] + rolled[1:]
    )
    wall = jnp.zeros_like(viscosity[:1])
    return jnp.concatenate((wall, interior, wall), axis=0)


def atmospheric_profile_diagnostics(
    velocity: StaggeredVelocity,
    pressure: jnp.ndarray,
    scalar: jnp.ndarray,
    grid: UniformGrid,
    boundaries: Boundaries,
    wall: MoninObukhovWall | None,
    scalar_model: PassiveScalar,
    subfilter: AnisotropicMinimumDissipation,
    *,
    x_velocity_offset: float = 0.0,
    y_velocity_offset: float = 0.0,
    lower_stress_x: jnp.ndarray | float | None = None,
    lower_stress_y: jnp.ndarray | float | None = None,
    lower_scalar_flux: jnp.ndarray | float | None = None,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    dict[str, jnp.ndarray],
]:
    """Return instantaneous fields and horizontally averaged diagnostics.

    Momentum and scalar SGS fluxes use the sign convention of resolved
    covariances: a downward surface momentum flux is negative and the imposed
    upward scalar flux is positive.  AMD has no prognostic SGS energy, so no
    modeled SGS-TKE contribution is fabricated.
    """

    u, v, w = cell_velocity(velocity)
    u = u + jnp.asarray(x_velocity_offset, u.dtype)
    v = v + jnp.asarray(y_velocity_offset, v.dtype)
    mean_u, u_fluctuation = _cell_fluctuations(u)
    mean_v, v_fluctuation = _cell_fluctuations(v)
    mean_w, w_fluctuation = _cell_fluctuations(w)
    mean_scalar, scalar_fluctuation = _cell_fluctuations(scalar)

    gradients = edge_gradients(velocity, grid, boundaries)
    viscosity = eddy_viscosity(
        velocity,
        grid,
        boundaries,
        subfilter,
        gradients=gradients,
    )
    tensor = cell_gradients(gradients)
    strain = [
        [0.5 * (tensor[i][j] + tensor[j][i]) for j in range(3)]
        for i in range(3)
    ]
    strain_contraction = jnp.zeros_like(viscosity)
    for i in range(3):
        for j in range(3):
            strain_contraction = strain_contraction + strain[i][j] ** 2
    tke_sgs_transfer = -2.0 * viscosity * strain_contraction

    momentum_flux_x = -_xz_edge_viscosity(viscosity) * (
        gradients["xz"] + gradients["zx"]
    )
    momentum_flux_y = -_yz_edge_viscosity(viscosity) * (
        gradients["yz"] + gradients["zy"]
    )
    if lower_stress_x is None or lower_stress_y is None:
        if wall is None:
            resolved_stress_x = jnp.asarray(0.0, velocity.x.dtype)
            resolved_stress_y = jnp.asarray(0.0, velocity.y.dtype)
        else:
            resolved_stress_x, resolved_stress_y = surface_stress(
                velocity, grid, wall
            )
    else:
        resolved_stress_x = lower_stress_x
        resolved_stress_y = lower_stress_y
    momentum_flux_x = momentum_flux_x.at[0].set(-resolved_stress_x)
    momentum_flux_y = momentum_flux_y.at[0].set(-resolved_stress_y)
    sgs_uw = 0.5 * _plane_mean(
        momentum_flux_x[:-1] + momentum_flux_x[1:]
    )
    sgs_vw = 0.5 * _plane_mean(
        momentum_flux_y[:-1] + momentum_flux_y[1:]
    )

    scalar_faces = jnp.concatenate(
        (
            scalar[:1],
            0.5 * (scalar[:-1] + scalar[1:]),
            scalar[-1:],
        ),
        axis=0,
    )
    _, scalar_face_fluctuation = _cell_fluctuations(scalar_faces)
    _, w_face_fluctuation = _cell_fluctuations(velocity.z)
    resolved_wc_faces = _plane_mean(
        w_face_fluctuation * scalar_face_fluctuation
    )
    resolved_wc = 0.5 * (
        resolved_wc_faces[:-1] + resolved_wc_faces[1:]
    )

    diffusivity = (
        jnp.asarray(scalar_model.diffusivity, scalar.dtype)
        + viscosity / scalar_model.turbulent_prandtl
    )
    interior_diffusivity = 0.5 * (diffusivity[:-1] + diffusivity[1:])
    interior_scalar_flux = -interior_diffusivity * (
        scalar[1:] - scalar[:-1]
    ) / grid.dz
    resolved_lower_flux = (
        scalar_model.lower_flux
        if lower_scalar_flux is None
        else lower_scalar_flux
    )
    lower_scalar_flux_field = jnp.full_like(
        scalar[:1], resolved_lower_flux
    )
    upper_scalar_flux = jnp.full_like(scalar[:1], scalar_model.upper_flux)
    scalar_flux_faces = jnp.concatenate(
        (lower_scalar_flux_field, interior_scalar_flux, upper_scalar_flux),
        axis=0,
    )
    sgs_wc = 0.5 * _plane_mean(
        scalar_flux_faces[:-1] + scalar_flux_faces[1:]
    )

    x_diffusivity = 0.5 * (diffusivity + jnp.roll(diffusivity, 1, axis=2))
    y_diffusivity = 0.5 * (diffusivity + jnp.roll(diffusivity, 1, axis=1))
    sgs_uc = _plane_mean(
        -x_diffusivity
        * (scalar - jnp.roll(scalar, 1, axis=2))
        / grid.dx
    )
    sgs_vc = _plane_mean(
        -y_diffusivity
        * (scalar - jnp.roll(scalar, 1, axis=1))
        / grid.dy
    )

    pressure_mean, pressure_fluctuation = _cell_fluctuations(pressure)
    del pressure_mean
    resolved_energy = 0.5 * (
        u_fluctuation**2 + v_fluctuation**2 + w_fluctuation**2
    )
    updraft = w_fluctuation > 0.0
    updraft_count = jnp.sum(updraft, axis=(-2, -1))
    safe_updraft_count = jnp.maximum(updraft_count, 1)
    plane_size = float(grid.nx * grid.ny)

    profiles = {
        "u": mean_u,
        "v": mean_v,
        "w": mean_w,
        "scalar": mean_scalar,
        "u_variance": _plane_mean(u_fluctuation**2),
        "v_variance": _plane_mean(v_fluctuation**2),
        "w_variance": _plane_mean(w_fluctuation**2),
        "scalar_variance": _plane_mean(scalar_fluctuation**2),
        "resolved_uw": _plane_mean(u_fluctuation * w_fluctuation),
        "resolved_vw": _plane_mean(v_fluctuation * w_fluctuation),
        "resolved_wc": resolved_wc,
        "resolved_uc": _plane_mean(u_fluctuation * scalar_fluctuation),
        "resolved_vc": _plane_mean(v_fluctuation * scalar_fluctuation),
        "sgs_tke": jnp.zeros_like(mean_u),
        "sgs_uw": sgs_uw,
        "sgs_vw": sgs_vw,
        "sgs_wc": sgs_wc,
        "sgs_uc": sgs_uc,
        "sgs_vc": sgs_vc,
        "resolved_tke_sgs_transfer": _plane_mean(tke_sgs_transfer),
        "momentum_diffusivity": _plane_mean(viscosity),
        "scalar_diffusivity": _plane_mean(diffusivity),
        "pressure_variance": _plane_mean(pressure_fluctuation**2),
        "w_third_moment": _plane_mean(w_fluctuation**3),
        "updraft_fraction": updraft_count / plane_size,
        "updraft_w": jnp.sum(
            jnp.where(updraft, w_fluctuation, 0.0), axis=(-2, -1)
        )
        / safe_updraft_count,
        "updraft_scalar_excess": jnp.sum(
            jnp.where(updraft, scalar_fluctuation, 0.0), axis=(-2, -1)
        )
        / safe_updraft_count,
        "resolved_energy_vertical_transport": _plane_mean(
            w_fluctuation * resolved_energy
        ),
        "pressure_vertical_transport": _plane_mean(
            pressure_fluctuation * w_fluctuation
        ),
    }
    return (u, v, w, scalar), profiles


def atmospheric_history_diagnostics(
    velocity: StaggeredVelocity,
    grid: UniformGrid,
    wall: MoninObukhovWall,
    *,
    coriolis: float,
    geostrophic_u: float,
    geostrophic_v: float,
) -> dict[str, jnp.ndarray]:
    """Return total resolved TKE and momentum-stationarity metrics."""

    u, v, w = cell_velocity(velocity)
    mean_u, u_fluctuation = _cell_fluctuations(u)
    mean_v, v_fluctuation = _cell_fluctuations(v)
    _, w_fluctuation = _cell_fluctuations(w)
    resolved_tke = 0.5 * _plane_mean(
        u_fluctuation**2 + v_fluctuation**2 + w_fluctuation**2
    )
    stress_x, stress_y = surface_stress(velocity, grid, wall)
    surface_uw = -jnp.mean(stress_x)
    surface_vw = -jnp.mean(stress_y)
    integrated_u_deficit = jnp.sum(mean_u - geostrophic_u) * grid.dz
    integrated_v_deficit = jnp.sum(mean_v - geostrophic_v) * grid.dz
    tiny = jnp.finfo(velocity.x.dtype).tiny
    stationarity_u = jnp.where(
        jnp.abs(surface_uw) > tiny,
        -coriolis * integrated_v_deficit / surface_uw,
        jnp.nan,
    )
    stationarity_v = jnp.where(
        jnp.abs(surface_vw) > tiny,
        coriolis * integrated_u_deficit / surface_vw,
        jnp.nan,
    )
    integrated = jnp.sum(resolved_tke) * grid.dz
    return {
        "surface_uw_m2_s2": surface_uw,
        "surface_vw_m2_s2": surface_vw,
        "momentum_stationarity_cu": stationarity_u,
        "momentum_stationarity_cv": stationarity_v,
        "integrated_resolved_tke_m3_s2": integrated,
        "integrated_sgs_tke_m3_s2": jnp.asarray(0.0, integrated.dtype),
        "integrated_total_tke_m3_s2": integrated,
    }


__all__ = [
    "atmospheric_history_diagnostics",
    "atmospheric_profile_diagnostics",
]
