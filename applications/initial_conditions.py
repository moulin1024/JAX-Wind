"""Interpret one generic tabulated Boussinesq initial state."""

from __future__ import annotations

import numpy as np

from .boussinesq import BoussinesqCase, TabulatedBoussinesqState


REQUIRED_COLUMNS = (
    "z_m",
    "u_m_s",
    "v_m_s",
    "w_upper_m_s",
    "scalar",
    "u_rms_m_s",
    "v_rms_m_s",
    "w_upper_rms_m_s",
    "scalar_rms",
)


def load_initial_profile(case: BoussinesqCase) -> np.ndarray:
    """Read one mean-plus-RMS profile shared by every ABL case."""

    table = np.genfromtxt(
        case.initial_condition.path,
        delimiter=",",
        names=True,
    )
    if table.dtype.names != REQUIRED_COLUMNS:
        raise ValueError(
            "initial profile columns must be " + ", ".join(REQUIRED_COLUMNS)
        )
    if table.shape != (case.physical_grid.nz,):
        raise ValueError("initial profile must contain one row per vertical cell")
    z = np.asarray(table["z_m"], dtype=np.float64)
    expected_z = (
        np.arange(case.physical_grid.nz, dtype=np.float64) + 0.5
    ) * case.physical_grid.dz
    if not np.allclose(z, expected_z, rtol=0.0, atol=1.0e-12):
        raise ValueError("initial profile heights must match the case grid")
    if not all(np.all(np.isfinite(table[name])) for name in REQUIRED_COLUMNS):
        raise ValueError("initial profile values must be finite")
    for name in (
        "u_rms_m_s",
        "v_rms_m_s",
        "w_upper_rms_m_s",
        "scalar_rms",
    ):
        if np.any(table[name] < 0.0):
            raise ValueError(f"initial profile {name} must be nonnegative")
    return table


def _unit_plane_noise(jax, jnp, key, shape, dtype):
    noise = jax.random.uniform(key, shape, dtype, minval=-0.5, maxval=0.5)
    noise -= jnp.mean(noise, axis=(-2, -1), keepdims=True)
    rms = jnp.sqrt(jnp.mean(noise * noise, axis=(-2, -1), keepdims=True))
    return noise / jnp.maximum(rms, jnp.finfo(dtype).tiny)


def build_initial_fields(
    case: BoussinesqCase,
    *,
    jax,
    jnp,
    solver,
):
    """Materialize and project the configured mean-plus-RMS initial state."""

    from jaxwind.domain import Accepted
    from jaxwind.physics import BoussinesqFields

    table = load_initial_profile(case)
    grid = case.physical_grid
    dtype = getattr(jnp, case.pressure.dtype)
    z = (jnp.arange(grid.nz, dtype=dtype) + 0.5) * grid.dz
    table_z = jnp.asarray(table["z_m"], dtype=dtype)

    def cell_profile(name: str):
        return jnp.interp(z, table_z, jnp.asarray(table[name], dtype=dtype))

    shape = (grid.nz, grid.ny, grid.nx)
    keys = jax.random.split(jax.random.PRNGKey(case.initial_condition.seed), 3)
    u_noise = _unit_plane_noise(jax, jnp, keys[0], shape, dtype)
    v_noise = _unit_plane_noise(jax, jnp, keys[1], shape, dtype)
    coupled_noise = _unit_plane_noise(jax, jnp, keys[2], shape, dtype)
    frame_u, frame_v = case.advection_frame_velocity_m_s
    u = cell_profile("u_m_s")[:, None, None] - frame_u + (
        cell_profile("u_rms_m_s")[:, None, None] * u_noise
    )
    v = cell_profile("v_m_s")[:, None, None] - frame_v + (
        cell_profile("v_rms_m_s")[:, None, None] * v_noise
    )
    w = jnp.asarray(table["w_upper_m_s"], dtype=dtype)[:, None, None] + (
        jnp.asarray(table["w_upper_rms_m_s"], dtype=dtype)[:, None, None]
        * coupled_noise
    )
    w = w.at[-1].set(0.0)
    scalar_physical = cell_profile("scalar")[:, None, None] + (
        cell_profile("scalar_rms")[:, None, None] * coupled_noise
    )

    lower = jnp.zeros((grid.ny, grid.nx), dtype=dtype)
    velocity = solver.candidate_velocity(
        case.mechanical_scales.to_execution_velocity(u).astype(dtype),
        case.mechanical_scales.to_execution_velocity(v).astype(dtype),
        case.mechanical_scales.to_execution_velocity(w).astype(dtype),
        lower_boundary=lower,
    )
    projected = solver.project_initial_velocity(velocity)
    scalar = solver.cell_field(
        case.scalar_scales.field_quantity,
        Accepted,
        case.scalar_scales.to_execution_scalar(scalar_physical).astype(dtype),
    )
    fields = BoussinesqFields(projected, scalar)
    return solver.initialize_fields(fields)


__all__ = [
    "TabulatedBoussinesqState",
    "build_initial_fields",
    "load_initial_profile",
]
