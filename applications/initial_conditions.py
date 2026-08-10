"""Interpret generic tabulated velocity/TKE initial conditions."""

from __future__ import annotations

import numpy as np

from .boussinesq import BoussinesqCase, TabulatedVelocityTKE


REQUIRED_COLUMNS = ("z_m", "u_m_s", "v_m_s", "tke_m2_s2")


def load_initial_profile(case: BoussinesqCase) -> np.ndarray:
    """Read and validate the configured cell-centred initial profile."""

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
    if np.any(table["tke_m2_s2"] < 0.0):
        raise ValueError("initial TKE must be nonnegative")
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
    decomposition,
    algebra,
    pressure_solver,
):
    """Materialize the configured profile and project its velocity."""

    from jaxwind.domain import (
        Accepted,
        AddressableField,
        Candidate,
        Cell,
        PassiveScalarConcentration,
        VerticalBoundary,
        VerticalVelocity,
        XVelocity,
        YVelocity,
        ZFace,
    )
    from jaxwind.interpreters.jax_zslab import ZFaceFieldContext
    from jaxwind.operators import VelocityVector, project
    from jaxwind.physics import BoussinesqFields

    table = load_initial_profile(case)
    grid = case.physical_grid
    dtype = getattr(jnp, case.pressure.dtype)
    z = (jnp.arange(grid.nz, dtype=dtype) + 0.5) * grid.dz
    upper_z = (jnp.arange(grid.nz, dtype=dtype) + 1.0) * grid.dz
    table_z = jnp.asarray(table["z_m"], dtype=dtype)
    table_u = jnp.asarray(table["u_m_s"], dtype=dtype)
    table_v = jnp.asarray(table["v_m_s"], dtype=dtype)
    table_tke = jnp.asarray(table["tke_m2_s2"], dtype=dtype)
    mean_u = jnp.interp(z, table_z, table_u)
    mean_v = jnp.interp(z, table_z, table_v)
    cell_tke = jnp.interp(z, table_z, table_tke)
    face_tke = jnp.interp(upper_z, table_z, table_tke, right=0.0)

    shape = (1, grid.nz, grid.ny, grid.nx)
    keys = jax.random.split(
        jax.random.PRNGKey(case.initial_condition.seed),
        3,
    )
    component_rms = jnp.sqrt((2.0 / 3.0) * cell_tke)[None, :, None, None]
    face_rms = jnp.sqrt((2.0 / 3.0) * face_tke)[None, :, None, None]
    u = mean_u[None, :, None, None] + component_rms * _unit_plane_noise(
        jax,
        jnp,
        keys[0],
        shape,
        dtype,
    )
    v = mean_v[None, :, None, None] + component_rms * _unit_plane_noise(
        jax,
        jnp,
        keys[1],
        shape,
        dtype,
    )
    w = face_rms * _unit_plane_noise(
        jax,
        jnp,
        keys[2],
        shape,
        dtype,
    )
    w = w.at[:, -1].set(0.0)
    lower = jnp.zeros((grid.ny, grid.nx), dtype=dtype)
    velocity = VelocityVector(
        AddressableField(
            XVelocity,
            Cell,
            decomposition.regions(Cell),
            Candidate,
            case.mechanical_scales.to_execution_velocity(u).astype(dtype),
        ),
        AddressableField(
            YVelocity,
            Cell,
            decomposition.regions(Cell),
            Candidate,
            case.mechanical_scales.to_execution_velocity(v).astype(dtype),
        ),
        ZFaceFieldContext(
            AddressableField(
                VerticalVelocity,
                ZFace,
                decomposition.regions(ZFace),
                Candidate,
                case.mechanical_scales.to_execution_velocity(w).astype(dtype),
            ),
            lower,
        ),
    )
    projected = project(
        velocity,
        dt=case.integrator.dt,
        normal_boundary=VerticalBoundary(0.0, 0.0),
        algebra=algebra,
        pressure_solver=pressure_solver,
    )
    scalar = AddressableField(
        PassiveScalarConcentration,
        Cell,
        decomposition.regions(Cell),
        Accepted,
        jnp.zeros(shape, dtype=dtype),
    )
    fields = BoussinesqFields(projected.velocity, scalar)
    return algebra.initialize_lasd_closure(fields, case.model)


__all__ = [
    "TabulatedVelocityTKE",
    "build_initial_fields",
    "load_initial_profile",
]
