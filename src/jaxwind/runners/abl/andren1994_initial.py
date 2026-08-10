"""Andrén et al. (1994) constants and Table A.1 initialization."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ...domain import (
    AddressableField,
    Candidate,
    Cell,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from ...interpreters.jax_zslab import ZFaceFieldContext
from ...operators import VelocityVector


F_CORIOLIS = 1.0e-4
GEOSTROPHIC_SPEED = 10.0
CANONICAL_HOURS = 10.0 / F_CORIOLIS / 3600.0
STATISTICS_START_HOURS = 7.0 / F_CORIOLIS / 3600.0
STATISTICS_WINDOW_HOURS = 3.0 / F_CORIOLIS / 3600.0


def load_initial_profiles(path: Path) -> np.ndarray:
    """Read and validate the 40-level Table A.1 profile."""

    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape != (40,):
        raise ValueError("Andrén Table A.1 must contain exactly 40 levels")
    if not np.allclose(np.diff(data["z_m"]), 37.5):
        raise ValueError("Andrén Table A.1 heights must use the 37.5 m grid")
    return data


def _unit_plane_noise(key, shape, dtype):
    noise = jax.random.uniform(key, shape, dtype, minval=-0.5, maxval=0.5)
    noise -= jnp.mean(noise, axis=(-2, -1), keepdims=True)
    rms = jnp.sqrt(jnp.mean(noise * noise, axis=(-2, -1), keepdims=True))
    return noise / jnp.maximum(rms, jnp.finfo(dtype).tiny)


def initial_velocity(
    physical_grid,
    decomposition,
    scales,
    dtype,
    *,
    profiles_path: Path,
    seed: int,
) -> VelocityVector:
    """Interpret Table A.1 and its uniform random perturbation law."""

    table = load_initial_profiles(profiles_path)
    z = (jnp.arange(physical_grid.nz, dtype=dtype) + 0.5) * physical_grid.dz
    upper_z = (jnp.arange(physical_grid.nz, dtype=dtype) + 1.0) * physical_grid.dz
    table_z = jnp.asarray(table["z_m"], dtype=dtype)
    table_u = jnp.asarray(table["u_m_s"], dtype=dtype)
    table_v = jnp.asarray(table["v_m_s"], dtype=dtype)
    table_tke = jnp.asarray(table["tke_m2_s2"], dtype=dtype)
    mean_u = jnp.interp(z, table_z, table_u, left=table_u[0], right=table_u[-1])
    mean_v = jnp.interp(z, table_z, table_v, left=table_v[0], right=table_v[-1])
    cell_tke = jnp.interp(z, table_z, table_tke, left=table_tke[0], right=0.0)
    face_tke = jnp.interp(
        upper_z,
        table_z,
        table_tke,
        left=table_tke[0],
        right=0.0,
    )
    shape = (1, physical_grid.nz, physical_grid.ny, physical_grid.nx)
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    component_rms = jnp.sqrt((2.0 / 3.0) * cell_tke)[None, :, None, None]
    face_rms = jnp.sqrt((2.0 / 3.0) * face_tke)[None, :, None, None]
    u = mean_u[None, :, None, None] + component_rms * _unit_plane_noise(
        keys[0], shape, dtype
    )
    v = mean_v[None, :, None, None] + component_rms * _unit_plane_noise(
        keys[1], shape, dtype
    )
    w = face_rms * _unit_plane_noise(keys[2], shape, dtype)
    w = w.at[:, -1].set(0.0)
    lower = jnp.zeros((physical_grid.ny, physical_grid.nx), dtype=dtype)
    return VelocityVector(
        AddressableField(
            XVelocity,
            Cell,
            decomposition.regions(Cell),
            Candidate,
            scales.to_execution_velocity(u).astype(dtype),
        ),
        AddressableField(
            YVelocity,
            Cell,
            decomposition.regions(Cell),
            Candidate,
            scales.to_execution_velocity(v).astype(dtype),
        ),
        ZFaceFieldContext(
            AddressableField(
                VerticalVelocity,
                ZFace,
                decomposition.regions(ZFace),
                Candidate,
                scales.to_execution_velocity(w).astype(dtype),
            ),
            lower,
        ),
    )


__all__ = [
    "CANONICAL_HOURS",
    "F_CORIOLIS",
    "GEOSTROPHIC_SPEED",
    "STATISTICS_START_HOURS",
    "STATISTICS_WINDOW_HOURS",
    "initial_velocity",
    "load_initial_profiles",
]
