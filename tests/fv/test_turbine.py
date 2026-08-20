from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from jaxwind.domain import ScaleSystem, UniformGrid
from jaxwind.fv import StaggeredVelocity, build_adbem_forcing
from jaxwind.windfarm import RigidBladeElementDisk, load_openfast_rigid_turbine


ROOT = Path(__file__).resolve().parents[2]
OPENFAST = (
    ROOT
    / "tests"
    / "fixtures"
    / "openfast"
    / "nrel5mw"
    / "NREL5MW_Rigid_Smoke.fst"
)


def test_adbem_forcing_maps_shared_turbine_loads_to_open_fv_faces() -> None:
    grid = UniformGrid(16, 8, 20, 320.0, 160.0, 240.0)
    rotor = load_openfast_rigid_turbine(OPENFAST)
    turbine = RigidBladeElementDisk(
        rotor=rotor,
        x_m=120.0,
        y_m=80.0,
        smoothing_width_m=20.0,
        hub_height_m=90.0,
        rotor_speed_rpm=8.0,
        pitch_degrees=0.0,
        body_smoothing_width_m=20.0,
    )
    scales = ScaleSystem(1.0, 1.0)
    forcing = build_adbem_forcing(
        grid,
        turbine.to_actuator_disk(scales=scales),
        turbine.to_nacelle_tower(scales=scales),
    )
    velocity = StaggeredVelocity(
        jnp.full((grid.nz, grid.ny, grid.nx + 1), 10.0, jnp.float32),
        jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx), jnp.float32),
    )

    result = jax.jit(forcing)(velocity, jnp.asarray(0.0, jnp.float32))

    assert result.x.shape == (grid.nz, grid.ny, grid.nx + 1)
    assert result.y.shape == (grid.nz, grid.ny, grid.nx)
    assert result.z.shape == (grid.nz + 1, grid.ny, grid.nx)
    assert bool(jnp.all(jnp.isfinite(result.x)))
    assert bool(jnp.all(jnp.isfinite(result.y)))
    assert bool(jnp.all(jnp.isfinite(result.z)))
    assert float(jnp.min(result.x)) < 0.0
    assert float(jnp.max(jnp.abs(result.y))) > 0.0
    assert float(jnp.max(jnp.abs(result.z))) > 0.0
    assert bool(jnp.all(result.z[0] == 0.0))
    assert bool(jnp.all(result.z[-1] == 0.0))
