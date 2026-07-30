from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest


JAX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JAX_ROOT))


def test_pressure_driven_case_uses_low_cfl_timestep() -> None:
    config_path = JAX_ROOT / "configs" / "pressure_driven_neutral_abl_lasd.toml"
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    dt = float(config["time"]["dt"])
    steps = int(config["time"]["steps"])
    log_every = int(config["time"]["log_every"])
    grid = config["grid"]
    physics = config["physics"]
    numerics = config["numerics"]
    sponge = config["sponge"]
    assert float(grid["lz"]) == pytest.approx(1500.0)
    assert float(grid["lx"]) == pytest.approx(2.0 * np.pi * 1000.0)
    assert float(grid["ly"]) == pytest.approx(2.0 * np.pi * 1000.0)
    assert int(grid["nx"]) == int(grid["ny"])
    assert float(grid["z_i"]) == pytest.approx(1000.0)
    assert float(physics["bl_height"]) == pytest.approx(1000.0)
    assert physics["initial_condition"] == "log_law"
    assert int(grid["nx"]) == 64
    assert int(grid["ny"]) == 64
    assert int(grid["nz"]) == 64
    assert dt == pytest.approx(0.625)
    assert steps * dt == pytest.approx(20000.0)
    assert steps // log_every == 40
    assert numerics["horizontal_dealias"] is True
    assert numerics["pressure_filter_nyquist"] is True
    assert sponge["enabled"] is True
    assert float(sponge["start_height"]) == pytest.approx(float(grid["lz"]) - 100.0)
    assert float(sponge["timescale"]) == pytest.approx(90.0)
    assert float(sponge["power"]) == pytest.approx(2.0)
    assert sponge["target"] == "plane_mean"


def test_pressure_gradient_is_confined_below_1000_m() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.rhs import assemble_rhs

    params = Params(
        nx=4,
        ny=4,
        nz=13,
        lz=1.2,
        z_i=1000.0,
        bl_height=1000.0,
        u_fric=0.4,
        zo=0.1,
        dtype=jnp.float32,
    )
    zeros = jnp.zeros((params.nx, params.ny, params.nz), dtype=params.dtype)
    rhs = np.asarray(jax.block_until_ready(assemble_rhs(zeros, zeros, params, pressure_force=True)))
    z = (np.arange(params.nz) + 0.5) * params.dz * params.z_i
    expected = np.where(z <= 1000.0, params.driving_pressure_force, 0.0)
    np.testing.assert_allclose(
        rhs,
        np.broadcast_to(expected[None, None, :], (params.nx, params.ny, params.nz)),
        atol=1.0e-7,
    )
    assert np.max(z[expected > 0.0]) <= 1000.0
    assert np.min(z[expected == 0.0]) > 1000.0


def test_pressure_gradient_height_can_cover_the_full_domain() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.rhs import assemble_rhs

    params = Params(
        nx=4,
        ny=4,
        nz=8,
        lz=1.0,
        z_i=3.6,
        bl_height=2.0,
        pressure_force_height=3.6,
        u_fric=0.16140448019041365,
        zo=1.6100320393803134e-5,
        dtype=jnp.float32,
    )
    zeros = jnp.zeros((params.nx, params.ny, params.nz), dtype=params.dtype)
    rhs = np.asarray(
        jax.block_until_ready(
            assemble_rhs(zeros, zeros, params, pressure_force=True)
        )
    )

    np.testing.assert_allclose(rhs, params.driving_pressure_force, rtol=2.0e-6)
    assert params.forcing_height == pytest.approx(1.0)
    assert params.pressure_ustar == pytest.approx(params.u_fric)
    assert params.driving_pressure_force == pytest.approx(params.u_fric**2)


@pytest.mark.parametrize("height", (0.0, -1.0, 3.6001))
def test_pressure_gradient_height_must_lie_inside_domain(height: float) -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params

    with pytest.raises(ValueError, match="pressure_force_height"):
        Params(
            nx=4,
            ny=4,
            nz=8,
            lz=1.0,
            z_i=3.6,
            bl_height=2.0,
            pressure_force_height=height,
            zo=1.0e-3,
            dtype=jnp.float32,
        )


def test_log_law_initial_state_is_uniform_above_1000_m() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params, initial_state

    params = Params(
        nx=4,
        ny=4,
        nz=13,
        lz=1.5,
        z_i=1000.0,
        bl_height=1000.0,
        u_fric=0.4,
        zo=0.1,
        initial_condition="log_law",
        initial_velocity_noise=0.0,
        dtype=jnp.float32,
    )
    state = jax.block_until_ready(initial_state(params))
    profile = np.asarray(state.u).mean(axis=(0, 1))
    z = (np.arange(profile.size) + 0.5) * params.dz * params.z_i
    below = z < params.bl_height
    above = ~below
    expected_below = params.u_fric / params.vonk * np.log(z[below] / params.zo)
    expected_cap = params.u_fric / params.vonk * np.log(params.bl_height / params.zo)

    np.testing.assert_allclose(profile[below], expected_below, rtol=2.0e-6)
    np.testing.assert_allclose(profile[above], expected_cap, rtol=2.0e-6)


def test_uniform_pressure_driven_initial_state_uses_dynamic_wall_stress() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from run_pressure_driven_neutral_abl import horizontal_u_profile, velocity_cross_sections
    from wireles_jax import Params, initial_state
    from wireles_jax.wall import wall_stress

    params = Params(
        nx=8,
        ny=8,
        nz=9,
        lz=1.5,
        z_i=100.0,
        u_fric=0.4,
        zo=0.1,
        bl_height=100.0,
        initial_condition="uniform",
        wall_stress_model="dynamic_neutral",
        initial_velocity_noise=0.0,
        sgs_model="lasd",
        dtype=jnp.float32,
    )
    state = jax.block_until_ready(initial_state(params))

    plotted_profile = horizontal_u_profile(state)
    assert plotted_profile.shape == (params.nz,)
    plotted_z = (np.arange(plotted_profile.size) + 0.5) * params.dz * params.z_i
    assert params.bl_height < plotted_z[-1] < params.lz * params.z_i

    xy, xz, yz, actual_z_over_h = velocity_cross_sections(state, params, 0.1)
    assert xy.shape == (params.ny, params.nx)
    assert xz.shape == (params.nz, params.nx)
    assert yz.shape == (params.nz, params.ny)
    assert 0.0 < actual_z_over_h < 1.0

    z = (np.arange(params.nz) + 0.5) * params.dz * params.z_i
    target = params.pressure_ustar / params.vonk * np.log(z / params.zo)
    expected_u = target[((np.arange(params.nz) + 0.5) * params.dz) <= params.forcing_height].mean()
    np.testing.assert_allclose(np.asarray(state.u), expected_u, rtol=2.0e-6)
    np.testing.assert_allclose(np.asarray(state.v), 0.0)
    np.testing.assert_allclose(np.asarray(state.w), 0.0)

    txz, _, _, _, ustar = jax.block_until_ready(wall_stress(state.u, state.v, params))
    expected_ustar = expected_u * params.vonk / np.log(params.wall_ref_height / params.zo)
    np.testing.assert_allclose(np.asarray(ustar), expected_ustar, rtol=2.0e-6)
    np.testing.assert_allclose(np.asarray(txz), -(expected_ustar**2), rtol=2.0e-6)
    assert not np.isclose(expected_ustar, params.u_fric)

    # The wall stress follows the resolved first-level speed; it is not fixed
    # by the u_fric value used to scale the pressure gradient.
    _, _, _, _, half_speed_ustar = jax.block_until_ready(
        wall_stress(0.5 * state.u, state.v, params)
    )
    np.testing.assert_allclose(np.asarray(half_speed_ustar), 0.5 * expected_ustar, rtol=2.0e-6)
    np.testing.assert_allclose(params.driving_pressure_force, params.u_fric**2, rtol=2.0e-6)
    np.testing.assert_allclose(params.pressure_ustar, params.u_fric, rtol=2.0e-6)
