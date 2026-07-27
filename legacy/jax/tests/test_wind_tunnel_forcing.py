from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wireles_jax.config import Params
from wireles_jax.scalar import buoyancy_from_theta_qv
from wireles_jax.wind_tunnel import (
    actuator_disk_kernel,
    classic_fringe_window,
    cold_source_kernel,
    fringe_mask,
    wind_tunnel_momentum_sources,
    wind_tunnel_scalar_sources,
)
from wireles_jax.wall import apply_porte_agel_wall_correction


def _base_params(**overrides) -> Params:
    values = dict(
        nx=32,
        ny=24,
        nz=24,
        lx=8.0,
        ly=4.0,
        lz=4.0,
        z_i=1.0,
        momentum_wall_model="free_slip",
        initial_condition="wind_tunnel",
        uniform_u=8.0,
        pressure_force=0.0,
        thermo_enabled=True,
        moisture_enabled=False,
        theta0=293.15,
        dtype=jnp.float32,
        sgs_dtype=jnp.float32,
    )
    values.update(overrides)
    return Params(**values)


def test_classic_fringe_window_is_smooth_at_the_periodic_seam() -> None:
    x = jnp.asarray([5.9, 6.0, 6.5, 7.0, 7.5, 8.0, 8.1], dtype=jnp.float32)
    window = np.asarray(classic_fringe_window(x, 6.0, 8.0))

    np.testing.assert_array_equal(window[[0, 1, 5, 6]], 0.0)
    assert float(window[3]) == pytest.approx(1.0)
    assert float(window[2]) == pytest.approx(float(window[4]), abs=1.0e-7)

    params = _base_params(
        nx=32,
        lx=8.0,
        fringe_enabled=True,
        fringe_start_x=6.0,
        fringe_timescale=1.0,
    )
    mask = np.asarray(fringe_mask(params))[:, 0, 0]
    assert np.all(mask[:24] == 0.0)
    assert mask[24] == pytest.approx(mask[-1], abs=1.0e-7)
    assert mask[24] < mask[27]
    assert mask[-1] < mask[-4]


def test_cold_source_preserves_specified_integral_budgets() -> None:
    params = _base_params(
        cold_source_enabled=True,
        cold_source_x=2.2,
        cold_source_y=2.0,
        cold_source_z=2.0,
        cold_source_sigma_x=0.15,
        cold_source_sigma_r=0.12,
        cold_source_momentum_flux=1.7,
        cold_source_cooling_power=850.0,
        cold_source_density=1.18,
        cold_source_heat_capacity=1005.0,
    )
    shape = (params.nx, params.ny, params.nz)
    u = jnp.full(shape, params.uniform_u, dtype=params.dtype)
    zeros = jnp.zeros(shape, dtype=params.dtype)
    theta = jnp.full(shape, params.theta0, dtype=params.dtype)
    cell_volume = params.dx * params.dy * params.dz * params.z_i**3

    kernel_integral = jnp.sum(cold_source_kernel(params)) * cell_volume
    source_u, _, _ = wind_tunnel_momentum_sources(u, zeros, zeros, params)
    source_theta, _ = wind_tunnel_scalar_sources(theta, zeros, params)
    integrated_force = (
        params.cold_source_density
        * jnp.sum(source_u / params.z_i)
        * cell_volume
    )
    integrated_cooling = (
        -params.cold_source_density
        * params.cold_source_heat_capacity
        * jnp.sum(source_theta / params.z_i)
        * cell_volume
    )

    assert float(kernel_integral) == pytest.approx(1.0, rel=5.0e-6)
    assert float(integrated_force) == pytest.approx(
        params.cold_source_momentum_flux, rel=5.0e-6
    )
    assert float(integrated_cooling) == pytest.approx(
        params.cold_source_cooling_power, rel=5.0e-6
    )


def test_actuator_disk_force_uses_local_disk_velocity_and_loaded_area() -> None:
    params = _base_params(
        actuator_disk_enabled=True,
        actuator_disk_x=2.0,
        actuator_disk_y=2.0,
        actuator_disk_z=2.0,
        actuator_disk_diameter=1.0,
        actuator_disk_hub_diameter=0.12,
        actuator_disk_ct_prime=4.0 / 3.0,
        actuator_disk_thickness=0.10,
    )
    shape = (params.nx, params.ny, params.nz)
    u = jnp.full(shape, params.uniform_u, dtype=params.dtype)
    zeros = jnp.zeros(shape, dtype=params.dtype)
    density = 1.2
    cell_volume = params.dx * params.dy * params.dz * params.z_i**3
    loaded_area = jnp.sum(actuator_disk_kernel(params)) * cell_volume

    source_u, source_v, source_w = wind_tunnel_momentum_sources(u, zeros, zeros, params)
    integrated_force = density * jnp.sum(source_u / params.z_i) * cell_volume
    expected_force = (
        -0.5
        * density
        * params.actuator_disk_ct_prime
        * params.uniform_u**2
        * loaded_area
    )
    assert float(integrated_force) == pytest.approx(float(expected_force), rel=5.0e-6)
    assert float(jnp.max(jnp.abs(source_v))) == 0.0
    assert float(jnp.max(jnp.abs(source_w))) == 0.0


@pytest.mark.parametrize("yaw_degrees", (10.0, 20.0, 30.0))
def test_yawed_actuator_disk_applies_only_rotor_normal_thrust(
    yaw_degrees: float,
) -> None:
    params = _base_params(
        actuator_disk_enabled=True,
        actuator_disk_x=2.0,
        actuator_disk_y=2.0,
        actuator_disk_z=2.0,
        actuator_disk_diameter=1.0,
        actuator_disk_ct_prime=1.2,
        actuator_disk_thickness=0.10,
        actuator_disk_yaw_degrees=yaw_degrees,
    )
    shape = (params.nx, params.ny, params.nz)
    u = jnp.full(shape, params.uniform_u, dtype=params.dtype)
    zeros = jnp.zeros(shape, dtype=params.dtype)

    source_u, source_v, source_w = wind_tunnel_momentum_sources(
        u, zeros, zeros, params
    )
    integrated_u = float(jnp.sum(source_u))
    integrated_v = float(jnp.sum(source_v))

    assert integrated_u < 0.0
    assert integrated_v / integrated_u == pytest.approx(
        np.tan(np.deg2rad(yaw_degrees)), rel=2.0e-6
    )
    assert float(jnp.max(jnp.abs(source_w))) == 0.0


def test_yawed_actuator_disk_kernel_rotates_with_the_rotor() -> None:
    base = dict(
        actuator_disk_enabled=True,
        actuator_disk_x=2.0,
        actuator_disk_y=2.0,
        actuator_disk_z=2.0,
        actuator_disk_diameter=1.0,
        actuator_disk_thickness=0.10,
    )
    zero_yaw = _base_params(**base, actuator_disk_yaw_degrees=0.0)
    yawed = _base_params(**base, actuator_disk_yaw_degrees=30.0)
    kernel_zero = np.asarray(actuator_disk_kernel(zero_yaw))
    kernel_yawed = np.asarray(actuator_disk_kernel(yawed))

    assert kernel_zero.shape == kernel_yawed.shape
    assert not np.allclose(kernel_zero, kernel_yawed)
    assert np.all(np.isfinite(kernel_yawed))


def test_ambient_buoyancy_retains_net_cold_anomaly() -> None:
    ambient = _base_params(buoyancy_reference="ambient")
    plane_mean = _base_params(buoyancy_reference="plane_mean")
    shape = (ambient.nx, ambient.ny, ambient.nz)
    theta = jnp.full(shape, ambient.theta0, dtype=ambient.dtype)
    theta = theta.at[ambient.nx // 2, ambient.ny // 2, ambient.nz // 2].add(-1.0)
    qv = jnp.zeros_like(theta)

    buoyancy_ambient = buoyancy_from_theta_qv(theta, qv, ambient)
    buoyancy_plane = buoyancy_from_theta_qv(theta, qv, plane_mean)

    assert float(jnp.sum(buoyancy_ambient)) < 0.0
    assert abs(float(jnp.sum(buoyancy_plane))) < 2.0e-7


def test_heterogeneous_flow_rejects_plane_averaged_closures() -> None:
    with pytest.raises(ValueError, match="plane-mean buoyancy"):
        _base_params(horizontal_homogeneous=False, buoyancy_reference="plane_mean")
    with pytest.raises(ValueError, match="momentum SGS"):
        _base_params(
            horizontal_homogeneous=False,
            buoyancy_reference="ambient",
            sgs_model="porte_agel_sd",
            scalar_sgs_model="fixed_prandtl",
        )


def test_heterogeneous_wall_gradient_correction_is_local() -> None:
    dudz = jnp.asarray([[[1.0]], [[3.0]]], dtype=jnp.float32)
    dvdz = jnp.zeros_like(dudz)
    corrected, _ = apply_porte_agel_wall_correction(
        dudz, dvdz, correction_index=0, horizontal_average=False
    )
    expected_factor = 1.0 / jnp.log(jnp.asarray(3.0, dtype=dudz.dtype))
    np.testing.assert_allclose(
        np.asarray(corrected[:, :, 0]),
        np.asarray(dudz[:, :, 0] * expected_factor),
        rtol=1.0e-6,
    )
