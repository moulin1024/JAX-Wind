from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = ROOT / "benchmark" / "Nieuwstadt1993"
sys.path.insert(0, str(ROOT / "legacy" / "jax"))
sys.path.insert(0, str(BENCHMARK_DIR))


def test_nieuwstadt_config_uses_lasd_for_momentum_and_scalar() -> None:
    from solve import load_settings

    settings = load_settings(BENCHMARK_DIR / "configs" / "lasd_scalar.toml")
    assert (settings["nx"], settings["ny"], settings["nz"]) == (40, 40, 48)
    assert settings["sgs_model"] == "lasd"
    assert settings["scalar_sgs_model"] == "lasd"
    assert settings["thermo_enabled"] is True
    assert settings["surface_theta_flux"] > 0.0
    assert settings["cs_count"] == 10
    assert settings["dt"] == pytest.approx(1.25)
    assert settings["time_scheme"] == "rk3"
    assert settings["steps"] == 9646
    assert settings["log_every"] == 96
    assert settings["scalar_vertical_scheme"] == "weno5z"
    assert settings["top_boundary_condition"] == "klemp_durran"
    assert settings["precision"] == "float32"
    assert settings["benchmark_initial_zi_fraction"] == pytest.approx(0.844)


def test_lasd_diagnostic_sgs_energy_uses_dynamic_momentum_and_heat_lengths() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from solve import (
        LASD_SGS_DISSIPATION_COEFFICIENT,
        _diagnostic_lasd_sgs_tke,
    )
    from wireles_jax import Params

    params = Params(
        nx=1,
        ny=1,
        nz=2,
        lz=1.0,
        z_i=1000.0,
        thermo_enabled=True,
        theta0=300.0,
        g=9.81,
        sgs_model="lasd",
        scalar_sgs_model="lasd",
        dtype=jnp.float32,
    )
    shape = (1, 1, params.nz)
    cs2 = np.full(shape, 0.04)
    scalar_c = np.full(shape, 0.08)
    strain = np.full(shape, 2.0)
    stability = np.full(shape, 0.5)

    neutral = _diagnostic_lasd_sgs_tke(
        cs2,
        scalar_c,
        strain,
        np.zeros(shape),
        stability,
        params,
    )
    dissipation_coefficient = LASD_SGS_DISSIPATION_COEFFICIENT
    expected_neutral = (
        0.04 * params.sgs_delta**3 * 2.0**3 / dissipation_coefficient
    ) ** (2.0 / 3.0)
    np.testing.assert_allclose(neutral, expected_neutral)

    n2_scaled = 0.5 * (0.04 / (0.08 * 0.5)) * 2.0**2
    stable_gradient = n2_scaled * params.theta_v0 / (params.z_i * params.g)
    stable = _diagnostic_lasd_sgs_tke(
        cs2,
        scalar_c,
        strain,
        np.full(shape, stable_gradient),
        stability,
        params,
    )
    np.testing.assert_allclose(stable, 0.5 ** (2.0 / 3.0) * expected_neutral)


def test_weno3_face_reconstruction_is_exact_for_linear_profile() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax.scalar import _weno3_face_value

    nz = 64
    phi = jnp.arange(nz + 2, dtype=jnp.float32)[None, None, :, None]
    positive_w = jnp.ones((1, 1, nz + 2), dtype=jnp.float32)
    negative_w = -positive_w
    expected = np.arange(nz + 2, dtype=np.float32) - 0.5
    np.testing.assert_allclose(
        np.asarray(_weno3_face_value(positive_w, phi))[0, 0, 2:-1, 0],
        expected[2:-1],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(_weno3_face_value(negative_w, phi))[0, 0, 2:-1, 0],
        expected[2:-1],
        atol=1.0e-6,
    )


def test_weno5z_face_reconstruction_is_exact_for_linear_profile() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax.scalar import _weno5z_face_value

    nz = 64
    phi = jnp.arange(nz + 2, dtype=jnp.float32)[None, None, :, None]
    positive_w = jnp.ones((1, 1, nz + 2), dtype=jnp.float32)
    negative_w = -positive_w
    expected = np.arange(nz + 2, dtype=np.float32) - 0.5
    interior = slice(4, -3)
    np.testing.assert_allclose(
        np.asarray(_weno5z_face_value(positive_w, phi))[0, 0, interior, 0],
        expected[interior],
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(_weno5z_face_value(negative_w, phi))[0, 0, interior, 0],
        expected[interior],
        atol=2.0e-6,
    )

    # The first interior face needs one virtual value below the stored ghost.
    # It must inherit the ghost-to-first-cell slope rather than repeat the
    # stored ghost, otherwise WENO5 introduces a wall-adjacent kink.
    first_interior_face = 2
    np.testing.assert_allclose(
        np.asarray(_weno5z_face_value(positive_w, phi))[0, 0, first_interior_face, 0],
        expected[first_interior_face],
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(_weno5z_face_value(negative_w, phi))[0, 0, first_interior_face, 0],
        expected[first_interior_face],
        atol=2.0e-6,
    )


def test_weno5z_float32_weights_remain_finite_for_large_flux_ghost() -> None:
    """A weak local diffusivity must not overflow normalized WENO-Z weights."""
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax.scalar import _weno5z_face_value

    nz = 8
    phi = jnp.full((1, 1, nz + 2, 1), 300.0, dtype=jnp.float32)
    # Large enough that an unscaled float32 Jiang--Shu indicator overflows.
    phi = phi.at[:, :, 0, :].set(5.0e24)
    w = jnp.ones((1, 1, nz + 2), dtype=jnp.float32)

    reconstructed = np.asarray(_weno5z_face_value(w, phi))
    assert np.all(np.isfinite(reconstructed))
    np.testing.assert_allclose(reconstructed[0, 0, 2:4, 0], 300.0, atol=2.0e-4)


def test_bounded_weno_z_weights_match_standard_formula() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax.scalar import _weno_z_weights

    beta = np.asarray([[0.2, 0.05, 0.4], [1.0e-3, 0.3, 0.02]], dtype=np.float32)
    epsilon = np.finfo(np.float32).eps
    tau = np.abs(beta[:, 0] - beta[:, 2])
    linear = np.asarray([0.1, 0.6, 0.3], dtype=np.float32)
    expected_alpha = linear[None, :] * (1.0 + (tau[:, None] / (beta + epsilon)) ** 2)
    expected = expected_alpha / expected_alpha.sum(axis=1, keepdims=True)

    actual = np.stack(
        tuple(np.asarray(weight) for weight in _weno_z_weights(*(jnp.asarray(beta[:, i]) for i in range(3)))),
        axis=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=3.0e-7)


def test_zero_speed_wall_stress_has_no_hidden_float32_nan() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.wall import wall_stress

    params = Params(
        nx=8,
        ny=8,
        nz=8,
        momentum_wall_model="abl",
        dtype=jnp.float32,
    )
    zero = jnp.zeros((params.nx, params.ny, params.nz), dtype=params.dtype)

    with jax.debug_nans(True):
        result = jax.jit(lambda u, v: wall_stress(u, v, params))(zero, zero)
        result = jax.block_until_ready(result)
    for value in result:
        np.testing.assert_array_equal(np.asarray(value), 0.0)


def test_theta_surface_ghost_matches_prescribed_flux() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.scalar import _apply_theta_surface_flux_ghost

    params = Params(
        nx=2,
        ny=2,
        nz=6,
        lz=1.0,
        z_i=1000.0,
        zo=0.01,
        thermo_enabled=True,
        theta_bc="flux",
        surface_theta_flux=0.06,
        dtype=jnp.float32,
    )
    theta = jnp.full((params.nx, params.ny, params.nz), 300.0, dtype=params.dtype)
    kappa = jnp.zeros_like(theta).at[:, :, 0].set(
        jnp.asarray([[0.02, 0.03], [0.04, 0.05]], dtype=params.dtype)
    )

    theta_ghost = _apply_theta_surface_flux_ghost(theta, kappa, params)
    reconstructed_flux = -kappa[:, :, 0] * (theta[:, :, 0] - theta_ghost) / params.dz
    np.testing.assert_allclose(
        np.asarray(reconstructed_flux),
        params.surface_theta_flux,
        rtol=7.0e-5,
        atol=5.0e-6,
    )
    assert np.all(np.asarray(theta_ghost) > np.asarray(theta[:, :, 0]))


def test_theta_surface_ghost_remains_finite_for_zero_diffusivity() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.scalar import _apply_theta_surface_flux_ghost

    params = Params(
        nx=2,
        ny=2,
        nz=6,
        lz=1.0,
        z_i=1000.0,
        zo=0.01,
        thermo_enabled=True,
        theta_bc="flux",
        surface_theta_flux=0.06,
        dtype=jnp.float32,
    )
    theta = jnp.full((params.nx, params.ny, params.nz), 300.0, dtype=params.dtype)
    theta_ghost = _apply_theta_surface_flux_ghost(theta, jnp.zeros_like(theta), params)
    np.testing.assert_allclose(np.asarray(theta_ghost), np.asarray(theta[:, :, 0]))
    assert np.all(np.isfinite(np.asarray(theta_ghost)))


def test_dirichlet_theta_uses_virtual_face_symmetric_ghost() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.scalar import _apply_theta_surface_flux_ghost

    params = Params(
        nx=2,
        ny=2,
        nz=4,
        theta_bc="dirichlet",
        theta_bottom=299.0,
        theta_top=305.0,
        thermo_enabled=True,
        dtype=jnp.float32,
    )
    theta = jnp.full((params.nx, params.ny, params.nz), 301.0, dtype=params.dtype)
    ghost = _apply_theta_surface_flux_ghost(theta, jnp.ones_like(theta), params)
    np.testing.assert_allclose(
        0.5 * (np.asarray(ghost) + np.asarray(theta[:, :, 0])),
        params.theta_bottom,
    )


def test_scalar_center_gradient_uses_only_physical_cells() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.scalar import _scalar_center_dz

    params = Params(nx=1, ny=1, nz=5, lz=1.0, dtype=jnp.float32)
    q = jnp.asarray([[[10.0, 12.0, 15.0, 19.0, 24.0]]])
    dz = params.dz
    derivative = np.asarray(_scalar_center_dz(q, params))[0, 0]
    assert derivative[0] == pytest.approx((12.0 - 10.0) / dz)
    assert derivative[1] == pytest.approx((15.0 - 10.0) / (2.0 * dz))
    assert derivative[-1] == pytest.approx((24.0 - 19.0) / dz)


@pytest.mark.parametrize("ny", [15, 16])
def test_radial_spectrum_obeys_parseval_for_rfft_storage(ny: int) -> None:
    from solve import radial_spectrum

    nx = 18
    dx = 120.0
    dy = 90.0
    zi0 = 1600.0
    rng = np.random.default_rng(829)
    field = rng.standard_normal((nx, ny))
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(ny, d=dy)
    maximum_kzi = np.max(
        np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2) * zi0
    )
    edges = np.linspace(0.0, np.nextafter(maximum_kzi, np.inf), 41)

    spectrum = radial_spectrum(field, dx, dy, zi0, edges)
    spectral_variance = np.sum(spectrum * np.diff(edges))
    physical_variance = np.mean((field - np.mean(field)) ** 2)
    assert spectral_variance == pytest.approx(
        physical_variance,
        rel=2.0e-14,
        abs=2.0e-14,
    )


def test_diagnostic_momentum_gradient_uses_wall_model_at_first_center() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from solve import _momentum_vertical_gradients_np

    params = Params(
        nx=16,
        ny=16,
        nz=4,
        lz=1.0,
        z_i=1600.0,
        zo=0.16,
        u_fric=0.2,
        wall_stress_model="prescribed_ustar",
        momentum_wall_model="abl",
        dtype=jnp.float32,
    )
    u = np.full((params.nx, params.ny, params.nz), 2.0)
    v = np.zeros_like(u)
    dudz, dvdz, dudz_face, dvdz_face = _momentum_vertical_gradients_np(
        u, v, params
    )
    expected_wall = params.u_fric / (
        params.vonk * 0.5 * params.dz
    )
    np.testing.assert_allclose(dudz[:, :, 0], expected_wall)
    np.testing.assert_allclose(dudz[:, :, 1:], 0.0)
    np.testing.assert_allclose(dvdz, 0.0)
    np.testing.assert_allclose(dudz_face, 0.0)
    np.testing.assert_allclose(dvdz_face, 0.0)


def test_lasd_departure_point_uses_one_trilinear_interpolation() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.sgs import _lagrangian_interp

    params = Params(
        nx=4,
        ny=4,
        nz=4,
        lx=4.0,
        ly=4.0,
        lz=3.0,
        dt=1.0,
        cs_count=1,
        momentum_wall_model="free_slip",
        dtype=jnp.float32,
    )
    shape = (params.nx, params.ny, params.nz)
    i, j, k = np.indices(shape)
    q = jnp.asarray(i + 10.0 * j + 100.0 * k, dtype=params.dtype)
    u = jnp.full(shape, 0.25, dtype=params.dtype)
    u = u.at[:, 0, :].set(0.75)
    v = jnp.full(shape, 0.5, dtype=params.dtype)
    w = jnp.full(shape, 0.25, dtype=params.dtype)

    result = np.asarray(_lagrangian_interp(q, u, v, w, params))
    expected = 211.0 - 0.25 / params.dx - 10.0 * 0.5 / params.dy - 100.0 * 0.25 / params.dz
    assert result[1, 1, 2] == pytest.approx(expected, abs=2.0e-5)


def test_lasd_history_boundary_centers_copy_nearest_interior() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax.scalar import _history_bc as scalar_history_bc
    from wireles_jax.sgs import _history_bc as momentum_history_bc

    history = jnp.arange(5.0)[None, None, :, None]
    expected = np.asarray([1.0, 1.0, 2.0, 3.0, 3.0])
    np.testing.assert_array_equal(
        np.asarray(momentum_history_bc(history))[0, 0, :, 0], expected
    )
    np.testing.assert_array_equal(
        np.asarray(scalar_history_bc(history))[0, 0, :, 0], expected
    )


def test_lasd_volume_filter_uses_original_fortran_nint_cutoff() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.scalar import _spectral_box_filter_concat as scalar_filter
    from wireles_jax.sgs import _spectral_box_filter_concat as momentum_filter

    params = Params(nx=32, ny=32, nz=1, lx=32.0, ly=32.0, lz=1.0)
    x = jnp.arange(params.nx, dtype=jnp.float32)[:, None, None, None]
    retained = jnp.sin(2.0 * jnp.pi * x / params.nx)
    retained2 = jnp.sin(2.0 * jnp.pi * 2.0 * x / params.nx)
    removed3 = jnp.sin(2.0 * jnp.pi * 3.0 * x / params.nx)
    field = jnp.broadcast_to(retained + retained2 + removed3, (32, 32, 1, 1))

    # At R=6, NINT(32/(2R))=3 and the strict Fortran cutoff keeps modes
    # |k|<3.  The separate wall-plane filter intentionally uses FLOOR.
    for filter_fn in (momentum_filter, scalar_filter):
        filtered = np.asarray(filter_fn(field, params, 6.0))[:, 0, 0, 0]
        np.testing.assert_allclose(
            filtered,
            np.asarray(retained + retained2)[:, 0, 0, 0],
            atol=2.0e-6,
        )


def test_wall_filter_keeps_original_fortran_floor_cutoff() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.wall import filter_2d_wall

    # nx/(2*fgr*tfr)=34/6=5.67: FLOOR removes mode 5, whereas NINT would
    # retain it.  This deliberately differs from the volume LASD filters.
    params = Params(nx=34, ny=34, nz=1, lx=34.0, ly=34.0, lz=1.0)
    x = jnp.arange(params.nx, dtype=jnp.float32)[:, None]
    mode4 = jnp.sin(2.0 * jnp.pi * 4.0 * x / params.nx)
    mode5 = jnp.sin(2.0 * jnp.pi * 5.0 * x / params.nx)
    field = jnp.broadcast_to(mode4 + mode5, (34, 34))
    filtered = np.asarray(filter_2d_wall(field, params))[:, 0]
    np.testing.assert_allclose(filtered, np.asarray(mode4)[:, 0], atol=2.0e-6)


def test_lasd_invalid_scale_ratio_fallback_only_changes_undefined_beta() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.sgs import _scale_dependence_beta

    c2 = jnp.asarray([0.0, 0.04, 0.04, 0.04], dtype=jnp.float32)
    c4 = jnp.asarray([0.0, 0.004, 0.01, 0.16], dtype=jnp.float32)
    legacy = _scale_dependence_beta(c2, c4, Params(), True)
    fallback = _scale_dependence_beta(
        c2,
        c4,
        Params(lasd_invalid_beta_fallback=True),
        True,
    )
    clipped = _scale_dependence_beta(
        c2,
        c4,
        Params(lasd_clipped_beta_fallback=True),
        True,
    )
    disabled = _scale_dependence_beta(c2, c4, Params(), False)

    np.testing.assert_allclose(np.asarray(legacy), [0.125, 0.125, 0.25, 4.0])
    np.testing.assert_allclose(np.asarray(fallback), [1.0, 0.125, 0.25, 4.0])
    np.testing.assert_allclose(np.asarray(clipped), [1.0, 1.0, 0.25, 4.0])
    np.testing.assert_allclose(np.asarray(disabled), 1.0)


def test_porte_agel_polynomial_selects_largest_positive_real_root() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax.scalar import _largest_positive_real_polynomial_root

    # Roots are 0.5, 0.8, -1, and the complex pair +/-i.
    descending = np.poly([0.5, 0.8, -1.0, 1.0j, -1.0j]).real
    ascending = jnp.asarray(descending[::-1], dtype=jnp.float32)
    root = _largest_positive_real_polynomial_root(ascending)
    np.testing.assert_allclose(float(root), 0.8, rtol=2.0e-5)

    # A vanishing fifth-order coefficient is a valid lower-order equation,
    # not a reason to discard the dynamic solution.  This quadratic has
    # roots -0.2 and 0.7.
    quadratic = jnp.asarray([-0.14, -0.5, 1.0, 0.0, 0.0, 0.0])
    quadratic_root = _largest_positive_real_polynomial_root(quadratic)
    np.testing.assert_allclose(float(quadratic_root), 0.7, rtol=2.0e-5)


def test_porte_agel_polynomial_matches_two_germano_quotients() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax.scalar import _porte_agel_polynomial

    contractions = tuple(
        jnp.asarray(value, dtype=jnp.float32)
        for value in (0.7, -0.13, 1.2, 0.17, -0.21, 0.41, 0.09, 0.83, 0.12, 0.16)
    )
    p, q, r, s, t, p2, q2, r2, s2, t2 = contractions
    coefficients = _porte_agel_polynomial(*contractions)

    for beta in (0.2, 0.8, 1.4):
        n1 = p - 4.0 * beta * q
        d1 = r - 8.0 * beta * t + 16.0 * beta**2 * s
        n2 = p2 - 16.0 * beta**2 * q2
        d2 = r2 - 32.0 * beta**2 * t2 + 256.0 * beta**4 * s2
        direct = n1 * d2 - n2 * d1
        expanded = sum(coefficients[i] * beta**i for i in range(6))
        np.testing.assert_allclose(float(expanded), float(direct), rtol=2.0e-5)


def test_porte_agel_momentum_coefficient_is_plane_uniform_and_positive() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.sgs import _porte_agel_plane_cs2

    params = Params(
        nx=32,
        ny=32,
        nz=4,
        lx=3200.0,
        ly=3200.0,
        lz=160.0,
        sgs_model="porte_agel_sd",
        scalar_sgs_model="fixed_prandtl",
    )
    key = jax.random.PRNGKey(7)
    velocity = jax.random.normal(key, (32, 32, 4, 3), dtype=jnp.float32)
    vel = velocity
    uu = jnp.stack(
        (
            vel[..., 0] ** 2,
            vel[..., 0] * vel[..., 1],
            vel[..., 0] * vel[..., 2],
            vel[..., 1] ** 2,
            vel[..., 1] * vel[..., 2],
            vel[..., 2] ** 2,
        ),
        axis=-1,
    )
    sij = 0.1 * jax.random.normal(
        jax.random.PRNGKey(8), (32, 32, 4, 6), dtype=jnp.float32
    )
    cs2, beta, valid = _porte_agel_plane_cs2(vel, uu, sij, params)

    assert cs2.shape == (32, 32, 4)
    assert np.all(np.isfinite(np.asarray(cs2)))
    assert np.all(np.asarray(cs2) > 0.0)
    for field in (cs2, beta, valid):
        actual = np.asarray(field)
        expected = np.broadcast_to(actual[:1, :1, :], actual.shape)
        np.testing.assert_allclose(actual, expected)


def test_scalar_lasd_test_strain_filters_tensor_before_magnitude() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.scalar import _scalar_lm_mm
    from wireles_jax.sgs import _strain_magnitude

    params = Params(
        nx=32,
        ny=32,
        nz=2,
        lx=32.0,
        ly=32.0,
        lz=2.0,
        momentum_wall_model="free_slip",
        thermo_enabled=True,
        sgs_model="lasd",
        scalar_sgs_model="lasd",
        dtype=jnp.float32,
    )
    shape = (params.nx, params.ny, params.nz)
    x = jnp.arange(params.nx, dtype=params.dtype)[:, None, None]
    sij = jnp.zeros(shape + (6,), dtype=params.dtype)
    sij = sij.at[..., 0].set(jnp.cos(2.0 * jnp.pi * 8.0 * x / params.nx))
    strain_mag = _strain_magnitude(sij)
    vel = jnp.zeros(shape + (3,), dtype=params.dtype)
    phi = jnp.zeros(shape + (2,), dtype=params.dtype)
    grad_phi = jnp.zeros(shape + (3, 2), dtype=params.dtype)
    grad_phi = grad_phi.at[..., 0, :].set(1.0)

    _, mm = _scalar_lm_mm(
        vel,
        phi,
        grad_phi,
        sij,
        strain_mag,
        params,
        params.tfr,
    )
    expected = 0.5 * params.sgs_delta**4
    np.testing.assert_allclose(
        np.asarray(mm),
        expected,
        rtol=2.0e-5,
        atol=1.0e-7,
    )


def test_scalar_lasd_relaxation_is_invariant_to_scalar_units() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.scalar import _lagrangian_average

    params = Params(
        nx=2,
        ny=2,
        nz=2,
        lx=2.0,
        ly=2.0,
        lz=2.0,
        dt=1.0,
        cs_count=1,
        momentum_wall_model="free_slip",
        dtype=jnp.float32,
    )
    shape = (params.nx, params.ny, params.nz, 2)
    zero_velocity = jnp.zeros(shape[:-1], dtype=params.dtype)
    one = jnp.ones(shape, dtype=params.dtype)
    momentum_a = jnp.full(shape, 0.03, dtype=params.dtype)
    momentum_b = one

    def averaged(scale: float) -> tuple[np.ndarray, np.ndarray]:
        avg_a, avg_b = _lagrangian_average(
            4.0 * scale * one,
            2.0 * scale * one,
            scale * one,
            scale * one,
            zero_velocity,
            zero_velocity,
            zero_velocity,
            params,
            timescale_a=momentum_a,
            timescale_b=momentum_b,
            ramp_numerator=True,
        )
        return np.asarray(avg_a) / scale, np.asarray(avg_b) / scale

    base_a, base_b = averaged(1.0)
    scaled_a, scaled_b = averaged(1.0e6)
    np.testing.assert_allclose(scaled_a, base_a, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(scaled_b, base_b, rtol=2.0e-6, atol=2.0e-6)


def test_scalar_lasd_negative_numerator_ramp_recovers() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.scalar import _lagrangian_average

    params = Params(
        nx=2,
        ny=2,
        nz=2,
        lx=2.0,
        ly=2.0,
        lz=2.0,
        dt=1.0,
        cs_count=1,
        momentum_wall_model="free_slip",
        dtype=jnp.float32,
    )
    shape = (params.nx, params.ny, params.nz, 2)
    zero_velocity = jnp.zeros(shape[:-1], dtype=params.dtype)
    one = jnp.ones(shape, dtype=params.dtype)
    kwargs = {
        "timescale_a": one,
        "timescale_b": one,
        "ramp_numerator": True,
    }
    ramped_a, first_b = _lagrangian_average(
        -1.0e6 * one,
        one,
        one,
        one,
        zero_velocity,
        zero_velocity,
        zero_velocity,
        params,
        **kwargs,
    )
    assert np.all(np.asarray(ramped_a) > 0.0)

    recovered_a, _ = _lagrangian_average(
        1.0e6 * one,
        one,
        ramped_a,
        first_b,
        zero_velocity,
        zero_velocity,
        zero_velocity,
        params,
        **kwargs,
    )
    assert np.min(np.asarray(recovered_a)) > 1.0e-32


def test_lasd_first_update_uses_exactly_cs_count_velocity_samples() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params, initial_state
    from wireles_jax.sgs import _update_lasd_coefficients

    params = Params(
        nx=4,
        ny=4,
        nz=4,
        lx=4.0,
        ly=4.0,
        lz=3.0,
        dt=0.1,
        cs_count=4,
        momentum_wall_model="free_slip",
        sgs_model="lasd",
        dtype=jnp.float32,
    )
    state = initial_state(params, seed=1)
    u = jnp.full_like(state.u, 2.0)
    zero = jnp.zeros_like(state.u)
    sij = jnp.zeros(state.u.shape + (6,), dtype=params.sgs_dtype)

    for step_index in range(params.cs_count):
        state = state._replace(step=jnp.asarray(step_index, dtype=state.step.dtype))
        _, lasd_state = _update_lasd_coefficients(
            state,
            u,
            zero,
            zero,
            sij,
            params,
            update=True,
        )
        state = state._replace(
            lm_old=lasd_state[0],
            mm_old=lasd_state[1],
            qn_old=lasd_state[2],
            nn_old=lasd_state[3],
            u_lag=lasd_state[4],
            v_lag=lasd_state[5],
            w_lag=lasd_state[6],
        )
        if step_index < params.cs_count - 1:
            expected = 2.0 * (step_index + 1) / params.cs_count
            np.testing.assert_allclose(np.asarray(state.u_lag), expected)

    np.testing.assert_array_equal(np.asarray(state.u_lag), 0.0)


def test_sharded_lasd_uses_same_single_departure_point_interpolation() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.timestep_sharded import _lagrangian_interp_halo

    params = Params(
        nx=4,
        ny=4,
        nz=4,
        lx=4.0,
        ly=4.0,
        lz=3.0,
        dt=1.0,
        cs_count=1,
        momentum_wall_model="free_slip",
        dtype=jnp.float32,
    )
    shape = (params.nx, params.ny, params.nz + 2)
    i, j, k = np.indices(shape)
    q = jnp.asarray(i + 10.0 * j + 100.0 * k, dtype=params.dtype)
    u = jnp.full(shape, 0.25, dtype=params.dtype).at[:, 0, :].set(0.75)
    v = jnp.full(shape, 0.5, dtype=params.dtype)
    w = jnp.full(shape, 0.25, dtype=params.dtype)
    result = np.asarray(_lagrangian_interp_halo(q, u, v, w, params))
    expected = 211.0 - 0.25 / params.dx - 10.0 * 0.5 / params.dy - 100.0 * 0.25 / params.dz
    assert result[1, 1, 2] == pytest.approx(expected, abs=2.0e-5)


def test_diagnostic_scalar_stability_uses_center_gradient_semantics() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from solve import _diagnostic_scalar_stability
    from wireles_jax import Params

    params = Params(
        nx=1,
        ny=1,
        nz=4,
        lz=1.0,
        z_i=1000.0,
        thermo_enabled=True,
        sgs_model="lasd",
        scalar_sgs_model="lasd",
        scalar_stability_correction=True,
        scalar_stability_beta=30.0,
        scalar_stability_power=2.0,
        dtype=jnp.float32,
    )
    strain = np.full((1, 1, params.nz), 2.0)
    center_gradient = np.full_like(strain, 3.0e-3)
    stability = _diagnostic_scalar_stability(strain, center_gradient, params)
    n2 = params.z_i * params.g * 3.0e-3 / params.theta_v0
    expected = (1.0 + params.scalar_stability_beta * n2 / 4.0) ** -2.0
    np.testing.assert_allclose(stability, expected)


def test_overall_and_lasd_cfl_limits() -> None:
    pytest.importorskip("jax.numpy")
    from wireles_jax import Params
    from wireles_jax.diagnostics import lasd_cfl_number, validate_cfl, validate_lasd_cfl
    from wireles_jax.state import Diagnostics

    params = Params(sgs_model="lasd", cs_count=10)

    def make_diag(cfl: float) -> Diagnostics:
        return Diagnostics(
            step=8,
            ustar=0.0,
            ke_max=0.0,
            div_max=0.0,
            cfl_x=cfl,
            cfl_y=0.5 * cfl,
            cfl_z=0.25 * cfl,
        )

    below = make_diag(0.099)
    assert validate_cfl(below) == pytest.approx(0.099)
    assert lasd_cfl_number(below, params) == pytest.approx(0.99)
    assert validate_lasd_cfl(below, params) == pytest.approx(0.99)

    at_lasd_limit = make_diag(0.1)
    assert validate_cfl(at_lasd_limit) == pytest.approx(0.1)
    with pytest.warns(RuntimeWarning, match=r"cs_count \* max\(CFL\) = 1\.000000 >= 1\.0"):
        validate_lasd_cfl(at_lasd_limit, params)

    above_overall_limit = make_diag(0.100001)
    with pytest.warns(RuntimeWarning, match=r"max\(CFL_x, CFL_y, CFL_z\).*= 0\.100001 > 0\.100000"):
        validate_cfl(above_overall_limit)


def test_lasd_scalar_transport_updates_coefficient_and_drives_buoyancy() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from initial_condition import initial_benchmark_state
    from wireles_jax import Params, step
    from wireles_jax.grid import make_operators
    from wireles_jax.scalar import buoyancy_from_theta_qv

    params = Params(
        nx=8,
        ny=8,
        nz=10,
        lx=4.0,
        ly=4.0,
        lz=1.5,
        z_i=1600.0,
        bl_height=1600.0,
        zo=0.16,
        dt=1.0 / 1600.0,
        cs_count=2,
        initial_condition="default",
        initial_velocity_noise=0.0,
        pressure_force=0.0,
        u_fric=0.0,
        sgs_model="lasd",
        sgs_delta_scale=1.0,
        thermo_enabled=True,
        theta_initial_gradient=0.003,
        theta_top_gradient=0.003,
        surface_theta_flux=0.06,
        scalar_sgs_model="lasd",
        scalar_stability_correction=True,
        scalar_stability_beta=30.0,
        scalar_stability_power=2.0,
        dtype=jnp.float32,
        sgs_dtype=jnp.float32,
    )
    state = initial_benchmark_state(params, seed=9)
    initial_scalar_c = np.asarray(state.scalar_c)
    initial_theta = np.asarray(state.theta)
    initial_buoyancy = np.asarray(
        jax.block_until_ready(buoyancy_from_theta_qv(state.theta, state.qv, params))
    )
    assert np.max(np.abs(initial_buoyancy)) > 0.0

    ops = make_operators(params)
    step_fn = jax.jit(lambda s: step(s, params, ops))
    for _ in range(4):
        state = step_fn(state)
    state = jax.block_until_ready(state)

    scalar_c = np.asarray(state.scalar_c[..., 0])
    assert np.all(np.isfinite(scalar_c))
    assert np.min(scalar_c) >= params.scalar_lasd_min
    assert np.max(scalar_c) <= params.scalar_lasd_max
    assert not np.allclose(np.asarray(state.scalar_c), initial_scalar_c)
    assert np.max(np.abs(np.asarray(state.theta) - initial_theta)) > 1.0e-5
    assert np.all(np.isfinite(np.asarray(state.w)))


def test_scalar_advection_flux_form_is_globally_conservative() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.grid import make_operators
    from wireles_jax.scalar import _scalar_advection_divergence

    params = Params(
        nx=8,
        ny=6,
        nz=9,
        lx=4.0,
        ly=3.0,
        lz=1.0,
        thermo_enabled=True,
        horizontal_dealias=True,
        dtype=jnp.float32,
    )
    key = jax.random.PRNGKey(31)
    key_u, key_v, key_w, key_phi = jax.random.split(key, 4)
    shape = (params.nx, params.ny, params.nz)
    u = jax.random.normal(key_u, shape)
    v = jax.random.normal(key_v, shape)
    w = jax.random.normal(key_w, shape)
    w = w.at[:, :, -1].set(0.0)
    phi = jax.random.normal(key_phi, shape + (2,))

    divergence = jax.block_until_ready(
        _scalar_advection_divergence(u, v, w, phi, params, make_operators(params))
    )
    global_sum = np.asarray(divergence).sum(axis=(0, 1, 2))
    np.testing.assert_allclose(global_sum, 0.0, atol=2.0e-4)
