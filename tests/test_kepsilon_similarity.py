"""Independent analytic and tensor checks before any profile assessment."""

import numpy as np
import pytest

from tools.kepsilon_similarity import (
    PopeConstants,
    normal_strain_production,
    similarity_fields,
    similarity_rhs,
)


def exact_mean(eta, a=23.0):
    f = (1 + a * eta**2) ** -2
    integral = eta**2 / (2 * (1 + a * eta**2))
    fp = -4 * a * eta * (1 + a * eta**2) ** -3
    return integral, f, fp


def test_constant_viscosity_exact_round_jet_and_continuity():
    r = np.r_[0.0, np.geomspace(1e-8, 2, 301)]
    a = 23.0
    integral, f, fp = exact_mean(r, a)
    n = 1 / (8 * a)
    k = np.full_like(r, 0.08)
    e = 0.09 * k**2 / n
    state = np.array([integral, f, k, r * 0, e, r * 0])
    rhs = similarity_rhs(r, state)
    np.testing.assert_allclose(rhs[1], fp, rtol=4e-15, atol=1e-15)
    viscosity, g, _ = similarity_fields(r, state)
    np.testing.assert_allclose(viscosity, n, rtol=2e-15)
    # Analytic derivative of r G = r^2 F - I, including its axis limit.
    radial_divergence = f + r * fp
    axial_derivative = -f - r * fp
    np.testing.assert_allclose(radial_divergence + axial_derivative, 0, atol=1e-15)
    np.testing.assert_allclose(g, r * f - r / (2 * (1 + a * r**2)))


@pytest.mark.parametrize("c3", [0.0, 0.79])
@pytest.mark.parametrize("normal", [False, True])
def test_original_dimensional_equations_against_manufactured_profiles(c3, normal):
    # Exact mean solution at constant viscosity, but arbitrary positive K/E.
    # Its nonzero k/epsilon residuals must match the dimensional PDE residuals.
    r = np.linspace(0.002, 0.4, 151)
    a = 23.0
    integral, f, fp = exact_mean(r, a)
    n = 1 / (8 * a)
    k = 0.08 * np.exp(-3 * r**2)
    kp = -6 * r * k
    kpp = (36 * r**2 - 6) * k
    e = 0.09 * k**2 / n
    ep = -12 * r * e
    epp = (144 * r**2 - 12) * e
    c = PopeConstants(c3=c3)
    qk = r * n / c.sigma_k * kp
    qe = r * n / c.sigma_e * ep
    state = np.array([integral, f, k, qk, e, qe])
    ode = similarity_rhs(r, state, c, normal_strain=normal)
    _, g, chi = similarity_fields(r, state, c)
    qkp = n / c.sigma_k * (kp + r * kpp)
    qep = n / c.sigma_e * (ep + r * epp)
    # Evaluate U d_x + V d_r and physical radial diffusion at unrelated A,x.
    amplitude, x = 3.7, 2.3
    radius = x * r
    u, v = amplitude / x * f, amplitude / x * g
    nu = amplitude * n
    kval = amplitude**2 / x**2 * k
    eps = amplitude**3 / x**4 * e
    ux_r = amplitude / x**2 * fp
    production = nu * ux_r**2
    if normal:
        ux_x = amplitude / x**2 * (-f - r * fp)
        hoop = amplitude / x**2 * g / r
        vr_r = -ux_x - hoop
        production += 2 * nu * (ux_x**2 + vr_r**2 + hoop**2)
    k_x = amplitude**2 / x**3 * (-2 * k - r * kp)
    k_r = amplitude**2 / x**3 * kp
    k_rr = amplitude**2 / x**4 * kpp
    e_x = amplitude**3 / x**5 * (-4 * e - r * ep)
    e_r = amplitude**3 / x**5 * ep
    e_rr = amplitude**3 / x**6 * epp
    k_residual = (
        u * k_x + v * k_r - nu / c.sigma_k * (k_rr + k_r / radius) - production + eps
    )
    e_residual = (
        u * e_x
        + v * e_r
        - nu / c.sigma_e * (e_rr + e_r / radius)
        - c.c1 * eps / kval * production
        + (c.c2 - c.c3 * chi) * eps**2 / kval
    )
    np.testing.assert_allclose(
        (ode[3] - qkp) / r, k_residual * x**4 / amplitude**3, rtol=2e-13, atol=3e-16
    )
    np.testing.assert_allclose(
        (ode[5] - qep) / r, e_residual * x**6 / amplitude**4, rtol=2e-13, atol=3e-16
    )


def test_signed_invariant_against_cartesian_tensor_contraction():
    r = np.linspace(0.01, 0.5, 77)
    integral, f, fp = exact_mean(r)
    k = np.full_like(r, 0.08)
    e = 0.09 * k**2 * (8 * 23)
    state = np.array([integral, f, k, r * 0, e, r * 0])
    _, g, chi = similarity_fields(r, state)
    computed = []
    for j in range(r.size):
        # Axial, radial and azimuthal normal strain; divergence is zero.
        sxx = -f[j] - r[j] * fp[j]
        stt = g[j] / r[j]
        strain = np.diag([sxx, -sxx - stt, stt])
        strain[0, 1] = strain[1, 0] = fp[j] / 2
        rotation = np.zeros((3, 3))
        rotation[0, 1] = fp[j] / 2
        rotation[1, 0] = -fp[j] / 2
        computed.append((k[j] / e[j]) ** 3 * np.trace(rotation @ rotation @ strain))
    np.testing.assert_allclose(chi, computed, rtol=2e-14, atol=1e-17)
    assert np.any(chi < 0) and np.any(chi > 0)


def test_nonphysical_states_and_inconsistent_axis_rejected():
    state = np.array([[0.0], [1.0], [0.08], [0.0], [0.12], [0.0]])
    assert np.all(np.isfinite(similarity_rhs(np.array([0.0]), state)))
    for index, value in [(2, 0), (4, -0.1), (1, np.nan)]:
        bad = state.copy()
        bad[index] = value
        with pytest.raises(ValueError, match="positive K and E"):
            similarity_rhs(np.array([0.0]), bad)
    state[3] = 0.1
    with pytest.raises(ValueError, match="axis"):
        similarity_rhs(np.array([0.0]), state)


def test_axis_uniaxial_strain_has_nonzero_production_without_shear():
    state = np.array([[0.0], [1.0], [0.08], [0.0], [0.12], [0.0]])
    viscosity = 0.09 * 0.08**2 / 0.12
    np.testing.assert_allclose(
        normal_strain_production(np.array([0.0]), state), 3 * viscosity, rtol=2e-15
    )
