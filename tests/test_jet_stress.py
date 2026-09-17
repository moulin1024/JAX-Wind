"""Independent dimensional-gradient and pressure identities for stress audit."""

import numpy as np
from scipy.integrate import simpson

from tools.jet_stress import modeled_stress, thin_jet_pressure, velocity_gradient


def test_self_similar_gradient_against_cartesian_finite_differences():
    a, amplitude = 17.0, 3.1

    def velocity(point):
        x, y, z = point
        r = np.hypot(y, z)
        eta = r / x
        f = np.exp(-a * eta**2)
        integral = -np.expm1(-a * eta**2) / (2 * a)
        g = eta * f - integral / eta if eta else 0.0
        return (
            amplitude / x * np.array([f, g * y / r if r else 0, g * z / r if r else 0])
        )

    for eta in [0.0, 0.07, 0.21]:
        point = np.array([1.6, 1.6 * eta, 0.0])
        h = 2e-6
        dimensional = np.column_stack(
            [
                (velocity(point + h * direction) - velocity(point - h * direction))
                / (2 * h)
                for direction in np.eye(3)
            ]
        )
        f = np.exp(-a * eta**2)
        integral = -np.expm1(-a * eta**2) / (2 * a)
        expected = velocity_gradient(
            np.array(eta), np.array(integral), np.array(f), np.array(-2 * a * eta * f)
        )
        np.testing.assert_allclose(
            dimensional * point[0] ** 2 / amplitude, expected, atol=2e-10, rtol=3e-9
        )
        assert abs(np.trace(expected)) < 1e-15


def test_isotropic_pressure_cancels_normal_stress_flux():
    eta = np.linspace(0, 6, 2049)
    k = np.exp(-(eta**2))
    stress = 2 / 3 * k
    pressure = thin_jet_pressure(eta, stress, stress)
    np.testing.assert_allclose(pressure, -stress + stress[-1], atol=1e-15)
    assert abs(simpson(eta * (stress + pressure), x=eta)) < 4e-15


def test_anisotropic_pressure_and_cross_section_identity():
    eta = np.linspace(0, 6, 4097)
    a = 0.7
    base = np.exp(-(eta**2))
    rr = (1 + a * eta**2) * base
    tt = base
    xx = 2 * base
    pressure = thin_jet_pressure(eta, rr, tt)
    exact = -rr + a / 2 * (base - base[-1]) + rr[-1]
    np.testing.assert_allclose(pressure, exact, atol=4e-12)
    direct = simpson(eta * (xx + pressure), x=eta)
    identity = simpson(eta * (xx - (rr + tt) / 2), x=eta)
    # Nonzero edge stress leaves an explicit finite-domain contribution.
    boundary = eta[-1] ** 2 * rr[-1] / 2
    assert abs(direct - identity - boundary) < 2e-11


def test_rotation_covariance_and_negative_eigenvalue_are_not_repaired():
    gradient = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    stress = modeled_stress(np.array(1.0), np.array(1.0), gradient)
    np.testing.assert_allclose(np.linalg.eigvalsh(stress), [-7 / 3, 2 / 3, 11 / 3])
    theta = 0.43
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    transformed = modeled_stress(
        np.array(1.0), np.array(1.0), rotation @ gradient @ rotation.T
    )
    np.testing.assert_allclose(transformed, rotation @ stress @ rotation.T, atol=1e-15)
    assert np.linalg.eigvalsh(transformed).min() < 0
    assert abs(np.trace(stress) - 2) < 1e-15
