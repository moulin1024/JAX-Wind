"""Check the implicit momentum/covariance algebra independently."""

import numpy as np
import pytest

from tools.jet_realizability import maximal_realizable_fraction, momentum_stress_affine
from tools.jet_stress import modeled_stress, velocity_gradient


@pytest.mark.parametrize("radial_only", [False, True])
def test_affine_stress_matches_recomputed_velocity_gradient(radial_only):
    eta = np.array([0, 0.07, 0.21])
    f = np.exp(-17 * eta**2)
    integral = -np.expm1(-17 * eta**2) / 34
    k, n = 0.03 * np.ones(3), 0.001 * np.ones(3)
    a, b = momentum_stress_affine(eta, integral, f, k, n, radial_shear_only=radial_only)
    for fraction in [1.0, 0.25, 0.001]:
        fp = -np.divide(integral * f, eta * n * fraction,
                        out=np.zeros(3), where=eta > 0)
        gradient = velocity_gradient(eta, integral, f, fp, radial_shear_only=radial_only)
        expected = modeled_stress(k, n * fraction, gradient)
        np.testing.assert_allclose(a + fraction * b, expected, atol=2e-16, rtol=1e-14)
        np.testing.assert_allclose(np.trace(a + fraction * b, axis1=1, axis2=2), 2*k)


def test_feasible_interval_need_not_contain_zero():
    a, b = np.diag([-0.2, 0.8, 1.4]), np.diag([1, -1, 0])
    result = maximal_realizable_fraction(a, b)
    assert result["feasible"]
    assert abs(result["fraction"] - 0.8) < 1e-12
    assert result["minimum_eigenvalue"] >= -1e-12


def test_infeasible_and_unchanged_states():
    assert not maximal_realizable_fraction(np.diag([-1, 1, 2]), np.zeros((3, 3)))["feasible"]
    assert maximal_realizable_fraction(np.eye(3), np.zeros((3, 3)))["fraction"] == 1


def test_prescribed_gradient_cap_does_not_enforce_momentum_covariance():
    # Simple shear: lowering N at a fixed gradient would reduce R_xr.
    # At fixed momentum flux it remains H, which can exceed the PSD bound.
    a, b = momentum_stress_affine(0.1, 0.01, 1.0, 0.01, 0.001,
                                 radial_shear_only=True)
    assert a[0, 1] == pytest.approx(0.1)
    assert b[0, 1] == 0
    result = maximal_realizable_fraction(a / 0.01, b / 0.01)
    assert not result["feasible"]


def test_invalid_states_rejected():
    with pytest.raises(ValueError):
        momentum_stress_affine(0, 1, 1, 1, 1, radial_shear_only=True)
    with pytest.raises(ValueError):
        maximal_realizable_fraction(np.eye(3), np.full((3, 3), np.nan))


def test_degenerate_interval_is_not_falsely_rejected():
    result = maximal_realizable_fraction(np.diag([0, -0.2, 0.8]), np.diag([0, 1, -1]))
    assert result["feasible"] is None
