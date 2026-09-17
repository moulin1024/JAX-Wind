"""Independent algebra, dimensional source, and constrained-edge checks."""

import numpy as np
import pytest

from tools.constrained_kepsilon import CONSTANTS, fields, rhs
from tools.jet_realizability import maximal_realizable_fraction, momentum_stress_affine
from tools.kepsilon_similarity import similarity_rhs


def state(eta, epsilon):
    f = np.exp(-17*eta**2)
    ii = -np.expm1(-17*eta**2)/34
    k = 0.5*np.exp(-3*eta**2)
    return np.array([ii, f, k, -eta*0.01, np.full_like(eta, epsilon), -eta*0.02])


def test_unconstrained_equations_recover_original_normal_production():
    eta = np.r_[0, np.linspace(0.001, 0.3, 61)]
    y = state(eta, 0.1)
    np.testing.assert_allclose(rhs(eta, y, constrained=False),
        similarity_rhs(eta, y, CONSTANTS, normal_strain=True), rtol=3e-14, atol=1e-13)


def test_analytic_cap_matches_independent_concave_eigenvalue_search():
    eta = np.r_[0, np.linspace(0.001, 0.3, 61)]
    y = state(eta, 0.001)
    result = fields(eta, y)
    n0 = CONSTANTS.c_mu*y[2]**2/y[4]
    a, b = momentum_stress_affine(eta, y[0], y[1], y[2], n0, radial_shear_only=True)
    for i in range(eta.size):
        independent = maximal_realizable_fraction(a[i]/y[2, i], b[i]/y[2, i])
        assert independent["feasible"]
        assert result["fraction"][i] == pytest.approx(independent["fraction"], abs=2e-13)
    assert np.all(result["fraction"] < 1)
    np.testing.assert_allclose(result["viscosity"]*result["fp"],
        -np.divide(y[0]*y[1], eta, out=np.zeros_like(eta), where=eta > 0), atol=1e-16)


def test_limited_timescale_controls_both_epsilon_sources_and_invariant():
    eta = np.array([0.08])
    y = state(eta, 0.001)
    value = fields(eta, y)
    n, time = value["viscosity"], value["time"]
    assert time < y[2]/y[4]/10
    derivative = rhs(eta, y)
    # Dimensional pointwise epsilon PDE at unrelated amplitude and distance.
    amplitude, x = 3.7, 2.3
    u = amplitude*y[1]/x
    v = amplitude*(eta*y[1]-y[0]/eta)/x
    eps = amplitude**3*y[4]/x**4
    eps_x = amplitude**3*(-4*y[4]-eta*derivative[4])/x**5
    eps_r = amplitude**3*derivative[4]/x**5
    physical_time = x*x/amplitude*time
    physical_production = amplitude**3/x**4*value["production"]
    residual = u*eps_x+v*eps_r-CONSTANTS.c1*physical_production/physical_time
    residual += (CONSTANTS.c2-CONSTANTS.c3*value["chi"])*eps/physical_time
    np.testing.assert_allclose(derivative[5]/eta, residual*x**6/amplitude**4, rtol=2e-14)
    g = np.zeros((3, 3))
    fp = value["fp"][0]
    g[0, 0] = -y[1, 0]-eta[0]*fp
    g[2, 2] = y[1, 0]-y[0, 0]/eta[0]**2
    g[1, 1] = -g[0, 0]-g[2, 2]
    g[0, 1] = fp
    strain, rotation = (g+g.T)/2, (g-g.T)/2
    np.testing.assert_allclose(value["chi"], time**3*np.trace(rotation@rotation@strain))
    np.testing.assert_allclose(value["production"], 2*n*np.sum(strain*strain))


def test_active_edge_powers_satisfy_leading_transport_balance():
    edge, ie, famp, eamp = 0.3, 0.02, 0.01, 0.01
    ncoef = ie/edge
    strain = np.diag([edge*famp, ie/edge**2-edge*famp, -ie/edge**2])
    strain[0, 1] = strain[1, 0] = -famp/2
    kamp = 3*ncoef*np.linalg.eigvalsh(strain)[-1]
    errors = []
    # Source corrections are O(s**0.7); test inside that asymptotic regime.
    for s in [1e-7, 1e-8, 1e-9]:
        eta = np.array([edge-s])
        f, k, e = famp*s, kamp*s, eamp*s**1.3
        ii = ie-famp*(edge*s*s/2-s**3/3)
        qk = -eta*ncoef*s*kamp
        qe = -eta*ncoef*eamp*s**1.3
        y = np.array([[ii], [f], [k], qk, [e], qe])
        v = fields(eta, y)
        derivative = rhs(eta, y)
        expected_qkp = ncoef*kamp*(edge-2*s)
        expected_qep = ncoef*eamp*(1.3*edge*s**0.3-2.3*s**1.3)
        error = max(abs(v["viscosity"][0]/(ncoef*s)-1),
                    abs(derivative[1, 0]/(-famp)-1),
                    abs(derivative[3, 0]/expected_qkp-1),
                    abs(derivative[5, 0]/expected_qep-1))
        errors.append(error)
        assert v["fraction"][0] < 1
    assert errors[-1] < 1e-4
    orders = np.log10(np.array(errors[:-1])/np.array(errors[1:]))
    assert np.all((orders > 0.6) & (orders < 0.8)), (errors, orders)


def test_infeasible_state_rejected_and_trial_extension_marked():
    y = np.array([[0.01], [1.0], [0.01], [0.0], [0.001], [0.0]])
    with pytest.raises(ValueError, match="infeasible"):
        fields(np.array([0.1]), y)
    v = fields(np.array([0.1]), y, trial=True)
    assert not v["feasible"][0]
    assert np.isfinite(v["viscosity"][0])
