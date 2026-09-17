"""Axisymmetric similarity gradients, modeled stress and thin-jet pressure.

These diagnostics do not alter a profile or make a covariance realizable.
The complete velocity gradient includes dV/dx; the radial-shear approximation
omits it. All fields use the common A/x and A^2/x^2 similarity scales.
"""

import numpy as np
from scipy.integrate import cumulative_simpson


def velocity_gradient(eta, integral, f, fp, *, radial_shear_only=False):
    eta, integral, f, fp = np.broadcast_arrays(eta, integral, f, fp)
    if np.any(eta < 0) or not all(
        np.all(np.isfinite(v)) for v in (eta, integral, f, fp)
    ):
        raise ValueError("finite similarity fields and nonnegative radius required")
    g = eta * f - np.divide(integral, eta, out=np.zeros_like(f), where=eta > 0)
    hoop = np.divide(g, eta, out=np.array(f / 2), where=eta > 0)
    axial = -f - eta * fp
    radial = -axial - hoop
    gradient = np.zeros(eta.shape + (3, 3))
    gradient[..., 0, 0] = axial
    gradient[..., 0, 1] = fp
    gradient[..., 1, 0] = 0 if radial_shear_only else -g - eta * radial
    gradient[..., 1, 1] = radial
    gradient[..., 2, 2] = hoop
    return gradient


def modeled_stress(k, viscosity, gradient):
    """Return the raw Boussinesq covariance; negative eigenvalues are retained."""
    k, viscosity = np.broadcast_arrays(k, viscosity)
    if (
        np.any(k < 0)
        or np.any(viscosity < 0)
        or not all(np.all(np.isfinite(v)) for v in (k, viscosity, gradient))
        or gradient.shape != k.shape + (3, 3)
    ):
        raise ValueError("finite gradient and nonnegative k/viscosity required")
    strain = (gradient + np.swapaxes(gradient, -1, -2)) / 2
    return (2 / 3 * k)[..., None, None] * np.eye(3) - 2 * viscosity[
        ..., None, None
    ] * strain


def thin_jet_pressure(eta, radial_stress, hoop_stress, outer_pressure=0.0):
    """Integrate radial stress equilibrium inward from a finite outer edge.

    The stress at that edge is retained explicitly. For compact profiles both
    outer normal stresses are zero. Mean radial inertia and axial shear-stress
    transport are omitted; this is not the complete RANS pressure field.
    """
    eta = np.asarray(eta, dtype=float)
    rr, tt = np.asarray(radial_stress), np.asarray(hoop_stress)
    if (
        eta.ndim != 1
        or eta.size < 3
        or np.any(np.diff(eta) <= 0)
        or eta[0] < 0
        or rr.shape != eta.shape
        or tt.shape != eta.shape
        or not all(np.all(np.isfinite(v)) for v in (eta, rr, tt, outer_pressure))
    ):
        raise ValueError(
            "finite increasing radial samples and matching stresses required"
        )
    if eta[0] == 0 and not np.isclose(rr[0], tt[0], rtol=1e-12, atol=1e-14):
        raise ValueError("axis requires equal radial and hoop stress")
    integrand = np.divide(rr - tt, eta, out=np.zeros_like(eta), where=eta > 0)
    inward = cumulative_simpson(integrand[::-1], x=-eta[::-1], initial=0)[::-1]
    return -rr + inward + rr[-1] + outer_pressure
