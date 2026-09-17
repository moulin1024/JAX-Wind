"""Source-frozen round-jet k-epsilon similarity equations; not a LES model.

Coordinates are eta=r/x, U=A F/x, V=A G/x, k=A**2 K/x**2,
epsilon=A**3 E/x**4. The state is (I,F,K,Qk,E,Qe), with I=int eta F,
Qk=eta N K'/sigma_k and Qe=eta N E'/sigma_e. No holdout is read.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PopeConstants:
    """Constants printed in Pope (1978), not the later 1.44/1.92 pair."""

    c_mu: float = 0.09
    c1: float = 1.45
    c2: float = 1.90
    sigma_k: float = 1.0
    sigma_e: float = 1.3
    c3: float = 0.0


STANDARD_CONSTANTS = PopeConstants()


def similarity_fields(eta, state, constants=STANDARD_CONSTANTS):
    """Return N, G and signed vortex-stretching invariant chi.

    The rotation tensor retains the boundary-layer radial axial-velocity
    gradient. Radial/azimuthal normal strain retains continuity. Axis data
    must satisfy I=Qk=Qe=0. Invalid turbulence states are rejected, not clipped.
    """
    eta = np.asarray(eta, dtype=float)
    y = np.asarray(state, dtype=float)
    if y.shape != (6,) + eta.shape:
        raise ValueError("state must have shape (6,) + eta.shape")
    if not np.all(np.isfinite(eta)) or np.any(eta < 0):
        raise ValueError("eta must be finite and nonnegative")
    if not np.all(np.isfinite(y)) or np.any(y[[2, 4]] <= 0):
        raise ValueError("finite state and positive K and E are required")
    if np.any((eta == 0) & (y[[0, 3, 5]] != 0)):
        raise ValueError("axis requires I=Qk=Qe=0")
    integral, f, k, _, e, _ = y
    n = constants.c_mu * k * k / e
    i_over_r = np.divide(integral, eta, out=np.zeros_like(eta), where=eta > 0)
    g = eta * f - i_over_r
    fp = -i_over_r * f / n
    g_over_r = np.divide(g, eta, out=np.array(f / 2), where=eta > 0)
    chi = (k / e) ** 3 * fp * fp / 4 * g_over_r
    return n, g, chi


def normal_strain_production(eta, state, constants=STANDARD_CONSTANTS):
    """Normal-strain contribution 2 N (Sxx^2+Srr^2+Stt^2).

    This diagnostic still neglects dV/dx in shear/rotation and streamwise
    stress transport in mean momentum. It is not a full RANS closure.
    """
    eta = np.asarray(eta, dtype=float)
    integral, f, _, _, _, _ = np.asarray(state, dtype=float)
    n, g, _ = similarity_fields(eta, state, constants)
    fp = -np.divide(integral, eta, out=np.zeros_like(eta), where=eta > 0) * f / n
    sxx = -f - eta * fp
    stt = np.divide(g, eta, out=np.array(f / 2), where=eta > 0)
    srr = -sxx - stt
    return 2 * n * (sxx**2 + srr**2 + stt**2)


def similarity_rhs(eta, state, constants=STANDARD_CONSTANTS, *, normal_strain=False):
    """First-order conservative-flux ODE, including removable axis limits."""
    eta = np.asarray(eta, dtype=float)
    y = np.asarray(state, dtype=float)
    n, _, chi = similarity_fields(eta, y, constants)
    integral, f, k, qk, e, qe = y
    inv_r = np.divide(1.0, eta, out=np.zeros_like(eta), where=eta > 0)
    fp = -integral * inv_r * f / n
    kp = constants.sigma_k * qk * inv_r / n
    ep = constants.sigma_e * qe * inv_r / n
    production = n * fp * fp
    if normal_strain:
        production += normal_strain_production(eta, y, constants)
    return np.array(
        [
            eta * f,
            fp,
            kp,
            -2 * eta * f * k - integral * kp - eta * production + eta * e,
            ep,
            -4 * eta * f * e
            - integral * ep
            - eta * constants.c1 * e / k * production
            + eta * (constants.c2 - constants.c3 * chi) * e * e / k,
        ]
    )
