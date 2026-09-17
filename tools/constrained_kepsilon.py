"""Experimental consistent-timescale Pope/Durbin boundary-layer equations.

This is a derived hybrid, not a published validated closure. The chosen strain
retains radial shear and continuity-compatible normal strains, as in the
existing normal-production diagnostic. A constrained point keeps N F'=-IF/eta.
"""

import numpy as np
from kepsilon_similarity import PopeConstants

CONSTANTS = PopeConstants(c1=1.44, c2=1.92, c3=0.79)


def fields(eta, state, constants=CONSTANTS, *, constrained=True, trial=False, active_branch=False):
    eta = np.asarray(eta, dtype=float)
    y = np.asarray(state, dtype=float)
    if y.shape != (6,) + eta.shape or not np.all(np.isfinite(y)):
        raise ValueError("finite six-component state required")
    ii, f, k, _, e, _ = y
    if np.any(eta < 0) or np.any(ii < 0) or np.any(f <= 0) or np.any(k <= 0) or np.any(e <= 0):
        raise ValueError("positive F,K,E and nonnegative eta,I required")
    if np.any((eta == 0) & (y[[0, 3, 5]] != 0)):
        raise ValueError("axis requires I=Qk=Qe=0")
    natural = constants.c_mu * k*k/e
    h = np.divide(ii*f, eta, out=np.zeros_like(f), where=eta > 0)
    j = np.divide(ii, eta*eta, out=np.array(f/2), where=eta > 0)
    hoop = f-j
    # Normalize R by K and write its 2x2 determinant as a polynomial in N/K.
    a, d = 2/3-2*eta*h/k, 2/3+2*eta*h/k
    aa = -4*f*j
    bb = 2*f*d-2*j*a
    cc = a*d-(h/k)**2
    discriminant = bb*bb-4*aa*cc
    root = np.sqrt(np.maximum(discriminant, 0))
    q = -0.5*(bb + np.where(bb >= 0, root, -root))
    first = np.divide(q, aa, out=np.full_like(f, np.nan), where=aa != 0)
    second = np.divide(cc, q, out=np.full_like(f, np.nan), where=q != 0)
    lo, hi = np.minimum(first, second), np.maximum(first, second)
    lo = np.maximum(lo, np.maximum(0, -a/(2*f)))
    radial_bound = np.divide(d, 2*j, out=np.full_like(f, np.inf), where=j > 0)
    hoop_bound = np.divide(1/3, hoop, out=np.full_like(f, np.inf), where=hoop > 0)
    hi = np.minimum(hi, np.minimum(radial_bound, hoop_bound))
    candidate = k*hi if active_branch else np.minimum(natural, k*hi)
    feasible = (discriminant >= 0) & (hi > 0) & (lo <= candidate/k) & np.isfinite(candidate)
    if constrained and not trial and not np.all(feasible):
        raise ValueError("momentum-compatible covariance constraint is infeasible")
    # Newton extension only: rejected states use the original finite viscosity.
    n = np.where(feasible, candidate, natural) if constrained else natural
    time = n/(constants.c_mu*k)
    fp = -h/n
    axial = -f-eta*fp
    radial = -axial-hoop
    production = n*(fp*fp + 2*(axial*axial+radial*radial+hoop*hoop))
    chi = time**3 * fp*fp*hoop/4
    return {"viscosity": n, "time": time, "fp": fp, "production": production,
            "chi": chi, "feasible": feasible, "fraction": n/natural}


def rhs(eta, state, constants=CONSTANTS, *, constrained=True, trial=False, active_branch=False):
    eta = np.asarray(eta, dtype=float)
    ii, f, k, qk, e, qe = np.asarray(state, dtype=float)
    values = fields(eta, state, constants, constrained=constrained, trial=trial, active_branch=active_branch)
    n, time, production, chi = (values[key] for key in ("viscosity", "time", "production", "chi"))
    invr = np.divide(1.0, eta, out=np.zeros_like(eta), where=eta > 0)
    kp, ep = constants.sigma_k*qk*invr/n, constants.sigma_e*qe*invr/n
    return np.array([
        eta*f, values["fp"], kp,
        -2*eta*f*k-ii*kp-eta*production+eta*e, ep,
        -4*eta*f*e-ii*ep-eta*constants.c1*production/time
        +eta*(constants.c2-constants.c3*chi)*e/time,
    ])
