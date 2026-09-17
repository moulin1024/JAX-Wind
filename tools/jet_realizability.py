"""Local covariance feasibility at fixed jet momentum flux; not a closure.

The existing boundary-layer momentum equation requires N F'=-I F/eta.
Consequently changing N changes the strain. Covariance is affine in N even
though the strain is not. No turbulence equation or measured data is fitted.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def momentum_stress_affine(eta, integral, f, k, viscosity, *, radial_shear_only):
    """Return A,B for R(t)=A+t B, with N=t*viscosity and momentum enforced.

    t=0 is an algebraic limiting stress, not a finite-gradient solution.
    The complete-gradient option is a diagnostic; mean momentum still uses
    the boundary-layer equation. Inputs are broadcastable nonnegative fields.
    """
    eta, integral, f, k, viscosity = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (eta, integral, f, k, viscosity)]
    )
    if any(not np.all(np.isfinite(v)) or np.any(v < 0)
           for v in (eta, integral, f, k, viscosity)):
        raise ValueError("finite nonnegative fields required")
    if np.any((eta == 0) & (integral != 0)):
        raise ValueError("axis integral must vanish")
    h = np.divide(integral * f, eta, out=np.zeros_like(f), where=eta > 0)
    hoop = f - np.divide(integral, eta**2, out=np.array(f / 2), where=eta > 0)
    a = np.zeros(eta.shape + (3, 3))
    b = np.zeros_like(a)
    a[..., 0, 0] = 2 * k / 3 - 2 * eta * h
    a[..., 1, 1] = 2 * k / 3 + 2 * eta * h
    a[..., 2, 2] = 2 * k / 3
    b[..., 0, 0] = 2 * viscosity * f
    b[..., 1, 1] = -2 * viscosity * (f - hoop)
    b[..., 2, 2] = -2 * viscosity * hoop
    a[..., 0, 1] = a[..., 1, 0] = h if radial_shear_only else h * (1 - eta**2)
    if not radial_shear_only:
        b[..., 0, 1] = b[..., 1, 0] = viscosity * eta * f
    return a, b


def maximal_realizable_fraction(a, b):
    """Largest positive t<=1 with PSD A+t B, or report infeasibility.

    Minimum eigenvalue is concave in t. A bounded scalar maximization locates
    its nonnegative interval, and a bracketed root locates the upper end.
    Inputs should be normalized by local K for a dimensionless tolerance.
    No negative eigenvalue is clipped and no modified field is returned.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (3, 3) or b.shape != (3, 3):
        raise ValueError("two 3-by-3 matrices required")
    if any(not np.all(np.isfinite(v)) or not np.allclose(v, v.T, atol=1e-14, rtol=0)
           for v in (a, b)):
        raise ValueError("finite symmetric matrices required")

    def eigenvalue(t):
        return float(np.linalg.eigvalsh(a + t * b)[0])

    original = eigenvalue(1.0)
    if original >= 0:
        return {"feasible": True, "fraction": 1.0, "minimum_eigenvalue": original}
    result = minimize_scalar(
        lambda t: -eigenvalue(t), bounds=(0.0, 1.0), method="bounded",
        options={"xatol": 1e-14, "maxiter": 200},
    )
    if not result.success:
        raise RuntimeError("local covariance maximization did not converge")
    candidates = [(eigenvalue(0.0), 0.0), (eigenvalue(result.x), float(result.x))]
    best, location = max(candidates)
    if best < -1e-12:
        return {"feasible": False, "fraction": None,
                "maximum_minimum_eigenvalue": best, "maximizer": location}
    if best <= 1e-12:
        return {"feasible": None, "fraction": None,
                "maximum_minimum_eigenvalue": best, "maximizer": location,
                "reason": "degenerate boundary feasibility requires separate analysis"}
    fraction = brentq(eigenvalue, location, 1.0, xtol=1e-14, rtol=1e-14)
    return {"feasible": True, "fraction": float(fraction),
            "minimum_eigenvalue": eigenvalue(fraction)}
