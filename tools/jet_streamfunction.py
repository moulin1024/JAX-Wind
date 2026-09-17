"""Positive finite-volume round-jet marching primitives in mass streamfunction.

psi=int_0^r s U ds, so r^2=2 int_0^psi 1/U dpsi. Momentum becomes
U_x|psi = d_psi(D U_psi), D=r^2 nu_t U. No physical model is selected here.
"""

import numpy as np
from scipy.linalg import solve_banded


def geometry(faces, velocity):
    """Piecewise-constant 1/U quadrature; return squared face/cell radii."""
    faces = np.asarray(faces, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    widths = np.diff(faces)
    if (
        faces.shape != (velocity.size + 1,)
        or faces[0] != 0
        or np.any(widths <= 0)
        or not np.all(np.isfinite(faces))
        or not np.all(np.isfinite(velocity))
        or np.any(velocity <= 0)
    ):
        raise ValueError(
            "increasing finite faces from zero and positive velocity required"
        )
    r2f = np.r_[0.0, 2 * np.cumsum(widths / velocity)]
    return r2f, (r2f[1:] + r2f[:-1]) / 2


def conductance(faces, velocity, viscosity, ambient_velocity, ambient_viscosity):
    """D/delta_psi on right faces; zero axis flux is implicit.

    Arithmetic face interpolation permits diffusion into the ambient stream.
    Harmonic interpolation can pin a nearly zero-velocity/turbulence front.
    """
    r2f, _ = geometry(faces, velocity)
    psi = (faces[1:] + faces[:-1]) / 2
    distances = np.r_[np.diff(psi), faces[-1] - psi[-1]]
    viscosity = np.broadcast_to(viscosity, velocity.shape)
    if (
        np.any(viscosity < 0)
        or not np.all(np.isfinite(viscosity))
        or not np.isfinite(ambient_velocity)
        or ambient_velocity <= 0
        or not np.isfinite(ambient_viscosity)
        or ambient_viscosity < 0
    ):
        raise ValueError(
            "nonnegative finite viscosity and positive ambient velocity required"
        )
    uf = (velocity + np.r_[velocity[1:], ambient_velocity]) / 2
    nf = (viscosity + np.r_[viscosity[1:], ambient_viscosity]) / 2
    return r2f[1:] * uf * nf / distances


def diffusion_rate(values, coefficients, widths, ambient):
    flux = coefficients * (np.r_[values[1:], ambient] - values)
    return (flux - np.r_[0, flux[:-1]]) / widths


def positive_step(values, coefficients, widths, ambient, dx, source=0.0, sink=0.0):
    """Implicit diffusion/destruction, explicit nonnegative production.

    The linear solve is an M-matrix transaction. No clipping/renormalization
    follows it. The caller must establish coefficient/time-step consistency.
    Returns state and outward boundary transfer in integral(values dpsi).
    """
    values, coefficients, widths = map(np.asarray, (values, coefficients, widths))
    source = np.broadcast_to(source, values.shape)
    sink = np.broadcast_to(sink, values.shape)
    if (
        values.ndim != 1
        or values.size < 2
        or coefficients.shape != values.shape
        or widths.shape != values.shape
        or any(
            not np.all(np.isfinite(q))
            for q in (values, coefficients, widths, source, sink)
        )
        or np.any(values < 0)
        or np.any(coefficients < 0)
        or np.any(widths <= 0)
        or np.any(source < 0)
        or np.any(sink < 0)
        or not np.isfinite(ambient)
        or ambient < 0
        or not np.isfinite(dx)
        or dx <= 0
    ):
        raise ValueError(
            "finite nonnegative state, rates and boundary; positive widths/dx required"
        )
    right = coefficients / widths
    left = np.r_[0, coefficients[:-1]] / widths
    band = np.zeros((3, values.size))
    band[1] = 1 + dx * (right + left + sink)
    band[0, 1:] = -dx * right[:-1]
    band[2, :-1] = -dx * left[1:]
    rhs = values + dx * source
    rhs[-1] += dx * right[-1] * ambient
    result = solve_banded((1, 1), band, rhs, check_finite=False)
    if np.any(result < 0) or not np.all(np.isfinite(result)):
        raise FloatingPointError("positive diffusion solve produced invalid state")
    export = dx * coefficients[-1] * (result[-1] - ambient)
    return result, export
