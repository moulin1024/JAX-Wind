"""Equilibrium smooth-wall stress for an unresolved rectangular duct.

Spalding's continuous law uses kappa=0.41, E=9.8 (OpenFOAM SpaldingsLaw).
This is a local velocity-based LES closure, not Fluent's k-based RANS wall
function. Cell-centre sampling, equilibrium boundary layers and adiabatic walls
are explicit approximations. No fitted roughness length is introduced.
"""

import jax
import jax.numpy as jnp

from .numerics.discretization import _cells_to_faces, cell_velocity
from .state import StaggeredVelocity


def smooth_friction_velocity(speed, distance, viscosity):
    """Invert Spalding's law for nonnegative speed and positive y and nu."""
    speed = jnp.asarray(speed)
    reynolds = speed * distance / viscosity

    # y+ >= u+, hence u+ <= sqrt(U*y/nu). Bisection also handles rest.
    def bisect(_, bracket):
        lo, hi = bracket
        up = 0.5 * (lo + hi)
        a = jnp.minimum(0.41 * up, 60.0)
        remainder = jnp.maximum(jnp.expm1(a) - a - a**2 / 2 - a**3 / 6, 0)
        yp = up + remainder / 9.8
        below = up * yp < reynolds
        return jnp.where(below, up, lo), jnp.where(below, hi, up)

    lo, hi = jax.lax.fori_loop(
        0, 48, bisect, (jnp.zeros_like(speed), jnp.sqrt(reynolds))
    )
    up = 0.5 * (lo + hi)
    return jnp.where(speed > 0, speed / jnp.maximum(up, 1e-30), 0.0)


def smooth_duct_tendency(velocity, grid, viscosity):
    """Apply opposing tangential wall stress to all four y/z boundary cells.

    Use with free-slip resolved fluxes, to avoid counting wall shear twice.
    Stress per unit density is u_tau^2. Its face-area/volume ratio is 1/delta.
    The normal components remain impermeable. No wall thermal flux is added.
    """
    if viscosity <= 0:
        raise ValueError("smooth-wall law requires positive molecular viscosity")
    if velocity.y.shape[1] != grid.ny + 1 or velocity.x.shape[2] != grid.nx + 1:
        raise ValueError("smooth duct requires nonperiodic x and y faces")
    cells = cell_velocity(velocity)
    force = [jnp.zeros_like(cells[0]) for _ in range(3)]
    for axis, widths, components in (
        (0, grid.z_widths, (0, 1)),
        (1, grid.y_widths, (0, 2)),
    ):
        for index, width in ((0, widths[0]), (-1, widths[-1])):
            location = [slice(None)] * 3
            location[axis] = index
            location = tuple(location)
            a, b = (cells[c][location] for c in components)
            speed = jnp.sqrt(a * a + b * b)
            friction = smooth_friction_velocity(speed, 0.5 * float(width), viscosity)
            coefficient = friction**2 / jnp.maximum(speed, 1e-30) / float(width)
            for c in components:
                force[c] = force[c].at[location].add(-coefficient * cells[c][location])
    return StaggeredVelocity(
        _cells_to_faces(force[0], grid, 2, periodic=False, boundary="copy"),
        _cells_to_faces(force[1], grid, 1, periodic=False, boundary="zero"),
        _cells_to_faces(force[2], grid, 0, periodic=False, boundary="zero"),
    )
