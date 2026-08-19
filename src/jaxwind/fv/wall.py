"""Monin-Obukhov surface stress for the finite-volume solver.

In a wall-modelled large-eddy simulation the near-wall flow is not resolved,
so the surface exerts a parameterised drag instead of a viscous stress.  Under
neutral stratification Monin-Obukhov similarity reduces to the logarithmic law
and the surface stress follows from the wind at one reference height,

    u_*  = kappa * U(z_1) / ln(z_1 / z_0),
    tau  = u_*^2 * (u, v) / U,

which enters a finite-volume momentum balance directly as the flux through the
bottom face of the wall-adjacent control volume.  That is the natural place
for it: no ghost value is invented, and the stress the model returns is the
stress the discrete equations feel.

The reference height needs care in a finite-volume code, and this is where the
finite-volume treatment departs from the finite-difference one.  A
finite-volume unknown is the average of the profile over its cell, whereas the
logarithmic law is a point value.  Because the logarithm is concave, the cell
average is smaller than the value at the cell centre, so feeding the average
into the law as though it were the centre value underestimates the surface
stress (Clement, Lemarie and Blayo, arXiv:2305.09254).  Integrating the law
over the first cell gives the height at which the two agree exactly,

    (1/dz) * integral_0^dz ln(z/z0) dz = ln(dz/z0) - 1  =>  z_1 = dz / e,

so ``CELL_AVERAGE`` sampling evaluates the law at ``dz / e`` instead of the
``dz / 2`` that a finite-difference code would use.  Both are available:
``CELL_CENTRE`` reproduces the usual finite-difference convention.

References
----------
Schumann (1975); Moeng, J. Atmos. Sci. 41, 2052 (1984) -- the surface stress
parameterisation and its planar-averaged variant.
Bou-Zeid, Meneveau and Parlange, Phys. Fluids 17, 025105 (2005) -- a local law
of the wall overpredicts the stress, which filtering the input wind mitigates.
Kawai and Larsson, Phys. Fluids 24, 015105 (2012) -- log-layer mismatch and the
contamination of the wall-adjacent LES data.
Clement, Lemarie and Blayo, arXiv:2305.09254 (2023) -- the finite-volume
reconstruction of the surface layer inside the first cell.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp

from jaxwind.domain.grid import UniformGrid

from .state import FREE_SLIP, Boundaries, StaggeredVelocity, Wall


CELL_AVERAGE = "cell-average"
CELL_CENTRE = "cell-centre"

LOCAL = "local"
PLANAR = "planar"


@dataclass(frozen=True, slots=True)
class MoninObukhovWall:
    """A rough wall whose drag follows the neutral logarithmic law.

    ``sampling`` selects the reference height inside the first cell, and
    ``averaging`` selects between the local law of the wall and the
    planar-averaged wind speed of Moeng (1984), which removes the bias a local
    law carries from the correlation between the speed and its fluctuations.
    """

    roughness: float
    von_karman: float = 0.4
    sampling: str = CELL_AVERAGE
    averaging: str = LOCAL

    def __post_init__(self) -> None:
        if self.roughness <= 0.0:
            raise ValueError("the roughness length must be positive")
        if self.von_karman <= 0.0:
            raise ValueError("the von Karman constant must be positive")
        if self.sampling not in (CELL_AVERAGE, CELL_CENTRE):
            raise ValueError(f"unsupported sampling: {self.sampling!r}")
        if self.averaging not in (LOCAL, PLANAR):
            raise ValueError(f"unsupported averaging: {self.averaging!r}")

    def reference_height(self, grid: UniformGrid) -> float:
        """Height inside the first cell at which the law is evaluated."""
        if self.sampling == CELL_AVERAGE:
            return grid.dz / math.e
        return 0.5 * grid.dz

    def drag_coefficient(self, grid: UniformGrid) -> float:
        """The factor relating ``U * (u, v)`` to the surface stress."""
        height = self.reference_height(grid)
        if height <= self.roughness:
            raise ValueError(
                "the first cell is not tall enough for the logarithmic law: "
                f"reference height {height:g} is below the roughness "
                f"{self.roughness:g}"
            )
        return (self.von_karman / math.log(height / self.roughness)) ** 2


def _first_level_speed(
    x_velocity: jnp.ndarray,
    y_velocity: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Cell-centred wind components and speed on the wall-adjacent level."""
    centred_x = 0.5 * (x_velocity + jnp.roll(x_velocity, -1, axis=1))
    centred_y = 0.5 * (y_velocity + jnp.roll(y_velocity, -1, axis=0))
    speed = jnp.sqrt(centred_x**2 + centred_y**2)
    return centred_x, centred_y, speed


def surface_stress(
    velocity: StaggeredVelocity,
    grid: UniformGrid,
    model: MoninObukhovWall,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the surface stress on the x-faces and the y-faces.

    The stress is formed at cell centres, where the wind speed is defined
    without bias, and then interpolated to the faces that carry the two
    horizontal momentum components.
    """
    first_x, first_y = velocity.x[0], velocity.y[0]
    centred_x, centred_y, speed = _first_level_speed(first_x, first_y)
    if model.averaging == PLANAR:
        speed = jnp.mean(speed)
    coefficient = model.drag_coefficient(grid)
    stress_x = coefficient * speed * centred_x
    stress_y = coefficient * speed * centred_y
    return (
        0.5 * (stress_x + jnp.roll(stress_x, 1, axis=1)),
        0.5 * (stress_y + jnp.roll(stress_y, 1, axis=0)),
    )


def friction_velocity(
    velocity: StaggeredVelocity,
    grid: UniformGrid,
    model: MoninObukhovWall,
) -> jnp.ndarray:
    """Planar-averaged friction velocity implied by the surface stress."""
    _, _, speed = _first_level_speed(velocity.x[0], velocity.y[0])
    if model.averaging == PLANAR:
        speed = jnp.mean(speed)
    return jnp.sqrt(jnp.mean(model.drag_coefficient(grid) * speed**2))


def wall_tendency(
    velocity: StaggeredVelocity,
    grid: UniformGrid,
    model: MoninObukhovWall,
) -> StaggeredVelocity:
    """Momentum tendency from the surface drag on the wall-adjacent cells.

    The stress is the flux through the bottom face of the first control
    volume, so it reaches only that cell, divided by the cell height.
    """
    stress_x, stress_y = surface_stress(velocity, grid, model)
    x_tendency = jnp.zeros_like(velocity.x).at[0].set(-stress_x / grid.dz)
    y_tendency = jnp.zeros_like(velocity.y).at[0].set(-stress_y / grid.dz)
    return StaggeredVelocity(
        x_tendency,
        y_tendency,
        jnp.zeros_like(velocity.z),
    )


def logarithmic_profile(
    grid: UniformGrid,
    friction: float,
    model: MoninObukhovWall,
) -> jnp.ndarray:
    """Cell-averaged logarithmic wind profile, the equilibrium of the model.

    The average over each cell is used rather than the value at the centre, so
    that the profile is the exact discrete equilibrium of a finite-volume
    solver using ``CELL_AVERAGE`` sampling.
    """
    upper = (jnp.arange(grid.nz) + 1.0) * grid.dz
    lower = jnp.arange(grid.nz) * grid.dz
    roughness = model.roughness

    def integral(height):
        # The antiderivative z * (ln(z / z0) - 1), whose limit at z = 0 is
        # zero. The logarithm is evaluated on a dummy height there so that no
        # infinity is formed: a small positive guard would underflow to zero
        # in single precision and produce a NaN.
        positive = height > 0.0
        safe = jnp.where(positive, height, 1.0)
        return jnp.where(positive, safe * (jnp.log(safe / roughness) - 1.0), 0.0)

    averaged = (integral(upper) - integral(lower)) / grid.dz
    return friction / model.von_karman * jnp.maximum(averaged, 0.0)


def monin_obukhov_boundaries() -> Boundaries:
    """Boundaries whose resolved viscous flux vanishes on both walls.

    The surface model supplies the entire wall stress, so the resolved viscous
    closure must not add a second one; the top is frictionless by assumption.
    """
    return Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP))


__all__ = [
    "CELL_AVERAGE",
    "CELL_CENTRE",
    "LOCAL",
    "PLANAR",
    "MoninObukhovWall",
    "friction_velocity",
    "logarithmic_profile",
    "monin_obukhov_boundaries",
    "surface_stress",
    "wall_tendency",
]
