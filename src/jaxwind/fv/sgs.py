"""The anisotropic minimum-dissipation subfilter model.

The AMD model of Rozema, Bae, Moin and Verstappen (Phys. Fluids 27, 085107,
2015) sets the eddy viscosity to the smallest value that keeps the subfilter
scales from growing, given the resolved gradients and an anisotropic filter
width:

    nu_e = max(0, -(d_k u_i)(d_k u_j) S_ij) / (d_m u_l d_m u_l)

where ``d_k`` is the derivative scaled by the filter width ``delta_k`` in that
direction.  Two properties matter for a Cartesian mesh solver: it is
consistent with anisotropic cells without an ad-hoc averaged width, and it
vanishes identically in laminar shear and in any two-component flow, so it
neither damps a developing boundary layer nor needs a wall damping function or
a dynamic procedure to behave near a wall.

Every strain component is evaluated where the staggered mesh already defines
it -- the normal strains at cell centres, the shear strains on the mesh edges
-- so the subfilter stress divergence telescopes exactly, and the wall-normal
gradients come from the same closure as the viscous flux.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from jaxwind.domain.grid import UniformGrid

from .operators import tangential_z_gradient
from .state import Boundaries, StaggeredVelocity


@dataclass(frozen=True, slots=True)
class AnisotropicMinimumDissipation:
    """AMD closure with the Poincare constant of the discretisation.

    ``poincare_constant`` is the square of the ratio between the filter width
    and the mesh spacing, ``delta_k**2 = poincare_constant * h_k**2``.  The
    default of one third is the modified Poincare constant that Rozema et al.
    derive for a second-order accurate scheme, which is what this solver uses.
    """

    poincare_constant: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        if self.poincare_constant <= 0.0:
            raise ValueError("the Poincare constant must be positive")


# Velocity gradients held where the staggered mesh defines them, keyed by the
# component and the direction of differentiation.
EdgeGradients = dict[str, jnp.ndarray]


def _to_cell_from_xy_edge(field: jnp.ndarray) -> jnp.ndarray:
    rolled = jnp.roll(field, -1, axis=2)
    return 0.25 * (
        field + rolled + jnp.roll(field, -1, axis=1) + jnp.roll(rolled, -1, axis=1)
    )


def _to_cell_from_xz_edge(field: jnp.ndarray) -> jnp.ndarray:
    rolled = jnp.roll(field, -1, axis=2)
    return 0.25 * (field[:-1] + field[1:] + rolled[:-1] + rolled[1:])


def _to_cell_from_yz_edge(field: jnp.ndarray) -> jnp.ndarray:
    rolled = jnp.roll(field, -1, axis=1)
    return 0.25 * (field[:-1] + field[1:] + rolled[:-1] + rolled[1:])


def _to_xy_edge_from_cell(field: jnp.ndarray) -> jnp.ndarray:
    rolled = jnp.roll(field, 1, axis=2)
    return 0.25 * (
        field + rolled + jnp.roll(field, 1, axis=1) + jnp.roll(rolled, 1, axis=1)
    )


def _to_xz_edge_from_cell(field: jnp.ndarray) -> jnp.ndarray:
    """Interpolate to the x-z edges, with a vanishing value on both walls."""
    rolled = jnp.roll(field, 1, axis=2)
    interior = 0.25 * (field[:-1] + field[1:] + rolled[:-1] + rolled[1:])
    wall = jnp.zeros_like(field[:1])
    return jnp.concatenate((wall, interior, wall), axis=0)


def _to_yz_edge_from_cell(field: jnp.ndarray) -> jnp.ndarray:
    rolled = jnp.roll(field, 1, axis=1)
    interior = 0.25 * (field[:-1] + field[1:] + rolled[:-1] + rolled[1:])
    wall = jnp.zeros_like(field[:1])
    return jnp.concatenate((wall, interior, wall), axis=0)


def edge_gradients(
    velocity: StaggeredVelocity,
    grid: UniformGrid,
    boundaries: Boundaries,
) -> EdgeGradients:
    """Return every velocity gradient at its natural staggered location."""
    x_velocity, y_velocity, z_velocity = velocity
    return dict(
        xx=(jnp.roll(x_velocity, -1, axis=2) - x_velocity) / grid.dx,
        yy=(jnp.roll(y_velocity, -1, axis=1) - y_velocity) / grid.dy,
        zz=(z_velocity[1:] - z_velocity[:-1]) / grid.dz,
        xy=(x_velocity - jnp.roll(x_velocity, 1, axis=1)) / grid.dy,
        yx=(y_velocity - jnp.roll(y_velocity, 1, axis=2)) / grid.dx,
        xz=tangential_z_gradient(x_velocity, grid, boundaries, "x_velocity"),
        zx=(z_velocity - jnp.roll(z_velocity, 1, axis=2)) / grid.dx,
        yz=tangential_z_gradient(y_velocity, grid, boundaries, "y_velocity"),
        zy=(z_velocity - jnp.roll(z_velocity, 1, axis=1)) / grid.dy,
    )


def cell_gradients(gradients: EdgeGradients) -> list[list[jnp.ndarray]]:
    """Collect the full gradient tensor at cell centres as ``g[i][k]``."""
    return [
        [
            gradients["xx"],
            _to_cell_from_xy_edge(gradients["xy"]),
            _to_cell_from_xz_edge(gradients["xz"]),
        ],
        [
            _to_cell_from_xy_edge(gradients["yx"]),
            gradients["yy"],
            _to_cell_from_yz_edge(gradients["yz"]),
        ],
        [
            _to_cell_from_xz_edge(gradients["zx"]),
            _to_cell_from_yz_edge(gradients["zy"]),
            gradients["zz"],
        ],
    ]


def eddy_viscosity(
    velocity: StaggeredVelocity,
    grid: UniformGrid,
    boundaries: Boundaries,
    model: AnisotropicMinimumDissipation,
    *,
    gradients: EdgeGradients | None = None,
) -> jnp.ndarray:
    """Cell-centred AMD eddy viscosity, non-negative by construction."""
    if gradients is None:
        gradients = edge_gradients(velocity, grid, boundaries)
    tensor = cell_gradients(gradients)
    strain = [
        [0.5 * (tensor[i][k] + tensor[k][i]) for k in range(3)] for i in range(3)
    ]
    widths = [
        model.poincare_constant * spacing**2
        for spacing in (grid.dx, grid.dy, grid.dz)
    ]
    numerator = jnp.zeros_like(tensor[0][0])
    for k, width in enumerate(widths):
        contraction = jnp.zeros_like(numerator)
        for i in range(3):
            for j in range(3):
                contraction = contraction + tensor[i][k] * tensor[j][k] * strain[i][j]
        numerator = numerator - width * contraction
    denominator = jnp.zeros_like(numerator)
    for i in range(3):
        for k in range(3):
            denominator = denominator + tensor[i][k] ** 2
    tiny = jnp.finfo(numerator.dtype).tiny
    return jnp.maximum(numerator, 0.0) / jnp.maximum(denominator, tiny)


def stress_divergence(
    velocity: StaggeredVelocity,
    viscosity: jnp.ndarray,
    grid: UniformGrid,
    boundaries: Boundaries,
    *,
    gradients: EdgeGradients | None = None,
) -> StaggeredVelocity:
    """Tendency ``d_j (2 nu_e S_ij)`` for a cell-centred eddy viscosity.

    The subfilter stress vanishes on the walls, which is the wall-resolved
    condition: AMD already drives the eddy viscosity to zero there, so no
    damping function is involved.
    """
    if gradients is None:
        gradients = edge_gradients(velocity, grid, boundaries)
    normal_x = 2.0 * viscosity * gradients["xx"]
    normal_y = 2.0 * viscosity * gradients["yy"]
    normal_z = 2.0 * viscosity * gradients["zz"]
    shear_xy = (
        2.0
        * _to_xy_edge_from_cell(viscosity)
        * 0.5
        * (gradients["xy"] + gradients["yx"])
    )
    shear_xz = (
        2.0
        * _to_xz_edge_from_cell(viscosity)
        * 0.5
        * (gradients["xz"] + gradients["zx"])
    )
    shear_yz = (
        2.0
        * _to_yz_edge_from_cell(viscosity)
        * 0.5
        * (gradients["yz"] + gradients["zy"])
    )
    x_tendency = (
        (normal_x - jnp.roll(normal_x, 1, axis=2)) / grid.dx
        + (jnp.roll(shear_xy, -1, axis=1) - shear_xy) / grid.dy
        + (shear_xz[1:] - shear_xz[:-1]) / grid.dz
    )
    y_tendency = (
        (jnp.roll(shear_xy, -1, axis=2) - shear_xy) / grid.dx
        + (normal_y - jnp.roll(normal_y, 1, axis=1)) / grid.dy
        + (shear_yz[1:] - shear_yz[:-1]) / grid.dz
    )
    z_interior = (
        (jnp.roll(shear_xz, -1, axis=2) - shear_xz)[1:-1] / grid.dx
        + (jnp.roll(shear_yz, -1, axis=1) - shear_yz)[1:-1] / grid.dy
        + (normal_z[1:] - normal_z[:-1]) / grid.dz
    )
    wall = jnp.zeros_like(velocity.z[:1])
    return StaggeredVelocity(
        x_tendency,
        y_tendency,
        jnp.concatenate((wall, z_interior, wall), axis=0),
    )


def subfilter_tendency(
    velocity: StaggeredVelocity,
    grid: UniformGrid,
    boundaries: Boundaries,
    model: AnisotropicMinimumDissipation,
) -> tuple[StaggeredVelocity, jnp.ndarray]:
    """Return the AMD momentum tendency and the eddy viscosity behind it."""
    gradients = edge_gradients(velocity, grid, boundaries)
    viscosity = eddy_viscosity(
        velocity,
        grid,
        boundaries,
        model,
        gradients=gradients,
    )
    tendency = stress_divergence(
        velocity,
        viscosity,
        grid,
        boundaries,
        gradients=gradients,
    )
    return tendency, viscosity


__all__ = [
    "AnisotropicMinimumDissipation",
    "EdgeGradients",
    "cell_gradients",
    "eddy_viscosity",
    "edge_gradients",
    "stress_divergence",
    "subfilter_tendency",
]
