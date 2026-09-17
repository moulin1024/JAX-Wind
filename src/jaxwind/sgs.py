"""Eddy-viscosity subfilter models for the finite-volume solver.

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

from jaxwind.domain.grid import Grid

from .metrics import (
    cell_volumes,
    center_distances,
    shaped_center_distances,
    shaped_widths,
)
from jaxwind.numerics.discretization import tangential_z_gradient
from .state import (
    Boundaries,
    StaggeredVelocity,
    spanwise_is_periodic,
    streamwise_is_periodic,
)


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


@dataclass(frozen=True, slots=True)
class StaticSmagorinsky:
    """Classical static Smagorinsky closure.

    The filter width is the cubic root of the cell volume and ``coefficient``
    is the usual Smagorinsky constant, conventionally about 0.16 for freely
    sheared turbulence. This is the undamped classical model.
    """

    coefficient: float = 0.16

    def __post_init__(self) -> None:
        if not 0.0 <= self.coefficient < 1.0:
            raise ValueError("the Smagorinsky coefficient must lie in [0, 1)")


@dataclass(frozen=True, slots=True)
class FluentSmagorinsky(StaticSmagorinsky):
    """Fluent 2026 R1 static model: l=min(kappa*ground_distance,Cs*V^(1/3)).

    The atmospheric upper boundary is a symmetry lid, not a second ground wall.
    Defaults follow the documented Cs=0.1. Existing SGS choices are unchanged.
    """

    coefficient: float = 0.1
    von_karman: float = 0.41

    def __post_init__(self):
        if not 0 < self.coefficient < 1 or not 0 < self.von_karman < 1:
            raise ValueError("positive finite Fluent LES constants required")

    def length_scale(self, grid, dtype):
        distance = jnp.asarray(grid.z_centers, dtype)[:, None, None]
        return jnp.minimum(self.von_karman*distance,
                           self.coefficient*cell_volumes(grid, dtype)**(1/3))


# Velocity gradients held where the staggered mesh defines them, keyed by the
# component and the direction of differentiation.
EdgeGradients = dict[str, jnp.ndarray]


def _to_cell_from_xy_edge(
    field: jnp.ndarray,
    *,
    open_x: bool = False,
    wall_y: bool = False,
) -> jnp.ndarray:
    x_average = (
        0.5 * (field[..., :-1] + field[..., 1:])
        if open_x
        else 0.5 * (field + jnp.roll(field, -1, axis=2))
    )
    return (
        0.5 * (x_average[:, :-1] + x_average[:, 1:])
        if wall_y
        else 0.5 * (x_average + jnp.roll(x_average, -1, axis=1))
    )


def _to_cell_from_xz_edge(
    field: jnp.ndarray,
    *,
    open_x: bool = False,
) -> jnp.ndarray:
    if open_x:
        x_average = 0.5 * (field[..., :-1] + field[..., 1:])
        return 0.5 * (x_average[:-1] + x_average[1:])
    rolled = jnp.roll(field, -1, axis=2)
    return 0.25 * (field[:-1] + field[1:] + rolled[:-1] + rolled[1:])


def _to_cell_from_yz_edge(field: jnp.ndarray, *, wall_y: bool = False) -> jnp.ndarray:
    z_average = 0.5 * (field[:-1] + field[1:])
    return (
        0.5 * (z_average[:, :-1] + z_average[:, 1:])
        if wall_y
        else 0.5 * (z_average + jnp.roll(z_average, -1, axis=1))
    )


def _cells_to_open_x_faces(field: jnp.ndarray) -> jnp.ndarray:
    interior = 0.5 * (field[..., :-1] + field[..., 1:])
    return jnp.concatenate((field[..., :1], interior, field[..., -1:]), axis=2)


def _to_xy_edge_from_cell(
    field: jnp.ndarray,
    *,
    open_x: bool = False,
    wall_y: bool = False,
) -> jnp.ndarray:
    x_faces = _cells_to_open_x_faces(field) if open_x else 0.5 * (field + jnp.roll(field, 1, axis=2))
    if not wall_y:
        return 0.5 * (x_faces + jnp.roll(x_faces, 1, axis=1))
    interior = 0.5 * (x_faces[:, :-1] + x_faces[:, 1:])
    wall = jnp.zeros_like(x_faces[:, :1])
    return jnp.concatenate((wall, interior, wall), axis=1)


def _to_xz_edge_from_cell(
    field: jnp.ndarray,
    *,
    open_x: bool = False,
) -> jnp.ndarray:
    """Interpolate to x-z edges, with zero wall values."""
    if open_x:
        faces = _cells_to_open_x_faces(field)
        interior = 0.5 * (faces[:-1] + faces[1:])
        wall = jnp.zeros_like(faces[:1])
    else:
        rolled = jnp.roll(field, 1, axis=2)
        interior = 0.25 * (field[:-1] + field[1:] + rolled[:-1] + rolled[1:])
        wall = jnp.zeros_like(field[:1])
    return jnp.concatenate((wall, interior, wall), axis=0)


def _to_yz_edge_from_cell(field: jnp.ndarray, *, wall_y: bool = False) -> jnp.ndarray:
    z_faces = 0.5 * (field[:-1] + field[1:])
    if wall_y:
        interior_y = 0.5 * (z_faces[:, :-1] + z_faces[:, 1:])
        side = jnp.zeros_like(z_faces[:, :1])
        z_faces = jnp.concatenate((side, interior_y, side), axis=1)
    else:
        z_faces = 0.5 * (z_faces + jnp.roll(z_faces, 1, axis=1))
    wall = jnp.zeros_like(z_faces[:1])
    return jnp.concatenate((wall, z_faces, wall), axis=0)


def _cell_x_derivative_to_open_faces(field: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    distance = center_distances(grid, 2, periodic=False, dtype=field.dtype)
    interior = (field[..., 1:] - field[..., :-1]) / distance[
        None, None, 1:-1
    ]
    zero = jnp.zeros_like(field[..., :1])
    return jnp.concatenate((zero, interior, zero), axis=2)


def edge_gradients(
    velocity: StaggeredVelocity,
    grid: Grid,
    boundaries: Boundaries,
) -> EdgeGradients:
    """Return every velocity gradient at its natural staggered location."""
    x_velocity, y_velocity, z_velocity = velocity
    periodic_x = streamwise_is_periodic(velocity, grid)
    periodic_y = spanwise_is_periodic(velocity, grid)
    if periodic_x:
        xx = (jnp.roll(x_velocity, -1, axis=2) - x_velocity) / shaped_widths(
            grid, 2, x_velocity.dtype
        )
        x_distance = shaped_center_distances(
            grid, 2, periodic=True, dtype=y_velocity.dtype
        )
        yx = (y_velocity - jnp.roll(y_velocity, 1, axis=2)) / x_distance
        zx = (z_velocity - jnp.roll(z_velocity, 1, axis=2)) / x_distance
    else:
        xx = (x_velocity[..., 1:] - x_velocity[..., :-1]) / shaped_widths(
            grid, 2, x_velocity.dtype
        )
        yx = _cell_x_derivative_to_open_faces(y_velocity, grid)
        zx = _cell_x_derivative_to_open_faces(z_velocity, grid)
    if periodic_y:
        yy = (jnp.roll(y_velocity, -1, axis=1) - y_velocity) / shaped_widths(
            grid, 1, y_velocity.dtype
        )
        y_distance = shaped_center_distances(
            grid, 1, periodic=True, dtype=x_velocity.dtype
        )
        xy = (x_velocity - jnp.roll(x_velocity, 1, axis=1)) / y_distance
        zy = (z_velocity - jnp.roll(z_velocity, 1, axis=1)) / y_distance
    else:
        yy = (y_velocity[:, 1:] - y_velocity[:, :-1]) / shaped_widths(
            grid, 1, y_velocity.dtype
        )
        y_distance = center_distances(
            grid, 1, periodic=False, dtype=x_velocity.dtype
        )[None, 1:-1, None]
        side_x = jnp.zeros_like(x_velocity[:, :1])
        xy = jnp.concatenate(
            (
                side_x,
                (x_velocity[:, 1:] - x_velocity[:, :-1]) / y_distance,
                side_x,
            ),
            axis=1,
        )
        side_z = jnp.zeros_like(z_velocity[:, :1])
        zy = jnp.concatenate(
            (
                side_z,
                (z_velocity[:, 1:] - z_velocity[:, :-1]) / y_distance,
                side_z,
            ),
            axis=1,
        )
    return dict(
        xx=xx,
        yy=yy,
        zz=(z_velocity[1:] - z_velocity[:-1])
        / shaped_widths(grid, 0, z_velocity.dtype),
        xy=xy,
        yx=yx,
        xz=tangential_z_gradient(
            x_velocity, grid, boundaries, "x_velocity"
        ),
        zx=zx,
        yz=tangential_z_gradient(
            y_velocity, grid, boundaries, "y_velocity"
        ),
        zy=zy,
    )


def cell_gradients(gradients: EdgeGradients) -> list[list[jnp.ndarray]]:
    """Collect the full gradient tensor at cell centres as ``g[i][k]``."""
    open_x = gradients["xy"].shape[-1] == gradients["xx"].shape[-1] + 1
    wall_y = gradients["xy"].shape[1] == gradients["xx"].shape[1] + 1
    return [
        [
            gradients["xx"],
            _to_cell_from_xy_edge(gradients["xy"], open_x=open_x, wall_y=wall_y),
            _to_cell_from_xz_edge(gradients["xz"], open_x=open_x),
        ],
        [
            _to_cell_from_xy_edge(gradients["yx"], open_x=open_x, wall_y=wall_y),
            gradients["yy"],
            _to_cell_from_yz_edge(gradients["yz"], wall_y=wall_y),
        ],
        [
            _to_cell_from_xz_edge(gradients["zx"], open_x=open_x),
            _to_cell_from_yz_edge(gradients["zy"], wall_y=wall_y),
            gradients["zz"],
        ],
    ]


def eddy_viscosity(
    velocity: StaggeredVelocity,
    grid: Grid,
    boundaries: Boundaries,
    model: AnisotropicMinimumDissipation | StaticSmagorinsky,
    *,
    gradients: EdgeGradients | None = None,
) -> jnp.ndarray:
    """Return the selected cell-centred, non-negative eddy viscosity."""
    if gradients is None:
        gradients = edge_gradients(velocity, grid, boundaries)
    tensor = cell_gradients(gradients)
    strain = [
        [0.5 * (tensor[i][k] + tensor[k][i]) for k in range(3)] for i in range(3)
    ]
    if isinstance(model, StaticSmagorinsky):
        strain_magnitude_squared = jnp.zeros_like(tensor[0][0])
        for i in range(3):
            for j in range(3):
                strain_magnitude_squared = (
                    strain_magnitude_squared + 2.0 * strain[i][j] ** 2
                )
        if isinstance(model, FluentSmagorinsky):
            return model.length_scale(grid, tensor[0][0].dtype)**2 * jnp.sqrt(
                strain_magnitude_squared)
        filter_width = cell_volumes(grid, tensor[0][0].dtype) ** (1.0 / 3.0)
        return (
            model.coefficient * filter_width
        ) ** 2 * jnp.sqrt(strain_magnitude_squared + 1.0e-20)
    widths = [
        model.poincare_constant
        * shaped_widths(grid, axis, tensor[0][0].dtype) ** 2
        for axis in (2, 1, 0)
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
    grid: Grid,
    boundaries: Boundaries,
    *,
    gradients: EdgeGradients | None = None,
) -> StaggeredVelocity:
    """Tendency ``d_j (2 nu_e S_ij)`` on mapped control volumes."""
    if gradients is None:
        gradients = edge_gradients(velocity, grid, boundaries)
    open_x = not streamwise_is_periodic(velocity, grid)
    wall_y = not spanwise_is_periodic(velocity, grid)
    normal_x = 2.0 * viscosity * gradients["xx"]
    normal_y = 2.0 * viscosity * gradients["yy"]
    normal_z = 2.0 * viscosity * gradients["zz"]
    shear_xy = (
        2.0
        * _to_xy_edge_from_cell(viscosity, open_x=open_x, wall_y=wall_y)
        * 0.5
        * (gradients["xy"] + gradients["yx"])
    )
    shear_xz = (
        2.0
        * _to_xz_edge_from_cell(viscosity, open_x=open_x)
        * 0.5
        * (gradients["xz"] + gradients["zx"])
    )
    shear_yz = (
        2.0
        * _to_yz_edge_from_cell(viscosity, wall_y=wall_y)
        * 0.5
        * (gradients["yz"] + gradients["zy"])
    )
    if open_x:
        edge = jnp.zeros_like(velocity.x[..., :1])
        x_distance = center_distances(
            grid, 2, periodic=False, dtype=normal_x.dtype
        )
        normal_x_divergence = jnp.concatenate(
            (
                edge,
                (normal_x[..., 1:] - normal_x[..., :-1])
                / x_distance[None, None, 1:-1],
                edge,
            ),
            axis=2,
        )
        shear_xy_divergence = (
            shear_xy[..., 1:] - shear_xy[..., :-1]
        ) / shaped_widths(grid, 2, shear_xy.dtype)
        shear_xz_divergence = (
            shear_xz[..., 1:] - shear_xz[..., :-1]
        ) / shaped_widths(grid, 2, shear_xz.dtype)
    else:
        x_distance = shaped_center_distances(
            grid, 2, periodic=True, dtype=normal_x.dtype
        )
        normal_x_divergence = (
            normal_x - jnp.roll(normal_x, 1, axis=2)
        ) / x_distance
        shear_xy_divergence = (
            jnp.roll(shear_xy, -1, axis=2) - shear_xy
        ) / shaped_widths(grid, 2, shear_xy.dtype)
        shear_xz_divergence = (
            jnp.roll(shear_xz, -1, axis=2) - shear_xz
        ) / shaped_widths(grid, 2, shear_xz.dtype)
    if wall_y:
        shear_xy_y_divergence = (
            shear_xy[:, 1:] - shear_xy[:, :-1]
        ) / shaped_widths(grid, 1, shear_xy.dtype)
        side = jnp.zeros_like(velocity.y[:, :1])
        y_distance = center_distances(
            grid, 1, periodic=False, dtype=normal_y.dtype
        )
        normal_y_y_divergence = jnp.concatenate(
            (
                side,
                (normal_y[:, 1:] - normal_y[:, :-1])
                / y_distance[None, 1:-1, None],
                side,
            ),
            axis=1,
        )
        shear_yz_y_divergence = (
            shear_yz[:, 1:] - shear_yz[:, :-1]
        ) / shaped_widths(grid, 1, shear_yz.dtype)
    else:
        y_distance = shaped_center_distances(
            grid, 1, periodic=True, dtype=normal_y.dtype
        )
        shear_xy_y_divergence = (
            jnp.roll(shear_xy, -1, axis=1) - shear_xy
        ) / shaped_widths(grid, 1, shear_xy.dtype)
        normal_y_y_divergence = (
            normal_y - jnp.roll(normal_y, 1, axis=1)
        ) / y_distance
        shear_yz_y_divergence = (
            jnp.roll(shear_yz, -1, axis=1) - shear_yz
        ) / shaped_widths(grid, 1, shear_yz.dtype)
    z_width = shaped_widths(grid, 0, shear_xz.dtype)
    x_tendency = (
        normal_x_divergence
        + shear_xy_y_divergence
        + (shear_xz[1:] - shear_xz[:-1]) / z_width
    )
    y_tendency = (
        shear_xy_divergence
        + normal_y_y_divergence
        + (shear_yz[1:] - shear_yz[:-1]) / z_width
    )
    z_distance = center_distances(
        grid, 0, periodic=False, dtype=normal_z.dtype
    )[1:-1, None, None]
    z_interior = (
        shear_xz_divergence[1:-1]
        + shear_yz_y_divergence[1:-1]
        + (normal_z[1:] - normal_z[:-1]) / z_distance
    )
    if wall_y:
        y_tendency = y_tendency.at[:, 0].set(0.0).at[:, -1].set(0.0)
    wall = jnp.zeros_like(velocity.z[:1])
    return StaggeredVelocity(
        x_tendency,
        y_tendency,
        jnp.concatenate((wall, z_interior, wall), axis=0),
    )


def subfilter_tendency(
    velocity: StaggeredVelocity,
    grid: Grid,
    boundaries: Boundaries,
    model: AnisotropicMinimumDissipation | StaticSmagorinsky,
    *,
    surface=None,
    mesh_stability=0.0,
) -> tuple[StaggeredVelocity, jnp.ndarray]:
    """Return the subfilter momentum tendency and its eddy viscosity.

    ``surface`` optionally supplies the wall model.  When it asks for the
    logarithmic gradient correction the wall-normal shear the closure sees is
    adjusted before the stress is formed; the momentum boundary condition
    itself is unchanged and still comes from the wall model.
    """
    gradients = edge_gradients(velocity, grid, boundaries)
    if surface is not None and getattr(surface, "gradient_correction", False):
        from .wall import log_law_gradient_correction

        gradients = log_law_gradient_correction(
            gradients, velocity, grid, surface, mesh_stability=mesh_stability
        )
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
    "StaticSmagorinsky",
    "EdgeGradients",
    "cell_gradients",
    "eddy_viscosity",
    "edge_gradients",
    "stress_divergence",
    "subfilter_tendency",
]
