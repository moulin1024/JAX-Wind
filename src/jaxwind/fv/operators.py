"""Second-order finite-volume operators on the staggered Cartesian mesh.

Every operator here is the conservative (flux-difference) form on control
volumes centred on the unknown it advances, so momentum telescopes exactly
across the periodic directions and across the impermeable walls.  The discrete
divergence and the discrete pressure gradient are exact negative adjoints,
which is what makes the projection in :mod:`jaxwind.fv.poisson` remove
divergence to round-off.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxwind.domain.grid import UniformGrid

from .state import (
    FREE_SLIP,
    Boundaries,
    StaggeredVelocity,
    Wall,
    streamwise_is_periodic,
)


def divergence(velocity: StaggeredVelocity, grid: UniformGrid) -> jnp.ndarray:
    """Cell-centred divergence of the face-normal velocity."""
    if streamwise_is_periodic(velocity, grid):
        x_divergence = (
            jnp.roll(velocity.x, -1, axis=2) - velocity.x
        ) / grid.dx
    else:
        x_divergence = (velocity.x[..., 1:] - velocity.x[..., :-1]) / grid.dx
    return (
        x_divergence
        + (jnp.roll(velocity.y, -1, axis=1) - velocity.y) / grid.dy
        + (velocity.z[1:] - velocity.z[:-1]) / grid.dz
    )


def pressure_gradient(
    pressure: jnp.ndarray,
    grid: UniformGrid,
    *,
    periodic_x: bool = True,
) -> StaggeredVelocity:
    """Face-normal gradient of a cell-centred field."""
    if periodic_x:
        x_gradient = (pressure - jnp.roll(pressure, 1, axis=2)) / grid.dx
    else:
        inlet = jnp.zeros_like(pressure[..., :1])
        interior = (pressure[..., 1:] - pressure[..., :-1]) / grid.dx
        outlet = -2.0 * pressure[..., -1:] / grid.dx
        x_gradient = jnp.concatenate((inlet, interior, outlet), axis=2)
    y_gradient = (pressure - jnp.roll(pressure, 1, axis=1)) / grid.dy
    wall = jnp.zeros_like(pressure[:1])
    z_gradient = jnp.concatenate(
        (wall, (pressure[1:] - pressure[:-1]) / grid.dz, wall),
        axis=0,
    )
    if not periodic_x:
        y_gradient = y_gradient.at[..., 0].set(0.0).at[..., -1].set(0.0)
        z_gradient = z_gradient.at[..., 0].set(0.0).at[..., -1].set(0.0)
    return StaggeredVelocity(x_gradient, y_gradient, z_gradient)


def _tangential_ghost(
    first: jnp.ndarray,
    second: jnp.ndarray,
    wall: Wall,
    wall_velocity: float,
) -> jnp.ndarray:
    """Return the value one cell outside a wall.

    A free-slip wall mirrors the adjacent cell, which makes the wall-tangential
    stress vanish exactly.  A no-slip or moving wall uses the quadratic through
    the wall value and the first two cell centres, so the wall stress is
    second-order accurate.  The plain mirror value would place a first-order
    error in the wall stress and cost the solver an order of accuracy in
    exactly the cells where a wall-bounded flow needs it most; the quadratic
    closure instead reproduces a parabolic profile exactly.
    """
    if wall.kind == FREE_SLIP:
        return first
    velocity = jnp.asarray(wall_velocity, first.dtype)
    if second is None:
        return 2.0 * velocity - first
    return (8.0 * velocity - 6.0 * first + second) / 3.0


def tangential_z_gradient(
    field: jnp.ndarray,
    grid: UniformGrid,
    boundaries: Boundaries,
    wall_velocity: str,
) -> jnp.ndarray:
    """z-derivative of a tangential component on all ``nz + 1`` z-faces.

    Every wall-normal flux in the solver -- viscous and subfilter alike -- is
    built from this one array, so the wall closure cannot disagree between
    them.
    """
    deep_enough = field.shape[0] > 1
    lower = _tangential_ghost(
        field[:1],
        field[1:2] if deep_enough else None,
        boundaries.lower,
        getattr(boundaries.lower, wall_velocity),
    )
    upper = _tangential_ghost(
        field[-1:],
        field[-2:-1] if deep_enough else None,
        boundaries.upper,
        getattr(boundaries.upper, wall_velocity),
    )
    padded = jnp.concatenate((lower, field, upper), axis=0)
    return (padded[1:] - padded[:-1]) / grid.dz


def _tangential_z_curvature(
    field: jnp.ndarray,
    grid: UniformGrid,
    boundaries: Boundaries,
    wall_velocity: str,
) -> jnp.ndarray:
    """Second z-derivative of a z-cell-centred tangential component."""
    gradient = tangential_z_gradient(field, grid, boundaries, wall_velocity)
    return (gradient[1:] - gradient[:-1]) / grid.dz


def _horizontal_curvature(
    field: jnp.ndarray,
    grid: UniformGrid,
    *,
    periodic_x: bool,
) -> jnp.ndarray:
    if periodic_x:
        x_curvature = (
            jnp.roll(field, -1, axis=2)
            - 2.0 * field
            + jnp.roll(field, 1, axis=2)
        ) / grid.dx**2
    else:
        interior = (
            field[..., 2:] - 2.0 * field[..., 1:-1] + field[..., :-2]
        ) / grid.dx**2
        edge = jnp.zeros_like(field[..., :1])
        x_curvature = jnp.concatenate((edge, interior, edge), axis=2)
    return x_curvature + (
        jnp.roll(field, -1, axis=1) - 2.0 * field + jnp.roll(field, 1, axis=1)
    ) / grid.dy**2


def diffusion(
    velocity: StaggeredVelocity,
    grid: UniformGrid,
    boundaries: Boundaries,
    viscosity: float,
) -> StaggeredVelocity:
    """Viscous tendency ``nu * laplacian(u)`` with wall mirror closures."""
    nu = jnp.asarray(viscosity, velocity.x.dtype)
    periodic_x = streamwise_is_periodic(velocity, grid)
    x_tendency = nu * (
        _horizontal_curvature(velocity.x, grid, periodic_x=periodic_x)
        + _tangential_z_curvature(velocity.x, grid, boundaries, "x_velocity")
    )
    y_tendency = nu * (
        _horizontal_curvature(velocity.y, grid, periodic_x=periodic_x)
        + _tangential_z_curvature(velocity.y, grid, boundaries, "y_velocity")
    )
    interior = velocity.z[1:-1]
    z_curvature = (
        velocity.z[2:] - 2.0 * interior + velocity.z[:-2]
    ) / grid.dz**2
    z_interior = nu * (
        _horizontal_curvature(interior, grid, periodic_x=periodic_x) + z_curvature
    )
    wall = jnp.zeros_like(velocity.z[:1])
    return StaggeredVelocity(
        x_tendency,
        y_tendency,
        jnp.concatenate((wall, z_interior, wall), axis=0),
    )


def _cells_to_open_x_faces(field: jnp.ndarray) -> jnp.ndarray:
    """Interpolate x-cell values to distinct inlet/interior/outlet faces."""
    interior = 0.5 * (field[..., :-1] + field[..., 1:])
    return jnp.concatenate((field[..., :1], interior, field[..., -1:]), axis=2)


def _edge_fluxes(
    velocity: StaggeredVelocity,
    grid: UniformGrid,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Momentum fluxes shared by the staggered control volumes."""
    x_velocity, y_velocity, z_velocity = velocity
    periodic_x = streamwise_is_periodic(velocity, grid)
    if periodic_x:
        y_on_x_faces = 0.5 * (y_velocity + jnp.roll(y_velocity, 1, axis=2))
        z_on_x_faces = 0.5 * (
            z_velocity[1:-1] + jnp.roll(z_velocity[1:-1], 1, axis=2)
        )
        wall_x = jnp.zeros_like(z_velocity[:1])
    else:
        y_on_x_faces = _cells_to_open_x_faces(y_velocity)
        z_on_x_faces = _cells_to_open_x_faces(z_velocity[1:-1])
        wall_x = jnp.zeros_like(x_velocity[:1])
    xy_edge = 0.5 * (
        x_velocity + jnp.roll(x_velocity, 1, axis=1)
    ) * y_on_x_faces
    xz_edge = jnp.concatenate(
        (
            wall_x,
            0.5 * (x_velocity[:-1] + x_velocity[1:]) * z_on_x_faces,
            wall_x,
        ),
        axis=0,
    )
    wall = jnp.zeros_like(z_velocity[:1])
    yz_edge = jnp.concatenate(
        (
            wall,
            0.5 * (y_velocity[:-1] + y_velocity[1:])
            * 0.5
            * (z_velocity[1:-1] + jnp.roll(z_velocity[1:-1], 1, axis=1)),
            wall,
        ),
        axis=0,
    )
    return xy_edge, xz_edge, yz_edge


def advection(velocity: StaggeredVelocity, grid: UniformGrid) -> StaggeredVelocity:
    """Convective momentum tendency in conservative flux form."""
    x_velocity, y_velocity, z_velocity = velocity
    periodic_x = streamwise_is_periodic(velocity, grid)
    xy_edge, xz_edge, yz_edge = _edge_fluxes(velocity, grid)

    if periodic_x:
        xx_cell = (
            0.5 * (x_velocity + jnp.roll(x_velocity, -1, axis=2))
        ) ** 2
        xx_divergence = (xx_cell - jnp.roll(xx_cell, 1, axis=2)) / grid.dx
        xy_x_divergence = (
            jnp.roll(xy_edge, -1, axis=2) - xy_edge
        ) / grid.dx
        xz_x_divergence = (
            jnp.roll(xz_edge, -1, axis=2) - xz_edge
        ) / grid.dx
    else:
        xx_cell = (0.5 * (x_velocity[..., :-1] + x_velocity[..., 1:])) ** 2
        interior = (xx_cell[..., 1:] - xx_cell[..., :-1]) / grid.dx
        edge = jnp.zeros_like(x_velocity[..., :1])
        xx_divergence = jnp.concatenate((edge, interior, edge), axis=2)
        xy_x_divergence = (xy_edge[..., 1:] - xy_edge[..., :-1]) / grid.dx
        xz_x_divergence = (xz_edge[..., 1:] - xz_edge[..., :-1]) / grid.dx
    yy_cell = (0.5 * (y_velocity + jnp.roll(y_velocity, -1, axis=1))) ** 2
    zz_cell = (0.5 * (z_velocity[:-1] + z_velocity[1:])) ** 2

    x_flux_divergence = (
        xx_divergence
        + (jnp.roll(xy_edge, -1, axis=1) - xy_edge) / grid.dy
        + (xz_edge[1:] - xz_edge[:-1]) / grid.dz
    )
    y_flux_divergence = (
        xy_x_divergence
        + (yy_cell - jnp.roll(yy_cell, 1, axis=1)) / grid.dy
        + (yz_edge[1:] - yz_edge[:-1]) / grid.dz
    )
    z_flux_divergence = (
        xz_x_divergence[1:-1]
        + (jnp.roll(yz_edge, -1, axis=1) - yz_edge)[1:-1] / grid.dy
        + (zz_cell[1:] - zz_cell[:-1]) / grid.dz
    )
    wall = jnp.zeros_like(z_velocity[:1])
    return StaggeredVelocity(
        -x_flux_divergence,
        -y_flux_divergence,
        -jnp.concatenate((wall, z_flux_divergence, wall), axis=0),
    )


def cell_velocity(
    velocity: StaggeredVelocity,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interpolate the staggered components to cell centres for diagnostics."""
    x_velocity = (
        0.5 * (velocity.x + jnp.roll(velocity.x, -1, axis=2))
        if velocity.x.shape[-1] == velocity.y.shape[-1]
        else 0.5 * (velocity.x[..., :-1] + velocity.x[..., 1:])
    )
    return (
        x_velocity,
        0.5 * (velocity.y + jnp.roll(velocity.y, -1, axis=1)),
        0.5 * (velocity.z[:-1] + velocity.z[1:]),
    )


def kinetic_energy(velocity: StaggeredVelocity, grid: UniformGrid) -> jnp.ndarray:
    """Volume-integrated kinetic energy of the staggered field."""
    interior = velocity.z[1:-1]
    volume = grid.dx * grid.dy * grid.dz
    return 0.5 * volume * (
        jnp.sum(velocity.x**2)
        + jnp.sum(velocity.y**2)
        + 0.5 * jnp.sum(velocity.z[0] ** 2 + velocity.z[-1] ** 2)
        + jnp.sum(interior**2)
    )


def courant_number(
    velocity: StaggeredVelocity,
    grid: UniformGrid,
    dt: float,
) -> jnp.ndarray:
    """Convective Courant number of a step."""
    return dt * (
        jnp.max(jnp.abs(velocity.x)) / grid.dx
        + jnp.max(jnp.abs(velocity.y)) / grid.dy
        + jnp.max(jnp.abs(velocity.z)) / grid.dz
    )


def stable_timestep(
    velocity: StaggeredVelocity,
    grid: UniformGrid,
    viscosity: float,
    *,
    courant: float = 0.5,
    diffusion_number: float = 0.25,
) -> jnp.ndarray:
    """Largest step satisfying both the convective and viscous limits.

    ``viscosity`` may be an array, in which case its largest value is used;
    pass the molecular viscosity plus the subfilter viscosity to bound a
    large-eddy simulation step.
    """
    speed = (
        jnp.max(jnp.abs(velocity.x)) / grid.dx
        + jnp.max(jnp.abs(velocity.y)) / grid.dy
        + jnp.max(jnp.abs(velocity.z)) / grid.dz
    )
    # The wall rows of the viscous operator carry twice the interior diagonal
    # because the quadratic wall closure reaches the wall itself, so the
    # vertical direction is weighted accordingly.
    inverse_squares = (
        1.0 / grid.dx**2 + 1.0 / grid.dy**2 + 2.0 / grid.dz**2
    )
    largest = jnp.max(jnp.asarray(viscosity))
    convective = jnp.where(speed > 0.0, courant / jnp.maximum(speed, 1e-300), jnp.inf)
    viscous = jnp.where(
        largest > 0.0,
        diffusion_number / jnp.maximum(largest * inverse_squares, 1e-300),
        jnp.inf,
    )
    return jnp.minimum(convective, viscous)


__all__ = [
    "advection",
    "cell_velocity",
    "courant_number",
    "diffusion",
    "divergence",
    "kinetic_energy",
    "pressure_gradient",
    "stable_timestep",
    "tangential_z_gradient",
]
