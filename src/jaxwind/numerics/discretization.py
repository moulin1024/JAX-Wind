"""Second-order finite-volume operators on a rectilinear MAC mesh.

Flux differences are divided by the physical control-volume widths sampled
from the analytical mesh mapping. On a mapped mesh divergence and pressure
gradient are adjoints in the natural volume/face-area inner products; their
composition remains the compact seven-point pressure operator.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxwind.domain.grid import Grid

from jaxwind.metrics import center_distances, shaped_center_distances, shaped_widths, widths
from jaxwind.state import (
    FREE_SLIP,
    Boundaries,
    StaggeredVelocity,
    Wall,
    spanwise_is_periodic,
    streamwise_is_periodic,
)


def _axis_shape(values: jnp.ndarray, axis: int, ndim: int = 3) -> jnp.ndarray:
    shape = [1] * ndim
    shape[axis] = values.size
    return values.reshape(shape)


def _axis_slice(ndim: int, axis: int, start, stop=None) -> tuple[slice, ...]:
    index = [slice(None)] * ndim
    index[axis] = slice(start, stop)
    return tuple(index)


def _cells_to_faces(
    field: jnp.ndarray,
    grid: Grid,
    axis: int,
    *,
    periodic: bool,
    boundary: str = "copy",
) -> jnp.ndarray:
    """Linearly interpolate cell values to physical faces along one axis."""
    cell_widths = widths(grid, axis, field.dtype)
    if periodic:
        lower = jnp.roll(field, 1, axis=axis)
        lower_width = jnp.roll(cell_widths, 1)
        total = lower_width + cell_widths
        lower_weight = _axis_shape(cell_widths / total, axis, field.ndim)
        upper_weight = _axis_shape(lower_width / total, axis, field.ndim)
        return lower_weight * lower + upper_weight * field
    lower_index = _axis_slice(field.ndim, axis, 0, -1)
    upper_index = _axis_slice(field.ndim, axis, 1, None)
    lower, upper = field[lower_index], field[upper_index]
    total = cell_widths[:-1] + cell_widths[1:]
    lower_weight = _axis_shape(cell_widths[1:] / total, axis, field.ndim)
    upper_weight = _axis_shape(cell_widths[:-1] / total, axis, field.ndim)
    interior = lower_weight * lower + upper_weight * upper
    edge_index = _axis_slice(field.ndim, axis, 0, 1)
    last_index = _axis_slice(field.ndim, axis, -1, None)
    if boundary == "copy":
        lower_edge, upper_edge = field[edge_index], field[last_index]
    elif boundary == "zero":
        lower_edge = jnp.zeros_like(field[edge_index])
        upper_edge = jnp.zeros_like(field[last_index])
    else:
        raise ValueError(f"unsupported face boundary interpolation: {boundary!r}")
    return jnp.concatenate((lower_edge, interior, upper_edge), axis=axis)


def divergence(velocity: StaggeredVelocity, grid: Grid) -> jnp.ndarray:
    """Cell-centred divergence of the face-normal velocity."""
    if streamwise_is_periodic(velocity, grid):
        x_difference = jnp.roll(velocity.x, -1, axis=2) - velocity.x
    else:
        x_difference = velocity.x[..., 1:] - velocity.x[..., :-1]
    if spanwise_is_periodic(velocity, grid):
        y_difference = jnp.roll(velocity.y, -1, axis=1) - velocity.y
    else:
        y_difference = velocity.y[:, 1:] - velocity.y[:, :-1]
    return (
        x_difference / shaped_widths(grid, 2, velocity.x.dtype)
        + y_difference / shaped_widths(grid, 1, velocity.y.dtype)
        + (velocity.z[1:] - velocity.z[:-1])
        / shaped_widths(grid, 0, velocity.z.dtype)
    )


def pressure_gradient(
    pressure: jnp.ndarray,
    grid: Grid,
    *,
    periodic_x: bool = True,
    periodic_y: bool = True,
    open_x_low: bool = False,
    open_y: bool = False,
    physical_transverse_inlet: bool = False,
) -> StaggeredVelocity:
    """Face-normal pressure gradient with optional physical transverse inlet.

    The opt-in uniform open-x mode releases transverse gradients in the first
    x-cell column. Inlet normal correction stays zero; the last-column
    transverse constraint remains unchanged. The matching Poisson operator
    must use the same option.
    """
    if physical_transverse_inlet and (
        periodic_x or open_x_low or open_y or not grid.is_uniform
    ):
        raise ValueError(
            "physical transverse inlet requires a uniform fixed-flux x inlet "
            "without lateral pressure outlets"
        )
    if periodic_x:
        x_gradient = (pressure - jnp.roll(pressure, 1, axis=2)) / (
            shaped_center_distances(grid, 2, periodic=True, dtype=pressure.dtype)
        )
    else:
        distances = center_distances(grid, 2, periodic=False, dtype=pressure.dtype)
        inlet = (
            pressure[..., :1] / distances[0]
            if open_x_low else jnp.zeros_like(pressure[..., :1])
        )
        interior = (pressure[..., 1:] - pressure[..., :-1]) / distances[
            None, None, 1:-1
        ]
        outlet = -pressure[..., -1:] / distances[-1]
        x_gradient = jnp.concatenate((inlet, interior, outlet), axis=2)
    if periodic_y:
        y_gradient = (pressure - jnp.roll(pressure, 1, axis=1)) / (
            shaped_center_distances(grid, 1, periodic=True, dtype=pressure.dtype)
        )
    else:
        distances_y = center_distances(grid, 1, periodic=False, dtype=pressure.dtype)
        side = jnp.zeros_like(pressure[:, :1])
        interior_y = (pressure[:, 1:] - pressure[:, :-1]) / distances_y[
            None, 1:-1, None
        ]
        low_y = pressure[:, :1] / distances_y[0] if open_y else side
        high_y = -pressure[:, -1:] / distances_y[-1] if open_y else side
        y_gradient = jnp.concatenate((low_y, interior_y, high_y), axis=1)
    distances_z = center_distances(grid, 0, periodic=False, dtype=pressure.dtype)
    wall = jnp.zeros_like(pressure[:1])
    z_gradient = jnp.concatenate(
        (
            wall,
            (pressure[1:] - pressure[:-1]) / distances_z[1:-1, None, None],
            wall,
        ),
        axis=0,
    )
    if not periodic_x:
        if not physical_transverse_inlet:
            y_gradient = y_gradient.at[..., 0].set(0.0)
            z_gradient = z_gradient.at[..., 0].set(0.0)
        # On a one-column coarse grid the retained outlet constraint wins.
        y_gradient = y_gradient.at[..., -1].set(0.0)
        z_gradient = z_gradient.at[..., -1].set(0.0)
    return StaggeredVelocity(x_gradient, y_gradient, z_gradient)


def _quadratic_wall_ghost_coefficients(
    first_width: float,
    second_width: float | None,
) -> tuple[float, float, float]:
    if second_width is None:
        return 2.0, -1.0, 0.0
    first_center = 0.5 * first_width
    center_distance = 0.5 * (first_width + second_width)
    second_center = first_center + center_distance
    wall = 2.0 * (first_center + second_center) / second_center
    first = -(first_center + second_center) / center_distance
    second = 2.0 * first_center**2 / (second_center * center_distance)
    return wall, first, second


def _tangential_ghost(
    first: jnp.ndarray,
    second: jnp.ndarray | None,
    wall: Wall,
    wall_velocity: float,
    first_width: float,
    second_width: float | None,
) -> jnp.ndarray:
    """Return the value at the reflected centre outside a physical wall."""
    if wall.kind == FREE_SLIP:
        return first
    velocity = jnp.asarray(wall_velocity, first.dtype)
    wall_coefficient, first_coefficient, second_coefficient = (
        _quadratic_wall_ghost_coefficients(first_width, second_width)
    )
    if second is None:
        return wall_coefficient * velocity + first_coefficient * first
    return (
        wall_coefficient * velocity
        + first_coefficient * first
        + second_coefficient * second
    )


def tangential_z_gradient(
    field: jnp.ndarray,
    grid: Grid,
    boundaries: Boundaries,
    wall_velocity: str,
) -> jnp.ndarray:
    """z derivative of a tangential component on all ``nz + 1`` faces."""
    deep_enough = field.shape[0] > 1
    z_widths = grid.z_widths
    lower = _tangential_ghost(
        field[:1],
        field[1:2] if deep_enough else None,
        boundaries.lower,
        getattr(boundaries.lower, wall_velocity),
        float(z_widths[0]),
        float(z_widths[1]) if deep_enough else None,
    )
    upper = _tangential_ghost(
        field[-1:],
        field[-2:-1] if deep_enough else None,
        boundaries.upper,
        getattr(boundaries.upper, wall_velocity),
        float(z_widths[-1]),
        float(z_widths[-2]) if deep_enough else None,
    )
    padded = jnp.concatenate((lower, field, upper), axis=0)
    interior = center_distances(grid, 0, periodic=False, dtype=field.dtype)[1:-1]
    distances = jnp.concatenate(
        (
            jnp.asarray(z_widths[:1], field.dtype),
            interior,
            jnp.asarray(z_widths[-1:], field.dtype),
        )
    )
    return (padded[1:] - padded[:-1]) / distances[:, None, None]


def _tangential_z_curvature(
    field: jnp.ndarray,
    grid: Grid,
    boundaries: Boundaries,
    wall_velocity: str,
) -> jnp.ndarray:
    gradient = tangential_z_gradient(field, grid, boundaries, wall_velocity)
    return (gradient[1:] - gradient[:-1]) / shaped_widths(
        grid, 0, field.dtype
    )


def _cell_curvature(
    field: jnp.ndarray,
    grid: Grid,
    axis: int,
    *,
    periodic: bool,
    zero_edge_tendency: bool = False,
) -> jnp.ndarray:
    """Second derivative of values stored at cell centres."""
    if periodic:
        distance = shaped_center_distances(
            grid, axis, periodic=True, dtype=field.dtype
        )
        lower_gradient = (field - jnp.roll(field, 1, axis=axis)) / distance
        curvature = (jnp.roll(lower_gradient, -1, axis=axis) - lower_gradient) / (
            shaped_widths(grid, axis, field.dtype)
        )
        return curvature
    distance = center_distances(grid, axis, periodic=False, dtype=field.dtype)
    lower_index = _axis_slice(field.ndim, axis, 0, -1)
    upper_index = _axis_slice(field.ndim, axis, 1, None)
    gradient = (field[upper_index] - field[lower_index]) / _axis_shape(
        distance[1:-1], axis, field.ndim
    )
    edge_index = _axis_slice(field.ndim, axis, 0, 1)
    zero = jnp.zeros_like(field[edge_index])
    flux = jnp.concatenate((zero, gradient, zero), axis=axis)
    upper_flux = flux[_axis_slice(flux.ndim, axis, 1, None)]
    lower_flux = flux[_axis_slice(flux.ndim, axis, 0, -1)]
    curvature = (upper_flux - lower_flux) / shaped_widths(
        grid, axis, field.dtype
    )
    if zero_edge_tendency:
        curvature = curvature.at[_axis_slice(field.ndim, axis, 0, 1)].set(0.0)
        curvature = curvature.at[_axis_slice(field.ndim, axis, -1, None)].set(0.0)
    return curvature


def _normal_face_curvature(
    field: jnp.ndarray,
    grid: Grid,
    axis: int,
    *,
    periodic: bool,
) -> jnp.ndarray:
    """Second derivative where the component lives on its normal faces."""
    cell_width = shaped_widths(grid, axis, field.dtype)
    if periodic:
        gradient = (jnp.roll(field, -1, axis=axis) - field) / cell_width
        return (gradient - jnp.roll(gradient, 1, axis=axis)) / (
            shaped_center_distances(grid, axis, periodic=True, dtype=field.dtype)
        )
    upper = field[_axis_slice(field.ndim, axis, 1, None)]
    lower = field[_axis_slice(field.ndim, axis, 0, -1)]
    gradient = (upper - lower) / cell_width
    interior = (
        gradient[_axis_slice(field.ndim, axis, 1, None)]
        - gradient[_axis_slice(field.ndim, axis, 0, -1)]
    ) / _axis_shape(
        center_distances(grid, axis, periodic=False, dtype=field.dtype)[1:-1],
        axis,
        field.ndim,
    )
    edge = jnp.zeros_like(field[_axis_slice(field.ndim, axis, 0, 1)])
    return jnp.concatenate((edge, interior, edge), axis=axis)


def diffusion(
    velocity: StaggeredVelocity,
    grid: Grid,
    boundaries: Boundaries,
    viscosity: float,
) -> StaggeredVelocity:
    """Viscous tendency ``nu * laplacian(u)`` on mapped control volumes."""
    nu = jnp.asarray(viscosity, velocity.x.dtype)
    periodic_x = streamwise_is_periodic(velocity, grid)
    periodic_y = spanwise_is_periodic(velocity, grid)
    x_tendency = nu * (
        _normal_face_curvature(velocity.x, grid, 2, periodic=periodic_x)
        + _cell_curvature(velocity.x, grid, 1, periodic=periodic_y)
        + _tangential_z_curvature(velocity.x, grid, boundaries, "x_velocity")
    )
    y_tendency = nu * (
        _cell_curvature(
            velocity.y,
            grid,
            2,
            periodic=periodic_x,
            zero_edge_tendency=not periodic_x,
        )
        + _normal_face_curvature(velocity.y, grid, 1, periodic=periodic_y)
        + _tangential_z_curvature(velocity.y, grid, boundaries, "y_velocity")
    )
    interior = velocity.z[1:-1]
    horizontal = _cell_curvature(
        interior,
        grid,
        2,
        periodic=periodic_x,
        zero_edge_tendency=not periodic_x,
    ) + _cell_curvature(interior, grid, 1, periodic=periodic_y)
    z_normal = _normal_face_curvature(velocity.z, grid, 0, periodic=False)[1:-1]
    wall = jnp.zeros_like(velocity.z[:1])
    z_tendency = nu * jnp.concatenate(
        (wall, horizontal + z_normal, wall), axis=0
    )
    if not periodic_y:
        y_tendency = y_tendency.at[:, 0].set(0.0).at[:, -1].set(0.0)
    return StaggeredVelocity(x_tendency, y_tendency, z_tendency)


def _centered_cells_to_faces(
    field: jnp.ndarray,
    axis: int,
    *,
    periodic: bool,
    boundary: str = "copy",
) -> jnp.ndarray:
    """Arithmetic midpoint used by the kinetic-energy-preserving flux."""
    if periodic:
        return 0.5 * (field + jnp.roll(field, 1, axis=axis))
    lower = field[_axis_slice(field.ndim, axis, 0, -1)]
    upper = field[_axis_slice(field.ndim, axis, 1, None)]
    interior = 0.5 * (lower + upper)
    first = field[_axis_slice(field.ndim, axis, 0, 1)]
    last = field[_axis_slice(field.ndim, axis, -1, None)]
    if boundary == "zero":
        first, last = jnp.zeros_like(first), jnp.zeros_like(last)
    elif boundary != "copy":
        raise ValueError(f"unsupported centered face boundary: {boundary!r}")
    return jnp.concatenate((first, interior, last), axis=axis)


def _volume_cells_to_faces(
    field: jnp.ndarray,
    grid: Grid,
    axis: int,
    *,
    periodic: bool,
    boundary: str = "copy",
) -> jnp.ndarray:
    """Average a contravariant volume flux over a staggered face."""
    cell_widths = widths(grid, axis, field.dtype)
    if periodic:
        lower = jnp.roll(field, 1, axis=axis)
        lower_width = jnp.roll(cell_widths, 1)
        total = lower_width + cell_widths
        return (
            _axis_shape(lower_width / total, axis, field.ndim) * lower
            + _axis_shape(cell_widths / total, axis, field.ndim) * field
        )
    lower = field[_axis_slice(field.ndim, axis, 0, -1)]
    upper = field[_axis_slice(field.ndim, axis, 1, None)]
    total = cell_widths[:-1] + cell_widths[1:]
    interior = (
        _axis_shape(cell_widths[:-1] / total, axis, field.ndim) * lower
        + _axis_shape(cell_widths[1:] / total, axis, field.ndim) * upper
    )
    first = field[_axis_slice(field.ndim, axis, 0, 1)]
    last = field[_axis_slice(field.ndim, axis, -1, None)]
    if boundary == "zero":
        first, last = jnp.zeros_like(first), jnp.zeros_like(last)
    elif boundary != "copy":
        raise ValueError(f"unsupported volume-flux boundary: {boundary!r}")
    return jnp.concatenate((first, interior, last), axis=axis)


def _edge_fluxes(
    velocity: StaggeredVelocity,
    grid: Grid,
) -> dict[str, jnp.ndarray]:
    """Component-specific, energy-preserving staggered momentum fluxes."""
    u, v, w = velocity
    periodic_x = streamwise_is_periodic(velocity, grid)
    periodic_y = spanwise_is_periodic(velocity, grid)
    return {
        "u_y": _centered_cells_to_faces(
            u, 1, periodic=periodic_y, boundary="copy"
        )
        * _volume_cells_to_faces(
            v, grid, 2, periodic=periodic_x, boundary="copy"
        ),
        "v_x": _centered_cells_to_faces(
            v, 2, periodic=periodic_x, boundary="copy"
        )
        * _volume_cells_to_faces(
            u, grid, 1, periodic=periodic_y, boundary="copy"
        ),
        "u_z": _centered_cells_to_faces(
            u, 0, periodic=False, boundary="zero"
        )
        * _volume_cells_to_faces(
            w, grid, 2, periodic=periodic_x, boundary="copy"
        ),
        "w_x": _centered_cells_to_faces(
            w, 2, periodic=periodic_x, boundary="copy"
        )
        * _volume_cells_to_faces(
            u, grid, 0, periodic=False, boundary="zero"
        ),
        "v_z": _centered_cells_to_faces(
            v, 0, periodic=False, boundary="zero"
        )
        * _volume_cells_to_faces(
            w, grid, 1, periodic=periodic_y, boundary="copy"
        ),
        "w_y": _centered_cells_to_faces(
            w, 1, periodic=periodic_y, boundary="copy"
        )
        * _volume_cells_to_faces(
            v, grid, 0, periodic=False, boundary="zero"
        ),
    }


def advection(velocity: StaggeredVelocity, grid: Grid) -> StaggeredVelocity:
    """Conservative and kinetic-energy-preserving momentum transport."""
    u, v, w = velocity
    periodic_x = streamwise_is_periodic(velocity, grid)
    periodic_y = spanwise_is_periodic(velocity, grid)
    flux = _edge_fluxes(velocity, grid)
    if periodic_x:
        uu_cell = (0.5 * (u + jnp.roll(u, -1, axis=2))) ** 2
        uu_x = (uu_cell - jnp.roll(uu_cell, 1, axis=2)) / (
            shaped_center_distances(grid, 2, periodic=True, dtype=u.dtype)
        )
        vu_x = (jnp.roll(flux["v_x"], -1, axis=2) - flux["v_x"]) / (
            shaped_widths(grid, 2, flux["v_x"].dtype)
        )
        wu_x = (jnp.roll(flux["w_x"], -1, axis=2) - flux["w_x"]) / (
            shaped_widths(grid, 2, flux["w_x"].dtype)
        )
    else:
        uu_cell = (0.5 * (u[..., :-1] + u[..., 1:])) ** 2
        edge_x = jnp.zeros_like(u[..., :1])
        uu_x = jnp.concatenate(
            (
                edge_x,
                (uu_cell[..., 1:] - uu_cell[..., :-1])
                / center_distances(grid, 2, periodic=False, dtype=u.dtype)[
                    None, None, 1:-1
                ],
                edge_x,
            ),
            axis=2,
        )
        vu_x = (flux["v_x"][..., 1:] - flux["v_x"][..., :-1]) / (
            shaped_widths(grid, 2, flux["v_x"].dtype)
        )
        wu_x = (flux["w_x"][..., 1:] - flux["w_x"][..., :-1]) / (
            shaped_widths(grid, 2, flux["w_x"].dtype)
        )
    if periodic_y:
        vv_cell = (0.5 * (v + jnp.roll(v, -1, axis=1))) ** 2
        uv_y = (jnp.roll(flux["u_y"], -1, axis=1) - flux["u_y"]) / (
            shaped_widths(grid, 1, flux["u_y"].dtype)
        )
        vv_y = (vv_cell - jnp.roll(vv_cell, 1, axis=1)) / (
            shaped_center_distances(grid, 1, periodic=True, dtype=v.dtype)
        )
        wv_y = (jnp.roll(flux["w_y"], -1, axis=1) - flux["w_y"]) / (
            shaped_widths(grid, 1, flux["w_y"].dtype)
        )
    else:
        vv_cell = (0.5 * (v[:, :-1] + v[:, 1:])) ** 2
        uv_y = (flux["u_y"][:, 1:] - flux["u_y"][:, :-1]) / (
            shaped_widths(grid, 1, flux["u_y"].dtype)
        )
        edge_y = jnp.zeros_like(v[:, :1])
        vv_y = jnp.concatenate(
            (
                edge_y,
                (vv_cell[:, 1:] - vv_cell[:, :-1])
                / center_distances(grid, 1, periodic=False, dtype=v.dtype)[
                    None, 1:-1, None
                ],
                edge_y,
            ),
            axis=1,
        )
        wv_y = (flux["w_y"][:, 1:] - flux["w_y"][:, :-1]) / (
            shaped_widths(grid, 1, flux["w_y"].dtype)
        )
    u_flux_divergence = (
        uu_x
        + uv_y
        + (flux["u_z"][1:] - flux["u_z"][:-1])
        / shaped_widths(grid, 0, flux["u_z"].dtype)
    )
    v_flux_divergence = (
        vu_x
        + vv_y
        + (flux["v_z"][1:] - flux["v_z"][:-1])
        / shaped_widths(grid, 0, flux["v_z"].dtype)
    )
    ww_cell = (0.5 * (w[:-1] + w[1:])) ** 2
    ww_z = (ww_cell[1:] - ww_cell[:-1]) / center_distances(
        grid, 0, periodic=False, dtype=w.dtype
    )[1:-1, None, None]
    w_flux_divergence = wu_x[1:-1] + wv_y[1:-1] + ww_z
    wall = jnp.zeros_like(w[:1])
    v_tendency = -v_flux_divergence
    if not periodic_y:
        v_tendency = v_tendency.at[:, 0].set(0.0).at[:, -1].set(0.0)
    return StaggeredVelocity(
        -u_flux_divergence,
        v_tendency,
        -jnp.concatenate((wall, w_flux_divergence, wall), axis=0),
    )


def cell_velocity(
    velocity: StaggeredVelocity,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interpolate the staggered components to cell centres."""
    x_velocity = (
        0.5 * (velocity.x + jnp.roll(velocity.x, -1, axis=2))
        if velocity.x.shape[-1] == velocity.y.shape[-1]
        else 0.5 * (velocity.x[..., :-1] + velocity.x[..., 1:])
    )
    y_velocity = (
        0.5 * (velocity.y + jnp.roll(velocity.y, -1, axis=1))
        if velocity.y.shape[1] == velocity.x.shape[1]
        else 0.5 * (velocity.y[:, :-1] + velocity.y[:, 1:])
    )
    return x_velocity, y_velocity, 0.5 * (velocity.z[:-1] + velocity.z[1:])


def kinetic_energy(velocity: StaggeredVelocity, grid: Grid) -> jnp.ndarray:
    """Volume-integrated kinetic energy of the staggered field."""
    periodic_x = streamwise_is_periodic(velocity, grid)
    periodic_y = spanwise_is_periodic(velocity, grid)
    x_width = shaped_center_distances(
        grid, 2, periodic=periodic_x, dtype=velocity.x.dtype
    )
    y_width = shaped_center_distances(
        grid, 1, periodic=periodic_y, dtype=velocity.y.dtype
    )
    z_width = shaped_center_distances(
        grid, 0, periodic=False, dtype=velocity.z.dtype
    )
    u_volume = (
        shaped_widths(grid, 0, velocity.x.dtype)
        * shaped_widths(grid, 1, velocity.x.dtype)
        * x_width
    )
    v_volume = (
        shaped_widths(grid, 0, velocity.y.dtype)
        * y_width
        * shaped_widths(grid, 2, velocity.y.dtype)
    )
    w_volume = (
        z_width
        * shaped_widths(grid, 1, velocity.z.dtype)
        * shaped_widths(grid, 2, velocity.z.dtype)
    )
    return 0.5 * (
        jnp.sum(u_volume * velocity.x**2)
        + jnp.sum(v_volume * velocity.y**2)
        + jnp.sum(w_volume * velocity.z**2)
    )


def courant_number(
    velocity: StaggeredVelocity,
    grid: Grid,
    dt: float,
) -> jnp.ndarray:
    """Conservative upper bound on the convective Courant number."""
    minimum_x = jnp.asarray(grid.x_widths.min(), velocity.x.dtype)
    minimum_y = jnp.asarray(grid.y_widths.min(), velocity.y.dtype)
    minimum_z = jnp.asarray(grid.z_widths.min(), velocity.z.dtype)
    return dt * (
        jnp.max(jnp.abs(velocity.x)) / minimum_x
        + jnp.max(jnp.abs(velocity.y)) / minimum_y
        + jnp.max(jnp.abs(velocity.z)) / minimum_z
    )


def stable_timestep(
    velocity: StaggeredVelocity,
    grid: Grid,
    viscosity: float,
    *,
    courant: float = 0.5,
    diffusion_number: float = 0.25,
) -> jnp.ndarray:
    """Largest explicit step satisfying convective and viscous limits."""
    minimum_x = jnp.asarray(grid.x_widths.min(), velocity.x.dtype)
    minimum_y = jnp.asarray(grid.y_widths.min(), velocity.y.dtype)
    minimum_z = jnp.asarray(grid.z_widths.min(), velocity.z.dtype)
    speed = (
        jnp.max(jnp.abs(velocity.x)) / minimum_x
        + jnp.max(jnp.abs(velocity.y)) / minimum_y
        + jnp.max(jnp.abs(velocity.z)) / minimum_z
    )
    inverse_squares = (
        1.0 / minimum_x**2 + 1.0 / minimum_y**2 + 2.0 / minimum_z**2
    )
    largest = jnp.max(jnp.asarray(viscosity))
    convective = jnp.where(
        speed > 0.0, courant / jnp.maximum(speed, 1e-300), jnp.inf
    )
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
