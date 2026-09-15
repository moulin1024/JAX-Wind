"""Time-dependent precursor inflow and second-order FV outflow closures."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from jaxwind.domain.grid import Grid

from .state import (
    FREE_SLIP,
    OPEN,
    PERIODIC,
    Boundaries,
    StaggeredVelocity,
    enforce_impermeability,
    spanwise_is_periodic,
    streamwise_is_periodic,
    validate,
)


class InflowPlane(NamedTuple):
    """One yz layer recorded from a periodic precursor state."""

    x_velocity: jnp.ndarray
    y_velocity: jnp.ndarray
    z_velocity: jnp.ndarray
    scalar: jnp.ndarray


def extract_inflow_plane(solution, grid: Grid, plane: int = 0) -> InflowPlane:
    """Extract exactly one streamwise layer from a periodic FV solution."""
    if not 0 <= plane < grid.nx:
        raise ValueError("the precursor inflow plane is outside the mesh")
    if not streamwise_is_periodic(solution.velocity, grid):
        raise ValueError("inflow recording requires a periodic precursor")
    return InflowPlane(
        solution.velocity.x[..., plane],
        solution.velocity.y[..., plane],
        solution.velocity.z[..., plane],
        solution.scalar[..., plane],
    )


def validate_inflow_plane(
    plane: InflowPlane,
    grid: Grid,
    *,
    wall_y: bool = False,
) -> None:
    """Validate the staggered yz shapes of a recorded layer."""
    expected = {
        "x_velocity": (grid.nz, grid.ny),
        "y_velocity": (grid.nz, grid.ny + 1 if wall_y else grid.ny),
        "z_velocity": (grid.nz + 1, grid.ny),
        "scalar": (grid.nz, grid.ny),
    }
    for name, shape in expected.items():
        if getattr(plane, name).shape != shape:
            raise ValueError(f"inflow {name} must have shape {shape}")


def _second_order_outflow(field: jnp.ndarray) -> jnp.ndarray:
    """Apply the three-point, zero-normal-gradient outlet extrapolation."""
    if field.shape[-1] < 3:
        raise ValueError("second-order outflow requires at least three x locations")
    outlet = (4.0 * field[..., -2] - field[..., -3]) / 3.0
    return field.at[..., -1].set(outlet)


def periodic_to_open_velocity(
    velocity: StaggeredVelocity,
    grid: Grid,
) -> StaggeredVelocity:
    """Give a periodic MAC field distinct inlet and outlet x faces."""
    if not streamwise_is_periodic(velocity, grid):
        raise ValueError("the source velocity is already nonperiodic")
    opened = StaggeredVelocity(
        jnp.concatenate((velocity.x, velocity.x[..., :1]), axis=2),
        velocity.y,
        velocity.z,
    )
    spanwise = FREE_SLIP if not spanwise_is_periodic(velocity, grid) else PERIODIC
    validate(opened, grid, Boundaries(streamwise=OPEN, spanwise=spanwise))
    return opened


def enforce_open_velocity(
    velocity: StaggeredVelocity,
    plane: InflowPlane,
    grid: Grid,
    *,
    extrapolate_normal_outflow: bool = True,
    open_y: bool = False,
) -> StaggeredVelocity:
    """Overwrite one inlet layer and apply second-order outlet extrapolation."""
    wall_y = not spanwise_is_periodic(velocity, grid)
    validate_inflow_plane(plane, grid, wall_y=wall_y)
    spanwise = OPEN if open_y else (FREE_SLIP if wall_y else PERIODIC)
    validate(velocity, grid, Boundaries(streamwise=OPEN, spanwise=spanwise))
    if open_y:
        def sides(field):
            low = (4.0 * field[:, 1] - field[:, 2]) / 3.0
            high = (4.0 * field[:, -2] - field[:, -3]) / 3.0
            return field.at[:, 0].set(low).at[:, -1].set(high)
        velocity = StaggeredVelocity(*(sides(field) for field in velocity))
    x_velocity = velocity.x.at[..., 0].set(plane.x_velocity)
    y_velocity = velocity.y.at[..., 0].set(plane.y_velocity)
    z_velocity = velocity.z.at[..., 0].set(plane.z_velocity)
    result = StaggeredVelocity(
        _second_order_outflow(x_velocity)
        if extrapolate_normal_outflow
        else x_velocity,
        _second_order_outflow(y_velocity),
        _second_order_outflow(z_velocity),
    )
    return enforce_impermeability(result, open_y=open_y)


def enforce_open_scalar(
    scalar: jnp.ndarray,
    plane: InflowPlane,
    grid: Grid,
) -> jnp.ndarray:
    """Overwrite the scalar inlet layer and extrapolate its outlet layer."""
    validate_inflow_plane(plane, grid, wall_y=plane.y_velocity.shape[1] == grid.ny + 1)
    expected = (grid.nz, grid.ny, grid.nx)
    if scalar.shape != expected:
        raise ValueError(f"scalar must have shape {expected}")
    return _second_order_outflow(scalar.at[..., 0].set(plane.scalar))


def enforce_two_outlet_velocity(
    velocity: StaggeredVelocity,
) -> StaggeredVelocity:
    """Extrapolate both x ends; pressure projection sets the final normal flux."""
    def both(field):
        field = _second_order_outflow(field)
        return field.at[..., 0].set((4.0 * field[..., 1] - field[..., 2]) / 3.0)

    return enforce_impermeability(StaggeredVelocity(
        both(velocity.x), both(velocity.y), both(velocity.z),
    ))


def enforce_two_outlet_scalar(scalar, velocity, ambient):
    """Zero-gradient outflow and ambient scalar on pressure-driven backflow."""
    left = jnp.where(velocity.x[..., 0] > 0.0, ambient, scalar[..., 1])
    right = jnp.where(velocity.x[..., -1] < 0.0, ambient, scalar[..., -2])
    return scalar.at[..., 0].set(left).at[..., -1].set(right)


__all__ = [
    "InflowPlane",
    "enforce_open_scalar",
    "enforce_open_velocity",
    "enforce_two_outlet_velocity",
    "enforce_two_outlet_scalar",
    "extract_inflow_plane",
    "periodic_to_open_velocity",
    "validate_inflow_plane",
]
