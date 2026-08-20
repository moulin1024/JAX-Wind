"""Time-dependent precursor inflow and second-order FV outflow closures."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from jaxwind.domain.grid import UniformGrid

from .state import (
    OPEN,
    Boundaries,
    StaggeredVelocity,
    enforce_impermeability,
    streamwise_is_periodic,
    validate,
)


class InflowPlane(NamedTuple):
    """One yz layer recorded from a periodic precursor state."""

    x_velocity: jnp.ndarray
    y_velocity: jnp.ndarray
    z_velocity: jnp.ndarray
    scalar: jnp.ndarray


def extract_inflow_plane(solution, grid: UniformGrid, plane: int = 0) -> InflowPlane:
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


def validate_inflow_plane(plane: InflowPlane, grid: UniformGrid) -> None:
    """Validate the staggered yz shapes of a recorded layer."""
    expected = {
        "x_velocity": (grid.nz, grid.ny),
        "y_velocity": (grid.nz, grid.ny),
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
    grid: UniformGrid,
) -> StaggeredVelocity:
    """Give a periodic MAC field distinct inlet and outlet x faces."""
    if not streamwise_is_periodic(velocity, grid):
        raise ValueError("the source velocity is already nonperiodic")
    opened = StaggeredVelocity(
        jnp.concatenate((velocity.x, velocity.x[..., :1]), axis=2),
        velocity.y,
        velocity.z,
    )
    validate(opened, grid, Boundaries(streamwise=OPEN))
    return opened


def enforce_open_velocity(
    velocity: StaggeredVelocity,
    plane: InflowPlane,
    grid: UniformGrid,
    *,
    extrapolate_normal_outflow: bool = True,
) -> StaggeredVelocity:
    """Overwrite one inlet layer and apply second-order outlet extrapolation."""
    validate_inflow_plane(plane, grid)
    validate(velocity, grid, Boundaries(streamwise=OPEN))
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
    return enforce_impermeability(result)


def enforce_open_scalar(
    scalar: jnp.ndarray,
    plane: InflowPlane,
    grid: UniformGrid,
) -> jnp.ndarray:
    """Overwrite the scalar inlet layer and extrapolate its outlet layer."""
    validate_inflow_plane(plane, grid)
    expected = (grid.nz, grid.ny, grid.nx)
    if scalar.shape != expected:
        raise ValueError(f"scalar must have shape {expected}")
    return _second_order_outflow(scalar.at[..., 0].set(plane.scalar))


__all__ = [
    "InflowPlane",
    "enforce_open_scalar",
    "enforce_open_velocity",
    "extract_inflow_plane",
    "periodic_to_open_velocity",
    "validate_inflow_plane",
]
