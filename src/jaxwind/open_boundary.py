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
    physical_transverse_inlet: bool = False,
) -> StaggeredVelocity:
    """Impose inlet data and apply second-order outlet extrapolation.

    physical_transverse_inlet preserves the v/w unknowns at x=dx/2; callers
    must supply physical-face transport/stress fluxes and a compatible pressure
    operator. The default retains the prescribed first-cell transverse layer.
    """
    wall_y = not spanwise_is_periodic(velocity, grid)
    validate_inflow_plane(plane, grid, wall_y=wall_y)
    spanwise = OPEN if open_y else (FREE_SLIP if wall_y else PERIODIC)
    validate(velocity, grid, Boundaries(streamwise=OPEN, spanwise=spanwise))
    if open_y:
        def sides(field):
            low = (4.0 * field[:, 1] - field[:, 2]) / 3.0
            high = (4.0 * field[:, -2] - field[:, -3]) / 3.0
            return field.at[:, 0].set(low).at[:, -1].set(high)
        velocity = StaggeredVelocity(
            sides(velocity.x),
            # Lateral pressure outlets retain their ordinary extrapolation.
            # The high-x backflow option must not impose a tangential-energy
            # pressure jump on the developing rough-wall layer.
            sides(velocity.y),
            sides(velocity.z),
        )
    x_velocity = velocity.x.at[..., 0].set(plane.x_velocity)
    # Transverse MAC unknowns lie half a cell inside the inlet. The physical
    # option prescribes their boundary fluxes instead of overwriting them.
    y_velocity = velocity.y if physical_transverse_inlet else velocity.y.at[..., 0].set(plane.y_velocity)
    z_velocity = velocity.z if physical_transverse_inlet else velocity.z.at[..., 0].set(plane.z_velocity)
    result = StaggeredVelocity(
        _second_order_outflow(x_velocity)
        if extrapolate_normal_outflow
        else x_velocity,
        _second_order_outflow(y_velocity),
        _second_order_outflow(z_velocity),
    )
    return enforce_impermeability(result, open_y=open_y)


def backflow_outlet_pressure(velocity: StaggeredVelocity, grid: Grid) -> jnp.ndarray:
    """Kinematic high-x pressure for the normal-traction backflow closure.

    With zero normal viscous traction, p_b = -|u_b|^2 for u_n < 0,
    otherwise zero (OBC-B, sharp-switch limit, Dong & Shen 2015,
    doi:10.1016/j.jcp.2015.03.012). The continuum boundary power is
    -(p_b + |u_b|^2/2) u_n = -|u_b|^2 |u_n|/2 <= 0.
    This is a boundary energy condition, not a discrete RK stability proof
    or a nonreflecting boundary condition. Tangential velocities are averaged
    onto the outlet u-face locations; no inlet data or opposite x edge enters.
    """
    if streamwise_is_periodic(velocity, grid):
        raise ValueError("backflow outlet pressure requires nonperiodic x")
    normal = velocity.x[..., -1]
    tangent_y = velocity.y[..., -1]
    if spanwise_is_periodic(velocity, grid):
        tangent_y = 0.5 * (tangent_y + jnp.roll(tangent_y, -1, axis=1))
    else:
        tangent_y = 0.5 * (tangent_y[:, :-1] + tangent_y[:, 1:])
    tangent_z = 0.5 * (velocity.z[:-1, :, -1] + velocity.z[1:, :, -1])
    speed_squared = normal**2 + tangent_y**2 + tangent_z**2
    return jnp.where(normal < 0., -speed_squared, 0.)


def backflow_lateral_pressures(velocity, grid, streamwise_reference):
    """Energy backflow pressure on y sides, relative to the ambient throughflow.

    The reference is tangential to both y sides. For the perturbation velocity
    q = u - (U_ref, 0, 0), pressure work plus advective perturbation-energy flux
    is -(p + |q|²/2) u_n. Setting p=-|q|² on reversal makes that nonpositive.
    Using the ambient reference also leaves uniform tangential flow unchanged.
    The projection retains its prescribed x-end/corner constraints.
    """
    if spanwise_is_periodic(velocity, grid) or streamwise_is_periodic(velocity, grid):
        raise ValueError("lateral backflow pressure requires nonperiodic x and y")
    u = 0.5 * (velocity.x[..., :-1] + velocity.x[..., 1:])
    w = 0.5 * (velocity.z[:-1] + velocity.z[1:])
    reference = jnp.asarray(streamwise_reference, velocity.x.dtype)
    low_normal, high_normal = -velocity.y[:, 0], velocity.y[:, -1]
    low_speed2 = (u[:, 0] - reference)**2 + low_normal**2 + w[:, 0]**2
    high_speed2 = (u[:, -1] - reference)**2 + high_normal**2 + w[:, -1]**2
    return (jnp.where(low_normal < 0., -low_speed2, 0.),
            jnp.where(high_normal < 0., -high_speed2, 0.))


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
    "backflow_outlet_pressure",
    "backflow_lateral_pressures",
    "enforce_open_scalar",
    "enforce_open_velocity",
    "enforce_two_outlet_velocity",
    "enforce_two_outlet_scalar",
    "extract_inflow_plane",
    "periodic_to_open_velocity",
    "validate_inflow_plane",
]
