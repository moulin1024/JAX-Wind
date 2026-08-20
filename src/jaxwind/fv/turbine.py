"""Adapters from the shared wind-turbine kernels to the FV MAC mesh."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jaxwind._jax.wind import (
    build_blade_element_disk_kernel,
    build_nacelle_tower_kernel,
)
from jaxwind.domain.grid import UniformGrid
from jaxwind.physics import BladeElementActuatorDisk, NacelleTowerDrag

from .operators import cell_velocity
from .state import StaggeredVelocity


def _x_faces(values: jnp.ndarray) -> jnp.ndarray:
    """Interpolate cell forcing to a distinct-inlet/outlet x-face mesh."""
    interior = 0.5 * (values[..., :-1] + values[..., 1:])
    return jnp.concatenate((values[..., :1], interior, values[..., -1:]), axis=2)


def _y_faces(values: jnp.ndarray) -> jnp.ndarray:
    """Interpolate cell forcing to periodic y faces."""
    return 0.5 * (values + jnp.roll(values, 1, axis=1))


def build_adbem_forcing(
    grid: UniformGrid,
    disk: BladeElementActuatorDisk,
    body: NacelleTowerDrag | None = None,
) -> Callable[[StaggeredVelocity, jnp.ndarray], StaggeredVelocity]:
    """Build single-device DTU-style AD-BEM forcing for an open FV domain.

    The aerodynamic calculation is the same annular kernel used by the
    spectral solver. Velocities are sampled at cell centres and upper z faces;
    the resulting accelerations are conservatively centred back onto the MAC
    component faces before entering the FV momentum tendency.
    """
    disk_kernel = build_blade_element_disk_kernel(
        grid=grid,
        axis_name="fv_adbem",
        partition_count=1,
    )

    def disk_local(u, v, w_upper):
        dtype = u.dtype
        return disk_kernel(
            u,
            v,
            w_upper,
            disk.x,
            disk.y,
            disk.z,
            disk.blade_count,
            disk.hub_radius,
            disk.tip_radius,
            disk.angular_velocity,
            jnp.asarray(disk.element_smoothing_widths, dtype=dtype),
            jnp.asarray(disk.element_radii, dtype=dtype),
            jnp.asarray(disk.element_widths, dtype=dtype),
            jnp.asarray(disk.element_chords, dtype=dtype),
            jnp.asarray(disk.element_twist_degrees, dtype=dtype),
            jnp.asarray(disk.element_airfoil_ids, dtype=jnp.int32),
            jnp.asarray(disk.polar_alpha_degrees, dtype=dtype),
            jnp.asarray(disk.polar_lift_coefficients, dtype=dtype),
            jnp.asarray(disk.polar_drag_coefficients, dtype=dtype),
            disk.pitch_degrees,
            disk.tip_loss,
            disk.root_loss,
        )

    disk_mapped = jax.pmap(disk_local, axis_name="fv_adbem")
    body_mapped = None
    if body is not None:
        body_kernel = build_nacelle_tower_kernel(
            grid=grid,
            axis_name="fv_turbine_body",
        )

        def body_local(u, v):
            return body_kernel(
                u,
                v,
                body.x,
                body.y,
                body.hub_height,
                body.nacelle_length,
                body.nacelle_diameter,
                body.nacelle_drag_coefficient,
                body.tower_base_diameter,
                body.tower_top_diameter,
                body.tower_drag_coefficient,
                body.smoothing_width,
            )

        body_mapped = jax.pmap(body_local, axis_name="fv_turbine_body")

    def forcing(
        velocity: StaggeredVelocity,
        _time: jnp.ndarray,
    ) -> StaggeredVelocity:
        u, v, _w = cell_velocity(velocity)
        w_upper = velocity.z[1:]
        disk_values = disk_mapped(u[None], v[None], w_upper[None])
        source_x = disk_values[0][0]
        source_y = disk_values[1][0]
        source_z_upper = disk_values[2][0]
        if body_mapped is not None:
            body_values = body_mapped(u[None], v[None])
            source_x = source_x + body_values[0][0]
            source_y = source_y + body_values[1][0]
            source_z_upper = source_z_upper + body_values[2][0]
        wall = jnp.zeros_like(source_z_upper[:1])
        return StaggeredVelocity(
            _x_faces(source_x),
            _y_faces(source_y),
            jnp.concatenate((wall, source_z_upper), axis=0),
        )

    return forcing


__all__ = ["build_adbem_forcing"]
