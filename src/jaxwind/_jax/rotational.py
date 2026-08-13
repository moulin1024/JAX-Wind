"""Rotational-advection kernel for the JAX solver."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def build_rotational_advection_kernel(*, grid, axis_name, wall_filter_local):
    """Build rotational convection while retaining grid-specialized closures."""

    def wall_gradient_local(
        context,
        enabled,
        roughness_length,
        von_karman,
        filtered,
        filter_width,
    ):
        wall_velocity = wall_filter_local(
            jnp.stack((context.u[0], context.v[0])),
            filter_width,
        )
        wall_u = jnp.where(filtered, wall_velocity[0], context.u[0])
        wall_v = jnp.where(filtered, wall_velocity[1], context.v[0])
        speed = jnp.hypot(wall_u, wall_v)
        safe_speed = jnp.maximum(speed, jnp.finfo(speed.dtype).tiny)
        denominator = jnp.log(0.5 * grid.dz / roughness_length)
        friction_velocity = speed * von_karman / denominator
        gradient = (
            jnp.stack((wall_u, wall_v))
            * friction_velocity
            / (safe_speed * von_karman * 0.5 * grid.dz)
        )
        moving = speed > jnp.finfo(speed.dtype).tiny
        bottom = lax.axis_index(axis_name) == 0
        return jnp.where(enabled & bottom & moving[None], gradient, 0.0)

    def dry_rotational_advection_local(
        context,
        wall_gradient_enabled,
        roughness_length,
        von_karman,
        wall_filtered,
        wall_filter_width,
    ):
        """Evaluate rotational convection on the hybrid cell/face layout."""
        wall_gradient = wall_gradient_local(
            context,
            wall_gradient_enabled,
            roughness_length,
            von_karman,
            wall_filtered,
            wall_filter_width,
        )
        lower_w = 2.0 * context.w_at_cells - context.w_upper
        lower_dudz = 2.0 * context.dudz_at_cells - context.dudz_upper
        lower_dvdz = 2.0 * context.dvdz_at_cells - context.dvdz_upper
        lower_dwdx = 2.0 * context.dwdx_at_cells - context.dwdx_upper
        lower_dwdy = 2.0 * context.dwdy_at_cells - context.dwdy_upper
        bottom = lax.axis_index(axis_name) == 0
        lower_dudz = lower_dudz.at[0].set(
            jnp.where(bottom & wall_gradient_enabled, wall_gradient[0], lower_dudz[0])
        )
        lower_dvdz = lower_dvdz.at[0].set(
            jnp.where(bottom & wall_gradient_enabled, wall_gradient[1], lower_dvdz[0])
        )
        x = context.v * (context.dudy - context.dvdx)
        x += 0.5 * (
            context.w_upper * (context.dudz_upper - context.dwdx_upper)
            + lower_w * (lower_dudz - lower_dwdx)
        )
        y = context.u * (context.dvdx - context.dudy)
        y += 0.5 * (
            context.w_upper * (context.dvdz_upper - context.dwdy_upper)
            + lower_w * (lower_dvdz - lower_dwdy)
        )
        z = context.u_upper * (context.dwdx_upper - context.dudz_upper)
        z += context.v_upper * (context.dwdy_upper - context.dvdz_upper)
        z = z.at[-1].set(jnp.where(context.upper_is_physical, 0.0, z[-1]))
        return -x, -y, -z

    return dry_rotational_advection_local
