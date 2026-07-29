"""Compilation of mapped kernels for the z-slab interpreter.

The public interpreter owns semantic validation and field construction.  This
module owns only JAX kernel assembly and mapping.
"""

from __future__ import annotations

from functools import partial
import math

import jax
import jax.numpy as jnp
from jax import lax

from jaxwind.domain import EqualZSlab

from ._jax_zslab_wind import (
    build_actuator_line_kernel,
    build_wind_tunnel_kernel,
)
from ._jax_zslab_lasd_kernels import build_lasd_kernels
from .jax_zslab import (
    JaxZSlabInterpreter,
    PackedHaloArrays,
    ZSlabDryFlowArrays,
    ZSlabScalarArrays,
)


def build_zslab_interpreter(
    decomposition: EqualZSlab,
    *,
    addressable_shards: tuple[int, ...] | None = None,
    axis_name: str = "jaxwind_z",
) -> JaxZSlabInterpreter:
    """Build mapped kernels without capturing any field-sized constants."""
    shard_count = decomposition.shard_count
    if addressable_shards is None:
        if shard_count != 1:
            raise ValueError(
                "addressable_shards is required for a multi-shard decomposition"
            )
        addressable_shards = (0,)
    if not addressable_shards:
        raise ValueError("at least one addressable shard is required")
    if len(set(addressable_shards)) != len(addressable_shards):
        raise ValueError("addressable shard indices must be unique")
    if any(not 0 <= index < shard_count for index in addressable_shards):
        raise ValueError("addressable shard index is outside the global z mesh")

    previous_permutation = tuple(
        (source, source + 1) for source in range(shard_count - 1)
    )
    next_permutation = tuple((source, source - 1) for source in range(1, shard_count))

    def exchange_local(packed):
        if packed.ndim != 4:
            raise ValueError("packed local fields must have shape (field, z, y, x)")
        index = lax.axis_index(axis_name)
        if shard_count == 1:
            lower = jnp.zeros_like(packed[:, -1])
            upper = jnp.zeros_like(packed[:, 0])
        else:
            lower = lax.ppermute(
                packed[:, -1],
                axis_name,
                previous_permutation,
            )
            upper = lax.ppermute(
                packed[:, 0],
                axis_name,
                next_permutation,
            )
        lower_physical = index == 0
        upper_physical = index == shard_count - 1
        lower = jnp.where(lower_physical, jnp.zeros_like(lower), lower)
        upper = jnp.where(upper_physical, jnp.zeros_like(upper), upper)
        return PackedHaloArrays(
            lower,
            upper,
            lower_physical,
            upper_physical,
        )

    def pressure_gradient_local(pressure, upper_boundary_gradient):
        halo = exchange_local(pressure[None, ...])
        next_cell = halo.upper[0]
        interface_gradient = (next_cell - pressure[-1]) / decomposition.grid.dz
        boundary_gradient = jnp.broadcast_to(
            jnp.asarray(upper_boundary_gradient, pressure.dtype),
            pressure.shape[1:],
        )
        last = jnp.where(halo.upper_is_physical, boundary_gradient, interface_gradient)
        interior = (pressure[1:] - pressure[:-1]) / decomposition.grid.dz
        return jnp.concatenate((interior, last[None, ...]), axis=0)

    def divergence_local(upper_faces, lower_boundary_face):
        halo = exchange_local(upper_faces[None, ...])
        boundary_face = jnp.broadcast_to(
            jnp.asarray(lower_boundary_face, upper_faces.dtype),
            upper_faces.shape[1:],
        )
        lower_first = jnp.where(
            halo.lower_is_physical,
            boundary_face,
            halo.lower[0],
        )
        lower_faces = jnp.concatenate(
            (lower_first[None, ...], upper_faces[:-1]),
            axis=0,
        )
        return (upper_faces - lower_faces) / decomposition.grid.dz

    def enforce_upper_boundary_local(upper_faces, upper_boundary_face):
        index = lax.axis_index(axis_name)
        boundary_face = jnp.broadcast_to(
            jnp.asarray(upper_boundary_face, upper_faces.dtype),
            upper_faces.shape[1:],
        )
        last = jnp.where(
            index == shard_count - 1,
            boundary_face,
            upper_faces[-1],
        )
        return upper_faces.at[-1].set(last)

    grid = decomposition.grid
    kx = 2.0 * jnp.pi * jnp.fft.rfftfreq(grid.nx, d=grid.lx / grid.nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(grid.ny, d=grid.ly / grid.ny)
    keep = jnp.ones((grid.ny, grid.nx // 2 + 1))
    if grid.nx % 2 == 0:
        kx = kx.at[-1].set(0.0)
        keep = keep.at[:, -1].set(0.0)
    if grid.ny % 2 == 0:
        ky = ky.at[grid.ny // 2].set(0.0)
        keep = keep.at[grid.ny // 2, :].set(0.0)

    x_mode = jnp.arange(grid.nx // 2 + 1)
    y_mode = jnp.fft.fftfreq(grid.ny) * grid.ny
    two_thirds = (jnp.abs(y_mode)[:, None] <= grid.ny // 3) & (
        x_mode[None, :] <= grid.nx // 3
    )

    def horizontal_derivative_local(values, axis):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        local_kx = kx.astype(values.real.dtype)
        local_ky = ky.astype(values.real.dtype)
        if axis == 0:
            multiplier = 1j * local_kx
        else:
            multiplier = 1j * local_ky[:, None]
        return jnp.fft.irfftn(
            spectrum * multiplier * keep.astype(values.real.dtype),
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def two_thirds_filter_local(values):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        return jnp.fft.irfftn(
            spectrum * two_thirds,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def wall_filter_local(values, filter_width):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        cutoff_x = jnp.floor(grid.nx / (2.0 * filter_width))
        cutoff_y = jnp.floor(grid.ny / (2.0 * filter_width))
        wall_keep = (jnp.abs(y_mode)[:, None] < cutoff_y) & (
            x_mode[None, :] < cutoff_x
        )
        return jnp.fft.irfftn(
            spectrum * wall_keep,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def truncated_derivative_local(values, axis):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        local_kx = kx.astype(values.real.dtype)
        local_ky = ky.astype(values.real.dtype)
        if axis == 0:
            multiplier = 1j * local_kx
        else:
            multiplier = 1j * local_ky[:, None]
        return jnp.fft.irfftn(
            spectrum * multiplier * two_thirds,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def strain_magnitude_local(
        dudx,
        dudy,
        dudz,
        dvdx,
        dvdy,
        dvdz,
        dwdx,
        dwdy,
        dwdz,
    ):
        sxy = 0.5 * (dudy + dvdx)
        sxz = 0.5 * (dudz + dwdx)
        syz = 0.5 * (dvdz + dwdy)
        symmetric_dot = (
            dudx * dudx
            + dvdy * dvdy
            + dwdz * dwdz
            + 2.0 * (sxy * sxy + sxz * sxz + syz * syz)
        )
        return jnp.sqrt(jnp.maximum(2.0 * symmetric_dot, 0.0))

    def dry_flow_context_local(u, v, w_upper, lower_boundary):
        halo = exchange_local(jnp.stack((u, v, w_upper), axis=0))
        lower_boundary_plane = jnp.broadcast_to(
            jnp.asarray(lower_boundary, dtype=w_upper.dtype),
            w_upper.shape[1:],
        )
        previous_u = jnp.where(halo.lower_is_physical, u[0], halo.lower[0])
        previous_v = jnp.where(halo.lower_is_physical, v[0], halo.lower[1])
        next_u_plane = jnp.where(halo.upper_is_physical, u[-1], halo.upper[0])
        next_v_plane = jnp.where(halo.upper_is_physical, v[-1], halo.upper[1])
        next_w_upper = jnp.where(
            halo.upper_is_physical,
            w_upper[-1],
            halo.upper[2],
        )
        w_lower_plane = jnp.where(
            halo.lower_is_physical,
            lower_boundary_plane,
            halo.lower[2],
        )
        lower_faces = jnp.concatenate((w_lower_plane[None], w_upper[:-1]), axis=0)
        w_at_cells = 0.5 * (lower_faces + w_upper)
        next_u = jnp.concatenate((u[1:], next_u_plane[None]), axis=0)
        next_v = jnp.concatenate((v[1:], next_v_plane[None]), axis=0)
        u_upper = 0.5 * (u + next_u)
        v_upper = 0.5 * (v + next_v)
        u_lower = 0.5 * (previous_u + u[0])
        v_lower = 0.5 * (previous_v + v[0])

        dudz_upper = (next_u - u) / grid.dz
        dvdz_upper = (next_v - v) / grid.dz
        dudz_upper = dudz_upper.at[-1].set(
            jnp.where(halo.upper_is_physical, 0.0, dudz_upper[-1])
        )
        dvdz_upper = dvdz_upper.at[-1].set(
            jnp.where(halo.upper_is_physical, 0.0, dvdz_upper[-1])
        )
        index = lax.axis_index(axis_name)
        wall_correction = 1.0 / math.log(3.0)
        dudz_upper = dudz_upper.at[0].set(
            jnp.where(
                index == 0,
                wall_correction * dudz_upper[0],
                dudz_upper[0],
            )
        )
        dvdz_upper = dvdz_upper.at[0].set(
            jnp.where(
                index == 0,
                wall_correction * dvdz_upper[0],
                dvdz_upper[0],
            )
        )
        lower_dudz = jnp.where(
            halo.lower_is_physical,
            jnp.zeros_like(previous_u),
            (u[0] - previous_u) / grid.dz,
        )
        lower_dvdz = jnp.where(
            halo.lower_is_physical,
            jnp.zeros_like(previous_v),
            (v[0] - previous_v) / grid.dz,
        )
        dudz_lower = jnp.concatenate((lower_dudz[None], dudz_upper[:-1]), axis=0)
        dvdz_lower = jnp.concatenate((lower_dvdz[None], dvdz_upper[:-1]), axis=0)

        dudx = horizontal_derivative_local(u, 0)
        dudy = horizontal_derivative_local(u, 1)
        dvdx = horizontal_derivative_local(v, 0)
        dvdy = horizontal_derivative_local(v, 1)
        dwdx_at_cells = horizontal_derivative_local(w_at_cells, 0)
        dwdy_at_cells = horizontal_derivative_local(w_at_cells, 1)
        dwdz = (w_upper - lower_faces) / grid.dz
        dwdx_upper = horizontal_derivative_local(w_upper, 0)
        dwdy_upper = horizontal_derivative_local(w_upper, 1)

        next_dudx = jnp.concatenate(
            (dudx[1:], horizontal_derivative_local(next_u_plane, 0)[None]),
            axis=0,
        )
        next_dudy = jnp.concatenate(
            (dudy[1:], horizontal_derivative_local(next_u_plane, 1)[None]),
            axis=0,
        )
        next_dvdx = jnp.concatenate(
            (dvdx[1:], horizontal_derivative_local(next_v_plane, 0)[None]),
            axis=0,
        )
        next_dvdy = jnp.concatenate(
            (dvdy[1:], horizontal_derivative_local(next_v_plane, 1)[None]),
            axis=0,
        )
        next_dwdz_plane = (next_w_upper - w_upper[-1]) / grid.dz
        next_dwdz = jnp.concatenate((dwdz[1:], next_dwdz_plane[None]), axis=0)
        next_w_cell_plane = 0.5 * (w_upper[-1] + next_w_upper)
        next_w_cell = jnp.concatenate(
            (w_at_cells[1:], next_w_cell_plane[None]),
            axis=0,
        )
        return ZSlabDryFlowArrays(
            u,
            v,
            w_upper,
            w_lower_plane,
            u_upper,
            v_upper,
            u_lower,
            v_lower,
            w_at_cells,
            next_w_cell,
            dudx,
            dudy,
            0.5 * (dudz_lower + dudz_upper),
            dvdx,
            dvdy,
            0.5 * (dvdz_lower + dvdz_upper),
            dwdx_at_cells,
            dwdy_at_cells,
            dwdz,
            dudz_upper,
            dvdz_upper,
            dwdx_upper,
            dwdy_upper,
            0.5 * (dudx + next_dudx),
            0.5 * (dudy + next_dudy),
            0.5 * (dvdx + next_dvdx),
            0.5 * (dvdy + next_dvdy),
            0.5 * (dwdz + next_dwdz),
            halo.upper_is_physical,
        )

    def scalar_context_local(theta):
        halo = exchange_local(theta[None, ...])
        previous_plane = jnp.where(
            halo.lower_is_physical,
            theta[0],
            halo.lower[0],
        )
        next_plane = jnp.where(
            halo.upper_is_physical,
            theta[-1],
            halo.upper[0],
        )
        next_theta = jnp.concatenate((theta[1:], next_plane[None]), axis=0)
        theta_upper = 0.5 * (theta + next_theta)
        theta_lower = jnp.concatenate(
            ((0.5 * (previous_plane + theta[0]))[None], theta_upper[:-1]),
            axis=0,
        )
        dtheta_dz_upper = (next_theta - theta) / grid.dz
        dtheta_dz_upper = dtheta_dz_upper.at[-1].set(
            jnp.where(halo.upper_is_physical, 0.0, dtheta_dz_upper[-1])
        )
        previous_theta = jnp.concatenate((previous_plane[None], theta[:-1]), axis=0)
        centered_dtheta_dz = (next_theta - previous_theta) / (2.0 * grid.dz)
        centered_dtheta_dz = centered_dtheta_dz.at[0].set(
            jnp.where(
                halo.lower_is_physical,
                (next_theta[0] - theta[0]) / grid.dz,
                centered_dtheta_dz[0],
            )
        )
        centered_dtheta_dz = centered_dtheta_dz.at[-1].set(
            jnp.where(
                halo.upper_is_physical,
                (theta[-1] - previous_theta[-1]) / grid.dz,
                centered_dtheta_dz[-1],
            )
        )
        return ZSlabScalarArrays(
            theta,
            theta_upper,
            theta_lower,
            horizontal_derivative_local(theta, 0),
            horizontal_derivative_local(theta, 1),
            centered_dtheta_dz,
            dtheta_dz_upper,
            halo.upper_is_physical,
        )

    def buoyancy_local(scalar, coefficient):
        local_coefficient = jnp.asarray(coefficient, dtype=scalar.theta.dtype)
        hydrostatic_free_theta = scalar.theta_upper - jnp.mean(
            scalar.theta_upper,
            axis=(-2, -1),
            keepdims=True,
        )
        z = local_coefficient * hydrostatic_free_theta
        return z.at[-1].set(jnp.where(scalar.upper_is_physical, 0.0, z[-1]))

    def rayleigh_damping_local(
        u,
        v,
        w_upper,
        start_height,
        maximum_rate,
        target_u,
        target_v,
    ):
        index = lax.axis_index(axis_name)
        local_nz = u.shape[0]
        global_cell = index * local_nz + jnp.arange(local_nz, dtype=u.dtype)
        cell_height = (global_cell + 0.5) * grid.dz
        upper_face_height = (global_cell + 1.0) * grid.dz
        depth = grid.lz - jnp.asarray(start_height, dtype=u.dtype)
        cell_eta = jnp.clip((cell_height - start_height) / depth, 0.0, 1.0)
        face_eta = jnp.clip(
            (upper_face_height - start_height) / depth,
            0.0,
            1.0,
        )
        cell_rate = jnp.asarray(maximum_rate, dtype=u.dtype) * cell_eta**2
        face_rate = jnp.asarray(maximum_rate, dtype=w_upper.dtype) * (
            face_eta.astype(w_upper.dtype) ** 2
        )
        return (
            -cell_rate[:, None, None] * (u - target_u),
            -cell_rate[:, None, None] * (v - target_v),
            -face_rate[:, None, None] * w_upper,
        )

    def scalar_advection_local(scalar, momentum):
        w_lower = jnp.concatenate(
            (momentum.w_lower[None], momentum.w_upper[:-1]),
            axis=0,
        )
        upper_flux = two_thirds_filter_local(momentum.w_upper * scalar.theta_upper)
        lower_flux = two_thirds_filter_local(w_lower * scalar.theta_lower)
        return -(
            truncated_derivative_local(momentum.u * scalar.theta, 0)
            + truncated_derivative_local(momentum.v * scalar.theta, 1)
            + (upper_flux - lower_flux) / grid.dz
        )

    def scalar_sgs_local(
        scalar,
        momentum,
        coefficient,
        lower_boundary_flux,
        upper_boundary_flux,
        stability_buoyancy_coefficient,
        stability_beta,
        stability_power,
    ):
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        cell_magnitude = strain_magnitude_local(
            momentum.dudx,
            momentum.dudy,
            momentum.dudz_at_cells,
            momentum.dvdx,
            momentum.dvdy,
            momentum.dvdz_at_cells,
            momentum.dwdx_at_cells,
            momentum.dwdy_at_cells,
            momentum.dwdz,
        )
        face_magnitude = strain_magnitude_local(
            momentum.dudx_upper,
            momentum.dudy_upper,
            momentum.dudz_upper,
            momentum.dvdx_upper,
            momentum.dvdy_upper,
            momentum.dvdz_upper,
            momentum.dwdx_upper,
            momentum.dwdy_upper,
            momentum.dwdz_upper,
        )
        local_coefficient = coefficient.astype(scalar.theta.dtype)
        n2 = jnp.maximum(
            jnp.asarray(stability_buoyancy_coefficient, dtype=scalar.theta.dtype)
            * scalar.dtheta_dz_at_cells,
            0.0,
        )
        richardson = n2 / jnp.maximum(cell_magnitude**2, 1.0e-24)
        stability = (
            1.0
            + jnp.asarray(stability_beta, dtype=scalar.theta.dtype) * richardson
        ) ** (-jnp.asarray(stability_power, dtype=scalar.theta.dtype))
        effective_coefficient = local_coefficient * stability
        coefficient_halo = exchange_local(effective_coefficient[None, ...])
        next_coefficient_plane = jnp.where(
            coefficient_halo.upper_is_physical,
            effective_coefficient[-1],
            coefficient_halo.upper[0],
        )
        next_coefficient = jnp.concatenate(
            (effective_coefficient[1:], next_coefficient_plane[None]),
            axis=0,
        )
        face_coefficient = 0.5 * (effective_coefficient + next_coefficient)
        cell_diffusivity = effective_coefficient * delta**2 * cell_magnitude
        face_diffusivity = face_coefficient * delta**2 * face_magnitude
        qx = -cell_diffusivity * scalar.dtheta_dx
        qy = -cell_diffusivity * scalar.dtheta_dy
        qz = -face_diffusivity * scalar.dtheta_dz_upper
        qz = qz.at[-1].set(
            jnp.where(scalar.upper_is_physical, upper_boundary_flux, qz[-1])
        )
        qz = two_thirds_filter_local(qz)
        flux_halo = exchange_local(qz[None, ...])
        lower_plane = jnp.where(
            flux_halo.lower_is_physical,
            jnp.full_like(qz[0], lower_boundary_flux),
            flux_halo.lower[0],
        )
        lower_qz = jnp.concatenate((lower_plane[None], qz[:-1]), axis=0)
        return -(
            truncated_derivative_local(qx, 0)
            + truncated_derivative_local(qy, 1)
            + (qz - lower_qz) / grid.dz
        )

    def dry_advection_local(context):
        upper_u_flux = two_thirds_filter_local(context.w_upper * context.u_upper)
        upper_v_flux = two_thirds_filter_local(context.w_upper * context.v_upper)
        lower_u_flux_plane = two_thirds_filter_local(context.w_lower * context.u_lower)
        lower_v_flux_plane = two_thirds_filter_local(context.w_lower * context.v_lower)
        lower_u_flux = jnp.concatenate(
            (lower_u_flux_plane[None], upper_u_flux[:-1]),
            axis=0,
        )
        lower_v_flux = jnp.concatenate(
            (lower_v_flux_plane[None], upper_v_flux[:-1]),
            axis=0,
        )
        x = -(
            truncated_derivative_local(context.u * context.u, 0)
            + truncated_derivative_local(context.v * context.u, 1)
            + (upper_u_flux - lower_u_flux) / grid.dz
        )
        y = -(
            truncated_derivative_local(context.u * context.v, 0)
            + truncated_derivative_local(context.v * context.v, 1)
            + (upper_v_flux - lower_v_flux) / grid.dz
        )
        vertical_flux = two_thirds_filter_local(context.w_at_cells * context.w_at_cells)
        next_vertical_flux = two_thirds_filter_local(
            context.w_next_cell * context.w_next_cell
        )
        z = -(
            truncated_derivative_local(context.u_upper * context.w_upper, 0)
            + truncated_derivative_local(context.v_upper * context.w_upper, 1)
            + (next_vertical_flux - vertical_flux) / grid.dz
        )
        z = z.at[-1].set(jnp.where(context.upper_is_physical, 0.0, z[-1]))
        return x, y, z

    def dry_wall_local(context, drag, filtered, filter_width):
        index = lax.axis_index(axis_name)
        wall_velocity = wall_filter_local(
            jnp.stack((context.u[0], context.v[0])),
            filter_width,
        )
        wall_u = jnp.where(filtered, wall_velocity[0], context.u[0])
        wall_v = jnp.where(filtered, wall_velocity[1], context.v[0])
        speed = jnp.sqrt(wall_u * wall_u + wall_v * wall_v)
        wall_x = -drag * speed * wall_u / grid.dz
        wall_y = -drag * speed * wall_v / grid.dz
        x = jnp.zeros_like(context.u).at[0].set(jnp.where(index == 0, wall_x, 0.0))
        y = jnp.zeros_like(context.v).at[0].set(jnp.where(index == 0, wall_y, 0.0))
        return x, y, jnp.zeros_like(context.w_upper)

    def dry_sgs_vertical_flux_local(context, coefficient):
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        face_magnitude = strain_magnitude_local(
            context.dudx_upper,
            context.dudy_upper,
            context.dudz_upper,
            context.dvdx_upper,
            context.dvdy_upper,
            context.dvdz_upper,
            context.dwdx_upper,
            context.dwdy_upper,
            context.dwdz_upper,
        )
        coefficient_halo = exchange_local(coefficient[None, ...])
        next_coefficient_plane = jnp.where(
            coefficient_halo.upper_is_physical,
            coefficient[-1],
            coefficient_halo.upper[0],
        )
        next_coefficient = jnp.concatenate(
            (coefficient[1:], next_coefficient_plane[None]),
            axis=0,
        )
        face_viscosity = (
            0.5 * (coefficient + next_coefficient) * delta**2 * face_magnitude
        )
        txz = -face_viscosity * (context.dudz_upper + context.dwdx_upper)
        tyz = -face_viscosity * (context.dvdz_upper + context.dwdy_upper)
        txz = txz.at[-1].set(jnp.where(context.upper_is_physical, 0.0, txz[-1]))
        tyz = tyz.at[-1].set(jnp.where(context.upper_is_physical, 0.0, tyz[-1]))
        txz = two_thirds_filter_local(txz)
        tyz = two_thirds_filter_local(tyz)
        return txz, tyz

    def dry_sgs_local(context, coefficient):
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        cell_magnitude = strain_magnitude_local(
            context.dudx,
            context.dudy,
            context.dudz_at_cells,
            context.dvdx,
            context.dvdy,
            context.dvdz_at_cells,
            context.dwdx_at_cells,
            context.dwdy_at_cells,
            context.dwdz,
        )
        cell_viscosity = coefficient * delta**2 * cell_magnitude
        txx = -2.0 * cell_viscosity * context.dudx
        txy = -cell_viscosity * (context.dudy + context.dvdx)
        tyy = -2.0 * cell_viscosity * context.dvdy
        tzz = -2.0 * cell_viscosity * context.dwdz
        txz, tyz = dry_sgs_vertical_flux_local(context, coefficient)
        tzz = two_thirds_filter_local(tzz)

        stress_halo = exchange_local(jnp.stack((txz, tyz, tzz), axis=0))
        lower_txz_plane = jnp.where(
            stress_halo.lower_is_physical,
            jnp.zeros_like(txz[0]),
            stress_halo.lower[0],
        )
        lower_tyz_plane = jnp.where(
            stress_halo.lower_is_physical,
            jnp.zeros_like(tyz[0]),
            stress_halo.lower[1],
        )
        lower_txz = jnp.concatenate((lower_txz_plane[None], txz[:-1]), axis=0)
        lower_tyz = jnp.concatenate((lower_tyz_plane[None], tyz[:-1]), axis=0)
        next_tzz_plane = jnp.where(
            stress_halo.upper_is_physical,
            tzz[-1],
            stress_halo.upper[2],
        )
        next_tzz = jnp.concatenate((tzz[1:], next_tzz_plane[None]), axis=0)
        x = -(
            truncated_derivative_local(txx, 0)
            + truncated_derivative_local(txy, 1)
            + (txz - lower_txz) / grid.dz
        )
        y = -(
            truncated_derivative_local(txy, 0)
            + truncated_derivative_local(tyy, 1)
            + (tyz - lower_tyz) / grid.dz
        )
        z = -(
            truncated_derivative_local(txz, 0)
            + truncated_derivative_local(tyz, 1)
            + (next_tzz - tzz) / grid.dz
        )
        z = z.at[-1].set(jnp.where(stress_halo.upper_is_physical, 0.0, z[-1]))
        return x, y, z

    def horizontal_divergence_local(x_velocity, y_velocity):
        x_spectrum = jnp.fft.rfftn(x_velocity, axes=(-2, -1))
        y_spectrum = jnp.fft.rfftn(y_velocity, axes=(-2, -1))
        spectrum = (
            1j * kx[None, None, :] * x_spectrum + 1j * ky[None, :, None] * y_spectrum
        ) * keep[None, ...]
        return jnp.fft.irfftn(
            spectrum,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(x_velocity.dtype)

    def horizontal_gradient_local(pressure):
        spectrum = jnp.fft.rfftn(pressure, axes=(-2, -1)) * keep[None, ...]
        gradient_x = jnp.fft.irfftn(
            spectrum * (1j * kx[None, None, :]),
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(pressure.dtype)
        gradient_y = jnp.fft.irfftn(
            spectrum * (1j * ky[None, :, None]),
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(pressure.dtype)
        return gradient_x, gradient_y

    def filter_horizontal_local(values):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        return jnp.fft.irfftn(
            spectrum * keep[None, ...],
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def filter_boundary(boundary, dtype_probe):
        plane = jnp.broadcast_to(
            jnp.asarray(boundary, dtype=dtype_probe.dtype),
            (grid.ny, grid.nx),
        )
        spectrum = jnp.fft.rfftn(plane, axes=(-2, -1))
        return jnp.fft.irfftn(
            spectrum * keep,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(plane.dtype)

    def correct_local(candidate, gradient, dt):
        local_dt = jnp.asarray(dt, dtype=candidate.dtype)
        return candidate - local_dt * gradient

    def ab2_update_local(
        state,
        current_tendency,
        previous_tendency,
        dt,
        current_weight,
        previous_weight,
    ):
        local_dt = jnp.asarray(dt, dtype=state.dtype)
        current_coefficient = jnp.asarray(current_weight, dtype=state.dtype)
        previous_coefficient = jnp.asarray(previous_weight, dtype=state.dtype)
        return state + local_dt * (
            current_coefficient * current_tendency
            + previous_coefficient * previous_tendency
        )

    def combine_payloads_local(payloads):
        total = payloads[0]
        for payload in payloads[1:]:
            total = total + payload
        return total

    def relax_lasd_field_local(current, target, blend):
        return current + blend * (target - current)

    (
        lasd_diagnostics_local,
        lasd_accumulate_local,
        lasd_accumulate_velocity_local,
        lasd_update_local,
    ) = build_lasd_kernels(
        grid=grid,
        axis_name=axis_name,
        shard_count=shard_count,
        exchange_local=exchange_local,
        strain_magnitude_local=strain_magnitude_local,
        two_thirds_filter_local=two_thirds_filter_local,
    )

    wind_tunnel_local = build_wind_tunnel_kernel(
        grid=grid,
        axis_name=axis_name,
    )
    actuator_line_local = build_actuator_line_kernel(
        grid=grid,
        axis_name=axis_name,
        shard_count=shard_count,
    )

    mapped = partial(jax.pmap, axis_name=axis_name, axis_size=shard_count)
    exchange_packed = mapped(exchange_local)
    pressure_gradient = mapped(pressure_gradient_local, in_axes=(0, None))
    divergence = mapped(divergence_local, in_axes=(0, None))
    enforce_upper_boundary = mapped(
        enforce_upper_boundary_local,
        in_axes=(0, None),
    )
    horizontal_divergence = mapped(
        horizontal_divergence_local,
        in_axes=(0, 0),
    )
    horizontal_gradient = mapped(horizontal_gradient_local)
    filter_horizontal = mapped(filter_horizontal_local)
    correct = mapped(correct_local, in_axes=(0, 0, None))
    ab2_update = mapped(
        ab2_update_local,
        in_axes=(0, 0, 0, None, None, None),
    )
    # Stage composite elementwise expressions once so Python composition does
    # not dispatch one full-grid GPU program for every arithmetic operator.
    combine_payloads = jax.jit(combine_payloads_local)
    relax_lasd_field = jax.jit(relax_lasd_field_local)
    wind_tunnel = mapped(
        wind_tunnel_local,
        in_axes=(0, 0, 0, 0, 0, 0) + (None,) * 16,
        # Geometry and fringe parameters are invariant throughout a case.
        static_broadcasted_argnums=tuple(range(6, 22)),
    )
    actuator_line = mapped(
        actuator_line_local,
        in_axes=(0, 0, 0) + (None,) * 31,
        static_broadcasted_argnums=(8,),
    )
    dry_flow_context = mapped(
        dry_flow_context_local,
        in_axes=(0, 0, 0, None),
    )
    dry_advection = mapped(dry_advection_local)
    dry_wall = mapped(dry_wall_local, in_axes=(0, None, None, None))
    dry_sgs = mapped(dry_sgs_local, in_axes=(0, 0))
    dry_sgs_vertical_flux = mapped(
        dry_sgs_vertical_flux_local,
        in_axes=(0, 0),
    )
    lasd_accumulate = mapped(
        lasd_accumulate_local,
        in_axes=(0, 0, 0, 0, 0, 0, None),
    )
    lasd_accumulate_velocity = mapped(
        lasd_accumulate_velocity_local,
        in_axes=(0, 0, 0, None, 0, 0, 0, None),
    )
    lasd_update = mapped(
        lasd_update_local,
        in_axes=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    lasd_diagnostics = mapped(
        lasd_diagnostics_local,
        in_axes=(
            0,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    scalar_context = mapped(scalar_context_local)
    scalar_advection = mapped(scalar_advection_local, in_axes=(0, 0))
    scalar_sgs = mapped(
        scalar_sgs_local,
        in_axes=(0, 0, 0, None, None, None, None, None),
    )
    buoyancy = mapped(buoyancy_local, in_axes=(0, None))
    rayleigh_damping = mapped(
        rayleigh_damping_local,
        in_axes=(0, 0, 0, None, None, None, None),
    )
    return JaxZSlabInterpreter(
        decomposition,
        addressable_shards,
        exchange_packed,
        pressure_gradient,
        divergence,
        enforce_upper_boundary,
        horizontal_divergence,
        horizontal_gradient,
        filter_horizontal,
        jax.jit(filter_boundary),
        correct,
        ab2_update,
        combine_payloads,
        relax_lasd_field,
        wind_tunnel,
        actuator_line,
        dry_flow_context,
        dry_advection,
        dry_wall,
        dry_sgs,
        dry_sgs_vertical_flux,
        lasd_accumulate,
        lasd_accumulate_velocity,
        lasd_update,
        lasd_diagnostics,
        scalar_context,
        scalar_advection,
        scalar_sgs,
        buoyancy,
        rayleigh_damping,
    )
