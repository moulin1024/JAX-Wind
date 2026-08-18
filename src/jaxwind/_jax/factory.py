"""Compilation of mapped kernels for the private JAX discretization."""

from __future__ import annotations

from functools import partial
import math

import jax
import jax.numpy as jnp
from jax import lax

from jaxwind.domain import EqualVerticalPartition

from .wind import (
    build_actuator_line_kernel,
    build_blade_element_disk_kernel,
    build_nacelle_tower_kernel,
    build_wind_tunnel_kernel,
)
from .lasd_kernels import build_lasd_kernels
from .fused_boussinesq import build_fused_neutral_boussinesq_kernels
from .rotational import build_rotational_advection_kernel
from .projection import build_projection_kernels
from .discretization import (
    _JaxDiscretization,
    PackedHaloArrays,
    ZSlabDryFlowArrays,
    ZSlabScalarArrays,
    _FlowKernels,
    _LasdKernels,
    _ProjectionKernels,
    _ScalarKernels,
    _WindKernels,
)
from .smagorinsky import build_smagorinsky_kernels
from .spectral import build_horizontal_spectral_kernels
from .sources import (
    build_boussinesq_source_kernels,
    strain_magnitude_local,
)
from .surface import build_monin_obukhov_surface_transfer_kernel


def build_discretization(
    decomposition: EqualVerticalPartition,
    *,
    addressable_partitions: tuple[int, ...] | None = None,
    axis_name: str = "jaxwind_z",
    porte_agel_wall_correction: bool = True,
    frozen_zero_scalar: bool = False,
    lasd_filter_backend: str = "jax",
) -> _JaxDiscretization:
    """Build mapped kernels with legacy WIRE-LES nonlinear ordering."""
    if not isinstance(porte_agel_wall_correction, bool):
        raise TypeError("Porté-Agel wall correction flag must be boolean")
    if not isinstance(frozen_zero_scalar, bool):
        raise TypeError("frozen zero scalar flag must be boolean")
    if lasd_filter_backend not in ("jax", "cufft"):
        raise ValueError("LASD filter backend must be jax or cufft")
    partition_count = decomposition.partition_count
    if addressable_partitions is None:
        if partition_count != 1:
            raise ValueError(
                "addressable_partitions is required for a multi-shard decomposition"
            )
        addressable_partitions = (0,)
    if not addressable_partitions:
        raise ValueError("at least one addressable shard is required")
    if len(set(addressable_partitions)) != len(addressable_partitions):
        raise ValueError("addressable partition ids must be unique")
    if any(not 0 <= index < partition_count for index in addressable_partitions):
        raise ValueError("addressable partition id is outside the global mesh")

    previous_permutation = tuple(
        (source, source + 1) for source in range(partition_count - 1)
    )
    next_permutation = tuple(
        (source, source - 1) for source in range(1, partition_count)
    )

    def exchange_local(packed):
        if packed.ndim != 4:
            raise ValueError("packed local fields must have shape (field, z, y, x)")
        index = lax.axis_index(axis_name)
        if partition_count == 1:
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
        upper_physical = index == partition_count - 1
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
            index == partition_count - 1,
            boundary_face,
            upper_faces[-1],
        )
        return upper_faces.at[-1].set(last)

    grid = decomposition.grid
    spectral = build_horizontal_spectral_kernels(grid)
    kx, ky, keep, state_keep = (
        spectral.kx,
        spectral.ky,
        spectral.keep,
        spectral.state_keep,
    )
    horizontal_spectrum_local = spectral.spectrum
    horizontal_gradient_pair_local = spectral.gradient_pair
    horizontal_spectral_flux_divergence_local = spectral.spectral_flux_divergence
    horizontal_flux_divergence_local = spectral.flux_divergence
    wall_filter_local = spectral.wall_filter
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
        porte_agel_factor = 1.0 / math.log(3.0) - 1.0
        corrected_dudz = dudz_upper[0] + porte_agel_factor * jnp.mean(dudz_upper[0])
        corrected_dvdz = dvdz_upper[0] + porte_agel_factor * jnp.mean(dvdz_upper[0])
        dudz_upper = dudz_upper.at[0].set(
            jnp.where(
                (index == 0) & porte_agel_wall_correction,
                corrected_dudz,
                dudz_upper[0],
            )
        )
        dvdz_upper = dvdz_upper.at[0].set(
            jnp.where(
                (index == 0) & porte_agel_wall_correction,
                corrected_dvdz,
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

        horizontal_x, horizontal_y = horizontal_gradient_pair_local(
            jnp.stack((u, v, w_at_cells, w_upper), axis=0)
        )
        dudx, dvdx, dwdx_at_cells, dwdx_upper = horizontal_x
        dudy, dvdy, dwdy_at_cells, dwdy_upper = horizontal_y
        dwdz = (w_upper - lower_faces) / grid.dz
        next_horizontal_x, next_horizontal_y = horizontal_gradient_pair_local(
            jnp.stack((next_u_plane, next_v_plane), axis=0)
        )
        next_dudx = jnp.concatenate(
            (dudx[1:], next_horizontal_x[0][None]),
            axis=0,
        )
        next_dudy = jnp.concatenate(
            (dudy[1:], next_horizontal_y[0][None]),
            axis=0,
        )
        next_dvdx = jnp.concatenate(
            (dvdx[1:], next_horizontal_x[1][None]),
            axis=0,
        )
        next_dvdy = jnp.concatenate(
            (dvdy[1:], next_horizontal_y[1][None]),
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
        dtheta_dx, dtheta_dy = horizontal_gradient_pair_local(theta)
        return ZSlabScalarArrays(
            theta,
            theta_upper,
            theta_lower,
            dtheta_dx,
            dtheta_dy,
            centered_dtheta_dz,
            dtheta_dz_upper,
            halo.upper_is_physical,
        )

    buoyancy_local, rayleigh_damping_local = build_boussinesq_source_kernels(
        grid=grid, axis_name=axis_name
    )

    def scalar_advection_from_fields_local(
        u,
        v,
        w_upper,
        w_lower,
        theta,
        theta_upper,
        theta_lower,
    ):
        horizontal = horizontal_flux_divergence_local(
            (u * theta)[None],
            (v * theta)[None],
        )[0]
        upper_flux = w_upper * theta_upper
        lower_flux = w_lower * theta_lower
        return -(horizontal + (upper_flux - lower_flux) / grid.dz)

    def scalar_advection_from_momentum_local(
        scalar,
        momentum_fields,
        lower_fields,
    ):
        w_lower = jnp.concatenate(
            (lower_fields[0][None], momentum_fields[2][:-1]),
            axis=0,
        )
        return scalar_advection_from_fields_local(
            momentum_fields[0],
            momentum_fields[1],
            momentum_fields[2],
            w_lower,
            scalar.theta,
            scalar.theta_upper,
            scalar.theta_lower,
        )

    def scalar_advection_local(scalar, momentum):
        w_lower = jnp.concatenate(
            (momentum.w_lower[None], momentum.w_upper[:-1]),
            axis=0,
        )
        return scalar_advection_from_fields_local(
            momentum.u,
            momentum.v,
            momentum.w_upper,
            w_lower,
            scalar.theta,
            scalar.theta_upper,
            scalar.theta_lower,
        )

    def scalar_sgs_from_momentum_gradients_local(
        scalar,
        momentum,
        momentum_gradients,
        face_gradients,
        coefficient,
        minimum_coefficient,
        maximum_coefficient,
        lower_boundary_flux,
        upper_boundary_flux,
        stability_buoyancy_coefficient,
        stability_beta,
        stability_power,
    ):
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        cell_magnitude = strain_magnitude_local(
            *momentum_gradients,
        )
        face_magnitude = strain_magnitude_local(
            *face_gradients,
        )
        local_coefficient = jnp.clip(
            coefficient.astype(scalar.theta.dtype),
            minimum_coefficient,
            maximum_coefficient,
        )
        n2 = jnp.maximum(
            jnp.asarray(stability_buoyancy_coefficient, dtype=scalar.theta.dtype)
            * scalar.dtheta_dz_at_cells,
            0.0,
        )
        richardson = n2 / jnp.maximum(cell_magnitude**2, 1.0e-24)
        stability = (
            1.0 + jnp.asarray(stability_beta, dtype=scalar.theta.dtype) * richardson
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
        flux_halo = exchange_local(qz[None, ...])
        lower_plane = jnp.where(
            flux_halo.lower_is_physical,
            jnp.full_like(qz[0], lower_boundary_flux),
            flux_halo.lower[0],
        )
        lower_qz = jnp.concatenate((lower_plane[None], qz[:-1]), axis=0)
        horizontal = horizontal_flux_divergence_local(qx[None], qy[None])[0]
        return -(horizontal + (qz - lower_qz) / grid.dz)

    def scalar_sgs_local(
        scalar,
        momentum,
        coefficient,
        minimum_coefficient,
        maximum_coefficient,
        lower_boundary_flux,
        upper_boundary_flux,
        stability_buoyancy_coefficient,
        stability_beta,
        stability_power,
    ):
        momentum_gradients = tuple(
            jnp.stack(
                (
                    momentum.dudx,
                    momentum.dudy,
                    momentum.dudz_at_cells,
                    momentum.dvdx,
                    momentum.dvdy,
                    momentum.dvdz_at_cells,
                    momentum.dwdx_at_cells,
                    momentum.dwdy_at_cells,
                    momentum.dwdz,
                ),
                axis=0,
            )
        )
        face_gradients = tuple(
            jnp.stack(
                (
                    momentum.dudx_upper,
                    momentum.dudy_upper,
                    momentum.dudz_upper,
                    momentum.dvdx_upper,
                    momentum.dvdy_upper,
                    momentum.dvdz_upper,
                    momentum.dwdx_upper,
                    momentum.dwdy_upper,
                    momentum.dwdz_upper,
                ),
                axis=0,
            )
        )
        return scalar_sgs_from_momentum_gradients_local(
            scalar,
            momentum,
            momentum_gradients,
            face_gradients,
            coefficient,
            minimum_coefficient,
            maximum_coefficient,
            lower_boundary_flux,
            upper_boundary_flux,
            stability_buoyancy_coefficient,
            stability_beta,
            stability_power,
        )

    def dry_wall_local(context, drag, filtered, filter_width):
        index = lax.axis_index(axis_name)
        wall_velocity = wall_filter_local(
            jnp.stack((context.u[0], context.v[0])),
            filter_width,
        )
        wall_u = jnp.where(filtered, wall_velocity[0], context.u[0])
        wall_v = jnp.where(filtered, wall_velocity[1], context.v[0])
        speed = jnp.hypot(wall_u, wall_v)
        wall_x = -drag * speed * wall_u / grid.dz
        wall_y = -drag * speed * wall_v / grid.dz
        x = jnp.zeros_like(context.u).at[0].set(jnp.where(index == 0, wall_x, 0.0))
        y = jnp.zeros_like(context.v).at[0].set(jnp.where(index == 0, wall_y, 0.0))
        return x, y, jnp.zeros_like(context.w_upper)

    filter_two_scales_external = None
    if lasd_filter_backend == "cufft":
        from .lasd_cufft import filter_two_scales as filter_two_scales_external

    (
        dry_sgs_local,
        dry_sgs_vertical_flux_local,
        dry_sgs_from_gradients_local,
        dry_sgs_tke_transfer_local,
    ) = build_smagorinsky_kernels(
        grid=grid,
        axis_name=axis_name,
        exchange_local=exchange_local,
        strain_magnitude_local=strain_magnitude_local,
        horizontal_spectrum_local=horizontal_spectrum_local,
        horizontal_spectral_flux_divergence_local=(
            horizontal_spectral_flux_divergence_local
        ),
    )

    dry_rotational_advection_local = build_rotational_advection_kernel(
        grid=grid,
        axis_name=axis_name,
        wall_filter_local=wall_filter_local,
    )
    (
        fused_lasd_boussinesq_local,
        fused_lasd_boussinesq_from_context_local,
    ) = build_fused_neutral_boussinesq_kernels(
        grid=grid,
        axis_name=axis_name,
        frozen_zero_scalar=frozen_zero_scalar,
        scalar_context_local=scalar_context_local,
        scalar_advection_from_momentum_local=(
            scalar_advection_from_momentum_local
        ),
        scalar_sgs_from_momentum_gradients_local=(
            scalar_sgs_from_momentum_gradients_local
        ),
        buoyancy_local=buoyancy_local,
        wall_filter_local=wall_filter_local,
        dry_flow_context_local=dry_flow_context_local,
        dry_sgs_from_gradients_local=dry_sgs_from_gradients_local,
    )
    (
        horizontal_divergence_local,
        horizontal_gradient_local,
        filter_horizontal_local,
        filter_velocity_local,
        filter_projection_velocity_local,
        prepare_projection_local,
        finish_projection_local,
        filter_boundary,
        correct_local,
    ) = build_projection_kernels(
        grid=grid,
        kx=kx,
        ky=ky,
        keep=keep,
        state_keep=state_keep,
        divergence_local=divergence_local,
        enforce_upper_boundary_local=enforce_upper_boundary_local,
        pressure_gradient_local=pressure_gradient_local,
    )

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

    def ab2_update_velocity_local(
        state_x,
        state_y,
        state_z,
        current_x,
        current_y,
        current_z,
        previous_x,
        previous_y,
        previous_z,
        dt,
        current_weight,
        previous_weight,
    ):
        return (
            ab2_update_local(
                state_x,
                current_x,
                previous_x,
                dt,
                current_weight,
                previous_weight,
            ),
            ab2_update_local(
                state_y,
                current_y,
                previous_y,
                dt,
                current_weight,
                previous_weight,
            ),
            ab2_update_local(
                state_z,
                current_z,
                previous_z,
                dt,
                current_weight,
                previous_weight,
            ),
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
        lasd_update_momentum_local,
    ) = build_lasd_kernels(
        grid=grid,
        axis_name=axis_name,
        partition_count=partition_count,
        exchange_local=exchange_local,
        strain_magnitude_local=strain_magnitude_local,
        filter_two_scales_external=filter_two_scales_external,
    )

    wind_tunnel_local = build_wind_tunnel_kernel(
        grid=grid,
        axis_name=axis_name,
    )
    actuator_line_local = build_actuator_line_kernel(
        grid=grid,
        axis_name=axis_name,
        partition_count=partition_count,
    )
    actuator_disk_bem_local = build_blade_element_disk_kernel(
        grid=grid,
        axis_name=axis_name,
        partition_count=partition_count,
    )
    nacelle_tower_local = build_nacelle_tower_kernel(
        grid=grid,
        axis_name=axis_name,
    )

    mapped = partial(jax.pmap, axis_name=axis_name, axis_size=partition_count)
    exchange_packed = mapped(exchange_local)
    pressure_gradient = mapped(pressure_gradient_local, in_axes=(0, None))
    divergence = mapped(divergence_local, in_axes=(0, None))
    enforce_upper_boundary = mapped(
        enforce_upper_boundary_local,
        in_axes=(0, None),
    )
    prepare_projection = mapped(
        prepare_projection_local,
        in_axes=(0, 0, 0, None, None),
    )
    finish_projection = mapped(
        finish_projection_local,
        in_axes=(0, 0, 0, 0, None),
    )
    horizontal_divergence = mapped(
        horizontal_divergence_local,
        in_axes=(0, 0),
    )
    horizontal_gradient = mapped(horizontal_gradient_local)
    filter_state = mapped(filter_velocity_local, in_axes=(0, 0, 0))
    filter_horizontal = mapped(
        filter_projection_velocity_local,
        in_axes=(0, 0, 0),
    )
    filter_scalar = mapped(filter_horizontal_local)
    correct = mapped(
        correct_local,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
    )
    ab2_update = mapped(
        ab2_update_local,
        in_axes=(0, 0, 0, None, None, None),
    )
    ab2_update_velocity = mapped(
        ab2_update_velocity_local,
        in_axes=(0,) * 9 + (None,) * 3,
    )
    # Stage composite elementwise expressions once so Python composition does
    # not dispatch one full-grid GPU program for every arithmetic operator.
    combine_payloads = jax.jit(combine_payloads_local)
    relax_lasd_field = jax.jit(relax_lasd_field_local)
    wind_tunnel = mapped(
        wind_tunnel_local,
        in_axes=(0, 0, 0, 0, 0, 0) + (None,) * 18,
        # Geometry and fringe parameters are invariant throughout a case.
        static_broadcasted_argnums=tuple(range(6, 24)),
    )
    actuator_line = mapped(
        actuator_line_local,
        in_axes=(0, 0, 0) + (None,) * 31,
        static_broadcasted_argnums=(8,),
    )
    actuator_disk_bem = mapped(
        actuator_disk_bem_local,
        in_axes=(0, 0, 0) + (None,) * 19,
        static_broadcasted_argnums=(6,),
    )
    nacelle_tower = mapped(
        nacelle_tower_local,
        in_axes=(0, 0) + (None,) * 10,
    )
    dry_flow_context = mapped(
        dry_flow_context_local,
        in_axes=(0, 0, 0, None),
    )
    dry_rotational_advection = mapped(
        dry_rotational_advection_local,
        in_axes=(0, None, None, None, None, None),
    )
    dry_wall = mapped(dry_wall_local, in_axes=(0, None, None, None))
    dry_sgs = mapped(dry_sgs_local, in_axes=(0, 0, None, None))
    dry_sgs_vertical_flux = mapped(
        dry_sgs_vertical_flux_local,
        in_axes=(0, 0, None, None),
    )
    dry_sgs_tke_transfer = mapped(
        dry_sgs_tke_transfer_local,
        in_axes=(0, 0, None, None, None),
    )
    fused_lasd_boussinesq = mapped(
        fused_lasd_boussinesq_local,
        in_axes=(0, 0, 0, None, 0, 0, 0)
        + (None,) * 19
        + (0, 0, 0)
        + (None,) * 17,
        static_broadcasted_argnums=(44, 45),
    )
    fused_lasd_boussinesq_from_context = mapped(
        fused_lasd_boussinesq_from_context_local,
        in_axes=(0, 0, 0, 0)
        + (None,) * 19
        + (0, 0, 0)
        + (None,) * 17,
        static_broadcasted_argnums=(41, 42),
    )
    surface_transfer = build_monin_obukhov_surface_transfer_kernel(axis_name)
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
    lasd_update_momentum = mapped(
        lasd_update_momentum_local,
        in_axes=(0,) * 8 + (None,) * 9,
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
        in_axes=(0, 0, 0) + (None,) * 7,
    )
    buoyancy = mapped(buoyancy_local, in_axes=(0, None))
    rayleigh_damping = mapped(
        rayleigh_damping_local,
        in_axes=(0, 0, 0, None, None, None, None),
    )
    return _JaxDiscretization(
        decomposition,
        addressable_partitions,
        frozen_zero_scalar,
        _ProjectionKernels(
            exchange_packed,
            pressure_gradient,
            divergence,
            enforce_upper_boundary,
            prepare_projection,
            finish_projection,
            horizontal_divergence,
            horizontal_gradient,
            filter_state,
            filter_horizontal,
            filter_scalar,
            jax.jit(filter_boundary),
            correct,
        ),
        _FlowKernels(
            ab2_update,
            ab2_update_velocity,
            combine_payloads,
            dry_flow_context,
            dry_rotational_advection,
            dry_wall,
            dry_sgs,
            dry_sgs_vertical_flux,
            dry_sgs_tke_transfer,
            fused_lasd_boussinesq,
            fused_lasd_boussinesq_from_context,
        ),
        _LasdKernels(
            relax_lasd_field,
            lasd_accumulate,
            lasd_accumulate_velocity,
            lasd_update,
            lasd_update_momentum,
            lasd_diagnostics,
        ),
        _ScalarKernels(
            scalar_context,
            scalar_advection,
            scalar_sgs,
            buoyancy,
            rayleigh_damping,
            surface_transfer,
        ),
        _WindKernels(
            wind_tunnel,
            actuator_line,
            actuator_disk_bem,
            nacelle_tower,
        ),
    )
